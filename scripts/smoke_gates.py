"""
Go/no-go gate for moving from Colab toy run -> RunPod production run.

Run AFTER `python main.py +experiment=toy` completes. Returns exit 0 if all
gates pass, exit 1 otherwise. Designed to be one of:

    python scripts/smoke_gates.py outputs/<DATE>/<TIME>

The script checks the 7 gates that block A100 spot spend:
  1. Final loss < initial loss (model is learning).
  2. Step-0 loss is in the expected range (sanity).
  3. No NaN/Inf in any logged metric.
  4. Throughput >= 20K tok/s for forward pass at seq=1024 on A100.
  5. Peak memory < 70% of GPU memory (room for the production batch).
  6. Checkpoint loads cleanly and reproduces a forward pass.
  7. Unit-test suite passes.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch


GATES = []  # populated by @gate


def gate(name):
    def deco(fn):
        GATES.append((name, fn))
        return fn
    return deco


def find_last_ckpt(run_dir: Path) -> Path:
    candidates = list(run_dir.rglob('last.ckpt'))
    assert candidates, f'no last.ckpt under {run_dir}'
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_wandb_or_lightning_logs(run_dir: Path):
    """Best-effort: parse Lightning's metrics.csv if it exists."""
    csvs = list(run_dir.rglob('metrics.csv'))
    if not csvs:
        return None
    import csv
    rows = []
    with open(csvs[0]) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


@gate('learning')
def check_learning(ctx):
    rows = ctx['log_rows']
    if not rows:
        return False, 'no metrics.csv found'
    losses = [float(r['trainer/loss']) for r in rows
              if r.get('trainer/loss') not in (None, '', 'nan')]
    if len(losses) < 5:
        return False, f'only {len(losses)} loss readings'
    initial = sum(losses[:3]) / 3
    final = sum(losses[-3:]) / 3
    ratio = final / initial
    return ratio < 0.95, f'loss {initial:.3f} -> {final:.3f} (ratio {ratio:.2f})'


@gate('step0_loss_sanity')
def check_step0(ctx):
    rows = ctx['log_rows']
    if not rows:
        return False, 'no logs'
    losses = [float(r['trainer/loss']) for r in rows
              if r.get('trainer/loss') not in (None, '', 'nan')]
    if not losses:
        return False, 'no loss values'
    init = losses[0]
    # MDLM step-0 loss should be near ln(vocab_size) ~= 10.83 for GPT-2 + mask.
    expected = math.log(50258)
    return abs(init - expected) < 2.0, \
        f'initial loss {init:.2f}, expected ~{expected:.2f}'


@gate('no_nan')
def check_no_nan(ctx):
    rows = ctx['log_rows']
    if not rows:
        return False, 'no logs'
    for r in rows:
        for k, v in r.items():
            if v in (None, ''):
                continue
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return False, f'NaN/Inf in {k}'
            except (TypeError, ValueError):
                pass
    return True, 'all logged metrics finite'


@gate('throughput')
def check_throughput(ctx):
    """Run the throughput benchmark and check forward-pass tok/s at seq=1024."""
    out_json = ctx['run_dir'] / 'smoke_throughput.json'
    cmd = [
        sys.executable, 'scripts/eval_throughput.py',
        '--checkpoint', str(ctx['ckpt']),
        '--backbone', 'dimamba',
        '--seq-lens', '1024',
        '--mode', 'forward',
        '--n-steps', '50',
        '--out', str(out_json),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return False, f'throughput script failed: {r.stderr[-500:]}'
    data = json.loads(out_json.read_text())
    tps = data['measurements'][0]['tokens_per_sec']
    # A100 forward pass at 130M, seq=1024, bs=1 should reach ~30K tok/s easily.
    # Toy uses micro-dimamba which is faster; gate at 20K as a generous floor.
    return tps >= 20_000, f'forward tok/s = {tps:.0f} at seq=1024'


@gate('memory_headroom')
def check_memory(ctx):
    """Toy run peak memory should leave room for the production batch."""
    # Inferred from the throughput benchmark output.
    out = ctx['run_dir'] / 'smoke_throughput.json'
    if not out.exists():
        return False, 'no throughput JSON to read memory from'
    data = json.loads(out.read_text())
    peak = data['measurements'][0].get('peak_mem_gb', 0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    frac = peak / total
    return frac < 0.7, f'peak {peak:.1f}GB / {total:.0f}GB ({frac:.0%})'


@gate('ckpt_loads')
def check_ckpt_loads(ctx):
    """Reload the checkpoint and run one forward pass."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import diffusion
    import dataloader
    import omegaconf

    ckpt = torch.load(ctx['ckpt'], map_location='cpu', weights_only=False)
    config = ckpt['hyper_parameters']['config']
    if isinstance(config, dict):
        config = omegaconf.OmegaConf.create(config)
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion.load_from_checkpoint(
        ctx['ckpt'], tokenizer=tokenizer, config=config,
        map_location='cuda')
    model.eval().cuda()
    ids = torch.full((1, 256), model.mask_index,
                     dtype=torch.long, device='cuda')
    sigma = torch.full((1,), 0.5, device='cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        out = model.backbone(ids, sigma)
    ok = (out.shape == (1, 256, model.vocab_size)
          and torch.isfinite(out).all())
    return bool(ok), f'forward output shape {tuple(out.shape)}'


@gate('unit_tests')
def check_unit_tests(ctx):
    r = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_bimamba2.py', '-q'],
        capture_output=True, text=True)
    return r.returncode == 0, r.stdout.splitlines()[-1] if r.stdout else r.stderr[-200:]


@gate('resume_roundtrip')
def check_resume(ctx):
    """Run the resume verification subprocess. Slow (~5 min) but critical."""
    r = subprocess.run(
        [sys.executable, 'scripts/verify_checkpoint_resume.py'],
        capture_output=True, text=True, timeout=600)
    last = r.stdout.strip().splitlines()[-1] if r.stdout else ''
    return r.returncode == 0, last or r.stderr[-200:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=Path,
                    help='Hydra run dir, e.g. outputs/openwebtext/2026.05.18/142312')
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    assert run_dir.exists(), run_dir
    ckpt = find_last_ckpt(run_dir)
    log_rows = parse_wandb_or_lightning_logs(run_dir) or []

    ctx = {'run_dir': run_dir, 'ckpt': ckpt, 'log_rows': log_rows}

    print(f'run_dir : {run_dir}')
    print(f'ckpt    : {ckpt}')
    print(f'log rows: {len(log_rows)}\n')

    all_pass = True
    for name, fn in GATES:
        try:
            ok, msg = fn(ctx)
        except Exception as e:
            ok, msg = False, f'EXCEPTION: {e}'
        marker = 'PASS' if ok else 'FAIL'
        print(f'  [{marker}] {name:<20} {msg}')
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print('All gates passed. Greenlight for RunPod.')
        sys.exit(0)
    else:
        print('One or more gates failed. DO NOT start the production run.')
        sys.exit(1)


if __name__ == '__main__':
    main()
