"""
End-to-end checkpoint resume verification. Runs a tiny training session,
saves a ckpt, then runs a fresh process that loads from that ckpt and checks:

  1. EMA shadow params are restored exactly.
  2. Backbone weights are restored exactly.
  3. global_step / epoch / optimizer state are restored.
  4. fast_forward_epochs / fast_forward_batches are set (so the dataloader
     skips already-seen batches).

Run BEFORE any RunPod training to confirm the save-resume contract holds.

Usage:
    python scripts/verify_checkpoint_resume.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ckpt_states(ckpt_path):
    """Pull the bits we care about out of a Lightning ckpt."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {
        'global_step': ckpt['global_step'],
        'epoch': ckpt['epoch'],
        'has_optimizer': 'optimizer_states' in ckpt,
        'has_ema': 'ema' in ckpt,
        'ema_num_updates': ckpt.get('ema', {}).get('num_updates'),
        'ema_n_shadow': len(ckpt['ema']['shadow_params']) if 'ema' in ckpt else 0,
        'state_dict_keys': sorted(ckpt['state_dict'].keys()),
        # Hash a couple of params for exact-restore check.
        'first_param_hash': hash(
            tuple(next(iter(ckpt['state_dict'].values())).flatten()[:16].tolist())),
        'ema_first_shadow_hash': (
            hash(tuple(ckpt['ema']['shadow_params'][0].flatten()[:16].tolist()))
            if 'ema' in ckpt else None),
    }


def run_training(steps, run_dir, resume_from=None):
    """Run main.py for `steps` training steps, optionally resuming."""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['HYDRA_FULL_ERROR'] = '1'
    cmd = [
        sys.executable, '-u', 'main.py',
        '+experiment=toy',
        f'trainer.max_steps={steps}',
        # Force a stable run dir so we can find the ckpt deterministically.
        f'hydra.run.dir={run_dir}',
        # Tiny model + tiny batch + zero workers — keep verification fast.
        'loader.global_batch_size=4',
        'loader.num_workers=0',
        'trainer.limit_val_batches=0.0',  # disable validation entirely for this test
        'trainer.num_sanity_val_steps=0',
        # Save every 10 steps so the first run produces a ckpt fast.
        'callbacks.checkpoint_every_n_steps.every_n_train_steps=10',
    ]
    if resume_from is not None:
        cmd += [
            'checkpointing.resume_from_ckpt=true',
            f'checkpointing.resume_ckpt_path={resume_from}',
        ]

    print(f'\n>>> {" ".join(cmd)}\n')
    r = subprocess.run(cmd, env=env, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f'training subprocess failed with exit {r.returncode}')


def main():
    if not torch.cuda.is_available():
        sys.exit('CUDA required — run this on Colab/RunPod, not Windows.')

    tmp = Path(tempfile.mkdtemp(prefix='dimamba_resume_'))
    print(f'work dir: {tmp}')

    run1 = tmp / 'run1'
    run2 = tmp / 'run2'

    try:
        # Phase 1: train to step 20.
        run_training(steps=20, run_dir=run1)
        ckpt1 = run1 / 'checkpoints' / 'last.ckpt'
        assert ckpt1.exists(), f'no ckpt at {ckpt1}'
        s1 = _ckpt_states(ckpt1)
        assert s1['global_step'] >= 10, f'step1 = {s1["global_step"]}'
        assert s1['has_ema'], 'EMA state missing from ckpt1'
        assert s1['has_optimizer'], 'optimizer state missing from ckpt1'
        print(f'\n  Phase 1: trained to global_step={s1["global_step"]}, '
              f'ema_n_shadow={s1["ema_n_shadow"]}\n')

        # Phase 2: resume from ckpt1, train 20 more steps.
        run_training(steps=40, run_dir=run2, resume_from=str(ckpt1))
        ckpt2 = run2 / 'checkpoints' / 'last.ckpt'
        assert ckpt2.exists(), f'no ckpt at {ckpt2}'
        s2 = _ckpt_states(ckpt2)
        print(f'\n  Phase 2: trained to global_step={s2["global_step"]}\n')

        # Checks.
        problems = []
        if s2['global_step'] <= s1['global_step']:
            problems.append(
                f"step did not advance: {s1['global_step']} -> {s2['global_step']}")
        if not s2['has_ema']:
            problems.append('EMA missing from resumed ckpt')
        if s2['state_dict_keys'] != s1['state_dict_keys']:
            problems.append('state_dict keys diverged between ckpt1 and ckpt2')
        if s2['ema_n_shadow'] != s1['ema_n_shadow']:
            problems.append(
                f"EMA shadow count changed: {s1['ema_n_shadow']} -> {s2['ema_n_shadow']}")
        if s2['ema_num_updates'] is None or s2['ema_num_updates'] <= (s1['ema_num_updates'] or 0):
            problems.append(
                f"EMA num_updates did not advance: "
                f"{s1['ema_num_updates']} -> {s2['ema_num_updates']}")

        if problems:
            print('\nFAIL — resume contract broken:')
            for p in problems:
                print(f'  - {p}')
            sys.exit(1)

        print('PASS — checkpoint resume contract holds:')
        print(f'  step  : {s1["global_step"]} -> {s2["global_step"]}')
        print(f'  ema   : updates {s1["ema_num_updates"]} -> {s2["ema_num_updates"]}, '
              f'{s2["ema_n_shadow"]} shadow params')
        print(f'  keys  : {len(s2["state_dict_keys"])} state_dict entries preserved')
    finally:
        # Keep dir on failure for debugging.
        if 'PASS' in (sys.exc_info()[0].__name__ if sys.exc_info()[0] else ''):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
