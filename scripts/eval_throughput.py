"""
Throughput benchmark for MDLM sampling. The HEADLINE result for the project:
tokens/sec vs sequence length for BiMamba-MDLM vs Transformer-MDLM.

Two modes:
  --mode forward    Pure backbone forward pass throughput (apples-to-apples).
  --mode sampling   Full MDLM sampling loop (closer to deployment).

Usage:
    python scripts/eval_throughput.py \
        --checkpoint outputs/runD_130m/checkpoints/last.ckpt \
        --backbone dimamba \
        --seq-lens 512 1024 2048 4096 8192 \
        --mode forward \
        --out results/throughput_runD.json

Always run with `nvidia-smi -l 1` in another shell to confirm steady-state GPU util.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add project root to path so `import diffusion` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to Lightning .ckpt file.')
    p.add_argument('--backbone', type=str, required=True,
                   choices=['dimamba', 'dit'],
                   help='Backbone type the checkpoint was trained with.')
    p.add_argument('--seq-lens', type=int, nargs='+',
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument('--batch-size', type=int, default=1,
                   help='Batch size for throughput. 1 = single-stream deployment.')
    p.add_argument('--n-steps', type=int, default=100,
                   help='Number of forward passes (or sampling steps) per measurement.')
    p.add_argument('--warmup', type=int, default=10,
                   help='Warmup forward passes (excluded from timing).')
    p.add_argument('--mode', choices=['forward', 'sampling'], default='forward')
    p.add_argument('--out', type=str, required=True,
                   help='Output JSON path.')
    p.add_argument('--dtype', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    return p.parse_args()


def load_model(ckpt_path: str, backbone: str):
    """Load a Lightning Diffusion checkpoint and return its backbone module."""
    import diffusion
    import dataloader
    import omegaconf

    # Use weights_only=False directly — Lightning's load_from_checkpoint
    # serialises arbitrary OmegaConf / typing internals that would require
    # an unbounded safe_globals allowlist under PyTorch 2.6+.
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['hyper_parameters']['config']
    if isinstance(config, dict):
        config = omegaconf.OmegaConf.create(config)

    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion.Diffusion(config, tokenizer=tokenizer)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    model.cuda()
    return model, config, tokenizer


@torch.no_grad()
def bench_forward(model, seq_len, batch_size, n_steps, warmup, dtype, mask_id):
    """Time pure forward passes through the backbone."""
    device = 'cuda'
    ids = torch.full((batch_size, seq_len), mask_id,
                     dtype=torch.long, device=device)
    sigma = torch.full((batch_size,), 0.5, device=device)

    amp_dtype = {'bf16': torch.bfloat16,
                 'fp16': torch.float16,
                 'fp32': torch.float32}[dtype]

    backbone = model.backbone

    # Warmup
    with torch.amp.autocast('cuda', dtype=amp_dtype):
        for _ in range(warmup):
            _ = backbone(ids, sigma)
    torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    with torch.amp.autocast('cuda', dtype=amp_dtype):
        for _ in range(n_steps):
            _ = backbone(ids, sigma)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = batch_size * seq_len * n_steps
    return {
        'tokens_per_sec': tokens / elapsed,
        'ms_per_forward': 1000.0 * elapsed / n_steps,
        'elapsed_sec': elapsed,
        'peak_mem_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


@torch.no_grad()
def bench_sampling(model, seq_len, batch_size, n_steps, warmup, dtype, mask_id):
    """Time the actual MDLM sampling loop (forward + token unmasking per step)."""
    device = 'cuda'
    amp_dtype = {'bf16': torch.bfloat16,
                 'fp16': torch.float16,
                 'fp32': torch.float32}[dtype]

    def one_sampling_pass():
        x = torch.full((batch_size, seq_len), mask_id,
                       dtype=torch.long, device=device)
        # Simple ancestral unmasking: each step, sample logits and unmask a
        # fraction of remaining positions by argmax. Matches the dominant cost
        # of MDLM sampling (forward pass per step).
        sigma = torch.full((batch_size,), 0.5, device=device)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            for s in range(n_steps):
                logits = model.backbone(x, sigma)
                # Update masked positions toward argmax (toy unmasking).
                preds = logits.argmax(dim=-1)
                # In MDLM real sampling, unmask 1/n_steps of positions per step.
                unmask_frac = 1.0 / max(n_steps - s, 1)
                mask = (x == mask_id) & (
                    torch.rand_like(x, dtype=torch.float32) < unmask_frac)
                x = torch.where(mask, preds, x)

    # Warmup
    for _ in range(max(1, warmup // n_steps)):
        one_sampling_pass()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    one_sampling_pass()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = batch_size * seq_len
    return {
        'tokens_per_sec': tokens / elapsed,
        'sec_per_sample': elapsed,
        'sampling_steps': n_steps,
        'peak_mem_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def main():
    args = parse_args()
    assert torch.cuda.is_available(), 'CUDA required.'

    print(f'Loading {args.backbone} checkpoint from {args.checkpoint}')
    model, config, tokenizer = load_model(args.checkpoint, args.backbone)
    mask_id = model.mask_index
    print(f'mask_id={mask_id}, vocab_size={model.vocab_size}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    results = {
        'backbone': args.backbone,
        'checkpoint': args.checkpoint,
        'mode': args.mode,
        'dtype': args.dtype,
        'batch_size': args.batch_size,
        'n_steps': args.n_steps,
        'gpu': torch.cuda.get_device_name(0),
        'measurements': [],
    }

    for sl in args.seq_lens:
        torch.cuda.reset_peak_memory_stats()
        try:
            if args.mode == 'forward':
                r = bench_forward(model, sl, args.batch_size, args.n_steps,
                                  args.warmup, args.dtype, mask_id)
            else:
                r = bench_sampling(model, sl, args.batch_size, args.n_steps,
                                   args.warmup, args.dtype, mask_id)
            r['seq_len'] = sl
            r['oom'] = False
            results['measurements'].append(r)
            print(f"  seq_len={sl:>5}: {r['tokens_per_sec']:>10.0f} tok/s "
                  f"  peak_mem={r['peak_mem_gb']:.1f}GB")
        except torch.cuda.OutOfMemoryError:
            results['measurements'].append({
                'seq_len': sl, 'oom': True, 'tokens_per_sec': None})
            print(f"  seq_len={sl:>5}: OOM")
            torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
