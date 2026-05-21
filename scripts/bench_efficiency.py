"""Inference-efficiency benchmark: BiMamba-2 vs DiT/Transformer denoiser.

Measures forward-pass latency, throughput (tokens/s), and peak GPU memory
across sequence lengths, at matched 130M parameter count. This reproduces the
core efficiency claim of DiffuApriel/DiffuMamba (arXiv 2511.15927) at small
scale: a linear-time SSM denoiser should scale better with sequence length than
quadratic attention.

Random-initialized models are used on purpose — we measure the architecture's
compute cost, not quality, so no checkpoints are needed.

Run on a GPU node with the diffmamba env (needs mamba_ssm + flash_attn):
    python scripts/bench_efficiency.py
    python scripts/bench_efficiency.py --batch 4 --lengths 1024 2048 4096 8192 16384
"""
import argparse
import os
import time

import hydra
import torch

import models.dimamba
import models.dit

_CONFIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
_VOCAB = 50258      # GPT-2 vocab + mask token (size only affects embed/head)
_PAD = 50257


def _build_backbones():
    """Compose both 130M configs in one Hydra context and build the backbones.

    Neither backbone allocates length-dependent params at construction (Mamba is
    length-agnostic; the DiT uses rotary embeddings), so a single build serves
    all sequence lengths.
    """
    with hydra.initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        cfg_mamba = hydra.compose(
            config_name='config', overrides=['+experiment=runD_130m'])
        cfg_dit = hydra.compose(
            config_name='config', overrides=['+experiment=runB_transformer_130m'])

    mamba = models.dimamba.DiMamba(
        cfg_mamba, vocab_size=_VOCAB, pad_token_id=_PAD).cuda().eval()
    dit = models.dit.DIT(cfg_dit, vocab_size=_VOCAB).cuda().eval()
    return [('BiMamba-2 (130M)', mamba), ('DiT/Transformer (130M)', dit)]


def _count_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


@torch.no_grad()
def _bench(backbone, batch, seq_len, iters=10, warmup=3):
    idx = torch.randint(0, _VOCAB, (batch, seq_len), device='cuda')
    sigma = torch.rand(batch, device='cuda') * 0.5 + 0.25
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            backbone(idx, sigma)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            backbone(idx, sigma)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return dt, batch * seq_len / dt, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lengths', type=int, nargs='+',
                    default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument('--iters', type=int, default=10)
    args = ap.parse_args()

    backbones = _build_backbones()
    for name, m in backbones:
        print(f'{name}: {_count_params(m):.1f}M params')
    print()
    header = (f"{'backbone':<26}{'seq_len':>8}{'ms/fwd':>10}"
              f"{'tok/s':>12}{'peak_GB':>10}")
    print(header)
    print('-' * len(header))
    for name, m in backbones:
        for L in args.lengths:
            try:
                dt, tps, gb = _bench(m, args.batch, L, iters=args.iters)
                print(f'{name:<26}{L:>8}{dt * 1000:>10.1f}'
                      f'{tps:>12.0f}{gb:>10.2f}')
            except torch.cuda.OutOfMemoryError:
                print(f'{name:<26}{L:>8}{"OOM":>10}{"OOM":>12}{"OOM":>10}')
                torch.cuda.empty_cache()
        print()


if __name__ == '__main__':
    main()
