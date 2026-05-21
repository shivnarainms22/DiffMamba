# DiffMamba — Bidirectional Mamba-2 backbones for Masked Diffusion Language Models

A small-scale, independent study of **state-space-model (Mamba-2) denoisers for
masked diffusion language models (MDLM)**. I replace the Transformer/DiT denoiser
in MDLM with a **bidirectional Mamba-2 backbone**, train a matched set of models
from scratch on OpenWebText, and run a controlled comparison plus a scaling study,
a learning-rate fairness check, and an inference-efficiency benchmark.

> **Full technical report:** [**DiffMamba_Report.md**](./DiffMamba_Report.md) —
> architecture, implementation, HPC engineering notes, results, and honest
> limitations.

This repository is a **fork/extension of the MDLM codebase** (Sahoo et al.,
NeurIPS 2024); the diffusion framework and the Transformer baseline are theirs.
The bidirectional Mamba-2 backbone, training/eval/benchmark tooling for it, and
all experiments here are mine. See **Attribution & Related Work** below.

---

## What I did

- Implemented a **bidirectional Mamba-2 denoiser** (`models/dimamba.py`): a
  forward + flipped-reverse Mamba-2 with weight-tied projections and AdaLN
  noise-level conditioning, as a drop-in replacement for the MDLM DiT.
- Trained **5 models from scratch** on OpenWebText (~5B tokens for 130M) on an
  academic SLURM cluster (single A100 per run, 8-hour wall limit, automatic
  checkpoint/resume job-chaining): BiMamba-130M ×2 seeds, a parameter-matched
  Transformer-130M, and BiMamba-50M / 100M for a scaling study.
- Evaluated validation perplexity (MDLM ELBO bound), generative perplexity
  (GPT-2-large), and **inference efficiency vs. sequence length** (`scripts/bench_efficiency.py`).

## Headline results

Validation perplexity (lower is better), matched 130M parameters / ~5B tokens:

| Model | Backbone | Params | Val PPL ↓ |
|-------|----------|--------|-----------|
| Transformer (DiT) | attention | 130M | **70.5** |
| BiMamba-2 (2 seeds) | SSM | 130M | ~84.7 |

BiMamba scaling (Val PPL): **50M → 136.3, 100M → 97.5, 130M → 84.7** (clean,
monotonic). Seed-stable (Δ≈2.4 between seeds).

**Honest finding:** at matched compute with the MDLM (Transformer-tuned) recipe,
the Transformer denoiser is modestly but consistently stronger on quality; the
*pure* BiMamba-2 trails — consistent with the prior work below, where a *hybrid*
Mamba+attention model is what recovers quality. The motivating advantage of the
SSM backbone is inference efficiency at long context (see the report). Full
numbers, caveats, and the in-progress LR-fairness sweep are in the
[report](./DiffMamba_Report.md).

---

## Quickstart

Train (per experiment):
```bash
python main.py +experiment=runD_130m seed=1 data.cache_dir=<path>/data
```
Experiments: `runD_130m` (BiMamba-130M), `runB_transformer_130m` (DiT-130M),
`scaling_50m`, `scaling_100m`.

Evaluate perplexity / generate samples / benchmark efficiency:
```bash
python main.py mode=ppl_eval +experiment=<exp> eval.checkpoint_path=<ckpt>
python main.py mode=sample_eval +experiment=<exp> eval.checkpoint_path=<ckpt> loader.eval_batch_size=4
python scripts/bench_efficiency.py
```

Code map: `models/dimamba.py` (BiMamba-2, this work) · `models/dit.py` (DiT
baseline) · `diffusion.py` (MDLM framework) · `configs/` (Hydra configs) ·
`scripts/` (HPC setup + SLURM chaining + efficiency benchmark).

---

## Attribution & Related Work

This work **builds directly on MDLM** and reproduces a research direction
recently introduced by **DiffuApriel/DiffuMamba**:

- **MDLM** — *Simple and Effective Masked Diffusion Language Models*, Sahoo et
  al., NeurIPS 2024. [[paper]](https://arxiv.org/abs/2406.07524)
  [[code]](https://github.com/kuleshov-group/mdlm). The diffusion framework
  (absorbing-state, SUBS parameterization) and the DiT baseline used here are
  from MDLM. This repo is an extension of their codebase.
- **DiffuApriel / DiffuMamba** — *High-Throughput Diffusion LMs with Mamba
  Backbone*, arXiv 2511.15927 (2025). Introduces bidirectional Mamba-2 (and
  hybrid) denoisers for MDLM at 240M–1.3B with large inference-throughput gains.
  This project is an independent small-scale (50–130M) reproduction and analysis
  of that direction; it is **not** claimed as a novel architecture.

### Citing MDLM (upstream framework)
```bibtex
@inproceedings{sahoo2024simple,
  title={Simple and Effective Masked Diffusion Language Models},
  author={Subham Sekhar Sahoo and Marianne Arriola and Aaron Gokaslan and Edgar Mariano Marroquin and Alexander M Rush and Yair Schiff and Justin T Chiu and Volodymyr Kuleshov},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=L4uaAR4ArM}
}
```

The MDLM repository was itself built on
[SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).
