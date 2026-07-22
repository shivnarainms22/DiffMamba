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

**Quality** — Validation perplexity (lower is better), 130M parameters / ~5B tokens:

| Model | Backbone | Params | LR | Val PPL ↓ |
|-------|----------|--------|----|-----------|
| Transformer (DiT) | attention | 130M | 3e-4 | **70.5** |
| BiMamba-2 (2 seeds) | SSM | 130M | 3e-4 | ~84.7 |
| **BiMamba-2 (LR-tuned)** | **SSM** | **130M** | **1e-3** | **79.3** |

BiMamba scaling (lr=3e-4 Val PPL): **50M → 136.3, 100M → 97.5, 130M → 84.7** —
clean, monotonic, seed-stable (Δ≈2.4 between seeds). A learning-rate sweep at
50M (§6.5) showed BiMamba prefers a **~3.3× higher LR** than the MDLM default;
retraining at 130M with the tuned LR (§6.6) closes **~43% of the gap** to the
Transformer (15.5 → 8.8 ppl) but does **not** close it.

**Throughput** — forward-pass benchmark, 130M-class denoisers on A100 (§6.7):

| seq_len | BiMamba ms | DiT ms | Speedup |
|--------:|-----------:|-------:|--------:|
|    1024 |       29.1 |   17.2 | 0.59× (DiT faster) |
|    4096 |       45.9 |   55.0 | 1.20× ← crossover |
|   32768 |      364.0 | 1136.2 | **3.12×** |

BiMamba latency is **textbook-linear** in seq length; DiT is empirically
O(L^1.55) even with FlashAttention. Crossover ~3K tokens.

**Honest finding:** at matched compute with the MDLM (Transformer-tuned) recipe,
the Transformer denoiser is modestly but consistently stronger on quality —
even after retuning BiMamba's learning rate, the gap narrows but does not close.
The *pure* BiMamba-2 trails on quality, but is **~3× faster** at long context.
This is consistent with the prior work below, where a *hybrid* Mamba+attention
model is what recovers quality. Full numbers, caveats, and the LR-fairness
analysis are in the [report](./DiffMamba_Report.md).

**Hybrid backbone — the quality gap, closed (best result here).** Inserting
sparse bidirectional attention into the BiMamba backbone recovers full DiT-class
quality. At matched 130M / 76k / lr 3e-4, just **3 of 12 layers as attention**
(`[3,7,11]`) reaches **69.6 val PPL**, statistically matching the Transformer
(70.5) while keeping 9/12 layers linear-time. An attention-layout ablation then
shows **placement matters more than count** — distribute attention through depth
(clustering it early is catastrophic, 80.8 PPL), and **4 evenly-spread layers**
(`[2,5,8,11]`) is best. Over-training that winner to 150k steps reaches **61.2
val PPL** (2-seed mean ±0.3) — the strongest quality result in this study, though
at ~2× the compute of the matched table above (so not a matched-compute claim
against the 70.5 DiT). Ablation grid, placement analysis, and the over-train
detail are in [VLM report §11.5](./DiffMamba_VLM_Report.md).

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
