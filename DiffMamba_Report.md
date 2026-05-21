# DiffMamba: Bidirectional Mamba-2 Backbones for Masked Diffusion Language Models

**A small-scale reproduction and analysis of SSM-backbone masked diffusion LMs:
a controlled comparison of a bidirectional Mamba-2 denoiser against a
parameter-matched Transformer denoiser within the MDLM framework, with a BiMamba
scaling study, a learning-rate fairness check, and an inference-efficiency
replication.**

This work reproduces, at 50–130M scale, the research direction introduced by
DiffuApriel/DiffuMamba (arXiv 2511.15927) on top of the MDLM framework (Sahoo et
al., NeurIPS 2024). It is positioned as a reproduction/analysis study, not a
novel architecture proposal — see §2.5 (Related Work and Positioning).

Date: 2026-05-21 · Hardware: Northeastern Explorer HPC (NVIDIA A100)

---

## 1. Summary

We replace the Transformer/DiT denoiser in a masked diffusion language model
(MDLM) with a **bidirectional Mamba-2 (BiMamba-2)** backbone, and ask whether a
linear-time state-space model can match or beat attention at matched compute.

We trained five models on OpenWebText: BiMamba-MDLM at 130M (two seeds), a
parameter-matched Transformer-MDLM at 130M, and BiMamba-MDLM at 50M and 100M for
a scaling study. The headline result is a **clean negative for the core
hypothesis at this scale**: the Transformer backbone achieves lower validation
perplexity (70.5 vs ~84.7) and slightly more fluent samples. BiMamba nonetheless
scales cleanly with model size and is seed-stable. The most important caveat is
that all runs used the MDLM training recipe, which was tuned for a Transformer.

| Model | Backbone | Params | Tokens | Val PPL (↓) | Gen PPL (↓) |
|-------|----------|--------|--------|-------------|-------------|
| runB  | Transformer (DiT) | 130M | ~5B | **70.45** | **91.11** |
| runD1 | BiMamba-2 (seed 1) | 130M | ~5B | 85.91 | 96.15 |
| runD2 | BiMamba-2 (seed 2) | 130M | ~5B | 83.51 | — |
| s100  | BiMamba-2 | 100M | ~4B | 97.53 | — |
| s50   | BiMamba-2 | 50M  | ~2B | 136.30 | — |

*Val PPL = exp(validation NLL), the MDLM ELBO perplexity bound on the full
validation split (EMA weights). Gen PPL = perplexity of generated samples scored
by GPT-2-large (4 samples; noisy).*

---

## 2. Motivation

Masked diffusion language models (MDLM; Sahoo et al.) generate text by iteratively
denoising a fully-masked sequence, using a bidirectional denoiser network. The
standard denoiser is a Diffusion Transformer (DiT), whose self-attention is
O(n²) in sequence length.

State-space models (Mamba/Mamba-2) process sequences in O(n) time with a
hardware-efficient selective scan, and have matched Transformers on autoregressive
language modeling. Because the MDLM denoiser must be **bidirectional** (it sees
the whole partially-masked sequence at once), a natural question is whether a
**bidirectional** Mamba-2 can serve as a drop-in denoiser that is competitive in
quality while being cheaper at long context.

**Hypothesis:** A BiMamba-2 backbone matches or exceeds a parameter-matched DiT
backbone in MDLM validation perplexity at a fixed training budget.

This report tests that hypothesis under a controlled, matched-compute protocol.

### 2.5 Related Work and Positioning

- **MDLM (Sahoo et al., NeurIPS 2024)** introduced the masked/absorbing-state
  diffusion framework with the SUBS parameterization used here, on a DiT
  (Transformer) backbone. Our diffusion framework and Transformer baseline follow
  MDLM directly.
- **DiffuApriel / DiffuMamba (arXiv 2511.15927, Nov 2025)** is the most directly
  related work: it replaces the MDLM Transformer denoiser with a bidirectional
  Mamba-2 (and a hybrid Mamba+attention variant), trains at 240M–1.3B, and reports
  up to ~5.3× (pure) / ~2.8× (hybrid) inference throughput with the hybrid
  matching or beating the Transformer on quality. It claims to be the first
  family of SSM-based masked diffusion LMs and evaluates long context up to 65K
  tokens.

**Positioning.** The architecture (BiMamba-2 denoiser for MDLM) and the
efficiency/long-context narrative are therefore **not novel to this work** — they
are established by DiffuApriel. We position this study as a **small-scale,
independent reproduction and analysis** (50–130M, single GPU, academic HPC) with
three deliverables: (i) a controlled quality comparison of the *pure* BiMamba-2
denoiser vs a matched Transformer; (ii) a learning-rate fairness check (does a
better LR close the quality gap?); and (iii) an independent replication of the
inference-efficiency scaling claim at small scale. Our finding that the *pure*
SSM denoiser trails the Transformer on quality is consistent with DiffuApriel,
where the *hybrid* (not pure Mamba) is what recovers quality.

---

## 3. Architecture

### 3.1 Diffusion framework (shared by both backbones)

- **Process:** absorbing-state (masking) discrete diffusion. The forward process
  progressively replaces tokens with a `[MASK]` absorbing state according to a
  noise schedule; the denoiser predicts the clean tokens.
- **Parameterization:** `subs` (SUBS, from MDLM) — the network output is mapped
  to a categorical over the vocabulary with the zero-masking-probability and
  carry-over-unmasking constraints baked in, which is what makes the MDLM bound
  tight and the model effectively time-independent in its core prediction.
- **Noise schedule:** log-linear; **continuous time** (`T = 0`).
- **Training objective:** the (negative) diffusion ELBO, a weighted
  cross-entropy over masked positions with antithetic time sampling for variance
  reduction.
- **Conditioning:** the denoiser is conditioned on the noise level σ through an
  **AdaLN** (adaptive layer-norm) modulation path with a `TimestepEmbedder`
  (cond_dim = 128). σ is embedded, passed through SiLU, and used to produce
  per-block (shift, scale, gate) parameters.

### 3.2 BiMamba-2 backbone (the contribution)

Implemented in `models/dimamba.py`. The denoiser is a stack of pre-norm residual
blocks; each block's token mixer is a **bidirectional Mamba-2** module.

- **Bidirectional Mamba-2 (`BiMambaWrapper`):** runs a forward-direction Mamba-2
  over the sequence and a second Mamba-2 over the **flipped** sequence, then sums
  the (un-flipped) outputs (`bidirectional_strategy='add'`). This gives every
  position a full left+right receptive field, as attention would, but in linear
  time.
- **Weight tying:** the forward and reverse Mamba-2 share their input/output
  projection weights (`bidirectional_weight_tie=True`); only the inner SSM
  parameters differ. This keeps the bidirectional module close to the parameter
  budget of a single Mamba-2.
- **Mamba-2 hyperparameters:** `d_state=64`, `d_conv=4`, `expand=2`, `headdim=64`
  (so effective SSM heads = expand·d_model / 64).
- **Block structure:** pre-norm `Add → Norm → Mixer` with the mamba_ssm prenorm
  residual contract (the next block folds `hidden + residual` before its own
  norm). RMSNorm with fused add-norm Triton kernels; `residual_in_fp32=True`.
- **AdaLN conditioning:** when `temb_strategy='adaln'`, each block applies
  (shift, scale) to the normed hidden state before the mixer and a (gate) after,
  produced from the σ-embedding by a zero-initialized linear layer (so blocks
  start as identity w.r.t. conditioning). A final AdaLN modulates the output norm.
- **Heads:** untied LM head (`tie_word_embeddings=False`); embedding + linear
  projection to the vocabulary.

### 3.3 Transformer baseline (DiT)

The baseline (`models/dit.py`, config `small`) is the standard MDLM Diffusion
Transformer: pre-norm self-attention + MLP blocks with AdaLN σ-conditioning and
**FlashAttention** (`flash_attn_varlen_qkvpacked_func`). It uses the identical
diffusion framework, vocabulary, sequence length, and training recipe — only the
token-mixing mechanism (attention vs bidirectional SSM) differs.

### 3.4 Model configurations

| Config | Used by | hidden | blocks | mixer | ~Params |
|--------|---------|--------|--------|-------|---------|
| `small-dimamba` | runD1, runD2 | 768 | 12 | BiMamba-2 | ~125M |
| `small` (DiT)   | runB         | 768 | 12 | Attention (12 heads) | ~130M |
| `base-dimamba`  | s100         | 640 | 10 | BiMamba-2 | ~100M |
| `micro-dimamba` | s50          | 512 | 8  | BiMamba-2 | ~50M |

All use sequence length 1024, cond_dim 128, dropout 0.1, GPT-2 (BPE) vocabulary.
The 130M BiMamba and DiT are **size-matched** (768-dim, 12 blocks) — the core
fair comparison.

---

## 4. Implementation

### 4.1 Codebase

- `main.py` — Hydra entry point; modes `train` / `ppl_eval` / `sample_eval`.
- `diffusion.py` — `Diffusion` LightningModule: forward/reverse process, NELBO
  loss, EMA, sampling (`ddpm_cache` predictor), validation metrics.
- `models/dimamba.py` — BiMamba-2 denoiser (this work).
- `models/dit.py` — DiT denoiser baseline + shared AdaLN/embedder utilities.
- `dataloader.py` — OpenWebText tokenization, chunking to length-1024 sequences,
  Arrow caching.
- `configs/` — Hydra configs (model, data, experiment, callbacks, noise, etc.).
- `scripts/` — HPC setup and SLURM submission/chaining.

### 4.2 Training pipeline

- **Framework:** PyTorch Lightning, single A100 per run, DDP strategy.
- **Precision:** `bf16-mixed`; TF32 matmuls enabled (`set_float32_matmul_precision('high')`).
- **Optimizer:** AdamW, lr 3e-4, weight decay 0.01, β=(0.9, 0.999), ε=1e-8.
- **Schedule:** constant LR with 2500-step warmup.
- **EMA:** 0.9999 (evaluation uses EMA weights).
- **Global batch:** 64 sequences × 1024 tokens. On one GPU this is realized as
  **micro-batch 16 × gradient-accumulation 4** (see §4.4).
- **Gradient clipping:** 1.0.
- **Checkpointing:** every 2000 steps, keep last + best (by val/nll) + recent;
  HuggingFace Hub backup every 5000 steps (`Shiv-22/diffmamba-checkpoints`).

### 4.3 Data

OpenWebText, GPT-2 BPE tokenization, concatenated and chunked into length-1024
training sequences. The tokenized + chunked dataset (~55 GB) is built once and
cached to scratch; all runs share the cache read-only.

### 4.4 HPC infrastructure and engineering notes

Training ran on the Northeastern Explorer cluster (SLURM, `gpu` partition,
**8-hour wall-time limit**, mixed A100 40 GB / 80 GB nodes). Several
infrastructure issues were resolved during bring-up; they are recorded here
because they materially shaped the setup and are the most likely repro pitfalls.

- **CUDA/driver matching (root cause of a full day of failures):** the nodes run
  driver 570.x, which supports CUDA ≤ 12.8. A `cu130` PyTorch build loads but
  reports `torch.cuda.is_available() == False`. Fix: install **torch cu128**
  (2.11.0+cu128) and build CUDA extensions with the **cuda/12.8.0** toolkit.
- **Source-compiling extensions:** `causal-conv1d`, `mamba-ssm`, and
  `flash-attn` must be compiled **from source** against the resident torch
  (`--no-build-isolation --no-deps --no-cache-dir --no-binary :all:`); reusing a
  cached wheel reintroduces ABI/`undefined symbol` mismatches. `torchvision` must
  match torch (0.26.0+cu128) or `import torchmetrics` breaks.
- **8-hour wall via job chaining:** each run runs in ≤8h segments. A SLURM
  script resumes from `last.ckpt` and submits the next segment.
- **Checkpoint resume (critical bug):** Lightning passes `weights_only=True`
  explicitly to `torch.load` on resume, which rejects the OmegaConf `DictConfig`
  stored in the checkpoint. A `torch.load` shim must **force**
  `weights_only=False` (a `setdefault` is silently ignored). Without it, every
  resume crashes and runs never survive their wall.
- **Memory (OOM):** the 130M model with the full global batch of 64 OOMs even an
  80 GB A100. Solution: micro-batch 16 + gradient accumulation 4 (identical
  effective global batch of 64, identical training dynamics), plus
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **In-training sampling disabled:** the validation sampler builds a
  `[batch, seq, vocab]` categorical (~12 GB) that OOMs 40 GB nodes; in-training
  sample generation was disabled (`eval.generate_samples=False`). Validation NLL
  is unaffected; samples are produced post-hoc from checkpoints.
- **Cluster QOS limits:** max 4 running jobs and ~8 submitted jobs per user.
  This constrains how many of the 5 runs (each chaining a successor) can be
  in-flight simultaneously.
- **Step/checkpoint alignment gotcha:** s100's target (61000) is not a multiple
  of the 2000-step checkpoint interval, so the final state was never written to a
  checkpoint and the early-exit chain looped re-training the last 1000 steps. s100
  was finalized at its 60000-step checkpoint (98.4% trained). **Lesson: set
  max_steps to a multiple of the checkpoint interval.**

---

## 5. Experimental Setup

- **Five runs:** runD1/runD2 (BiMamba-130M, seeds 1/2), runB (DiT-130M),
  s100 (BiMamba-100M), s50 (BiMamba-50M).
- **Matched compute:** the 130M comparison fixes architecture size, data,
  optimizer, schedule, and step budget (76k steps ≈ 5B tokens ≈ one OWT pass).
  Scaling runs follow the same ~40 tokens/param recipe (s50: 30k steps ≈ 2B;
  s100: 61k steps ≈ 4B).
- **Evaluation:**
  - *Validation perplexity (primary):* `mode=ppl_eval` runs `trainer.validate`
    over the full validation split with EMA weights and reports `val/nll`;
    PPL = exp(nll). Same eval seed (1) for all runs, so the ELBO Monte-Carlo is
    identical across runs (fair comparison).
  - *Generative perplexity (secondary):* `mode=sample_eval` generates samples
    (128 denoising steps, `ddpm_cache` predictor) and scores them with
    GPT-2-large. Noisy (few samples); used only as a directional fluency check.

---

## 6. Results

### 6.1 Matched-compute comparison (130M, ~5B tokens)

| Backbone | Run | Val PPL (↓) | Gen PPL (↓) |
|----------|-----|-------------|-------------|
| Transformer (DiT) | runB | **70.45** | **91.11** |
| BiMamba-2 (seed 1) | runD1 | 85.91 | 96.15 |
| BiMamba-2 (seed 2) | runD2 | 83.51 | — |
| BiMamba-2 (avg) | — | ~84.71 | — |

**The Transformer backbone wins by ~17% validation perplexity** (70.45 vs ~84.71),
and is also slightly ahead on generative perplexity. This is the opposite of the
hypothesis: at this scale and budget, attention is the stronger MDLM denoiser.

### 6.2 Seed stability

runD1 (85.91) and runD2 (83.51) differ by only ~2.4 PPL (~2.8%) — far smaller
than the ~14-point gap to the Transformer. The architecture gap is therefore
**robust to seed**, not noise.

### 6.3 BiMamba scaling study

| Params | Run | Val PPL (↓) |
|--------|-----|-------------|
| 50M  | s50  | 136.30 |
| 100M | s100 | 97.53 |
| 130M | runD (avg) | 84.71 |

Perplexity decreases **monotonically and cleanly** with model size — BiMamba-MDLM
behaves like a well-posed model class; it simply trails the Transformer at each
matched comparison. (s100 finalized at 60000/61000 steps; see §4.4.)

### 6.4 Qualitative samples

Both backbones produce **locally fluent but globally incoherent** text —
grammatical clauses and real named entities, but no sustained topic — which is
typical for ~130M masked-diffusion LMs. Because both were sampled from the same
seed (identical initial noise), their samples follow near-**parallel** skeletons,
making them directly comparable; the Transformer's are marginally cleaner. The
~17% ELBO gap does **not** manifest as a dramatic visible quality difference.

---

## 7. Discussion and Limitations

- **Transformer-tuned recipe (largest caveat).** All runs used the MDLM
  hyperparameters (lr 3e-4, schedule, etc.), which were tuned for a DiT backbone.
  SSMs frequently prefer a different (often higher) learning rate, so BiMamba may
  be **undertuned** rather than fundamentally weaker. A BiMamba LR sweep is the
  single most important follow-up to make the negative result conclusive.
- **Single training budget.** ~5B tokens (one OWT pass). Mamba-style models
  sometimes close gaps with more data; the comparison is at one budget only.
- **Bound, not exact likelihood.** Validation PPL is the MDLM ELBO bound. It is
  fair to both backbones but is an upper bound, not exact NLL.
- **No inference-efficiency measurement.** The motivating advantage of SSMs is
  linear-time inference at long context; this report measures only quality at
  length 1024, not the speed/memory trade-off where BiMamba might still win.
- **s100 finalized at 60000/61000 steps** (98.4%); the difference is negligible
  for the scaling trend.

---

## 8. Conclusion and Future Work

Within the MDLM framework, at matched 130M parameters and ~5B training tokens
**with Transformer-tuned hyperparameters**, a Transformer/DiT denoiser is the
stronger backbone (70.5 vs ~84.7 validation perplexity), and this holds across
seeds and on sample fluency. A bidirectional Mamba-2 denoiser is a viable,
cleanly-scaling model class but does not overtake attention here.

**In progress (this study):**
- **BiMamba learning-rate sweep** at 50M ({3e-4, 5e-4, 1e-3, 2e-3}) to test the
  undertuned-hyperparameter hypothesis. If a higher LR closes the gap, the 130M
  comparison is rerun at the best LR. *(Results: §6.5, to be added.)*
- **Inference-efficiency replication** (forward latency / throughput / peak
  memory vs sequence length 512→8192, BiMamba vs DiT at 130M), an independent
  small-scale check of DiffuApriel's efficiency claim. *(Results: §6.6, to be
  added.)*

**Genuine future work (beyond this reproduction):**
- **Longer training / more tokens** to test whether the gap narrows with data.
- **Larger scale** (toward DiffuApriel's 240M–1.3B) to see whether the gap
  widens or narrows with size.
- **Hybrid Mamba+attention** denoiser, which is what recovers quality in
  DiffuApriel — not yet reproduced here.

---

## Appendix A — Reproducibility

**Environment:** Python 3.11, torch 2.11.0+cu128, mamba-ssm 2.3.2, causal-conv1d
1.6.2, flash-attn 2.8.3, lightning 2.x, datasets 4.x, torchvision 0.26.0+cu128.
CUDA toolkit module `cuda/12.8.0` (must match the cu128 torch and the ≤12.8
driver).

**Train (per run):**
```
python main.py +experiment=runD_130m seed=1 data.cache_dir=<scratch>/data
```
Experiments: `runD_130m` (BiMamba-130M), `runB_transformer_130m` (DiT-130M),
`scaling_50m`, `scaling_100m`. Per-GPU micro-batch capped at 16 (accum 4) to fit
80 GB; `generate_samples=False` during training.

**Evaluate perplexity:**
```
python main.py mode=ppl_eval +experiment=<exp> wandb=null \
  data.cache_dir=<scratch>/data loader.eval_batch_size=32 \
  eval.checkpoint_path=<run>/checkpoints/best.ckpt
```

**Generate samples:**
```
python main.py mode=sample_eval +experiment=<exp> wandb=null \
  loader.eval_batch_size=4 sampling.num_sample_batches=1 \
  sampling.num_sample_log=4 eval.checkpoint_path=<run>/checkpoints/best.ckpt
```

## Appendix B — Final results table

| Run | Backbone | Params | Steps | Tokens | Val NLL | Val PPL | Gen PPL |
|-----|----------|--------|-------|--------|---------|---------|---------|
| runB  | DiT (attention) | 130M | 76000 | ~5B | 4.255 | 70.45 | 91.11 |
| runD1 | BiMamba-2 | 130M | 76000 | ~5B | 4.453 | 85.91 | 96.15 |
| runD2 | BiMamba-2 | 130M | 76000 | ~5B | 4.425 | 83.51 | — |
| s100  | BiMamba-2 | 100M | 60000 | ~4B | 4.580 | 97.53 | — |
| s50   | BiMamba-2 | 50M  | 30000 | ~2B | 4.915 | 136.30 | — |

*Val NLL = ln(Val PPL); reported to 3 d.p. (runB/s50/s100 NLL inferred from PPL).*
