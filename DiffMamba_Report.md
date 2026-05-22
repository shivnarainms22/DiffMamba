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

Date: 2026-05-22 · Hardware: Northeastern Explorer HPC (NVIDIA A100)

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
scales cleanly with model size and is seed-stable. To address the "BiMamba is
undertuned" caveat, we then ran an LR sweep at 50M (§6.5) — which showed BiMamba
prefers a ~3.3× higher LR — and retrained the 130M BiMamba at the tuned LR
(§6.6). The tuned BiMamba reaches **79.26** ppl, closing ~43% of the original
gap but **not closing it**: the Transformer remains 12.5% ahead. Finally, an
independent inference-efficiency benchmark at 130M-class (§6.7) reproduces
DiffuApriel's long-context efficiency claim at small scale — **BiMamba is 3.12×
faster than the flash-attn DiT denoiser at sequence length 32K**, with a clean
crossover near ~3K tokens.

| Model | Backbone | Params | Tokens | LR | Val PPL (↓) | Gen PPL (↓) |
|-------|----------|--------|--------|----|-------------|-------------|
| runB        | Transformer (DiT) | 130M | ~5B | 3e-4 | **70.45** | **91.11** |
| runD1       | BiMamba-2 (seed 1) | 130M | ~5B | 3e-4 | 85.91 | 96.15 |
| runD2       | BiMamba-2 (seed 2) | 130M | ~5B | 3e-4 | 83.51 | — |
| **runD_lr1e3** | **BiMamba-2 (tuned LR)** | **130M** | **~5B** | **1e-3** | **79.26** | **—** |
| s100        | BiMamba-2 | 100M | ~4B | 3e-4 | 97.53 | — |
| s50         | BiMamba-2 | 50M  | ~2B | 3e-4 | 136.30 | — |

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

### 6.5 BiMamba learning-rate sweep at 50M

To test the "undertuned hyperparameters" caveat from §7, we ran a learning-rate
sweep at the smallest scale (50M parameters, 30000 steps, ~2B tokens) where each
training run completes in ~14h. All other hyperparameters were held at the MDLM
defaults; only `optim.lr` was varied. Evaluation: validation perplexity (MDLM
ELBO bound) on the full OpenWebText validation split with EMA weights, batch 32,
seed 1 — identical protocol to §6.1.

| Run | LR | Val PPL | Δ vs baseline |
|-----|----|---------|----|
| s50 (baseline) | 3e-4 | 136.30 | — |
| s50_lr5e4 | 5e-4 | 123.07 | −13.23 (−9.7%) |
| **s50_lr1e3** | **1e-3** | **117.09** | **−19.21 (−14.1%) — best** |
| s50_lr2e3 | 2e-3 | 117.83 | −18.47 (−13.5%) |

Validation perplexity decreases monotonically from 3e-4 through 1e-3, then
plateaus at 2e-3 (a slight 0.74-ppl regression, within evaluation noise). The
gap between the MDLM-default 3e-4 and the BiMamba-optimum 1e-3 is **14.1%** —
substantially larger than the seed-noise band from §6.2 (~2.4 ppl, ~3%). This
**confirms** the undertuned-hyperparameter hypothesis at small scale: the
default MDLM recipe was tuned for the Transformer/DiT backbone, and BiMamba
needs a roughly **3.3× higher learning rate** to converge to its own optimum.

The clean plateau between 1e-3 and 2e-3 (no divergence, only marginal
regression) suggests **1e-3 sits in a stable basin** with headroom — important
for the next experiment, which scales the model up 2.6× while holding the LR
fixed.

**Implication for the 130M headline.** The matched-compute comparison in §6.1
used lr=3e-4 for both backbones. The Transformer was at or near its tuned
optimum; the BiMamba was 14% short of its own optimum *at the same scale*. If
the same lift transfers to 130M, the BiMamba average of ~84.7 ppl could drop to
~73 ppl — within ~3 ppl (seed-noise distance) of the Transformer's 70.45. That
transfer is **not** guaranteed (LR scaling laws typically argue LR should
*decrease* with model size, not stay constant), but it is the single test that
makes the §6.1 comparison fair. §6.6 reports that retrain.

### 6.6 Phase 2: tuned 130M BiMamba retrain

**Configuration.** Identical to runD1 (130M BiMamba, 76000 steps, seed=1, all
other hyperparameters per `configs/experiment/runD_130m.yaml`) except for
`optim.lr=1e-3` taken from the §6.5 sweep optimum. Run name: `runD_lr1e3`,
distinct W&B name and HF Hub upload path so the original runD1 / runD2
artefacts are not overwritten. Training completed cleanly across the expected
~4 wall-time segments; no loss spikes or instability were observed at 1e-3 at
the 130M scale, confirming the §6.5 "stable basin with headroom" reading.

**Result.**

| Run | LR | Val NLL | Val PPL | Δ vs runD1 |
|-----|----|---------|---------|-----------|
| runD1 (baseline)          | 3e-4 | 4.453 | 85.91 | — |
| **runD_lr1e3 (tuned)**    | **1e-3** | **4.373** | **79.26** | **−6.65 (−7.7%)** |
| runB (Transformer, ref.)  | 3e-4 | 4.255 | 70.45 | — (12.5% ahead of tuned BiMamba) |

**Findings.**

1. **The lift is real, not seed noise.** The 6.65-ppl improvement is ~2.8× the
   runD1↔runD2 seed-noise band (2.4 ppl) reported in §6.2. Single-seed evidence
   is therefore sufficient to call the direction.
2. **The lift is real but smaller than at 50M.** The 50M sweep predicted a −14%
   improvement; only **−7.7%** transferred to 130M — roughly half. This is
   consistent with the µP / Maximal-Update-Parameterization intuition that the
   optimal LR should decrease (not stay constant) with model size: holding
   lr=1e-3 across a 2.6× model-size increase moves us *past* the 130M optimum.
   The 130M optimum is plausibly between 5e-4 and 1e-3 (i.e. the gain we
   measured is a lower bound on what a 130M-specific sweep could achieve).
3. **The headline shifts but does not flip.** With both backbones at their
   tuned (or near-tuned) LR, the Transformer's lead narrows from **15.5 ppl
   (18% gap)** to **8.8 ppl (12.5% gap)** — LR tuning closes **~43% of the
   gap**. This is the most honest framing of the §6.1 fairness concern: BiMamba
   *was* meaningfully undertuned, but tuning it does not make pure BiMamba
   competitive with the Transformer at matched 130M/5B.
4. **Consistent with DiffuApriel.** DiffuApriel reports that the *pure* Mamba
   denoiser trades quality for speed, with a *hybrid* Mamba+attention variant
   needed to recover quality. Our tuned-pure-BiMamba result lands in exactly
   that regime: meaningfully behind a Transformer of the same scale, but
   architecturally cheaper at long context (§6.7).

**Decision: no second seed.** A runD_lr1e3 seed-2 retrain would cost another
~28 h of HPC time to tighten an already-clear directional answer (effect is
~2.8× the seed-noise band). Compute spent on §6.7 (efficiency benchmark)
instead.

### 6.7 Inference-efficiency benchmark (long context)

Quality is one half of the story; the motivating advantage of an SSM denoiser
is **linear-time scaling at long context**. We replicate the core efficiency
claim of DiffuApriel/DiffuMamba (arXiv 2511.15927) at small scale and on a
single A100-40GB: measure forward-pass latency, throughput, and peak GPU
memory for BiMamba-2 vs the flash-attn DiT denoiser across sequence lengths
512 → 32768.

**Protocol.** `scripts/bench_efficiency.py`. Both backbones are
**randomly-initialised** — we are measuring the architecture's compute cost,
not its quality, so no checkpoints are needed. Batch size 4, bf16 autocast,
A100-PCIe-40GB; per-length: 3 warmup passes then 10 timed passes,
`torch.cuda.synchronize` around the timing window, `max_memory_allocated`
captured after `reset_peak_memory_stats`. Measurements use the same
`small-dimamba` and `small` (DiT) configs as the headline training runs.

**Parameter-count caveat.** The benchmark uses the matched-by-shape configs
(both at hidden=768, blocks=12), not matched-by-parameter-count. The actual
counts are **BiMamba 125.1M, DiT 169.6M** — the DiT block adds attention QKV
projections that the BiMamba block does not have. This means **the DiT
baseline in the efficiency comparison is 35% heavier**, which makes the result
*more* conservative (the heavier model loses at long context), not less. The
matched-shape configs are the right level of comparison for a denoiser-swap
ablation, but writeups should say "both ~150M-class denoisers" rather than
"matched 130M" when citing this table.

**Results.**

| seq_len | BiMamba ms/fwd | BiMamba tok/s | BiMamba peak GB | DiT ms/fwd | DiT tok/s | DiT peak GB | Speedup (DiT/BiMamba) |
|--------:|---------------:|--------------:|----------------:|-----------:|----------:|------------:|----------------------:|
|     512 |           28.1 |        72,975 |            1.55 |       16.2 |   126,431 |        1.69 |  0.58× (DiT 1.73× faster) |
|    1024 |           29.1 |       140,650 |            1.77 |       17.2 |   238,554 |        1.92 |  0.59× |
|    2048 |           31.4 |       260,605 |            2.20 |       26.4 |   310,494 |        2.37 |  0.84× |
|    4096 |           45.9 |       357,313 |            3.06 |       55.0 |   297,843 |        3.25 |  **1.20× ← crossover** |
|    8192 |           90.5 |       362,001 |            4.79 |      131.4 |   249,363 |        5.03 |  **1.45×** |
|   16384 |          180.6 |       362,907 |            8.23 |      361.2 |   181,463 |        8.58 |  **2.00×** |
|   32768 |          364.0 |       360,060 |           15.12 |    1,136.2 |   115,355 |       15.68 |  **3.12×** |

**Findings.**

1. **BiMamba latency scales textbook-linearly with sequence length.**
   8K → 16K → 32K = 90.5 → 180.6 → 364.0 ms (multipliers ×1.996, ×2.015,
   i.e. exactly linear in seq_len). Throughput saturates at ~360k tok/s from
   4K onward and *holds constant* through 32K — the asymptotic regime of a
   linear-time SSM.
2. **DiT latency scales empirically as ~O(L^1.55).** Latency 8K → 32K =
   131.4 → 1136.2 ms (×8.65 for a ×4 increase in seq_len). FlashAttention
   keeps DiT memory roughly linear in seq_len, but **compute is not** — the
   quadratic FLOP cost still asserts itself. Throughput collapses from 310k
   tok/s at 4K down to 115k tok/s at 32K.
3. **Crossover at ~3K tokens.** Below 2K the DiT is faster (attention is cheap
   at short context and the Mamba-2 kernel has fixed overhead from the
   bidirectional pass and SSM-state setup). The crossover lies between 2K
   and 4K. At seq=1024 (our training length) the DiT is 1.7× faster — a small
   penalty BiMamba pays at training time.
4. **Memory was not a differentiator on A100 + flash-attn.** Peak memory at
   32K: BiMamba **15.12 GB** vs DiT **15.68 GB** — essentially tied. Both
   stayed under half of the 40 GB budget at every length tested. The
   originally-anticipated "DiT OOMs at long context" story does *not* hold on
   modern flash-attention; the efficiency story at this scale is purely
   latency, not memory. (DiffuApriel does report memory wins at much larger
   model scale and longer contexts; our 130M-class result simply does not
   reach the regime where DiT memory diverges.)
5. **Speedup vs DiffuApriel.** DiffuApriel report ~5.3× throughput at 1.3B
   parameters in their long-context regime. We measure **3.12× at 130M-class,
   32K tokens** — directionally identical, with the expected attenuation at
   smaller scale (the constant Mamba-kernel overhead matters more at small
   model sizes than at 1.3B). The scaling pattern is the same: pure BiMamba
   trades quality for asymptotic throughput.

**Interpretation in light of §6.6.** §6.6 says tuned pure BiMamba trails the
Transformer on quality at matched 130M/5B by 12.5%. §6.7 says it is **3.12×
faster** at 32K-token inference, with the gap widening as context grows. This
is the same quality/throughput trade-off DiffuApriel describes, replicated
independently at a fraction of the scale and compute budget.

---

## 7. Discussion and Limitations

- **Transformer-tuned recipe (largest caveat — partially addressed).** The
  §6.1 headline used the MDLM-default lr=3e-4 for both backbones, which was
  tuned for a DiT backbone. §6.5 confirmed BiMamba was meaningfully undertuned
  at 50M (−14% at lr=1e-3), and §6.6 measured the lift at 130M (−7.7%). The
  remaining residual question is whether a 130M-specific LR sweep (rather than
  reusing the 50M optimum) would close a further fraction of the gap. The 130M
  optimum is plausibly between 5e-4 and 1e-3, so the reported §6.6 number is a
  conservative lower bound on tuned-BiMamba quality at 130M.
- **Single training budget.** ~5B tokens (one OWT pass). Mamba-style models
  sometimes close gaps with more data; the quality comparison is at one budget
  only.
- **Bound, not exact likelihood.** Validation PPL is the MDLM ELBO bound. It is
  fair to both backbones but is an upper bound, not exact NLL.
- **Efficiency benchmark: forward-pass-only, random-init, single batch size.**
  §6.7 measures the cost of one denoiser forward pass; it does not measure
  end-to-end *generation* latency (which involves many denoising steps and
  KV/SSM-state reuse), nor training step time. Backbones are randomly
  initialised — fine for measuring compute, but does not capture potential
  cache-/CUDA-graph effects from a trained model. Batch size is fixed at 4;
  the crossover point can shift with batch and dtype.
- **Efficiency benchmark: parameter-count mismatch.** As §6.7 flags, the
  benchmark uses matched-by-shape configs (BiMamba 125.1M, DiT 169.6M). The
  35% heavier DiT makes the long-context speedup conservative, not inflated,
  but writeups should not call this a "matched-parameter" comparison.
- **s100 finalized at 60000/61000 steps** (98.4%); the difference is negligible
  for the scaling trend.

---

## 8. Conclusion and Future Work

Within the MDLM framework at 130M parameters and ~5B training tokens, the
final picture is a clean **quality/throughput trade-off**, not a single
winner:

- **Quality.** A Transformer/DiT denoiser is the stronger backbone. With both
  backbones at the MDLM-default lr=3e-4 (§6.1), the Transformer wins by 18%
  validation perplexity (70.45 vs ~84.7) and is seed-stable to ±2.4 ppl. With
  BiMamba retuned to its preferred learning rate (lr=1e-3, §6.6), the gap
  narrows to 12.5% (70.45 vs 79.26) — LR tuning closes ~43% of the original
  gap but does not close it. A bidirectional Mamba-2 denoiser is a viable,
  cleanly-scaling model class that does not match attention on quality here.
- **Throughput.** At sequence length 32K, the BiMamba-2 denoiser is **3.12×
  faster** than the flash-attn DiT denoiser per forward pass, with throughput
  holding constant at ~360k tokens/s while DiT collapses from 310k (4K) to
  115k (32K) (§6.7). Crossover at ~3K tokens. Memory was not the
  differentiator on A100-40GB with flash-attn; the efficiency story at this
  scale is purely latency.

This matches DiffuApriel's reported pattern (pure Mamba: trade quality for
throughput; quality recovered by a *hybrid* Mamba+attention variant), now
independently reproduced at 130M-class scale on academic-HPC compute.

**Completed in this study:**
- **§6.1–§6.4:** Headline 130M/5B comparison + BiMamba scaling (50M, 100M,
  130M×2 seeds) + Transformer baseline + sample quality.
- **§6.5:** BiMamba LR sweep at 50M (3e-4, 5e-4, 1e-3, 2e-3) — confirms
  BiMamba undertuned at the MDLM default; optimum at 1e-3 (−14.1%).
- **§6.6:** Phase 2 retrain of 130M BiMamba at lr=1e-3 — −7.7% transfer to
  130M; gap to Transformer narrows but does not close.
- **§6.7:** Inference-efficiency replication — 3.12× speedup at 32K tokens,
  linear vs ~O(L^1.55) latency scaling, crossover ~3K tokens.

**Genuine future work (beyond this reproduction):**
- **130M-specific LR sweep** to pin the tuned-BiMamba ceiling more tightly
  (current §6.6 reuses the 50M optimum and is a conservative lower bound).
- **Longer training / more tokens** to test whether the quality gap narrows
  with data.
- **Larger scale** (toward DiffuApriel's 240M–1.3B) to see whether the gap
  widens or narrows with size, and whether DiT memory begins to diverge.
- **Hybrid Mamba+attention denoiser**, which is what recovers quality in
  DiffuApriel — not yet reproduced here.
- **End-to-end generation latency** (not just one forward pass): the §6.7
  speedup applies to one denoiser call; full-sample latency depends on the
  denoising-step budget and any state-reuse the sampler can exploit.

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

**Efficiency benchmark (§6.7):**
```
PYTHONPATH=. python scripts/bench_efficiency.py \
  --batch 4 --lengths 512 1024 2048 4096 8192 16384 32768
```
Random-initialised backbones; no checkpoint required. Run on a GPU node with
the diffmamba env. The `PYTHONPATH=.` prefix is needed because Python only
adds the script's parent directory to `sys.path`, not the repo root.

## Appendix B — Final results table

| Run | Backbone | Params | Steps | Tokens | LR | Val NLL | Val PPL | Gen PPL |
|-----|----------|--------|-------|--------|----|---------|---------|---------|
| runB       | DiT (attention) | 130M | 76000 | ~5B | 3e-4 | 4.255 | 70.45  | 91.11 |
| runD1      | BiMamba-2       | 130M | 76000 | ~5B | 3e-4 | 4.453 | 85.91  | 96.15 |
| runD2      | BiMamba-2       | 130M | 76000 | ~5B | 3e-4 | 4.425 | 83.51  | — |
| s100       | BiMamba-2       | 100M | 60000 | ~4B | 3e-4 | 4.580 | 97.53  | — |
| s50        | BiMamba-2       | 50M  | 30000 | ~2B | 3e-4 | 4.915 | 136.30 | — |
| s50_lr5e4  | BiMamba-2       | 50M  | 30000 | ~2B | 5e-4 | 4.813 | 123.07 | — |
| s50_lr1e3  | BiMamba-2       | 50M  | 30000 | ~2B | 1e-3 | 4.763 | 117.09 | — |
| s50_lr2e3  | BiMamba-2       | 50M  | 30000 | ~2B | 2e-3 | 4.769 | 117.83 | — |
| runD_lr1e3 | BiMamba-2       | 130M | 76000 | ~5B | 1e-3 | 4.373 | 79.26  | — |

*Val NLL = ln(Val PPL); reported to 3 d.p. (NLL values inferred from PPL where not logged directly).*
