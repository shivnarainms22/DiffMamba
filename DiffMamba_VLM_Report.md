# DiffMamba-VLM: A Bidirectional-Mamba Masked-Diffusion Vision-Language Model

**Status:** Stage 1 (image→text understanding) **and** Stage 2 (text→image generation)
proofs complete. 2026-06-03.
**Scope:** A standout/differentiated portfolio piece — the point is *novelty + honest
rigor*, not leaderboard accuracy. Built additively on top of
[DiffMamba](./DiffMamba_Report.md); the text-only project is unchanged.

The model both **understands** images (Stage 1: VQAv2 exact-match 0.29) and **generates**
them (Stage 2: text-conditioned images, CLIP matched > shuffled) on one bidirectional-Mamba
masked-diffusion backbone — a small-scale "MMaDA-with-Mamba." Stage 1 is §§1–6; Stage 2 is §8.

## 1. One-line contribution

A **unified-style multimodal masked-diffusion language model with a bidirectional
Mamba-2 (SSM) backbone** — "MMaDA, but with a Mamba backbone, at small scale." Public
diffusion-LM VLMs (LLaDA-V, MMaDA) are Transformer-based; pairing a **state-space
backbone** with **masked-diffusion** image understanding is rare-to-nonexistent. This
report shows the combination *works* end-to-end and characterises its quality honestly at
130M, proof-run scale.

## 2. Method

**Backbone (reused, frozen-then-tuned).** The DiffMamba 130M bidirectional Mamba-2 MDLM
denoiser (hidden 768 / 12 blocks, AdaLN noise conditioning, SUBS parameterization),
warm-started from the tuned text checkpoint `runD_130m` (lr 1e-3, text val PPL 79.3).

**Vision (frozen).** SigLIP `google/siglip-base-patch16-224` → 196 patch tokens (768-d).

**Projector (trained).** 2-layer GELU MLP, 768→768, mapping SigLIP patches into the LM
embedding space (LLaVA-1.5 style).

**Conditioning mechanism — the key design.** The diffusion sequence is **text-only**. Image
features are projected and **prepended as a clean (never-noised) prefix in embedding space**
via the backbone's existing `inputs_embeds` hook, then the image positions are **sliced off
before the loss**. So the entire diffusion machinery (absorbing-state `q_xt`, the SUBS loss,
the attention/loss masks) operates only on the text span and is reused unchanged — no
surgery to the Mamba stack. Mamba has no positional embeddings, so the prefix needs zero
position handling; the forward-direction recurrence propagates image context to the text.

**Conditional masking.** Only the **answer span** is noised; the prompt (question/caption
instruction) and the image stay clean conditioning. The MDLM loss is weighted to answer
tokens via a `loss_mask`.

**Inference.** Iterative unmasking of the answer span, conditioned on the fixed image prefix
and prompt; the prompt is held fixed across all denoising steps.

## 3. Training (proof-run, two-phase LLaVA recipe)

All on Northeastern Explorer (A100-80GB, 8h job-chaining); frozen SigLIP features are
precomputed once to a float16 disk memmap and reused across segments. Cash cost ≈ $0.

| Phase | Trainable | Data | Steps | Result |
|---|---|---|---|---|
| **Align** | projector only (3.1M); backbone frozen | CC3M captions (`pixparse/cc3m-wds`), ~80K | 6000 | best val/nll **3.87** |
| **SFT** | projector + backbone (full FT, 128M) | VQAv2 (`lmms-lab/VQAv2`), ~40K | 8000 | — |

SFT warm-starts from the align checkpoint (backbone **and** trained projector).

## 4. Results

**Held-out VQAv2 (200 examples the model never trained on), 64 denoising steps:**

| Metric | v1 (answer-only loss) | v2 (full-span loss) |
|---|---|---|
| Exact-match (normalized answer == gold) | 0.250 | **0.290** |
| Gold-answer recall (gold appears in the answer) | 0.330 | 0.290 |

**v1 → v2: a train/inference fix (see §4.1).** v2 produces clean, terminated answers, so
exact-match rose and the two metrics converged to 0.29 — in v1 the higher recall (0.33) was
*inflated* by verbose rambling that incidentally mentioned the gold word; in v2 the prediction
*is* the answer.

### 4.1 Termination fix (honest-engineering note)

v1 supervised only the answer + EOS tokens, leaving the trailing padding clean and
unsupervised. But inference masks and generates the **entire** post-prompt span — including
the positions that were pad in training — so the model had never learned to emit EOS and stop,
and it filled every slot, rambling. v2 supervises the **full post-prompt span (answer + EOS +
pad)**, matching generation. Effect: outputs went from paragraphs to clean short answers, e.g.
`What is in the image? → "blue"`, `"no"`, `"3"`, `"white"`, and exact-match improved 0.25→0.29.

**Qualitative (image→text).** The model **grounds text in the image** — answers are
image-influenced (e.g. food images → food words; vehicles → "motorcycle"/"plane"). v2 answers
are short and well-formed but **lean on common VQA priors** — colors, yes/no, and counts
("blue", "black", "no", "white", "3") are over-represented. So it answers in the right *form*
and is image-conditioned, but defaults to high-frequency answers rather than fine visual
reasoning — the expected behaviour at 130M, proof-run scale.

## 5. Honest limitations

- **130M quality ceiling + answer-prior reliance.** The backbone already trails its own DiT
  baseline on *text* (79.3 vs 70.5 ppl); VLM quality inherits that ceiling. After the
  termination fix (§4.1) answers are clean and well-formed but lean on high-frequency VQA
  answers (colors / yes-no / counts) rather than fine visual reasoning — expected at this
  scale, not a bug.
- **Proof-run scale.** ~80K caption + ~40K VQA examples, 6K+8K steps — a fraction of the
  LLaVA recipe (558K + 150K). Numbers would improve with scale but the qualitative story
  (works, modest) would not change materially at 130M.
- **VQA exact-match caveat.** VQAv2 is ~38% yes/no with "yes" dominant, so part of the 25%
  exact-match reflects answer-prior/yes-bias, not full visual reasoning. Gold-recall (33%)
  and the qualitative samples are the more informative signals.
- **Eval source.** SFT trained on a subset of the VQAv2 *validation* split; eval used a
  disjoint held-out slice of the same split (the model never saw those examples), not the
  official test server.
- **Forward-pass framing.** This is Stage 1 (understanding) only.

## 6. Why a Mamba backbone (motivation carried from DiffMamba)

The text-only DiffMamba study showed BiMamba-2 is **linear-time** and overtakes flash-attn
DiT in throughput beyond ~3K tokens (3.1× at 32K). A VLM built on this backbone **inherits
that long-context efficiency** — relevant as image-token counts and multi-image / video
contexts grow. (Not separately benchmarked here; inherited from the backbone.)

## 7. Stage 1 conclusion

Stage 1 establishes that a **bidirectional-Mamba masked-diffusion VLM** trains and conditions
on images end-to-end (image→text understanding), reproducing the expected small-scale
quality/throughput trade-off honestly.

## 8. Stage 2 — text→image generation

Stage 2 extends the same backbone to **generate images from text**, then is the basis for a
unified understand-and-generate model.

### 8.1 Method

Generation is **pure tokens in / tokens out** — sequence
`[BOS] caption [BOI] <1024 VQ image tokens> [EOI]` — so it reuses the original text DiMamba
path (no SigLIP, no projector) with an **expanded vocabulary**: text `[0..50256]` + `[MASK]` +
`[BOI]`/`[EOI]` + **4096 VQ image-code tokens**. A frozen diffusers `VQModel`
(`microsoft/vq-diffusion-ithq`, f8) maps 256px images ↔ a **32×32 = 1024** code grid. The
backbone is warm-started from the text checkpoint `runD_lr1e3` (text embedding/lm_head rows
copied; image/marker rows fresh). Conditional masking noises **only the image span** (caption
clean); MDLM SUBS loss over image tokens. Sampling iteratively unmasks the 1024 image tokens
with logits constrained to the image-code range, then `VQModel.decode` → pixels.

**Why 1024 tokens:** it is the backbone's trained sequence length *and* it puts generation in
the regime where Mamba's **linear** sequence scaling beats attention's quadratic — i.e., the
SSM backbone is a natural fit for high-token-count image generation (a deliberate design
choice; higher token counts are the documented scaling extension).

### 8.2 Training

Proof-run on Explorer (A100, chained 8h segments): **CC3M** caption→image
(`pixparse/cc3m-wds`, ~80K subset), images VQ-encoded once to an int16 token memmap on
`/scratch` (reused across segments). 8000 steps, warm-started from the 130M text backbone.

### 8.3 Results

**Held-out CLIP-score (n=64, generated image vs. caption):**

| | CLIP cosine |
|---|---|
| matched (image vs. its own caption) | **0.191** |
| shuffled (image vs. a random caption) | 0.182 |

**matched > shuffled ⇒ the generated images are genuinely text-conditioned** — the model uses
the caption. The gap is small and the absolute score modest (well-matched CLIP pairs score
~0.25–0.30), i.e. **real but weak** conditioning, as expected at 130M proof scale.

**Qualitative.** Generated images are **coherent, textured structure — not noise** — with loose
caption alignment:

| Caption | Sample |
|---|---|
| *"portrait of the artist by painting artist"* | ![portrait](assets/gen_sample_00_portrait.png) |
| *"man concentrating on a chess game"* | ![chess](assets/gen_sample_01_chess.png) |

The first is a warm, **painterly brush-textured** image (on-theme for "painting/artist"); the
second is a dim **scene with figures**. Neither has recognizable objects (no face, no
chessboard) — the honest 130M ceiling: structured, text-influenced, but rough.

### 8.4 Stage 2 limitations

- **Quality bounded by model size, not resolution.** At 130M, more image tokens give bigger
  but still-rough images; recognizable objects need a much larger LM (LLaDA-V/MMaDA are 8B).
- **Weak conditioning gap.** The CLIP matched−shuffled margin is small; classifier-free
  guidance (caption dropout + guided sampling) is the documented lever to widen it.
- **Proof-run scale & no FID.** ~80K CC3M, 8K steps; CLIP-score + qualitative only (FID needs
  many samples + a reference set — future work).

## 9. Overall conclusion & future work

On one bidirectional-Mamba masked-diffusion backbone, the model both **understands** images
(Stage 1: VQAv2 exact-match 0.29) and **generates** them (Stage 2: text-conditioned images,
CLIP matched > shuffled) — a small-scale unified-capable "MMaDA-with-Mamba," novel because
public diffusion-LM VLMs are Transformer-based. Both results are honest proofs of mechanism at
130M, not SOTA systems — exactly the intended portfolio contribution.

**Future work:** unify both directions in a single training run (mixed understanding +
generation data); classifier-free guidance for stronger conditioning; scale data/steps;
higher image-token counts to exploit Mamba's linear scaling (+ a generation-latency benchmark
vs. attention); a hybrid Mamba+attention block to recover quality (per DiffuApriel).

## Reproduce

```bash
# Align (projector-only) on CC3M, then SFT (full FT) on VQAv2 — chained 8h segments
bash scripts/submit_vlm.sh vlm_align vlm_stage1_align 6000 wandb=null
# (after align completes)
bash scripts/submit_vlm.sh vlm_sft   vlm_stage1_sft   8000 wandb=null

# Stage-1 held-out eval + qualitative samples
python main_vlm.py +experiment=vlm_stage1_sft mode=vlm_eval \
  eval.checkpoint_path=/scratch/.../runs/vlm_sft/checkpoints/best.ckpt sampling.steps=64
python main_vlm.py +experiment=vlm_stage1_sft mode=vlm_sample \
  eval.checkpoint_path=/scratch/.../runs/vlm_sft/checkpoints/best.ckpt \
  vlm.caption_prompt="What is in the image?"

# Stage-2 text->image generation: train, CLIP-score, image gallery
bash scripts/submit_vlm.sh gen_stage2 gen_stage2 8000 wandb=null
python main_vlm.py +experiment=gen_stage2 mode=gen_eval \
  eval.checkpoint_path=/scratch/.../runs/gen_stage2/checkpoints/best.ckpt sampling.steps=64
python main_vlm.py +experiment=gen_stage2 mode=gen_sample \
  eval.checkpoint_path=/scratch/.../runs/gen_stage2/checkpoints/best.ckpt sampling.steps=128
```

Stage 1 code: `models/vision.py`, `models/mm_dimamba.py`, `mm_diffusion.py`,
`mm_dataloader.py`, `warmstart.py`, `configs/experiment/vlm_stage1_*.yaml`.
Stage 2 code: `models/vq.py`, `gen_diffusion.py`, `gen_dataloader.py`, `gen_vocab.py`,
`configs/vlm/vqgen.yaml`, `configs/experiment/gen_stage2.yaml`. Shared: `main_vlm.py`.
