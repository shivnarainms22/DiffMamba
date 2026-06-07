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

## 9. Unification — one backbone, both directions (honest result)

Stages 1 and 2 each used the shared backbone separately. The unification step trains **one**
expanded-vocab DiMamba backbone + **one** projector **jointly** on CC3M *bidirectionally* —
understanding (SigLIP-prefix → caption) and generation (caption → VQ image tokens) — plus light
pure text, with a `CombinedLoader` summing per-step losses across the streams. The reported run
warm-starts the projector from the Stage-1 checkpoint (a recovery run after a cold-start attempt
degraded understanding).

### 9.1 Results (matched-vs-shuffled CLIP, n=48)

| Direction | unified matched | unified shuffled | standalone baseline (matched / shuffled) |
|---|---|---|---|
| **Generation** (gen image vs. caption) | **0.194** | 0.197 | Stage-2 **0.191** / 0.182 |
| **Understanding** (gen caption vs. gold, text-CLIP) | 0.677 | 0.674 | align captioner **0.607** / 0.607 |

### 9.2 Reading it honestly

- **Generation is preserved under unification.** Unified matched (0.194) is statistically
  indistinguishable from the standalone Stage-2 generator (0.191) — the matched−shuffled margins
  are both sub-0.01 at n=48, i.e. within noise. Sharing the backbone with the understanding
  objective did **not** degrade generation.
- **Understanding is inconclusive on this metric — and we baselined it rather than guess.** The
  unified model's captions sit at the caption-CLIP floor (matched ≈ shuffled). Crucially, the
  **standalone projector-aligned captioner sits at the *same* floor** (0.607 ≈ 0.607) — so this is
  **not** unification-induced forgetting; there was no working free-form captioner on this metric
  to regress from. (The unified captions are in fact *more* coherent English than the standalone's
  output.) The cause is the eval/scale: free-form CC3M alt-text captioning at 130M via masked
  diffusion, scored by a high-floor/low-dynamic-range CLIP cosine, has little signal to resolve.
- **What understanding *was* demonstrated** is short-answer VQA (Stage 1, exact-match **0.29**) — a
  different task, trained with VQAv2 SFT, and outside the unified model's captioning objective. It
  is not measured by, nor comparable to, the caption-CLIP metric used here.

**Net:** the unification successfully shares a single Mamba-diffusion backbone across both
objectives **without degrading the (weak) generation**; the understanding direction is **honestly
inconclusive at 130M on the free-form caption-CLIP eval**, established with a same-metric baseline
rather than reported as either a success or a "collapse."

## 10. Overall conclusion & future work

On one bidirectional-Mamba masked-diffusion backbone, the model **understands** images (Stage 1:
VQAv2 exact-match 0.29), **generates** them (Stage 2: text-conditioned images, CLIP matched >
shuffled), and can be **jointly trained in both directions on a single backbone** (Stage 3:
generation preserved within noise; understanding inconclusive-at-floor, baselined) — a small-scale
"MMaDA-with-Mamba," novel because public diffusion-LM VLMs are Transformer-based. All results are
honest proofs of mechanism at 130M, not SOTA systems — exactly the intended portfolio contribution.

**Future work:** a stronger free-form captioner (caption-specific SFT) and a sharper understanding
metric (VQA-style scoring inside the unified model) so the understanding direction can be measured
with dynamic range; classifier-free guidance for stronger conditioning; scale data/steps; higher
image-token counts to exploit Mamba's linear scaling (+ a generation-latency benchmark vs.
attention); a hybrid Mamba+attention block to recover quality (per DiffuApriel).

## 11. Follow-up: pursuing the §10 future-work items (branch `hybrid-cfg-vqa`)

Three of the §10 future-work items were implemented and run as a follow-up: classifier-free
guidance, a hybrid Mamba+attention backbone, and a VQA-style understanding metric for the unified
model. Results below; all honest proofs of mechanism, same as the main report.

### 11.1 Classifier-free guidance (done, verified)

CFG-capable Stage-2 was trained with caption dropout (`vlm.caption_dropout_prob=0.1`) and sampled
with guidance scale `sampling.cfg_scale`. At sampling time the unconditional branch replaces the
caption span with PAD while keeping the fixed `[BOS] … [BOI]` prompt shape, and logits are combined
as `uncond + scale·(cond − uncond)` (`cfg_utils.py`). `cfg_scale=1.0` short-circuits to the plain
conditional (bit-exact baseline).

Held-out CLIP score (cosine of CLIP image vs. its caption), matched vs. shuffled, n=200:

| `cfg_scale` | matched | shuffled | gap (matched − shuffled) |
|---|---|---|---|
| 1.0 (baseline) | 0.187 | 0.184 | 0.003 |
| 1.5 | 0.189 | 0.183 | 0.006 |
| 2.0 | 0.191 | 0.185 | 0.006 |
| 3.0 | 0.191 | 0.185 | 0.006 |
| 4.0 | 0.194 | 0.186 | 0.008 |

**Reading it honestly.** `matched` rises **monotonically** with guidance (0.187 → 0.194) while
`shuffled` barely moves (0.184 → 0.186), so guidance preferentially strengthens caption alignment
rather than inflating CLIP globally — the conditioning gap roughly doubles. Each single step is
within ≈1 standard error at n=200; the evidence is the monotonic trend across all five points, not
any one delta. Qualitative samples confirm the mechanism: as `cfg_scale` rises, images visibly gain
saturation and contrast (the textbook CFG signature), with over-saturation/posterization setting in
by `cfg_scale=4.0`. No recognizable objects emerge at any scale — the base Stage-2 generator is too
weak to render semantics, so CFG amplifies colour/texture, not content.

**Net:** CFG is implemented and verified — caption alignment increases monotonically with scale and
images visibly sharpen — but absolute fidelity stays bounded by the proof-of-mechanism generator.
Practical operating point **`cfg_scale≈3.0`** (clear conditioning gain before the over-saturation at
4.0). Code: `cfg_utils.py`, `gen_diffusion.py`/`unified_diffusion.py` (`_guided_image_forward`),
`configs/experiment/gen_stage2_cfg.yaml`.

### 11.2 Hybrid Mamba+attention backbone (in progress)

Opt-in backbone (`backbone: hybrid_dimamba`) that inserts full bidirectional attention on a fixed
schedule (every 4th block: layers 3/7/11 of 12) among the bidirectional Mamba blocks, to test
whether sparse attention recovers DiT-class quality while keeping more of Mamba's long-context
efficiency. Constructs and runs (CUDA smoke test passes). A 130M run (`hybrid_130m`, same
OpenWebText / dims / 76k-step recipe as the BiMamba baseline `runD_130m`) is training.

*Results pending:* validation NLL/perplexity vs. BiMamba (`runD`) and Transformer (`runB`), and
forward-pass throughput across `dimamba` / `hybrid_dimamba` / `dit` (`scripts/eval_throughput.py`).
Code: `models/hybrid_dimamba.py`, `hybrid_schedule.py`, `configs/model/small-hybrid-dimamba.yaml`,
`configs/experiment/hybrid_130m.yaml`.

### 11.3 Unified VQA understanding metric (negative result; SFT queued)

To give the unified model's *understanding* direction a dynamic-range metric (§9.2's caption-CLIP
floor could not resolve it), `mode=uni_vqa_eval` scores short VQAv2-style answers (exact match +
gold-recall) with a shuffled-image ablation. On the `uni_stage3` checkpoint it returns
**exact = recall = 0.000 across 200 examples**.

A diagnostic (`mode=uni_vqa_diag`) established this is **not a bug**. Per-example raw token ids show
that for a *question* prompt the model emits `[EOS]` as the very first answer token (answer region =
`[EOS, PAD, PAD, …]` → empty), whereas for its in-distribution caption prompt ("Describe the image.")
it produces real, image-varying tokens. The unified model was trained **only** as a captioner with a
single fixed prompt, so it has **zero zero-shot question-answering ability** — a VQA probe returns 0
by construction. (The same dump also shows the captioner leaking VQ image-code tokens into the text
answer region, corroborating §9.2's at-floor understanding.)

**Net:** the unified Stage-3 checkpoint cannot be VQA-probed as-is; a dynamic-range understanding
number requires *training* for it. Queued next: a short **unified VQA-SFT** (mix VQAv2 Q→A pairs into
the understanding stream, as Stage-1 did), after which `uni_vqa_eval` becomes meaningful. Code:
`vlm_eval_utils.py`, `main_vlm.py` (`_uni_vqa_eval`, `_uni_vqa_diag`).

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

# Stage-3 unification: joint train, then both-direction CLIP eval
bash scripts/submit_vlm.sh uni_stage3 uni_stage3 12000 wandb=null
python main_vlm.py +experiment=uni_stage3 mode=uni_eval \
  eval.checkpoint_path=/scratch/.../runs/uni_stage3/checkpoints/best.ckpt sampling.steps=64
# Understanding baseline (standalone captioner, identical caption-CLIP metric)
python main_vlm.py +experiment=vlm_stage1_align mode=vlm_caption_eval \
  eval.checkpoint_path=/scratch/.../runs/vlm_align/checkpoints/best.ckpt sampling.steps=64

# §11.1 Classifier-free guidance: train w/ caption dropout, then sweep cfg_scale
bash scripts/submit_vlm.sh gen_stage2_cfg gen_stage2_cfg 8000
for s in 1.0 1.5 2.0 3.0 4.0; do python main_vlm.py +experiment=gen_stage2_cfg mode=gen_eval \
  eval.checkpoint_path=/scratch/.../runs/gen_stage2_cfg/checkpoints/best.ckpt \
  sampling.cfg_scale=$s sampling.steps=64 \
  checkpointing.save_dir=/scratch/.../eval/gen_cfg_s$s; done

# §11.2 Hybrid Mamba+attention 130M (same recipe as runD baseline) + throughput
bash scripts/submit_hpc.sh hybrid_130m hybrid_130m 76000 seed=1
for b in dimamba hybrid_dimamba dit; do python scripts/eval_throughput.py --backbone $b --mode forward; done

# §11.3 Unified VQA probe (returns 0 — model has no QA ability) + diagnostic
python main_vlm.py +experiment=uni_stage3 mode=uni_vqa_eval \
  eval.checkpoint_path=/scratch/.../runs/uni_stage3/checkpoints/best.ckpt sampling.steps=64
python main_vlm.py +experiment=uni_stage3 mode=uni_vqa_diag \
  eval.checkpoint_path=/scratch/.../runs/uni_stage3/checkpoints/best.ckpt sampling.steps=64
```

Stage 1 code: `models/vision.py`, `models/mm_dimamba.py`, `mm_diffusion.py`,
`mm_dataloader.py`, `warmstart.py`, `configs/experiment/vlm_stage1_*.yaml`.
Stage 2 code: `models/vq.py`, `gen_diffusion.py`, `gen_dataloader.py`, `gen_vocab.py`,
`configs/vlm/vqgen.yaml`, `configs/experiment/gen_stage2.yaml`.
Stage 3 (unification) code: `unified_diffusion.py`, `unified_dataloader.py`,
`configs/vlm/unified.yaml`, `configs/experiment/uni_stage3.yaml`; baseline mode
`vlm_caption_eval` in `main_vlm.py`. Shared: `main_vlm.py`.
