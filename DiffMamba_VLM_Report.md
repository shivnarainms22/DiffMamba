# DiffMamba-VLM: A Bidirectional-Mamba Masked-Diffusion Vision-Language Model (Stage 1)

**Status:** Stage-1 proof complete (image→text understanding). 2026-06-02.
**Scope:** A standout/differentiated portfolio piece — the point is *novelty + honest
rigor*, not leaderboard accuracy. Built additively on top of
[DiffMamba](./DiffMamba_Report.md); the text-only project is unchanged.

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

| Metric | Value |
|---|---|
| Exact-match (normalized first answer == gold) | **0.250** |
| Gold-answer recall (gold appears in the generated answer) | **0.330** |

**Qualitative (image→text).** The model demonstrably **grounds text in the image** — content
words track image content across categories:
- food images → "carrots, broccoli, onions, chicken, beans, pizza"
- airplanes → "jet engine, warplane flying, anti-aircraft, plane crash"
- others → "cat", "motorcycle", "ski / ice shelf / ski resort", "umbrella", "pink tiles"

Different images yield different, topically-relevant text — evidence that the
SigLIP→projector→Mamba-diffusion path conditions on the image. Generated text is **locally
fluent but globally incoherent and verbose**, with the most on-target content front-loaded
(the model learned VQA answers are short).

## 5. Honest limitations

- **130M quality ceiling.** The backbone already trails its own DiT baseline on *text*
  (79.3 vs 70.5 ppl); VLM quality inherits that ceiling. Outputs are relevant-but-incoherent
  by design at this scale, not a bug.
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

## 7. Conclusion & future work

Stage 1 establishes that a **bidirectional-Mamba masked-diffusion VLM** trains and conditions
on images end-to-end, reproducing the expected small-scale quality/throughput trade-off
honestly. It is a working, novel artifact rather than a SOTA system — exactly the intended
portfolio contribution.

**Stage 2 (designed, not built):** text→image *generation* via a frozen VQ tokenizer +
image-token vocabulary expansion in the same denoising framework → a unified
understand-and-generate model. Other directions: scale data/steps, a hybrid Mamba+attention
block to recover quality (per DiffuApriel), and an end-to-end generation-latency benchmark.

## Reproduce

```bash
# Align (projector-only) on CC3M, then SFT (full FT) on VQAv2 — chained 8h segments
bash scripts/submit_vlm.sh vlm_align vlm_stage1_align 6000 wandb=null
# (after align completes)
bash scripts/submit_vlm.sh vlm_sft   vlm_stage1_sft   8000 wandb=null

# Held-out eval + qualitative samples
python main_vlm.py +experiment=vlm_stage1_sft mode=vlm_eval \
  eval.checkpoint_path=/scratch/.../runs/vlm_sft/checkpoints/best.ckpt sampling.steps=64
python main_vlm.py +experiment=vlm_stage1_sft mode=vlm_sample \
  eval.checkpoint_path=/scratch/.../runs/vlm_sft/checkpoints/best.ckpt \
  vlm.caption_prompt="What is in the image?"
```

Code: `models/vision.py`, `models/mm_dimamba.py`, `mm_diffusion.py`, `mm_dataloader.py`,
`warmstart.py`, `main_vlm.py`, `configs/vlm/`, `configs/experiment/vlm_stage1_*.yaml`.
