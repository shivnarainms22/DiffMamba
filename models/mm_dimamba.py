"""Multimodal wrapper around the DiMamba diffusion backbone.

Stage-1 understanding VLM: a frozen vision encoder's patch features (provided
pre-projected as `image_features`) are mapped into the LM embedding space by a
trainable MLP projector, prepended as a clean (never-noised) prefix to the text
token embeddings, run through the existing BiMamba-2 + AdaLN denoiser via its
`inputs_embeds` hook, and the image positions are sliced off so only the text
logits leave the backbone. The diffusion machinery (q_xt, SUBS loss,
attention_mask) operates only on the text span and is reused unchanged.

The vision *tower* deliberately lives outside this module (in the data /
sampling path): its features are frozen and precomputed, so keeping it out of
the training graph saves memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dimamba import DiMamba
from models.mm_ops import assemble_mm_embeds, slice_text_logits  # re-exported
from models.vision import MLPProjector

__all__ = ['MMDiMamba', 'assemble_mm_embeds', 'slice_text_logits']


class MMDiMamba(nn.Module):
    """DiMamba backbone + MLP projector, conditioned on image features."""

    def __init__(self, config, vocab_size: int, pad_token_id: int):
        super().__init__()
        self.backbone = DiMamba(config, vocab_size=vocab_size,
                                pad_token_id=pad_token_id)
        self.num_image_tokens = config.vlm.num_image_tokens
        self.projector = MLPProjector(
            in_dim=config.vlm.vision_dim,
            out_dim=config.model.hidden_size,
            hidden_dim=config.vlm.projector_hidden,
        )

    def forward(self, input_ids, sigma, image_features):
        """input_ids: (B, L) text ids. sigma: (B,) noise level.
        image_features: (B, N, vision_dim). Returns text logits (B, L, vocab)."""
        # Noise-level conditioning embedding (mirrors DiMamba.forward).
        c = None
        if self.backbone.temb_strategy is not None:
            c = F.silu(self.backbone.sigma_map(sigma))

        model = self.backbone.model
        text_embeds = model.get_input_embeddings()(input_ids)
        # Project image features into LM space; match the embedding dtype so the
        # concat is homogeneous (embedding lookup is fp32; autocast happens
        # inside the backbone, same as the text-only DiMamba path).
        image_embeds = self.projector(image_features).to(text_embeds.dtype)
        fused = assemble_mm_embeds(image_embeds, text_embeds)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(inputs_embeds=fused, time_embeds=c).logits

        return slice_text_logits(logits, self.num_image_tokens)
