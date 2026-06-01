"""Pure splice/slice helpers for the multimodal backbone.

No mamba_ssm import — safe to use on CPU-only environments.
"""
import torch


def assemble_mm_embeds(image_embeds: torch.Tensor,
                       text_embeds: torch.Tensor) -> torch.Tensor:
    """Prepend image embeddings as a clean prefix: (B, N+L, D)."""
    return torch.cat([image_embeds, text_embeds], dim=1)


def slice_text_logits(logits: torch.Tensor, num_image_tokens: int) -> torch.Tensor:
    """Drop the first num_image_tokens (image) positions -> text logits."""
    return logits[:, num_image_tokens:]
