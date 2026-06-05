"""Classifier-free guidance helpers shared by image-token samplers."""

import torch


def make_unconditional_prompt(x: torch.Tensor, prompt_len: int,
                              pad_token_id: int) -> torch.Tensor:
    """Drop caption tokens while preserving BOS, BOI, image span, and EOI.

    The generation prompt layout is fixed as `[BOS] caption... [BOI]`.
    Positions `1:prompt_len-1` are the caption conditioning span.
    """
    out = x.clone()
    if prompt_len > 2:
        out[:, 1:prompt_len - 1] = pad_token_id
    return out


def cfg_combine_log_probs(cond: torch.Tensor, uncond: torch.Tensor,
                          scale: float) -> torch.Tensor:
    """Classifier-free guidance interpolation in log-prob/logit space."""
    return uncond + scale * (cond - uncond)
