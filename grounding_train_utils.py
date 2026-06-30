"""In-batch image-contrastive loss for grounding the unified VLM.

The standard answer-NLL lets the language prior win: P(answer | correct image, Q)
== P(answer | wrong image, Q) under a prior. This term requires the answer to be
MORE likely under its own image than under negatives, so the only way to lower it
is to actually use the image. Bounded cross-entropy (not NLL-maximization) for
stability.
"""

import torch
import torch.nn.functional as F


def image_contrastive_loss(scores: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that the correct image gives the highest answer likelihood.

    Args:
        scores: ``(B, 1 + num_neg)`` per-image total answer log-likelihoods for
            each example; column 0 is the correct image, the rest are negatives.

    Returns:
        Scalar loss. ~0 when the correct image dominates; ``log(1+num_neg)`` when
        all images are equally likely (the language-prior failure mode).
    """
    if scores.dim() != 2:
        raise ValueError(f'expected (B, 1+num_neg), got {tuple(scores.shape)}')
    target = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores, target)
