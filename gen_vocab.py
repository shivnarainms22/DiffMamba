"""
gen_vocab.py — vocabulary-id math and embedding-resize helpers for generation VLM.

Vocabulary layout:
  [0 .. 50256]          GPT-2 BPE text tokens
   50257                [MASK]  absorbing/mask token
   50258                [BOI]   begin-image marker (conditioning, not noised)
   50259                [EOI]   end-image marker (generated, supervised)
  [50260 .. 50260+C)    image VQ tokens  (C = codebook size, e.g. 16384)

An image code `k` maps to token id `image_base + k` where image_base=50260.
"""
import torch


def code_to_id(code: int, image_base: int) -> int:
    """Convert a VQ codebook index to its vocab token id."""
    return image_base + code


def id_to_code(token_id: int, image_base: int) -> int:
    """Convert a vocab token id back to a VQ codebook index."""
    return token_id - image_base


def resize_embedding_with_warmstart(
    old_weight: torch.Tensor,
    new_vocab: int,
    std: float = 0.02,
) -> torch.Tensor:
    """Return a (new_vocab, dim) embedding weight tensor.

    The first `old_vocab` rows are copied verbatim from `old_weight`.
    New rows (indices [old_vocab, new_vocab)) are initialised from N(0, std);
    when std=0 the new rows are zeros (torch normal_ with std=0 yields mean).

    Args:
        old_weight: existing embedding weight of shape (old_vocab, dim).
        new_vocab:  target vocabulary size; must be >= old_vocab.
        std:        standard deviation for new-row initialisation (default 0.02).

    Returns:
        Float tensor of shape (new_vocab, dim) on the same device/dtype as old_weight.
    """
    old_vocab, dim = old_weight.shape
    new = torch.empty(new_vocab, dim, dtype=old_weight.dtype, device=old_weight.device)
    new.normal_(0.0, std)
    new[:old_vocab] = old_weight.detach()
    return new
