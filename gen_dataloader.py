"""
gen_dataloader.py — data pipeline for the masked-diffusion text→image generation VLM.

- build_gen_sequence: pure sequence template builder (CPU-testable, no torch).
- GenDataset / get_gen_dataloaders: VQ-token memmap dataset (Task 6).
"""
from typing import Dict, List


def build_gen_sequence(
    caption_ids,
    image_tokens,
    bos: int,
    boi: int,
    eoi: int,
    pad: int,
    caption_len: int,
    num_image_tokens: int,
) -> Dict[str, List[int]]:
    """Build a single generation sequence: [BOS] caption(padded) [BOI] image_tokens [EOI].

    Args:
        caption_ids:      token ids for the caption text (will be truncated/padded to
                          caption_len).
        image_tokens:     exactly num_image_tokens already-offset vocab ids for the
                          image span (e.g. image_base + code).
        bos:              BOS token id.
        boi:              begin-image marker token id (clean, not noised).
        eoi:              end-image marker token id (generated, supervised).
        pad:              padding token id used to fill a short caption.
        caption_len:      fixed caption length (truncate or right-pad with `pad`).
        num_image_tokens: expected number of image tokens; asserted against len(image_tokens).

    Returns:
        dict with keys:
          input_ids      — [BOS] + caption(padded) + [BOI] + image_tokens + [EOI].
          attention_mask — 1 everywhere except caption-padding positions (which are 0).
          loss_mask      — 0 on BOS+caption+BOI (clean conditioning),
                           1 on image_tokens+EOI (generated targets).
    """
    assert len(image_tokens) == num_image_tokens, (
        f"Expected {num_image_tokens} image tokens, got {len(image_tokens)}"
    )

    # Truncate then right-pad caption to exactly caption_len.
    cap = list(caption_ids[:caption_len])
    cap += [pad] * (caption_len - len(cap))

    ids = [bos] + cap + [boi] + list(image_tokens) + [eoi]

    prompt_end = 1 + caption_len + 1          # BOS + caption + BOI
    loss = [0] * prompt_end + [1] * (num_image_tokens + 1)  # image tokens + EOI

    # Attention: 1 everywhere except caption pad positions.
    attn = (
        [1]
        + [1 if t != pad else 0 for t in cap]
        + [1] * (1 + num_image_tokens + 1)    # BOI + image + EOI always attended
    )

    return {"input_ids": ids, "attention_mask": attn, "loss_mask": loss}
