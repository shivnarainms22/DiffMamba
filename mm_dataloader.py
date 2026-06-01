"""
mm_dataloader.py — text template utilities for masked-diffusion VLM training.

Currently contains only build_prompt_labels (Task 4).
MMDataset and get_mm_dataloaders will be added in Task 5.
"""
from typing import Dict, List


def build_prompt_labels(tokenizer, prompt: str, answer: str,
                        text_len: int) -> Dict[str, List[int]]:
    """Build a single text example: [BOS] prompt answer [EOS], padded to text_len.

    Args:
        tokenizer: any tokenizer with bos_token_id, eos_token_id, pad_token_id
                   and an encode(str, add_special_tokens=False) -> List[int] method.
        prompt:    the conditioning text (question / caption instruction).
        answer:    the supervised response — only these tokens are denoised.
        text_len:  fixed sequence length; output lists are always this length.

    Returns:
        dict with keys:
          input_ids      — [BOS] + prompt + answer + [EOS], padded with pad_token_id.
          attention_mask — 1 on real tokens (non-pad), 0 on pad.
          loss_mask      — 1 on answer tokens and the final EOS, 0 elsewhere.

    Truncation: if the raw sequence exceeds text_len, it is clipped to
    text_len-1 tokens and a final EOS (loss_mask=1) is appended, so the
    output is always exactly text_len tokens with a valid EOS at the end.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)

    ids  = [bos] + p_ids + a_ids + [eos]
    loss = [0]   + [0] * len(p_ids) + [1] * len(a_ids) + [1]

    if len(ids) > text_len:           # truncate, always keep a final EOS
        ids  = ids[:text_len - 1]  + [eos]
        loss = loss[:text_len - 1] + [1]

    attn  = [1] * len(ids)
    pad_n = text_len - len(ids)
    ids  += [pad] * pad_n
    attn += [0]   * pad_n
    loss += [0]   * pad_n

    return {"input_ids": ids, "attention_mask": attn, "loss_mask": loss}
