"""
Unit tests for generation sequence template (Task 3) and vocab helpers (Task 4).
All tests are pure CPU — no CUDA, no mamba_ssm, no diffusers required.
Run: python -m pytest tests/test_gen_template.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gen_dataloader import build_gen_sequence

BOS, MASK, BOI, EOI = 1, 50257, 50258, 50259


def test_gen_sequence_layout_and_loss_mask():
    cap = [10, 11, 12]
    img = list(range(50260, 50260 + 8))          # 8 image-token ids
    out = build_gen_sequence(cap, img, bos=BOS, boi=BOI, eoi=EOI, pad=0,
                             caption_len=5, num_image_tokens=8)
    ids, attn, loss = out['input_ids'], out['attention_mask'], out['loss_mask']
    n = 1 + 5 + 1 + 8 + 1                          # BOS + cap(pad to 5) + BOI + img + EOI
    assert len(ids) == len(attn) == len(loss) == n
    assert ids[0] == BOS
    assert ids[1 + 5] == BOI                        # BOI after padded caption
    assert ids[1 + 5 + 1: 1 + 5 + 1 + 8] == img   # image span
    assert ids[-1] == EOI
    # caption (BOS+caption+BOI) is clean conditioning -> loss 0
    assert all(l == 0 for l in loss[:1 + 5 + 1])
    # image tokens + EOI are generated -> loss 1
    assert all(l == 1 for l in loss[1 + 5 + 1:])


# ---------------------------------------------------------------------------
# Task 4 — vocab-id math + embedding-resize tensor op
# ---------------------------------------------------------------------------
import torch
from gen_vocab import code_to_id, id_to_code, resize_embedding_with_warmstart


def test_code_id_roundtrip():
    base = 50260
    assert code_to_id(0, base) == 50260
    assert id_to_code(50265, base) == 5
    assert id_to_code(code_to_id(123, base), base) == 123


def test_resize_copies_text_rows_and_inits_new():
    old = torch.arange(12, dtype=torch.float32).reshape(4, 3)   # old_vocab=4, dim=3
    new = resize_embedding_with_warmstart(old, new_vocab=7, std=0.0)
    assert new.shape == (7, 3)
    assert torch.equal(new[:4], old)                             # text rows copied
    assert torch.equal(new[4:], torch.zeros(3, 3))              # std=0 -> zeros for new rows
