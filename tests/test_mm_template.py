"""
Unit tests for build_prompt_labels (pure, CPU-testable).
Run: python -m pytest tests/test_mm_template.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mm_dataloader import build_prompt_labels


class _ToyTok:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def encode(self, s, add_special_tokens=False):
        # deterministic: one id per word, offset to avoid specials
        return [10 + (ord(w[0]) % 50) for w in s.split()]


def test_loss_mask_covers_answer_not_prompt_or_pad():
    tok = _ToyTok()
    out = build_prompt_labels(tok, prompt="describe the image",
                              answer="a cat", text_len=16)
    ids, attn, loss = out["input_ids"], out["attention_mask"], out["loss_mask"]
    assert len(ids) == len(attn) == len(loss) == 16
    # BOS at 0, then prompt(3), then answer(2), then EOS, rest pad.
    assert ids[0] == tok.bos_token_id
    # prompt positions (1..3) are NOT in loss; answer + EOS ARE.
    assert loss[1] == 0 and loss[2] == 0 and loss[3] == 0
    answer_start = 1 + 3
    assert loss[answer_start] == 1 and loss[answer_start + 1] == 1   # "a cat"
    assert loss[answer_start + 2] == 1                                # EOS
    # padding: attn and loss both 0 at the tail.
    assert attn[-1] == 0 and loss[-1] == 0
    assert ids[-1] == tok.pad_token_id


def test_truncates_when_too_long():
    tok = _ToyTok()
    out = build_prompt_labels(tok, prompt="a " * 30, answer="b " * 30, text_len=16)
    assert len(out["input_ids"]) == 16
    assert out["input_ids"][-1] in (tok.eos_token_id, tok.pad_token_id)
