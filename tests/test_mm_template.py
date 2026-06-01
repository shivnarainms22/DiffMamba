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


# ---------------------------------------------------------------------------
# Task 6 — splice / slice helpers
# ---------------------------------------------------------------------------
import torch
from models.mm_ops import assemble_mm_embeds, slice_text_logits


def test_assemble_prepends_image_and_slice_recovers_text():
    B, N, L, D, V = 2, 5, 7, 8, 13
    text_embeds = torch.randn(B, L, D)
    image_embeds = torch.randn(B, N, D)
    fused = assemble_mm_embeds(image_embeds, text_embeds)
    assert fused.shape == (B, N + L, D)
    # image occupies the first N positions
    assert torch.equal(fused[:, :N], image_embeds)
    assert torch.equal(fused[:, N:], text_embeds)
    # slicing logits drops the N image positions
    logits = torch.randn(B, N + L, V)
    text_logits = slice_text_logits(logits, num_image_tokens=N)
    assert text_logits.shape == (B, L, V)
    assert torch.equal(text_logits, logits[:, N:])


# ---------------------------------------------------------------------------
# Task 8 — warm-start key remap
# ---------------------------------------------------------------------------
from warmstart import remap_diffmamba_backbone_state


def test_remap_keeps_backbone_drops_nonbackbone():
    ckpt_state = {
        'backbone.model.bimamba.backbone.embeddings.word_embeddings.weight': 1,
        'backbone.sigma_map.mlp.0.weight': 2,
        'noise.sigma': 3,           # not a backbone weight -> dropped
        'ema.shadow_params.0': 4,   # EMA bookkeeping -> dropped
    }
    out = remap_diffmamba_backbone_state(ckpt_state)
    # all kept keys are re-prefixed under the MM wrapper's `backbone.`
    assert 'backbone.model.bimamba.backbone.embeddings.word_embeddings.weight' in out
    assert 'backbone.sigma_map.mlp.0.weight' in out
    assert all(k.startswith('backbone.') for k in out)
    assert 'noise.sigma' not in out and not any('ema' in k for k in out)
