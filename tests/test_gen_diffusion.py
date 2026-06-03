"""CUDA-only tests for GenDiffusion (text->image VQ-token generation).
    python -m pytest tests/test_gen_diffusion.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA required', allow_module_level=True)
pytest.importorskip('mamba_ssm')

import transformers
from hydra import compose, initialize_config_dir

from gen_diffusion import GenDiffusion

_CONFIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

_TINY = [
    '+experiment=gen_stage2',
    'model.hidden_size=256', 'model.n_blocks=4', 'model.cond_dim=128',
    'model.length=64',
    'vlm.codebook_size=64', 'vlm.num_image_tokens=8', 'vlm.caption_len=4',
    'vlm.warmstart_path=',
]


def _build():
    with initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        cfg = compose(config_name='config', overrides=_TINY)
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    return GenDiffusion(cfg, tokenizer=tok).cuda()


def _batch(model, B=2, cap_len=4, n_img=8):
    bos = model.tokenizer.bos_token_id
    cap = torch.randint(0, 50257, (B, cap_len), device='cuda')
    img = torch.randint(model.image_base, model.image_base + 64, (B, n_img),
                        device='cuda')
    boi = torch.full((B, 1), model.boi_id, device='cuda')
    eoi = torch.full((B, 1), model.eoi_id, device='cuda')
    bos_col = torch.full((B, 1), bos, device='cuda')
    ids = torch.cat([bos_col, cap, boi, img, eoi], dim=1)         # (B, 15)
    attn = torch.ones_like(ids, dtype=torch.float)
    loss = torch.zeros_like(ids, dtype=torch.float)
    loss[:, 1 + cap_len + 1:] = 1.0                                # image + EOI
    return {'input_ids': ids, 'attention_mask': attn, 'loss_mask': loss}


def test_vocab_layout():
    model = _build()
    assert model.boi_id == 50258
    assert model.eoi_id == 50259
    assert model.image_base == 50260
    assert model.vocab_size == 50260 + 64


def test_gen_loss_is_finite():
    model = _build()
    b = _batch(model)
    losses = model._loss(b['input_ids'], b['attention_mask'], b['loss_mask'])
    assert torch.isfinite(losses.loss), 'loss not finite'
    assert losses.loss.item() > 0


def test_sample_image_returns_valid_codes():
    model = _build()
    B, cap_len = 2, 4
    bos = model.tokenizer.bos_token_id
    cap = torch.randint(0, 50257, (B, cap_len), device='cuda')
    prompt = torch.cat([
        torch.full((B, 1), bos, device='cuda'),
        cap,
        torch.full((B, 1), model.boi_id, device='cuda')], dim=1)   # [BOS] cap [BOI]
    codes = model._sample_image(prompt, num_steps=8)
    assert codes.shape == (B, 8)                                   # num_image_tokens
    assert int(codes.min()) >= 0 and int(codes.max()) < model.codebook_size
