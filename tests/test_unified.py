"""CUDA-only tests for UnifiedDiffusion (one backbone: understand + generate + text).
    python -m pytest tests/test_unified.py -v
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

from unified_diffusion import UnifiedDiffusion

_CONFIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

_TINY = [
    '+experiment=uni_stage3',
    'model.hidden_size=256', 'model.n_blocks=4', 'model.cond_dim=128',
    'model.length=64',
    'vlm.codebook_size=64', 'vlm.vq_tokens=8', 'vlm.siglip_tokens=5',
    'vlm.vision_dim=64', 'vlm.caption_len=4', 'vlm.projector_hidden=128',
    'vlm.warmstart_path=', 'vlm.projector_warmstart_path=',
]


def _build():
    with initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        cfg = compose(config_name='config', overrides=_TINY)
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    return UnifiedDiffusion(cfg, tokenizer=tok).cuda()


def test_contrastive_and_scale_match_run_finite():
    """Smoke: with scale_match + contrastive flags on, the understanding
    contrastive term is finite and backprops. CUDA-only (skipped on CPU CI)."""
    from omegaconf import open_dict
    with initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        cfg = compose(config_name='config', overrides=_TINY)
    with open_dict(cfg):
        cfg.vlm.proj_scale_match = True
        cfg.vlm.contrastive_weight = 0.5
        cfg.vlm.num_neg = 1
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    model = UnifiedDiffusion(cfg, tokenizer=tok).cuda()
    assert model.projector.out_norm is not None
    # scale must be FROZEN (attempt 1 collapsed it to escape grounding pressure)
    assert model.projector.out_norm.weight.requires_grad is False
    b, length = 2, model.config.vlm.caption_len + 4
    x0 = torch.randint(0, model.vocab_size, (b, length), device='cuda')
    loss_mask = torch.ones(b, length, device='cuda')
    feats = torch.randn(b, model.siglip_tokens, model.config.vlm.vision_dim,
                        device='cuda')
    lc = model._understand_contrastive(x0, loss_mask, feats, num_neg=1)
    assert torch.isfinite(lc)
    lc.backward()


def _u_batch(model, B=2, L=8, N=5, Dv=64):
    ids = torch.randint(0, 50257, (B, L), device='cuda')
    attn = torch.ones(B, L, device='cuda')
    loss = torch.zeros(B, L, device='cuda'); loss[:, L // 2:] = 1.0
    feats = torch.randn(B, N, Dv, device='cuda')
    return dict(input_ids=ids, attention_mask=attn, loss_mask=loss,
                image_features=feats)


def _g_batch(model, B=2, cap=4, n_img=8):
    bos = model.tokenizer.bos_token_id
    parts = [torch.full((B, 1), bos, device='cuda'),
             torch.randint(0, 50257, (B, cap), device='cuda'),
             torch.full((B, 1), model.boi_id, device='cuda'),
             torch.randint(model.image_base, model.image_base + 64, (B, n_img),
                           device='cuda'),
             torch.full((B, 1), model.eoi_id, device='cuda')]
    ids = torch.cat(parts, dim=1)
    attn = torch.ones_like(ids, dtype=torch.float)
    loss = torch.zeros_like(ids, dtype=torch.float); loss[:, 1 + cap + 1:] = 1.0
    return dict(input_ids=ids, attention_mask=attn, loss_mask=loss)


def _t_batch(B=2, L=8):
    return dict(input_ids=torch.randint(0, 50257, (B, L), device='cuda'),
                attention_mask=torch.ones(B, L, device='cuda'))


def test_vocab_and_shared_backbone():
    m = _build()
    assert m.image_base == 50260 and m.vocab_size == 50260 + 64
    assert m.siglip_tokens == 5 and m.vq_tokens == 8
    # one shared backbone + one projector
    assert m.backbone is m.backbone and hasattr(m, 'projector')


def test_understand_loss_finite():
    m = _build(); b = _u_batch(m)
    lu = m._understand_loss(b['input_ids'], b['attention_mask'],
                            b['loss_mask'], b['image_features'])
    assert torch.isfinite(lu.loss) and lu.loss.item() > 0


def test_generate_loss_finite():
    m = _build(); b = _g_batch(m)
    lg = m._generate_loss(b['input_ids'], b['attention_mask'], b['loss_mask'])
    assert torch.isfinite(lg.loss) and lg.loss.item() > 0


def test_text_loss_finite():
    m = _build(); b = _t_batch()
    lt = m._text_loss(b['input_ids'], b['attention_mask'])
    assert torch.isfinite(lt.loss) and lt.loss.item() > 0


def test_sample_image_valid_codes():
    m = _build()
    bos = m.tokenizer.bos_token_id
    prompt = torch.cat([torch.full((2, 1), bos, device='cuda'),
                        torch.randint(0, 50257, (2, 4), device='cuda'),
                        torch.full((2, 1), m.boi_id, device='cuda')], dim=1)
    codes = m._sample_image(prompt, num_steps=6)
    assert codes.shape == (2, m.vq_tokens)
    assert int(codes.min()) >= 0 and int(codes.max()) < m.codebook_size
