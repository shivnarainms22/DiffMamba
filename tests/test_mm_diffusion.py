"""CUDA-only tests for MMDiffusion (image-conditioned masked-diffusion VLM).
Run on a GPU node:
    python -m pytest tests/test_mm_diffusion.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import itertools

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA required for Mamba-2 kernels', allow_module_level=True)
pytest.importorskip('mamba_ssm')

import transformers
from hydra import compose, initialize_config_dir

from mm_diffusion import MMDiffusion

_CONFIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')

_TINY = [
    '+experiment=vlm_stage1_align',
    'model.hidden_size=256', 'model.n_blocks=4', 'model.cond_dim=128',
    'model.length=64',
    'vlm.num_image_tokens=5', 'vlm.vision_dim=64', 'vlm.projector_hidden=128',
    'vlm.warmstart_path=',          # no warm-start in the unit test
]


def _build(phase='align'):
    with initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        cfg = compose(config_name='config',
                      overrides=_TINY + [f'vlm.phase={phase}'])
    tok = transformers.AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.add_special_tokens({'pad_token': '[PAD]'})
    model = MMDiffusion(cfg, tokenizer=tok).cuda()
    return model


def _batch(B=2, L=16, N=5, Dv=64, vocab=50257):
    ids = torch.randint(0, vocab, (B, L), device='cuda')
    attn = torch.ones(B, L, device='cuda')
    loss_mask = torch.zeros(B, L, device='cuda')
    loss_mask[:, L // 2:] = 1.0                 # second half = answer span
    feats = torch.randn(B, N, Dv, device='cuda')
    return {'input_ids': ids, 'attention_mask': attn,
            'loss_mask': loss_mask, 'image_features': feats}


def test_loss_is_finite():
    model = _build('align')
    b = _batch()
    losses = model._loss(b['input_ids'], b['attention_mask'],
                         b['loss_mask'], b['image_features'])
    assert torch.isfinite(losses.loss), 'loss is not finite'
    assert losses.loss.item() > 0, 'untrained loss should be positive'


def test_zero_loss_mask_is_finite():
    model = _build('align')
    b = _batch()
    b['loss_mask'] = torch.zeros_like(b['loss_mask'])
    losses = model._loss(b['input_ids'], b['attention_mask'],
                         b['loss_mask'], b['image_features'])
    assert torch.isfinite(losses.loss), 'zero-supervision must not NaN'


def test_align_phase_freezes_backbone_trains_projector():
    model = _build('align')
    assert all(not p.requires_grad
               for p in model.backbone.backbone.parameters()), \
        'backbone must be frozen in align phase'
    assert all(p.requires_grad
               for p in model.backbone.projector.parameters()), \
        'projector must be trainable in align phase'


def test_sft_phase_trains_backbone():
    model = _build('sft')
    assert any(p.requires_grad
               for p in model.backbone.backbone.parameters()), \
        'backbone must be trainable in sft phase'


def test_ema_shadow_matches_trainable_params_align():
    """Regression: EMA must be built AFTER phase-freeze, so its shadow set
    matches the trainable params ema.update() filters to (else a frozen-backbone
    shadow misaligns against the projector-only live params)."""
    model = _build('align')
    trainable = [p for p in itertools.chain(model.backbone.parameters(),
                                            model.noise.parameters())
                 if p.requires_grad]
    assert len(model.ema.shadow_params) == len(trainable)
    # update() must not raise (the size-mismatch bug surfaced here).
    model.ema.update(itertools.chain(model.backbone.parameters(),
                                     model.noise.parameters()))
