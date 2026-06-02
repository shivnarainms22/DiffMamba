"""CUDA-only tests for the MMDiMamba multimodal backbone wrapper.
Run on a GPU node (mamba-ssm Triton kernels required):
    python -m pytest tests/test_mm_dimamba.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA required for Mamba-2 kernels', allow_module_level=True)
pytest.importorskip('mamba_ssm')

import omegaconf
from models.mm_dimamba import MMDiMamba


def _cfg(hidden=256, blocks=4, cond=128, vision_dim=64, n_img=5):
    return omegaconf.OmegaConf.create({
        'model': {'hidden_size': hidden, 'n_blocks': blocks, 'cond_dim': cond,
                  'tie_word_embeddings': False, 'temb_strategy': 'adaln',
                  'length': 32, 'dropout': 0.1},
        'vlm': {'vision_dim': vision_dim, 'num_image_tokens': n_img,
                'projector_hidden': 128},
    })


def test_mm_forward_shape_and_conditioning():
    cfg = _cfg()
    model = MMDiMamba(cfg, vocab_size=50258, pad_token_id=0).cuda()
    B, L, N, Dv = 2, 16, 5, 64
    ids = torch.randint(0, 50257, (B, L), device='cuda')
    sigma = torch.rand(B, device='cuda')
    feats = torch.randn(B, N, Dv, device='cuda')

    out1 = model(ids, sigma, feats)
    assert out1.shape == (B, L, 50258), \
        f'expected text logits (B, L, vocab), got {tuple(out1.shape)}'
    assert torch.isfinite(out1).all(), 'non-finite logits'

    # Changing the image must change the text logits (image actually conditions).
    out2 = model(ids, sigma, torch.randn(B, N, Dv, device='cuda'))
    assert not torch.allclose(out1, out2, atol=1e-4), \
        'text logits unchanged when image changed — image is not conditioning'


def test_projector_is_trainable_backbone_present():
    cfg = _cfg()
    model = MMDiMamba(cfg, vocab_size=50258, pad_token_id=0).cuda()
    assert any(p.requires_grad for p in model.projector.parameters())
    # The backbone embedding must be reachable for warm-start / training.
    assert model.backbone.model.get_input_embeddings() is not None
