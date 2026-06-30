"""CUDA-only smoke tests for the hybrid Mamba+attention denoiser.

Run on a GPU node:
    python -m pytest tests/test_hybrid_dimamba.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip('CUDA required for Mamba-2/flash-attn kernels', allow_module_level=True)
pytest.importorskip('mamba_ssm')
pytest.importorskip('flash_attn')

from hydra import compose, initialize_config_dir

from models.hybrid_dimamba import HybridDiMamba

_CONFIGS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')


def _cfg():
    with initialize_config_dir(version_base=None, config_dir=_CONFIGS):
        return compose(config_name='config', overrides=[
            '+experiment=hybrid_130m',
            'model.hidden_size=256',
            'model.n_blocks=4',
            'model.n_heads=4',
            'model.cond_dim=128',
            'model.length=64',
            'model.hybrid_attention_every=2',
            'model.hybrid_attention_offset=1',
        ])


def test_hybrid_layer_schedule_and_forward_shape():
    cfg = _cfg()
    model = HybridDiMamba(cfg, vocab_size=128, pad_token_id=0).cuda().eval()
    assert model.layer_types == ['mamba', 'attention', 'mamba', 'attention']
    ids = torch.randint(0, 128, (2, 32), device='cuda')
    sigma = torch.ones(2, device='cuda')
    with torch.no_grad():
        logits = model(ids, sigma)
    assert logits.shape == (2, 32, 128)
    assert torch.isfinite(logits).all()
