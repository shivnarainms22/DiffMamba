"""VQ tokenizer encode<->decode round-trip. Needs diffusers + the VQModel
checkpoint (downloads once). CPU-runnable (slow); skips if diffusers absent.
    python -m pytest tests/test_vq.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

REPO = 'microsoft/vq-diffusion-ithq'
SUB = 'vqvae'


def test_encode_decode_roundtrip():
    pytest.importorskip('diffusers')
    from models.vq import VQTokenizer
    vq = VQTokenizer(REPO, subfolder=SUB).eval()

    img = torch.rand(2, 3, 256, 256) * 2 - 1            # [-1, 1]
    tokens = vq.encode(img)
    assert tokens.shape == (2, 1024), f'expected (2,1024), got {tuple(tokens.shape)}'
    assert tokens.dtype == torch.long
    assert int(tokens.max()) < vq.codebook_size and int(tokens.min()) >= 0
    assert vq.codebook_size == 4096

    recon = vq.decode(tokens)
    assert recon.shape[0] == 2 and recon.shape[1] == 3      # (B,3,H,W) pixels
    assert torch.isfinite(recon).all()
    # Frozen.
    assert all(not p.requires_grad for p in vq.parameters())
