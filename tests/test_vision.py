import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.vision import MLPProjector


def test_projector_maps_vision_dim_to_lm_dim():
  proj = MLPProjector(in_dim=768, out_dim=768, hidden_dim=2048)
  x = torch.randn(2, 196, 768)          # (B, num_image_tokens, vision_dim)
  y = proj(x)
  assert y.shape == (2, 196, 768)
  assert y.dtype == x.dtype


def test_projector_handles_dim_change():
  proj = MLPProjector(in_dim=1152, out_dim=768, hidden_dim=2048)  # so400m -> 768
  y = proj(torch.randn(1, 729, 1152))
  assert y.shape == (1, 729, 768)


import pytest


def test_vision_tower_returns_patch_features():
  transformers = pytest.importorskip("transformers")
  from models.vision import SiglipVisionTower
  tower = SiglipVisionTower("google/siglip-base-patch16-224").eval()
  # pixel_values: (B, 3, 224, 224)
  pixel_values = torch.randn(1, 3, 224, 224)
  with torch.no_grad():
    feats = tower(pixel_values)
  assert feats.shape[0] == 1
  assert feats.shape[1] == tower.num_image_tokens   # 196 for patch16-224
  assert feats.shape[2] == tower.hidden_size        # 768
  # Frozen: no parameter requires grad.
  assert all(not p.requires_grad for p in tower.parameters())
