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


def test_scale_match_output_rms_is_stable_across_input_scale():
  """RMSNorm on the projector output -> output RMS ~= norm weight, regardless of
  how hot the input is. This is what un-throttles the projector gradient."""
  proj = MLPProjector(in_dim=8, out_dim=16, hidden_dim=32, scale_match=True)
  proj.out_norm.weight.data.fill_(1.0)
  for scale in (0.1, 1.0, 24.0):
    y = proj(torch.randn(64, 8) * scale)
    assert 0.8 < y.pow(2).mean().sqrt().item() < 1.25  # ~1.0 independent of input


def test_scale_match_rms_tracks_norm_weight():
  proj = MLPProjector(8, 16, 32, scale_match=True)
  proj.out_norm.weight.data.fill_(0.17)
  y = proj(torch.randn(64, 8))
  assert 0.10 < y.pow(2).mean().sqrt().item() < 0.25


def test_without_scale_match_output_rms_grows_with_input():
  proj = MLPProjector(8, 16, 32, scale_match=False)
  assert proj.out_norm is None
  small = proj(torch.randn(64, 8) * 0.1).pow(2).mean().sqrt().item()
  big = proj(torch.randn(64, 8) * 24.0).pow(2).mean().sqrt().item()
  assert big > 3 * small


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
