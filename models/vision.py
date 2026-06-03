import torch
import torch.nn as nn


class MLPProjector(nn.Module):
  """LLaVA-1.5 style 2-layer GELU projector: vision_dim -> lm_dim."""

  def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class SiglipVisionTower(nn.Module):
  """Frozen SigLIP vision encoder. Returns last_hidden_state patch features."""

  def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
    super().__init__()
    from transformers import SiglipVisionModel
    self.model = SiglipVisionModel.from_pretrained(model_name)
    self.model.eval()
    for p in self.model.parameters():
      p.requires_grad = False
    self.hidden_size = self.model.config.hidden_size
    patch = self.model.config.patch_size
    img = self.model.config.image_size
    self.num_image_tokens = (img // patch) ** 2

  @torch.no_grad()
  def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    out = self.model(pixel_values=pixel_values)
    return out.last_hidden_state  # (B, num_image_tokens, hidden_size)
