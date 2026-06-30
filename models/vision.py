import torch
import torch.nn as nn


class MLPProjector(nn.Module):
  """LLaVA-1.5 style 2-layer GELU projector: vision_dim -> lm_dim.

  When ``scale_match`` is set, an RMSNorm on the output fixes its scale (its
  learnable weight is initialized by the caller to the text-embed std). The
  backbone's first block is RMSNorm pre-norm (scale-invariant), so a too-hot
  projector output does not change the forward but throttles the projector's
  training gradient ~1/scale; this normalizes the output so that gradient flows.
  """

  def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048,
               scale_match: bool = False):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_dim, hidden_dim),
      nn.GELU(),
      nn.Linear(hidden_dim, out_dim),
    )
    self.out_norm = nn.RMSNorm(out_dim) if scale_match else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = self.net(x)
    if self.out_norm is not None:
      y = self.out_norm(y)
    return y


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
