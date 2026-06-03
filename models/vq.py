"""Frozen VQ image tokenizer for Stage-2 generation.

Wraps a diffusers VQModel: images <-> flattened discrete code-index grids.
Default = microsoft/vq-diffusion-ithq (vqvae): 256px -> 32x32 = 1024 tokens,
codebook 4096. Used only in the data pipeline (encode) and at inference
(decode) — never in the training graph.
"""
import torch
import torch.nn as nn
from diffusers import VQModel


class VQTokenizer(nn.Module):
    def __init__(self, repo: str = 'microsoft/vq-diffusion-ithq',
                 subfolder: str | None = 'vqvae'):
        super().__init__()
        kwargs = {'subfolder': subfolder} if subfolder else {}
        self.model = VQModel.from_pretrained(repo, **kwargs)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.codebook_size = (getattr(self.model.config, 'num_vq_embeddings', None)
                              or self.model.config.n_embed)
        self.num_image_tokens = None    # set on first encode (= grid H*W)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B,3,H,W) in [-1,1]. Returns (B, num_image_tokens) long codes."""
        h = self.model.encode(images).latents
        _, _, info = self.model.quantize(h)
        idx = info[2]                                   # (B*Hg*Wg,) min-encoding indices
        b, _, hg, wg = h.shape
        self.num_image_tokens = hg * wg
        return idx.view(b, hg * wg).long()

    @torch.no_grad()
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, num_image_tokens) long codes. Returns (B,3,H,W) pixels."""
        b, n = tokens.shape
        hw = int(round(n ** 0.5))
        quant = self.model.quantize.get_codebook_entry(
            tokens.reshape(-1), shape=(b, hw, hw, -1))
        return self.model.decode(quant).sample
