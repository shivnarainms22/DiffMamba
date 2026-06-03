"""Throwaway probe: which diffusers VQModel gives a 32x32 latent grid (=1024
image tokens) and what is its codebook size?

Tries candidate checkpoints, encodes a 256x256 image, prints the latent grid
and codebook size (or the failure). Picks the f8 / 32x32 tokenizer for Stage-2
generation. Run:  python scripts/probe_vq.py
"""
import torch

# (repo, subfolder) candidates — diffusers VQModel checkpoints.
CANDIDATES = [
    ("CompVis/ldm-celebahq-256", "vqvae"),
    ("CompVis/ldm-super-resolution-4x-openimages", "vqvae"),
    ("CompVis/ldm-text2im-large-256", "vqvae"),
    ("microsoft/vq-diffusion-ithq", "vqvae"),
    ("stabilityai/sd-vq-f8", None),
]


def probe(repo, subfolder):
    from diffusers import VQModel
    kwargs = {"subfolder": subfolder} if subfolder else {}
    m = VQModel.from_pretrained(repo, **kwargs).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        h = m.encode(x).latents
    n_embed = (getattr(m.config, "num_vq_embeddings", None)
               or getattr(m.config, "n_embed", "?"))
    _, c, hh, ww = h.shape
    print(f"OK   {repo} sub={subfolder} | latent={1, c, hh, ww} "
          f"| grid={hh}x{ww}={hh*ww} tokens | codebook={n_embed}")


def main():
    for repo, sub in CANDIDATES:
        try:
            probe(repo, sub)
        except Exception as e:  # noqa: BLE001 - probe wants every failure reason
            print(f"FAIL {repo} sub={sub} | {repr(e)[:110]}")


if __name__ == "__main__":
    main()
