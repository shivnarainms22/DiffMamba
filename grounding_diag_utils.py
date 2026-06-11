"""Pure-math probes for the unified-model grounding diagnostic.

These functions take plain tensors and return plain floats — no model, no CUDA,
no I/O — so they are unit-testable on CPU. The GPU harness in
``main_vlm.py::_uni_grounding_diag`` feeds them the projected image prefix,
answer-region logits, and input-embedding gradients from the frozen
``uni_stage3`` checkpoint.

Three probes, mapping directly to the scope's decision tree:
  A ``image_prefix_variation`` — does the connector map distinct images to
    distinct prefixes, or collapse them to ~one vector ('connector dead')?
  B ``logit_distribution_shift`` — does swapping the image actually move the
    answer-token distribution (causal, sampler-free)?
  C ``grad_norm_ratio`` — does the answer log-prob carry gradient back to the
    image prefix at all, or only to the text ('attention-dead prefix')?
"""

import torch

_EPS = 1e-8


def image_prefix_variation(proj_embeds: torch.Tensor,
                           ref_scale: float | None = None) -> dict:
    """Probe A: how much the projected image prefix varies across images.

    Args:
        proj_embeds: ``(K, N, D)`` — projector output for ``K`` *distinct*
            images, ``N`` image tokens, model dim ``D``.
        ref_scale: optional reference std (e.g. the text embedding table's std)
            to express ``mean_feature_std`` as a relative ratio.

    Returns dict with:
        mean_pairwise_cosine: mean cosine between the flattened per-image
            prefixes over all distinct image pairs. ~1.0 == collapsed/dead.
        mean_feature_std: mean over features of the std across images.
            ~0.0 == collapsed/dead.
        std_vs_ref: ``mean_feature_std / ref_scale`` when ``ref_scale`` given.
    """
    if proj_embeds.dim() != 3:
        raise ValueError(
            f'expected (K, N, D), got shape {tuple(proj_embeds.shape)}')
    k = proj_embeds.shape[0]
    if k < 2:
        raise ValueError(f'need >=2 images to compare, got K={k}')

    flat = proj_embeds.reshape(k, -1).float()  # (K, N*D)

    normed = torch.nn.functional.normalize(flat, dim=1)
    cos = normed @ normed.t()  # (K, K)
    off_diag = cos[~torch.eye(k, dtype=torch.bool, device=cos.device)]
    mean_cosine = off_diag.mean().item()

    # std across images, per feature, then averaged.
    mean_feature_std = flat.std(dim=0, unbiased=False).mean().item()

    out = {
        'num_images': k,
        'mean_pairwise_cosine': mean_cosine,
        'mean_feature_std': mean_feature_std,
    }
    if ref_scale is not None:
        out['std_vs_ref'] = mean_feature_std / (ref_scale + _EPS)
    return out


def logit_distribution_shift(logits_a: torch.Tensor,
                             logits_b: torch.Tensor) -> dict:
    """Probe B: how far the answer-token distribution moves between two runs.

    Args:
        logits_a, logits_b: ``(P, V)`` answer-region logits from two forward
            passes that differ only in the image prefix.

    Returns dict with:
        l2: mean over positions of the L2 distance between the raw logit
            vectors.
        sym_kl: mean over positions of the symmetric KL between the softmax
            distributions, KL(p||q) + KL(q||p). Symmetric in (a, b).
    """
    if logits_a.shape != logits_b.shape:
        raise ValueError(
            f'shape mismatch: {tuple(logits_a.shape)} vs '
            f'{tuple(logits_b.shape)}')

    a = logits_a.float()
    b = logits_b.float()

    l2 = (a - b).norm(dim=-1).mean().item()

    log_p = torch.log_softmax(a, dim=-1)
    log_q = torch.log_softmax(b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(dim=-1)
    kl_qp = (q * (log_q - log_p)).sum(dim=-1)
    sym_kl = (kl_pq + kl_qp).mean().item()

    return {'l2': l2, 'sym_kl': sym_kl}


def grad_norm_ratio(image_grad: torch.Tensor,
                    text_grad: torch.Tensor) -> dict:
    """Probe C: image-prefix gradient norm relative to the text gradient.

    Args:
        image_grad: gradient of the answer log-prob w.r.t. the image-prefix
            input embeddings.
        text_grad: gradient w.r.t. the text input embeddings.

    Returns dict with the two L2 norms and ``image_to_text_ratio``
    (``image_norm / text_norm``). A ratio ~0 means the answer carries no
    gradient back to the image -> attention-dead prefix.
    """
    image_norm = image_grad.float().norm().item()
    text_norm = text_grad.float().norm().item()
    return {
        'image_grad_norm': image_norm,
        'text_grad_norm': text_norm,
        'image_to_text_ratio': image_norm / (text_norm + _EPS),
    }
