"""Pure-math probes for the grounding diagnostic (CPU, no CUDA, no model).

    python -m pytest tests/test_grounding_diag.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pytest
import torch

from grounding_diag_utils import (
    grad_norm_ratio,
    image_prefix_variation,
    logit_distribution_shift,
)


# ---- Probe A: image_prefix_variation -------------------------------------
def test_identical_images_give_max_cosine_zero_std():
    """K copies of one image -> the projected prefix is constant: cosine 1, std 0.

    This is the 'connector dead' signature we are hunting for.
    """
    one = torch.randn(3, 8)  # (N, D)
    embeds = one[None].repeat(5, 1, 1)  # (K=5, N, D), all identical
    out = image_prefix_variation(embeds)
    assert out['mean_pairwise_cosine'] == pytest.approx(1.0, abs=1e-5)
    assert out['mean_feature_std'] == pytest.approx(0.0, abs=1e-6)


def test_distinct_images_lower_cosine_positive_std():
    """Independent random images -> cosine well below 1, std clearly positive."""
    embeds = torch.randn(6, 3, 8)  # genuinely different images
    out = image_prefix_variation(embeds)
    assert out['mean_pairwise_cosine'] < 0.6
    assert out['mean_feature_std'] > 0.1


def test_variation_reports_std_vs_ref_when_given():
    """A reference scale (e.g. text-embedding std) yields a relative ratio."""
    embeds = torch.randn(4, 2, 5)
    out = image_prefix_variation(embeds, ref_scale=2.0)
    assert out['std_vs_ref'] == pytest.approx(out['mean_feature_std'] / 2.0)


def test_variation_requires_at_least_two_images():
    with pytest.raises(ValueError):
        image_prefix_variation(torch.randn(1, 3, 8))


def test_variation_rejects_wrong_rank():
    with pytest.raises(ValueError):
        image_prefix_variation(torch.randn(3, 8))  # missing image axis


# ---- Probe B: logit_distribution_shift -----------------------------------
def test_identical_logits_zero_shift():
    logits = torch.randn(4, 50)  # (P positions, V vocab)
    out = logit_distribution_shift(logits, logits.clone())
    assert out['l2'] == pytest.approx(0.0, abs=1e-6)
    assert out['sym_kl'] == pytest.approx(0.0, abs=1e-6)


def test_different_logits_positive_shift():
    a = torch.randn(4, 50)
    b = a + torch.randn(4, 50)
    out = logit_distribution_shift(a, b)
    assert out['l2'] > 0.0
    assert out['sym_kl'] > 0.0


def test_sym_kl_is_symmetric():
    a = torch.randn(3, 20)
    b = torch.randn(3, 20)
    assert logit_distribution_shift(a, b)['sym_kl'] == pytest.approx(
        logit_distribution_shift(b, a)['sym_kl'], abs=1e-6)


def test_shift_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        logit_distribution_shift(torch.randn(4, 50), torch.randn(4, 49))


# ---- Probe C: grad_norm_ratio --------------------------------------------
def test_grad_ratio_matches_known_norms():
    img = torch.tensor([3.0, 4.0])      # L2 norm 5
    txt = torch.tensor([0.0, 10.0])     # L2 norm 10
    out = grad_norm_ratio(img, txt)
    assert out['image_grad_norm'] == pytest.approx(5.0)
    assert out['text_grad_norm'] == pytest.approx(10.0)
    assert out['image_to_text_ratio'] == pytest.approx(0.5)


def test_grad_ratio_zero_image_grad_signals_dead_prefix():
    """An image gradient of ~0 vs a real text gradient => attention-dead prefix."""
    out = grad_norm_ratio(torch.zeros(8), torch.randn(8) + 5.0)
    assert out['image_grad_norm'] == pytest.approx(0.0, abs=1e-6)
    assert out['image_to_text_ratio'] == pytest.approx(0.0, abs=1e-6)


def test_grad_ratio_handles_zero_text_grad_without_nan():
    out = grad_norm_ratio(torch.randn(4), torch.zeros(4))
    assert math.isfinite(out['image_to_text_ratio'])
