import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import pytest
import torch

from grounding_train_utils import image_contrastive_loss


def test_zero_loss_when_correct_image_dominates():
    scores = torch.tensor([[10.0, -10.0], [10.0, -10.0]])  # col 0 = correct image
    assert image_contrastive_loss(scores).item() < 0.01


def test_max_loss_when_all_images_equal_the_prior_case():
    # 1 correct + 2 negatives, identical answer-likelihood under every image:
    # this is exactly the language prior -> contrastive cannot be satisfied.
    scores = torch.zeros(4, 3)
    assert abs(image_contrastive_loss(scores).item() - math.log(3)) < 1e-4


def test_gradient_pushes_correct_up_negatives_down():
    scores = torch.zeros(2, 2, requires_grad=True)
    image_contrastive_loss(scores).backward()
    assert scores.grad[:, 0].mean().item() < 0   # raise correct-image likelihood
    assert scores.grad[:, 1].mean().item() > 0   # lower wrong-image likelihood


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        image_contrastive_loss(torch.zeros(4))
