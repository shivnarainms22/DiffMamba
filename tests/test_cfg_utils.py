import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cfg_utils import cfg_combine_log_probs, make_unconditional_prompt


def test_make_unconditional_prompt_drops_caption_but_keeps_markers():
    x = torch.tensor([[1, 10, 11, 12, 50258, 50260, 50259]])
    out = make_unconditional_prompt(x, prompt_len=5, pad_token_id=0)
    assert out.tolist() == [[1, 0, 0, 0, 50258, 50260, 50259]]


def test_make_unconditional_prompt_noops_without_caption_span():
    x = torch.tensor([[1, 50258, 50260]])
    out = make_unconditional_prompt(x, prompt_len=2, pad_token_id=0)
    assert torch.equal(out, x)
    assert out.data_ptr() != x.data_ptr()


def test_cfg_combine_log_probs_scale_one_is_conditional():
    cond = torch.tensor([[[0.1, 0.9]]])
    uncond = torch.tensor([[[0.5, 0.5]]])
    out = cfg_combine_log_probs(cond, uncond, scale=1.0)
    assert torch.equal(out, cond)


def test_cfg_combine_log_probs_extrapolates_from_unconditional():
    cond = torch.tensor([[[2.0, 4.0]]])
    uncond = torch.tensor([[[1.0, 1.5]]])
    out = cfg_combine_log_probs(cond, uncond, scale=2.0)
    assert torch.equal(out, torch.tensor([[[3.0, 6.5]]]))
