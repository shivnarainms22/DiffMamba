"""Pure-config tests for _understanding_view / _generation_view.
No GPU, no torch, no external data — runs locally.

python -m pytest tests/test_unified_dataloader_views.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf, open_dict

from unified_dataloader import _understanding_view, _generation_view


def _make_config(understand_task=None):
    """Build a minimal OmegaConf config mirroring configs/vlm/unified.yaml."""
    vlm = {
        # understanding pathway
        'siglip_tokens': 196,
        'vq_tokens': 1024,
        'caption_len': 64,
        # CC3M (main dataset, both streams)
        'dataset': 'pixparse/cc3m-wds',
        'split': 'train',
        'image_column': 'jpg',
        'caption_column': 'txt',
        'streaming': True,
        'max_examples': 80000,
        # vqa_* keys already declared in unified.yaml
        'vqa_dataset': 'lmms-lab/VQAv2',
        'vqa_split': 'validation',
        'vqa_image_column': 'image',
        'vqa_caption_column': 'multiple_choice_answer',
        'vqa_question_column': 'question',
        'vqa_max_examples': 40000,
        'vqa_text_len': 64,
    }
    if understand_task is not None:
        vlm['understand_task'] = understand_task

    cfg = OmegaConf.create({'vlm': vlm})
    return cfg


def test_vqa_understanding_view_routes_to_vqa_dataset():
    """understand_task='vqa' → understanding view uses VQAv2 dataset & columns."""
    cfg = _make_config(understand_task='vqa')
    view = _understanding_view(cfg)

    assert view.vlm.dataset == 'lmms-lab/VQAv2'
    assert view.vlm.split == 'validation'
    assert view.vlm.image_column == 'image'
    assert view.vlm.question_column == 'question'
    assert view.vlm.caption_column == 'multiple_choice_answer'
    assert view.vlm.text_len == 64            # vqa_text_len
    assert view.vlm.max_examples == 40000     # vqa_max_examples
    assert view.vlm.streaming is True
    assert view.vlm.label_as_caption is False
    assert view.vlm.num_image_tokens == 196   # siglip_tokens


def test_generation_view_unaffected_by_vqa_task():
    """_generation_view must always point at CC3M regardless of understand_task."""
    cfg = _make_config(understand_task='vqa')
    view = _generation_view(cfg)

    assert view.vlm.dataset == 'pixparse/cc3m-wds'
    assert view.vlm.image_column == 'jpg'
    assert view.vlm.num_image_tokens == 1024  # vq_tokens


def test_caption_back_compat_no_understand_task():
    """Default (understand_task absent): understanding view keeps CC3M caption path."""
    cfg = _make_config()  # no understand_task key
    view = _understanding_view(cfg)

    assert view.vlm.dataset == 'pixparse/cc3m-wds'
    # Caption path must not inject a VQA-style question_column (else mm_dataloader
    # would switch to the VQA prompt=question path).
    assert 'question_column' not in view.vlm


def test_caption_back_compat_explicit_caption():
    """understand_task='caption': same as absent — CC3M caption path preserved."""
    cfg = _make_config(understand_task='caption')
    view = _understanding_view(cfg)

    assert view.vlm.dataset == 'pixparse/cc3m-wds'
    assert 'question_column' not in view.vlm
