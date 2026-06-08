"""Unified dataloader: combine understanding + generation (+ optional pure-text)
streams for joint training, reusing the Stage-1 and Stage-2 pipelines.

- understanding: CC3M SigLIP features + caption  (mm_dataloader, reused)
- generation:    CC3M VQ tokens + caption        (gen_dataloader, reused; reuses
                 the Stage-2 VQ cache when dataset/split/max_examples/vq_repo match)
- pure text:     existing text MDLM data          (dataloader, only if text_weight>0)

Per-stream config "views" set the right num_image_tokens (SigLIP 196 vs VQ 1024),
which the two underlying loaders read from config.vlm.num_image_tokens.
"""
import copy

from omegaconf import open_dict


def _understanding_view(config):
    cfg = copy.deepcopy(config)
    with open_dict(cfg):
        cfg.vlm.num_image_tokens = config.vlm.siglip_tokens
        if config.vlm.get('understand_task', 'caption') == 'vqa':
            v = config.vlm
            cfg.vlm.dataset = v.get('vqa_dataset', 'lmms-lab/VQAv2')
            cfg.vlm.split = v.get('vqa_split', 'validation')
            cfg.vlm.image_column = v.get('vqa_image_column', 'image')
            cfg.vlm.caption_column = v.get('vqa_caption_column', 'multiple_choice_answer')
            cfg.vlm.question_column = v.get('vqa_question_column', 'question')
            cfg.vlm.text_len = v.get('vqa_text_len', v.caption_len)
            cfg.vlm.max_examples = v.get('vqa_max_examples', 40000)
            cfg.vlm.streaming = True
            cfg.vlm.label_as_caption = False
        else:
            cfg.vlm.text_len = config.vlm.caption_len
            cfg.vlm.label_as_caption = False
    return cfg


def _generation_view(config):
    cfg = copy.deepcopy(config)
    with open_dict(cfg):
        cfg.vlm.num_image_tokens = config.vlm.vq_tokens
    return cfg


def get_unified_dataloaders(config, tokenizer):
    import dataloader
    import gen_dataloader
    import mm_dataloader
    from lightning.pytorch.utilities.combined_loader import CombinedLoader

    u_train, u_valid = mm_dataloader.get_mm_dataloaders(
        _understanding_view(config), tokenizer)
    g_train, g_valid = gen_dataloader.get_gen_dataloaders(
        _generation_view(config), tokenizer)

    train_loaders = {'u': u_train, 'g': g_train}
    valid_loaders = {'u': u_valid, 'g': g_valid}

    if config.vlm.get('text_weight', 0.0) > 0:
        t_train, _ = dataloader.get_dataloaders(
            config, tokenizer, skip_valid=True)
        train_loaders['t'] = t_train

    train = CombinedLoader(train_loaders, mode='max_size_cycle')
    valid = CombinedLoader(valid_loaders, mode='max_size_cycle')
    return train, valid
