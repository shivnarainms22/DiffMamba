"""
mm_dataloader.py — data pipeline for the masked-diffusion understanding VLM.

- build_prompt_labels: pure text-template builder (CPU-testable).
- MMDataset / get_mm_dataloaders: a *generic* image-caption loader. It reads any
  HF dataset that exposes an inline image column and a caption column (configured
  via vlm.image_column / vlm.caption_column), precomputes frozen SigLIP features
  once (kept out of the training graph and out of checkpoints), and yields
  {input_ids, attention_mask, loss_mask, image_features} batches.
"""
import itertools
from typing import Dict, List

import torch


def build_prompt_labels(tokenizer, prompt: str, answer: str,
                        text_len: int) -> Dict[str, List[int]]:
    """Build a single text example: [BOS] prompt answer [EOS], padded to text_len.

    Args:
        tokenizer: any tokenizer with bos_token_id, eos_token_id, pad_token_id
                   and an encode(str, add_special_tokens=False) -> List[int] method.
        prompt:    the conditioning text (question / caption instruction).
        answer:    the supervised response — only these tokens are denoised.
        text_len:  fixed sequence length; output lists are always this length.

    Returns:
        dict with keys:
          input_ids      — [BOS] + prompt + answer + [EOS], padded with pad_token_id.
          attention_mask — 1 on real tokens (non-pad), 0 on pad.
          loss_mask      — 1 on answer tokens and the final EOS, 0 elsewhere.

    Truncation: if the raw sequence exceeds text_len, it is clipped to
    text_len-1 tokens and a final EOS (loss_mask=1) is appended, so the
    output is always exactly text_len tokens with a valid EOS at the end.
    """
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)

    ids  = [bos] + p_ids + a_ids + [eos]
    loss = [0]   + [0] * len(p_ids) + [1] * len(a_ids) + [1]

    if len(ids) > text_len:           # truncate, always keep a final EOS
        ids  = ids[:text_len - 1]  + [eos]
        loss = loss[:text_len - 1] + [1]

    attn  = [1] * len(ids)
    pad_n = text_len - len(ids)
    ids  += [pad] * pad_n
    attn += [0]   * pad_n
    loss += [0]   * pad_n

    return {"input_ids": ids, "attention_mask": attn, "loss_mask": loss}


def _first_caption(value):
    """Datasets store captions as either a str or a list[str]; take one str."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else ''
    return value


def _load_records(name, split, n, streaming, label_field=None):
    """Load up to n records from an HF dataset (streaming keeps the smoke cheap).

    Returns (records, label_names). label_names is the ClassLabel name list when
    label_field is given (so an image-classification set can stand in as a
    caption source: caption = class name), else None.
    """
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=streaming)
    label_names = None
    if label_field is not None:
        feat = ds.features.get(label_field) if ds.features else None
        label_names = getattr(feat, 'names', None)
    if streaming:
        if n:
            ds = ds.take(n)
        records = list(ds)
    else:
        if n:
            ds = ds.select(range(min(n, len(ds))))
        records = list(ds)
    return records, label_names


@torch.no_grad()
def _precompute_features(records, image_column, processor, tower, device,
                         batch_size=16):
    """Run the frozen SigLIP tower over all images once → (N, num_img, dim) cpu."""
    tower = tower.to(device).eval()
    out = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        imgs = [r[image_column].convert('RGB') for r in chunk]
        pixel_values = processor(
            images=imgs, return_tensors='pt')['pixel_values'].to(device)
        feats = tower(pixel_values)
        out.append(feats.float().cpu())
    return torch.cat(out, dim=0)


class MMDataset(torch.utils.data.Dataset):
    """Map-style dataset over precomputed image features + caption text."""

    def __init__(self, records, image_features, tokenizer, caption_prompt,
                 text_len, caption_column, label_names=None):
        assert len(records) == len(image_features)
        self.records = records
        self.image_features = image_features
        self.tokenizer = tokenizer
        self.caption_prompt = caption_prompt
        self.text_len = text_len
        self.caption_column = caption_column
        self.label_names = label_names

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        raw = self.records[i][self.caption_column]
        if self.label_names is not None:   # classification label -> class name
            caption = self.label_names[int(raw)]
        else:
            caption = _first_caption(raw)
        tl = build_prompt_labels(
            self.tokenizer, self.caption_prompt, caption, self.text_len)
        return {
            'input_ids': torch.tensor(tl['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tl['attention_mask'],
                                           dtype=torch.float),
            'loss_mask': torch.tensor(tl['loss_mask'], dtype=torch.float),
            'image_features': self.image_features[i],
        }


def get_mm_dataloaders(config, tokenizer):
    """Build train/valid loaders. Precomputes SigLIP features up front; the
    vision tower is freed before training so it never enters the train graph."""
    from transformers import AutoImageProcessor

    from models.vision import SiglipVisionTower

    v = config.vlm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = AutoImageProcessor.from_pretrained(v.encoder_name)
    tower = SiglipVisionTower(v.encoder_name)
    assert tower.num_image_tokens == v.num_image_tokens, (
        f'config.vlm.num_image_tokens={v.num_image_tokens} != '
        f'tower={tower.num_image_tokens}')

    n = v.get('max_examples', None)
    streaming = v.get('streaming', True)
    label_field = v.caption_column if v.get('label_as_caption', False) else None
    records, label_names = _load_records(
        v.dataset, v.split, n, streaming, label_field)
    if not records:
        raise ValueError(f'No records loaded from {v.dataset}:{v.split}')
    feats = _precompute_features(records, v.image_column, processor, tower,
                                 device)

    # Free the tower — features are cached; it is not needed during training.
    del tower
    if device == 'cuda':
        torch.cuda.empty_cache()

    n_valid = max(2, len(records) // 10)
    train_ds = MMDataset(records, feats, tokenizer, v.caption_prompt,
                         v.text_len, v.caption_column, label_names)
    valid_ds = MMDataset(records[:n_valid], feats[:n_valid], tokenizer,
                         v.caption_prompt, v.text_len, v.caption_column,
                         label_names)

    num_workers = config.loader.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.loader.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=config.loader.pin_memory,
        persistent_workers=num_workers > 0)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config.loader.eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=config.loader.pin_memory,
        persistent_workers=num_workers > 0)
    train_loader.tokenizer = tokenizer
    valid_loader.tokenizer = tokenizer
    return train_loader, valid_loader
