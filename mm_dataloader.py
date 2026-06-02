"""
mm_dataloader.py — data pipeline for the masked-diffusion understanding VLM.

- build_prompt_labels: pure text-template builder (CPU-testable).
- MMDataset / get_mm_dataloaders: a *generic* image-caption loader. It reads any
  HF dataset that exposes an inline image column and a caption column (configured
  via vlm.image_column / vlm.caption_column), precomputes frozen SigLIP features
  once (kept out of the training graph and out of checkpoints), and yields
  {input_ids, attention_mask, loss_mask, image_features} batches.
"""
import hashlib
import json
import os
from typing import Dict, List

import numpy as np
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


def _first_str(value):
    """Datasets store target text as either a str or a list[str]; take one str."""
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ''
    return str(value)


def _resolve_prompt_answer(rec, v, label_names):
    """Map one raw record to (prompt, answer) text given the column config.

    - label_as_caption: answer = class name for the int label; prompt fixed.
    - question_column set (VQA): prompt = the question; answer = target text.
    - else (captioning): prompt = fixed caption_prompt; answer = caption text.
    """
    if label_names is not None:
        return v.caption_prompt, str(label_names[int(rec[v.caption_column])])
    answer = _first_str(rec[v.caption_column])
    q_col = v.get('question_column', None)
    if q_col:
        return _first_str(rec[q_col]), answer
    return v.caption_prompt, answer


def _cache_key(v):
    raw = f'{v.dataset}|{v.split}|{v.get("max_examples", None)}|{v.encoder_name}'
    return hashlib.md5(raw.encode()).hexdigest()[:16]


@torch.no_grad()
def _build_or_load_cache(v, cache_dir, processor, tower, device,
                         num_img, vis_dim, batch_size=32):
    """Stream the dataset once: encode images → float16 memmap on disk, collect
    (prompt, answer) text. Reuses an existing cache (critical so chained 8h job
    segments don't re-encode). Returns (feat_path, shape, text_records)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(v)
    feat_path = os.path.join(cache_dir, f'{key}.f16.dat')
    text_path = os.path.join(cache_dir, f'{key}.text.json')
    meta_path = os.path.join(cache_dir, f'{key}.meta.json')

    if all(os.path.exists(p) for p in (feat_path, text_path, meta_path)):
        meta = json.load(open(meta_path))
        text_records = json.load(open(text_path))
        return feat_path, tuple(meta['shape']), text_records

    n = v.get('max_examples', None)
    if not n:
        raise ValueError(
            'vlm.max_examples must be set for the memmap feature cache '
            '(streaming has no length to size the memmap).')

    from datasets import load_dataset
    ds = load_dataset(v.dataset, split=v.split, streaming=True)
    label_names = None
    if v.get('label_as_caption', False):
        feat = ds.features.get(v.caption_column) if ds.features else None
        label_names = getattr(feat, 'names', None)
    ds = ds.take(n)

    tower = tower.to(device).eval()
    mm = np.memmap(feat_path, dtype='float16', mode='w+',
                   shape=(n, num_img, vis_dim))
    text_records = []
    img_buf, idx = [], 0

    def _flush():
        nonlocal idx
        if not img_buf:
            return
        pixel_values = processor(
            images=img_buf, return_tensors='pt')['pixel_values'].to(device)
        feats = tower(pixel_values).float().cpu().numpy().astype('float16')
        mm[idx:idx + feats.shape[0]] = feats
        idx += feats.shape[0]
        img_buf.clear()

    for rec in ds:
        prompt, answer = _resolve_prompt_answer(rec, v, label_names)
        text_records.append({'prompt': prompt, 'answer': answer})
        img_buf.append(rec[v.image_column].convert('RGB'))
        if len(img_buf) == batch_size:
            _flush()
    _flush()

    count = idx
    mm.flush()
    del mm
    # Trim to the actual count (streamed set may be < n).
    shape = (count, num_img, vis_dim)
    text_records = text_records[:count]
    json.dump(text_records, open(text_path, 'w'))
    json.dump({'shape': list(shape)}, open(meta_path, 'w'))
    return feat_path, shape, text_records


class MMDataset(torch.utils.data.Dataset):
    """Map-style dataset over a disk feature memmap + (prompt, answer) text.

    The memmap is opened lazily (per worker process) to avoid pickling it across
    DataLoader workers. `index_offset` maps local indices into the shared memmap
    so a contiguous train/valid split reuses one cache file.
    """

    def __init__(self, text_records, feat_path, feat_shape, tokenizer, text_len,
                 index_offset=0):
        self.text_records = text_records
        self.feat_path = feat_path
        self.feat_shape = feat_shape
        self.tokenizer = tokenizer
        self.text_len = text_len
        self.index_offset = index_offset
        self._mm = None

    def _feats(self):
        if self._mm is None:
            self._mm = np.memmap(self.feat_path, dtype='float16', mode='r',
                                 shape=self.feat_shape)
        return self._mm

    def __len__(self):
        return len(self.text_records)

    def __getitem__(self, i):
        rec = self.text_records[i]
        row = np.asarray(self._feats()[self.index_offset + i])
        tl = build_prompt_labels(
            self.tokenizer, rec['prompt'], rec['answer'], self.text_len)
        return {
            'input_ids': torch.tensor(tl['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tl['attention_mask'],
                                           dtype=torch.float),
            'loss_mask': torch.tensor(tl['loss_mask'], dtype=torch.float),
            'image_features': torch.from_numpy(row).float(),
        }


def get_mm_dataloaders(config, tokenizer):
    """Build train/valid loaders backed by a disk-cached SigLIP feature memmap.
    The vision tower is used only to build the cache, then freed — it never
    enters the training graph or the checkpoint."""
    from transformers import AutoImageProcessor

    from models.vision import SiglipVisionTower

    v = config.vlm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = AutoImageProcessor.from_pretrained(v.encoder_name)
    tower = SiglipVisionTower(v.encoder_name)
    assert tower.num_image_tokens == v.num_image_tokens, (
        f'config.vlm.num_image_tokens={v.num_image_tokens} != '
        f'tower={tower.num_image_tokens}')

    cache_dir = os.path.join(config.data.cache_dir, 'vlm_feats')
    feat_path, shape, text_records = _build_or_load_cache(
        v, cache_dir, processor, tower, device,
        tower.num_image_tokens, tower.hidden_size)

    del tower
    if device == 'cuda':
        torch.cuda.empty_cache()

    total = len(text_records)
    if total == 0:
        raise ValueError(f'No records cached from {v.dataset}:{v.split}')
    n_valid = max(1, min(total // 10, 512))
    n_train = total - n_valid

    train_ds = MMDataset(text_records[:n_train], feat_path, shape, tokenizer,
                         v.text_len, index_offset=0)
    valid_ds = MMDataset(text_records[n_train:], feat_path, shape, tokenizer,
                         v.text_len, index_offset=n_train)

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
