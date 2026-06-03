"""
gen_dataloader.py — data pipeline for the masked-diffusion text→image generation VLM.

- build_gen_sequence: pure sequence template builder (CPU-testable, no torch).
- GenDataset / get_gen_dataloaders: CC3M -> VQ-token int16 disk-memmap dataset.
"""
import hashlib
import json
import os
from typing import Dict, List

import numpy as np
import torch

from gen_vocab import code_to_id


def build_gen_sequence(
    caption_ids,
    image_tokens,
    bos: int,
    boi: int,
    eoi: int,
    pad: int,
    caption_len: int,
    num_image_tokens: int,
) -> Dict[str, List[int]]:
    """Build a single generation sequence: [BOS] caption(padded) [BOI] image_tokens [EOI].

    Args:
        caption_ids:      token ids for the caption text (will be truncated/padded to
                          caption_len).
        image_tokens:     exactly num_image_tokens already-offset vocab ids for the
                          image span (e.g. image_base + code).
        bos:              BOS token id.
        boi:              begin-image marker token id (clean, not noised).
        eoi:              end-image marker token id (generated, supervised).
        pad:              padding token id used to fill a short caption.
        caption_len:      fixed caption length (truncate or right-pad with `pad`).
        num_image_tokens: expected number of image tokens; asserted against len(image_tokens).

    Returns:
        dict with keys:
          input_ids      — [BOS] + caption(padded) + [BOI] + image_tokens + [EOI].
          attention_mask — 1 everywhere except caption-padding positions (which are 0).
          loss_mask      — 0 on BOS+caption+BOI (clean conditioning),
                           1 on image_tokens+EOI (generated targets).
    """
    assert len(image_tokens) == num_image_tokens, (
        f"Expected {num_image_tokens} image tokens, got {len(image_tokens)}"
    )

    # Truncate then right-pad caption to exactly caption_len.
    cap = list(caption_ids[:caption_len])
    cap += [pad] * (caption_len - len(cap))

    ids = [bos] + cap + [boi] + list(image_tokens) + [eoi]

    prompt_end = 1 + caption_len + 1          # BOS + caption + BOI
    loss = [0] * prompt_end + [1] * (num_image_tokens + 1)  # image tokens + EOI

    # Attention: 1 everywhere except caption pad positions.
    attn = (
        [1]
        + [1 if t != pad else 0 for t in cap]
        + [1] * (1 + num_image_tokens + 1)    # BOI + image + EOI always attended
    )

    return {"input_ids": ids, "attention_mask": attn, "loss_mask": loss}


def _first_str(value):
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ''
    return str(value)


def _preprocess_image(pil, size=256):
    """PIL -> (3, size, size) float tensor in [-1, 1] (VQModel input range)."""
    pil = pil.convert('RGB').resize((size, size))
    arr = np.asarray(pil, dtype=np.float32)            # (H, W, 3) in [0, 255]
    return torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0


def _cache_key(v):
    raw = f'{v.dataset}|{v.split}|{v.get("max_examples", None)}|{v.vq_repo}'
    return hashlib.md5(raw.encode()).hexdigest()[:16]


@torch.no_grad()
def _build_or_load_vq_cache(v, cache_dir, vq, device, batch_size=16):
    """Stream CC3M once: VQ-encode each image to an int16 code memmap on disk
    (reused across chained segments); collect captions. Returns
    (codes_path, shape, captions)."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(v)
    codes_path = os.path.join(cache_dir, f'{key}.codes.i16')
    cap_path = os.path.join(cache_dir, f'{key}.caps.json')
    meta_path = os.path.join(cache_dir, f'{key}.meta.json')

    if all(os.path.exists(p) for p in (codes_path, cap_path, meta_path)):
        meta = json.load(open(meta_path))
        captions = json.load(open(cap_path))
        return codes_path, tuple(meta['shape']), captions

    n = v.get('max_examples', None)
    if not n:
        raise ValueError('vlm.max_examples must be set for the VQ-token cache.')

    from datasets import load_dataset
    ds = load_dataset(v.dataset, split=v.split, streaming=True).take(n)

    vq = vq.to(device).eval()
    n_tok = v.num_image_tokens
    mm = np.memmap(codes_path, dtype='int16', mode='w+', shape=(n, n_tok))
    captions = []
    img_buf, idx = [], 0

    def _flush():
        nonlocal idx
        if not img_buf:
            return
        px = torch.stack(img_buf).to(device)
        codes = vq.encode(px).cpu().numpy().astype('int16')
        mm[idx:idx + codes.shape[0]] = codes
        idx += codes.shape[0]
        img_buf.clear()

    for rec in ds:
        captions.append(_first_str(rec[v.caption_column]))
        img_buf.append(_preprocess_image(rec[v.image_column]))
        if len(img_buf) == batch_size:
            _flush()
    _flush()

    count = idx
    mm.flush()
    del mm
    shape = (count, n_tok)
    captions = captions[:count]
    json.dump(captions, open(cap_path, 'w'))
    json.dump({'shape': list(shape)}, open(meta_path, 'w'))
    return codes_path, shape, captions


class GenDataset(torch.utils.data.Dataset):
    """Map-style dataset: caption + VQ code grid -> generation sequence.
    The code memmap is opened lazily per worker (avoids pickling)."""

    def __init__(self, captions, codes_path, codes_shape, tokenizer, v,
                 index_offset=0):
        self.captions = captions
        self.codes_path = codes_path
        self.codes_shape = codes_shape
        self.tokenizer = tokenizer
        self.v = v
        self.index_offset = index_offset
        self._mm = None

    def _codes(self):
        if self._mm is None:
            self._mm = np.memmap(self.codes_path, dtype='int16', mode='r',
                                 shape=self.codes_shape)
        return self._mm

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        codes = np.asarray(self._codes()[self.index_offset + i]).astype('int64')
        image_tokens = [code_to_id(int(c), self.v.image_base) for c in codes]
        cap_ids = self.tokenizer.encode(self.captions[i], add_special_tokens=False)
        tl = build_gen_sequence(
            cap_ids, image_tokens,
            bos=self.tokenizer.bos_token_id, boi=self.v.boi_id,
            eoi=self.v.eoi_id, pad=self.tokenizer.pad_token_id,
            caption_len=self.v.caption_len,
            num_image_tokens=self.v.num_image_tokens)
        return {
            'input_ids': torch.tensor(tl['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tl['attention_mask'], dtype=torch.float),
            'loss_mask': torch.tensor(tl['loss_mask'], dtype=torch.float),
        }


def get_gen_dataloaders(config, tokenizer):
    """Build train/valid loaders backed by a disk-cached VQ-token memmap. The VQ
    tokenizer is used only to build the cache, then freed."""
    from models.vq import VQTokenizer

    v = config.vlm
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vq = VQTokenizer(v.vq_repo, subfolder=v.get('vq_subfolder', None))

    cache_dir = os.path.join(config.data.cache_dir, 'vq_tokens')
    codes_path, shape, captions = _build_or_load_vq_cache(v, cache_dir, vq, device)
    del vq
    if device == 'cuda':
        torch.cuda.empty_cache()

    total = len(captions)
    if total == 0:
        raise ValueError(f'No records cached from {v.dataset}:{v.split}')
    n_valid = max(1, min(total // 10, 256))
    n_train = total - n_valid

    train_ds = GenDataset(captions[:n_train], codes_path, shape, tokenizer, v,
                          index_offset=0)
    valid_ds = GenDataset(captions[n_train:], codes_path, shape, tokenizer, v,
                          index_offset=n_train)

    nw = config.loader.num_workers
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.loader.batch_size, shuffle=True,
        num_workers=nw, pin_memory=config.loader.pin_memory,
        persistent_workers=nw > 0)
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=config.loader.eval_batch_size, shuffle=False,
        num_workers=nw, pin_memory=config.loader.pin_memory,
        persistent_workers=nw > 0)
    train_loader.tokenizer = tokenizer
    valid_loader.tokenizer = tokenizer
    return train_loader, valid_loader
