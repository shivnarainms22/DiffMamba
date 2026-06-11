"""Hydra entry for the Stage-1 understanding VLM (image -> text).

Mirrors main.py but builds MMDiffusion + the multimodal dataloader. Kept
separate so the text-only DiffMamba entry (main.py) stays untouched.

  python main_vlm.py +experiment=vlm_smoke                 # pipeline smoke
  python main_vlm.py +experiment=vlm_stage1_align mode=vlm_train
  python main_vlm.py +experiment=vlm_stage1_align mode=vlm_sample \
    eval.checkpoint_path=.../checkpoints/best.ckpt         # (Task 12)
"""
import itertools
import os
import copy

import fsspec
import hydra
import lightning as L
import omegaconf
import torch
from omegaconf import open_dict

# Same self-produced-checkpoint torch.load shim as main.py (Lightning passes
# weights_only=True explicitly on resume; our ckpts store OmegaConf objects).
_real_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs['weights_only'] = False
    return _real_torch_load(*args, **kwargs)


torch.load = _torch_load_compat
torch.set_float32_matmul_precision('high')

import dataloader
import gen_dataloader
import mm_dataloader
import unified_dataloader
import utils
from gen_diffusion import GenDiffusion
from mm_diffusion import MMDiffusion
from unified_diffusion import UnifiedDiffusion
from vlm_eval_utils import score_vqa_answer, summarize_vqa_rows
from grounding_diag_utils import (
    grad_norm_ratio, image_prefix_variation, logit_distribution_shift)

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _build_logger(config):
    if config.get('wandb', None) is not None:
        return L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config), **config.wandb)
    return L.pytorch.loggers.CSVLogger(
        save_dir=config.checkpointing.save_dir, name='logs',
        flush_logs_every_n_steps=10)


def _callbacks(config):
    cbs = []
    if 'callbacks' in config:
        for _, cb in config.callbacks.items():
            cbs.append(hydra.utils.instantiate(cb))
    return cbs


def _vlm_train(config, logger, tokenizer):
    logger.info('Starting VLM training.')
    wandb_logger = _build_logger(config)

    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    train_ds, valid_ds = mm_dataloader.get_mm_dataloaders(config, tokenizer)
    model = MMDiffusion(config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=_callbacks(config),
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _vlm_sample(config, logger, tokenizer):
    logger.info('Generating image -> text samples.')
    model = _load_eval_model(config, tokenizer)

    _, valid_ds = mm_dataloader.get_mm_dataloaders(config, tokenizer)
    batch = next(iter(valid_ds))
    image_features = batch['image_features'].to('cuda')

    prompt = [tokenizer.bos_token_id] + tokenizer.encode(
        config.vlm.caption_prompt, add_special_tokens=False)
    prompt_ids = torch.tensor(prompt, device='cuda')[None].repeat(
        image_features.shape[0], 1)

    out = model._sample_conditional(
        image_features, prompt_ids, num_steps=config.sampling.steps)
    for i, text in enumerate(tokenizer.batch_decode(out)):
        print(f'[sample {i}] {text}')


def _load_eval_model(config, tokenizer):
    model = MMDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config).to('cuda')
    if config.eval.disable_ema:
        model.ema = None
    elif model.ema is not None:
        import itertools
        model.ema.move_shadow_params_to_device(model.device)
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.noise.parameters()))
    model.eval()
    return model


def _norm(s):
    import re
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def _vlm_eval(config, logger, tokenizer):
    """Held-out eval: generate an answer per real question and score against the
    gold answer. exact = normalized first answer segment equals gold;
    recall = gold string appears anywhere in the generated answer."""
    import json
    model = _load_eval_model(config, tokenizer)
    _, valid_ds = mm_dataloader.get_mm_dataloaders(config, tokenizer)
    ds = valid_ds.dataset            # MMDataset: holds prompt/answer text
    n_eval = min(len(ds), int(config.eval.get('num_eval', 200)))
    steps = config.sampling.steps

    exact = recall = 0
    rows = []
    for i in range(n_eval):
        rec = ds.text_records[i]
        q, gold = rec['prompt'], rec['answer']
        feats = ds[i]['image_features'][None].to('cuda')
        p = [tokenizer.bos_token_id] + tokenizer.encode(
            q, add_special_tokens=False)
        prompt_ids = torch.tensor(p, device='cuda')[None]
        out = model._sample_conditional(feats, prompt_ids, num_steps=steps)
        ans_ids = out[0, len(p):].tolist()
        if tokenizer.eos_token_id in ans_ids:
            ans_ids = ans_ids[:ans_ids.index(tokenizer.eos_token_id)]
        gen = tokenizer.decode(ans_ids).strip()

        ng, ngold = _norm(gen), _norm(gold)
        is_exact = bool(ngold) and (ng == ngold or ng.startswith(ngold + ' '))
        is_recall = bool(ngold) and ngold in ng
        exact += is_exact
        recall += is_recall
        rows.append({'question': q, 'gold': gold, 'generated': gen,
                     'exact': is_exact, 'recall': is_recall})

    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    out_path = os.path.join(config.checkpointing.save_dir, 'vlm_eval.json')
    summary = {'n': n_eval, 'exact_match': exact / n_eval,
               'gold_recall': recall / n_eval, 'sampling_steps': steps}
    json.dump({'summary': summary, 'rows': rows}, open(out_path, 'w'), indent=2)
    print(f'VLM eval ({n_eval} held-out VQAv2): '
          f'exact-match={summary["exact_match"]:.3f}  '
          f'gold-recall={summary["gold_recall"]:.3f}')
    print(f'Per-example rows written to {out_path}')


def _gen_train(config, logger, tokenizer):
    logger.info('Starting text->image generation training.')
    wandb_logger = _build_logger(config)
    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    train_ds, valid_ds = gen_dataloader.get_gen_dataloaders(config, tokenizer)
    model = GenDiffusion(config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=_callbacks(config),
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _gen_build_prompts(captions, tokenizer, v):
    """[BOS] caption(padded to caption_len) [BOI] — fixed length P for all rows."""
    rows = []
    for c in captions:
        cap = tokenizer.encode(c, add_special_tokens=False)[:v.caption_len]
        cap = cap + [tokenizer.pad_token_id] * (v.caption_len - len(cap))
        rows.append([tokenizer.bos_token_id] + cap + [v.boi_id])
    return torch.tensor(rows, device='cuda')


def _gen_sample(config, logger, tokenizer):
    logger.info('Generating images from captions.')
    import numpy as np
    from PIL import Image

    from models.vq import VQTokenizer

    model = GenDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config).to('cuda')
    if not config.eval.disable_ema and model.ema is not None:
        model.ema.move_shadow_params_to_device(model.device)
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.noise.parameters()))
    model.eval()

    _, valid_ds = gen_dataloader.get_gen_dataloaders(config, tokenizer)
    n = config.sampling.get('num_sample_log', 8)
    captions = valid_ds.dataset.captions[:n]
    prompt_ids = _gen_build_prompts(captions, tokenizer, config.vlm)
    codes = model._sample_image(prompt_ids, num_steps=config.sampling.steps)

    vq = VQTokenizer(config.vlm.vq_repo,
                     subfolder=config.vlm.get('vq_subfolder', None)).to('cuda').eval()
    imgs = vq.decode(codes)                              # (B,3,H,W) in [-1,1]
    out_dir = os.path.join(config.checkpointing.save_dir, 'gen_samples')
    os.makedirs(out_dir, exist_ok=True)
    for i, (cap, img) in enumerate(zip(captions, imgs)):
        arr = ((img.float() + 1) * 127.5).clamp(0, 255).byte()
        arr = arr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_dir, f'sample_{i:02d}.png'))
        print(f'[gen {i}] "{cap[:60]}" -> sample_{i:02d}.png')
    print(f'Saved {len(captions)} images to {out_dir}')


def _gen_generate_pils(model, vq, captions, tokenizer, config, batch_size=8):
    import numpy as np
    from PIL import Image
    pil_imgs = []
    for s in range(0, len(captions), batch_size):
        prompt_ids = _gen_build_prompts(captions[s:s + batch_size], tokenizer,
                                        config.vlm)
        codes = model._sample_image(prompt_ids, num_steps=config.sampling.steps)
        for img in vq.decode(codes):
            arr = ((img.float() + 1) * 127.5).clamp(0, 255).byte()
            pil_imgs.append(Image.fromarray(
                arr.permute(1, 2, 0).cpu().numpy().astype(np.uint8)))
    return pil_imgs


def _gen_eval(config, logger, tokenizer):
    """Held-out CLIP-score: cosine(CLIP image, CLIP caption). Compare matched
    vs shuffled captions — matched > shuffled means images are text-conditioned."""
    import json
    from transformers import CLIPModel, CLIPProcessor

    from models.vq import VQTokenizer

    model = GenDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config).to('cuda')
    if not config.eval.disable_ema and model.ema is not None:
        model.ema.move_shadow_params_to_device(model.device)
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.noise.parameters()))
    model.eval()

    _, valid_ds = gen_dataloader.get_gen_dataloaders(config, tokenizer)
    n = min(len(valid_ds.dataset), int(config.eval.get('num_eval', 64)))
    captions = valid_ds.dataset.captions[:n]

    vq = VQTokenizer(config.vlm.vq_repo,
                     subfolder=config.vlm.get('vq_subfolder', None)).to('cuda').eval()
    pil_imgs = _gen_generate_pils(model, vq, captions, tokenizer, config)

    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda').eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    with torch.no_grad():
        inputs = proc(text=captions, images=pil_imgs, return_tensors='pt',
                      padding=True, truncation=True).to('cuda')
        out = clip(**inputs)
        ifeat = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        tfeat = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    matched = (ifeat * tfeat).sum(-1).mean().item()
    mismatched = (ifeat * torch.roll(tfeat, 1, 0)).sum(-1).mean().item()

    summary = {'n': n, 'clip_matched': matched,
               'clip_mismatched_shuffled': mismatched,
               'sampling_steps': config.sampling.steps}
    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    out = os.path.join(config.checkpointing.save_dir, 'gen_eval.json')
    json.dump(summary, open(out, 'w'), indent=2)
    print(f'Gen CLIP-score (n={n}): matched={matched:.3f} vs '
          f'shuffled={mismatched:.3f}  '
          f'(matched>shuffled => images are caption-conditioned)')
    print(f'Written {out}')


def _uni_train(config, logger, tokenizer):
    logger.info('Starting UNIFIED understand+generate training.')
    wandb_logger = _build_logger(config)
    if (config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    train_ds, valid_ds = unified_dataloader.get_unified_dataloaders(config, tokenizer)
    warmstart = config.vlm.get('unified_warmstart_path', '')
    if warmstart and ckpt_path is None:
        # SFT: load full unified weights (backbone+projector+noise+EMA) from a prior
        # unified checkpoint; fresh optimizer/step (trainer.fit ckpt_path stays None).
        # NOTE: the warm-start config (uni_vqa_sft.yaml) empties vlm.warmstart_path and
        # vlm.projector_warmstart_path so __init__'s PARTIAL warm-starts are skipped —
        # this full checkpoint load supplies all weights. Keep them empty alongside
        # unified_warmstart_path, else the partial loads fire and are clobbered here.
        model = UnifiedDiffusion.load_from_checkpoint(warmstart, config=config, tokenizer=tokenizer)
    else:
        model = UnifiedDiffusion(config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=_callbacks(config),
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


def _clip_text_embeds(clip, proc, texts):
    ti = proc(text=texts, return_tensors='pt', padding=True, truncation=True).to('cuda')
    pooled = clip.text_model(input_ids=ti['input_ids'],
                            attention_mask=ti['attention_mask']).pooler_output
    e = clip.text_projection(pooled)
    return e / e.norm(dim=-1, keepdim=True)


def _uni_eval(config, logger, tokenizer):
    """Evaluate BOTH directions of the unified model, each as matched-vs-shuffled
    CLIP: understanding = text-CLIP(generated caption, gold caption);
    generation = CLIP(generated image, caption)."""
    import json
    from transformers import CLIPModel, CLIPProcessor

    from models.vq import VQTokenizer

    model = UnifiedDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config).to('cuda')
    if not config.eval.disable_ema and model.ema is not None:
        model.ema.move_shadow_params_to_device(model.device)
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.projector.parameters(),
                                          model.noise.parameters()))
    model.eval()

    _, valid = unified_dataloader.get_unified_dataloaders(config, tokenizer)
    u_ds = valid.iterables['u'].dataset
    g_ds = valid.iterables['g'].dataset
    n = min(int(config.eval.get('num_eval', 48)), len(u_ds), len(g_ds))
    v = config.vlm

    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda').eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    # --- understanding: caption held-out images, compare to gold caption ---
    gen_caps, gold_caps = [], []
    p = [tokenizer.bos_token_id] + tokenizer.encode(
        v.caption_prompt, add_special_tokens=False)
    with torch.no_grad():
        for s in range(0, n, 8):
            idxs = list(range(s, min(s + 8, n)))
            feats = torch.stack([u_ds[i]['image_features'] for i in idxs]).to('cuda')
            prompt_ids = torch.tensor(p, device='cuda')[None].repeat(len(idxs), 1)
            out = model._sample_caption(feats, prompt_ids,
                                        num_steps=config.sampling.steps)
            for j, i in enumerate(idxs):
                ids = out[j].tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                gen_caps.append(tokenizer.decode(ids).strip() or ' ')
                gold_caps.append(u_ds.text_records[i]['answer'])
        ge, go = _clip_text_embeds(clip, proc, gen_caps), _clip_text_embeds(clip, proc, gold_caps)
    u_matched = (ge * go).sum(-1).mean().item()
    u_shuffled = (ge * torch.roll(go, 1, 0)).sum(-1).mean().item()

    # --- DIAGNOSTIC: are generated captions actually varying per image? ---
    print('--- DEBUG: first 5 (generated | gold) captions ---')
    for gc, gd in list(zip(gen_caps, gold_caps))[:5]:
        print(f'  GEN : {gc[:70]!r}')
        print(f'  GOLD: {gd[:70]!r}')
    print(f'  unique generated captions: {len(set(gen_caps))} / {len(gen_caps)}')

    # --- generation: caption -> image, CLIP(image, caption) ---
    caps = g_ds.captions[:n]
    vq = VQTokenizer(v.vq_repo, subfolder=v.get('vq_subfolder', None)).to('cuda').eval()
    pil_imgs = _gen_generate_pils(model, vq, caps, tokenizer, config)
    with torch.no_grad():
        inp = proc(text=caps, images=pil_imgs, return_tensors='pt',
                   padding=True, truncation=True).to('cuda')
        o = clip(**inp)
        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
    g_matched = (ie * te).sum(-1).mean().item()
    g_shuffled = (ie * torch.roll(te, 1, 0)).sum(-1).mean().item()

    summary = {'n': n,
               'understand_matched': u_matched, 'understand_shuffled': u_shuffled,
               'generate_matched': g_matched, 'generate_shuffled': g_shuffled}
    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    out_path = os.path.join(config.checkpointing.save_dir, 'uni_eval.json')
    json.dump(summary, open(out_path, 'w'), indent=2)
    print(f'UNIFIED eval (n={n}):')
    print(f'  understand (gen caption vs gold, text-CLIP): '
          f'matched={u_matched:.3f} vs shuffled={u_shuffled:.3f}')
    print(f'  generate   (gen image vs caption, CLIP):     '
          f'matched={g_matched:.3f} vs shuffled={g_shuffled:.3f}')
    print('  matched>shuffled in BOTH => one model understands AND generates')
    print(f'  written {out_path}')


def _load_unified_eval_model(config, tokenizer):
    model = UnifiedDiffusion.load_from_checkpoint(
        config.eval.checkpoint_path, tokenizer=tokenizer, config=config).to('cuda')
    if config.eval.disable_ema:
        model.ema = None
    elif model.ema is not None:
        model.ema.move_shadow_params_to_device(model.device)
        model.ema.copy_to(itertools.chain(model.backbone.parameters(),
                                          model.projector.parameters(),
                                          model.noise.parameters()))
    model.eval()
    return model


def _unified_vqa_view(config):
    cfg = copy.deepcopy(config)
    v = config.vlm
    with open_dict(cfg):
        cfg.vlm.num_image_tokens = v.siglip_tokens
        cfg.vlm.text_len = v.get('vqa_text_len', v.caption_len)
        cfg.vlm.dataset = v.get('vqa_dataset', 'lmms-lab/VQAv2')
        cfg.vlm.split = v.get('vqa_split', 'validation')
        cfg.vlm.image_column = v.get('vqa_image_column', 'image')
        cfg.vlm.caption_column = v.get(
            'vqa_caption_column', 'multiple_choice_answer')
        cfg.vlm.question_column = v.get('vqa_question_column', 'question')
        cfg.vlm.label_as_caption = False
        cfg.vlm.streaming = True
        cfg.vlm.max_examples = v.get('vqa_max_examples', 40000)
    return cfg


def _decode_answer(tokenizer, ids):
    ids = list(ids)
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    if tokenizer.pad_token_id is not None:
        ids = [i for i in ids if i != tokenizer.pad_token_id]
    return tokenizer.decode(ids).strip()


def _uni_vqa_eval(config, logger, tokenizer):
    """Unified-model VQA eval with shuffled-image ablation.

    This measures the understanding behavior Stage 3 previously could not
    resolve with free-form caption CLIP: exact/recalled short answers against
    VQAv2-style gold labels, plus the same prompts paired with shuffled image
    features to estimate how much accuracy is genuinely image-conditioned.
    """
    import json

    model = _load_unified_eval_model(config, tokenizer)
    _, valid_ds = mm_dataloader.get_mm_dataloaders(
        _unified_vqa_view(config), tokenizer)
    ds = valid_ds.dataset
    n_eval = min(len(ds), int(config.eval.get('num_eval', 200)))
    steps = config.sampling.steps

    rows = []
    with torch.no_grad():
        for i in range(n_eval):
            rec = ds.text_records[i]
            q, gold = rec['prompt'], rec['answer']
            prompt = [tokenizer.bos_token_id] + tokenizer.encode(
                q, add_special_tokens=False)
            prompt_ids = torch.tensor(prompt, device='cuda')[None]

            feats = ds[i]['image_features'][None].to('cuda')
            # Ablation image must be a DIFFERENT image. VQAv2 stores several
            # questions per image consecutively, so the adjacent example (i+1)
            # often shares i's image -> a no-op "shuffle". Offset by ~half the
            # eval set to guarantee a different image_id.
            shuf_idx = (i + max(1, n_eval // 2)) % n_eval
            shuffled_feats = ds[shuf_idx]['image_features'][None].to('cuda')

            out = model._sample_caption(
                feats, prompt_ids, num_steps=steps)
            shuffled = model._sample_caption(
                shuffled_feats, prompt_ids, num_steps=steps)

            gen = _decode_answer(tokenizer, out[0].tolist())
            gen_shuf = _decode_answer(tokenizer, shuffled[0].tolist())
            exact, recall = score_vqa_answer(gen, gold)
            shuf_exact, shuf_recall = score_vqa_answer(gen_shuf, gold)

            rows.append({
                'question': q,
                'gold': gold,
                'generated_correct': gen,
                'generated_shuffled': gen_shuf,
                'correct_exact': exact,
                'correct_recall': recall,
                'shuffled_exact': shuf_exact,
                'shuffled_recall': shuf_recall,
            })

    summary = summarize_vqa_rows(rows, sampling_steps=steps)
    summary['checkpoint'] = config.eval.checkpoint_path
    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    out_path = os.path.join(config.checkpointing.save_dir, 'uni_vqa_eval.json')
    json.dump({'summary': summary, 'rows': rows}, open(out_path, 'w'), indent=2)

    print(f'UNIFIED VQA eval ({n_eval} held-out VQAv2-style examples):')
    print(f'  correct image:  exact={summary["correct_exact_match"]:.3f}  '
          f'recall={summary["correct_gold_recall"]:.3f}')
    print(f'  shuffled image: exact={summary["shuffled_exact_match"]:.3f}  '
          f'recall={summary["shuffled_gold_recall"]:.3f}')
    print(f'  image delta:    exact={summary["image_ablation_exact_delta"]:.3f}  '
          f'recall={summary["image_ablation_recall_delta"]:.3f}')
    print(f'  written {out_path}')


def _uni_vqa_diag(config, logger, tokenizer):
    """Diagnostic for the all-empty uni_vqa_eval generations.

    For the first few VQAv2-style examples, generate an answer region with BOTH
    the actual question prompt AND the in-distribution caption prompt
    ("Describe the image."), printing the raw token ids and the decoded string.

    Reads the result like this:
      - caption decoded is non-empty (real words) but the question answer is
        empty/EOS-first  -> the image+feature pipeline is fine; the unified
        captioner simply has no question-answering ability (OOD prompt).
      - caption answer is ALSO empty -> the VQAv2 feature pipeline is broken,
        not the prompt; chase the features.
    """
    model = _load_unified_eval_model(config, tokenizer)
    _, valid_ds = mm_dataloader.get_mm_dataloaders(
        _unified_vqa_view(config), tokenizer)
    ds = valid_ds.dataset
    steps = config.sampling.steps
    cap_prompt = config.vlm.caption_prompt
    n = min(len(ds), int(config.eval.get('num_diag', 8)))

    def _gen(feats, text):
        p = [tokenizer.bos_token_id] + tokenizer.encode(
            text, add_special_tokens=False)
        pids = torch.tensor(p, device='cuda')[None]
        return model._sample_caption(feats, pids, num_steps=steps)[0].tolist()

    print(f'EOS id={tokenizer.eos_token_id}  PAD id={tokenizer.pad_token_id}  '
          f'caption_prompt={cap_prompt!r}  steps={steps}')
    with torch.no_grad():
        for i in range(n):
            rec = ds.text_records[i]
            q, gold = rec['prompt'], rec['answer']
            feats = ds[i]['image_features'][None].to('cuda')
            qout = _gen(feats, q)
            cout = _gen(feats, cap_prompt)
            print(f'[{i}] Q={q!r} gold={gold!r}')
            print(f'    question -> ids[:12]={qout[:12]}  '
                  f'decoded={_decode_answer(tokenizer, qout)!r}')
            print(f'    caption  -> ids[:12]={cout[:12]}  '
                  f'decoded={_decode_answer(tokenizer, cout)!r}')


def _uni_grounding_diag(config, logger, tokenizer):
    """Localize WHERE image grounding fails, read-only on the frozen ckpt.

    The behavioral shuffle (uni_vqa_eval) proved the model answers VQA blind but
    not *why*. Three probes triangulate the failure (see
    docs/superpowers/specs/2026-06-10-grounding-diagnostic-scope.md):

      A connector collapse -> does the projector map distinct images to distinct
        prefixes, or to ~one vector? (cosine ~1 / std ~0 == dead)
      B logit sensitivity  -> swapping the image (A vs B vs zeroed), sampler-free,
        does the answer-region distribution actually move?
      C gradient saliency  -> does the answer log-prob carry gradient back to the
        image prefix at all, vs the text? (ratio ~0 == attention-dead prefix)

    Every probe runs on uni_stage3/best.ckpt with no training. The result picks
    which retrain is justified, so we don't spend GPU guessing.
    """
    import json

    from models.mm_ops import assemble_mm_embeds, slice_text_logits

    model = _load_unified_eval_model(config, tokenizer)
    _, valid_ds = mm_dataloader.get_mm_dataloaders(
        _unified_vqa_view(config), tokenizer)
    ds = valid_ds.dataset
    steps = config.sampling.steps
    siglip = config.vlm.siglip_tokens
    n_probe = min(len(ds), int(config.eval.get('num_probe', 16)))
    pdtype = next(model.projector.parameters()).dtype

    def _masked_state(prompt_ids):
        """Initial caption state: prompt fixed, answer region all MASK (mirrors
        _sample_caption)."""
        b, p = prompt_ids.shape
        length = config.vlm.caption_len + 8
        x = torch.full((b, p + length), model.mask_index,
                       dtype=torch.long, device='cuda')
        x[:, :p] = prompt_ids
        return x, p

    def _first_sigma(b):
        st = model.noise(torch.ones(b, 1, device='cuda'))[0]
        return st.squeeze(-1) if st.ndim > 1 else st

    # ---- Probe A: connector collapse across DISTINCT images ----
    # VQAv2 groups questions per image; stride the indices to hit different images.
    k = min(n_probe, len(ds))
    stride = max(1, len(ds) // k)
    a_idxs = [(j * stride) % len(ds) for j in range(k)]
    with torch.no_grad():
        feats_k = torch.stack([ds[i]['image_features'] for i in a_idxs]).to('cuda')
        proj = model.projector(feats_k.to(pdtype))  # (K, N, D)
    ref_scale = model.backbone.model.get_input_embeddings().weight.float().std().item()
    probe_a = image_prefix_variation(proj, ref_scale=ref_scale)

    # ---- Probe B: logit sensitivity to the image (sampler-free) ----
    n_b = min(8, n_probe)
    ab, a0 = [], []
    with torch.no_grad():
        for i in range(n_b):
            rec = ds.text_records[i]
            prompt = [tokenizer.bos_token_id] + tokenizer.encode(
                rec['prompt'], add_special_tokens=False)
            prompt_ids = torch.tensor(prompt, device='cuda')[None]
            x, p = _masked_state(prompt_ids)
            st = _first_sigma(1)

            featsA = ds[i]['image_features'][None].to('cuda')
            shuf = (i + max(1, n_b // 2)) % n_b
            featsB = ds[shuf]['image_features'][None].to('cuda')
            zeros = torch.zeros_like(featsA)

            lpA = model._forward_understand(x, st[:, None], featsA)[:, p:, :]
            lpB = model._forward_understand(x, st[:, None], featsB)[:, p:, :]
            lp0 = model._forward_understand(x, st[:, None], zeros)[:, p:, :]
            ab.append(logit_distribution_shift(lpA[0], lpB[0]))
            a0.append(logit_distribution_shift(lpA[0], lp0[0]))

    def _avg(rows, key):
        return sum(r[key] for r in rows) / max(1, len(rows))

    probe_b = {
        'n': n_b,
        'image_a_vs_b': {'l2': _avg(ab, 'l2'), 'sym_kl': _avg(ab, 'sym_kl')},
        'image_a_vs_zero': {'l2': _avg(a0, 'l2'), 'sym_kl': _avg(a0, 'sym_kl')},
    }

    # ---- Probe C: gradient saliency, image prefix vs text ----
    # Reconstruct _forward_understand (unified_diffusion.py:99-110) with the image
    # and text embeds as leaves so backward populates their grads.
    n_c = min(4, n_probe)
    ratios = []
    for i in range(n_c):
        rec = ds.text_records[i]
        prompt = [tokenizer.bos_token_id] + tokenizer.encode(
            rec['prompt'], add_special_tokens=False)
        prompt_ids = torch.tensor(prompt, device='cuda')[None]
        x, p = _masked_state(prompt_ids)
        st = _first_sigma(1)
        feats = ds[i]['image_features'][None].to('cuda')

        with torch.cuda.amp.autocast(dtype=torch.float32):
            sigma = model._process_sigma(st[:, None])
            c = None
            if model.backbone.temb_strategy is not None:
                c = torch.nn.functional.silu(model.backbone.sigma_map(sigma))
            text_embeds = model.backbone.model.get_input_embeddings()(
                x).detach().requires_grad_(True)
            img = model.projector(feats.to(pdtype)).to(
                text_embeds.dtype).detach().requires_grad_(True)
            fused = assemble_mm_embeds(img, text_embeds)
            logits = model.backbone.model(
                inputs_embeds=fused, time_embeds=c).logits
            logits = slice_text_logits(logits, siglip)
            ans = torch.log_softmax(logits[:, p:, :], dim=-1)
            scalar = ans.max(dim=-1).values.sum()
        model.zero_grad(set_to_none=True)
        scalar.backward()
        ratios.append(grad_norm_ratio(img.grad, text_embeds.grad))

    probe_c = {
        'n': n_c,
        'image_grad_norm': _avg(ratios, 'image_grad_norm'),
        'text_grad_norm': _avg(ratios, 'text_grad_norm'),
        'image_to_text_ratio': _avg(ratios, 'image_to_text_ratio'),
    }

    # ---- Verdict (heuristic; raw numbers are the ground truth) ----
    if (probe_a['mean_pairwise_cosine'] > 0.98
            and probe_a.get('std_vs_ref', 1.0) < 0.05):
        verdict = ('CONNECTOR COLLAPSE (Probe A): projector maps distinct images '
                   'to ~one vector -> re-align the projector.')
    elif (probe_b['image_a_vs_b']['sym_kl'] < 1e-3
          and probe_c['image_to_text_ratio'] < 0.05):
        verdict = ('BACKBONE IGNORES PREFIX (B+C): connector varies but the answer '
                   'is image-invariant and carries ~no gradient to the image -> '
                   'conditioning/objective fix (image-token loss weight / longer SFT).')
    elif probe_b['image_a_vs_b']['sym_kl'] >= 1e-3:
        verdict = ('SIGNAL EXISTS, DECODE WASHES OUT (B): logits move with the image '
                   'yet sampling was image-invariant -> sampler/loss fix, maybe no retrain.')
    else:
        verdict = ('INCONCLUSIVE: inspect raw numbers; thresholds are heuristic.')

    summary = {
        'checkpoint': config.eval.checkpoint_path,
        'sampling_steps': steps,
        'probe_a_connector_variation': probe_a,
        'probe_b_logit_sensitivity': probe_b,
        'probe_c_gradient_saliency': probe_c,
        'suggested_reading': verdict,
    }
    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    out_path = os.path.join(
        config.checkpointing.save_dir, 'uni_grounding_diag.json')
    json.dump(summary, open(out_path, 'w'), indent=2)

    print('GROUNDING DIAGNOSTIC (read-only, frozen uni_stage3):')
    print(f'  [A] connector: pairwise_cosine={probe_a["mean_pairwise_cosine"]:.4f}  '
          f'feature_std={probe_a["mean_feature_std"]:.4f}  '
          f'std_vs_ref={probe_a.get("std_vs_ref", float("nan")):.4f}  '
          f'(cosine~1 / std~0 => connector dead)')
    print(f'  [B] logit shift A-vs-B:    sym_kl={probe_b["image_a_vs_b"]["sym_kl"]:.4f}  '
          f'l2={probe_b["image_a_vs_b"]["l2"]:.4f}')
    print(f'      logit shift A-vs-zero: sym_kl={probe_b["image_a_vs_zero"]["sym_kl"]:.4f}  '
          f'l2={probe_b["image_a_vs_zero"]["l2"]:.4f}  (~0 => image ignored at logits)')
    print(f'  [C] grad ratio image/text={probe_c["image_to_text_ratio"]:.4f}  '
          f'(img={probe_c["image_grad_norm"]:.4f} txt={probe_c["text_grad_norm"]:.4f}; '
          f'~0 => attention-dead prefix)')
    print(f'  => {verdict}')
    print(f'  written {out_path}')


def _vlm_caption_eval(config, logger, tokenizer):
    """Baseline for the unified-eval UNDERSTANDING metric.

    Captions held-out images with the STANDALONE Stage-1 model and scores
    text-CLIP(generated caption, gold caption) matched vs shuffled — the exact
    same metric and CLIP path as `_uni_eval`'s understanding side. This is the
    reference the unified number needs to be interpreted: if a dedicated
    captioner also sits at matched~=shuffled, the metric is at its floor for a
    130M model on noisy CC3M alt-text (not a unification collapse).

    Run under the Stage-1 captioner config so the data is CC3M captioning and
    the config matches the checkpoint, e.g.:
      python main_vlm.py +experiment=vlm_stage1_align mode=vlm_caption_eval \\
        eval.checkpoint_path=.../runs/vlm_align/checkpoints/best.ckpt
    (Generation has no analogous gap: `mode=gen_eval` already reports the
    standalone image-CLIP matched-vs-shuffled baseline.)
    """
    import json
    from transformers import CLIPModel, CLIPProcessor

    model = _load_eval_model(config, tokenizer)
    _, valid_ds = mm_dataloader.get_mm_dataloaders(config, tokenizer)
    ds = valid_ds.dataset
    n = min(len(ds), int(config.eval.get('num_eval', 48)))
    steps = config.sampling.steps

    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to('cuda').eval()
    proc = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    p = [tokenizer.bos_token_id] + tokenizer.encode(
        config.vlm.caption_prompt, add_special_tokens=False)
    gen_caps, gold_caps = [], []
    with torch.no_grad():
        for s in range(0, n, 8):
            idxs = list(range(s, min(s + 8, n)))
            feats = torch.stack([ds[i]['image_features'] for i in idxs]).to('cuda')
            prompt_ids = torch.tensor(p, device='cuda')[None].repeat(len(idxs), 1)
            out = model._sample_conditional(feats, prompt_ids, num_steps=steps)
            for j, i in enumerate(idxs):
                ids = out[j, len(p):].tolist()
                if tokenizer.eos_token_id in ids:
                    ids = ids[:ids.index(tokenizer.eos_token_id)]
                gen_caps.append(tokenizer.decode(ids).strip() or ' ')
                gold_caps.append(ds.text_records[i]['answer'])
        ge = _clip_text_embeds(clip, proc, gen_caps)
        go = _clip_text_embeds(clip, proc, gold_caps)
    matched = (ge * go).sum(-1).mean().item()
    shuffled = (ge * torch.roll(go, 1, 0)).sum(-1).mean().item()

    print('--- DEBUG: first 5 (generated | gold) captions ---')
    for gc, gd in list(zip(gen_caps, gold_caps))[:5]:
        print(f'  GEN : {gc[:70]!r}')
        print(f'  GOLD: {gd[:70]!r}')
    print(f'  unique generated captions: {len(set(gen_caps))} / {len(gen_caps)}')

    summary = {'n': n, 'understand_matched': matched,
               'understand_shuffled': shuffled, 'sampling_steps': steps,
               'checkpoint': config.eval.checkpoint_path}
    os.makedirs(config.checkpointing.save_dir, exist_ok=True)
    out_path = os.path.join(config.checkpointing.save_dir, 'vlm_caption_eval.json')
    json.dump(summary, open(out_path, 'w'), indent=2)
    print(f'STANDALONE caption baseline (n={n}): '
          f'understand matched={matched:.3f} vs shuffled={shuffled:.3f}')
    print('  compare to uni_eval understand matched/shuffled: a clear matched>'
          'shuffled here but not in uni_eval => real forgetting; ~equal in both '
          '=> metric floor, not collapse')
    print(f'  written {out_path}')


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'uni_train':
        _uni_train(config, logger, tokenizer)
        return
    if config.mode == 'uni_eval':
        _uni_eval(config, logger, tokenizer)
        return
    if config.mode == 'uni_vqa_eval':
        _uni_vqa_eval(config, logger, tokenizer)
        return
    if config.mode == 'uni_vqa_diag':
        _uni_vqa_diag(config, logger, tokenizer)
        return
    if config.mode == 'uni_grounding_diag':
        _uni_grounding_diag(config, logger, tokenizer)
        return
    if config.mode == 'vlm_caption_eval':
        _vlm_caption_eval(config, logger, tokenizer)
        return

    if config.mode == 'vlm_sample':
        _vlm_sample(config, logger, tokenizer)
    elif config.mode == 'vlm_eval':
        _vlm_eval(config, logger, tokenizer)
    elif config.mode == 'gen_train':
        _gen_train(config, logger, tokenizer)
    elif config.mode == 'gen_sample':
        _gen_sample(config, logger, tokenizer)
    elif config.mode == 'gen_eval':
        _gen_eval(config, logger, tokenizer)
    else:
        _vlm_train(config, logger, tokenizer)


if __name__ == '__main__':
    main()
