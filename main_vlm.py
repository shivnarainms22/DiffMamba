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

import fsspec
import hydra
import lightning as L
import omegaconf
import torch

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
    model = UnifiedDiffusion(config, tokenizer=tokenizer)

    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=_callbacks(config),
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'uni_train':
        _uni_train(config, logger, tokenizer)
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
