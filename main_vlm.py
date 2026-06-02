"""Hydra entry for the Stage-1 understanding VLM (image -> text).

Mirrors main.py but builds MMDiffusion + the multimodal dataloader. Kept
separate so the text-only DiffMamba entry (main.py) stays untouched.

  python main_vlm.py +experiment=vlm_smoke                 # pipeline smoke
  python main_vlm.py +experiment=vlm_stage1_align mode=vlm_train
  python main_vlm.py +experiment=vlm_stage1_align mode=vlm_sample \
    eval.checkpoint_path=.../checkpoints/best.ckpt         # (Task 12)
"""
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
import mm_dataloader
import utils
from mm_diffusion import MMDiffusion

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


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    L.seed_everything(config.seed)
    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)

    if config.mode == 'vlm_sample':
        _vlm_sample(config, logger, tokenizer)
    else:
        _vlm_train(config, logger, tokenizer)


if __name__ == '__main__':
    main()
