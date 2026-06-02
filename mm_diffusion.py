"""MMDiffusion — the Stage-1 understanding VLM training/inference module.

Subclasses the text-only `diffusion.Diffusion` (kept untouched) and changes only
what multimodality requires:
  - swaps the text backbone for `MMDiMamba` (image-conditioned),
  - warm-starts the LM backbone from a DiffMamba checkpoint,
  - applies LLaVA-style phase freezing (align = projector-only; sft = full FT),
  - noises ONLY the answer span (prompt + image stay clean conditioning) and
    weights the MDLM loss to answer tokens via `loss_mask`.

All the diffusion math (noise schedule, q_xt, SUBS parameterization, EMA,
checkpoint plumbing) is inherited verbatim.
"""
import itertools

import hydra.utils
import torch

import models
import utils
from diffusion import Diffusion, Loss
from models.mm_dimamba import MMDiMamba
from warmstart import load_warmstart


class MMDiffusion(Diffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        assert self.parameterization == 'subs', \
            'MMDiffusion supports the SUBS parameterization only.'

        # Swap the text-only backbone for the multimodal wrapper. self.vocab_size
        # already includes the [MASK] row, matching the DiffMamba lm_head.
        self.backbone = MMDiMamba(
            config, vocab_size=self.vocab_size,
            pad_token_id=tokenizer.pad_token_id)

        # Warm-start the LM backbone from DiffMamba (into the MMDiMamba, whose
        # `.backbone` attribute is the DiMamba — so the 'backbone.*' ckpt keys
        # match directly; projector keys are newly initialized).
        warmstart_path = config.vlm.get('warmstart_path', '')
        if warmstart_path:
            info = load_warmstart(self.backbone, warmstart_path)
            assert info['unexpected'] == 0, (
                f'warm-start key mismatch (unexpected={info["unexpected"]}, '
                f'missing={info["missing"]}) — check the MMDiMamba nesting.')

        # Freeze BEFORE rebuilding EMA: ema.py keeps shadows only for
        # requires_grad params and update() re-filters the same way, so the EMA
        # must be built against the final trainable set. Building it before the
        # freeze would shadow the (then-trainable) backbone, then update() would
        # see only the projector → a positional shadow/param misalignment.
        self._apply_phase_freeze(config.vlm.phase)

        # Rebuild EMA over the NEW backbone parameters (the parent built it over
        # the now-discarded text backbone).
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()),
                decay=self.config.training.ema)

    def _apply_phase_freeze(self, phase):
        if phase == 'align':
            for p in self.backbone.backbone.parameters():
                p.requires_grad_(False)
            for p in self.backbone.projector.parameters():
                p.requires_grad_(True)
        elif phase != 'sft':
            raise ValueError(f'Unknown vlm.phase: {phase!r} (align|sft)')
        # sft: everything trainable (SigLIP is not part of this module).

    def forward(self, x, sigma, image_features):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(x, sigma, image_features)
        return self._subs_parameterization(logits=logits, xt=x)

    def _forward_pass_diffusion(self, x0, image_features, loss_mask):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            t += (1 / self.T)

        if self.change_of_variables:
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]
            sigma = None
            dsigma = None
            unet_conditioning = t[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        # Conditional masking: only the answer span is noised. Prompt and pad
        # positions keep their ground-truth ids and act as clean context —
        # exactly how the image prefix is treated.
        xt = torch.where(loss_mask.bool(), xt, x0)

        model_output = self.forward(xt, unet_conditioning, image_features)
        utils.print_nans(model_output, 'model_output')

        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(
                -torch.exp(-self.noise.sigma_min))
        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def _loss(self, x0, attention_mask, loss_mask, image_features):
        per_tok = self._forward_pass_diffusion(x0, image_features, loss_mask)
        # Supervise only real answer tokens.
        weight = (attention_mask * loss_mask).to(per_tok.dtype)
        nlls = per_tok * weight
        count = weight.sum()
        token_nll = nlls.sum() / count.clamp(min=1.0)
        return Loss(loss=token_nll, nlls=nlls, token_mask=weight)

    def _compute_loss(self, batch, prefix):
        losses = self._loss(
            batch['input_ids'],
            batch['attention_mask'],
            batch['loss_mask'],
            batch['image_features'])
        loss = losses.loss

        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Only optimize trainable params (phase freezing zeros requires_grad on
        # the frozen backbone during alignment).
        params = [p for p in itertools.chain(self.backbone.parameters(),
                                              self.noise.parameters())
                  if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)
        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]
