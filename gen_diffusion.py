"""GenDiffusion — the Stage-2 text->image generation module.

Subclasses the text-only `diffusion.Diffusion` (untouched) and changes only what
generation needs:
  - rebuilds the backbone with an EXPANDED vocab (text + [BOI]/[EOI] + VQ codebook),
  - warm-starts the text rows from a DiffMamba checkpoint (image/marker rows fresh),
  - noises ONLY the image-token span (caption stays clean conditioning) and weights
    the MDLM loss to the image span via loss_mask.

Pure tokens in / tokens out — the backbone is a plain DiMamba, so forward and the
SUBS parameterization are inherited unchanged. Image-token sampling is added for
inference.
"""
import itertools

import torch

from cfg_utils import cfg_combine_log_probs, make_unconditional_prompt
import models
import utils
from diffusion import Diffusion, Loss, _sample_categorical
from models.dimamba import DiMamba
from warmstart import load_text_rows_into_expanded


class GenDiffusion(Diffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        assert self.parameterization == 'subs', \
            'GenDiffusion supports the SUBS parameterization only.'

        base_vocab = self.vocab_size            # text + [MASK] (e.g. 50258)
        C = config.vlm.codebook_size
        self.boi_id = base_vocab                # [BOI]
        self.eoi_id = base_vocab + 1            # [EOI]
        self.image_base = base_vocab + 2        # first image-code token id
        self.codebook_size = C
        new_vocab = base_vocab + 2 + C
        self.vocab_size = new_vocab

        # Rebuild the backbone with the expanded vocab (parent built it at base).
        self.backbone = DiMamba(
            config, vocab_size=new_vocab, pad_token_id=tokenizer.pad_token_id)

        warmstart_path = config.vlm.get('warmstart_path', '')
        if warmstart_path:
            info = load_text_rows_into_expanded(
                self.backbone, warmstart_path, base_vocab)
            assert info['unexpected'] == 0 and info['vocab_rows_copied'] >= 1, (
                f'warm-start mismatch: {info}')

        # Rebuild EMA over the new (expanded) backbone params.
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()),
                decay=self.config.training.ema)

    # forward(x, sigma) is inherited: backbone is a DiMamba (tokens in), then
    # _subs_parameterization — identical to the text path.

    def _forward_pass_diffusion(self, x0, loss_mask):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            t += (1 / self.T)

        if self.change_of_variables:
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))[:, None]
            sigma = dsigma = None
            unet_conditioning = t[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        # Conditional masking: only the image span (loss_mask==1) is noised; the
        # caption + markers stay clean conditioning.
        xt = torch.where(loss_mask.bool(), xt, x0)

        model_output = self.forward(xt, unet_conditioning)
        utils.print_nans(model_output, 'model_output')

        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)
        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))
        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def _loss(self, x0, attention_mask, loss_mask):
        per_tok = self._forward_pass_diffusion(x0, loss_mask)
        weight = (attention_mask * loss_mask).to(per_tok.dtype)
        nlls = per_tok * weight
        count = weight.sum()
        token_nll = nlls.sum() / count.clamp(min=1.0)
        return Loss(loss=token_nll, nlls=nlls, token_mask=weight)

    def _compute_loss(self, batch, prefix):
        losses = self._loss(batch['input_ids'], batch['attention_mask'],
                            batch['loss_mask'])
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

    @torch.no_grad()
    def _sample_image(self, prompt_ids, num_steps=None, eps=1e-5):
        """Generate the image-token grid given clean caption-prompt ids.
        prompt_ids: (B, P) = [BOS] caption [BOI]. Returns (B, num_image_tokens)
        code indices in [0, codebook_size)."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        device = prompt_ids.device
        B = prompt_ids.shape[0]
        P = prompt_ids.shape[1]
        n_img = self.config.vlm.num_image_tokens

        # [prompt | masked image span | EOI]
        x = torch.full((B, P + n_img + 1), self.mask_index,
                       dtype=torch.long, device=device)
        x[:, :P] = prompt_ids
        x[:, -1] = self.eoi_id

        timesteps = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        img_lo, img_hi = self.image_base, self.image_base + self.codebook_size
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(B, 1, device=device)
            x = self._ddpm_image_update(x, t, dt, img_lo, img_hi,
                                        prompt_len=P)
            x[:, :P] = prompt_ids                       # keep prompt fixed
            x[:, -1] = self.eoi_id
        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(B, 1, device=device)
            sigma = self.noise(t)[0]
            logits = self._guided_image_forward(x, sigma, prompt_len=P)
            logits[:, :, :img_lo] = self.neg_infinity
            logits[:, :, img_hi:] = self.neg_infinity
            x = logits.argmax(-1)

        img = x[:, P:P + n_img]
        return img - self.image_base                    # -> code indices

    def _guided_image_forward(self, x, sigma, prompt_len):
        scale = float(self.config.sampling.get('cfg_scale', 1.0))
        if scale == 1.0:
            return self.forward(x, sigma)
        x_uncond = make_unconditional_prompt(
            x, prompt_len=prompt_len,
            pad_token_id=self.tokenizer.pad_token_id)
        cond = self.forward(x, sigma)
        uncond = self.forward(x_uncond, sigma)
        return cfg_combine_log_probs(cond, uncond, scale)

    def _ddpm_image_update(self, x, t, dt, img_lo, img_hi, prompt_len):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        log_p_x0 = self._guided_image_forward(x, sigma_t, prompt_len)
        # Restrict generation to valid image-code tokens (mask/text/markers off).
        log_p_x0[:, :, :img_lo] = self.neg_infinity
        log_p_x0[:, :, img_hi:] = self.neg_infinity
        log_p_x0 = log_p_x0 - torch.logsumexp(log_p_x0, dim=-1, keepdim=True)

        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        x_next = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * x_next
