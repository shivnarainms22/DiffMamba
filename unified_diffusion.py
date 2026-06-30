"""UnifiedDiffusion — one Mamba-diffusion model that BOTH understands and generates.

Janus-style decoupled design on a single expanded-vocab DiMamba backbone:
  - understand (image->text): inputs_embeds = [projector(SigLIP) | text], mask caption span
  - generate  (text->image): token ids [caption | image tokens], mask image span
  - pure text: standard MDLM over text tokens (optional, prevents language drift)

Warm-started from the text backbone (runD_lr1e3). Reuses Stage-1/Stage-2 building
blocks (mm_ops splice, projector, vocab math, warm-start row-copy, samplers).
diffusion.py / models/dimamba.py are untouched.
"""
import itertools

import torch

from cfg_utils import cfg_combine_log_probs, make_unconditional_prompt
import models
import utils
from diffusion import Diffusion, Loss, _sample_categorical
from models.dimamba import DiMamba
from models.mm_ops import assemble_mm_embeds, slice_text_logits
from models.vision import MLPProjector
from warmstart import load_projector_from_mm, load_text_rows_into_expanded


class UnifiedDiffusion(Diffusion):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        assert self.parameterization == 'subs'

        v = config.vlm
        base_vocab = self.vocab_size
        C = v.codebook_size
        self.boi_id = base_vocab
        self.eoi_id = base_vocab + 1
        self.image_base = base_vocab + 2
        self.codebook_size = C
        self.vocab_size = base_vocab + 2 + C
        self.siglip_tokens = v.siglip_tokens
        self.vq_tokens = v.vq_tokens
        self.w_u = v.get('understand_weight', 1.0)
        self.w_g = v.get('generate_weight', 1.0)
        self.w_t = v.get('text_weight', 0.0)
        self.w_c = v.get('contrastive_weight', 0.0)
        self.num_neg = v.get('num_neg', 1)

        # One shared expanded-vocab backbone + one projector.
        self.backbone = DiMamba(
            config, vocab_size=self.vocab_size, pad_token_id=tokenizer.pad_token_id)
        self.projector = MLPProjector(
            in_dim=v.vision_dim, out_dim=config.model.hidden_size,
            hidden_dim=v.projector_hidden,
            scale_match=v.get('proj_scale_match', False))

        ws = v.get('warmstart_path', '')
        if ws:
            info = load_text_rows_into_expanded(self.backbone, ws, base_vocab)
            assert info['unexpected'] == 0 and info['vocab_rows_copied'] >= 1, info

        # Warm-start the projector from the trained Stage-1 projector so the
        # backbone doesn't learn to ignore a random image prefix (the collapse).
        pws = v.get('projector_warmstart_path', '')
        if pws:
            pinfo = load_projector_from_mm(self.projector, pws)
            assert pinfo['loaded'] > 0 and pinfo['unexpected'] == 0, pinfo

        # Fix the scale-match norm at the text-embed scale (un-throttles the
        # projector gradient; see grounding diagnostic Probe D). Done post-warmstart
        # so it reflects the loaded rows. FROZEN (requires_grad=False): when the
        # weight was learnable, training drove it to ~0 to shrink the image away and
        # answer from the language prior (std_vs_ref 24->0.019, attempt 1). Freezing
        # removes that escape — the model cannot turn the image off via the scale.
        if self.projector.out_norm is not None:
            text_std = self.backbone.model.get_input_embeddings().weight.detach().std()
            self.projector.out_norm.weight.data.fill_(text_std.item())
            self.projector.out_norm.weight.requires_grad_(False)

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(),
                                self.projector.parameters(),
                                self.noise.parameters()),
                decay=self.config.training.ema)

    def on_train_start(self):
        # Skip the parent's fault-tolerant sampler rebuild (it assumes a single
        # loader; we use a CombinedLoader). Just move EMA shadows to device.
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    # The parent's on_save/on_load_checkpoint poke a single train_dataloader's
    # .sampler for fault-tolerant resume — which doesn't exist for a
    # CombinedLoader (dict of loaders). We disabled that sampler in
    # on_train_start, so just persist EMA; Lightning saves global_step, model,
    # and optimizer state by default (enough for the chained-run early-exit).
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])

    def optimizer_step(self, *args, **kwargs):
        # Parent's EMA update iterates backbone+noise; include the projector.
        super(Diffusion, self).optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(self.backbone.parameters(),
                                            self.projector.parameters(),
                                            self.noise.parameters()))

    # ---- understand (image -> text) ----
    def _forward_understand(self, xt, sigma, image_features):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            c = None
            if self.backbone.temb_strategy is not None:
                c = torch.nn.functional.silu(self.backbone.sigma_map(sigma))
            text_embeds = self.backbone.model.get_input_embeddings()(xt)
            img = self.projector(image_features).to(text_embeds.dtype)
            fused = assemble_mm_embeds(img, text_embeds)
            logits = self.backbone.model(inputs_embeds=fused, time_embeds=c).logits
            logits = slice_text_logits(logits, self.siglip_tokens).contiguous()
        return self._subs_parameterization(logits=logits, xt=xt)

    def _understand_loss(self, x0, attention_mask, loss_mask, image_features):
        per_tok = self._cond_fpd(x0, loss_mask,
                                 lambda xt, s: self._forward_understand(xt, s, image_features))
        return self._weighted(per_tok, attention_mask, loss_mask)

    def _understand_contrastive(self, x0, loss_mask, image_features, num_neg=1):
        """In-batch contrastive: the gold answer must be more likely under its
        OWN image than under rolled-in negatives. Same noised xt for every image
        so only the image differs. Reuses _forward_understand."""
        from grounding_train_utils import image_contrastive_loss
        t = self._sample_t(x0.shape[0], x0.device)
        sigma, _ = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = self.q_xt(x0, move_chance)
        xt = torch.where(loss_mask.bool(), xt, x0)  # keep non-answer span clean

        def answer_loglik(feats):
            lp = self._forward_understand(xt, sigma[:, None], feats)  # (B,T,V) logp
            tok = torch.gather(lp, -1, x0[:, :, None]).squeeze(-1)    # (B,T)
            return (tok * loss_mask.to(tok.dtype)).sum(-1)            # (B,)

        cols = [answer_loglik(image_features)]
        for k in range(1, num_neg + 1):
            cols.append(answer_loglik(torch.roll(image_features, shifts=k, dims=0)))
        return image_contrastive_loss(torch.stack(cols, dim=1))

    # ---- generate (text -> image) ----
    def _generate_loss(self, x0, attention_mask, loss_mask):
        per_tok = self._cond_fpd(x0, loss_mask, lambda xt, s: self.forward(xt, s))
        return self._weighted(per_tok, attention_mask, loss_mask)

    # ---- pure text ----
    def _text_loss(self, x0, attention_mask):
        per_tok = self._cond_fpd(x0, None, lambda xt, s: self.forward(xt, s))
        nlls = per_tok * attention_mask
        return Loss(loss=nlls.sum() / attention_mask.sum().clamp(min=1.0),
                    nlls=nlls, token_mask=attention_mask)

    # ---- shared conditional forward-pass-diffusion ----
    def _cond_fpd(self, x0, loss_mask, forward_fn):
        t = self._sample_t(x0.shape[0], x0.device)
        sigma, dsigma = self.noise(t)
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = self.q_xt(x0, move_chance)
        if loss_mask is not None:                       # keep non-target span clean
            xt = torch.where(loss_mask.bool(), xt, x0)
        model_output = forward_fn(xt, sigma[:, None])
        utils.print_nans(model_output, 'model_output')
        log_p = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        return -log_p * (dsigma / torch.expm1(sigma))[:, None]

    @staticmethod
    def _weighted(per_tok, attention_mask, loss_mask):
        weight = (attention_mask * loss_mask).to(per_tok.dtype)
        nlls = per_tok * weight
        return Loss(loss=nlls.sum() / weight.sum().clamp(min=1.0),
                    nlls=nlls, token_mask=weight)

    def _compute_loss(self, batch, prefix):
        metrics = (self.train_metrics if prefix == 'train'
                   else self.valid_metrics if prefix == 'val'
                   else self.test_metrics)
        total = 0.0
        parts = {}
        if 'u' in batch:
            b = batch['u']
            lu = self._understand_loss(b['input_ids'], b['attention_mask'],
                                       b['loss_mask'], b['image_features'])
            total = total + self.w_u * lu.loss
            metrics.update(lu.nlls, lu.token_mask)
            parts['understand'] = lu.loss
            if self.w_c > 0:
                lc = self._understand_contrastive(
                    b['input_ids'], b['loss_mask'], b['image_features'],
                    self.num_neg)
                total = total + self.w_c * lc
                parts['contrastive'] = lc
        if 'g' in batch:
            b = batch['g']
            lg = self._generate_loss(b['input_ids'], b['attention_mask'],
                                     b['loss_mask'])
            total = total + self.w_g * lg.loss
            metrics.update(lg.nlls, lg.token_mask)
            parts['generate'] = lg.loss
        if 't' in batch and self.w_t > 0:
            b = batch['t']
            lt = self._text_loss(b['input_ids'], b['attention_mask'])
            total = total + self.w_t * lt.loss
            parts['text'] = lt.loss

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        for k, val in parts.items():
            self.log(f'{prefix}/{k}_loss', val, on_step=(prefix == 'train'),
                     on_epoch=True, sync_dist=True)
        return total

    # ---- sampling (reuse Stage-1/Stage-2 logic on the shared backbone) ----
    @torch.no_grad()
    def _sample_image(self, prompt_ids, num_steps=None, eps=1e-5):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        device = prompt_ids.device
        B, P = prompt_ids.shape
        x = torch.full((B, P + self.vq_tokens + 1), self.mask_index,
                       dtype=torch.long, device=device)
        x[:, :P] = prompt_ids
        x[:, -1] = self.eoi_id
        ts = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        lo, hi = self.image_base, self.image_base + self.codebook_size
        for i in range(num_steps):
            t = ts[i] * torch.ones(B, 1, device=device)
            x = self._ddpm_range(x, t, dt, lo, hi, prompt_len=P)
            x[:, :P] = prompt_ids
            x[:, -1] = self.eoi_id
        # Final noise removal: fill any remaining MASK with the best valid image
        # code so no MASK (-> out-of-range code) reaches the VQ decoder.
        st = self.noise(ts[-1] * torch.ones(B, 1, device=device))[0]
        st = st.squeeze(-1) if st.ndim > 1 else st
        lp = self._guided_image_forward(x, st, prompt_len=P)
        lp[:, :, :lo] = self.neg_infinity
        lp[:, :, hi:] = self.neg_infinity
        x = torch.where(x == self.mask_index, lp.argmax(-1), x)
        img = x[:, P:P + self.vq_tokens]
        return (img - self.image_base).clamp(0, self.codebook_size - 1)

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

    def _ddpm_range(self, x, t, dt, lo, hi, prompt_len):
        st, _ = self.noise(t)
        ss, _ = self.noise(t - dt)
        st = st.squeeze(-1) if st.ndim > 1 else st
        ss = ss.squeeze(-1) if ss.ndim > 1 else ss
        mt = (1 - torch.exp(-st))[:, None, None]
        ms = (1 - torch.exp(-ss))[:, None, None]
        lp = self._guided_image_forward(x, st, prompt_len)
        lp[:, :, :lo] = self.neg_infinity
        lp[:, :, hi:] = self.neg_infinity
        lp = lp - torch.logsumexp(lp, dim=-1, keepdim=True)
        q = lp.exp() * (mt - ms)
        q[:, :, self.mask_index] = ms[:, :, 0]
        nx = _sample_categorical(q)
        copy = (x != self.mask_index).to(x.dtype)
        return copy * x + (1 - copy) * nx

    @torch.no_grad()
    def _sample_caption(self, image_features, prompt_ids, num_steps=None, eps=1e-5):
        if num_steps is None:
            num_steps = self.config.sampling.steps
        device = image_features.device
        B, P = prompt_ids.shape
        L = self.config.vlm.caption_len + 8           # answer budget
        x = torch.full((B, P + L), self.mask_index, dtype=torch.long, device=device)
        x[:, :P] = prompt_ids
        ts = torch.linspace(1, eps, num_steps + 1, device=device)
        dt = (1 - eps) / num_steps
        for i in range(num_steps):
            t = ts[i] * torch.ones(B, 1, device=device)
            st, _ = self.noise(t)
            ss, _ = self.noise(t - dt)
            st = st.squeeze(-1) if st.ndim > 1 else st
            ss = ss.squeeze(-1) if ss.ndim > 1 else ss
            mt = (1 - torch.exp(-st))[:, None, None]
            ms = (1 - torch.exp(-ss))[:, None, None]
            lp = self._forward_understand(x, st[:, None], image_features)
            q = lp.exp() * (mt - ms)
            q[:, :, self.mask_index] = ms[:, :, 0]
            nx = _sample_categorical(q)
            copy = (x != self.mask_index).to(x.dtype)
            x = copy * x + (1 - copy) * nx
            x[:, :P] = prompt_ids
        # Final fill: no MASK token left in the generated caption.
        st = self.noise(ts[-1] * torch.ones(B, 1, device=device))[0]
        st = st.squeeze(-1) if st.ndim > 1 else st
        lp = self._forward_understand(x, st[:, None], image_features)
        x = torch.where(x == self.mask_index, lp.argmax(-1), x)
        x[:, :P] = prompt_ids
        return x[:, P:]
