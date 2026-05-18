import math
from functools import partial
from typing import Optional, Tuple, Union

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
    MaskedLMOutput,
)

# mamba-ssm renamed the triton submodule between 1.x ("layernorm") and
# 2.x ("layer_norm"). Try both and provide a uniform `rms_norm_fn` shim if
# only the unified `layer_norm_fn` is available (2.x).
RMSNorm = layer_norm_fn = rms_norm_fn = None
try:  # mamba-ssm 2.x
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
    try:
        from mamba_ssm.ops.triton.layer_norm import rms_norm_fn
    except ImportError:
        # 2.x exposes a unified layer_norm_fn with is_rms_norm flag.
        def rms_norm_fn(*args, **kwargs):
            kwargs['is_rms_norm'] = True
            return layer_norm_fn(*args, **kwargs)
except ImportError:
    try:  # mamba-ssm 1.x
        from mamba_ssm.ops.triton.layernorm import (
            RMSNorm,
            layer_norm_fn,
            rms_norm_fn,
        )
    except ImportError:
        RMSNorm = layer_norm_fn = rms_norm_fn = None

from models.dit import (
    TimestepEmbedder,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)

# Mamba-2 defaults for this project (d_state=64 per plan, headdim=64 divides
# d_inner=expand*d_model for all three scales: 512/640/768 with expand=2).
_MAMBA2_DEFAULTS = {
    'd_state': 64,
    'd_conv': 4,
    'expand': 2,
    'headdim': 64,
}


class Block(nn.Module):
    def __init__(
      self,
      dim,
      mixer_cls,
      norm_cls=nn.LayerNorm,
      fused_add_norm=False,
      residual_in_fp32=False,
      modulate=False,
      t_dim=0,
      dropout=0.1,
    ):
      """
      Pre-norm residual block: Add -> LN -> Mixer.
      Returns (hidden_states=mixer_out, residual=pre-merge-residual) — the
      next block folds them via `residual = hidden_states + residual` before
      its own norm. This matches the mamba_ssm Block contract; do not add
      `residual` inside this block.
      """
      super().__init__()
      self.residual_in_fp32 = residual_in_fp32
      self.fused_add_norm = fused_add_norm
      self.mixer = mixer_cls(dim)
      self.norm = norm_cls(dim)

      if self.fused_add_norm:
        assert RMSNorm is not None, 'RMSNorm import fails'
        assert isinstance(
            self.norm, (nn.LayerNorm, RMSNorm)
        ), 'Only LayerNorm and RMSNorm are supported for fused_add_norm'

      self.dropout = dropout

      self.modulate = modulate
      self.t_dim = t_dim
      if modulate:
        assert t_dim > 0, 'modulate=True requires t_dim > 0'
        self.adaLN_modulation = nn.Linear(t_dim, 3 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
      return (
        bias_dropout_add_scale_fused_train
        if self.training
        else bias_dropout_add_scale_fused_inference
      )

    def forward(
      self,
      hidden_states: Tensor,
      residual: Optional[Tensor] = None,
      inference_params=None,
      time_embeds=None,
    ):
        if not self.fused_add_norm:
          residual = (
            (hidden_states + residual)
            if residual is not None
            else hidden_states
          )
          hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
          if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        else:
          fused_add_norm_fn = (
            rms_norm_fn
            if isinstance(self.norm, RMSNorm)
            else layer_norm_fn
          )
          hidden_states, residual = fused_add_norm_fn(
            hidden_states,
            self.norm.weight,
            getattr(self.norm, 'bias', None),
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
            eps=self.norm.eps)

        if self.modulate and time_embeds is not None:
          (shift_msa, scale_msa, gate_msa) = self.adaLN_modulation(
              time_embeds)[:, None].chunk(3, dim=-1)
          hidden_states = modulate_fused(hidden_states, shift_msa, scale_msa)

        mixer_out = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = mixer_out

        if self.modulate and time_embeds is not None:
          # IMPORTANT: pass residual=None. The prenorm Block contract is that
          # the *next* block folds `hidden_states + residual` before its norm.
          # Adding `residual` here would double-count it (residual stream
          # accumulates 2x per layer).
          bias_dropout_scale_fn = self._get_bias_dropout_scale()
          hidden_states = bias_dropout_scale_fn(
            hidden_states, None, gate_msa, None, self.dropout)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
      return self.mixer.allocate_inference_cache(
        batch_size, max_seqlen, dtype=dtype, **kwargs)


class BiMambaConfig(PretrainedConfig):
    """Config for bidirectional Mamba-2 used as a diffusion denoiser."""

    model_type = 'bimamba'

    def __init__(
        self,
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        tie_word_embeddings: bool = True,
        norm_epsilon: float = 1e-5,
        initializer_cfg: Optional[dict] = None,
        bidirectional: bool = True,
        bidirectional_strategy: Union[str, None] = 'add',
        bidirectional_weight_tie: bool = True,
        temb_strategy: Union[str, None] = None,
        d_temb: int = 0,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_word_embeddings = tie_word_embeddings
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.temb_strategy = temb_strategy
        self.d_temb = d_temb
        self.dropout = dropout


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bidirectional=True,
    bidirectional_strategy='add',
    bidirectional_weight_tie=True,
    device=None,
    dtype=None,
    modulate=False,
    d_temb=0,
    dropout=0.1,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    # Merge project-level Mamba-2 defaults; YAML values override.
    ssm_cfg = {**_MAMBA2_DEFAULTS, **ssm_cfg}

    factory_kwargs = {'device': device, 'dtype': dtype}
    bidirectional_kwargs = {
        'bidirectional': bidirectional,
        'bidirectional_strategy': bidirectional_strategy,
        'bidirectional_weight_tie': bidirectional_weight_tie,
    }
    mixer_cls = partial(
        BiMambaWrapper,
        layer_idx=layer_idx,
        **ssm_cfg,
        **bidirectional_kwargs,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        t_dim=d_temb,
        modulate=modulate,
        dropout=dropout,
    )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Bidirectional Mamba-2: forward pass + flipped-reverse pass, outputs summed."""

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = 'add',
        bidirectional_weight_tie: bool = True,
        **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = 'add'
        if bidirectional and bidirectional_strategy not in ['add', 'ew_multiply']:
            raise NotImplementedError(
                f'`{bidirectional_strategy}` bi-directionality strategy not implemented'
            )
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy

        self.mamba_fwd = Mamba2(d_model=d_model, **mamba_kwargs)

        self.mamba_rev = None
        if bidirectional:
            self.mamba_rev = Mamba2(d_model=d_model, **mamba_kwargs)
            if bidirectional_weight_tie:
                # Tie the bulk of the parameters (in/out projections).
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                if self.mamba_fwd.in_proj.bias is not None:
                    self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                if self.mamba_fwd.out_proj.bias is not None:
                    self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states.

        `inference_params` is forwarded to the forward-direction Mamba2 (it does
        support caches); the reverse direction cannot meaningfully reuse a cache
        across autoregressive steps, so we always pass None there.
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)

        if self.bidirectional:
            out_rev = self.mamba_rev(hidden_states.flip(1)).flip(1)
            if self.bidirectional_strategy == 'add':
                out = out + out_rev
            elif self.bidirectional_strategy == 'ew_multiply':
                out = out * out_rev

        return out


class BiMambaEmbeddings(nn.Module):
    def __init__(
        self,
        config: BiMambaConfig,
        input_dim=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if input_dim is None:
            input_dim = config.vocab_size
        self.word_embeddings = nn.Embedding(
            input_dim, config.d_model, **factory_kwargs
        )

    def forward(self, input_ids):
        return self.word_embeddings(input_ids)


class BiMambaMixerModel(nn.Module):
    def __init__(
        self,
        config: BiMambaConfig,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.temb_strategy = config.temb_strategy
        self.config = config
        # The embedding lookup always has `vocab_size` rows; only the feature
        # dim downstream grows when concatenating the time embedding.
        d_model = config.d_model
        if self.temb_strategy and self.temb_strategy == 'concat':
            d_model += config.d_temb
        if self.temb_strategy is None:
            config.d_temb = 0

        self.fused_add_norm = config.fused_add_norm
        self.residual_in_fp32 = config.residual_in_fp32

        self.embeddings = BiMambaEmbeddings(
            config, input_dim=config.vocab_size, **factory_kwargs)

        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError('Failed to import Triton LayerNorm / RMSNorm kernels')

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    modulate=True if config.temb_strategy and 'adaln' in config.temb_strategy else False,
                    d_temb=config.d_temb,
                    dropout=getattr(config, 'dropout', 0.1),
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        if self.temb_strategy and 'adaln' in self.temb_strategy:
            self.adaLN_modulation_final = nn.Linear(
                config.d_temb, 2 * d_model, bias=True
            )
            self.adaLN_modulation_final.weight.data.zero_()
            self.adaLN_modulation_final.bias.data.zero_()

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f

    def pre_apply_temb(self, input_embeds, time_embeds):
        if self.temb_strategy == 'concat':
            input_embeds = torch.cat([time_embeds.unsqueeze(1).expand(
                -1, input_embeds.shape[1], -1), input_embeds], dim=-1)
        elif self.temb_strategy == 'add':
            input_embeds = input_embeds + time_embeds.unsqueeze(1).expand(
                -1, input_embeds.shape[1], -1)
        return input_embeds

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        output_hidden_states=False,
        time_embeds=None,
    ):
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        if (
            time_embeds is not None
            and self.temb_strategy in ['concat', 'add']
        ):
            hidden_states = self.pre_apply_temb(hidden_states, time_embeds)

        residual = None

        for ind, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            layer_out = layer(
                hidden_states, residual, inference_params=None, time_embeds=time_embeds
            )
            hidden_states, residual = layer_out

        if not self.fused_add_norm:
            if self.temb_strategy and 'adaln' in self.temb_strategy:
                raise NotImplementedError('adaln only implemented for fused_add_norm')
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            if time_embeds is not None and self.temb_strategy and 'adaln' in self.temb_strategy:
                shift, scale = self.adaLN_modulation_final(time_embeds)[:, None].chunk(
                    2, dim=2
                )

            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                getattr(self.norm_f, 'bias', None),
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            if time_embeds is not None and self.temb_strategy and 'adaln' in self.temb_strategy:
                hidden_states = modulate_fused(hidden_states, shift, scale)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return hidden_states, all_hidden_states


def cross_entropy(logits, y, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    return F.cross_entropy(logits, y, ignore_index=ignore_index)


def weighted_cross_entropy(logits, y, loss_weights, ignore_index=-100):
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)
    ce = F.cross_entropy(logits, y, ignore_index=ignore_index, reduction='none')
    # Clone so we don't mutate the caller's tensor; .view returns a view.
    loss_weights = loss_weights.reshape(-1).clone()
    loss_weights[y == ignore_index] = 0.0
    return (ce * (loss_weights / loss_weights.sum())).sum()


class BiMambaPreTrainedModel(PreTrainedModel):
    config_class = BiMambaConfig
    base_model_prefix = 'bimamba'
    supports_gradient_checkpointing = False
    _no_split_modules = ['BiMambaWrapper']

    def _init_weights(
        self,
        module,
        initializer_range=0.02,
        **kwargs,
    ):
        n_layer = self.config.n_layer
        initialized_cfg = (
            self.config.initializer_cfg
            if self.config.initializer_cfg is not None
            else {}
        )
        rescale_prenorm_residual = initialized_cfg.get('rescale_prenorm_residual', True)
        initializer_range = initialized_cfg.get('initializer_range', initializer_range)
        n_residuals_per_layer = initialized_cfg.get('n_residuals_per_layer', 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, '_no_reinit', False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Apply prenorm residual rescaling at most once per Parameter.
            # `self.apply(_initialize_weights)` walks every submodule and the
            # match `name in [...]` fires for every immediate parent of an
            # `out_proj`. With bidirectional weight tying the fwd and rev
            # Mamba2 modules both expose the *same* Parameter object, which
            # otherwise gets kaiming-init + rescaled twice.
            for name, p in module.named_parameters():
                if name in ['out_proj.weight', 'fc2.weight']:
                    if getattr(p, '_no_reinit', False):
                        continue
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
                    p._no_reinit = True


class BiMamba(BiMambaPreTrainedModel):
    def __init__(self, config: BiMambaConfig, device=None, dtype=None, **kwargs):
        super().__init__(config)
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        self.config = config
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.backbone = BiMambaMixerModel(config, **factory_kwargs, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        time_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        backbone_out = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            time_embeds=time_embeds,
        )
        hidden_states, all_hidden_states = backbone_out

        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
            )
        # Always return a tuple in the non-dict path so callers can use
        # `outputs[0]` / `outputs[1:]` uniformly. The second slot is the
        # hidden-states list (empty when not requested).
        return (
            hidden_states,
            all_hidden_states if output_hidden_states else None,
        )


class BiMambaForMaskedLM(BiMambaPreTrainedModel):
    def __init__(self, config: BiMambaConfig, device=None, dtype=None, **kwargs):
        super().__init__(config, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.bimamba = BiMamba(config, **factory_kwargs, **kwargs)
        self.config = config
        self.temb_strategy = config.temb_strategy
        lm_head_in_dim = config.d_model
        if (
            not config.tie_word_embeddings
            and config.temb_strategy == 'concat'
        ):
            lm_head_in_dim += config.d_temb
        self.lm_head = nn.Linear(
            lm_head_in_dim,
            self.config.vocab_size,
            bias=False,
            **factory_kwargs,
        )
        self.post_init()
        if self.config.tie_word_embeddings:
            self.tie_weights()

    def init_weights(self):
        self.apply(self._initialize_weights)

    def post_init(self):
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def get_input_embeddings(self):
        return self.bimamba.backbone.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bimamba.backbone.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        super().tie_weights()

    def get_decoder(self):
        return self.bimamba

    def set_decoder(self, decoder):
        self.bimamba = decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_weights: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        time_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bimamba(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            time_embeds=time_embeds,
        )
        hidden_states = outputs[0]
        if (
            self.config.tie_word_embeddings
            and time_embeds is not None
            and self.temb_strategy is not None
            and self.temb_strategy == 'concat'
        ):
            hidden_states = hidden_states[:, :, self.config.d_temb:]

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if loss_weights is not None:
                loss = weighted_cross_entropy(
                    logits, labels, loss_weights, ignore_index=self.config.pad_token_id
                )
            else:
                loss = cross_entropy(
                    logits, labels, ignore_index=self.config.pad_token_id
                )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


class DiMamba(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int, pad_token_id: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)

        # Normalize the 'none' string sentinel so the rest of the stack only
        # has to handle None.
        temb_strategy = config.model.temb_strategy
        if temb_strategy == 'none':
            temb_strategy = None
        self.temb_strategy = temb_strategy

        if self.temb_strategy == 'add':
            self.sigma_map = TimestepEmbedder(config.model.hidden_size)
        elif self.temb_strategy is not None:
            self.sigma_map = TimestepEmbedder(config.model.cond_dim)

        # Allow per-run ssm_cfg overrides from YAML; fall back to Mamba-2 defaults.
        ssm_cfg_raw = getattr(config.model, 'ssm_cfg', None)
        ssm_cfg = omegaconf.OmegaConf.to_object(ssm_cfg_raw) if ssm_cfg_raw is not None else None

        mamba_config = BiMambaConfig(
            d_model=config.model.hidden_size,
            n_layer=config.model.n_blocks,
            pad_token_id=pad_token_id,
            vocab_size=vocab_size,
            pad_vocab_size_multiple=1,
            tie_word_embeddings=config.model.tie_word_embeddings,
            temb_strategy=self.temb_strategy,
            d_temb=config.model.cond_dim,
            bidirectional=True,
            ssm_cfg=ssm_cfg,
            dropout=getattr(config.model, 'dropout', 0.1),
        )

        self.model = BiMambaForMaskedLM(config=mamba_config)

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, indices, sigma):
        c = None
        if self.temb_strategy is not None:
            c = F.silu(self.sigma_map(sigma))

        # Use the modern torch.amp.autocast API; torch.cuda.amp.autocast is
        # deprecated in PyTorch 2.x. A100 supports bf16 natively (tensor cores).
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x = self.model(indices, time_embeds=c).logits

        return x
