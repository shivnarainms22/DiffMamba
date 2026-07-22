"""Hybrid Mamba+attention denoiser for masked diffusion LMs.

This is an opt-in backbone: most layers are bidirectional Mamba-2 residual
blocks, with full bidirectional attention inserted on a configurable schedule.
It is meant to test whether sparse attention recovers DiT quality while keeping
more of Mamba's long-context efficiency than an all-attention denoiser.
"""

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybrid_schedule import attention_layer_indices, is_attention_layer
from models.dimamba import BiMambaWrapper, _MAMBA2_DEFAULTS
from models.dit import (
    DDiTBlock,
    DDitFinalLayer,
    EmbeddingLayer,
    Rotary,
    TimestepEmbedder,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
    modulate_fused,
)


class HybridMambaBlock(nn.Module):
    """Standard residual block around the existing bidirectional Mamba mixer."""

    def __init__(self, dim, cond_dim, ssm_cfg=None, dropout=0.1):
        super().__init__()
        if ssm_cfg is None:
            ssm_cfg = {}
        ssm_cfg = {**_MAMBA2_DEFAULTS, **ssm_cfg}
        self.norm = nn.LayerNorm(dim)
        self.mixer = BiMambaWrapper(
            d_model=dim,
            bidirectional=True,
            bidirectional_strategy='add',
            bidirectional_weight_tie=True,
            **ssm_cfg)
        self.dropout = dropout
        self.adaLN_modulation = nn.Linear(cond_dim, 3 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        return (bias_dropout_add_scale_fused_train if self.training
                else bias_dropout_add_scale_fused_inference)

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c)[:, None].chunk(3, dim=2)
        h = modulate_fused(self.norm(x), shift, scale)
        h = self.mixer(h, inference_params=None)
        return self._get_bias_dropout_scale()(h, None, gate, x, self.dropout)


class HybridDiMamba(nn.Module):
    """A DiT-compatible denoiser API backed by sparse attention + BiMamba."""

    def __init__(self, config, vocab_size: int, pad_token_id: int = None):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size

        m = config.model
        hidden = m.hidden_size
        cond_dim = m.cond_dim
        n_layers = m.n_blocks
        every = int(m.get('hybrid_attention_every', 4))
        offset = int(m.get('hybrid_attention_offset', every - 1))
        explicit_raw = m.get('hybrid_attention_layers', None)
        explicit = ([int(x) for x in explicit_raw]
                    if explicit_raw is not None else None)
        dropout = float(m.get('dropout', 0.1))

        ssm_cfg_raw = getattr(m, 'ssm_cfg', None)
        ssm_cfg = (omegaconf.OmegaConf.to_object(ssm_cfg_raw)
                   if ssm_cfg_raw is not None else None)

        self.vocab_embed = EmbeddingLayer(hidden, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(hidden // m.n_heads)

        self.blocks = nn.ModuleList()
        self.layer_types = []
        for i in range(n_layers):
            if is_attention_layer(i, n_layers, every, offset, explicit=explicit):
                self.blocks.append(DDiTBlock(
                    hidden, m.n_heads, cond_dim, dropout=dropout))
                self.layer_types.append('attention')
            else:
                self.blocks.append(HybridMambaBlock(
                    hidden, cond_dim, ssm_cfg=ssm_cfg, dropout=dropout))
                self.layer_types.append('mamba')

        # Log the resolved schedule once at construction so any run can be
        # grep-confirmed to train the INTENDED architecture (a wrong schedule
        # silently trained for 76k steps is the costly failure to avoid).
        attn_idx = attention_layer_indices(n_layers, every, offset,
                                           explicit=explicit)
        print(f'[HybridDiMamba] n_blocks={n_layers} '
              f'attention_layers={attn_idx} ({len(attn_idx)} attn / '
              f'{n_layers - len(attn_idx)} mamba)', flush=True)

        self.output_layer = DDitFinalLayer(hidden, vocab_size, cond_dim)
        self.scale_by_sigma = m.scale_by_sigma
        self.pad_token_id = pad_token_id

    def forward(self, indices, sigma):
        x = self.vocab_embed(indices)
        c = F.silu(self.sigma_map(sigma))
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for kind, block in zip(self.layer_types, self.blocks):
                if kind == 'attention':
                    x = block(x, rotary_cos_sin, c, seqlens=None)
                else:
                    x = block(x, c)
            x = self.output_layer(x, c)
        return x
