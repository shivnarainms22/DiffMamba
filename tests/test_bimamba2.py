"""
Unit tests for the BiMamba-2 backbone.
Run on a GPU instance (requires mamba-ssm + CUDA):
    cd /path/to/DiffMamba && python -m pytest tests/test_bimamba2.py -v

Tests are CUDA-only because mamba-ssm's Mamba2 requires Triton kernels.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

# Hard skip the whole module if CUDA or mamba-ssm aren't available — keeps the
# suite at least importable on machines without GPU/Triton (e.g. CPU CI).
if not torch.cuda.is_available():
    pytest.skip('CUDA required for Mamba-2 kernels', allow_module_level=True)
pytest.importorskip('mamba_ssm')

from models.dimamba import (
    BiMambaConfig,
    BiMambaForMaskedLM,
    BiMambaWrapper,
    _MAMBA2_DEFAULTS,
)


VOCAB_SIZE = 50258  # GPT-2 vocab + [MASK]
MASK_ID = 50257
DEVICE = 'cuda'


def make_model(hidden_size=256, n_blocks=4, tie=False,
               temb_strategy=None, d_temb=0):
    """Small model for fast unit tests."""
    cfg = BiMambaConfig(
        d_model=hidden_size,
        n_layer=n_blocks,
        vocab_size=VOCAB_SIZE,
        pad_vocab_size_multiple=1,
        tie_word_embeddings=tie,
        temb_strategy=temb_strategy,
        d_temb=d_temb,
        bidirectional=True,
    )
    return BiMambaForMaskedLM(config=cfg).to(DEVICE).eval()


# ── Test 1: bidirectionality ────────────────────────────────────────────────

def test_bidirectional():
    """Perturbing a future token must change logits at past positions."""
    model = make_model()
    ids = torch.randint(0, VOCAB_SIZE - 1, (2, 64), device=DEVICE)

    with torch.no_grad():
        out1 = model(ids).logits

    ids2 = ids.clone()
    ids2[:, 50] = 0  # perturb token 50

    with torch.no_grad():
        out2 = model(ids2).logits

    # Token 20 is before the perturbation — a causal model would not change.
    # BiMamba must change it.
    assert not torch.allclose(out1[:, 20, :], out2[:, 20, :], atol=1e-5), \
        "Model appears to be causal — token 20 logits unchanged after perturbing token 50"


# ── Test 2: output shape ────────────────────────────────────────────────────

def test_output_shape():
    model = make_model()
    ids = torch.randint(0, VOCAB_SIZE - 1, (2, 64), device=DEVICE)
    with torch.no_grad():
        logits = model(ids).logits
    assert logits.shape == (2, 64, VOCAB_SIZE), \
        f"Expected (2, 64, {VOCAB_SIZE}), got {logits.shape}"


# ── Test 3: masked input ────────────────────────────────────────────────────

def test_masked_input():
    """[MASK] tokens must not cause NaN or crash."""
    model = make_model()
    ids = torch.randint(0, VOCAB_SIZE - 1, (2, 64), device=DEVICE)
    ids[:, 20:40] = MASK_ID

    with torch.no_grad():
        logits = model(ids).logits

    assert not torch.isnan(logits).any(), "NaN in output with masked input"
    assert not torch.isinf(logits).any(), "Inf in output with masked input"


# ── Test 4: weight tying ────────────────────────────────────────────────────

def test_weight_tying():
    """Fwd and rev Mamba-2 must share in_proj and out_proj weights."""
    cfg = BiMambaConfig(
        d_model=128,
        n_layer=2,
        vocab_size=VOCAB_SIZE,
        pad_vocab_size_multiple=1,
        tie_word_embeddings=False,
        temb_strategy=None,
        d_temb=0,
        bidirectional=True,
        bidirectional_weight_tie=True,
    )
    model = BiMambaForMaskedLM(config=cfg).to(DEVICE)
    wrapper = model.bimamba.backbone.layers[0].mixer

    assert wrapper.mamba_fwd.in_proj.weight.data_ptr() == \
           wrapper.mamba_rev.in_proj.weight.data_ptr(), \
        "in_proj weights not tied between fwd and rev"
    assert wrapper.mamba_fwd.out_proj.weight.data_ptr() == \
           wrapper.mamba_rev.out_proj.weight.data_ptr(), \
        "out_proj weights not tied between fwd and rev"


# ── Test 5: AdaLN path — finite output and bounded residual growth ─────────

def test_adaln_finite_and_bounded():
    """
    Regression guard for the AdaLN residual-double-count bug.
    With gate-zero init (zeroed adaLN_modulation), the gated update is 0 at
    init, so the residual stream should be exactly the embedding + final-norm.
    With a non-zero time embedding the magnitudes must stay finite and not
    blow up with depth.
    """
    d_temb = 64
    n_layer = 6
    model = make_model(hidden_size=128, n_blocks=n_layer,
                       temb_strategy='adaln', d_temb=d_temb)
    ids = torch.randint(0, VOCAB_SIZE - 1, (2, 64), device=DEVICE)
    time_embeds = torch.randn(2, d_temb, device=DEVICE)

    with torch.no_grad():
        out = model(ids, time_embeds=time_embeds, output_hidden_states=True)

    logits = out.logits
    assert torch.isfinite(logits).all(), \
        'AdaLN path produced non-finite logits — residual stream may be drifting'

    # At init, adaLN_modulation weights/biases are zero → gate = 0 → blocks
    # are residual-only. Hidden-state norms should NOT double per layer (the
    # symptom of the old double-count bug).
    hs = out.hidden_states  # list of (B, L, D)
    if hs is not None and len(hs) >= 2:
        norms = [h.float().norm().item() for h in hs]
        # Ratio between successive layers should be ~1 with gate=0; allow
        # generous slack but well under 2x (the bug regime).
        for a, b in zip(norms[:-1], norms[1:]):
            assert b / max(a, 1e-6) < 1.5, \
                f'Per-layer norm ratio {b/a:.3f} > 1.5 — suggests residual ' \
                f'double-count regression. Norms: {norms}'


# ── Test 6: dropout from config is respected ───────────────────────────────

def test_dropout_threaded_from_config():
    cfg = BiMambaConfig(
        d_model=128, n_layer=2, vocab_size=VOCAB_SIZE,
        pad_vocab_size_multiple=1, tie_word_embeddings=False,
        temb_strategy='adaln', d_temb=32, bidirectional=True,
        dropout=0.42,
    )
    model = BiMambaForMaskedLM(config=cfg).to(DEVICE)
    for block in model.bimamba.backbone.layers:
        assert block.dropout == 0.42, \
            f'Block dropout {block.dropout} did not pick up config.dropout=0.42'


# ── Test 7: weight rescale runs at most once per Parameter ─────────────────

def test_no_double_rescale_on_tied_weights():
    cfg = BiMambaConfig(
        d_model=128, n_layer=4, vocab_size=VOCAB_SIZE,
        pad_vocab_size_multiple=1, tie_word_embeddings=False,
        temb_strategy=None, d_temb=0, bidirectional=True,
        bidirectional_weight_tie=True,
    )
    model = BiMambaForMaskedLM(config=cfg).to(DEVICE)
    # Every out_proj.weight that we rescaled should carry the sentinel.
    seen = set()
    for layer in model.bimamba.backbone.layers:
        for direction in (layer.mixer.mamba_fwd, layer.mixer.mamba_rev):
            p = direction.out_proj.weight
            if id(p) in seen:
                continue
            seen.add(id(p))
            assert getattr(p, '_no_reinit', False), \
                'out_proj.weight missing _no_reinit sentinel after init'


# ── Param count report (not a test, just informational) ────────────────────

def report_param_counts():
    configs = [
        ("micro-dimamba (50M target)", 512, 8),
        ("base-dimamba (100M target)", 640, 10),
        ("small-dimamba (130M target)", 768, 12),
    ]
    print("\n── Parameter counts ──────────────────────────────────")
    for name, hidden, blocks in configs:
        cfg = BiMambaConfig(
            d_model=hidden,
            n_layer=blocks,
            vocab_size=VOCAB_SIZE,
            pad_vocab_size_multiple=1,
            tie_word_embeddings=False,
            temb_strategy='adaln',
            d_temb=128,
            bidirectional=True,
        )
        model = BiMambaForMaskedLM(config=cfg).to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        unique = sum(p.numel() for p in set(model.parameters()))
        print(f"  {name}: {total/1e6:.1f}M total, {unique/1e6:.1f}M unique (after weight tying)")


if __name__ == '__main__':
    print("Running BiMamba-2 unit tests...")
    test_bidirectional()
    test_output_shape()
    test_masked_input()
    test_weight_tying()
    test_adaln_finite_and_bounded()
    test_dropout_threaded_from_config()
    test_no_double_rescale_on_tied_weights()
    report_param_counts()
    print("\nAll tests passed.")
