# Tolerate mamba_ssm being absent on CPU-only dev boxes (ModuleNotFoundError),
# but re-raise a *broken* install — e.g. an undefined-symbol ImportError from a
# torch/CUDA mismatch — so it surfaces loudly on GPU nodes instead of becoming a
# confusing downstream AttributeError on `models.dimamba`.
try:
    from . import dimamba
except ModuleNotFoundError:
    pass
try:
    from . import hybrid_dimamba
except ModuleNotFoundError:
    pass
try:
    from . import ema
except ModuleNotFoundError:
    pass
# dit + autoregressive depend on flash_attn, which isn't needed for the
# dimamba code path or its tests. Tolerate its absence so the package is
# still importable in environments where flash-attn isn't installed.
try:
    from . import dit
except ImportError:
    pass
try:
    from . import autoregressive
except ImportError:
    pass
