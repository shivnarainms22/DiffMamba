try:
    from . import dimamba
except (ImportError, ModuleNotFoundError):
    pass
try:
    from . import ema
except (ImportError, ModuleNotFoundError):
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
