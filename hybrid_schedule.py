"""Pure helpers for scheduling attention layers inside a hybrid backbone."""


def attention_layer_indices(n_layers: int, every: int,
                            offset: int = 0,
                            explicit: list[int] | None = None) -> list[int]:
    """Return zero-based layer indices that should use attention.

    `every=4, offset=3` gives layers 3, 7, 11 for a 12-block model.
    `every <= 0` disables attention insertion.

    If `explicit` is a non-empty list, it takes precedence over the
    every/offset schedule: its indices (deduped, sorted, and clamped to
    ``0 <= i < n_layers``) become the attention layers. An empty or ``None``
    `explicit` falls back to every/offset. This lets placement ablations pin
    attention at arbitrary layers (e.g. early `[0,1,2]` vs late `[9,10,11]`).
    """
    if n_layers <= 0:
        return []
    if explicit:  # non-None and non-empty -> overrides every/offset
        return sorted({int(i) for i in explicit if 0 <= int(i) < n_layers})
    if every <= 0:
        return []
    start = max(0, offset)
    return list(range(start, n_layers, every))


def is_attention_layer(layer_idx: int, n_layers: int, every: int,
                       offset: int = 0,
                       explicit: list[int] | None = None) -> bool:
    return layer_idx in set(
        attention_layer_indices(n_layers, every, offset, explicit))
