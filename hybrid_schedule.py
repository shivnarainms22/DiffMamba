"""Pure helpers for scheduling attention layers inside a hybrid backbone."""


def attention_layer_indices(n_layers: int, every: int,
                            offset: int = 0) -> list[int]:
    """Return zero-based layer indices that should use attention.

    `every=4, offset=3` gives layers 3, 7, 11 for a 12-block model.
    `every <= 0` disables attention insertion.
    """
    if every <= 0:
        return []
    if n_layers <= 0:
        return []
    start = max(0, offset)
    return list(range(start, n_layers, every))


def is_attention_layer(layer_idx: int, n_layers: int, every: int,
                       offset: int = 0) -> bool:
    return layer_idx in set(attention_layer_indices(n_layers, every, offset))
