import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_schedule import attention_layer_indices, is_attention_layer


def test_attention_layer_indices_every_four_upper_layers():
    assert attention_layer_indices(n_layers=12, every=4, offset=3) == [3, 7, 11]


def test_attention_layer_indices_disabled_when_every_zero():
    assert attention_layer_indices(n_layers=12, every=0, offset=3) == []


def test_is_attention_layer_uses_schedule():
    assert is_attention_layer(7, n_layers=12, every=4, offset=3)
    assert not is_attention_layer(8, n_layers=12, every=4, offset=3)
