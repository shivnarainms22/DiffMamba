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


def test_explicit_layers_override_every_offset():
    # An explicit layer list takes precedence over the every/offset schedule.
    assert attention_layer_indices(
        n_layers=12, every=4, offset=3, explicit=[0, 1, 2]) == [0, 1, 2]


def test_explicit_layers_sorted_deduped_and_bounds_filtered():
    # Out-of-range and duplicate indices are dropped; result is sorted.
    assert attention_layer_indices(
        n_layers=12, every=4, offset=3, explicit=[11, 5, 5, 99, -1]) == [5, 11]


def test_empty_explicit_falls_back_to_every_offset():
    assert attention_layer_indices(
        n_layers=12, every=4, offset=3, explicit=[]) == [3, 7, 11]


def test_none_explicit_falls_back_to_every_offset():
    assert attention_layer_indices(
        n_layers=12, every=4, offset=3, explicit=None) == [3, 7, 11]


def test_is_attention_layer_honors_explicit():
    assert is_attention_layer(
        9, n_layers=12, every=4, offset=3, explicit=[9, 10, 11])
    assert not is_attention_layer(
        3, n_layers=12, every=4, offset=3, explicit=[9, 10, 11])


# --- ablation grid: lock the exact attention pattern of each Phase-1 config ---

def test_ablation_arm_a_counts():
    # Arm A varies HOW MANY attention layers (evenly spaced).
    assert attention_layer_indices(12, every=3, offset=2) == [2, 5, 8, 11]   # hyb_e3  (4)
    assert attention_layer_indices(12, every=4, offset=3) == [3, 7, 11]      # A0      (3, current)
    assert attention_layer_indices(12, every=6, offset=5) == [5, 11]         # hyb_e6  (2)
    assert attention_layer_indices(12, every=12, offset=11) == [11]          # hyb_e12 (1)


def test_ablation_arm_b_placements_three_attn_layers():
    # Arm B fixes the COUNT at 3 and varies WHERE (vs A0's distributed [3,7,11]).
    assert attention_layer_indices(12, every=4, offset=3, explicit=[0, 1, 2]) == [0, 1, 2]      # hyb_early
    assert attention_layer_indices(12, every=4, offset=3, explicit=[4, 5, 6]) == [4, 5, 6]      # hyb_mid
    assert attention_layer_indices(12, every=4, offset=3, explicit=[9, 10, 11]) == [9, 10, 11]  # hyb_late
