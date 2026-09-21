"""Per-condition numbers come from slicing one whole-run pass, not from cutting the run.

This is the primitive the per-condition views rest on, and it is pure array work: the
matrix and its window grid are already computed, so a condition is a column selection. The
alternative, cropping each condition and measuring it, filters every piece against its own
two edges and lands each on a grid starting at its own onset, so its windows are not the
run's windows. Nothing here can catch that; what it can catch is the selection rule going
wrong, which is what these pin.
"""

import numpy as np
import pytest

from fnirs_pipe.qc.metrics.windowed import condition_window_means

# 40 windows on a 10 s grid, centres 5, 15, ... 395
CENTERS = np.arange(5.0, 400.0, 10.0)
PAIRS = np.stack([CENTERS - 5, CENTERS + 5], axis=1)


def _ramp(n_ch: int = 3) -> np.ndarray:
    """Channel i holds i * 100 + window index, so a mean names the columns it came from."""
    return np.arange(n_ch)[:, None] * 100 + np.arange(len(CENTERS))[None, :]


# ---- which columns a condition keeps ----

def test_a_condition_keeps_the_windows_whose_centres_fall_inside_it():
    # 100 to 200 s holds the centres 105 .. 195, which are columns 10 to 19
    out = condition_window_means(_ramp(), CENTERS, [("mid", 100.0, 200.0)])
    assert out["mid"] == pytest.approx([14.5, 114.5, 214.5])


def test_a_window_straddling_an_edge_counts_for_one_side_only():
    # the centre rule is what makes this unambiguous: 100 to 150 keeps 105 .. 145 and the
    # window centred at 95 belongs to whatever covers 95, never to both
    first = condition_window_means(_ramp(1), CENTERS, [("a", 0.0, 100.0)])["a"]
    second = condition_window_means(_ramp(1), CENTERS, [("b", 100.0, 200.0)])["b"]
    assert first[0] == pytest.approx(4.5)    # columns 0..9
    assert second[0] == pytest.approx(14.5)  # columns 10..19


def test_several_conditions_are_sliced_out_of_the_one_matrix():
    out = condition_window_means(
        _ramp(1), CENTERS, [("a", 0.0, 100.0), ("b", 100.0, 200.0), ("c", 200.0, 300.0)])
    assert list(out) == ["a", "b", "c"]
    assert [v[0] for v in out.values()] == pytest.approx([4.5, 14.5, 24.5])


# ---- the shapes the callers actually hold ----

def test_the_mne_nirs_start_end_pairs_are_accepted_as_well_as_centres():
    # attach_windowed_series hands back the [start, end] pairs, and an (n, 2) array would
    # broadcast against the window bounds and mask nothing correctly
    by_pairs = condition_window_means(_ramp(), PAIRS, [("mid", 100.0, 200.0)])
    by_centers = condition_window_means(_ramp(), CENTERS, [("mid", 100.0, 200.0)])
    assert by_pairs["mid"] == pytest.approx(by_centers["mid"])


def test_a_single_series_reduces_to_one_scalar_per_condition():
    # GVTD is one series over time, not one per channel
    out = condition_window_means(np.arange(len(CENTERS), dtype=float), CENTERS,
                                 [("mid", 100.0, 200.0)])
    assert out["mid"].shape == ()
    assert float(out["mid"]) == pytest.approx(14.5)


def test_a_per_channel_matrix_keeps_one_value_per_channel():
    out = condition_window_means(_ramp(5), CENTERS, [("mid", 100.0, 200.0)])
    assert out["mid"].shape == (5,)


# ---- conditions with nothing in them ----

def test_a_condition_holding_no_whole_window_is_left_out():
    # an annotation shorter than the grid spacing, so no centre falls inside it
    out = condition_window_means(_ramp(), CENTERS, [("blink", 20.0, 24.0),
                                                    ("mid", 100.0, 200.0)])
    assert list(out) == ["mid"]


def test_no_conditions_gives_nothing_rather_than_the_whole_run():
    # an empty window list must not fall through to "no scope keeps every window", which is
    # what _in_scope does on its own and would silently label the run as a condition
    assert condition_window_means(_ramp(), CENTERS, []) == {}
