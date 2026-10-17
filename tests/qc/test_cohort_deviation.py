"""The cohort page's distance from the median and its rolling median, on inputs worked by hand."""

import numpy as np
import pytest

from nirspipe.qc.figures.subject.group_figures import _robust_z, _smooth


# ---- the distance a strip point and a highlight are drawn at ----

def test_the_distance_is_in_robust_sds_from_the_median():
    # median 0.94 and MAD 0.04, so a step of 0.04 is 1 / 1.4826 robust SDs; the NaN stays out
    z = _robust_z(np.array([0.90, np.nan, 0.94, 0.98]))
    assert z == pytest.approx([-1 / 1.4826, np.nan, 0.0, 1 / 1.4826], nan_ok=True)


def test_a_zero_mad_falls_back_to_the_sd():
    # the MAD of 1, 1, 1, 5 is 0 and its population SD sqrt(3)
    assert _robust_z(np.array([1.0, 1.0, 1.0, 5.0])) == pytest.approx(
        [0.0, 0.0, 0.0, 4 / np.sqrt(3)])


def test_a_column_with_no_spread_has_no_distance():
    assert np.isnan(_robust_z(np.array([28.0, 28.0, 28.0]))).all()


# ---- the rolling median the grid draws ----

def test_the_rolling_median_drops_a_spike_narrower_than_its_window():
    assert _smooth(np.array([0.0, 0.0, 9.0, 0.0, 0.0]), 3).tolist() == [0.0] * 5


def test_the_edges_are_held_at_their_own_value_not_dropped():
    # padded with the end sample, the first window reads 5, 5, 0
    assert _smooth(np.array([5.0, 0.0, 0.0, 0.0, 0.0]), 3).tolist() == [5.0, 0.0, 0.0, 0.0, 0.0]
