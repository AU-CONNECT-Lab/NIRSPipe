"""The cohort page's distance from the median and its rolling median, on inputs worked by hand."""

import numpy as np
import pandas as pd
import pytest

from nirspipe.qc.figures.subject.group_figures import (
    _robust_z, _smooth, detect_outliers, deviation_scores, metric_family,
)


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


# ---- the outlier score: one quantity counts once ----

def _cohort():
    # SCI written under three channel sets (one quantity), PSP once. Both columns have median
    # 0 and MAD 1, so |z| is the value over 1.4826: SCI 0, 0, 1, 1, 10 and PSP 0, 1, 1, 2, 2
    sci = [0.0, 0.0, 1.0, -1.0, 10.0]
    return pd.DataFrame({"bids_name": ["r1", "r2", "r3", "r4", "r5"],
                         "raw_sci_mean": sci, "raw_long_sci_mean": sci,
                         "raw_short_sci_mean": sci,
                         "raw_psp_mean": [0.0, 1.0, -1.0, 2.0, -2.0]})


def test_the_score_averages_quantities_not_columns():
    # (SCI + PSP) / 2 per run; counting columns would give r5 (3 * 10 + 2) / 4 = 8
    df = _cohort()
    got = deviation_scores(df, list(df.columns[1:]))
    assert got == pytest.approx(np.array([0.0, 0.5, 1.0, 1.5, 6.0]) / 1.4826)


def test_an_outlier_names_each_quantity_once():
    # r5 scores 6 / 1.4826 = 4.05, past 2; its PSP |z| of 1.35 is not past 2 and is not named
    df = _cohort()
    assert detect_outliers(df, list(df.columns[1:])) == {"r5": ["raw_sci_mean"]}


def test_a_count_and_its_share_are_one_quantity():
    assert (metric_family("motion_post_gvtd_num_above_thresh")
            == metric_family("raw_long_gvtd_pct_above_thresh"))
    assert metric_family("excluded_kept_s") == metric_family("excluded_total_s")
    assert metric_family("raw_sci_mean") != metric_family("raw_psp_mean")


# ---- the rolling median the grid draws ----

def test_the_rolling_median_drops_a_spike_narrower_than_its_window():
    assert _smooth(np.array([0.0, 0.0, 9.0, 0.0, 0.0]), 3).tolist() == [0.0] * 5


def test_the_edges_are_held_at_their_own_value_not_dropped():
    # padded with the end sample, the first window reads 5, 5, 0
    assert _smooth(np.array([5.0, 0.0, 0.0, 0.0, 0.0]), 3).tolist() == [5.0, 0.0, 0.0, 0.0, 0.0]
