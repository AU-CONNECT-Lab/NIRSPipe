"""The circular phase statistic on a WTC band: the angle, its spread, and what masks it."""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.synchrony import (
    WTCResult,
    _circular_stats,
    roi_mean_of_channels,
    wtc_band_mean,
    wtc_phase_by_scale,
)

FREQS = np.array([0.05, 0.10, 0.20])
TIMES = np.arange(4.0)
ONES  = np.ones((3, 4))
WIDE  = np.full(4, 1e6)      # a cone that admits every cell


def _result(phase, wtc=None, coi=None, sig=None, label="S1_D1") -> WTCResult:
    data = {"wtc": ONES if wtc is None else wtc,
            "coi": WIDE if coi is None else coi,
            "phase": phase}
    if sig is not None:
        data["sig"] = sig
    return WTCResult(pairs={("sub-A", "sub-B"): {label: data}},
                     freqs=FREQS, times=TIMES)


# ---- the circular mean itself ----

def test_the_mean_wraps_instead_of_averaging_across_the_cut():
    # the whole reason an angle cannot go through `mean`: these two straddle 0 deg
    angle, _, n = _circular_stats(np.radians([10.0, 350.0]))
    assert np.degrees(angle) == pytest.approx(0.0, abs=1e-9)
    assert n == 2


def test_spread_is_zero_when_the_angles_agree_and_large_when_they_do_not():
    _, tight, _ = _circular_stats(np.radians([30.0, 30.0, 30.0]))
    _, spread, _ = _circular_stats(np.linspace(-np.pi, np.pi, 360, endpoint=False))
    assert tight == pytest.approx(0.0, abs=1e-9)
    assert spread > 3.0


def test_no_finite_angle_gives_nan_at_zero_count():
    angle, spread, n = _circular_stats(np.array([np.nan, np.nan]))
    assert np.isnan(angle) and np.isnan(spread) and n == 0


# ---- the band column ----

def test_a_constant_lead_comes_back_as_its_angle_with_the_sign_kept():
    # positive means the first member leads, the convention the phase arrows are drawn with
    row = wtc_band_mean(_result(np.full((3, 4), np.pi / 4)), 0.04, 0.25).iloc[0]
    assert row["phase_angle"] == pytest.approx(45.0)
    assert row["phase_sd"] == pytest.approx(0.0, abs=1e-6)
    assert row["phase_n"] == 12
    lagging = wtc_band_mean(_result(np.full((3, 4), -np.pi / 4)), 0.04, 0.25).iloc[0]
    assert lagging["phase_angle"] == pytest.approx(-45.0)


def test_cells_outside_the_cone_do_not_enter_the_angle():
    # a cone of 5 s admits 0.2 Hz only; the 0.2 Hz row holds 0 deg, the masked rows 180 deg
    phase = np.array([[np.pi] * 4, [np.pi] * 4, [0.0] * 4])
    masked = wtc_band_mean(_result(phase, coi=np.full(4, 5.0)), 0.04, 0.25).iloc[0]
    assert masked["phase_angle"] == pytest.approx(0.0, abs=1e-9)
    assert masked["phase_n"] == 4
    unmasked = wtc_band_mean(_result(phase, coi=np.full(4, 5.0)), 0.04, 0.25,
                             mask_coi=False).iloc[0]
    assert abs(unmasked["phase_angle"]) == pytest.approx(180.0)
    assert unmasked["phase_n"] == 12


def test_cells_below_the_monte_carlo_level_do_not_enter_the_angle():
    # the level is per frequency: only the 0.2 Hz row clears it, and it is the row at 0 deg
    phase = np.array([[np.pi] * 4, [np.pi] * 4, [0.0] * 4])
    wtc   = np.array([[0.2] * 4, [0.2] * 4, [0.9] * 4])
    row = wtc_band_mean(_result(phase, wtc=wtc, sig=np.array([0.5, 0.5, 0.5])),
                        0.04, 0.25).iloc[0]
    assert row["phase_angle"] == pytest.approx(0.0, abs=1e-9)
    assert row["phase_n"] == 4


def test_a_result_carrying_no_phase_keeps_its_row_at_zero_count():
    bare = WTCResult(pairs={("sub-A", "sub-B"): {"S1_D1": {"wtc": ONES, "coi": WIDE}}},
                     freqs=FREQS, times=TIMES)
    row = wtc_band_mean(bare, 0.04, 0.25).iloc[0]
    assert np.isnan(row["phase_angle"]) and row["phase_n"] == 0


def test_a_pairing_that_failed_keeps_its_row():
    blank = WTCResult(pairs={("sub-A", "sub-B"): {"S1_D1": None}},
                      freqs=FREQS, times=TIMES)
    row = blank and wtc_band_mean(blank, 0.04, 0.25).iloc[0]
    assert np.isnan(row["phase_angle"]) and row["phase_n"] == 0


# ---- per scale, where the angle becomes a delay ----

def test_the_angle_becomes_a_lag_in_seconds_at_its_own_frequency():
    # +45 deg at 0.10 Hz is an eighth of a 10 s period: 1.25 s, the first member ahead
    out = wtc_phase_by_scale(_result(np.full((3, 4), np.pi / 4)), 0.04, 0.25)
    at = lambda f: out[np.isclose(out["freq"], f)].iloc[0]
    assert at(0.10)["lag_s"] == pytest.approx(1.25)
    # the same angle is a different delay at a different scale, which is why the band mean
    # cannot be converted
    assert at(0.20)["lag_s"] == pytest.approx(0.625)


def test_every_band_frequency_gets_a_row_and_a_failed_pairing_gets_none():
    assert len(wtc_phase_by_scale(_result(np.zeros((3, 4))), 0.04, 0.25)) == 3
    blank = WTCResult(pairs={("sub-A", "sub-B"): {"S1_D1": None}},
                      freqs=FREQS, times=TIMES)
    assert wtc_phase_by_scale(blank, 0.04, 0.25).empty


# ---- the ROI average ----

def _band_frame(angles) -> pd.DataFrame:
    return pd.DataFrame({
        "sub1": "sub-A", "sub2": "sub-B", "label": ["S1_D1", "S1_D2"],
        "coherence": [0.5, 0.5], "coherence_z": [0.55, 0.55], "n_valid_frac": [1.0, 1.0],
        "phase_angle": angles, "phase_sd": [1.0, 1.0], "phase_n": [10, 20],
    })


def test_the_roi_average_wraps_too():
    out = roi_mean_of_channels(_band_frame([10.0, -10.0]),
                               {"pfc": ["S1_D1", "S1_D2"]}).iloc[0]
    assert out["phase_angle"] == pytest.approx(0.0, abs=1e-9)
    assert out["phase_n"] == 30       # a cell count, so it is summed rather than averaged


def test_a_frame_with_no_phase_columns_still_averages():
    # callers that built a band frame by hand predate these columns
    frame = _band_frame([0.0, 0.0]).drop(columns=["phase_angle", "phase_sd", "phase_n"])
    out = roi_mean_of_channels(frame, {"pfc": ["S1_D1", "S1_D2"]})
    assert "phase_angle" not in out.columns
    assert out.iloc[0]["coherence"] == pytest.approx(0.5)
