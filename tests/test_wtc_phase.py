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

    def at(f):
        return out[np.isclose(out["freq"], f)].iloc[0]

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


# ---- end to end, against a lag that is known because it was put there ----
# The unit tests above feed the statistic a phase map. These build two recordings one of
# which really is a fixed time ahead of the other, run the real transform over them, and
# ask for the delay back. A sign error, a conjugate the wrong way round, or a frequency axis
# off by a row all survive the tests above and none of them survives these.

SFREQ    = 5.0
DURATION = 400.0
SIG_FREQ = 0.05           # period 20 s
LAG_S    = 2.5            # an eighth of that period: +45 deg at SIG_FREQ
E2E_BAND = (0.03, 0.10)
E2E_LABELS = ["S1_D1", "S2_D2", "S3_D3"]


def _raw_carrying(signal: np.ndarray, seed: int) -> "object":
    """A haemoglobin recording whose HbO channels all carry ``signal`` plus their own noise."""
    import mne
    rng = np.random.default_rng(seed)
    names = [f"{label} {c}" for label in E2E_LABELS for c in ("hbo", "hbr")]
    types = [c for _ in E2E_LABELS for c in ("hbo", "hbr")]
    info = mne.create_info(names, SFREQ, types)
    for i, ch in enumerate(info["chs"]):
        loc = np.zeros(12)
        loc[3:6] = [i * 0.05, 0.0, 0.0]
        loc[6:9] = [i * 0.05 + 0.03, 0.0, 0.0]       # 30 mm, so nothing reads as short
        loc[:3] = (loc[3:6] + loc[6:9]) / 2
        ch["loc"] = loc
    data = np.empty((len(names), signal.size))
    for i, ch_type in enumerate(types):
        noise = rng.standard_normal(signal.size)
        data[i] = 1e-6 * (signal + 0.2 * noise if ch_type == "hbo" else noise)
    return mne.io.RawArray(data, info, verbose="ERROR")


def _dyad_lagged(lag_s: float) -> dict:
    """Two recordings of one rhythm, the second ``lag_s`` behind the first."""
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    lead   = np.sin(2 * np.pi * SIG_FREQ * t)
    follow = np.sin(2 * np.pi * SIG_FREQ * (t - lag_s))
    return {"sub-01": _raw_carrying(lead, 1), "sub-02": _raw_carrying(follow, 2)}


def _at_signal_frequency(raws) -> pd.Series:
    """The per-scale row nearest the rhythm that was put in, averaged over the label pairs."""
    from fnirs_pipe.pipeline.synchrony import compute_wtc
    result = compute_wtc(raws, fmin=0.02, fmax=0.2, ch_type="hbo")
    by = wtc_phase_by_scale(result, *E2E_BAND)
    freq = by["freq"].unique()[np.argmin(np.abs(by["freq"].unique() - SIG_FREQ))]
    rows = by[by["freq"] == freq]
    angle, _, _ = _circular_stats(np.radians(rows["phase_angle"].to_numpy()))
    return pd.Series({"freq": freq, "angle_deg": np.degrees(angle),
                      "lag_s": rows["lag_s"].mean(), "n": len(rows)})


@pytest.fixture(scope="module")
def ahead():
    return _at_signal_frequency(_dyad_lagged(LAG_S))


def test_the_delay_that_was_put_in_comes_back_in_seconds(ahead):
    # the number a reader acts on: sub-01 leads sub-02 by LAG_S, and lag_s says so
    assert ahead["lag_s"] == pytest.approx(LAG_S, rel=0.05)
    assert ahead["n"] == len(E2E_LABELS)


def test_the_angle_is_the_delay_times_the_frequency_of_its_own_row(ahead):
    # pinned against theory rather than against a recorded number: a phase of 2 pi f tau,
    # evaluated at the row's own frequency, not at the nominal one
    expected = np.degrees(2 * np.pi * ahead["freq"] * LAG_S)
    assert ahead["angle_deg"] == pytest.approx(expected, abs=3.0)


def test_reversing_who_leads_reverses_the_sign():
    # the half of the measurement coherence cannot give: the same coupling, the other way
    behind = _at_signal_frequency(_dyad_lagged(-LAG_S))
    assert behind["lag_s"] == pytest.approx(-LAG_S, rel=0.05)


def test_two_members_in_step_report_no_lead():
    together = _at_signal_frequency(_dyad_lagged(0.0))
    assert together["angle_deg"] == pytest.approx(0.0, abs=2.0)
    assert together["lag_s"] == pytest.approx(0.0, abs=0.2)


def test_the_spread_is_what_says_the_angle_is_unreadable():
    """Independent members still produce an angle; only ``phase_sd`` distinguishes it."""
    from fnirs_pipe.pipeline.synchrony import compute_wtc
    t = np.arange(int(SFREQ * DURATION)) / SFREQ
    rng = np.random.default_rng(7)
    apart = {"sub-01": _raw_carrying(rng.standard_normal(t.size), 11),
             "sub-02": _raw_carrying(rng.standard_normal(t.size), 12)}
    band = wtc_band_mean(compute_wtc(apart, fmin=0.02, fmax=0.2, ch_type="hbo"), *E2E_BAND)
    locked = wtc_band_mean(compute_wtc(_dyad_lagged(LAG_S), fmin=0.02, fmax=0.2,
                                       ch_type="hbo"), *E2E_BAND)
    assert band["phase_sd"].mean() > locked["phase_sd"].mean()
    assert band["phase_angle"].notna().all()      # an angle is still reported, hence the test


def test_the_lag_is_only_right_where_the_spread_says_it_is():
    """``phase_sd`` is the gate, and this is the measurement that says it works."""
    from fnirs_pipe.pipeline.synchrony import compute_wtc
    by = wtc_phase_by_scale(
        compute_wtc(_dyad_lagged(LAG_S), fmin=0.02, fmax=0.2, ch_type="hbo"), *E2E_BAND)
    per_freq = by.groupby("freq").agg(lag_s=("lag_s", "mean"),
                                      sd=("phase_sd", "mean")).reset_index()
    # the rhythm is a single tone, so away from it there is nothing shared to be late by
    # and the reported lag runs off. Every row whose angle is tight still has it right.
    tight = per_freq[per_freq["sd"] < 2.0]
    assert len(tight) >= 4
    assert tight["lag_s"].to_numpy() == pytest.approx(LAG_S, rel=0.20)
    # and the row that gets it most wrong is not a quiet failure: it is flagged by a
    # spread an order of magnitude wider than the best row's
    worst = per_freq.iloc[int(np.argmax(np.abs(per_freq["lag_s"] - LAG_S)))]
    assert worst["sd"] > 10 * per_freq["sd"].min()
