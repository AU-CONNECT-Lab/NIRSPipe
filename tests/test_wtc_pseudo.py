"""The phase-scrambled null, and the two band-mean changes that came with it.

`phase_scramble` has to preserve the magnitude spectrum exactly and return a real series;
those are the two properties that make the surrogate a fair null rather than a different
signal. What it must *not* do is decorrelate: a surrogate of a signal whose power sits in a
few low-frequency bins keeps that rhythm and only shifts its phase, which is the point. The
null asks whether two people's timing is related given both have these rhythms, not whether
one of them has a rhythm at all.

The rest covers `wtc_band_mean` now defaulting to no COI mask while still reporting the share
inside the cone, the Fisher z column, and the ROI grouping that replaced the ROI-signal route.
"""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.synchrony import (
    WTCResult,
    _fisher_z,
    phase_scramble,
    roi_maps_from_channels,
    roi_mean_of_channels,
    wtc_band_mean,
)

FREQS = np.linspace(0.02, 0.30, 24)
TIMES = np.arange(30.0)


def _map(value, coi=1e6):
    """One synthetic WTC map, flat at `value`; a huge coi keeps every cell inside it."""
    return {"wtc": np.full((len(FREQS), len(TIMES)), float(value)),
            "coi": np.full(len(TIMES), float(coi)), "sig": None}


def _result(labels: dict) -> WTCResult:
    return WTCResult(pairs={("s1", "s2"): labels}, freqs=FREQS, times=TIMES)


# ---- phase_scramble ----

@pytest.mark.parametrize("n", [512, 511])
def test_the_surrogate_is_real_and_keeps_its_length(n):
    rng = np.random.default_rng(0)
    out = phase_scramble(np.cumsum(rng.normal(size=n)), rng)
    assert np.isrealobj(out)
    assert len(out) == n


@pytest.mark.parametrize("n", [512, 511])
def test_the_magnitude_spectrum_survives_scrambling(n):
    rng = np.random.default_rng(1)
    sig = np.cumsum(rng.normal(size=n))
    out = phase_scramble(sig, rng)
    before = np.abs(np.fft.rfft(sig))
    after = np.abs(np.fft.rfft(out))
    assert np.allclose(before, after, atol=1e-8)


@pytest.mark.parametrize("offset", [100.0, -100.0])
def test_the_mean_is_not_shifted(offset):
    """A negative offset is the case that catches taking the magnitude of the DC bin."""
    rng = np.random.default_rng(2)
    sig = np.cumsum(rng.normal(size=256)) + offset
    assert phase_scramble(sig, rng).mean() == pytest.approx(sig.mean(), abs=1e-8)


def test_two_surrogates_of_one_signal_differ():
    rng = np.random.default_rng(3)
    sig = np.cumsum(rng.normal(size=256))
    assert not np.allclose(phase_scramble(sig, rng), phase_scramble(sig, rng))


def test_white_noise_decorrelates_but_a_dominant_rhythm_does_not():
    """Not a defect: preserving the spectrum means preserving the rhythm, phase aside."""
    rng = np.random.default_rng(4)
    white = rng.normal(size=1024)
    walk = np.cumsum(rng.normal(size=1024))
    r_white = np.mean([abs(np.corrcoef(white, phase_scramble(white, rng))[0, 1])
                       for _ in range(50)])
    r_walk = np.mean([abs(np.corrcoef(walk, phase_scramble(walk, rng))[0, 1])
                      for _ in range(50)])
    assert r_white < 0.2
    assert r_walk > r_white


# ---- wtc_band_mean ----

def _half_outside_the_cone():
    """A map whose second half is padding: 0.5 inside the cone, 1.0 outside it."""
    data = _map(0.5)
    data["coi"] = np.where(np.arange(len(TIMES)) < 15, 1e6, 1e-12)
    data["wtc"][:, 15:] = 1.0
    return data


def test_the_cone_is_masked_by_default():
    """The padded half is dropped without being asked for. The default flipped on
    2026-09-10; this used to assert the other way and was not updated with it."""
    df = wtc_band_mean(_result({"A": _half_outside_the_cone()}), 0.06, 0.15)
    assert df["coherence"].iloc[0] == pytest.approx(0.5)


def test_the_mask_can_still_be_turned_off():
    """Off, the padded cells average in and pull the number up by half their distance."""
    df = wtc_band_mean(_result({"A": _half_outside_the_cone()}), 0.06, 0.15, mask_coi=False)
    assert df["coherence"].iloc[0] == pytest.approx(0.75)


def test_the_cone_share_is_reported_even_when_it_is_not_applied():
    """The diagnostic has to survive the default, or turning the mask off hides the reason."""
    coi = np.where(np.arange(len(TIMES)) < 15, 1e6, 1e-12)
    data = _map(0.5)
    data["coi"] = coi
    df = wtc_band_mean(_result({"A": data}), 0.06, 0.15)
    assert df["n_valid_frac"].iloc[0] == pytest.approx(0.5)


def test_the_band_mean_carries_its_fisher_z():
    df = wtc_band_mean(_result({"A": _map(0.6)}), 0.06, 0.15)
    assert df["coherence_z"].iloc[0] == pytest.approx(np.arctanh(0.6))


def test_a_failed_pair_keeps_its_row_with_both_values_missing():
    df = wtc_band_mean(_result({"A": None}), 0.06, 0.15)
    assert len(df) == 1
    assert np.isnan(df["coherence"].iloc[0])
    assert np.isnan(df["coherence_z"].iloc[0])


def test_fisher_z_of_one_is_finite():
    """Clipping, not arctanh(1), so a perfectly locked pair does not become inf."""
    assert np.isfinite(_fisher_z(1.0))
    assert np.isnan(_fisher_z(float("nan")))


# ---- ROI grouping ----

def test_an_roi_under_the_channel_minimum_is_dropped():
    df = wtc_band_mean(_result({"A1": _map(0.4), "A2": _map(0.6), "B1": _map(0.2)}),
                       0.06, 0.15)
    roi_map = {"roiA": ["A1", "A2"], "roiB": ["B1"]}
    kept = roi_mean_of_channels(df, roi_map, min_channels=2)
    assert list(kept["label"]) == ["roiA"]
    assert list(roi_mean_of_channels(df, roi_map, min_channels=1)["label"]) == ["roiA", "roiB"]


def test_the_roi_fisher_z_matches_the_roi_coherence_not_the_channel_average():
    """arctanh is nonlinear, so averaging z per channel would not equal z of the average."""
    df = wtc_band_mean(_result({"A1": _map(0.2), "A2": _map(0.8)}), 0.06, 0.15)
    out = roi_mean_of_channels(df, {"roiA": ["A1", "A2"]}, min_channels=2)
    assert out["coherence"].iloc[0] == pytest.approx(0.5)
    assert out["coherence_z"].iloc[0] == pytest.approx(np.arctanh(0.5))


def test_the_roi_map_average_gives_the_number_the_roi_table_reports():
    """The figure and the table are the same average taken in the other order."""
    result = _result({"A1": _map(0.4), "A2": _map(0.6)})
    roi_map = {"roiA": ["A1", "A2"]}
    from_table = roi_mean_of_channels(
        wtc_band_mean(result, 0.06, 0.15), roi_map, min_channels=2)
    from_maps = wtc_band_mean(roi_maps_from_channels(result, roi_map), 0.06, 0.15)
    assert from_maps["coherence"].iloc[0] == pytest.approx(from_table["coherence"].iloc[0])


def test_a_channel_in_no_roi_reaches_neither_the_maps_nor_the_table():
    result = _result({"A1": _map(0.4), "A2": _map(0.6), "Z9": _map(0.9)})
    roi_map = {"roiA": ["A1", "A2"]}
    assert set(roi_maps_from_channels(result, roi_map).pairs[("s1", "s2")]) == {"roiA"}
    table = roi_mean_of_channels(wtc_band_mean(result, 0.06, 0.15), roi_map, min_channels=2)
    assert table["n_ch"].iloc[0] == 2


def test_crossed_channels_group_into_an_roi_by_roi_matrix():
    result = _result({("A1", "A1"): _map(0.4), ("A1", "B1"): _map(0.2),
                      ("B1", "A1"): _map(0.3), ("B1", "B1"): _map(0.5)})
    roi_map = {"roiA": ["A1"], "roiB": ["B1"]}
    table = roi_mean_of_channels(wtc_band_mean(result, 0.06, 0.15), roi_map, min_channels=1)
    assert set(zip(table["label"], table["label2"])) == {
        ("roiA", "roiA"), ("roiA", "roiB"), ("roiB", "roiA"), ("roiB", "roiB")}
    maps = roi_maps_from_channels(result, roi_map)
    assert ("roiA", "roiB") in maps.pairs[("s1", "s2")]


def test_a_group_of_three_is_refused_rather_than_half_scrambled():
    """Only one subject is scrambled, so a third member would leave real pairs in the null."""
    from fnirs_pipe.pipeline.synchrony import compute_wtc_pseudo

    with pytest.raises(ValueError, match="exactly 2 subjects"):
        compute_wtc_pseudo({"s1": None, "s2": None, "s3": None}, 0.06, 0.15, n_iter=1)


# ---- the null follows the windows the real table was read at ----
#
# A whole-run null against a per-condition real table is anticonservative on the short
# conditions: a long record's surrogate coherence is lower than a short window's, so the
# real value looks further above chance than it is. These pin that the null is windowed off
# the same transform rather than recomputed on the cut, which is the same decision
# `window_result` records for the real side.

def _ramp_map(first_half, second_half):
    """A map that is `first_half` over the first half of TIMES and `second_half` over the rest.

    Flat in frequency, so a band mean over any window is just the value that window holds.
    """
    wtc = np.empty((len(FREQS), len(TIMES)))
    mid = len(TIMES) // 2
    wtc[:, :mid] = float(first_half)
    wtc[:, mid:] = float(second_half)
    return {"wtc": wtc, "coi": np.full(len(TIMES), 1e6),
            "phase": np.zeros_like(wtc), "sig": None}


@pytest.fixture
def stub_pseudo(monkeypatch):
    """compute_wtc_pseudo with the transform and the channel picking replaced.

    Neither is what these tests are about, and stubbing both keeps them exact: the map is
    fixed, so every number below is arithmetic rather than a coherence estimate.
    """
    from fnirs_pipe.pipeline import synchrony

    result = _result({"S1_D1": _ramp_map(0.2, 0.8)})
    monkeypatch.setattr(synchrony, "_long_signals",
                        lambda raw, ch_type, sep_bands: {"S1_D1": np.arange(8.0)})
    # evaluated as an argument to the stubbed `_wtc_over_pairs`, so stubbing that one is not
    # enough: it reads the montage off recordings these tests do not have
    monkeypatch.setattr(synchrony, "long_axis_over", lambda *a, **k: ["S1_D1"])
    monkeypatch.setattr(synchrony, "_wtc_over_pairs", lambda *a, **k: result)
    return synchrony


def test_no_windows_leaves_the_second_table_unbuilt(stub_pseudo):
    whole, by_cond = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1)
    assert by_cond is None
    assert whole["coherence"].iloc[0] == pytest.approx(0.5)


def test_a_window_spanning_the_record_reproduces_the_whole_run_number(stub_pseudo):
    """The one identity that says "windowed, not recomputed": a window over everything is
    the whole run. Recomputing on a cut would not give this back, because a cut has edges
    of its own."""
    whole, by_cond = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1,
        windows=[("all", float(TIMES[0]), float(TIMES[-1]))])
    assert by_cond["condition"].tolist() == ["all"]
    assert by_cond["coherence"].iloc[0] == pytest.approx(whole["coherence"].iloc[0])


def test_each_window_gets_its_own_null_level(stub_pseudo):
    mid = float(TIMES[len(TIMES) // 2])
    _, by_cond = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=2,
        windows=[("early", float(TIMES[0]), mid - 1e-9),
                 ("late", mid, float(TIMES[-1]))])
    levels = dict(zip(by_cond["condition"], by_cond["coherence"]))
    assert levels["early"] == pytest.approx(0.2)
    assert levels["late"] == pytest.approx(0.8)


def test_the_windowed_null_carries_the_same_columns_as_the_whole_run_one(stub_pseudo):
    """So a real per-condition table and this one subtract cell by cell, `condition` aside."""
    whole, by_cond = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1,
        windows=[("all", float(TIMES[0]), float(TIMES[-1]))])
    assert list(by_cond.columns) == ["condition"] + list(whole.columns)


# ---- --tstart/--tend reaches the whole-run row too ----
#
# `windows` fixed the per-condition rows; the whole-run row had the same defect and no
# parameter to fix it. With `--tstart`/`--tend` the real whole-run table describes the
# window, and the null described the whole recording: one row, compared against a row
# measuring a different span, with nothing saying so. The ramp map makes the arithmetic
# exact, 0.2 over the first half and 0.8 over the second.

def test_without_an_analysis_window_the_whole_run_row_covers_the_record(stub_pseudo):
    whole, _ = stub_pseudo.compute_wtc_pseudo({"s1": None, "s2": None}, 0.02, 0.30, n_iter=1)
    assert whole["coherence"].iloc[0] == pytest.approx(0.5)


@pytest.mark.parametrize("half, expected", [("first", 0.2), ("second", 0.8)])
def test_the_whole_run_row_is_the_window_when_one_is_given(stub_pseudo, half, expected):
    mid = float(TIMES[len(TIMES) // 2])
    window = ((float(TIMES[0]), mid - 1e-9) if half == "first"
              else (mid, float(TIMES[-1])))
    whole, _ = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1, analysis_window=window)
    assert whole["coherence"].iloc[0] == pytest.approx(expected)


def test_a_window_spanning_everything_is_the_unwindowed_number(stub_pseudo):
    """The same identity the per-condition side is pinned by: windowed off the transform,
    never recomputed on a cut."""
    whole, _ = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1,
        analysis_window=(float(TIMES[0]), float(TIMES[-1])))
    assert whole["coherence"].iloc[0] == pytest.approx(0.5)


def test_the_conditions_are_unaffected_by_the_analysis_window(stub_pseudo):
    """Condition windows are absolute times and already lie inside the analysis window, so
    they are read off the same transform either way. Windowing twice would move them."""
    mid = float(TIMES[len(TIMES) // 2])
    args = dict(n_iter=1, windows=[("late", mid, float(TIMES[-1]))])
    _, plain = stub_pseudo.compute_wtc_pseudo({"s1": None, "s2": None}, 0.02, 0.30, **args)
    _, windowed = stub_pseudo.compute_wtc_pseudo(
        {"s1": None, "s2": None}, 0.02, 0.30,
        analysis_window=(mid, float(TIMES[-1])), **args)
    assert windowed["coherence"].iloc[0] == pytest.approx(plain["coherence"].iloc[0])
    assert plain["coherence"].iloc[0] == pytest.approx(0.8)


def test_the_writer_passes_the_window_down(monkeypatch, tmp_path):
    """The wiring, which is where this bug lived: both functions had the parameter for the
    conditions and neither had it for the run."""
    from fnirs_pipe.qc import wtc_null
    seen = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame({"label": ["S1_D1"], "label2": ["S1_D1"],
                             "coherence": [0.5], "coherence_z": [0.55]}), None

    monkeypatch.setattr("fnirs_pipe.pipeline.hyperscanning.compute_wtc_pseudo", _spy)
    monkeypatch.setattr("fnirs_pipe.pipeline.hyperscanning._hyper_sidecar",
                        lambda *a, **k: None)
    monkeypatch.setattr("fnirs_pipe.utils.lineage.path_from", lambda r: None)
    # both read the montage off the recordings, which these stubs do not have
    monkeypatch.setattr("fnirs_pipe.pipeline.synchrony.wtc_grid_params", lambda raws: {})
    wtc_null.write_wtc_null(
        group_id="G1", task="tap", aligned_raws={"s1": None, "s2": None},
        output_dir=tmp_path, n_iter=1, chroma=("hbo",), analysis_window=(60.0, 300.0))
    assert seen["analysis_window"] == (60.0, 300.0)
