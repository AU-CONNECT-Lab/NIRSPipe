"""The phase-scrambled null, and the band mean it is read against.

`phase_scramble` has to preserve the magnitude spectrum exactly and return a real series;
those are the two properties that make the surrogate a fair null rather than a different
signal. What it must *not* do is decorrelate: a surrogate of a signal whose power sits in a
few low-frequency bins keeps that rhythm and only shifts its phase, which is the point. The
null asks whether two people's timing is related given both have these rhythms, not whether
one of them has a rhythm at all.

The rest covers `wtc_band_mean` masking the cone by default while still reporting the share
inside it, the Fisher z column, and the ROI grouping of the channel values.
"""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.hyper.roi import roi_maps_from_channels, roi_mean_of_channels
from fnirs_pipe.pipeline.hyper.surrogate import phase_scramble
from fnirs_pipe.pipeline.hyper.wtc import WTCResult, wtc_band_mean

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
    """The padded half is dropped without being asked for."""
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


# ---- ROI grouping ----

def test_an_roi_under_the_channel_minimum_is_dropped():
    """Dropped from the numbers, not from the table: the row stays, blank."""
    df = wtc_band_mean(_result({"A1": _map(0.4), "A2": _map(0.6), "B1": _map(0.2)}),
                       0.06, 0.15)
    roi_map = {"roiA": ["A1", "A2"], "roiB": ["B1"]}
    kept = roi_mean_of_channels(df, roi_map, min_channels=2).set_index("label")
    assert list(kept.index) == ["roiA", "roiB"]
    assert np.isfinite(kept.loc["roiA", "coherence"])
    assert kept.loc["roiB", ["coherence", "coherence_z", "n_valid_frac"]].isna().all()
    assert kept.loc["roiB", "n_ch"] == 1
    assert roi_mean_of_channels(df, roi_map, min_channels=1)["coherence"].notna().all()


def _crossed_band(values):
    return pd.DataFrame([{"sub1": "a", "sub2": "b", "label": l1, "label2": l2,
                          "coherence": v, "n_valid_frac": 1.0}
                         for (l1, l2), v in values.items()])


def test_a_crossed_roi_cell_needs_channels_on_both_sides_not_just_pairings():
    roi_map = {"roiA": ["A1", "A2"], "roiB": ["B1"]}
    df = _crossed_band({(a, b): 0.4 for a in ("A1", "A2", "B1") for b in ("A1", "A2", "B1")})
    out = roi_mean_of_channels(df, roi_map, min_channels=2)
    filled = out[out["coherence"].notna()]
    # roiA x roiB rests on two pairings but on one channel of the second member
    assert list(zip(filled["label"], filled["label2"])) == [("roiA", "roiA")]
    assert len(out) == 4                  # the blank cells keep their rows


def test_a_channel_without_a_value_does_not_count_towards_the_minimum():
    roi_map = {"roiA": ["A1", "A2"]}
    df = _crossed_band({("A1", "A1"): 0.4, ("A1", "A2"): float("nan"),
                        ("A2", "A1"): 0.4, ("A2", "A2"): float("nan")})
    # A2 of the second member was rejected, leaving one channel on that side
    out = roi_mean_of_channels(df, roi_map, min_channels=2)
    assert len(out) == 1 and np.isnan(out["coherence"].iloc[0])


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
    maps = roi_maps_from_channels(result, roi_map, min_channels=1)
    assert ("roiA", "roiB") in maps.pairs[("s1", "s2")]


def test_a_cell_the_table_leaves_blank_gets_no_map():
    """One rule for both, so no figure stands for a region the table leaves empty."""
    cells = {(a, b): _map(0.4) for a in ("A1", "A2", "B1") for b in ("A1", "A2", "B1")}
    result = _result(cells)
    roi_map = {"roiA": ["A1", "A2"], "roiB": ["B1"]}
    table = roi_mean_of_channels(wtc_band_mean(result, 0.06, 0.15), roi_map, min_channels=2)
    filled = set(zip(*table[table["coherence"].notna()][["label", "label2"]].T.values))
    assert set(roi_maps_from_channels(result, roi_map).pairs[("s1", "s2")]) == filled
    assert filled == {("roiA", "roiA")}


def test_a_channel_two_rois_list_counts_in_both():
    """The rule ISC and resting state follow. Mapped one-to-one, the shared channel would
    land only in the last ROI and the first would average one channel fewer than it lists."""
    result = _result({"A1": _map(0.2), "S": _map(0.8), "B1": _map(0.4)})
    roi_map = {"roiA": ["A1", "S"], "roiB": ["S", "B1"]}
    table = roi_mean_of_channels(wtc_band_mean(result, 0.06, 0.15), roi_map, min_channels=2)
    means = dict(zip(table["label"], table["coherence"]))

    assert means == pytest.approx({"roiA": 0.5, "roiB": 0.6})
    maps = wtc_band_mean(roi_maps_from_channels(result, roi_map), 0.06, 0.15)
    assert dict(zip(maps["label"], maps["coherence"])) == pytest.approx(means)


def test_a_group_of_three_is_refused_rather_than_half_scrambled():
    """Only one subject is scrambled, so a third member would leave real pairs in the null."""
    from fnirs_pipe.pipeline.hyper.surrogate import compute_wtc_phase_null

    with pytest.raises(ValueError, match="exactly 2 subjects"):
        compute_wtc_phase_null({"s1": None, "s2": None, "s3": None}, 0.06, 0.15, n_iter=1)


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


def _null(frame, cond_frames=(), levels=None):
    """A NullDraws around an already-made frame, for the tests that stub the draw away."""
    from fnirs_pipe.pipeline.hyper.surrogate import NullDraws

    keys = ["sub1", "sub2", "label"] + (["label2"] if "label2" in frame.columns else [])
    return NullDraws(draws=[frame], cond_draws=list(cond_frames), keys=keys,
                      levels=levels or {})


@pytest.fixture
def stub_null(monkeypatch):
    """compute_wtc_phase_null with the transform and the channel picking replaced.

    Neither is what these tests are about, and stubbing both keeps them exact: the map is
    fixed, so every number below is arithmetic rather than a coherence estimate.
    """
    from fnirs_pipe.pipeline.hyper import surrogate

    result = _result({"S1_D1": _ramp_map(0.2, 0.8)})
    monkeypatch.setattr(surrogate, "_long_signals",
                        lambda raw, ch_type, sep_bands: {"S1_D1": np.arange(8.0)})
    # evaluated as an argument to the stubbed `_wtc_over_pairs`, so stubbing that one is not
    # enough: it reads the montage off recordings these tests do not have
    monkeypatch.setattr(surrogate, "long_axis_over", lambda *a, **k: ["S1_D1"])
    monkeypatch.setattr(surrogate, "_wtc_over_pairs", lambda *a, **k: result)
    return surrogate


def test_no_windows_leaves_the_second_table_unbuilt(stub_null):
    whole, by_cond = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1).summarise()
    assert by_cond is None
    assert whole["null_mean"].iloc[0] == pytest.approx(0.5)


def test_a_window_spanning_the_record_reproduces_the_whole_run_number(stub_null):
    """The one identity that says "windowed, not recomputed": a window over everything is
    the whole run. Recomputing on a cut would not give this back, because a cut has edges
    of its own."""
    whole, by_cond = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1,
        windows=[("all", float(TIMES[0]), float(TIMES[-1]))]).summarise()
    assert by_cond["condition"].tolist() == ["all"]
    assert by_cond["null_mean"].iloc[0] == pytest.approx(whole["null_mean"].iloc[0])


def test_each_window_gets_its_own_null_level(stub_null):
    mid = float(TIMES[len(TIMES) // 2])
    _, by_cond = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=2,
        windows=[("early", float(TIMES[0]), mid - 1e-9),
                 ("late", mid, float(TIMES[-1]))]).summarise()
    levels = dict(zip(by_cond["condition"], by_cond["null_mean"]))
    assert levels["early"] == pytest.approx(0.2)
    assert levels["late"] == pytest.approx(0.8)


def test_the_windowed_null_carries_the_same_columns_as_the_whole_run_one(stub_null):
    """So a real per-condition table and this one subtract cell by cell, `condition` aside."""
    whole, by_cond = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1,
        windows=[("all", float(TIMES[0]), float(TIMES[-1]))]).summarise()
    assert list(by_cond.columns) == ["condition"] + list(whole.columns)


# ---- --tstart/--tend reaches the whole-run row too ----
#
# With `--tstart`/`--tend` the real whole-run table describes the window, so the null's
# whole-run row has to describe the same window rather than the whole recording, as
# `windows` does for the per-condition rows. The ramp map makes the arithmetic exact, 0.2
# over the first half and 0.8 over the second.

def test_without_an_analysis_window_the_whole_run_row_covers_the_record(stub_null):
    whole, _ = stub_null.compute_wtc_phase_null({"s1": None, "s2": None}, 0.02, 0.30, n_iter=1).summarise()
    assert whole["null_mean"].iloc[0] == pytest.approx(0.5)


@pytest.mark.parametrize("half, expected", [("first", 0.2), ("second", 0.8)])
def test_the_whole_run_row_is_the_window_when_one_is_given(stub_null, half, expected):
    mid = float(TIMES[len(TIMES) // 2])
    window = ((float(TIMES[0]), mid - 1e-9) if half == "first"
              else (mid, float(TIMES[-1])))
    whole, _ = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1, analysis_window=window).summarise()
    assert whole["null_mean"].iloc[0] == pytest.approx(expected)


def test_a_window_spanning_everything_is_the_unwindowed_number(stub_null):
    """The same identity the per-condition side is pinned by: windowed off the transform,
    never recomputed on a cut."""
    whole, _ = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30, n_iter=1,
        analysis_window=(float(TIMES[0]), float(TIMES[-1]))).summarise()
    assert whole["null_mean"].iloc[0] == pytest.approx(0.5)


def test_the_conditions_are_unaffected_by_the_analysis_window(stub_null):
    """Condition windows are absolute times and already lie inside the analysis window, so
    they are read off the same transform either way. Windowing twice would move them."""
    mid = float(TIMES[len(TIMES) // 2])
    args = dict(n_iter=1, windows=[("late", mid, float(TIMES[-1]))])
    _, plain = stub_null.compute_wtc_phase_null({"s1": None, "s2": None}, 0.02, 0.30, **args).summarise()
    _, windowed = stub_null.compute_wtc_phase_null(
        {"s1": None, "s2": None}, 0.02, 0.30,
        analysis_window=(mid, float(TIMES[-1])), **args).summarise()
    assert windowed["null_mean"].iloc[0] == pytest.approx(plain["null_mean"].iloc[0])
    assert plain["null_mean"].iloc[0] == pytest.approx(0.8)


def test_the_writer_passes_the_window_down(monkeypatch, tmp_path):
    """The wiring: both functions take the window for the run as well as the ones for the
    conditions."""
    from fnirs_pipe.pipeline.hyper import wtc_null
    seen = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return _null(pd.DataFrame({"sub1": ["s1"], "sub2": ["s2"], "label": ["S1_D1"],
                                   "label2": ["S1_D1"], "coherence": [0.5],
                                   "n_valid_frac": [1.0]}))

    monkeypatch.setattr("fnirs_pipe.pipeline.hyper.compute_wtc_phase_null", _spy)
    monkeypatch.setattr("fnirs_pipe.pipeline.hyper._hyper_sidecar",
                        lambda *a, **k: None)
    monkeypatch.setattr("fnirs_pipe.utils.lineage.path_from", lambda r: None)
    # both read the montage off the recordings, which these stubs do not have
    monkeypatch.setattr("fnirs_pipe.pipeline.hyper.wtc.wtc_grid_params", lambda raws: {})
    wtc_null.run_wtc_null(
        group_id="G1", task="tap", aligned_raws={"s1": None, "s2": None},
        output_dir=tmp_path, n_iter=1, chroma=("hbo",), analysis_window=(60.0, 300.0))
    assert seen["analysis_window"] == (60.0, 300.0)


# ---- the ROI null ----

def _roi_draws(values_per_iter):
    """NullDraws with hand-made draws: [{label: value}] per iteration, two channels an ROI."""
    from fnirs_pipe.pipeline.hyper.surrogate import NullDraws

    frames = []
    for values in values_per_iter:
        frames.append(pd.DataFrame({
            "sub1": ["s1"] * len(values), "sub2": ["s2"] * len(values),
            "label": list(values), "coherence": list(values.values()),
            "coherence_z": [np.arctanh(v) for v in values.values()],
            "n_valid_frac": [1.0] * len(values)}))
    return NullDraws(draws=frames, cond_draws=[], keys=["sub1", "sub2", "label"], levels={})


ROI_MAP = {"r1": ["S1_D1", "S1_D2"]}


def test_the_roi_null_groups_inside_each_iteration():
    """The ROI null is the spread of the ROI mean, not of the channels it averages.

    Two channels moving together give the ROI mean their own spread; two moving oppositely
    give it none. Summarising the channels first would report the same sd for both, which is
    the whole reason the grouping happens per iteration.
    """
    together = _roi_draws([{"S1_D1": 0.2, "S1_D2": 0.2},
                           {"S1_D1": 0.4, "S1_D2": 0.4},
                           {"S1_D1": 0.6, "S1_D2": 0.6}])
    opposed = _roi_draws([{"S1_D1": 0.2, "S1_D2": 0.6},
                          {"S1_D1": 0.4, "S1_D2": 0.4},
                          {"S1_D1": 0.6, "S1_D2": 0.2}])

    t_whole, _ = together.summarise_roi(ROI_MAP, min_channels=1)
    o_whole, _ = opposed.summarise_roi(ROI_MAP, min_channels=1)

    # same channel-level spread on both sides, so a bracket from null_sd could not tell them apart
    assert together.summarise()[0]["null_sd"].iloc[0] == pytest.approx(
        opposed.summarise()[0]["null_sd"].iloc[0])
    assert t_whole["null_mean"].iloc[0] == pytest.approx(0.4)
    assert o_whole["null_mean"].iloc[0] == pytest.approx(0.4)
    assert t_whole["null_sd"].iloc[0] == pytest.approx(0.2)
    assert o_whole["null_sd"].iloc[0] == pytest.approx(0.0)


def test_the_roi_null_ranks_the_real_roi_value():
    null = _roi_draws([{"S1_D1": 0.2, "S1_D2": 0.2}, {"S1_D1": 0.4, "S1_D2": 0.4},
                       {"S1_D1": 0.6, "S1_D2": 0.6}, {"S1_D1": 0.8, "S1_D2": 0.8}])
    real = pd.DataFrame({"sub1": ["s1"], "sub2": ["s2"], "label": ["r1"],
                         "coherence": [0.7], "n_valid_frac": [1.0]})

    whole, _ = null.summarise_roi(ROI_MAP, real=real, min_channels=1)

    assert whole["label"].iloc[0] == "r1"
    assert whole["percentile"].iloc[0] == pytest.approx(75.0)


def test_a_crossed_null_still_ranks_only_the_homologous_roi_value():
    """A crossed draw carries within-ROI cross pairings the reported ROI value does not."""
    from fnirs_pipe.pipeline.hyper.surrogate import NullDraws

    rows = [("S1_D1", "S1_D1", 0.4), ("S1_D2", "S1_D2", 0.4),
            ("S1_D1", "S1_D2", 0.9), ("S1_D2", "S1_D1", 0.9)]
    frame = pd.DataFrame({"sub1": ["s1"] * 4, "sub2": ["s2"] * 4,
                          "label": [r[0] for r in rows], "label2": [r[1] for r in rows],
                          "coherence": [r[2] for r in rows], "n_valid_frac": [1.0] * 4})
    null = NullDraws(draws=[frame], cond_draws=[],
                      keys=["sub1", "sub2", "label", "label2"], levels={})

    whole, _ = null.summarise_roi(ROI_MAP, min_channels=1)

    assert len(whole) == 1
    assert whole["null_mean"].iloc[0] == pytest.approx(0.4)


def test_a_suffixed_map_passed_straight_to_the_api_matches_the_bare_one():
    """The CLI strips suffixes on load; a caller going straight to these functions does not
    pass through it, and a "A1 hbo" entry used to match no channel at all."""
    from fnirs_pipe.pipeline.hyper.isc import roi_mean_of_isc

    df = wtc_band_mean(_result({"A1": _map(0.2), "A2": _map(0.8)}), 0.06, 0.15)
    bare = roi_mean_of_channels(df, {"roiA": ["A1", "A2"]}, min_channels=2)
    suffixed = roi_mean_of_channels(df, {"roiA": ["A1 hbo", "A2 hbr"]}, min_channels=2)
    pd.testing.assert_frame_equal(bare, suffixed)

    mat = np.array([[0.5, 0.1], [0.2, 0.4]])
    a, _ = roi_mean_of_isc(mat, ["A1", "A2"], {"r": ["A1", "A2"]}, min_channels=1)
    b, _ = roi_mean_of_isc(mat, ["A1", "A2"], {"r": ["A1 hbo", "A2 hbo"]}, min_channels=1)
    np.testing.assert_array_equal(a, b)
