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

def test_the_cone_is_not_masked_by_default():
    """Half the record outside the cone: masking would change the number, and must not."""
    coi = np.where(np.arange(len(TIMES)) < 15, 1e6, 1e-12)
    data = _map(0.5)
    data["coi"] = coi
    data["wtc"][:, 15:] = 1.0
    df = wtc_band_mean(_result({"A": data}), 0.06, 0.15)
    assert df["coherence"].iloc[0] == pytest.approx(0.75)


def test_masking_the_cone_drops_the_padded_half():
    coi = np.where(np.arange(len(TIMES)) < 15, 1e6, 1e-12)
    data = _map(0.5)
    data["coi"] = coi
    data["wtc"][:, 15:] = 1.0
    df = wtc_band_mean(_result({"A": data}), 0.06, 0.15, mask_coi=True)
    assert df["coherence"].iloc[0] == pytest.approx(0.5)


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
