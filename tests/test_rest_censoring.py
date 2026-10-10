"""FC and ALFF use only the frames outside every BAD_ span, and nothing changes without one."""

import json
from datetime import datetime, timezone

import mne
import numpy as np
import pandas as pd
import pytest
from scipy import signal

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.glm_censored import kept_frames
from nirspipe.pipeline.restingstate import (compute_alff, compute_fc, compute_fc_roi,
                                            compute_fc_seed, lomb_scargle)
from nirspipe.utils.spans import UNSELECTED

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

SFREQ = 5.0
ROI_MAP = {"A": ["S1_D1", "S2_D2"], "B": ["S3_D3", "S4_D4"]}


def _raw(seconds=400.0, onsets=(), durations=(), descs=(), seed=0):
    rng = np.random.default_rng(seed)
    n = int(SFREQ * seconds)
    t = np.arange(n) / SFREQ
    shared = np.sin(2 * np.pi * 0.03 * t)
    hbo = np.vstack([shared * w + rng.standard_normal(n) for w in (1.0, 0.8, -0.5, 0.0)])
    hbr = -0.3 * hbo + 0.5 * rng.standard_normal(hbo.shape)
    names = [f"S{i}_D{i} {c}" for c in ("hbo", "hbr") for i in range(1, 5)]
    raw = mne.io.RawArray(np.vstack([hbo, hbr]) * 1e-7,
                          mne.create_info(names, SFREQ, ["hbo"] * 4 + ["hbr"] * 4),
                          verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


def _spanned():
    return _raw(onsets=[30.0, 150.0, 200.0, 330.0], durations=[40.0, 0.0, 25.0, 70.0],
                descs=["BAD_gvtd", "tap", UNSELECTED, UNSELECTED])


def _scrambled(raw):
    """The same recording with garbage at every BAD_ frame."""
    out = raw.copy()
    drop = ~kept_frames(raw)
    data = out.get_data()
    data[:, drop] = np.random.default_rng(9).standard_normal((len(data), drop.sum())) * 1e-4
    out._data[:] = data
    return out


# ---- the spectrum ----

@pytest.mark.parametrize("n", [2000, 2001])
def test_lomb_scargle_with_every_sample_kept_is_the_periodogram(n):
    x = np.random.default_rng(1).standard_normal((2, n))
    freqs, power = lomb_scargle(x, np.ones(n, dtype=bool), SFREQ)
    ref_f, ref_p = signal.periodogram(x - x.mean(axis=1, keepdims=True), SFREQ,
                                      scaling="spectrum")
    np.testing.assert_array_equal(freqs, ref_f)
    np.testing.assert_allclose(power[:, 1:], ref_p[:, 1:], rtol=1e-8)
    assert (power[:, 0] == 0).all()


@pytest.mark.parametrize("n", [3000, 3001])
def test_lomb_scargle_with_gaps_is_scipys(n):
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(n)) * 0.05 + rng.standard_normal(n)
    keep = np.ones(n, dtype=bool)
    keep[400:900] = keep[1500:1520] = keep[-60:] = False
    freqs, power = lomb_scargle(x[np.newaxis], keep, SFREQ)
    y = x[keep] - x[keep].mean()
    ref = signal.lombscargle(np.flatnonzero(keep) / SFREQ, y, 2 * np.pi * freqs[1:],
                             normalize=True) * y.var()
    np.testing.assert_allclose(power[0, 1:], ref, rtol=1e-7, atol=1e-12 * ref.max())


# ---- without spans nothing moves ----

def test_without_spans_alff_is_todays_periodogram_bit_for_bit():
    raw = _raw(onsets=[50.0], durations=[0.0], descs=["tap"])
    out = compute_alff(raw, low_pass=0.08, high_pass=0.01)
    for i, ch in enumerate(raw.get_data()):
        freqs, power = signal.periodogram(ch - np.nanmean(ch), SFREQ, scaling="spectrum")
        amp = np.sqrt(power)
        band = amp[(freqs >= 0.01) & (freqs <= 0.08)]
        assert out["alff"].iloc[i] == np.nanmean(band)
        assert out["falff"].iloc[i] == np.nansum(band) / np.nansum(amp[1:])


def test_without_spans_fc_is_todays_corrcoef_bit_for_bit():
    raw = _raw()
    picks = [c for c in raw.ch_names if c.endswith(" hbo")]
    np.testing.assert_array_equal(compute_fc(raw, "hbo").to_numpy(),
                                  np.corrcoef(raw.get_data(picks=picks)))


# ---- with spans only kept frames count ----

def test_fc_is_pearson_on_the_kept_frames():
    raw = _spanned()
    keep = kept_frames(raw)
    picks = [c for c in raw.ch_names if c.endswith(" hbr")]
    np.testing.assert_array_equal(compute_fc(raw, "hbr").to_numpy(),
                                  np.corrcoef(raw.get_data(picks=picks)[:, keep]))


def test_nothing_inside_a_span_reaches_fc_or_alff():
    raw = _spanned()
    junk = _scrambled(raw)
    for chromo in ("hbo", "hbr"):
        pd.testing.assert_frame_equal(compute_fc(junk, chromo), compute_fc(raw, chromo))
        pd.testing.assert_frame_equal(compute_fc_roi(junk, ROI_MAP, chromo),
                                      compute_fc_roi(raw, ROI_MAP, chromo))
        pd.testing.assert_frame_equal(compute_fc_seed(junk, ROI_MAP, chromo),
                                      compute_fc_seed(raw, ROI_MAP, chromo))
    pd.testing.assert_frame_equal(compute_alff(junk, low_pass=0.08, high_pass=0.01),
                                  compute_alff(raw, low_pass=0.08, high_pass=0.01),
                                  rtol=1e-10)
    assert not compute_alff(junk, 0.08, 0.01).equals(
        compute_alff(_raw(), low_pass=0.08, high_pass=0.01))


def test_a_channel_constant_on_its_kept_samples_keeps_zero_alff():
    raw = _spanned()
    keep = kept_frames(raw)
    raw._data[0, keep] = 1e-7
    out = compute_alff(raw, low_pass=0.08, high_pass=0.01)
    assert out["alff"].iloc[0] == 0.0 and out["falff"].iloc[0] == 0.0


# ---- the whole run ----

def _rest_run(tmp_path, *extra):
    bids, _ = make_fingerprint_dataset(tmp_path, task="rest", rest=True)
    table = tmp_path / "keep.tsv"
    table.write_text("onset\tduration\n0\t200\n", encoding="utf-8")
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "rest",
                  "--keep-spans", str(table), *extra, "--no-report", "--skip-bids-validation"])
    return out


def test_a_rest_run_correlates_and_measures_its_kept_stretch(tmp_path):
    out = _rest_run(tmp_path, "--min-time", "150")
    (errts,) = out.rglob("*desc-errts_nirs.snirf")
    resid = read_snirf(errts)
    keep = kept_frames(resid)
    assert keep.sum() == 200 * resid.info["sfreq"]

    (fc_path,) = out.rglob("*chromo-hbo_stat-pearson_relmat.tsv")
    fc = pd.read_csv(fc_path, sep="\t", index_col="channel")
    picks = list(fc.index)
    expected = np.corrcoef(resid.get_data(picks=picks)[:, keep])
    good = fc.notna().to_numpy()
    np.testing.assert_allclose(fc.to_numpy()[good], expected[good], rtol=1e-6, atol=1e-6)

    for path in (fc_path, *out.rglob("*stat-alff_nirsmap.tsv")):
        params = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["parameters"]
        assert params["kept_s"] == pytest.approx(200.0)
        assert params["min_time"] == 150.0


def test_a_rest_run_under_min_time_writes_no_fc_or_alff_and_says_why(tmp_path):
    out = _rest_run(tmp_path, "--min-time", "300")
    assert not list(out.rglob("*_relmat.tsv"))
    assert not list(out.rglob("*stat-alff*"))
    (errts,) = out.rglob("*desc-errts_nirs.json")
    (broad,) = out.rglob("*desc-errtsbroad_nirs.json")
    for path, what in ((errts, "FC"), (broad, "ALFF")):
        (note,) = json.loads(path.read_text(encoding="utf-8"))["skipped_outputs"]
        assert note["output"] == what
        assert note["kept_s"] == pytest.approx(200.0) and note["min_time"] == 300.0
