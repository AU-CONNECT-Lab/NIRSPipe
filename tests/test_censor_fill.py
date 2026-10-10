"""Corrupted BAD_ spans are filled before the bandpass; BAD_unselected and kept samples are not."""

import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.exceptions import StageError
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.censor_fill import corrupted_mask, fill_corrupted, fill_values
from nirspipe.pipeline.denoise import bandpass_filter
from nirspipe.utils.lineage import lineage_of, stage_of
from nirspipe.utils.spans import UNSELECTED

from tests._fingerprint import CLI_ARGS, fingerprint_raw
from tests._synth import _write_dataset_root, _write_subject

SFREQ = 10.0


def _haemo(data, onsets=(), durations=(), descs=()):
    data = np.atleast_2d(np.asarray(data, dtype=float))
    names = [f"S{i}_D{i} hbo" for i in range(1, data.shape[0] + 1)]
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, "hbo"), verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


# ---- the three fills ----

def test_linear_joins_the_samples_either_side_and_holds_the_ends():
    x = np.array([[0.0, 1.0, 9.0, 9.0, 4.0, 5.0, 9.0, 9.0]])
    gap = np.array([False, False, True, True, False, False, True, True])
    out = fill_values(x, gap, "linear", SFREQ)
    np.testing.assert_array_equal(out, [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]])


def test_spline_runs_through_every_kept_sample():
    t = np.arange(40.0)
    x = (0.01 * t ** 3 - 0.2 * t ** 2 + t)[None, :]
    gap = np.zeros(40, dtype=bool)
    gap[15:22] = True
    out = fill_values(x, gap, "spline", SFREQ)
    # a cubic is what a not-a-knot spline reproduces exactly
    np.testing.assert_allclose(out, x, rtol=1e-9, atol=1e-9)
    gap[:3] = True
    assert (fill_values(x, gap, "spline", SFREQ)[0, :3] == x[0, 3]).all()


def test_lomb_fits_the_kept_samples_and_matches_their_spread():
    from nirspipe.pipeline.censor_fill import _lomb_reconstruction

    t = np.arange(int(SFREQ * 400)) / SFREQ
    x = np.vstack([np.sin(2 * np.pi * 0.05 * t), np.zeros_like(t)])
    gap = (t >= 200) & (t < 215)
    rebuilt = _lomb_reconstruction(x, ~gap, SFREQ)
    assert np.corrcoef(rebuilt[0, ~gap], x[0, ~gap])[0, 1] > 0.999
    assert rebuilt[0, ~gap].std() == pytest.approx(x[0, ~gap].std())
    out = fill_values(x, gap, "lomb", SFREQ)
    assert np.isfinite(out).all()          # the flat channel has no SD to scale to


@pytest.mark.parametrize("method", ["linear", "spline", "lomb"])
def test_every_kept_sample_comes_back_bit_for_bit(method):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 600))
    gap = np.zeros(600, dtype=bool)
    gap[100:130] = gap[400:405] = True
    out = fill_values(x, gap, method, SFREQ)
    assert np.array_equal(out[:, ~gap], x[:, ~gap])
    assert not np.array_equal(out[:, gap], x[:, gap])


# ---- which spans ----

def test_only_corrupted_spans_are_filled():
    raw = _haemo(np.zeros(100), [1.0, 3.0, 6.0], [0.3, 1.0, 0.5],
                 ["BAD_gvtd", UNSELECTED, "BAD_motion"])
    assert np.flatnonzero(corrupted_mask(raw)).tolist() == [10, 11, 12, 60, 61, 62, 63, 64]


def test_a_recording_without_corrupted_spans_comes_back_untouched():
    rng = np.random.default_rng(2)
    raw = _haemo(rng.standard_normal(300), [0.0], [10.0], [UNSELECTED])
    before = raw.get_data().copy()
    assert fill_corrupted(raw, "lomb") is raw
    assert np.array_equal(raw.get_data(), before)
    assert lineage_of(raw) is None


def test_a_recording_corrupted_end_to_end_is_refused():
    raw = _haemo(np.zeros(100), [0.0], [10.0], ["BAD_gvtd"])
    with pytest.raises(StageError, match="no sample lies outside its corrupted BAD_ spans"):
        fill_corrupted(raw)


def test_the_fill_is_stamped_with_its_method():
    raw = fill_corrupted(_haemo(np.arange(100.0), [2.0], [1.0], ["BAD_gvtd"]), "linear")
    assert stage_of(raw) == "filled"
    assert lineage_of(raw).params == {"censor_fill": "linear", "censor_filled_s": 1.0}


# ---- the whole run ----

def _dataset(root):
    raw, truth = fingerprint_raw("01", "tapping")
    # a corrupted span the recording already carries, inside the first rest
    raw.set_annotations(raw.annotations + mne.Annotations(
        [150.0], [6.0], ["BAD_motion"], orig_time=raw.annotations.orig_time))
    bids = root / "bids"
    _write_dataset_root(bids, ["01"])
    _write_subject(bids, "01", "tapping", raw)
    return bids


def _run(bids, out, *extra):
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "denoise",
                  "--high-pass", "0.01", "--low-pass", "0.1", *extra,
                  "--no-report", "--skip-bids-validation"])
    return {p.name.split("_desc-")[1].split("_")[0]: p for p in out.rglob("*_nirs.snirf")}


def test_the_bandpass_sees_the_fill_and_preproc_keeps_the_measured_data(tmp_path, caplog):
    bids = _dataset(tmp_path)
    with caplog.at_level(logging.WARNING, logger="nirspipe"):
        files = _run(bids, tmp_path / "out", "--censor-fill", "spline")
    assert "may inflate coherence" in caplog.text

    preproc = read_snirf(files["preproc"])
    expected = bandpass_filter(fill_corrupted(preproc.copy(), "spline"), 0.01, 0.1)
    filtered = read_snirf(files["filtered"])
    np.testing.assert_array_equal(filtered.get_data(), expected.get_data())
    unfilled = bandpass_filter(read_snirf(files["preproc"]), 0.01, 0.1)
    assert not np.array_equal(filtered.get_data(), unfilled.get_data())

    sidecar = json.loads(files["filtered"].with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["parameters"]["censor_fill"] == "spline"
    assert sidecar["excluded_spans"] == [{"onset": 150.0, "duration": 6.0,
                                          "description": "BAD_motion", "kind": "corrupted",
                                          "source": "input file"}]


def test_the_exported_script_fills_as_the_run_filled(tmp_path):
    bids = _dataset(tmp_path)
    out, script_out = tmp_path / "out", tmp_path / "script"
    files = _run(bids, out, "--censor-fill", "lomb")
    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)
    (script_filtered,) = script_out.rglob("*desc-filtered_nirs.snirf")
    np.testing.assert_array_equal(read_snirf(script_filtered).get_data(),
                                  read_snirf(files["filtered"]).get_data())
