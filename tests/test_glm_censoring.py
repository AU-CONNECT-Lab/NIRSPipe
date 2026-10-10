"""The GLM and the confound regressions leave every BAD_ frame out of the fit, nothing else."""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.exceptions import StageError
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.pipeline.glm_censored import (ar_order, events_left, fit_channel,
                                            gap_ar_coefficients, kept_frames)
from nirspipe.utils.lineage import lineage_of
from nirspipe.utils.spans import UNSELECTED

from tests._fingerprint import CLI_ARGS, EVENT_DURATION, make_fingerprint_dataset

SFREQ = 5.0


def _haemo(seconds=300.0, onsets=(), durations=(), descs=(), n_ch=2, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n_ch, int(SFREQ * seconds))) * 1e-7
    names = [f"S{i}_D{i} hbo" for i in range(1, n_ch + 1)]
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, "hbo"), verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


def _glm(raw, noise_model="ols", **kw):
    return run_glm_pipeline(raw, stim_dur=5.0, hrf_model="spm", noise_model=noise_model,
                            drift_model="polynomial", high_pass=None, drift_order=1,
                            fir_delays=None, **kw)


# ---- pieces ----

def test_both_kinds_of_span_leave_the_fit():
    raw = _haemo(10.0, [1.0, 4.0, 7.0], [1.0, 0.0, 0.4], ["BAD_gvtd", "tap", UNSELECTED])
    assert np.flatnonzero(~kept_frames(raw)).tolist() == [5, 6, 7, 8, 9, 35, 36]


@pytest.mark.parametrize("model, order", [("ols", 0), ("ar1", 1), ("ar7", 7), ("auto", 20)])
def test_each_noise_model_names_its_order(model, order):
    assert ar_order(model, SFREQ) == order


def test_ar_coefficients_ignore_what_lies_in_the_gaps():
    rng = np.random.default_rng(3)
    x = np.zeros(20000)
    for t in range(2, len(x)):
        x[t] = 0.6 * x[t - 1] - 0.2 * x[t - 2] + rng.standard_normal()
    keep = np.ones(len(x), dtype=bool)
    for start in range(1000, 19000, 2000):
        keep[start:start + 300] = False
        x[start:start + 300] = rng.standard_normal(300) * 50      # garbage nobody may read
    np.testing.assert_allclose(gap_ar_coefficients(x, keep, 2), [0.6, -0.2], atol=0.02)
    np.testing.assert_allclose(gap_ar_coefficients(x, keep, 6, search=True), [0.6, -0.2],
                               atol=0.02)


def test_too_few_lag_windows_are_refused():
    keep = np.tile([True] * 3 + [False] * 3, 50)
    with pytest.raises(StageError, match="too few kept stretches longer than 5 frames"):
        gap_ar_coefficients(np.ones(300), keep, 4)


def test_ols_on_the_kept_rows_is_plain_least_squares():
    rng = np.random.default_rng(4)
    design = np.column_stack([rng.standard_normal(200), np.ones(200)])
    y = design @ [2.0, 1.0] + rng.standard_normal(200)
    keep = np.ones(200, dtype=bool)
    keep[50:80] = False
    y_spoiled = y.copy()
    y_spoiled[~keep] = 1e6
    expected = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    np.testing.assert_allclose(fit_channel(y_spoiled, design, keep, 0).theta.ravel(), expected)


def test_events_left_are_those_touching_no_span():
    events = pd.DataFrame({"trial_type": ["tap", "tap", "tap", "rest"],
                           "onset": [10.0, 50.0, 70.0, 60.0], "duration": [5.0, 5.0, 0.0, 0.0]})
    spans = [(48.0, 52.0, "BAD_gvtd"), (60.0, 61.0, UNSELECTED)]
    assert events_left(events, spans) == {"tap": {"left": 2, "total": 3},
                                          "rest": {"left": 0, "total": 1}}


# ---- the fit ----

def test_without_spans_the_fit_takes_the_existing_path_and_records_nothing_left_out():
    raw = _haemo(onsets=[20.0, 120.0, 220.0], durations=[0.0] * 3, descs=["tap"] * 3)
    *_, resid = _glm(raw)
    params = lineage_of(resid).params
    assert params["censored_frac"] == 0.0 and params["kept_frames"] == raw.n_times
    assert params["events_left"] == {"tap": {"left": 3, "total": 3}}
    assert "dropped_conditions" not in params


@pytest.mark.parametrize("noise_model", ["ols", "ar1", "auto", "ar_irls"])
def test_the_residual_is_full_length_and_the_fit_its_kept_rows(noise_model):
    raw = _haemo(onsets=[20.0, 120.0, 220.0, 100.0], durations=[0.0, 0.0, 0.0, 30.0],
                 descs=["tap", "tap", "tap", "BAD_gvtd"])
    _, est, dm, resid = _glm(raw, noise_model)
    assert resid.get_data().shape == raw.get_data().shape
    theta = np.column_stack([est.data[ch].theta.ravel() for ch in raw.ch_names])
    np.testing.assert_allclose(resid.get_data(), raw.get_data() - (dm.values @ theta).T)
    params = lineage_of(resid).params
    assert params["kept_frames"] == raw.n_times - int(30 * SFREQ)
    assert params["censored_corrupted_frac"] == pytest.approx(0.1)
    assert params["events_left"] == {"tap": {"left": 2, "total": 3}}


def test_the_rows_a_span_covers_do_not_move_an_ols_fit():
    onsets, durations, descs = [20.0, 120.0, 220.0, 100.0], [0.0, 0.0, 0.0, 30.0], \
        ["tap", "tap", "tap", "BAD_gvtd"]
    raw = _haemo(onsets=onsets, durations=durations, descs=descs)
    spoiled = raw.copy()
    spoiled._data[:, int(100 * SFREQ):int(130 * SFREQ)] = 1.0
    a, b = _glm(raw)[1], _glm(spoiled)[1]
    for ch in raw.ch_names:
        np.testing.assert_allclose(a.data[ch].theta, b.data[ch].theta, rtol=1e-9, atol=1e-18)


def test_a_condition_with_no_signal_in_kept_time_is_dropped_and_named():
    raw = _haemo(onsets=[20.0, 120.0, 200.0, 180.0], durations=[0.0, 0.0, 0.0, 120.0],
                 descs=["tap", "tap", "rest", UNSELECTED])
    _, est, dm, resid = _glm(raw)
    assert "rest" not in dm.columns and "tap" in dm.columns
    params = lineage_of(resid).params
    assert params["dropped_conditions"] == ["rest"]
    assert params["conditions"] == ["tap"]
    with pytest.raises(StageError, match=r"weights \['rest'\], dropped from the design"):
        _glm(raw, contrast_def={"rest_minus_tap": {"rest": 1, "tap": -1}})


def test_too_few_kept_frames_are_refused():
    raw = _haemo(onsets=[20.0, 0.5], durations=[0.0, 299.0], descs=["tap", UNSELECTED])
    with pytest.raises(StageError, match=r"frames lie outside its BAD_ spans .* at least 3 frames "
                                         r"per regressor"):
        _glm(raw)


def test_the_saved_design_flags_the_rows_it_left_out(tmp_path):
    raw = _haemo(onsets=[20.0, 120.0, 100.0], durations=[0.0, 0.0, 30.0],
                 descs=["tap", "tap", "BAD_gvtd"])
    nirs = tmp_path / "sub-01" / "nirs"
    source = nirs / "sub-01_task-tapping_desc-preproc_nirs.snirf"
    _glm(raw, output_dir=str(nirs), source_path=str(source))
    design = pd.read_csv(nirs / "sub-01_task-tapping_design.tsv", sep="\t")
    assert len(design) == raw.n_times
    assert design["censored"].tolist() == (~kept_frames(raw)).astype(int).tolist()
    sidecar = json.loads((nirs / "sub-01_task-tapping_design.json").read_text("utf-8"))
    assert sidecar["parameters"]["kept_frames"] == raw.n_times - 150


# ---- the whole run ----

def test_a_glm_run_with_kept_stretches_through_the_cli_and_its_script(tmp_path):
    bids, _ = make_fingerprint_dataset(tmp_path)
    table = tmp_path / "keep.tsv"
    table.write_text("onset\tduration\n0\t200\n", encoding="utf-8")
    out, script_out = tmp_path / "out", tmp_path / "script"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "glm",
                  "--stim-dur", f"{EVENT_DURATION:g}", "--drift-model", "polynomial",
                  "--noise-model", "ar1", "--keep-spans", str(table),
                  "--no-report", "--skip-bids-validation"])
    (errts,) = out.rglob("*desc-errts_nirs.json")
    params = json.loads(errts.read_text(encoding="utf-8"))["parameters"]
    assert params["censored_unselected_frac"] == pytest.approx(0.5)
    assert [v["left"] for v in params["events_left"].values()] == [3]

    script = (out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    (tmp_path / "script.py").write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(tmp_path / "script.py")], check=True, cwd=tmp_path)
    (script_errts,) = script_out.rglob("*desc-errts_nirs.snirf")
    np.testing.assert_array_equal(read_snirf(script_errts).get_data(),
                                  read_snirf(errts.with_suffix(".snirf")).get_data())
