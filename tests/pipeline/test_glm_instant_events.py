"""A condition whose events last 0 s is fitted as an instantaneous response peaking at 1."""
import json

import mne
import numpy as np
import pandas as pd
import pytest
from nilearn.glm.first_level import make_first_level_design_matrix
from scipy.stats import gamma

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _refuse_glm_without_durations
from nirspipe.pipeline.glm import build_design_matrix, instant_conditions
from nirspipe.qc.subject.report import _design_notes

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

AMP = 2.5
DURATION = 300.0


def _events(names, onsets, durations):
    return pd.DataFrame({"trial_type": names, "onset": onsets, "duration": durations})


def _raw(sfreq):
    info = mne.create_info(["S1_D1 hbo"], sfreq=sfreq, ch_types="hbo")
    return mne.io.RawArray(np.zeros((1, int(DURATION * sfreq))), info, verbose="ERROR")


# ---- which conditions, and which stop ----

def test_the_zero_duration_conditions_are_named():
    events = _events(["tap", "rest", "rest"], [10.0, 40.0, 70.0], [5.0, 0.0, 0.0])
    assert instant_conditions(events) == ["rest"]


def test_a_condition_mixing_zero_and_longer_events_stops():
    with pytest.raises(ValueError, match="tap mix 0 s events"):
        instant_conditions(_events(["tap", "tap"], [10.0, 40.0], [0.0, 5.0]))


def test_a_negative_duration_stops():
    with pytest.raises(ValueError, match="negative"):
        instant_conditions(_events(["tap"], [10.0], [-1.0]))


def test_stim_dur_zero_is_accepted_and_a_negative_one_refused():
    _refuse_glm_without_durations({"mode": "glm", "stim_dur": 0.0})
    with pytest.raises(SystemExit, match="negative"):
        _refuse_glm_without_durations({"mode": "glm", "stim_dur": -1.0})


# ---- the beta is the response's peak, at any rate ----

def _canonical(t):
    # SPM's canonical response (gamma peak shape 6, undershoot shape 16, ratio 1/6), peak 1
    def h(x):
        return np.where(x > 0, gamma.pdf(x, 6) - gamma.pdf(x, 16) / 6, 0.0)
    return h(t) / h(np.arange(0, 32, 0.001)).max()


def _beta(sfreq, onsets):
    times = np.arange(int(DURATION * sfreq)) / sfreq
    signal = AMP * sum(_canonical(times - on) for on in onsets)
    dm = build_design_matrix(_raw(sfreq), None, "spm", "none", None, None,
                             events=_events(["a"] * len(onsets), onsets, [0.0] * len(onsets)))
    coef, *_ = np.linalg.lstsq(dm.to_numpy(), signal, rcond=None)
    return coef[list(dm.columns).index("a")]


@pytest.mark.parametrize("sfreq", [5.0, 10.0, 25.0])
def test_the_beta_is_the_planted_peak(sfreq):
    onsets = [20.3, 51.7, 83.1, 114.9, 146.2, 177.6, 209.4, 240.8]
    assert _beta(sfreq, onsets) == pytest.approx(AMP, rel=0.02)


@pytest.mark.parametrize("sfreq", [5.0, 25.0])
def test_overlapping_events_add_up_rather_than_set_the_scale(sfreq):
    # 3 s apart, so the responses pile up well past one event's peak
    onsets = [30.1 + 3.0 * k for k in range(8)] + [150.4 + 3.0 * k for k in range(8)]
    assert _beta(sfreq, onsets) == pytest.approx(AMP, rel=0.02)


def test_each_fir_delay_is_the_response_at_that_delay():
    sfreq = 5.0
    times = np.arange(int(DURATION * sfreq)) / sfreq
    onsets = [20.0, 60.0, 100.0, 140.0, 180.0, 220.0]
    signal = np.zeros_like(times)
    for on in onsets:
        signal[int(round(on * sfreq)) + 2] = AMP
    dm = build_design_matrix(_raw(sfreq), None, "fir", "none", None, None, fir_delays=(0, 1, 2, 3),
                             events=_events(["a"] * 6, onsets, [0.0] * 6))
    coef, *_ = np.linalg.lstsq(dm.to_numpy(), signal, rcond=None)
    fitted = dict(zip(dm.columns, coef))
    assert fitted["a_delay_2"] == pytest.approx(AMP)
    assert [fitted[f"a_delay_{d}"] for d in (0, 1, 3)] == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_a_condition_with_durations_keeps_nilearns_columns():
    raw = _raw(10.0)
    events = _events(["a", "b", "a", "b"], [20.0, 50.0, 120.0, 150.0], [0.0, 10.0, 0.0, 10.0])
    ours = build_design_matrix(raw, None, "spm", "none", None, None, events=events)
    theirs = make_first_level_design_matrix(raw.times, events, hrf_model="spm", drift_model=None)
    pd.testing.assert_series_equal(ours["b"], theirs["b"])
    assert ours["a"].max() == pytest.approx(1.0, rel=0.01)


# ---- a run end to end ----

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("instant")
    bids, _ = make_fingerprint_dataset(root)
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "glm",
                  "--stim-dur", "0", "--noise-model", "ols", "--drift-high-pass", "0.01",
                  "--participant-label", "01", "--skip-bids-validation", "--no-report"])
    return out


def test_the_run_records_its_instant_conditions(run):
    sidecar = next(run.rglob("*desc-glm_nirsmap.json"))
    params = json.loads(sidecar.read_text(encoding="utf-8"))["parameters"]
    assert params["stim_dur"] == 0
    assert params["instant_conditions"] == params["conditions"]


def test_the_report_says_what_the_betas_mean(run):
    sidecar = next(run.rglob("*desc-glm_nirsmap.json"))
    params = json.loads(sidecar.read_text(encoding="utf-8"))["parameters"]
    notes = _design_notes(sidecar.parent, sidecar.name.split("_desc-")[0], "01")
    assert notes == ["Events lasting 0 s, fitted as an instantaneous response scaled to a peak "
                     "of 1, so each beta is the height of the response's peak: "
                     + ", ".join(params["instant_conditions"]) + "."]
