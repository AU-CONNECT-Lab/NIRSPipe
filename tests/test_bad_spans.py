"""`BAD_` spans are censoring marks, never conditions or task time, and no stage may lose them."""

from datetime import datetime, timezone

import mne
import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.pipeline.glm import run_glm_pipeline
from fnirs_pipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
from fnirs_pipe.pipeline.restingstate import compute_alff
from fnirs_pipe.qc.metrics.windowed import task_scope_windows
from fnirs_pipe.qc.subject.sqm_record import _condition_cnr
from fnirs_pipe.utils.lineage import lineage_of

from tests._synth import synth_raw

SFREQ = 10.0


def _hbo_raw(data, onsets=(), durations=(), descs=()):
    data = np.atleast_2d(data)
    names = [f"S{i}_D{i} hbo" for i in range(1, data.shape[0] + 1)]
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, "hbo"), verbose="error")
    # without a date mne reads a cropped copy's annotations from the crop, not the recording
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descs)))
    return raw


# ---- GLM events table ----

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="D1: rows read from --events-path are not filtered for BAD_ spans")
def test_a_bad_row_in_an_events_table_is_not_a_condition(tmp_path):
    raw = synth_raw("01", "tapping", duration=120.0, bad_pair=None, motion_onset=None)
    haemo = od_to_haemo(intensity_to_od(raw), [6.0])
    path = tmp_path / "events.tsv"
    pd.DataFrame({"onset": [30.0, 60.0], "duration": [5.0, 40.0],
                  "trial_type": ["tap", "BAD_gvtd"]}).to_csv(path, sep="\t", index=False)
    *_, dm, resid = run_glm_pipeline(
        haemo, stim_dur=None, hrf_model="spm", noise_model="ols", drift_model="polynomial",
        high_pass=None, drift_order=1, fir_delays=None, events_path=str(path))
    assert "BAD_gvtd" not in dm.columns
    assert lineage_of(resid).params["conditions"] == ["tap"]


# ---- screening scope ----

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="D2: every annotation of 20 s or more is a task block, BAD_ spans included")
def test_task_scope_leaves_out_a_long_bad_span():
    raw = _hbo_raw(np.zeros(int(SFREQ * 600)), [20.0, 400.0], [300.0, 60.0], ["rest", "BAD_gvtd"])
    assert task_scope_windows(raw) == [("rest", 20.0, 320.0)]


# ---- per-condition CNR ----

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="D3: a BAD_ span starting before the condition is dropped with the other annotations")
def test_a_bad_span_reaching_into_a_condition_still_rejects_its_epochs():
    rng = np.random.default_rng(0)
    # epochs run 5 s before to 15 s after each tap, so the span covers the first two
    haemo = _hbo_raw(rng.standard_normal((2, int(SFREQ * 200))) * 1e-6,
                     [50.0, 65.0, 85.0, 105.0, 125.0], [45.0, 0.0, 0.0, 0.0, 0.0],
                     ["BAD_gvtd", "tap", "tap", "tap", "tap"])
    assert _condition_cnr(haemo, 65.0, 125.0)["cnr_n_epochs"] == 2


# ---- ALFF ----

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="D4: nansum over an all-NaN spectrum is 0, so fALFF reads 0.0")
def test_an_all_nan_channel_has_no_falff():
    t = np.arange(int(SFREQ * 100)) / SFREQ
    data = np.vstack([np.sin(2 * np.pi * 0.05 * t), np.full(t.size, np.nan)])
    out = compute_alff(_hbo_raw(data), low_pass=0.08, high_pass=0.01)
    assert np.isnan(out["falff"].iloc[1])
