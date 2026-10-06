"""Recordings that are wrong in the ways real acquisition goes wrong.

Each case either runs to a sensible result or stops with an error that names the problem.
"""

import json
import logging
import warnings

import mne
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from fnirs_pipe.exceptions import AlignmentError, FilterDesignError, StageError
from fnirs_pipe.pipeline.denoise import filter_kwargs
from fnirs_pipe.pipeline.glm import build_design_matrix
from fnirs_pipe.pipeline.hyper.alignment import align_recordings
from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import (
    PrepConfig,
    intensity_to_od,
    mark_bad_channels,
    run_prep,
)
from fnirs_pipe.qc.metrics import compute_sci_scores, screening_scores

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)


def _prep(raw, tmp_path, **overrides):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8,
                        motion_correction="none", **_BANDS, **overrides)
    return run_prep(raw, config, tmp_path, source_entities={"task": "rest"})


def _with(raw, edit):
    data = raw.get_data()
    edit(data)
    out = mne.io.RawArray(data, raw.info, verbose="error")
    return out.set_annotations(raw.annotations)


# ---- prep on broken recordings ----

def test_an_all_zero_channel_is_rejected_in_both_wavelengths_and_leaves_the_rest_finite(tmp_path):
    raw = synth_raw("01", "rest", bad_pair=None)
    data = raw.get_data()
    data[0] = 0.0
    result = _prep(mne.io.RawArray(data, raw.info, verbose="error"), tmp_path)
    assert result.bad_channels == ["S1_D1 760", "S1_D1 850"]
    good = [c for c in result.raw_haemo.ch_names if c not in result.bad_channels]
    assert np.isfinite(result.raw_haemo.get_data(picks=good)).all()


def test_a_single_pair_montage_runs_and_keeps_its_pair(tmp_path):
    raw = synth_raw("01", "rest", n_long_pairs=1, short_channels=False, bad_pair=None)
    result = _prep(raw, tmp_path)
    assert result.bad_channels == []
    assert len(result.raw_haemo.ch_names) == 2


def test_a_recording_two_screening_windows_long_still_screens(tmp_path):
    raw = synth_raw("01", "rest", duration=20.0, motion_onset=None)
    assert _prep(raw, tmp_path).bad_channels == ["S3_D3 760", "S3_D3 850"]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_a_non_finite_stretch_rejects_its_pair_and_no_longer_reaches_the_glm(tmp_path, value):
    # before the fix prep passed the NaN through in a rejected channel and post died in KMeans
    def spoil(data):
        data[0, 100:200] = value

    result = _prep(_with(synth_raw("01", "rest", bad_pair=None), spoil), tmp_path / "prep")
    assert result.bad_channels == ["S1_D1 760", "S1_D1 850"]
    assert np.isfinite(result.raw_haemo.get_data()).all()

    sidecar = next((tmp_path / "prep").rglob("*desc-od_nirs.json"))
    assert json.loads(sidecar.read_text())["non_finite_samples"]["S1_D1 760"] > 0

    preproc = next((tmp_path / "prep").rglob("*desc-preproc_nirs.snirf"))
    post = PostConfig(subject="01", high_pass=0.01, drift_model="polynomial", drift_order=1,
                      short_channel="mean", **_BANDS)
    # into the tree prep wrote, as the command does, so the stage it read is its own source
    residual = run_post(result.raw_haemo.copy(), post, tmp_path / "prep", mode="denoise",
                        source_entities={"task": "rest"}, source_path=preproc)[0]
    good = [c for c in residual.ch_names if c not in residual.info["bads"]]
    assert np.isfinite(residual.get_data(picks=good)).all()


def test_a_finite_recording_records_no_non_finite_samples(tmp_path):
    _prep(synth_raw("01", "rest"), tmp_path)
    sidecar = next(tmp_path.rglob("*desc-od_nirs.json"))
    assert "non_finite_samples" not in json.loads(sidecar.read_text())


def test_non_finite_and_manual_marks_that_leave_nothing_name_both(tmp_path):
    def spoil(data):
        data[0, 100:200] = np.nan

    raw = _with(synth_raw("01", "rest", bad_pair=None), spoil)
    rest = sorted({c.rsplit(" ", 1)[0] for c in raw.ch_names} - {"S1_D1"})
    with pytest.raises(StageError, match="--bad-channels and non-finite samples leave no usable"):
        _prep(raw, tmp_path, bad_channels=rest)


def test_manual_marks_alone_that_leave_nothing_still_say_so(tmp_path):
    raw = synth_raw("01", "rest", bad_pair=None)
    every = sorted({c.rsplit(" ", 1)[0] for c in raw.ch_names})
    with pytest.raises(StageError, match="--bad-channels leaves no usable channel"):
        _prep(raw, tmp_path, bad_channels=every)


_ABOVE_NYQUIST = r"--cardiac-h-freq 1.5 Hz .* Nyquist frequency \(1 Hz\)"


@pytest.fixture(scope="module")
def slow_intensity():
    return synth_raw("01", "rest").resample(2.0)


def test_a_cardiac_band_above_nyquist_is_refused_naming_the_flag(slow_intensity):
    od = intensity_to_od(slow_intensity.copy())
    with pytest.raises(ValueError, match=_ABOVE_NYQUIST):
        mark_bad_channels(od, 0.8, _BANDS["cardiac_l_freq"], _BANDS["cardiac_h_freq"])


def test_the_qc_sci_refuses_it_instead_of_scoring_every_channel_one(slow_intensity):
    with pytest.raises(ValueError, match=_ABOVE_NYQUIST):
        compute_sci_scores(slow_intensity.copy(), _BANDS["cardiac_l_freq"], _BANDS["cardiac_h_freq"])


def test_screening_refuses_it_instead_of_screening_nothing(slow_intensity):
    # SCI handed in, so only the window criterion is left for the catch-all to swallow
    od = intensity_to_od(slow_intensity.copy())
    have = {"sci": {ch: 1.0 for ch in od.ch_names}}
    with pytest.raises(ValueError, match=_ABOVE_NYQUIST):
        screening_scores(od, _BANDS["cardiac_l_freq"], _BANDS["cardiac_h_freq"], have=have)


# ---- a filter longer than the recording ----

def test_a_fir_longer_than_the_recording_is_refused_and_points_at_iir():
    with pytest.raises(FilterDesignError, match="iir"):
        filter_kwargs(10.0, 600, 0.01, 0.2, method="fir")


def test_the_iir_design_has_no_length_to_run_out_of():
    assert filter_kwargs(10.0, 600, 0.01, 0.2, method="iir")["method"] == "iir"


# ---- events ----

def _tapping_events(raw):
    return pd.DataFrame({"trial_type": list(raw.annotations.description),
                         "onset": raw.annotations.onset, "duration": 5.0})


_DESIGN = dict(stim_dur=None, hrf_model="spm", drift_model="polynomial",
               high_pass=None, drift_order=1)


def test_event_order_does_not_change_the_design():
    raw = synth_raw("01", "tapping")
    events = _tapping_events(raw)
    ordered = build_design_matrix(raw, events=events, **_DESIGN)
    shuffled = build_design_matrix(raw, events=events.iloc[::-1].reset_index(drop=True), **_DESIGN)
    pd.testing.assert_frame_equal(ordered, shuffled)


def test_a_repeated_trigger_is_dropped_rather_than_doubling_its_block(caplog):
    raw = synth_raw("01", "tapping")
    events = _tapping_events(raw)
    doubled = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)       # nilearn's summing warning must not fire
        design = build_design_matrix(raw, events=doubled, **_DESIGN)
    pd.testing.assert_frame_equal(design, build_design_matrix(raw, events=events, **_DESIGN))
    assert "dropped 1 duplicated event" in caplog.text


def test_two_events_at_one_onset_with_different_durations_are_both_kept():
    # not a duplicate by nilearn's own identity, so not ours to drop either
    raw = synth_raw("01", "tapping")
    events = _tapping_events(raw)
    longer = events.iloc[[0]].assign(duration=10.0)
    design = build_design_matrix(raw, events=pd.concat([events, longer], ignore_index=True),
                                 **_DESIGN)
    assert not design.equals(build_design_matrix(raw, events=events, **_DESIGN))


def test_glm_mode_without_events_is_refused_and_points_at_denoise(tmp_path):
    raw = synth_raw("01", "rest")
    haemo = _prep(raw, tmp_path / "prep").raw_haemo
    post = PostConfig(subject="01", hrf_model="spm", drift_model="polynomial", drift_order=1,
                      stim_dur=5.0, **_BANDS)
    with pytest.raises(ValueError, match="no events in the recording's annotations.*--mode denoise"):
        run_post(haemo, post, tmp_path / "post", mode="glm", source_entities={"task": "rest"})


# ---- hyperscanning alignment ----

def _pair(ann_a, ann_b, dur_a=400.0, dur_b=400.0):
    a = synth_raw("01", "rest", duration=dur_a)
    b = synth_raw("02", "rest", duration=dur_b)
    a.set_annotations(mne.Annotations(*ann_a))
    b.set_annotations(mne.Annotations(*ann_b))
    return {"01": a, "02": b}


def test_a_pair_with_no_triggers_is_refused():
    with pytest.raises(AlignmentError, match="No non-BAD annotations"):
        align_recordings(_pair(([], [], []), ([], [], [])), "rest")


def test_motion_marks_alone_are_not_triggers():
    bad = ([30.0], [2.0], ["BAD_motion"])
    with pytest.raises(AlignmentError, match="No non-BAD annotations"):
        align_recordings(_pair(bad, bad), "rest")


def test_a_pair_whose_triggers_share_no_name_is_refused_and_lists_both():
    with pytest.raises(AlignmentError, match=r"No shared trigger.*'x'.*'y'"):
        align_recordings(_pair(([5.0], [0.0], ["x"]), ([5.0], [0.0], ["y"])), "rest")


def test_members_of_different_length_come_out_cut_to_the_shorter_one():
    pair = _pair(([10.0], [0.0], ["start"]), ([20.0], [0.0], ["start"]), dur_b=300.0)
    aligned, offsets = align_recordings(pair, "rest")
    assert offsets == {"01": 10.0, "02": 20.0}
    n_times = [raw.n_times for raw in aligned.values()]
    assert n_times[0] == n_times[1]
    assert_array_equal(aligned["01"].times, aligned["02"].times)
    assert aligned["02"].times[-1] <= 300.0 - 20.0
