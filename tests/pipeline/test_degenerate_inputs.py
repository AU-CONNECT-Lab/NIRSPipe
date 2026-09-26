"""Recordings that are wrong in the ways real acquisition goes wrong.

Each case either runs to a sensible result or stops with an error that names the problem.
Cases that currently do neither are not here; they are findings, recorded in the handoff.
"""

import mne
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from fnirs_pipe.exceptions import AlignmentError, FilterDesignError
from fnirs_pipe.pipeline.denoise import filter_kwargs
from fnirs_pipe.pipeline.glm import build_design_matrix
from fnirs_pipe.pipeline.hyper.alignment import align_recordings
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)


def _prep(raw, tmp_path):
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8,
                        motion_correction="none", **_BANDS)
    return run_prep(raw, config, tmp_path, source_entities={"task": "rest"})


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


# ---- a filter longer than the recording ----

def test_a_fir_longer_than_the_recording_is_refused_and_points_at_iir():
    with pytest.raises(FilterDesignError, match="iir"):
        filter_kwargs(10.0, 600, 0.01, 0.2, method="fir")


def test_the_iir_design_has_no_length_to_run_out_of():
    assert filter_kwargs(10.0, 600, 0.01, 0.2, method="iir")["method"] == "iir"


# ---- events out of order ----

def test_event_order_does_not_change_the_design():
    raw = synth_raw("01", "tapping")
    events = pd.DataFrame({"trial_type": list(raw.annotations.description),
                           "onset": raw.annotations.onset,
                           "duration": 5.0})
    kwargs = dict(stim_dur=None, hrf_model="spm", drift_model="polynomial",
                  high_pass=None, drift_order=1)
    ordered = build_design_matrix(raw, events=events, **kwargs)
    shuffled = build_design_matrix(raw, events=events.iloc[::-1].reset_index(drop=True), **kwargs)
    pd.testing.assert_frame_equal(ordered, shuffled)


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
