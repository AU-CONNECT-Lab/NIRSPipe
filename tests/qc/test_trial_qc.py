"""Which trial windows the per-trial panel scores, and which it leaves blank."""

from nirspipe.qc.subject.trial_qc import (
    MIN_SCI_TRIAL_S, MIN_TRIAL_S, SHORT_TRIAL_KEYS, score_trials, short_trial_counts,
    trial_fits, trial_windows,
)
from tests._synth import synth_raw


def test_a_trial_shorter_than_one_window_keeps_its_label_and_is_left_unscored():
    raw = synth_raw("01", "tapping", duration=120.0, motion_onset=None)
    markers = [{"onset": 20.0, "duration": 1.0, "description": "tap"},
               {"onset": 60.0, "duration": MIN_TRIAL_S, "description": "tap"}]
    labels, sqms = score_trials(raw, markers, 0.8, 0.7, 2.0)
    assert len(labels) == 2
    assert sqms[0] == {}
    assert "sci_mean" in sqms[1] and "cv_mean" in sqms[1]


def test_a_trial_between_the_two_lines_is_scored_on_its_window_free_rows_only():
    raw = synth_raw("01", "tapping", duration=120.0, motion_onset=None)
    markers = [{"onset": 20.0, "duration": MIN_SCI_TRIAL_S, "description": "tap"}]
    _, (sqm,) = score_trials(raw, markers, 0.8, 0.7, 2.0)
    assert set(sqm) == set(SHORT_TRIAL_KEYS)
    assert 0.0 < sqm["sci_mean"] <= 1.0


def test_the_counts_split_the_partly_scored_from_the_unscored():
    markers = [{"onset": 10.0 * i, "duration": d, "description": "tap"}
               for i, d in enumerate((1.0, MIN_SCI_TRIAL_S, 8.0, MIN_TRIAL_S), start=1)]
    windows = trial_windows(markers, None, None, 120.0)
    assert short_trial_counts(windows, 10.0) == (2, 1)


def test_a_window_is_counted_the_way_mne_crops_it():
    assert trial_fits(10.0, 30.0, 30.0 + MIN_TRIAL_S)
    assert not trial_fits(10.0, 30.0, 30.0 + MIN_TRIAL_S - 0.2)
    # a 1 s event on a 7.8 Hz recording is nine samples
    assert not trial_fits(7.8125, 33.408, 34.408)
    assert trial_fits(10.0, 30.0, 30.0 + MIN_SCI_TRIAL_S, MIN_SCI_TRIAL_S)
