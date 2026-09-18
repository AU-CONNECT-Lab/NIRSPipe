"""A cosine drift cutoff is bracketed: at least the data's high-pass, below the task's rhythm."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline.post_pipeline import (
    PostConfig,
    _repeat_intervals,
    _warn_drift_absorbs_task,
    _warn_lowpass_breaks_whitening,
    _warn_unmatched_design_band,
)


def _raw_with(descriptions, onsets):
    info = mne.create_info(["S1_D1 hbo"], sfreq=10.0, ch_types="hbo")
    n = int((max(onsets) + 60) * 10)          # long enough to hold every annotation
    raw = mne.io.RawArray(np.zeros((1, n)), info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[1.0] * len(onsets),
                                        description=descriptions))
    return raw


def _config(**kw):
    base = dict(subject="01", drift_model="cosine", drift_high_pass=0.01,
                cardiac_l_freq=0.7, cardiac_h_freq=1.5,
                resp_l_freq=0.1, resp_h_freq=0.5)
    return PostConfig(**{**base, **kw})


# ---- the design side ----

def test_the_longest_gap_is_measured_per_condition():
    raw = _raw_with(["a", "a", "b", "b"], [0.0, 30.0, 0.0, 120.0])
    assert _repeat_intervals(raw) == {"a": pytest.approx(30.0), "b": pytest.approx(120.0)}


def test_rejected_time_is_not_a_condition():
    raw = _raw_with(["a", "a", "BAD_gvtd", "BAD_gvtd"], [0.0, 30.0, 0.0, 200.0])
    assert set(_repeat_intervals(raw)) == {"a"}


def test_a_condition_seen_once_has_no_interval():
    assert _repeat_intervals(_raw_with(["a"], [0.0])) == {}


def test_the_warning_names_the_conditions_it_spans(caplog):
    """A block marker and a stimulus marker in one file have different rhythms."""
    raw = _raw_with(["block", "block", "stim", "stim"], [0.0, 400.0, 0.0, 10.0])
    with caplog.at_level("WARNING"):
        _warn_drift_absorbs_task(_config(drift_high_pass=0.01), raw)
    assert "block" in caplog.text
    assert "stim" not in caplog.text
    assert "1 of 2 conditions" in caplog.text


def test_a_cutoff_above_the_task_rhythm_warns(caplog):
    raw = _raw_with(["a", "a"], [0.0, 120.0])          # task rhythm 1/120 = 0.0083 Hz
    with caplog.at_level("WARNING"):
        _warn_drift_absorbs_task(_config(drift_high_pass=0.01), raw)
    assert "absorb" in caplog.text


def test_a_cutoff_below_the_task_rhythm_is_quiet(caplog):
    raw = _raw_with(["a", "a"], [0.0, 30.0])           # task rhythm 1/30 = 0.033 Hz
    with caplog.at_level("WARNING"):
        _warn_drift_absorbs_task(_config(drift_high_pass=0.0167), raw)
    assert caplog.text == ""


def test_only_a_cosine_basis_is_bounded_this_way(caplog):
    raw = _raw_with(["a", "a"], [0.0, 120.0])
    with caplog.at_level("WARNING"):
        _warn_drift_absorbs_task(_config(drift_model="polynomial"), raw)
    assert caplog.text == ""


# ---- the data side, which was already checked ----

def test_a_cutoff_below_the_data_high_pass_warns(caplog):
    with caplog.at_level("WARNING"):
        _warn_unmatched_design_band(_config(high_pass=0.02, drift_high_pass=0.01))
    assert "underestimated" in caplog.text


def test_matching_the_data_high_pass_is_quiet(caplog):
    with caplog.at_level("WARNING"):
        _warn_unmatched_design_band(_config(high_pass=0.01, drift_high_pass=0.01))
    assert caplog.text == ""


def test_no_data_high_pass_leaves_nothing_to_match(caplog):
    """The default: the drift basis is the only detrend, so no cutoff has to be covered."""
    with caplog.at_level("WARNING"):
        _warn_unmatched_design_band(_config(high_pass=None, drift_high_pass=0.005))
    assert caplog.text == ""


def test_the_two_bounds_can_exclude_each_other(caplog):
    """A band too aggressive for the design: no cutoff satisfies both, and both say so."""
    raw = _raw_with(["a", "a"], [0.0, 120.0])          # task rhythm 0.0083 Hz
    config = _config(high_pass=0.01, drift_high_pass=0.01)
    with caplog.at_level("WARNING"):
        _warn_unmatched_design_band(config)            # quiet: 0.01 >= 0.01
        _warn_drift_absorbs_task(config, raw)          # warns: 0.01 >= 0.0083
    assert "absorb" in caplog.text


# ---- the high end: a low-pass and an AR noise model do not mix ----

def test_a_low_pass_under_an_ar_model_is_warned_about(caplog):
    """Measured consequence, so it is not a style note: whitening a low-passed series takes
    the AR coefficient to the stationarity boundary and leaves the residual as correlated as
    it started, which moves the t values and not the betas."""
    import logging

    with caplog.at_level(logging.WARNING):
        _warn_lowpass_breaks_whitening(_config(noise_model="ar1", low_pass=0.2))
    assert "--low-pass 0.2 Hz" in caplog.text
    assert "anti-conservative" in caplog.text


def test_resampling_counts_as_a_low_pass(caplog):
    """It anti-aliases at the new Nyquist, so it empties the same band under another name.
    The warning has to name the Nyquist, or a reader sees no low-pass in their command."""
    import logging

    with caplog.at_level(logging.WARNING):
        _warn_lowpass_breaks_whitening(_config(noise_model="ar1", resample_sfreq=2.0))
    assert "--resample-sfreq 2 Hz" in caplog.text and "1 Hz" in caplog.text


def test_ols_and_an_unfiltered_run_are_both_silent(caplog):
    """ols does not whiten, so there is nothing for the filter to corrupt, and it is what
    denoise and rest hard-code. An AR model on unfiltered data is the correct combination."""
    import logging

    with caplog.at_level(logging.WARNING):
        _warn_lowpass_breaks_whitening(_config(noise_model="ols", low_pass=0.2))
        _warn_lowpass_breaks_whitening(_config(noise_model="ar1"))
        _warn_lowpass_breaks_whitening(_config(noise_model=None, low_pass=0.2))
    assert "anti-conservative" not in caplog.text
