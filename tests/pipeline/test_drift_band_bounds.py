"""A cosine drift cutoff is bracketed: at least the data's high-pass, below the task's rhythm."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
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


# ---- the data side ----

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
    """Whitening a low-passed series takes the AR coefficient to the stationarity boundary
    and leaves the residual as correlated as it started, which moves the t values and not the
    betas."""
    import logging

    with caplog.at_level(logging.WARNING):
        _warn_lowpass_breaks_whitening(_config(noise_model="ar1", low_pass=0.2), "glm")
    assert "--low-pass 0.2 Hz" in caplog.text
    assert "anti-conservative" in caplog.text


def test_the_modes_whose_product_is_the_residual_are_warned_too(caplog):
    """denoise and rest fit the requested noise model too, so the same combination moves the
    file they exist to write. The warning has to name that, not the t values they do not compute:
    a reader whose coherence changed needs to be told the residual did."""
    import logging

    for mode in ("denoise", "rest"):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _warn_lowpass_breaks_whitening(_config(noise_model="auto", low_pass=0.2), mode)
        assert "desc-errts" in caplog.text
        assert "anti-conservative" not in caplog.text


def test_resampling_counts_as_a_low_pass(caplog):
    """It anti-aliases at the new Nyquist, so it empties the same band under another name.
    The warning has to name the Nyquist, or a reader sees no low-pass in their command."""
    import logging

    with caplog.at_level(logging.WARNING):
        _warn_lowpass_breaks_whitening(_config(noise_model="ar1", resample_sfreq=2.0), "glm")
    assert "--resample-sfreq 2 Hz" in caplog.text and "1 Hz" in caplog.text


def test_ols_and_an_unfiltered_run_are_both_silent(caplog):
    """ols does not whiten, so there is nothing for the filter to corrupt. An AR model on
    unfiltered data is the correct combination."""
    import logging

    with caplog.at_level(logging.WARNING):
        _warn_lowpass_breaks_whitening(_config(noise_model="ols", low_pass=0.2), "glm")
        _warn_lowpass_breaks_whitening(_config(noise_model="ar1"), "glm")
        _warn_lowpass_breaks_whitening(_config(noise_model=None, low_pass=0.2), "denoise")
    assert "fit the filter rather than the noise" not in caplog.text


# ---- the noise model a caller lands on by default ----

def test_the_cli_takes_any_order_the_library_takes():
    """The CLI validates by rule rather than against a closed list. The suggestion list still
    has to be acceptable to the validator, or the GUI dropdown would offer a value the CLI
    rejects."""
    import argparse

    import pytest as _pytest

    from fnirs_pipe.cli.run import _NOISE_CHOICES, _noise_model
    from fnirs_pipe.pipeline.glm import NoiseModel

    for value in set(_NOISE_CHOICES) | set(NoiseModel.__args__):
        assert _noise_model(value) == value
    for value in ("ar16", "ar31", "ar100"):
        assert _noise_model(value) == value
    for value in ("ar0", "ar", "banana", "AR16", "ar1.5", ""):
        with _pytest.raises(argparse.ArgumentTypeError):
            _noise_model(value)


def test_the_default_does_not_shadow_a_config_file():
    """`pick` takes the CLI value when it is not None, so a default declared in argparse
    would make the TOML unreachable. It belongs in the `pick` call, as every other defaulted
    field here has it."""
    import inspect

    from fnirs_pipe.cli import workflows
    from fnirs_pipe.cli.run import _build_parser

    assert 'noise_model=pick("noise_model", default="auto")' in inspect.getsource(workflows)
    action = next(a for a in _build_parser()._actions
                  if "--noise-model" in getattr(a, "option_strings", []))
    assert action.default is None, "a default here would make the TOML unreachable"



def test_every_mode_fits_the_model_that_was_asked_for():
    """denoise and rest pass `--noise-model` through to their fit rather than accepting it
    and dropping it."""
    import inspect

    from fnirs_pipe.pipeline import post_pipeline

    src = inspect.getsource(post_pipeline.run_post)
    assert 'noise_model="ols"' not in src
    assert src.count("noise_model=config.noise_model") >= 3


def test_the_gui_field_takes_what_the_cli_takes():
    """The Analysis page's field is free text with a suggestion list, and its `pattern` is
    the CLI's rule.

    The rule is written once and the page imports it, so what is left to pin is that the
    page still reaches for it: a `pattern=` spelled out again here would be the second copy
    this test exists to prevent. A user typing `ar16` into the page and having the browser
    refuse it, or the page accepting something the CLI then rejects, are both silent.
    """
    import argparse
    import re

    from pathlib import Path

    from fnirs_pipe.cli.run import NOISE_MODEL_PATTERN, _noise_model

    src = Path("fnirs_pipe/interface/pages/analysis.py").read_text(encoding="utf-8")
    assert 'id="an-noise-model"' in src and 'value="auto"' in src, "the page must default to auto"
    assert "pattern=NOISE_MODEL_PATTERN" in src, "the page must use the CLI's rule, not its own copy"
    assert 'pattern="' not in src, "a literal pattern here is the second copy of the rule"
    pattern = NOISE_MODEL_PATTERN

    for value in ("auto", "ols", "ar1", "ar5", "ar16", "ar31", "ar100", "ar_irls", "ar_irls40",
                  "ar0", "banana", "ar", "AR16", "ar1.5", "", "ar_irls0", "irls"):
        browser = bool(re.fullmatch(pattern, value))
        try:
            _noise_model(value)
            cli = True
        except argparse.ArgumentTypeError:
            cli = False
        assert browser == cli, f"{value!r}: page says {browser}, CLI says {cli}"


# ---- which events the warning reads ----

def test_an_events_table_overrides_the_annotations():
    """--events-path is what the design is built from, so it is what the warning measures."""
    raw = _raw_with(["15.0", "15.0", "a", "a"], [0.0, 2900.0, 0.0, 30.0])
    events = pd.DataFrame({"trial_type": ["a", "a"], "onset": [0.0, 30.0],
                           "duration": [5.0, 5.0]})
    assert _repeat_intervals(raw, events) == {"a": pytest.approx(30.0)}


def test_the_warning_ignores_a_marker_the_design_leaves_out(caplog):
    raw = _raw_with(["15.0", "15.0", "a", "a"], [0.0, 2900.0, 0.0, 30.0])
    events = pd.DataFrame({"trial_type": ["a", "a"], "onset": [0.0, 30.0],
                           "duration": [5.0, 5.0]})
    with caplog.at_level("WARNING"):
        _warn_drift_absorbs_task(_config(drift_high_pass=0.001), raw, events)
    assert "15.0" not in caplog.text
    assert caplog.text == ""
