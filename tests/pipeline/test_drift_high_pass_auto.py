"""`--drift-high-pass auto`: 1/(2 x the slowest condition repeat) per run, capped, never below the high-pass."""

import argparse
import json
import math
import tomllib

import mne
import numpy as np
import pandas as pd
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.run import _drift_cutoff
from nirspipe.cli.workflows import _refuse_drift_auto_outside_glm
from nirspipe.interface.hints import band_note
from nirspipe.io.tables import read_table
from nirspipe.pipeline.post_pipeline import PostConfig, resolve_drift_high_pass, run_post
from nirspipe.qc.boilerplate.vocabulary import _drift_phrase
from nirspipe.qc.subject.report import _drift_auto_notes

from tests._fingerprint import CLI_ARGS, DURATION, EVENT_DURATION, make_fingerprint_dataset


def _raw_with(descriptions, onsets):
    info = mne.create_info(["S1_D1 hbo"], sfreq=10.0, ch_types="hbo")
    raw = mne.io.RawArray(np.zeros((1, int((max(onsets) + 60) * 10))), info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[1.0] * len(onsets),
                                        description=descriptions))
    return raw


def _config(**kw):
    base = dict(subject="01", drift_model="cosine", drift_high_pass="auto",
                cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.5)
    return PostConfig(**{**base, **kw})


# ---- the rule ----

def test_the_slowest_condition_sets_the_cutoff():
    # a: gaps 100, 50; b: gaps 30, 30. The slowest is a's 100 s, so 1/200 Hz
    raw = _raw_with(["a", "a", "a", "b", "b", "b"], [0.0, 100.0, 150.0, 0.0, 30.0, 60.0])
    config = resolve_drift_high_pass(_config(), raw)
    assert config.drift_high_pass == pytest.approx(0.005)
    assert config.drift_high_pass_auto == {"condition": "a", "interval_s": pytest.approx(100.0),
                                           "trials": 3, "bound": None, "seen_once": []}


def test_a_fast_design_is_capped_at_a_hundredth_of_a_hertz():
    raw = _raw_with(["a"] * 4, [0.0, 10.0, 20.0, 30.0])     # 1/(2 x 10) = 0.05 Hz
    config = resolve_drift_high_pass(_config(), raw)
    assert config.drift_high_pass == pytest.approx(0.01)
    assert config.drift_high_pass_auto["bound"] == "cap"


def test_the_cutoff_is_raised_to_the_data_high_pass():
    raw = _raw_with(["a", "a", "a"], [0.0, 100.0, 150.0])   # 0.005 Hz, under the 0.01 filter
    config = resolve_drift_high_pass(_config(high_pass=0.01), raw)
    assert config.drift_high_pass == pytest.approx(0.01)
    assert config.drift_high_pass_auto["bound"] == "high_pass"


def test_the_high_pass_outranks_the_cap():
    raw = _raw_with(["a"] * 4, [0.0, 10.0, 20.0, 30.0])     # 0.05, capped to 0.01, then 0.02
    config = resolve_drift_high_pass(_config(high_pass=0.02), raw)
    assert config.drift_high_pass == pytest.approx(0.02)
    assert config.drift_high_pass_auto["bound"] == "high_pass"


def test_a_typed_cutoff_is_left_alone():
    config = _config(drift_high_pass=0.008)
    assert resolve_drift_high_pass(config, _raw_with(["a", "a"], [0.0, 10.0])) is config


def test_the_events_table_is_read_in_place_of_the_annotations():
    raw = _raw_with(["a", "a"], [0.0, 30.0])
    events = pd.DataFrame({"trial_type": ["x", "x", "x", "BAD_x", "BAD_x"],
                           "onset": [0.0, 100.0, 200.0, 0.0, 500.0],
                           "duration": [5.0] * 5})
    config = resolve_drift_high_pass(_config(), raw, events)
    assert config.drift_high_pass == pytest.approx(0.005)
    assert config.drift_high_pass_auto["condition"] == "x"


# ---- warned, not refused ----

def test_a_deciding_condition_seen_twice_is_warned_about(caplog):
    # start at 0 and 400 s sets 1/800 Hz; the task repeats every 20 s
    raw = _raw_with(["start", "start", "tap", "tap", "tap"], [0.0, 400.0, 10.0, 30.0, 50.0])
    with caplog.at_level("WARNING"):
        config = resolve_drift_high_pass(_config(), raw)
    assert config.drift_high_pass == pytest.approx(1 / 800)
    assert "start, which occurs only twice, 400 s apart" in caplog.text


def test_three_trials_are_not_warned_about(caplog):
    raw = _raw_with(["a", "a", "a"], [0.0, 100.0, 150.0])
    with caplog.at_level("WARNING"):
        resolve_drift_high_pass(_config(), raw)
    assert caplog.text == ""


def test_a_condition_seen_once_is_skipped_and_named(caplog):
    # c's two annotations sit on one onset, a doubled marker rather than a repeat
    raw = _raw_with(["a", "a", "a", "b", "c", "c"], [0.0, 60.0, 90.0, 50.0, 20.0, 20.0])
    with caplog.at_level("WARNING"):
        config = resolve_drift_high_pass(_config(), raw)
    assert config.drift_high_pass == pytest.approx(1 / 120)
    assert config.drift_high_pass_auto["seen_once"] == ["b", "c"]
    assert "b, c occur(s) once" in caplog.text


def test_a_run_where_nothing_repeats_has_no_cutoff_to_compute():
    with pytest.raises(ValueError, match="no condition occurs twice"):
        resolve_drift_high_pass(_config(), _raw_with(["a", "b"], [0.0, 10.0]))


def test_only_glm_mode_reads_events_for_it(tmp_path):
    with pytest.raises(ValueError, match="only glm mode"):
        run_post(_raw_with(["a", "a"], [0.0, 10.0]), _config(), tmp_path, mode="rest")


def test_a_string_other_than_auto_is_refused():
    with pytest.raises(ValueError, match="expected a cutoff in Hz or 'auto'"):
        _config(drift_high_pass="0.01")


# ---- the command line and the form ----

def test_the_flag_takes_a_number_or_auto():
    assert _drift_cutoff(" AUTO ") == "auto"
    assert _drift_cutoff("0.008") == 0.008
    with pytest.raises(argparse.ArgumentTypeError):
        _drift_cutoff("slow")


def test_auto_outside_glm_stops_before_any_subject():
    with pytest.raises(SystemExit, match="only --mode glm"):
        _refuse_drift_auto_outside_glm({"mode": "rest", "drift_high_pass": "auto"})
    _refuse_drift_auto_outside_glm({"mode": "glm", "drift_high_pass": "auto"})
    _refuse_drift_auto_outside_glm({"mode": "rest", "drift_high_pass": 0.01})


def test_the_form_accepts_auto_in_glm_mode_only():
    assert band_note("glm", "0.01", "cosine", "auto") is None
    assert "needs the glm mode" in band_note("rest", "", "", "auto")
    assert "type auto" in band_note("glm", "", "cosine", "").lower()


# ---- the Methods ----

def _auto(bound=None):
    return {"condition": "tap", "interval_s": 60.0, "trials": 6, "bound": bound, "seen_once": []}


def test_the_methods_say_how_the_cutoff_was_set():
    assert _drift_phrase({"drift_model": "cosine", "drift_high_pass": 1 / 120,
                          "drift_high_pass_auto": _auto()}) == (
        "a discrete cosine drift basis (high-pass cutoff: 0.00833333 Hz, set from the run's "
        "events to 1/(2T), T being the longest interval between two trials of one condition "
        "(tap, 60 s))")
    assert _drift_phrase({"drift_model": "cosine", "drift_high_pass": 0.01,
                          "drift_high_pass_auto": _auto("cap")}).endswith(
        "(tap, 60 s), and capped at 0.01 Hz)")
    assert _drift_phrase({"drift_model": "cosine", "drift_high_pass": 0.02,
                          "drift_high_pass_auto": _auto("high_pass")}).endswith(
        "(tap, 60 s), and raised to the data's high-pass cutoff, which the basis has to cover)")


def test_a_typed_cutoff_reads_as_before():
    assert _drift_phrase({"drift_model": "cosine", "drift_high_pass": 0.01}) == (
        "a discrete cosine drift basis (high-pass cutoff: 0.01 Hz)")


# ---- a run end to end, and its rerun ----

QUIET = ["--participant-label", "01", "--skip-bids-validation", "--no-report"]
GLM = ["--mode", "glm", "--stim-dur", f"{EVENT_DURATION:g}", "--noise-model", "ols"]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The block dataset: ca and cb once each, the trials inside them every 60 s."""
    root = tmp_path_factory.mktemp("drift_auto")
    bids, truth = make_fingerprint_dataset(root, blocks=True)
    first, rerun = root / "first", root / "rerun"
    run_cli.main([str(bids), str(first), "participant", *CLI_ARGS, *GLM,
                  "--drift-high-pass", "auto", *QUIET])
    run_cli.main([str(bids), str(rerun), "participant",
                  "--config", str(first / "sub-01" / "logs" / "sub-01.toml"), *QUIET])
    return first, rerun, truth


def _glm_params(out):
    sidecar = next(out.rglob("*desc-glm_nirsmap.json"))
    return json.loads(sidecar.read_text(encoding="utf-8"))["parameters"]


def test_the_run_records_auto_and_its_sidecar_the_cutoff(runs):
    first, _, truth = runs
    record = tomllib.loads((first / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8"))
    assert record["post"]["drift_high_pass"] == "auto"
    trials = sorted(on for on, _, name in truth.events if name == "trial")
    gap = max(np.diff(trials))
    params = _glm_params(first)
    assert params["drift_high_pass"] == pytest.approx(1 / (2 * gap))
    assert params["drift_high_pass_auto"] == {"condition": "trial", "interval_s": gap,
                                              "trials": len(trials), "bound": None,
                                              "seen_once": ["ca", "cb"]}


def test_the_design_holds_the_cosines_below_that_cutoff(runs):
    first, _, _ = runs
    design = read_table(next(first.rglob("*_design.tsv")))
    cutoff = _glm_params(first)["drift_high_pass"]
    # nilearn's basis: floor(2 x duration x cutoff) cosines plus the constant
    assert sum(c.startswith("drift") for c in design.columns) == math.floor(2 * DURATION * cutoff)


def test_the_rerun_is_auto_again_and_lands_on_the_same_cutoff(runs):
    first, rerun, _ = runs
    record = tomllib.loads((rerun / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8"))
    assert record["post"]["drift_high_pass"] == "auto"
    assert _glm_params(rerun)["drift_high_pass_auto"] == _glm_params(first)["drift_high_pass_auto"]


def test_the_report_notes_the_conditions_seen_once(runs):
    first, _, _ = runs
    sidecar = next(first.rglob("*desc-glm_nirsmap.json"))
    notes = _drift_auto_notes(sidecar.parent, sidecar.name.split("_desc-")[0], "01")
    assert notes == ["Seen once, so no interval for --drift-high-pass auto to read: ca, cb."]
