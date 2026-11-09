"""`--stim-dur` per condition: NAME=SEC, a plain number for the rest, read back from the record."""

import json
import tomllib

import mne
import numpy as np
import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _refuse_glm_without_durations
from nirspipe.pipeline.glm import _annotation_events, stim_durations
from nirspipe.qc.boilerplate.vocabulary import _conditions_phrase

from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

GLM = ("--mode", "glm", "--noise-model", "ols", "--drift-high-pass", "0.01",
       "--participant-label", "01", "--skip-bids-validation", "--no-report")


# ---- reading the flag ----

@pytest.mark.parametrize("given, expected", [
    (["20"], 20.0),
    (20, 20.0),
    (["Tapping=20", "Control=15"], {"Tapping": 20.0, "Control": 15.0}),
    (["20", "Press=0"], {"*": 20.0, "Press": 0.0}),
    (["Electrical VAS 3=12"], {"Electrical VAS 3": 12.0}),
    ({"*": 20, "Press": 0}, {"*": 20.0, "Press": 0.0}),
    (None, None),
])
def test_the_flag_reads_as_one_duration_or_a_condition_map(given, expected):
    assert stim_durations(given) == expected


@pytest.mark.parametrize("given, message", [
    (["20", "15"], "more than one plain number"),
    (["A=1", "A=2"], "condition A twice"),
    (["A=-1"], "cannot be negative"),
    (["A=x"], "not a number"),
    (["=5"], "condition name goes before"),
])
def test_a_flag_that_cannot_be_read_stops_before_any_subject(given, message):
    with pytest.raises(SystemExit, match=message):
        _refuse_glm_without_durations({"mode": "glm", "stim_dur": given})


# ---- each event's duration ----

def _raw(names):
    info = mne.create_info(["S1_D1 hbo"], 10.0, "hbo")
    raw = mne.io.RawArray(np.zeros((1, 3000)), info, verbose="error")
    raw.set_annotations(mne.Annotations([10.0 + 60 * i for i in range(len(names))],
                                        [1.0] * len(names), names))
    return raw


def test_each_event_takes_its_conditions_duration():
    events = _annotation_events(_raw(["tap", "rest", "tap"]), {"tap": 20.0, "rest": 15.0})
    assert list(events["duration"]) == [20.0, 15.0, 20.0]


def test_the_plain_number_covers_the_conditions_not_named():
    events = _annotation_events(_raw(["tap", "rest", "press"]), {"*": 20.0, "press": 0.0})
    assert list(events["duration"]) == [20.0, 20.0, 0.0]


def test_a_condition_left_without_a_duration_is_named():
    with pytest.raises(ValueError, match="no duration for rest: name each"):
        _annotation_events(_raw(["tap", "rest"]), {"tap": 20.0})


# ---- the Methods ----

def test_the_methods_list_each_conditions_boxcar():
    params = {"hrf_model": "glover", "conditions": ["rest", "tap", "walk"],
              "stim_dur": {"tap": 20.0, "walk": 20.0, "rest": 15.0}}
    assert _conditions_phrase(params) == (
        "as boxcars of 20 s (tap and walk) and 15 s (rest) convolved with the Glover "
        "haemodynamic response function")


def test_an_instant_condition_is_left_to_its_own_clause():
    params = {"hrf_model": "glover", "conditions": ["press", "tap"],
              "stim_dur": {"tap": 20.0, "press": 0.0}, "instant_conditions": ["press"]}
    assert _conditions_phrase(params) == (
        "as 20 s boxcars convolved with the Glover haemodynamic response function (press, "
        "whose events last 0 s, as instantaneous responses scaled to a peak of 1)")


# ---- a run end to end, and its record read back ----

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("stim_dur_map")
    bids, _ = make_fingerprint_dataset(root, blocks=True)
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, *GLM,
                  "--stim-dur", "160", "trial=10"])
    return root, bids, out


def test_the_run_records_each_conditions_duration(run):
    _, _, out = run
    sidecar = next(out.rglob("*desc-glm_nirsmap.json"))
    params = json.loads(sidecar.read_text(encoding="utf-8"))["parameters"]
    assert params["stim_dur"] == {"ca": 160.0, "cb": 160.0, "trial": 10.0}


def test_the_record_reruns_the_same_map(run):
    root, bids, out = run
    record = out / "sub-01" / "logs" / "sub-01.toml"
    first = tomllib.loads(record.read_text(encoding="utf-8"))
    assert first["post"]["stim_dur"] == {"*": 160.0, "trial": 10.0}
    again = root / "again"
    run_cli.main([str(bids), str(again), "participant", "--config", str(record),
                  "--participant-label", "01", "--skip-bids-validation", "--no-report",
                  "--dry-run"])
    rerun = tomllib.loads((again / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8"))
    assert rerun["post"]["stim_dur"] == first["post"]["stim_dur"]
