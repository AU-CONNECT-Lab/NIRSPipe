"""`--events-path` with {sub}, {ses}, {task}, {acq} and {run}: each run reads its own table."""

import json
import tomllib

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.cli.workflows import _refuse_bad_events_path, events_path_for

from tests._fingerprint import CLI_ARGS, EVENT_DURATION, EVENT_ONSETS, make_fingerprint_dataset

RUN = "sub-02_ses-b_task-tap_acq-x_run-03_desc-preproc_nirs.snirf"


def test_each_placeholder_takes_the_runs_own_entity(tmp_path):
    (tmp_path / "02_b_tap_x_03.tsv").write_text("onset\tduration\ttrial_type\n", encoding="utf-8")
    template = str(tmp_path / "{sub}_{ses}_{task}_{acq}_{run}.tsv")
    assert events_path_for(template, RUN) == str(tmp_path / "02_b_tap_x_03.tsv")


def test_a_path_without_placeholders_serves_every_run_as_typed():
    assert events_path_for("events.tsv", RUN) == "events.tsv"


def test_a_run_without_the_entity_named_is_refused(tmp_path):
    with pytest.raises(ValueError, match="sub-01_task-tap_nirs.snirf has no ses"):
        events_path_for(str(tmp_path / "{sub}_{ses}.tsv"), "sub-01_task-tap_nirs.snirf")


def test_a_runs_missing_file_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="02_events.tsv, the file for"):
        events_path_for(str(tmp_path / "{sub}_events.tsv"), RUN)


def test_a_placeholder_no_run_can_fill_stops_before_any_subject():
    with pytest.raises(SystemExit, match=r"\{subject\} is not one of \{sub\}"):
        _refuse_bad_events_path({"events_path": "ev/{subject}_events.tsv"})


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("events_template")
    bids, _ = make_fingerprint_dataset(root)
    events = root / "events"
    events.mkdir()
    rows = "".join(f"{o:g}\t{EVENT_DURATION:g}\ttapping\n" for o in EVENT_ONSETS)
    (events / "sub-01_task-tapping_events.tsv").write_text(
        "onset\tduration\ttrial_type\n" + rows, encoding="utf-8")
    template = str(events / "sub-{sub}_task-{task}_events.tsv")
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *CLI_ARGS, "--mode", "glm",
                  "--events-path", template, "--noise-model", "ols", "--drift-high-pass",
                  "0.01", "--participant-label", "01", "--skip-bids-validation", "--no-report"])
    return out, template


def test_the_glm_reads_the_runs_own_table(run):
    out, _ = run
    sidecar = json.loads(next(out.rglob("*desc-glm_nirsmap.json")).read_text(encoding="utf-8"))
    assert [s.rsplit("/", 1)[-1] for s in sidecar["external_sources"]] == [
        "sub-01_task-tapping_events.tsv"]


def test_the_record_keeps_the_template(run):
    out, template = run
    record = tomllib.loads((out / "sub-01" / "logs" / "sub-01.toml").read_text(encoding="utf-8"))
    assert record["post"]["events_path"] == template.replace("\\", "/")
