"""The BIDS App contract at the command line: the input dataset is never written to, labels
that name no participant are refused, and the input is checked with bids-validator unless the
user skips it."""

import json
import subprocess
import sys

import pytest

from fnirs_pipe.cli import workflows
from fnirs_pipe.cli.run import main
from fnirs_pipe.io import bids as bids_io

REQUIRED = ["--dpf", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
            "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.1", "--resp-h-freq", "0.4"]

_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    bids_filter_file=None, work_dir=None, verbose=False, ignore=None, n_jobs=1,
    no_report=True, dry_run=True, dpf=[6.0, 6.0], sci_threshold=0.8,
    motion_correction="tddr", cardiac_l_freq=0.7, cardiac_h_freq=1.5,
    resp_l_freq=0.2, resp_h_freq=0.5,
)


def _dry_run(bids_dir, out_dir, **overrides):
    args = {**_ARGS, "bids_dir": bids_dir, "output_dir": out_dir,
            "participant_label": ["01"], "skip_bids_validation": False, **overrides}
    original = sys.argv
    sys.argv = ["fnirs-pipe", str(bids_dir), str(out_dir), "participant", "--dry-run"]
    try:
        workflows.run_participant_level(args)
    finally:
        sys.argv = original


# ---- the input is not written to ----

def test_an_output_directory_that_is_the_input_is_refused(mini_bids, capsys):
    before = (mini_bids / "dataset_description.json").read_text()
    with pytest.raises(SystemExit) as exit_:
        main([str(mini_bids), str(mini_bids), "participant", *REQUIRED])
    assert exit_.value.code != 0
    assert "output directory is the input" in capsys.readouterr().err
    assert (mini_bids / "dataset_description.json").read_text() == before


def test_a_work_directory_inside_the_input_is_refused(mini_bids, tmp_path):
    with pytest.raises(SystemExit) as exit_:
        main([str(mini_bids), str(tmp_path / "out"), "participant", *REQUIRED,
              "--work-dir", str(mini_bids / "work")])
    assert exit_.value.code != 0
    assert not (mini_bids / "work").exists()


# ---- participant labels ----

def test_a_label_naming_no_participant_is_refused_before_anything_is_written(
        mini_bids, tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "validate_bids", lambda bids_dir: None)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="99"):
        _dry_run(mini_bids, out, participant_label=["01", "99"])
    assert not out.exists()


# ---- validation ----

def test_the_input_is_validated_unless_skipped(mini_bids, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(workflows, "validate_bids", seen.append)
    _dry_run(mini_bids, tmp_path / "a")
    _dry_run(mini_bids, tmp_path / "b", skip_bids_validation=True)
    _dry_run(mini_bids, tmp_path / "c", ignore=["bids-validation"])
    assert seen == [mini_bids]


def test_the_output_names_the_dataset_it_was_computed_from(mini_bids, tmp_path, monkeypatch):
    monkeypatch.setattr(workflows, "validate_bids", lambda bids_dir: None)
    _dry_run(mini_bids, tmp_path / "out")
    desc = json.loads((tmp_path / "out" / "dataset_description.json").read_text())
    assert desc["SourceDatasets"][0]["URL"] == mini_bids.resolve().as_uri()


def _fake_validator(monkeypatch, report: dict, code: int):
    monkeypatch.setattr(bids_io, "_validator_command", lambda: ["bids-validator"])
    monkeypatch.setattr(bids_io.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], code, stdout=json.dumps(report), stderr=""))


def test_validator_errors_stop_the_run_and_name_the_problem(tmp_path, monkeypatch):
    issues = [{"code": "NOT_INCLUDED", "severity": "error", "location": f"/sub-01/x{i}.txt"}
              for i in range(5)]
    issues.append({"code": "README_FILE_MISSING", "severity": "warning", "location": "/"})
    _fake_validator(monkeypatch, {"issues": {"issues": issues}}, 16)
    with pytest.raises(SystemExit) as exit_:
        bids_io.validate_bids(tmp_path)
    message = str(exit_.value.code)
    assert "NOT_INCLUDED x5" in message and "/sub-01/x0.txt" in message
    assert "and 2 more" in message and "--skip-bids-validation" in message
    assert "README_FILE_MISSING" not in message


def test_the_older_validator_report_is_read_too(tmp_path, monkeypatch):
    report = {"issues": {"errors": [{"key": "NOT_INCLUDED",
                                     "files": [{"file": {"relativePath": "/sub-01/x.txt"}}]}],
                         "warnings": []}}
    _fake_validator(monkeypatch, report, 1)
    with pytest.raises(SystemExit, match="NOT_INCLUDED x1: /sub-01/x.txt"):
        bids_io.validate_bids(tmp_path)


def test_warnings_alone_let_the_run_go_on(tmp_path, monkeypatch):
    report = {"issues": {"issues": [{"code": "README_FILE_MISSING", "severity": "warning"}]}}
    _fake_validator(monkeypatch, report, 0)
    bids_io.validate_bids(tmp_path)


def test_without_a_validator_the_run_goes_on_and_says_so(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(bids_io, "_validator_command", lambda: None)
    bids_io.validate_bids(tmp_path)
    assert "bids-validator not found" in caplog.text


def test_a_validator_that_prints_no_report_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bids_io, "_validator_command", lambda: ["bids-validator"])
    monkeypatch.setattr(bids_io.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a[0], 2, stdout="", stderr="boom"))
    with pytest.raises(SystemExit, match="boom"):
        bids_io.validate_bids(tmp_path)


# ---- the other commands that read a BIDS dataset ----

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5"]
SAME_DIR_COMMANDS = {
    "crop":           ("prep", ["crop", "{b}", "{b}", "--participant-label", "01", "--tmin", "0"]),
    "align":          ("prep", ["align", "{b}", "{b}", "--group-csv", "{b}/pairs.csv"]),
    "markers export": ("prep", ["edit-markers", "export", "{b}", "{b}",
                                "--participant-label", "01"]),
    "markers apply":  ("prep", ["edit-markers", "apply", "{b}", "{b}", "--participant-label", "01",
                                "--shift", "1"]),
    "prep-raw":       ("qc", ["prep-raw", "{b}", "{b}", "--participant-label", "01", *PHYS]),
    "hyper-raw":      ("qc", ["hyper-raw", "{b}", "{b}", "group", "--pairs-csv", "{b}/pairs.csv",
                                *PHYS]),
}


def _entry(module: str):
    from fnirs_pipe.cli import prep, qc
    return {"prep": prep.main, "qc": qc.main}[module]


@pytest.mark.parametrize("name", list(SAME_DIR_COMMANDS))
def test_every_command_refuses_to_write_into_its_input(name, mini_bids, capsys):
    module, argv = SAME_DIR_COMMANDS[name]
    before = sorted(p.relative_to(mini_bids) for p in mini_bids.rglob("*"))
    with pytest.raises(SystemExit) as exit_:
        _entry(module)([a.format(b=mini_bids) for a in argv] + ["--skip-bids-validation"])
    assert exit_.value.code != 0
    assert "output directory is the input" in capsys.readouterr().err
    assert sorted(p.relative_to(mini_bids) for p in mini_bids.rglob("*")) == before


def test_cropping_a_pipeline_output_back_into_its_own_tree_is_allowed(tmp_path, capsys):
    from fnirs_pipe.cli import prep
    tree = tmp_path / "out"
    tree.mkdir()
    try:
        prep.main(["crop", str(tree), str(tree), "--participant-label", "01", "--tmin", "0",
                   "--input-desc", "errts", "--skip-bids-validation"])
    except SystemExit:
        pass          # nothing to crop in an empty tree; the point is what it was stopped for
    assert "output directory is the input" not in capsys.readouterr().err
