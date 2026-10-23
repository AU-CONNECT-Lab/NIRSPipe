"""One bad recording must not strand the subjects queued behind it, serial or parallel.

A failure is logged and the loop moves on to the next subject. With `--nprocs` the same
loop runs several subjects at once, which is only safe if each one's log goes to its own
file: `setup_logging` replaces the root handlers, so a shared log would be taken over by the
next subject.
"""

import sys

import pytest

from nirspipe.cli.run_record import record_stem
from nirspipe.cli.workflows import run_participant_level

_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    bids_filter_file=None, verbose=False, skip_bids_validation=True,
    ignore=None, nprocs=None, no_report=True, dry_run=False,
    dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
    cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
    mode="denoise", high_pass=0.01, low_pass=0.5,
)


@pytest.fixture
def attempted(mini_bids, tmp_path, monkeypatch):
    """Run two subjects whose prep always raises; return the ones the loop reached."""
    seen = []

    def _boom(raw, config, *args, **kwargs):
        seen.append(config.subject)
        raise RuntimeError("synthetic prep failure")

    monkeypatch.setattr("nirspipe.cli.workflows.run_prep", _boom)
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": tmp_path / "out",
            "participant_label": ["01", "02"]}

    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(tmp_path / "out"), "participant"]
    try:
        with pytest.raises(SystemExit) as exit_info:
            run_participant_level(args)
    finally:
        sys.argv = original
    return seen, exit_info.value.code


def test_the_second_subject_still_runs(attempted):
    seen, _ = attempted
    assert seen == ["01", "02"]


def test_the_batch_still_exits_non_zero(attempted):
    _, code = attempted
    assert code == 1


@pytest.fixture
def parallel_attempted(mini_bids, tmp_path, monkeypatch):
    """The same two subjects over two jobs; returns what ran and where the logs went."""
    seen = []

    def _boom(raw, config, *args, **kwargs):
        seen.append(config.subject)
        raise RuntimeError("synthetic prep failure")

    monkeypatch.setattr("nirspipe.cli.workflows.run_prep", _boom)
    out = tmp_path / "out"
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": out,
            "participant_label": ["01", "02"], "nprocs": 2}

    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(out), "participant", "--nprocs", "2"]
    try:
        with pytest.raises(SystemExit):
            run_participant_level(args)
    finally:
        sys.argv = original
    return seen, out


def test_parallel_runs_every_subject(parallel_attempted):
    seen, _ = parallel_attempted
    assert sorted(seen) == ["01", "02"]


def test_each_subject_keeps_its_own_log(parallel_attempted):
    _, out = parallel_attempted
    for sub, other in (("01", "02"), ("02", "01")):
        log = out / f"sub-{sub}" / "logs" / f"{record_stem(sub, tasks=['tapping'])}.log"
        assert log.exists(), f"sub-{sub} has no log"
        text = log.read_text(encoding="utf-8")
        assert f"sub-{sub} | starting" in text
        assert f"sub-{other} |" not in text


# ---- each subject's run record ends with how it went ----

@pytest.fixture
def mixed_outcome(mini_bids, tmp_path, monkeypatch):
    """sub-01's prep raises, sub-02's runs for real; returns the output tree."""
    from nirspipe.cli import workflows

    real_prep = workflows.run_prep

    def _fail_one(raw, config, *args, **kwargs):
        if config.subject == "01":
            raise RuntimeError("synthetic prep failure\nsecond line of the message")
        return real_prep(raw, config, *args, **kwargs)

    monkeypatch.setattr("nirspipe.cli.workflows.run_prep", _fail_one)
    out = tmp_path / "out"
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": out, "participant_label": ["01", "02"]}

    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(out), "participant"]
    try:
        with pytest.raises(SystemExit):
            run_participant_level(args)
    finally:
        sys.argv = original
    return out


def _outcome(out, sub):
    from nirspipe.utils import load_toml
    return load_toml(_record(out, sub)).get("outcome")


def _record(out, sub):
    return out / f"sub-{sub}" / "logs" / f"{record_stem(sub, tasks=['tapping'])}.toml"


def test_a_failed_subject_records_the_first_line_of_its_error(mixed_outcome):
    outcome = _outcome(mixed_outcome, "01")
    assert outcome["status"] == "failed"
    assert outcome["error"] == "RuntimeError: synthetic prep failure"
    assert outcome["duration_s"] >= 0


def test_a_finished_subject_records_success_and_no_error(mixed_outcome):
    outcome = _outcome(mixed_outcome, "02")
    assert outcome["status"] == "success"
    assert "error" not in outcome


def test_a_record_with_an_outcome_still_reads_back_as_config(mixed_outcome):
    from nirspipe.cli.run import read_config

    settings = read_config(_record(mixed_outcome, "01"))
    assert settings["prep"]["sci_threshold"] == 0.8


def test_a_dry_run_record_has_no_outcome(mini_bids, tmp_path):
    out = tmp_path / "out"
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": out, "participant_label": ["01"],
            "dry_run": True}
    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(out), "participant", "--dry-run"]
    try:
        run_participant_level(args)
    finally:
        sys.argv = original
    assert _outcome(out, "01") is None
