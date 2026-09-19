"""One bad recording must not strand the subjects queued behind it, serial or parallel.

The participant loop used to re-raise, so the first failure ended the batch and every
subject after it went unprocessed. `fnirs-qc` and `fnirs-prep` already isolated theirs.
`--n-jobs` then made the same loop run several subjects at once, which is only safe if each
one's log goes to its own file: `setup_logging` replaces the root handlers, so the second
subject used to take the first one's log away.
"""

import sys

import pytest

from fnirs_pipe.cli.workflows import run_participant_level

_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    bids_filter_file=None, work_dir=None, verbose=False, skip_bids_validation=True,
    ignore=None, n_jobs=1, no_report=True, dry_run=False,
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

    monkeypatch.setattr("fnirs_pipe.cli.workflows.run_prep", _boom)
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": tmp_path / "out",
            "participant_label": ["01", "02"]}

    original = sys.argv
    sys.argv = ["fnirs-pipe", str(mini_bids), str(tmp_path / "out"), "participant"]
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

    monkeypatch.setattr("fnirs_pipe.cli.workflows.run_prep", _boom)
    out = tmp_path / "out"
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": out,
            "participant_label": ["01", "02"], "n_jobs": 2}

    original = sys.argv
    sys.argv = ["fnirs-pipe", str(mini_bids), str(out), "participant", "--n-jobs", "2"]
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
        log = out / f"sub-{sub}" / "logs" / f"sub-{sub}.log"
        assert log.exists(), f"sub-{sub} has no log"
        text = log.read_text(encoding="utf-8")
        assert f"sub-{sub} | starting" in text
        assert f"sub-{other} |" not in text
