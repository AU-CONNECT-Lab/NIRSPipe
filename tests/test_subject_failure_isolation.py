"""One bad recording must not strand the subjects queued behind it.

The participant loop used to re-raise, so the first failure ended the batch and every
subject after it went unprocessed. `fnirs-qc` and `fnirs-prep` already isolated theirs.
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
