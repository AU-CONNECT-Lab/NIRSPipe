"""--dry-run has to stop before the pipeline, not merely be recorded as having been asked for."""

import sys

import pytest

from nirspipe.cli.workflows import run_participant_level

_ARGS = dict(
    analysis_level="participant", session_label=None, task_label=["tapping"],
    bids_filter_file=None, work_dir=None, verbose=False, skip_bids_validation=True,
    ignore=None, n_jobs=1, no_report=True,
    dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
    cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
    mode="denoise", high_pass=0.01, low_pass=0.5,
)


@pytest.fixture(scope="module")
def dry_out(mini_bids, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("dry") / "out"
    args = {**_ARGS, "bids_dir": mini_bids, "output_dir": out_dir,
            "participant_label": ["01"], "dry_run": True}

    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(out_dir), "participant", "--dry-run"]
    try:
        run_participant_level(args)
    finally:
        sys.argv = original
    return out_dir


def test_the_script_and_the_record_are_written(dry_out):
    logs = dry_out / "sub-01" / "logs"
    assert (logs / "sub-01_script.py").exists()
    assert list(logs.glob("sub-01*.toml"))


def test_nothing_is_processed(dry_out):
    assert not list((dry_out / "sub-01").glob("nirs/*.snirf"))
    assert not list((dry_out / "sub-01").glob("*.html"))
