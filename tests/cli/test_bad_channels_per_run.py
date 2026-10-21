"""A --bad-channels row limited to one task marks that task's recording and no other."""

import json
import sys

from nirspipe.cli.workflows import run_participant_level


def _marked(out_dir, task):
    (sidecar,) = out_dir.rglob(f"sub-01_task-{task}_desc-sci_nirs.json")
    return sorted(json.loads(sidecar.read_text(encoding="utf-8"))["bad_channels"])


def test_a_task_row_reaches_only_that_task(mini_bids, tmp_path):
    table = tmp_path / "bads.tsv"
    table.write_text("participant_id\ttask\tbad_channels\nsub-01\ttask-rest\tS1_D1\n",
                     encoding="utf-8")
    cli_out = tmp_path / "cli"
    args = dict(
        analysis_level="participant", session_label=None, task_label=None,
        bids_filter_file=None, verbose=False, skip_bids_validation=True,
        ignore=None, nprocs=None, no_report=True, mode=None,
        dpf=[6.0, 6.0], sci_threshold=0.8, motion_correction="tddr",
        cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5,
        bad_channels=str(table),
        bids_dir=mini_bids, output_dir=cli_out, participant_label=["01"])
    original = sys.argv
    sys.argv = ["nirspipe", str(mini_bids), str(cli_out), "participant"]
    try:
        run_participant_level(args)
    finally:
        sys.argv = original

    assert {"S1_D1 760", "S1_D1 850"} <= set(_marked(cli_out, "rest"))
    assert "S1_D1 760" not in _marked(cli_out, "tapping")


def test_a_table_without_participant_id_reaches_the_fingerprint_run(tmp_path):
    from nirspipe.cli import run as run_cli
    from tests._fingerprint import CLI_ARGS, make_fingerprint_dataset

    bids, _ = make_fingerprint_dataset(tmp_path)
    table = tmp_path / "bads.tsv"
    table.write_text("task\tbad_channels\ntask-tapping\tS1_D1\n", encoding="utf-8")
    run_cli.main([str(bids), str(tmp_path / "out"), "participant", *CLI_ARGS,
                  "--bad-channels", str(table), "--no-report", "--skip-bids-validation"])

    assert {"S1_D1 760", "S1_D1 850"} <= set(_marked(tmp_path / "out", "tapping"))
