"""A --bad-channels row limited to one task marks that task's recording and no other."""

import json
import re
import subprocess
import sys

from nirspipe.cli.workflows import run_participant_level


def _marked(out_dir, task):
    (sidecar,) = out_dir.rglob(f"sub-01_task-{task}_desc-sci_nirs.json")
    return sorted(json.loads(sidecar.read_text(encoding="utf-8"))["bad_channels"])


def test_a_task_row_reaches_only_that_task_in_the_run_and_its_script(mini_bids, tmp_path):
    table = tmp_path / "bads.tsv"
    table.write_text("participant_id\ttask\tbad_channels\nsub-01\ttask-rest\tS1_D1\n",
                     encoding="utf-8")
    cli_out, script_out = tmp_path / "cli", tmp_path / "script"
    args = dict(
        analysis_level="participant", session_label=None, task_label=None,
        bids_filter_file=None, work_dir=None, verbose=False, skip_bids_validation=True,
        ignore=None, n_jobs=1, no_report=True, mode=None,
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

    script = (cli_out / "sub-01" / "logs" / "sub-01_script.py").read_text(encoding="utf-8")
    script = re.sub(r"^OUTPUT_DIR = .*$", f"OUTPUT_DIR = Path({script_out.as_posix()!r})",
                    script, count=1, flags=re.M)
    script_path = tmp_path / "script.py"
    script_path.write_text(script, encoding="utf-8")
    subprocess.run([sys.executable, str(script_path)], check=True, cwd=tmp_path)

    for out_dir in (cli_out, script_out):
        assert {"S1_D1 760", "S1_D1 850"} <= set(_marked(out_dir, "rest"))
        assert "S1_D1 760" not in _marked(out_dir, "tapping")
