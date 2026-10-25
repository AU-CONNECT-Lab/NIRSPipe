"""Raw QC --run-label: a chosen run gets a report of its own, and the task's report is kept."""

import json
import re

import pytest

from nirspipe.cli import qc
from nirspipe.io.naming import report_name
from tests._synth import _write_dataset_root, _write_subject, synth_raw

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5", "--sci-threshold", "0.8",
        "--skip-bids-validation"]


@pytest.fixture(scope="module")
def two_runs(tmp_path_factory):
    """sub-01, task-tapping, runs 1 and 2."""
    bids = tmp_path_factory.mktemp("runs") / "bids"
    _write_dataset_root(bids, ["01"])
    for run in ("1", "2"):
        snirf = _write_subject(bids, "01", "tapping", synth_raw("01", "tapping", duration=120.0,
                                                                seed=int(run)))
        for path in list(snirf.parent.glob("sub-01_task-tapping_*")):
            if "_run-" not in path.name:
                path.rename(path.with_name(path.name.replace("_task-tapping_", f"_task-tapping_run-{run}_")))
    return bids


def _run_labels(html):
    return json.loads(re.search(r"var _RUN_LABELS\s*=\s*(\[.*?\]);", html.read_text(encoding="utf-8")).group(1))


def test_a_chosen_run_gets_its_own_report_and_the_tasks_report_is_kept(two_runs, tmp_path):
    out = tmp_path / "qc"
    qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", *PHYS])
    task_page = out / "sub-01" / report_name("sub-01_task-tapping", desc="raw")
    assert _run_labels(task_page) == ["sub-01_task-tapping_run-1", "sub-01_task-tapping_run-2"]
    before = task_page.read_text(encoding="utf-8")

    qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", "--run-label", "2", *PHYS])

    run_page = out / "sub-01" / report_name("sub-01_task-tapping_run-2", desc="raw")
    assert _run_labels(run_page) == ["sub-01_task-tapping_run-2"]
    assert task_page.read_text(encoding="utf-8") == before


def test_a_run_label_no_recording_has_writes_no_report(two_runs, tmp_path, capsys):
    out = tmp_path / "qc"
    try:
        qc.main([str(two_runs), str(out), "participant", "--participant-label", "01", "--run-label", "9", *PHYS])
    except SystemExit:
        pass
    assert not list(out.glob("sub-01/*_desc-raw_report.html"))
    assert "no SNIRF files found for sub-01" in capsys.readouterr().err
