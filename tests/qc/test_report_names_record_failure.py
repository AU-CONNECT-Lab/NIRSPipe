"""A run whose SQM record could not be written says why on its report, not that the file is missing."""

from nirspipe.cli import run as run_cli
from nirspipe.io.naming import report_name
from nirspipe.qc.subject import sqm_record

from tests._synth import _write_dataset_root, _write_subject, synth_raw

REASON = "per-window series of unequal length [50, 51]; they cannot share one table"
FLAGS = ("--dpf", "6", "--sci-threshold", "0.5", "--cardiac-l-freq", "0.7",
         "--cardiac-h-freq", "2.0", "--resp-l-freq", "0.1", "--resp-h-freq", "0.5",
         "--skip-bids-validation")


def _refuse(*_args, **_kwargs):
    raise ValueError(REASON)


def test_the_report_gives_the_writers_reason_once(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    _write_subject(bids, "01", "tap", synth_raw("01", "tap", duration=300.0))
    monkeypatch.setattr(sqm_record, "compute_run_sections", _refuse)
    out = tmp_path / "out"
    run_cli.main([str(bids), str(out), "participant", *FLAGS])

    html = (out / "sub-01" / report_name("sub-01_task-tap")).read_text(encoding="utf-8")
    assert html.count(f"SQM record: not written: {REASON}") == 1
    assert "No such file" not in html
