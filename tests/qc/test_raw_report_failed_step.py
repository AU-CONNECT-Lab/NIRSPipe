"""A step raw QC goes on without is reported as not run, whatever an earlier run left in the tree."""

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.io.derivatives import write_dataset_description
from nirspipe.pipeline import prep_pipeline
from nirspipe.qc.subject.prep_raw_report import _process_run

from tests._raw_qc import SCREENING, raw_run
from tests._synth import make_bids_dataset


def _boom(*_args, **_kwargs):
    raise RuntimeError("planted failure")


@pytest.mark.parametrize("step, stage", [("correct_motion", "motcorrected"),
                                         ("od_to_haemo", "preproc")])
def test_a_step_failing_on_a_rerun_keeps_no_file_of_the_run_before(tmp_path, monkeypatch, step, stage):
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("rest",), duration=60.0)
    args = [str(bids), str(tmp_path / "out"), "participant", *SCREENING, "--skip-bids-validation"]
    qc_cli.main(args)
    (last,) = (tmp_path / "out" / "stages").rglob(f"*_desc-{stage}_nirs.snirf")
    monkeypatch.setattr(prep_pipeline, step, _boom)
    qc_cli.main(args)
    assert not last.exists()


def test_a_correction_that_failed_is_not_named_as_the_one_the_page_shows(tmp_path, monkeypatch):
    bids = make_bids_dataset(tmp_path, subjects=("01",), tasks=("tapping",), duration=60.0)
    (path,) = bids.rglob("*_nirs.snirf")
    write_dataset_description(tmp_path / "out", source=bids)
    monkeypatch.setattr(prep_pipeline, "correct_motion", _boom)
    inline, _ = _process_run(raw_run(bids, path, "tapping", motion_correction="tddr"),
                             tmp_path / "out" / "sub-01")
    assert any("planted failure" in error for error in inline["errors"])
    assert inline["summary"]["motion_correction"] == "none"
