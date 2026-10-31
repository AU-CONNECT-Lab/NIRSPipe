"""A step raw QC goes on without is reported as not run, whatever an earlier run left in the tree."""

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.pipeline import prep_pipeline

from tests._raw_qc import SCREENING
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
