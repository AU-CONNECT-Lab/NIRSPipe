"""The raw QC trees describe themselves: who wrote them and which files a validator skips."""

import json

import pytest

from nirspipe.cli import qc
from nirspipe.pipeline import hyper as hyper_pkg

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5", "--sci-threshold", "0.8"]


def _generated_by(tree):
    desc = json.loads((tree / "dataset_description.json").read_text())
    return [e["Name"] for e in desc["GeneratedBy"]], desc["Name"]


@pytest.fixture
def no_stage_reads(monkeypatch):
    monkeypatch.setattr(hyper_pkg, "load_group_stage", lambda *a, **k: {})


def test_prep_raw_tree_lists_what_a_validator_skips(mini_bids, tmp_path):
    out = tmp_path / "qc"
    with pytest.raises(SystemExit):
        qc.main(["prep-raw", str(mini_bids), str(out), "--participant-label", "zz",
                 "--skip-bids-validation", *PHYS])
    assert "*.html" in (out / ".bidsignore").read_text().splitlines()


def test_hyper_raw_tree_of_its_own_carries_its_own_stamp(mini_hyper_bids, tmp_path, no_stage_reads):
    bids, pairs = mini_hyper_bids
    out = tmp_path / "qc_hyper"
    qc.main(["hyper-raw", str(bids), str(out), "group", "--pairs-csv", str(pairs),
             "--task-label", "hold", "--skip-bids-validation", *PHYS])

    assert _generated_by(out) == (["nirspipe-qc hyper-raw"], "nirspipe-qc hyper-raw output")
    assert (out / ".bidsignore").exists()


def test_hyper_raw_into_the_hyper_tree_keeps_that_stamp(mini_hyper_bids, tmp_path, no_stage_reads):
    from nirspipe.io.derivatives import write_dataset_description

    bids, pairs = mini_hyper_bids
    out = tmp_path / "nirspipe-hyper"
    write_dataset_description(out, name="nirspipe-hyper output", generated_by="nirspipe-hyper")
    qc.main(["hyper-raw", str(bids), str(out), "group", "--pairs-csv", str(pairs),
             "--task-label", "hold", "--skip-bids-validation", *PHYS])

    assert _generated_by(out) == (["nirspipe-hyper"], "nirspipe-hyper output")
