"""The raw QC trees describe themselves: who wrote them and which files a validator skips."""

import json

import pytest

from nirspipe.cli import qc

from tests._raw_qc import raw_qc

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5", "--sci-threshold", "0.8"]


def _generated_by(tree):
    desc = json.loads((tree / "dataset_description.json").read_text())
    return [e["Name"] for e in desc["GeneratedBy"]], desc["Name"]


def test_the_raw_qc_tree_lists_what_a_validator_skips(mini_bids, tmp_path):
    out = tmp_path / "qc"
    qc.main([str(mini_bids), str(out), "participant", "--participant-label", "01",
             "--task-label", "rest", "--skip-bids-validation", *PHYS])
    assert "*.html" in (out / ".bidsignore").read_text().splitlines()


def test_an_absent_participant_is_refused_before_the_tree_is_written(mini_bids, tmp_path,
                                                                     capsys):
    out = tmp_path / "qc"
    with pytest.raises(SystemExit):
        qc.main([str(mini_bids), str(out), "participant", "--participant-label", "zz",
                 "--skip-bids-validation", *PHYS])
    assert "participant label(s) not in" in capsys.readouterr().err
    assert not out.exists()


def test_the_group_level_keeps_the_one_stamp(mini_hyper_bids, tmp_path):
    bids, pairs = mini_hyper_bids
    out = tmp_path / "qc"
    raw_qc(bids, out, pairs, participant=(*PHYS, "--task-label", "hold"),
           group=("--task-label", "hold"))

    assert _generated_by(out) == (["nirspipe"], "nirspipe output")
    assert (out / ".bidsignore").exists()
