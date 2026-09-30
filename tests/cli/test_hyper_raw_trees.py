"""`fnirs-qc hyper-raw` writes the fnirs-hyper tree and reads the corrected recordings from the
fnirs-pipe tree, never one directory standing in for the other."""

import json

import pytest

from fnirs_pipe.cli import qc
from fnirs_pipe.pipeline import hyper as hyper_pkg

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5"]


def _argv(bids, pairs, out, *extra):
    return ["hyper-raw", str(bids), str(out), "group", "--pairs-csv", str(pairs),
            "--task-label", "hold", "--skip-bids-validation", *PHYS, *extra]


@pytest.fixture
def stage_reads(monkeypatch):
    seen = []

    def _record(output_dir, group, desc):
        seen.append((output_dir, desc))
        return {}
    monkeypatch.setattr(hyper_pkg, "load_group_stage", _record)
    return seen


def test_the_corrected_side_is_read_from_the_derivatives_tree(
        mini_hyper_bids, tmp_path, stage_reads):
    bids, pairs = mini_hyper_bids
    deriv, out = tmp_path / "fnirs-pipe", tmp_path / "fnirs-hyper"
    deriv.mkdir()
    qc.main(_argv(bids, pairs, out, "--derivatives-dir", str(deriv)))

    assert stage_reads == [(deriv, "motcorrected")]
    assert list((out / "group-G01").glob("*_desc-raw_report.html"))
    desc = json.loads((out / "dataset_description.json").read_text())
    assert desc["SourceDatasets"][0]["URL"] == deriv.resolve().as_uri()
    assert not any(deriv.iterdir())


def test_without_a_derivatives_tree_no_stage_is_looked_for(
        mini_hyper_bids, tmp_path, stage_reads, capsys):
    bids, pairs = mini_hyper_bids
    qc.main(_argv(bids, pairs, tmp_path / "fnirs-hyper"))

    assert stage_reads == []
    assert "no --derivatives-dir" in capsys.readouterr().err


def test_the_hyper_tree_cannot_be_the_fnirs_pipe_tree(mini_hyper_bids, tmp_path, capsys):
    bids, pairs = mini_hyper_bids
    tree = tmp_path / "shared"
    tree.mkdir()
    with pytest.raises(SystemExit) as exit_:
        qc.main(_argv(bids, pairs, tree, "--derivatives-dir", str(tree)))
    assert exit_.value.code != 0
    assert "fnirs-hyper tree" in capsys.readouterr().err
    assert not any(tree.iterdir())
