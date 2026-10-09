"""`fnirs-qc hyper-raw` writes the fnirs-hyper tree and reads the corrected recordings from the
fnirs-pipe tree, never one directory standing in for the other."""

import json

import pytest

from fnirs_pipe.cli import qc
from fnirs_pipe.pipeline import hyper as hyper_pkg

PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5", "--sci-threshold", "0.8"]


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
    # it reads both, so both are linked, under the names fnirs-hyper and fnirs-pipe use
    links = desc["DatasetLinks"]
    assert (out / links["preprocessed"]).resolve() == deriv.resolve()
    assert (out / links["raw"]).resolve() == bids.resolve()
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
    assert "are the same" in capsys.readouterr().err
    assert not any(tree.iterdir())


@pytest.mark.parametrize("entry", ["main", "main_pair_null"])
def test_fnirs_hyper_refuses_to_write_into_the_tree_it_reads(entry, tmp_path, capsys):
    from fnirs_pipe.cli import hyper

    tree = tmp_path / "shared"
    tree.mkdir()
    (tmp_path / "pairs.csv").write_text("group_id,subject_id,task\nG01,sub-a,main\n")
    with pytest.raises(SystemExit) as exit_:
        getattr(hyper, entry)([str(tree), str(tree), "group",
                               "--pairs-csv", str(tmp_path / "pairs.csv")])
    assert exit_.value.code != 0
    assert "are the same" in capsys.readouterr().err
    assert not any(tree.iterdir())


def test_a_fnirs_pipe_tree_is_refused_as_the_output(mini_hyper_bids, tmp_path, capsys):
    from fnirs_pipe.io.derivatives import write_dataset_description

    bids, pairs = mini_hyper_bids
    tree = tmp_path / "fnirs-pipe"
    write_dataset_description(tree)
    before = (tree / "dataset_description.json").read_text()
    with pytest.raises(SystemExit) as exit_:
        qc.main(_argv(bids, pairs, tree))
    assert exit_.value.code != 0
    assert "is a fnirs-pipe tree" in capsys.readouterr().err
    assert (tree / "dataset_description.json").read_text() == before
    assert not list(tree.glob("group-*"))


def test_without_a_derivatives_tree_the_existing_stamp_is_kept(
        mini_hyper_bids, tmp_path, stage_reads):
    from fnirs_pipe.io.derivatives import LINK_PREPROCESSED, write_dataset_description

    bids, pairs = mini_hyper_bids
    deriv, out = tmp_path / "fnirs-pipe", tmp_path / "fnirs-hyper"
    deriv.mkdir()
    write_dataset_description(out, name="fnirs-hyper output", generated_by="fnirs-hyper",
                              source=deriv, link=LINK_PREPROCESSED)
    qc.main(_argv(bids, pairs, out))

    # the link fnirs-hyper wrote stays, and the raw recordings this read are added beside it
    desc = json.loads((out / "dataset_description.json").read_text())
    assert (out / desc["DatasetLinks"]["preprocessed"]).resolve() == deriv.resolve()
    assert (out / desc["DatasetLinks"]["raw"]).resolve() == bids.resolve()


def test_session_label_picks_the_session_read_and_every_output_follows_it(tmp_path, stage_reads):
    """A pairs table without sessions, as a BIDS App reads --session-label: which session to
    process, every output of it under ses-."""
    from tests._synth import make_hyper_dataset

    bids, _ = make_hyper_dataset(tmp_path, tasks=("hold",), sessions=("a", "b"))
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("group_id,subject_id,task\nG01,sub-11,hold\nG01,sub-12,hold\n")
    out = tmp_path / "out"
    qc.main(_argv(bids, pairs, out, "--session-label", "b"))

    group = out / "group-G01"
    written = [p for p in group.rglob("*") if p.is_file() and p.parent.name == "nirs"]
    assert written
    assert all(p.parent == group / "ses-b" / "nirs" and "_ses-b_" in p.name for p in written)
    for sidecar in (p for p in written if p.suffix == ".json"):
        sources = json.loads(sidecar.read_text()).get("Sources") or []
        assert all("ses-a" not in s for s in sources), (sidecar.name, sources)
    assert (group / "group-G01_ses-b_task-hold_desc-raw_report.html").exists()
