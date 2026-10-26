"""Raw QC as two BIDS App levels in one tree: the group level reads what the participant level wrote."""

import json

import pytest

from nirspipe.cli import qc

from tests._raw_qc import SCREENING, raw_qc

SKIP = ("--skip-bids-validation",)


@pytest.fixture(scope="module")
def tree(mini_hyper_bids, tmp_path_factory):
    bids, pairs = mini_hyper_bids
    out = tmp_path_factory.mktemp("levels") / "qc"
    raw_qc(bids, out, pairs, group=("--task-label", "hold"))
    return out


def test_the_dyad_pages_sit_beside_the_subjects_in_one_tree(tree):
    assert sorted(p.name for p in tree.glob("sub-*")) == ["sub-11", "sub-12"]
    assert list((tree / "group-G01").glob("group-G01_task-hold_desc-raw_report.html"))
    assert (tree / "desc-subjects_report.html").exists()
    assert (tree / "desc-groups_report.html").exists()
    desc = json.loads((tree / "dataset_description.json").read_text())
    assert [e["Name"] for e in desc["GeneratedBy"]] == ["nirspipe"]


def test_a_finished_raw_qc_run_is_not_listed_as_unfinished(tree):
    from nirspipe.qc.subject.group_writer import unfinished_subjects
    from nirspipe.utils import load_toml

    records = sorted(tree.glob("sub-*/logs/*_desc-raw.toml"))
    assert [load_toml(r)["outcome"]["status"] for r in records] == ["success", "success"]
    assert unfinished_subjects(tree) == []
    assert 'id="Unfinished"' not in (tree / "desc-subjects_report.html").read_text(encoding="utf-8")


def test_each_member_is_read_from_its_own_record(tree):
    from nirspipe.qc.metrics import GOOD_FRAC_PASS

    sidecar = json.loads(next(tree.glob("group-G01/nirs/*_desc-subject_qc.json")).read_text())
    # the lines each member was screened by, the share at its default as none was typed
    assert sidecar["parameters"]["members"] == {
        sid: {"sci_threshold": 0.8, "psp_threshold": None, "min_good_frac": GOOD_FRAC_PASS,
              "cardiac_l_freq": 0.7, "cardiac_h_freq": 1.5} for sid in ("sub-11", "sub-12")}
    records = {p.name for p in tree.glob("sub-*/nirs/*_task-hold_desc-sqmraw_qc.json")}
    assert {s.rsplit("/", 1)[-1] for s in sidecar["Sources"]} >= records


def test_a_member_without_a_record_fails_its_dyad_and_names_the_fix(mini_hyper_bids, tmp_path,
                                                                      capsys):
    bids, pairs = mini_hyper_bids
    out = tmp_path / "qc"
    qc.main([str(bids), str(out), "participant", "--participant-label", "11", "--task-label",
             "hold", *SCREENING, *SKIP])
    with pytest.raises(SystemExit) as exit_:
        qc.main([str(bids), str(out), "group", "--pairs-csv", str(pairs), "--task-label", "hold",
                 *SKIP])
    assert exit_.value.code == 1
    err = capsys.readouterr().err
    assert "sub-12: no raw QC record" in err and "participant" in err
    # the cohort page is still written for the subjects there are
    assert (out / "desc-subjects_report.html").exists()


def test_the_motion_panel_corrects_each_member_as_its_record_says(mini_hyper_bids, tmp_path):
    bids, pairs = mini_hyper_bids
    figures = {}
    for method in ("tddr", "none"):
        out = tmp_path / method
        raw_qc(bids, out, pairs, participant=(*SCREENING, "--task-label", "hold",
                                              "--motion-correction", method),
               group=("--task-label", "hold"))
        figures[method] = {p.name for p in (out / "group-G01" / "figures").glob("*motion*")}
    assert "group-G01_task-hold_desc-rawmotionafter_nirs.html" in figures["tddr"]
    assert "group-G01_task-hold_desc-rawmotionafter_nirs.html" not in figures["none"]


@pytest.mark.parametrize("flags, level, says", [
    (("participant", "--pairs-csv", "p.csv"), "participant", "--pairs-csv is a group-level"),
    (("group", "--sci-threshold", "0.8"), "group", "--sci-threshold is a participant-level"),
])
def test_a_flag_of_the_other_level_is_refused(flags, level, says, capsys):
    with pytest.raises(SystemExit) as exit_:
        qc.main(["/bids", "/out", *flags])
    assert exit_.value.code == 1
    assert says in capsys.readouterr().err


def test_a_participant_run_is_rerun_from_its_record(mini_hyper_bids, tmp_path):
    bids, _pairs = mini_hyper_bids
    first, again = tmp_path / "first", tmp_path / "again"
    qc.main([str(bids), str(first), "participant", "--participant-label", "11", "--task-label",
             "hold", *SCREENING, "--motion-correction", "none", *SKIP])
    (record,) = (first / "sub-11" / "logs").glob("*_desc-raw.toml")
    qc.main([str(bids), str(again), "participant", "--config", str(record), *SKIP])
    params = [json.loads(next(t.glob("sub-11/nirs/*_desc-sqmraw_qc.json")).read_text())["parameters"]
              for t in (first, again)]
    assert params[0] == params[1]
    assert params[1]["motion_correction"] == "none"
    # the record's selection came with it: one subject, not the dataset
    assert sorted(p.name for p in again.glob("sub-*")) == ["sub-11"]


@pytest.mark.parametrize("entry", ["main", "main_pair_null"])
def test_nirspipe_hyper_refuses_to_write_into_the_tree_it_reads(entry, tmp_path, capsys):
    from nirspipe.cli import hyper

    tree = tmp_path / "shared"
    tree.mkdir()
    (tmp_path / "pairs.csv").write_text("group_id,subject_id,task\nG01,sub-a,main\n")
    with pytest.raises(SystemExit) as exit_:
        getattr(hyper, entry)([str(tree), str(tree), "group",
                               "--pairs-csv", str(tmp_path / "pairs.csv")])
    assert exit_.value.code != 0
    assert "are the same" in capsys.readouterr().err
    assert not any(tree.iterdir())
