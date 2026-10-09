"""Members of one sitting with different session labels, joined by an occasion column."""

import json
import re

import pandas as pd
import pytest

from fnirs_pipe.exceptions import GroupCSVError
from fnirs_pipe.pipeline.hyper.group_io import parse_group_csv

TASK = "mixed"
OCCASION = "7"
# no member carries the occasion's label, so an output named off a member shows
MEMBER_SESSIONS = {"11": "3", "12": "1", "13": "2", "14": "4"}
GROUPS = {"G01": ("11", "12"), "G02": ("13", "14")}
PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
        "--sci-threshold", "0.8"]


def _csv(tmp_path, text):
    path = tmp_path / "pairs.csv"
    path.write_text(text)
    return path


def test_an_occasion_joins_members_with_different_sessions(tmp_path):
    groups = parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,session,occasion\n"
                                            "G1,sub-01,main,3,1\nG1,sub-02,main,1,1\n"))
    assert list(groups) == [("G1", "main", "1")]
    assert [(m.subject_id, m.session) for m in groups[("G1", "main", "1")]] == [
        ("sub-01", "3"), ("sub-02", "1")]


def test_different_sessions_without_an_occasion_are_refused_naming_the_column(tmp_path):
    with pytest.raises(GroupCSVError, match="occasion"):
        parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,session\n"
                                       "G1,sub-01,main,3\nG1,sub-02,main,1\n"))


def test_a_blank_occasion_falls_back_to_the_row_s_session(tmp_path):
    groups = parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,session,occasion\n"
                                            "G1,sub-01,main,3,1\nG1,sub-02,main,1,1\n"
                                            "G2,sub-03,main,a,\nG2,sub-04,main,a,\n"))
    assert sorted(groups) == [("G1", "main", "1"), ("G2", "main", "a")]


def test_an_occasion_that_is_not_a_bids_label_is_refused(tmp_path):
    with pytest.raises(GroupCSVError, match="not a BIDS label"):
        parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,session,occasion\n"
                                       "G1,sub-01,main,3,ses-1\nG1,sub-02,main,1,ses-1\n"))


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """Two dyads whose members each carry their own session label, through every dyad command."""
    from fnirs_pipe.cli import hyper as hyper_cli
    from fnirs_pipe.cli import qc as qc_cli
    from fnirs_pipe.cli import run as run_cli
    from tests._fingerprint import CLI_ARGS
    from tests._synth import make_hyper_dataset

    root = tmp_path_factory.mktemp("occasion")
    bids, pairs = make_hyper_dataset(root, groups=GROUPS, tasks=(TASK,),
                                     member_sessions=MEMBER_SESSIONS, occasion=OCCASION)
    deriv, hyper = root / "deriv", root / "hyper"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--skip-bids-validation"])
    qc_cli.main(["hyper-raw", str(bids), str(hyper), "group", "--pairs-csv", str(pairs),
                 "--skip-bids-validation", "--session-label", OCCASION, *PHYS])
    hyper_cli.main([str(deriv), str(hyper), "group", "--pairs-csv", str(pairs),
                    "--wtc-fmin", "0.02", "--wtc-phase-null", "2", "--wtc-seed", "0"])
    hyper_cli.main_pair_null([str(deriv), str(hyper), "group", "--pairs-csv", str(pairs)])
    hyper_cli.main_group_null([str(hyper), "group", "--task-label", TASK, "--n-resample", "50",
                               "--seed", "0"])
    hyper_cli.main_merge([str(hyper), "group"])
    qc_cli.main(["cohort-hyper", str(hyper)])
    return {"bids": bids, "pairs": pairs, "hyper": hyper}


def test_every_dyad_output_is_named_by_the_occasion(tree):
    hyper = tree["hyper"]
    for gid in GROUPS:
        stem = f"group-{gid}_ses-{OCCASION}_task-{TASK}"
        nirs = hyper / f"group-{gid}" / f"ses-{OCCASION}" / "nirs"
        for name in (f"{stem}_stat-wtc_relmat.tsv", f"{stem}_cond-all_stat-wtc_relmat.tsv",
                     f"{stem}_stat-isc_relmat.tsv",
                     f"{stem}_chromo-hbo_null-phase_stat-wtc_desc-level_relmat.npz",
                     f"{stem}_cond-all_null-pair_stat-wtc_relmat.tsv",
                     f"{stem}_desc-bad_qc.tsv", f"{stem}_desc-subject_qc.tsv",
                     f"{stem}_desc-channel_qc.tsv", f"{stem}_desc-sqm_qc.json"):
            assert (nirs / name).exists(), name
        assert (hyper / f"group-{gid}" / f"{stem}_report.html").exists()
        assert (hyper / f"group-{gid}" / f"{stem}_desc-raw_report.html").exists()
        assert (hyper / f"group-{gid}" / "logs" / f"{stem}.toml").exists()
        # nothing under a member's own session, and nothing session-less
        assert [p.name for p in (hyper / f"group-{gid}").glob("ses-*")] == [f"ses-{OCCASION}"]
        assert not (hyper / f"group-{gid}" / "nirs").exists()


def test_each_member_is_read_from_its_own_session(tree):
    for gid, members in GROUPS.items():
        stem = f"group-{gid}_ses-{OCCASION}_task-{TASK}"
        tsv = tree["hyper"] / f"group-{gid}" / f"ses-{OCCASION}" / "nirs" / f"{stem}_stat-wtc_relmat.tsv"
        sources = json.loads(tsv.with_suffix(".json").read_text())["Sources"]
        assert len(sources) == 2, sources
        for sub in members:
            assert any(f"sub-{sub}_ses-{MEMBER_SESSIONS[sub]}_" in s for s in sources), sources


def test_the_index_links_the_occasion_s_files(tree):
    group_dir = tree["hyper"] / "group-G01"
    html = (group_dir / "group-G01_desc-index_report.html").read_text(encoding="utf-8")
    hrefs = set(re.findall(r'href="([^"#]+)"', html))
    assert f"group-G01_ses-{OCCASION}_task-{TASK}_report.html" in hrefs
    local = [h for h in hrefs if not h.startswith(("http", "../"))]
    assert local and all((group_dir / h).exists() for h in local), local


def test_the_group_null_ranks_each_dyad_under_its_occasion(tree):
    occasions = set()
    for tsv in tree["hyper"].glob("*.tsv"):
        frame = pd.read_csv(tsv, sep="\t")
        if "occasion" in frame.columns:
            occasions |= set(frame["occasion"].astype(str))
    assert occasions == {f"{gid} ses-{OCCASION}" for gid in GROUPS}


def test_the_merged_table_and_the_cohort_page_carry_the_occasion(tree):
    merged = pd.read_csv(tree["hyper"] / "stat-wtc_relmat.tsv", sep="\t", dtype=str)
    assert set(merged["session"]) == {OCCASION}
    table = pd.read_csv(tree["hyper"] / "desc-groups_qc.tsv", sep="\t", dtype=str)
    assert sorted(table["label"]) == [f"group-{gid}_ses-{OCCASION}_task-{TASK}" for gid in GROUPS]


def test_hyper_raw_session_label_selects_occasions_not_member_sessions(tree, tmp_path):
    from fnirs_pipe.cli import qc as qc_cli

    with pytest.raises(SystemExit) as exc:
        qc_cli.main(["hyper-raw", str(tree["bids"]), str(tmp_path), "group",
                     "--pairs-csv", str(tree["pairs"]), "--skip-bids-validation",
                     "--session-label", MEMBER_SESSIONS["11"], *PHYS])
    assert exc.value.code == 1
    assert not list(tmp_path.glob("group-*"))
