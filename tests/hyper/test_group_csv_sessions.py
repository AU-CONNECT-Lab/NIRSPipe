"""A pairs CSV naming one group and task in two sessions keeps them two groups, end to end."""

import json
import re

import pandas as pd
import pytest

from fnirs_pipe.pipeline.hyper.group_io import parse_group_csv

SESSIONS = ("a", "b")
TASK = "mixed"
PHYS = ["--dpf", "6", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
        "--sci-threshold", "0.8"]


def test_two_sessions_of_one_group_are_not_merged_into_one(tmp_path):
    csv = tmp_path / "pairs.csv"
    csv.write_text("group_id,subject_id,task,session\n"
                   "G1,sub-01,main,a\nG1,sub-02,main,a\n"
                   "G1,sub-01,main,b\nG1,sub-02,main,b\n")
    groups = parse_group_csv(csv)
    assert sorted(groups) == [("G1", "main", "a"), ("G1", "main", "b")]
    for (_, _, ses), members in groups.items():
        assert [m.subject_id for m in members] == ["sub-01", "sub-02"]
        assert {m.session for m in members} == {ses}


def test_a_csv_without_sessions_keys_every_group_under_none(tmp_path):
    csv = tmp_path / "pairs.csv"
    csv.write_text("group_id,subject_id,task\nG1,sub-01,main\nG1,sub-02,main\n")
    assert list(parse_group_csv(csv)) == [("G1", "main", None)]


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """One dyad recorded in two sessions, through every dyad command."""
    from fnirs_pipe.cli import hyper as hyper_cli
    from fnirs_pipe.cli import qc as qc_cli
    from fnirs_pipe.cli import run as run_cli
    from tests._fingerprint import CLI_ARGS
    from tests._synth import make_hyper_dataset

    root = tmp_path_factory.mktemp("sessions")
    bids, pairs = make_hyper_dataset(root, tasks=(TASK,), sessions=SESSIONS)
    deriv, hyper = root / "deriv", root / "hyper"
    run_cli.main([str(bids), str(deriv), "participant", *CLI_ARGS, "--skip-bids-validation"])
    qc_cli.main(["hyper-raw", str(bids), str(hyper), "group", "--pairs-csv", str(pairs),
                 "--skip-bids-validation", "--seed", "0", *PHYS])
    hyper_cli.main([str(deriv), str(hyper), "group", "--pairs-csv", str(pairs),
                    "--wtc-fmin", "0.02", "--wtc-phase-null", "2", "--wtc-seed", "0"])
    hyper_cli.main_pair_null([str(deriv), str(hyper), "group", "--pairs-csv", str(pairs)])
    hyper_cli.main_group_null([str(hyper), "group", "--task-label", TASK, "--n-resample", "50",
                               "--seed", "0"])
    hyper_cli.main_merge([str(hyper), "group"])
    qc_cli.main(["cohort-hyper", str(hyper)])
    return hyper


def _nirs(tree, ses):
    return tree / "group-G01" / f"ses-{ses}" / "nirs"


def test_each_session_writes_its_own_tables_and_pages(tree):
    for ses in SESSIONS:
        stem = f"group-G01_ses-{ses}_task-{TASK}"
        for name in (f"{stem}_stat-wtc_relmat.tsv", f"{stem}_cond-all_stat-wtc_relmat.tsv",
                     f"{stem}_stat-isc_relmat.tsv",
                     f"{stem}_chromo-hbo_null-phase_stat-wtc_desc-level_relmat.npz",
                     f"{stem}_cond-all_null-pair_stat-wtc_relmat.tsv",
                     f"{stem}_desc-sqm_qc.json"):
            assert (_nirs(tree, ses) / name).exists(), name
        assert (tree / "group-G01" / f"{stem}_report.html").exists()
        assert (tree / "group-G01" / f"{stem}_desc-raw_report.html").exists()
        assert (tree / "group-G01" / "logs" / f"{stem}.toml").exists()
    assert not list((tree / "group-G01").glob("nirs/*_relmat.tsv"))


def test_each_session_s_table_is_read_from_that_session_only(tree):
    values = []
    for ses in SESSIONS:
        tsv = _nirs(tree, ses) / f"group-G01_ses-{ses}_task-{TASK}_stat-wtc_relmat.tsv"
        sources = json.loads(tsv.with_suffix(".json").read_text())["Sources"]
        assert len(sources) == 2 and all(f"_ses-{ses}_" in s for s in sources), sources
        values.append(pd.read_csv(tsv, sep="\t")["coherence"].to_numpy())
    assert not (values[0] == values[1]).all()


def test_the_index_lists_both_sessions_with_working_links(tree):
    html = (tree / "group-G01" / "group-G01_desc-index_report.html").read_text(encoding="utf-8")
    assert "<th>Session</th>" in html
    hrefs = set(re.findall(r'href="([^"#]+)"', html))
    for ses in SESSIONS:
        assert f"group-G01_ses-{ses}_task-{TASK}_report.html" in hrefs
        assert f"ses-{ses}/nirs/group-G01_ses-{ses}_task-{TASK}_stat-wtc_relmat.tsv" in hrefs
    local = [h for h in hrefs if not h.startswith(("http", "../"))]
    assert local and all((tree / "group-G01" / h).exists() for h in local), local


def test_the_group_null_ranks_each_session_as_an_occasion(tree):
    occasions = set()
    for tsv in tree.glob("*.tsv"):
        frame = pd.read_csv(tsv, sep="\t")
        if "occasion" in frame.columns:
            occasions |= set(frame["occasion"].astype(str))
    assert occasions == {f"G01 ses-{ses}" for ses in SESSIONS}


def test_the_merged_table_carries_the_session(tree):
    merged = pd.read_csv(tree / "stat-wtc_relmat.tsv", sep="	", dtype=str)
    assert set(merged["session"]) == set(SESSIONS)


def test_the_cohort_page_has_a_row_per_session(tree):
    table = pd.read_csv(tree / "desc-groups_qc.tsv", sep="	", dtype=str)
    assert sorted(table["label"]) == [f"group-G01_ses-{ses}_task-{TASK}" for ses in SESSIONS]
