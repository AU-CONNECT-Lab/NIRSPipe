"""The subject landing page: the two things it says that no single run's report can.

Which run stands apart from the subject's others on a metric, and which channels were
rejected in which run. Both are read back off disk from files the runs already wrote, so
these build a subject folder rather than running anything.
"""

import json

from fnirs_pipe.io.naming import report_name
from fnirs_pipe.qc.common.channel_table import CHANNEL_METRICS_SUFFIX
from fnirs_pipe.qc.common.report_shell import outlier_flags as _outlier_flags
from fnirs_pipe.qc.subject.subject_index import (
    _COLUMNS,
    _condition_hrefs,
    _links,
    collect_bad_channels,
    collect_runs,
)

TAB = chr(9)


def _run(sub_dir, task, *, sci=0.96, gvtd=9e-3, bad_pairs=(), channels=("S1_D1", "S2_D2")):
    """One run's record and channel metrics, the two files the index reads."""
    label = f"sub-01_task-{task}"
    nirs = sub_dir / "nirs"
    nirs.mkdir(parents=True, exist_ok=True)

    (nirs / f"{label}_desc-sqm_qc.json").write_text(json.dumps({
        "step": "sqm",
        "raw": {"sci_mean": sci, "channel_retention_rate": 1.0, "gvtd_p95": gvtd},
        "motion": {"motion_corrected_pct": 0.1},
        "preproc": {"hbo_hbr_corr_mean": 0.2},
    }))
    (nirs / f"{label}_desc-preproc_nirs.json").write_text(json.dumps({
        "step": "beer_lambert",
        "data": {"n_channels": 2 * len(channels), "n_bad": 2 * len(bad_pairs),
                 "sfreq": 10.0, "duration_s": 300.0},
    }))

    lines = [TAB.join(["name", "sci", "snr", "cv", "corr", "is_bad"])]
    for pair in channels:
        for wavelength in (760, 850):
            lines.append(TAB.join([f"{pair} {wavelength}", "0.9", "20", "0.1", "0.5",
                                   str(pair in bad_pairs)]))
    (nirs / (label + CHANNEL_METRICS_SUFFIX)).write_text("\n".join(lines) + "\n")
    return label


# ---- which run stands apart ----

def test_the_run_that_sits_apart_is_marked():
    assert _outlier_flags([0.96, 0.95, 0.96, 0.40, 0.97]) == [False] * 3 + [True, False]


def test_three_runs_are_too_few_to_compare():
    # with three values one of them is always the furthest out, and calling it an outlier
    # would mark a run on every three-task subject in the study
    assert _outlier_flags([0.96, 0.95, 0.10]) == [False, False, False]


def test_runs_that_agree_exactly_produce_no_marks():
    assert _outlier_flags([1.0, 1.0, 1.0, 1.0, 1.0]) == [False] * 5


def test_one_run_away_from_a_set_that_agrees_is_still_caught():
    # the median absolute deviation is zero here, so the scale has to come from elsewhere
    assert _outlier_flags([1.0, 1.0, 1.0, 1.0, 0.2]) == [False] * 4 + [True]


def test_two_runs_sharing_a_deviation_are_not_outliers():
    assert _outlier_flags([1.0, 0.95, 1.0, 1.0, 0.95]) == [False] * 5


def test_a_missing_value_is_neither_marked_nor_counted():
    assert _outlier_flags([0.96, None, 0.95, 0.96, 0.97]) == [False] * 5


# ---- which channels were rejected where ----

def test_only_the_rejected_pairs_get_a_row(tmp_path):
    pairs = ("S1_D1", "S2_D2", "S3_D3")
    labels = [
        _run(tmp_path, "rest", channels=pairs),
        _run(tmp_path, "hold", bad_pairs=("S2_D2",), channels=pairs),
    ]

    channels = collect_bad_channels(tmp_path, labels)
    assert [r["pair"] for r in channels["rejected"]] == ["S2_D2"]
    assert channels["rejected"][0]["bad_in"] == [False, True]
    assert channels["clean"] == ["S1_D1", "S3_D3"]
    assert (channels["n_pairs"], channels["n_clean"]) == (3, 2)


def test_the_two_wavelengths_of_a_pair_are_one_row(tmp_path):
    # the CSV has one row per wavelength; rejecting either rejects the optode
    labels = [_run(tmp_path, "rest", bad_pairs=("S1_D1",))]
    channels = collect_bad_channels(tmp_path, labels)
    assert len(channels["rejected"]) == 1
    assert channels["n_pairs"] == 2


def test_the_pair_rejected_most_often_comes_first(tmp_path):
    pairs = ("S1_D1", "S2_D2")
    labels = [
        _run(tmp_path, "rest", bad_pairs=("S2_D2",), channels=pairs),
        _run(tmp_path, "hold", bad_pairs=("S1_D1", "S2_D2"), channels=pairs),
    ]
    rows = collect_bad_channels(tmp_path, labels)["rejected"]
    assert [(r["pair"], r["n_bad"]) for r in rows] == [("S2_D2", 2), ("S1_D1", 1)]


def test_a_tree_without_channel_metrics_says_nothing(tmp_path):
    # the section is skipped rather than drawn empty, which is what an older tree gets
    (tmp_path / "nirs").mkdir()
    assert collect_bad_channels(tmp_path, ["sub-01_task-rest"]) == {}


# ---- the other products of a run ----

def test_only_the_artefacts_on_disk_are_linked(tmp_path):
    label = _run(tmp_path, "rest")
    (tmp_path / report_name(label, desc="mne")).write_text("")

    texts = [link["text"] for link in _links(tmp_path, label)]
    assert texts == ["MNE", "channels"]      # no provenance png, no aux table


def test_the_index_rows_carry_the_links_and_the_marks(tmp_path):
    for task, gvtd in (("a", 9e-3), ("b", 9e-3), ("c", 9e-3), ("d", 9e-3), ("e", 5e-1)):
        _run(tmp_path, task, gvtd=gvtd)

    rows = collect_runs(tmp_path)
    gvtd_column = next(i for i, (head, _, _) in enumerate(_COLUMNS) if head == "GVTD p95")
    assert [row["metrics"][gvtd_column]["flagged"] for row in rows] == [False] * 4 + [True]
    assert all(link["text"] == "channels" for row in rows for link in row["links"])


# ---- the condition pages, whose name the writer and this reader must agree on ----

def test_a_condition_page_is_found_where_the_report_writes_it(tmp_path):
    """The two ends of one name, bound together rather than spelled twice.

    They came apart once already: the writer still spelled `_desc-<slug>_qc.html` after the
    reports were renamed, the nav strip inside those pages already asked for the new name,
    and this reader looked for a third spelling. Nothing was red, and every per-condition
    link on the index and on the pages themselves was dead.
    """
    from fnirs_pipe.qc.subject.report import condition_page_name

    label = _run(tmp_path, "rest")
    page = tmp_path / condition_page_name(label, "game 1")
    page.write_text("")

    assert _condition_hrefs(tmp_path, label, ["game 1", "video"]) == [page.name, None]


def _write_raw_condition_pages(monkeypatch, sub_dir, run_label, report_stem, conditions):
    """Drive the raw viewer's page writer with the payloads stubbed, which is all it names."""
    from fnirs_pipe.qc.subject import condition_views, prep_raw_report as prr

    record = sub_dir / f"{run_label}_record.json"
    record.write_text(json.dumps({"by_condition": {c: {} for c in conditions}}))
    monkeypatch.setattr(condition_views, "condition_payloads",
                        lambda payload, **kw: [(c, {}) for c in conditions])
    out = sub_dir / report_name(report_stem, desc="raw")
    ctx = {"sqm_path": record, "fig_dir": sub_dir / "figures", "sci_scores": {},
           "bad_channels": set(), "channel_pairs": [], "series": {}, "cutoffs": {},
           "shell": prr._shell_vars([], out, sub_dir, 0.8)}
    prr._write_condition_views(ctx, {}, out, run_label, 0.8)


def test_a_raw_condition_page_is_found_where_prep_raw_writes_it(tmp_path, monkeypatch):
    """The raw viewer's writer and this reader, end to end.

    The writer took its name from the report's stem, `..._desc-raw_report` since the rename,
    while this reader still asked for `..._desc-raw_nirs`, so every raw condition link on
    the index was dead.
    """
    label = _run(tmp_path, "rest")
    _write_raw_condition_pages(monkeypatch, tmp_path, label, label, ["game 1", "video"])

    hrefs = _condition_hrefs(tmp_path, label, ["game 1", "video"])
    assert all(href and (tmp_path / href).exists() for href in hrefs)


def test_two_runs_in_one_raw_report_get_pages_of_their_own(tmp_path, monkeypatch):
    # one raw report holds every run of a task, so a page named off the report's stem
    # would leave the second run's condition written over the first's
    for run in ("01", "02"):
        _write_raw_condition_pages(monkeypatch, tmp_path, f"sub-01_task-rest_run-{run}",
                                   "sub-01_task-rest", ["game1"])

    pages = sorted(p.name for p in tmp_path.glob("*_cond-game1_desc-raw_report.html"))
    assert pages == ["sub-01_task-rest_run-01_cond-game1_desc-raw_report.html",
                     "sub-01_task-rest_run-02_cond-game1_desc-raw_report.html"]


def test_a_label_that_repeats_once_reduced_gets_no_second_page(tmp_path, monkeypatch):
    # its figures and URL fragments carry the same slug, so a page of its own would show
    # the first condition's files under the second one's numbers
    _write_raw_condition_pages(monkeypatch, tmp_path, "sub-01_task-rest", "sub-01_task-rest",
                               ["game-1", "game 1"])

    assert len(list(tmp_path.glob("*_cond-*_report.html"))) == 1
