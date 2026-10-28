"""The cohort page's count of subjects that kept each source-detector pair, per task.

Each run's channel table is written by hand below, so every expected count is read off the
rows: a pair is rejected when either wavelength is, a subject keeps it when any of its runs
of the task did, and a subject whose montage lacks the pair is not counted for it.
"""

from __future__ import annotations

import json

from nirspipe.cli import qc as qc_cli
from nirspipe.cli.workflows import run_group_level
from nirspipe.qc.subject.group_writer import (build_group_raw_report, channels_kept,
                                              optode_losses, region_losses)
from nirspipe.qc.subject.record_io import write_record
from tests.figure_accuracy._dyad import table_under

PAIRS = ("S1_D1", "S2_D2", "S3_D3")


def _channel_table(nirs, label, bad, suffix="_desc-channel_qc.tsv", pairs=PAIRS):
    """Two wavelength rows per pair; ``bad`` names the wavelength rows rejected."""
    rows = ["name\tseparation\tis_bad"]
    for pair in pairs:
        for wl in (760, 850):
            sep = "short" if pair == "S3_D3" else "long"
            rows.append(f"{pair} {wl}\t{sep}\t{f'{pair} {wl}' in bad}")
    nirs.mkdir(parents=True, exist_ok=True)
    (nirs / f"{label}{suffix}").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _tree(out):
    s1, s2, s3 = (out / f"sub-0{i}" / "nirs" for i in (1, 2, 3))
    # sub-01 rejects S1_D1 in run 1 through one wavelength, keeps it in run 2
    _channel_table(s1, "sub-01_task-rest_run-1", {"S1_D1 850"})
    _channel_table(s1, "sub-01_task-rest_run-2", set())
    _channel_table(s1, "sub-01_task-nback", {"S2_D2 760", "S2_D2 850"})
    # sub-02's montage has no S3_D3, and it rejects S1_D1 in its one run
    _channel_table(s2, "sub-02_task-rest", {"S1_D1 760"}, pairs=("S1_D1", "S2_D2"))
    # sub-03: raw QC rejects S2_D2, the pipeline's own table keeps it and wins
    _channel_table(s3, "sub-03_task-rest", {"S2_D2 760"}, suffix="_desc-rawchannel_qc.tsv")
    _channel_table(s3, "sub-03_task-rest", set())
    for nirs, label in ((s1, "sub-01_task-rest_run-1"), (s2, "sub-02_task-rest"),
                        (s3, "sub-03_task-rest")):
        write_record(nirs / f"{label}_desc-sqm_qc.json",
                     {"step": "sqm", "raw": {"psp_mean": 0.3, "gvtd_p95": 1e-3}})


def test_each_pair_counts_the_subjects_that_kept_it_per_task(tmp_path):
    _tree(tmp_path)
    kept = channels_kept(tmp_path)
    assert kept["tasks"] == ["task-nback", "task-rest"]
    assert kept["rows"] == [
        {"pair": "S1_D1", "separation": "long", "cells": ["1 of 1", "2 of 3"]},
        {"pair": "S2_D2", "separation": "long", "cells": ["0 of 1", "3 of 3"]},
        {"pair": "S3_D3", "separation": "short", "cells": ["1 of 1", "2 of 2"]},
    ]


def test_no_channel_table_leaves_the_section_out(tmp_path):
    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    write_record(nirs / "sub-01_task-rest_desc-sqm_qc.json",
                 {"step": "sqm", "raw": {"psp_mean": 0.3, "gvtd_p95": 1e-3}})
    assert channels_kept(tmp_path) == {}
    html = build_group_raw_report(tmp_path).read_text(encoding="utf-8")
    assert 'id="ChannelsKept"' not in html and 'href="#ChannelsKept"' not in html


def test_the_page_prints_the_counts_under_quality_metrics(tmp_path):
    _tree(tmp_path)
    page = build_group_raw_report(tmp_path)
    html = page.read_text(encoding="utf-8")
    assert (html.index('<h2 id="Metrics">') < html.index('<section id="ChannelsKept"')
            < html.index('<section id="Table"'))
    assert 'href="#ChannelsKept"' in html
    assert table_under(page, "Channels kept") == [
        ["Source-detector", "Separation", "task-nback", "task-rest"],
        ["S1_D1", "long", "1 of 1", "2 of 3"],
        ["S2_D2", "long", "0 of 1", "3 of 3"],
        ["S3_D3", "short", "1 of 1", "2 of 2"],
    ]


# ---- each optode's long channels rejected, pooled over the task's runs ----

SHARED = ("S1_D1", "S1_D2", "S2_D2", "S3_D3")


def _shared_tree(out):
    """S1 reaches D1 and D2, S2 reaches D2, S3_D3 is short."""
    s1, s2 = (out / f"sub-0{i}" / "nirs" for i in (1, 2))
    # sub-01 loses both of S1's channels, one wavelength each
    _channel_table(s1, "sub-01_task-rest", {"S1_D1 760", "S1_D2 850", "S3_D3 760"}, pairs=SHARED)
    _channel_table(s2, "sub-02_task-rest", set(), pairs=SHARED)
    _channel_table(s2, "sub-02_task-nback", {"S2_D2 760"}, pairs=SHARED)
    for nirs, label in ((s1, "sub-01_task-rest"), (s2, "sub-02_task-rest")):
        write_record(nirs / f"{label}_desc-sqm_qc.json",
                     {"step": "sqm", "raw": {"psp_mean": 0.3, "gvtd_p95": 1e-3}})


def test_each_optode_pools_its_long_channels_over_the_runs_of_a_task(tmp_path):
    _shared_tree(tmp_path)
    got = optode_losses(tmp_path)
    assert got["tasks"] == ["task-nback", "task-rest"]
    # rest: S1 2 of 2 then 0 of 2; D1 1 of 1 then 0 of 1; D2 1 of 2 then 0 of 2; S2 0 of 1
    # twice. nback: S2_D2 alone rejected. S3 and D3 have only the short pair
    assert got["rows"] == [
        {"optode": "S1", "cells": ["0% (0 of 2)", "50% (2 of 4)"]},
        {"optode": "S2", "cells": ["100% (1 of 1)", "0% (0 of 2)"]},
        {"optode": "D1", "cells": ["0% (0 of 1)", "50% (1 of 2)"]},
        {"optode": "D2", "cells": ["50% (1 of 2)", "25% (1 of 4)"]},
    ]


def test_the_page_prints_the_optodes_after_the_channels(tmp_path):
    _shared_tree(tmp_path)
    page = build_group_raw_report(tmp_path)
    html = page.read_text(encoding="utf-8")
    assert (html.index('<section id="ChannelsKept"') < html.index('<section id="Optodes"')
            < html.index('<section id="Table"'))
    assert 'href="#Optodes"' in html
    assert table_under(page, "Optodes")[0] == ["Optode", "task-nback", "task-rest"]
    assert table_under(page, "Optodes")[1] == ["S1", "0% (0 of 2)", "50% (2 of 4)"]


def test_no_channel_table_leaves_the_optodes_out(tmp_path):
    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    write_record(nirs / "sub-01_task-rest_desc-sqm_qc.json",
                 {"step": "sqm", "raw": {"psp_mean": 0.3, "gvtd_p95": 1e-3}})
    assert optode_losses(tmp_path) == {}
    html = build_group_raw_report(tmp_path).read_text(encoding="utf-8")
    assert 'id="Optodes"' not in html and 'href="#Optodes"' not in html


# ---- the runs in which each --roi-mapping region kept no long channel ----

# L is S1's two channels, R adds the short S3_D3 to S2_D2, F lists a pair no montage has
ROI_MAP = {"L": ["S1_D1", "S1_D2 hbo"], "R": ["S2_D2", "S3_D3"], "F": ["S9_D9"]}


def test_each_region_counts_the_runs_of_a_task_that_kept_none_of_it(tmp_path):
    _shared_tree(tmp_path)
    got = region_losses(tmp_path, ROI_MAP)
    assert got["tasks"] == ["task-nback", "task-rest"]
    # rest: sub-01 loses S1_D1 and S1_D2, sub-02 loses nothing; nback: sub-02 loses S2_D2,
    # R's only long channel. F has no channel in any run, so every run counts
    assert got["rows"] == [
        {"region": "L", "cells": [{"text": "0 of 1", "runs": []},
                                  {"text": "1 of 2", "runs": ["sub-01_task-rest"]}]},
        {"region": "R", "cells": [{"text": "1 of 1", "runs": ["sub-02_task-nback"]},
                                  {"text": "0 of 2", "runs": []}]},
        {"region": "F", "cells": [{"text": "1 of 1", "runs": ["sub-02_task-nback"]},
                                  {"text": "2 of 2",
                                   "runs": ["sub-01_task-rest", "sub-02_task-rest"]}]},
    ]


def test_no_map_leaves_the_regions_out(tmp_path):
    _shared_tree(tmp_path)
    assert region_losses(tmp_path, None) == {}
    html = build_group_raw_report(tmp_path).read_text(encoding="utf-8")
    assert 'id="Regions"' not in html and 'href="#Regions"' not in html


def test_the_page_prints_the_regions_before_the_channels(tmp_path):
    _shared_tree(tmp_path)
    page = build_group_raw_report(tmp_path, ROI_MAP)
    html = page.read_text(encoding="utf-8")
    assert (html.index('<h2 id="Metrics">') < html.index('<section id="Regions"')
            < html.index('<section id="ChannelsKept"'))
    assert 'href="#Regions"' in html
    assert table_under(page, "Regions") == [
        ["Region", "task-nback", "task-rest"],
        ["L", "0 of 1", "1 of 2"],
        ["R", "1 of 1", "0 of 2"],
        ["F", "1 of 1", "2 of 2"],
    ]
    assert 'title="sub-01_task-rest, sub-02_task-rest"' in html


def _map_file(tmp_path):
    path = tmp_path / "roi.json"
    path.write_text(json.dumps(ROI_MAP), encoding="utf-8")
    return path


def test_raw_qc_group_level_reads_the_map(tmp_path):
    out = tmp_path / "qc"
    _shared_tree(out)
    qc_cli.main([str(tmp_path / "bids"), str(out), "group", "--roi-mapping",
                 str(_map_file(tmp_path))])
    assert table_under(next(out.glob("*desc-subjects_report.html")), "Regions")[1] == [
        "L", "0 of 1", "1 of 2"]


def test_the_pipeline_group_level_reads_the_map(tmp_path):
    out = tmp_path / "out"
    _shared_tree(out)
    run_group_level({"output_dir": str(out), "roi_mapping": str(_map_file(tmp_path))})
    assert table_under(next(out.glob("*desc-subjects_report.html")), "Regions")[1] == [
        "L", "0 of 1", "1 of 2"]
