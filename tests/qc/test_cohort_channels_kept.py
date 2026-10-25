"""The cohort page's count of subjects that kept each source-detector pair, per task.

Each run's channel table is written by hand below, so every expected count is read off the
rows: a pair is rejected when either wavelength is, a subject keeps it when any of its runs
of the task did, and a subject whose montage lacks the pair is not counted for it.
"""

from __future__ import annotations

from nirspipe.qc.subject.group_writer import build_group_raw_report, channels_kept
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
