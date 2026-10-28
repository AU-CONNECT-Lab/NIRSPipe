"""The cohort page's connectivity panels, read from each run's channel matrices.

Every matrix and channel table is written by hand below, so each expected value is read off
the rows: only kept long channels take part, a run without the matrix before its
regression is left out, and runs pool by task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nirspipe.qc.figures.subject.group_figures import build_fc_distributions
from nirspipe.qc.subject.group_writer import build_group_raw_report, fc_distributions
from nirspipe.qc.subject.record_io import write_record

CHANNELS = ("S1_D1", "S2_D2", "S3_D3", "S4_D4")


def _channel_table(nirs, label, bad=()):
    rows = ["name\tseparation\tis_bad"]
    for pair in CHANNELS:
        for wl in (760, 850):
            sep = "short" if pair == "S3_D3" else "long"
            rows.append(f"{pair} {wl}\t{sep}\t{pair in bad}")
    (nirs / f"{label}_desc-channel_qc.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _matrix(nirs, name, chroma, r):
    """``r`` maps a pair of channels to its correlation; the rest of the matrix is 0."""
    names = [f"{c} {chroma}" for c in CHANNELS]
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for (a, b), v in r.items():
        m.loc[f"{a} {chroma}", f"{b} {chroma}"] = m.loc[f"{b} {chroma}", f"{a} {chroma}"] = v
    m.index.name = "channel"
    m.to_csv(nirs / name, sep="\t", na_rep="n/a")


def _run(out, sub, label, before, after, bad=(), with_before=True):
    nirs = out / f"sub-{sub}" / "nirs"
    nirs.mkdir(parents=True, exist_ok=True)
    _channel_table(nirs, label, bad)
    for chroma in ("hbo", "hbr"):
        _matrix(nirs, f"{label}_chromo-{chroma}_stat-pearson_relmat.tsv", chroma, after)
        if with_before:
            _matrix(nirs, f"{label}_chromo-{chroma}_stat-pearson_desc-filtered_relmat.tsv",
                    chroma, before)
    write_record(nirs / f"{label}_desc-sqm_qc.json",
                 {"step": "sqm", "raw": {"psp_mean": 0.3, "gvtd_p95": 1e-3}})


def _tree(out):
    # S3_D3 is short, so its 0.9s never count
    _run(out, "01", "sub-01_task-rest_acq-a", {("S1_D1", "S2_D2"): 0.8, ("S1_D1", "S3_D3"): 0.9},
         {("S1_D1", "S2_D2"): 0.3, ("S2_D2", "S4_D4"): -0.1})
    # S4_D4 rejected through one wavelength row's pair
    _run(out, "02", "sub-02_task-rest_acq-b", {("S1_D1", "S2_D2"): 0.6},
         {("S1_D1", "S2_D2"): 0.2, ("S1_D1", "S4_D4"): 0.7}, bad=("S4_D4",))
    # no regression ran: no matrix before it, so no curve
    _run(out, "03", "sub-03_task-rest", {}, {("S1_D1", "S2_D2"): 0.5}, with_before=False)


def test_each_run_gives_its_kept_long_channels_pairs_before_and_after(tmp_path):
    _tree(tmp_path)
    by_task = fc_distributions(tmp_path)
    assert list(by_task) == ["task-rest"], "acquisitions pool under their task"
    rest = by_task["task-rest"]
    assert rest["runs"] == ["sub-01_task-rest_acq-a", "sub-02_task-rest_acq-b"]
    values = rest["values"]
    # sub-01: S1, S2, S4 long and kept -> pairs (S1,S2), (S1,S4), (S2,S4)
    assert sorted(values[("hbo", "before")][0]) == [0.0, 0.0, 0.8]
    assert sorted(values[("hbo", "after")][0]) == [-0.1, 0.0, 0.3]
    # sub-02: S4 rejected -> (S1,S2) alone
    assert list(values[("hbo", "before")][1]) == [0.6]
    assert list(values[("hbr", "after")][1]) == [0.2]


def test_the_figure_peaks_where_the_runs_sit(tmp_path):
    nirs = tmp_path
    for k in range(3):
        _run(nirs, f"0{k}", f"sub-0{k}_task-rest", {p: 0.6 for p in
             [("S1_D1", "S2_D2"), ("S1_D1", "S4_D4"), ("S2_D2", "S4_D4")]},
             {p: 0.1 for p in [("S1_D1", "S2_D2"), ("S1_D1", "S4_D4"), ("S2_D2", "S4_D4")]})
    fig = build_fc_distributions(fc_distributions(tmp_path))
    medians = [t for t in fig.data if t.name == "cohort median"]
    # rows HbO then HbR, columns before then after
    assert [t.line.color for t in medians] == ["#e74c3c", "#e74c3c", "#3498db", "#3498db"]
    peaks = [float(np.asarray(t.x)[np.argmax(t.y)]) for t in medians]
    # within the bin either side of the value, as an edge on it may fall either way
    assert peaks[0] == pytest.approx(0.6, abs=0.02) and peaks[2] == peaks[0]
    assert peaks[1] == pytest.approx(0.1, abs=0.02) and peaks[3] == peaks[1]
    assert fig.layout.yaxis.range[1] >= max(max(t.y) for t in medians)


def test_the_page_draws_it_and_leaves_it_out_without_a_regression(tmp_path):
    _tree(tmp_path / "a")
    html = build_group_raw_report(tmp_path / "a").read_text(encoding="utf-8")
    assert '<section id="Connectivity"' in html and 'href="#Connectivity"' in html
    assert "figures/desc-fcdistribution_nirs.html" in html
    _run(tmp_path / "b", "01", "sub-01_task-rest", {}, {("S1_D1", "S2_D2"): 0.5},
         with_before=False)
    html = build_group_raw_report(tmp_path / "b").read_text(encoding="utf-8")
    assert 'id="Connectivity"' not in html
