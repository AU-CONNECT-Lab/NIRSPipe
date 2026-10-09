"""The cohort test page: each panel draws its table, and the marks read the paired test."""

import re

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.io.naming import derivative_path
from fnirs_pipe.qc.figures.hyper.groupnull_figures import (
    build_occasion_panels, build_region_lift,
)
from fnirs_pipe.qc.hyper.groupnull_report import write_groupnull_report

OCCASIONS = ("G01", "G02", "G03", "G04")
CONDITIONS = ("ca", "cb")


def _byoccasion(pairings=("homologous", "all")) -> pd.DataFrame:
    rows = []
    for pairing in pairings:
        for c, cond in enumerate(CONDITIONS):
            for o, occ in enumerate(OCCASIONS):
                null = 0.30 + 0.01 * o
                # the all-pairings level is offset so a mix-up of the two shows
                real = null + (0.02 if (o + c) % 2 == 0 else -0.01) + (0.1 if pairing == "all" else 0)
                rows.append({"granularity": "whole", "level": "whole", "pairings": pairing,
                             "condition": cond, "occasion": occ, "coherence": real,
                             "null_mean": null, "null_sd": 0.01, "null_p95": null + 0.02,
                             "lift": real - null, "percentile": 50.0, "p": 0.5, "n_iter": 20})
    return pd.DataFrame(rows)


def _cohort(n_occasions=4, crossed=True, corrected=False) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        for test in ("paired", "resample"):
            row = {"granularity": "whole", "level": "whole", "pairings": "all",
                   "condition": cond, "test": test, "coherence": 0.4, "null_mean": 0.3,
                   "lift": 0.1, "p": 0.01, "n_occasions": n_occasions, "n_positive": 3}
            if test == "paired" and n_occasions >= 3:
                row.update(t=3.1, df=n_occasions - 1)
            rows.append(row)
    levels = ([("L>L", 0.04, 0.01), ("L>R", -0.03, 0.02), ("R>L", 0.01, 0.4), ("R>R", 0.02, 0.03)]
              if crossed else [("L", 0.04, 0.01), ("R", -0.02, 0.3)])
    for cond in CONDITIONS:
        for level, lift, p in levels:
            rows.append({"granularity": "roi", "level": level,
                         "pairings": "all" if crossed else "homologous", "condition": cond,
                         "test": "paired", "coherence": 0.3 + lift, "null_mean": 0.3,
                         "lift": lift, "p": p, "n_occasions": n_occasions, "n_positive": 2})
    frame = pd.DataFrame(rows)
    if corrected:
        # corrected p lifts L>R out of significance, so a mark must follow this column
        frame["p_fdr_bh"] = frame["p"].where(frame["level"] != "L>R", 0.2)
    return frame


# ---- the occasion panels ----

def test_each_condition_draws_its_occasions_against_their_null_means():
    table = _byoccasion()
    fig = build_occasion_panels(table, "coherence", "WTC")
    real = [t for t in fig.data if t.name == "real dyad"]
    assert len(real) == len(CONDITIONS)
    for cond, trace in zip(CONDITIONS, real):
        rows = table[(table.pairings == "all") & (table.condition == cond)]
        assert list(trace.x) == list(OCCASIONS)
        assert list(trace.y) == pytest.approx(rows.coherence.tolist())
        # hollow where the real value is not above its null mean
        hollow = [c == "white" for c in trace.marker.color]
        assert hollow == (rows.coherence <= rows.null_mean).tolist()


def test_the_all_pairings_level_is_drawn_when_the_null_was_crossed():
    drawn = [t for t in build_occasion_panels(_byoccasion(), "coherence", "WTC").data
             if t.name == "real dyad"][0]
    homologous = [t for t in build_occasion_panels(_byoccasion(("homologous",)), "coherence",
                                                   "WTC").data if t.name == "real dyad"][0]
    assert np.mean(drawn.y) == pytest.approx(np.mean(homologous.y) + 0.1)


def test_the_caption_goes_on_its_condition():
    fig = build_occasion_panels(_byoccasion(), "coherence", "WTC", {"cb": "3/4 above"})
    titles = [a.text for a in fig.layout.annotations]
    assert "cb  3/4 above" in titles and "ca" in titles


# ---- the region matrix ----

def test_a_crossed_region_matrix_holds_each_pair_s_lift_where_it_belongs():
    fig = build_region_lift(_cohort(), "WTC")
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == len(CONDITIONS)
    z = np.asarray(heat[0].z, dtype=float)
    regions = list(heat[0].y)
    assert z[regions.index("L"), regions.index("R")] == pytest.approx(-0.03)
    assert z[regions.index("R"), regions.index("L")] == pytest.approx(0.01)
    assert heat[0].zmid == 0.0 and heat[0].zmax == pytest.approx(0.04)


def test_dots_mark_the_paired_test_and_follow_the_corrected_p_when_there_is_one():
    def dots(frame):
        fig = build_region_lift(frame, "WTC")
        return {(y, x) for t in fig.data if t.type == "scatter" for x, y in zip(t.x, t.y)}

    assert dots(_cohort()) == {("L", "L"), ("L", "R"), ("R", "R")}
    assert dots(_cohort(corrected=True)) == {("L", "L"), ("R", "R")}


def test_homologous_regions_alone_make_a_regions_by_conditions_grid():
    fig = build_region_lift(_cohort(crossed=False), "WTC")
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == 1
    assert list(heat[0].x) == list(CONDITIONS) and list(heat[0].y) == ["L", "R"]
    dots = [t for t in fig.data if t.type == "scatter"][0]
    assert set(zip(dots.x, dots.y)) == {("ca", "L"), ("cb", "L")}


def test_no_region_rows_draw_no_region_matrix():
    frame = _cohort()
    assert build_region_lift(frame[frame.granularity == "whole"], "WTC") is None


# ---- the page ----

def _write(tree, byocc, cohort, chroma="hbo", null="pair", stat="wtc"):
    common = dict(chromophore=chroma, task="main", condition="all", nulldist=null,
                  statistic=stat)
    for frame, desc in ((byocc, "byoccasion"), (cohort, "cohort")):
        path = derivative_path(tree, "relmat", ".tsv", **common, desc=desc)
        frame.to_csv(path, sep="\t", index=False)
        path.with_suffix(".json").write_text("{}")


def test_the_page_has_a_section_per_table_pair_and_every_link_is_on_disk(tmp_path):
    _write(tmp_path, _byoccasion(), _cohort(corrected=True))
    _write(tmp_path, _byoccasion(), _cohort(), chroma="hbr", null="phase")
    page = write_groupnull_report(tmp_path, "main")
    html = page.read_text(encoding="utf-8")
    assert page.name == "task-main_desc-groupnull_report.html"
    assert "WTC, re-paired null, HBO" in html and "WTC, phase-scrambled null, HBR" in html
    assert "<th>p_fdr_bh</th>" in html
    assert "paired t(3) = 3.10" in (tmp_path / "figures").joinpath(
        next(p.name for p in (tmp_path / "figures").glob("*null-pair*occasions*"))).read_text(
        encoding="utf-8")
    links = set(re.findall(r'(?:src|href)="([^"#]+)"', html))
    local = [h for h in links if not h.startswith(("http", "data:"))]
    assert local and all((tmp_path / h).exists() for h in local), local


def test_under_three_occasions_the_caption_says_there_is_no_paired_test(tmp_path):
    byocc = _byoccasion()
    byocc = byocc[byocc.occasion.isin(OCCASIONS[:2])]
    _write(tmp_path, byocc, _cohort(n_occasions=2))
    write_groupnull_report(tmp_path, "main")
    fig = next((tmp_path / "figures").glob("*occasions*")).read_text(encoding="utf-8")
    assert "no paired test under three occasions" in fig


def test_another_task_s_tables_stay_off_the_page(tmp_path):
    _write(tmp_path, _byoccasion(), _cohort())
    assert write_groupnull_report(tmp_path, "other") is None
