"""The page for `nirspipe-hyper-groupnull`: one task's cohort tests, drawn off the tables on disk.

Rebuilt from every ``desc-byoccasion`` / ``desc-cohort`` pair at the tree's root for the
task, so a run for a second chromophore or null adds its section rather than replacing the
page.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nirspipe.io.derivatives import entity_of
from nirspipe.io.naming import derivative_path
from nirspipe.io.tables import read_table
from nirspipe.qc.boilerplate import collect_software_versions
from nirspipe.qc.common.figure_io import _save_figure_html
from nirspipe.qc.common.report_shell import footer_vars, guard, note, page_vars, render
from nirspipe.qc.figures.hyper.groupnull_figures import (
    ALPHA, build_occasion_panels, build_region_lift, p_column, whole_level,
)
from nirspipe.utils.logging import get_logger

logger = get_logger("qc.groupnull_report")

_MEASURES = {"wtc": "WTC", "isc": "ISC"}
_NULLS = {"pair": "re-paired", "phase": "phase-scrambled"}
_VALUE_AXIS = {"coherence": "WTC", "r_z": "ISC (Fisher z)", "abs_r_z": "|ISC| (Fisher z)"}
_TABLE_COLUMNS = ("condition", "test", "real", "null_mean", "lift", "p", "n_occasions")


def _paired_caption(row: "pd.Series | None", p_col: str) -> str:
    """``12/15 above · paired t(14) = 2.90, p = 0.006``, or what is missing instead."""
    if row is None:
        return ""
    above = f"{int(row['n_positive'])}/{int(row['n_occasions'])} above"
    if not np.isfinite(row.get("t", np.nan)):
        return f"{above} · no paired test under three occasions"
    p = row.get(p_col, row.get("p"))
    star = " *" if p < ALPHA else ""
    return f"{above} · paired t({int(row['df'])}) = {row['t']:.2f}, {p_col} = {p:.3g}{star}"


def _section(byocc_path: Path, fig_dir: Path, errors: list, scope: str) -> "dict | None":
    cohort_path = byocc_path.with_name(byocc_path.name.replace("desc-byoccasion", "desc-cohort"))
    byocc = read_table(byocc_path)
    cohort = read_table(cohort_path) if cohort_path.exists() else pd.DataFrame()
    value = next((c for c in _VALUE_AXIS if c in byocc.columns), None)
    if value is None:
        return None
    entities = {k: entity_of(byocc_path.name, k)
                for k in ("chromo", "task", "cond", "null", "stat")}
    measure = _MEASURES.get(entities["stat"], entities["stat"])
    title = (f"{measure}, {_NULLS.get(entities['null'], entities['null'])} null, "
             f"{(entities['chromo'] or '').upper()}")
    p_col = p_column(cohort) if not cohort.empty else "p"

    whole = whole_level(cohort) if not cohort.empty else cohort
    paired = (whole[whole["test"] == "paired"].set_index("condition")
              if not whole.empty else pd.DataFrame())
    captions = {cond: _paired_caption(row, p_col) for cond, row in paired.iterrows()}

    def _name(desc: str) -> str:
        return derivative_path(fig_dir, "relmat", ".html", chromophore=entities["chromo"],
                               task=entities["task"], condition=entities["cond"],
                               nulldist=entities["null"], statistic=entities["stat"],
                               desc=desc).name

    figures: dict = {}
    with guard(f"{title} occasions", errors, scope):
        fig = build_occasion_panels(byocc, value, _VALUE_AXIS[value], captions)
        if fig is not None:
            fname = _name("occasions")
            figures["occasions"] = {"src": f"figures/{fname}",
                                    "h": _save_figure_html(fig, fig_dir / fname)}
    with guard(f"{title} regions", errors, scope):
        fig = build_region_lift(cohort, _VALUE_AXIS[value]) if not cohort.empty else None
        if fig is not None:
            fname = _name("regions")
            figures["regions"] = {"src": f"figures/{fname}",
                                  "h": _save_figure_html(fig, fig_dir / fname)}

    rows = []
    for _, r in whole.iterrows():
        rows.append({"condition": r["condition"], "test": r["test"],
                     "real": f"{r[value]:.4f}", "null_mean": f"{r['null_mean']:.4f}",
                     "lift": f"{r['lift']:+.4f}",
                     "p": "" if pd.isna(r.get(p_col)) else f"{r[p_col]:.3g}",
                     "n_occasions": int(r["n_occasions"])})
    return {"title": title, "anchor": byocc_path.stem.replace("_desc-byoccasion_relmat", ""),
            "figures": figures, "rows": rows, "p_col": p_col,
            "pairings": whole["pairings"].iloc[0] if not whole.empty else "",
            "byoccasion": byocc_path.name,
            "cohort": cohort_path.name if cohort_path.exists() else None}


def write_groupnull_report(output_dir: Path, task: str) -> "Path | None":
    """``task-<task>_desc-groupnull_report.html`` at the root, or None with no tables."""
    output_dir = Path(output_dir)
    tables = sorted(p for p in output_dir.glob("*_desc-byoccasion_relmat.tsv")
                    if entity_of(p.name, "task") == task)
    if not tables:
        return None
    errors: list = []
    notes: list = []
    scope = f"task-{task}"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    for path in tables:
        with guard(f"section {path.name}", errors, scope):
            section = _section(path, fig_dir, errors, scope)
            if section is None:
                note(notes, scope, f"{path.name} carries no value column this page reads")
                continue
            sections.append(section)

    versions = collect_software_versions()
    html = render(
        "groupnull_report.html.j2",
        **page_vars(title=f"nirspipe cohort test, task {task} ({output_dir.name})",
                    heading="nirspipe Cohort test (hyperscanning)",
                    nav_meta=[("task", task), ("sections", len(sections))]),
        **footer_vars(versions=versions, errors=errors, notes=notes),
        sections=sections,
        nav_sections=[(s["anchor"], s["title"]) for s in sections],
        table_columns=_TABLE_COLUMNS,
        alpha=ALPHA,
    )
    out = output_dir / derivative_path("", "report", ".html", task=task, desc="groupnull").name
    out.write_text(html, encoding="utf-8")
    logger.info("group-null page: %s", out)
    return out
