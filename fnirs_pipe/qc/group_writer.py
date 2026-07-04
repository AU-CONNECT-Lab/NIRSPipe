"""Group-level QC aggregation: glob per-subject (or per-group) SQM JSONs into
TSV + an HTML viewer with heatmap / boxplots / sortable table / outlier panel."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.qc.figure_io import _save_figure_html
from fnirs_pipe.qc.figures.group_figures import (
    build_boxplot_per_metric,
    build_heatmap,
    build_time_subject_heatmap,
    detect_outliers,
)

# Per-window metrics that should produce a time × subject heatmap.
_WINDOWED_METRICS = [
    ("sci",  "sci_per_window",  "sci_window_times_s",  "SCI per window"),
    ("psp",  "psp_per_window",  "psp_window_times_s",  "PSP per window"),
    ("gvtd", "gvtd_per_window", "gvtd_window_times_s", "GVTD per window"),
]
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.group_writer")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BASE_CSS     = (_TEMPLATE_DIR / "_base.css").read_text(encoding="utf-8")


def _scalars(sqm: dict) -> dict:
    """Keep numeric scalar fields only (drop dicts/lists/strings)."""
    return {k: v for k, v in sqm.items() if isinstance(v, (int, float))}


def _bids_name_from_sqm_path(path: Path) -> str:
    return path.name.removesuffix("_desc-sqm_nirs.json")


def _collect_sqm(
    output_dir: Path, entity_glob: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Glob SQM JSONs under output_dir matching entity prefix (e.g. 'sub-*' or 'group-*').

    Returns (df, full_rows):
      - df:        scalar SQM columns (bids_name + numeric scalars), for TSV/heatmap/boxplot
      - full_rows: each row keeps the full SQM dict (incl. windowed list fields)
    """
    full_rows: list[dict] = []
    for sqm_path in sorted(output_dir.glob(f"{entity_glob}/**/nirs/*_desc-sqm_nirs.json")):
        try:
            sqm = json.loads(sqm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skip %s: %s", sqm_path, exc)
            continue
        full_rows.append({"bids_name": _bids_name_from_sqm_path(sqm_path), **sqm})

    if not full_rows:
        return pd.DataFrame(columns=["bids_name"]), []
    scalar_rows = [{"bids_name": r["bids_name"], **_scalars(r)} for r in full_rows]
    cols = ["bids_name"] + sorted({k for r in scalar_rows for k in r if k != "bids_name"})
    return pd.DataFrame(scalar_rows, columns=cols), full_rows


def _render_group(
    output_dir: Path,
    out_stem: str,
    title: str,
    df: pd.DataFrame,
    full_rows: list[dict],
    template_name: str = "group_report.html.j2",
) -> Path:
    """Render TSV + HTML for an already-collected group of SQM rows.

    Shared by `_build_group` (globs SQM JSONs) and `window_writer` (recomputes SQM
    after cropping). Empty df renders an empty report (logged as warning)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        logger.warning("rendering empty group report: %s", out_stem)

    tsv_path = output_dir / f"{out_stem}.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info("group TSV  -> %s (rows=%d, cols=%d)", tsv_path, len(df), len(df.columns))

    metric_cols = [c for c in df.columns if c != "bids_name"]
    fig_dir = output_dir / out_stem
    fig_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict = {}

    def _save(name: str, desc: str, fig) -> None:
        if fig is None:
            return
        fname = f"{out_stem}_desc-{desc}_nirs.html"
        h = _save_figure_html(fig, fig_dir / fname)
        figure_paths[name] = {"src": f"{out_stem}/{fname}", "h": h}

    if not df.empty and metric_cols:
        _save("heatmap", "heatmap", build_heatmap(df, metric_cols))
        _save("boxplot", "boxplot", build_boxplot_per_metric(df, metric_cols))

    windowed_panels: list[dict] = []
    for key, val_field, time_field, panel_title in _WINDOWED_METRICS:
        if not any(val_field in r and r[val_field] for r in full_rows):
            continue
        fig = build_time_subject_heatmap(full_rows, val_field, time_field, panel_title)
        if fig is None:
            continue
        _save(f"window_{key}", f"window{key}", fig)
        windowed_panels.append({"key": key, "title": panel_title})

    outliers = detect_outliers(df, metric_cols) if metric_cols else {}

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template(template_name).render(
        base_css=_BASE_CSS,
        title=title,
        n_rows=len(df),
        n_metrics=len(metric_cols),
        figure_paths=figure_paths,
        windowed_panels=windowed_panels,
        tsv_name=tsv_path.name,
        table_columns=list(df.columns),
        table_rows=df.values.tolist(),
        outliers=outliers,
    )
    html_path = output_dir / f"{out_stem}.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info("group HTML -> %s", html_path)
    return html_path


def _build_group(
    output_dir: Path,
    entity_glob: str,
    out_stem: str,
    title: str,
    template_name: str = "group_report.html.j2",
) -> Path:
    """Glob SQM JSONs and render group report."""
    df, full_rows = _collect_sqm(output_dir, entity_glob)
    return _render_group(output_dir, out_stem, title, df, full_rows, template_name)


def rows_to_dataframe(full_rows: list[dict]) -> pd.DataFrame:
    """Public helper for callers (e.g. window_writer) that pre-compute SQM rows."""
    if not full_rows:
        return pd.DataFrame(columns=["bids_name"])
    scalar_rows = [{"bids_name": r["bids_name"], **_scalars(r)} for r in full_rows]
    cols = ["bids_name"] + sorted({k for r in scalar_rows for k in r if k != "bids_name"})
    return pd.DataFrame(scalar_rows, columns=cols)


def build_group_raw_report(output_dir: Path) -> Path:
    """Aggregate all sub-XX/nirs/...desc-sqm_nirs.json into group_nirs.{tsv,html}."""
    return _build_group(
        output_dir, entity_glob="sub-*",
        out_stem="group_nirs",
        title="Group-level QC (individual)",
    )


def build_group_hyper_raw_report(output_dir: Path) -> Path:
    """Aggregate all group-XX/nirs/...desc-sqm_nirs.json into group_hyper_nirs.{tsv,html}."""
    return _build_group(
        output_dir, entity_glob="group-*",
        out_stem="group_hyper_nirs",
        title="Group-level QC (hyperscanning)",
    )
