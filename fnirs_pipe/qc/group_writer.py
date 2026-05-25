"""Group-level QC aggregation: glob per-subject (or per-group) IQM JSONs into
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
    detect_outliers,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.group_writer")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BASE_CSS     = (_TEMPLATE_DIR / "_base.css").read_text(encoding="utf-8")


def _scalars(iqm: dict) -> dict:
    """Keep numeric scalar fields only (drop dicts/lists/strings)."""
    return {k: v for k, v in iqm.items() if isinstance(v, (int, float))}


def _bids_name_from_iqm_path(path: Path) -> str:
    return path.name.removesuffix("_desc-iqm_nirs.json")


def _collect_iqm(
    output_dir: Path, entity_glob: str,
) -> pd.DataFrame:
    """Glob IQM JSONs under output_dir matching entity prefix (e.g. 'sub-*' or 'group-*').

    Each row has bids_name + all scalar IQM columns. Union schema across rows.
    """
    rows: list[dict] = []
    for iqm_path in sorted(output_dir.glob(f"{entity_glob}/**/nirs/*_desc-iqm_nirs.json")):
        try:
            iqm = json.loads(iqm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skip %s: %s", iqm_path, exc)
            continue
        row = {"bids_name": _bids_name_from_iqm_path(iqm_path), **_scalars(iqm)}
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["bids_name"])
    cols = ["bids_name"] + sorted({k for r in rows for k in r if k != "bids_name"})
    return pd.DataFrame(rows, columns=cols)


def _build_group(
    output_dir: Path,
    entity_glob: str,
    out_stem: str,
    title: str,
    template_name: str = "group_report.html.j2",
) -> Path:
    """Write {out_stem}.tsv + {out_stem}.html into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _collect_iqm(output_dir, entity_glob)
    if df.empty:
        logger.warning("no IQM JSON found under %s/%s", output_dir, entity_glob)

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

    outliers = detect_outliers(df, metric_cols) if metric_cols else {}

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template(template_name).render(
        base_css=_BASE_CSS,
        title=title,
        n_rows=len(df),
        n_metrics=len(metric_cols),
        figure_paths=figure_paths,
        tsv_name=tsv_path.name,
        table_columns=list(df.columns),
        table_rows=df.values.tolist(),
        outliers=outliers,
    )
    html_path = output_dir / f"{out_stem}.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info("group HTML -> %s", html_path)
    return html_path


def build_group_raw_report(output_dir: Path) -> Path:
    """Aggregate all sub-XX/nirs/...desc-iqm_nirs.json into group_nirs.{tsv,html}."""
    return _build_group(
        output_dir, entity_glob="sub-*",
        out_stem="group_nirs",
        title="Group-level QC (individual)",
    )


def build_group_hyper_raw_report(output_dir: Path) -> Path:
    """Aggregate all group-XX/nirs/...desc-iqm_nirs.json into group_hyper_nirs.{tsv,html}."""
    return _build_group(
        output_dir, entity_glob="group-*",
        out_stem="group_hyper_nirs",
        title="Group-level QC (hyperscanning)",
    )
