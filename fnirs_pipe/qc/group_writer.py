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
    build_grouped_boxes,
    build_heatmap,
    build_time_subject_heatmap,
    detect_outliers,
    group_metrics,
)
from fnirs_pipe.qc.sqm_record import SECTIONS
from fnirs_pipe.utils.logging import get_logger

# Click a strip point -> open that subject's raw report (sibling of the group HTML,
# one dir up from this figure iframe). Multi-run viewers open at their first run.
_STRIP_CLICK_JS = (
    "<script>(function(){function bind(){var gd=document.querySelector('.plotly-graph-div');"
    "if(!gd||!gd.on){return setTimeout(bind,150);}"
    "gd.on('plotly_click',function(e){var p=e.points&&e.points[0];if(!p||p.customdata==null)return;"
    "var b=Array.isArray(p.customdata)?p.customdata[0]:p.customdata;"
    "if(b)window.open('../'+b+'_desc-raw_nirs.html','_blank');});}bind();})();</script>"
)

# Per-window metrics that should produce a time × subject heatmap.
_WINDOWED_METRICS = [
    ("sci",  "sci_per_window",  "sci_window_times_s",  "SCI per window"),
    ("psp",  "psp_per_window",  "psp_window_times_s",  "PSP per window"),
    ("gvtd", "gvtd_per_window", "gvtd_window_times_s", "GVTD mean per window"),
    ("gvtd_p95", "gvtd_p95_per_window", "gvtd_window_times_s", "GVTD p95 (worst-moment) per window"),
    ("gvtd_filt", "gvtd_filt_per_window", "gvtd_window_times_s", "GVTD filtered (0.01-0.5 Hz) mean per window"),
    ("gvtd_filt_p95", "gvtd_filt_p95_per_window", "gvtd_window_times_s", "GVTD filtered (0.01-0.5 Hz) p95 per window"),
]

logger = get_logger("qc.group_writer")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BASE_CSS     = (_TEMPLATE_DIR / "_base.css").read_text(encoding="utf-8")


def _warn_on_mixed_windows(rows: list) -> None:
    """A time x subject heatmap only reads straight if every subject was binned the same way.

    prep-raw rows carry the scalar under a ``raw_`` prefix and pipeline rows do not, so both
    names are checked. Silence means either one window length across the cohort or records
    written before the length was stored.
    """
    lengths = {r.get("qc_window_s", r.get("raw_qc_window_s")) for r in rows}
    lengths.discard(None)
    if len(lengths) > 1:
        logger.warning(
            "cohort mixes QC window lengths %s; the per-window heatmaps put subjects with "
            "different window grids in one column",
            sorted(lengths),
        )


def _scalars(sqm: dict) -> dict:
    """Keep numeric scalar fields only (drop dicts/lists/strings).

    A sectioned record keeps its metrics one level down, so the known sections flatten to
    ``section_metric`` for the group table, where column names have to be unique. Only
    those sections are descended into: per_channel would explode into one column per
    channel.
    """
    flat = {k: v for k, v in sqm.items() if isinstance(v, (int, float))}
    for section in SECTIONS:
        values = sqm.get(section)
        if isinstance(values, dict):
            flat.update({f"{section}_{k}": v for k, v in values.items()
                         if isinstance(v, (int, float))})
    return flat


# The two records a run can leave behind, best first. `sqm` is the pipeline's sectioned
# record; `sqmraw` is what `fnirs-qc prep-raw` writes, measuring the original recording
# only. A run that saw both commands has both files.
_SQM_DESCS = ("sqm", "sqmraw")
_PREP_RAW_DESC = "sqmraw"


def _bids_name_from_sqm_path(path: Path) -> str:
    for desc in _SQM_DESCS:
        if (name := path.name.removesuffix(f"_desc-{desc}_nirs.json")) != path.name:
            return name
    return path.stem


def _sqm_row(bids_name: str, sqm: dict, desc: str) -> dict:
    """One group-table row, in the sectioned record's column vocabulary.

    prep-raw measures the original recording over every channel, which is exactly the
    sectioned record's ``raw`` view, so its scalars take the same ``raw_`` prefix and a
    cohort holding both kinds compares in one set of columns. Windowed series keep the
    names prep-raw writes: the time x subject heatmaps look them up by those.
    """
    if desc != _PREP_RAW_DESC:
        return {"bids_name": bids_name, **sqm}
    return {
        "bids_name": bids_name,
        **{(f"raw_{k}" if isinstance(v, (int, float)) else k): v for k, v in sqm.items()},
    }


def _collect_sqm(
    output_dir: Path, entity_glob: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """Glob SQM JSONs under output_dir matching entity prefix (e.g. 'sub-*' or 'group-*').

    One row per run, never one per file: a run that was processed by both the pipeline and
    `prep-raw` has two records, and the sectioned one wins because it is a superset.

    Returns (df, full_rows):
      - df:        scalar SQM columns (bids_name + numeric scalars), for TSV/heatmap/boxplot
      - full_rows: each row keeps the full SQM dict (incl. windowed list fields)
    """
    by_run: dict[str, tuple[Path, str]] = {}
    for desc in _SQM_DESCS:
        for sqm_path in sorted(output_dir.glob(f"{entity_glob}/**/nirs/*_desc-{desc}_nirs.json")):
            by_run.setdefault(_bids_name_from_sqm_path(sqm_path), (sqm_path, desc))

    full_rows: list[dict] = []
    for bids_name, (sqm_path, desc) in sorted(by_run.items()):
        try:
            sqm = json.loads(sqm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skip %s: %s", sqm_path, exc)
            continue
        full_rows.append(_sqm_row(bids_name, sqm, desc))

    return rows_to_dataframe(full_rows), full_rows


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

    def _save(name: str, desc: str, fig, extra_js: str = "") -> None:
        if fig is None:
            return
        fname = f"{out_stem}_desc-{desc}_nirs.html"
        h = _save_figure_html(fig, fig_dir / fname, extra_js=extra_js)
        figure_paths[name] = {"src": f"{out_stem}/{fname}", "h": h}

    box_panels: list[dict] = []
    if not df.empty and metric_cols:
        _, ordered_cols = group_metrics(metric_cols)
        _save("heatmap", "heatmap", build_heatmap(df, ordered_cols))
        for i, (box_title, fig) in enumerate(build_grouped_boxes(df, ordered_cols)):
            fname = f"{out_stem}_desc-box{i}_nirs.html"
            h = _save_figure_html(fig, fig_dir / fname, extra_js=_STRIP_CLICK_JS)
            box_panels.append({
                "src": f"{out_stem}/{fname}", "h": h,
                "w": int(getattr(fig.layout, "width", None) or 300), "title": box_title,
            })

    windowed_panels: list[dict] = []
    _warn_on_mixed_windows(full_rows)
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
        box_panels=box_panels,
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
