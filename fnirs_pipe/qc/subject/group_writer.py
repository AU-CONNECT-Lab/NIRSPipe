"""Cohort-level QC aggregation: glob per-subject (or per-group) SQM JSONs into
TSV + an HTML viewer with heatmap / boxplots / sortable table / outlier panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.boilerplate.notes import section_note
from fnirs_pipe.qc.metrics.coupling import SCI_WINDOW_S
from fnirs_pipe.qc.common.figure_io import _save_figure_html
from fnirs_pipe.qc.figures.subject.group_figures import (
    SCORE_THRESHOLD,
    SMOOTH_S,
    _split_column,
    build_condition_matrix,
    build_condition_panels,
    build_deviation_strip,
    build_grouped_boxes,
    build_window_grid,
    condition_names,
    detect_outliers,
    deviation_scores,
    group_metrics,
)
from fnirs_pipe.qc.common.report_shell import (
    footer_vars,
    guard,
    note,
    page_vars,
    render,
)
from fnirs_pipe.io.naming import derivative_path
from fnirs_pipe.qc.subject.sqm_record import (
    OPTIONAL_SECTIONS,
    POST_BANDPASS_HAEMO_STAGES,
    PRE_BANDPASS_HAEMO_STAGE,
    SECTIONS,
    RECORD_SUFFIXES,
    SQM_DESCS,
    fill_skipped_long_sections,
)
from fnirs_pipe.utils.logging import get_logger

# Click a strip point -> open that subject's raw report, which lives in sub-<id>/ next to
# the rest of that subject's files. The figure is an iframe one dir down from the group HTML,
# so the hop is up and then into the subject folder, whose name is the label's first field.
# Multi-run viewers open at their first run.
_STRIP_CLICK_JS = (
    "<script>(function(){function bind(){var gd=document.querySelector('.plotly-graph-div');"
    "if(!gd||!gd.on){return setTimeout(bind,150);}"
    "gd.on('plotly_click',function(e){var p=e.points&&e.points[0];if(!p||p.customdata==null)return;"
    "var b=Array.isArray(p.customdata)?p.customdata[0]:p.customdata;"
    "if(b)window.open('../'+b.split('_')[0]+'/'+b+'_desc-raw_report.html','_blank');});}"
    "bind();})();</script>"
)

logger = get_logger("qc.group_writer")

# Below this many runs a spread is not measured, it is drawn: the box, the robust z-score and
# the Tukey fence each need a middle to sit in. The panels still render, carrying a line that
# says what they cannot tell the reader at this cohort size.
SMALL_COHORT_N = 5

# What the summary reports, in this order. Each is read off the whole-channel-set column of
# the earliest stage the cohort carries, that being the one every record has.
_HEADLINE_METRICS = ("sci_win_mean", "psp_mean", "snr_mean", "cv_mean",
                     "channel_retention_rate", "gvtd_pct_above_thresh")
_STAGE_ORDER = ("raw", "motion", "motion_post", "preproc", "filtered", "resampled", "errts")

# How a record's desc reads in the summary.
_DESC_LABELS = {"sqm": "pipeline", "sqmraw": "prep-raw"}



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


def _bandpass_span_metrics(columns: list) -> list:
    """Metric stems this table carries on both sides of the bandpass, sorted.

    ::

        ["preproc_gcor_hbo", "errts_gcor_hbo", "raw_sci_mean"]  ->  ["gcor_hbo"]
    """
    def stems(stage: str) -> set:
        return {c[len(stage) + 1:] for c in columns if c.startswith(f"{stage}_")}

    post = set().union(*(stems(s) for s in POST_BANDPASS_HAEMO_STAGES))
    return sorted(stems(PRE_BANDPASS_HAEMO_STAGE) & post)


def _scalars(sqm: dict) -> dict:
    """Keep numeric scalar fields only (drop dicts/lists/strings).

    A sectioned record keeps its metrics one level down, so the known sections flatten to
    ``section_metric`` for the group table, where column names have to be unique. Only
    those sections are descended into: per_channel would explode into one column per
    channel.
    """
    flat = {k: v for k, v in sqm.items() if isinstance(v, (int, float))}
    for section in (*SECTIONS, *OPTIONAL_SECTIONS):
        values = sqm.get(section)
        if isinstance(values, dict):
            flat.update({f"{section}_{k}": v for k, v in values.items()
                         if isinstance(v, (int, float))})
    return flat


def _bids_name_from_sqm_path(path: Path) -> str:
    for desc in SQM_DESCS:
        if (name := path.name.removesuffix(RECORD_SUFFIXES[desc])) != path.name:
            return name
    return path.stem


def _sqm_row(bids_name: str, sqm: dict) -> dict:
    """One group-table row, in the sectioned record's column vocabulary.

    A legacy record is flat: every scalar sits at the top level and describes every
    channel. That is exactly the sectioned record's ``raw`` view, so it takes the same
    ``raw_`` prefix and a cohort holding both shapes compares in one set of columns.

    The ``_long`` sections an all-long montage skipped are filled in first, or a cohort
    compared on a ``_long`` column would drop those runs without saying so.
    """
    if any(isinstance(sqm.get(section), dict) for section in SECTIONS):
        row = {"bids_name": bids_name, **fill_skipped_long_sections(sqm)}
        # the record keeps the series in a `windowed` section and the heatmaps below look
        # them up by their bare names. Lifting here means one shape reaches the panels;
        # the section stays as written on disk
        windowed = row.pop("windowed", None)
        if isinstance(windowed, dict):
            row.update(windowed)
        return row
    return {
        "bids_name": bids_name,
        **{(f"raw_{k}" if isinstance(v, (int, float)) else k): v for k, v in sqm.items()},
    }


def _collect_sqm(
    output_dir: Path, entity_glob: str,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Glob SQM JSONs under output_dir matching entity prefix (e.g. 'sub-*' or 'group-*').

    One row per run, never one per file: a run that was processed by both the pipeline and
    `prep-raw` has two records, and the sectioned one wins because it is a superset.

    Returns (df, full_rows, descs):
      - df:        scalar SQM columns (bids_name + numeric scalars), for TSV/heatmap/boxplot
      - full_rows: each row keeps the full SQM dict (incl. windowed list fields)
      - descs:     which record kinds the cohort was built from, in SQM_DESCS order
    """
    by_run: dict[str, tuple[Path, str]] = {}
    for desc in SQM_DESCS:
        for sqm_path in sorted(output_dir.glob(f"{entity_glob}/**/nirs/*{RECORD_SUFFIXES[desc]}")):
            by_run.setdefault(_bids_name_from_sqm_path(sqm_path), (sqm_path, desc))

    full_rows: list[dict] = []
    descs: set[str] = set()
    for bids_name, (sqm_path, desc) in sorted(by_run.items()):
        try:
            sqm = json.loads(sqm_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("skip %s: %s", sqm_path, exc)
            continue
        full_rows.append(_sqm_row(bids_name, sqm))
        descs.add(desc)

    return (rows_to_dataframe(full_rows), full_rows,
            [d for d in SQM_DESCS if d in descs])


def _headline_rows(df: pd.DataFrame) -> list[dict]:
    """One row per headline metric: the cohort's median and its range.

    ::

        raw_sci_mean   median 0.95   0.94 - 0.96

    No verdict colour. What counts as a passing SCI is a per-run setting the group table
    does not carry, so the summary reports the spread and leaves the call to the reader.
    """
    by_metric: dict[str, dict[str, str]] = {}
    for col in df.columns:
        stage, channel_set, metric = _split_column(col)
        if channel_set == "all":
            by_metric.setdefault(metric, {})[stage] = col

    rows: list[dict] = []
    for metric in _HEADLINE_METRICS:
        stages = by_metric.get(metric, {})
        col = next((stages[st] for st in _STAGE_ORDER if st in stages), None)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        rows.append({
            "label": col,
            "median": f"{np.median(finite):.4g}",
            "range": f"{finite.min():.4g} to {finite.max():.4g}",
        })
    return rows


def _summary_meta(df: pd.DataFrame, metric_cols: list[str], descs: list[str]) -> list[tuple]:
    """What the cohort is: its size, its shape, and what it was measured from."""
    channel_sets = sorted({_split_column(c)[1] for c in metric_cols})
    windows = sorted({
        float(v) for v in pd.to_numeric(df.get("qc_window_s"), errors="coerce").dropna()
    }) if "qc_window_s" in df.columns else []

    meta: list[tuple] = [
        ("Runs", len(df)),
        ("Metrics", len(metric_cols)),
        ("Channel sets", ", ".join(channel_sets) or "n/a"),
    ]
    if windows:
        meta.append(("QC window (s)", ", ".join(f"{w:g}" for w in windows)))
    meta.append(("Records", " + ".join(_DESC_LABELS.get(d, d) for d in descs) or "none"))
    return meta


def _render_group(
    output_dir: Path,
    out_desc: str,
    title: str,
    df: pd.DataFrame,
    full_rows: list[dict],
    descs: list[str] | None = None,
    template_name: str = "group_report.html.j2",
) -> Path:
    """Render TSV + HTML for an already-collected group of SQM rows.

    Every panel is built under its own guard, so a metric one cohort cannot plot costs that
    panel and not the report, as in every other report builder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    notes: list[str] = []
    if df.empty:
        note(notes, out_desc, "no quality records found, so the report is empty; run the "
                              "pipeline or `fnirs-qc prep-raw` first")

    tsv_path = output_dir / derivative_path("", "qc", ".tsv", desc=out_desc).name
    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info("group TSV  -> %s (rows=%d, cols=%d)", tsv_path, len(df), len(df.columns))

    metric_cols = [c for c in df.columns if c != "bids_name"]

    # the subject report re-applies the passband before comparing stages; this table is built
    # from records rather than recordings and cannot, so it names the pairs instead
    spanning = _bandpass_span_metrics(list(df.columns))
    if spanning:
        shown = ", ".join(spanning[:6])
        more = f", and {len(spanning) - 6} more" if len(spanning) > 6 else ""
        note(notes, out_desc, section_note(
            "caveat.bandpass_pairs", pre=PRE_BANDPASS_HAEMO_STAGE,
            post="/".join(POST_BANDPASS_HAEMO_STAGES), n=len(spanning), shown=shown + more))

    # under figures/ so the .bidsignore line covers these as it covers every other figure
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figure_paths: dict = {}

    def _save(name: str, desc: str, fig, extra_js: str = "") -> None:
        if fig is None:
            return
        fname = derivative_path("", "nirs", ".html", datatype="figures", desc=desc).name
        h = _save_figure_html(fig, fig_dir / fname, extra_js=extra_js)
        figure_paths[name] = {"src": f"figures/{fname}", "h": h,
                              "w": getattr(fig.layout, "width", None)}

    # one number per run, and the few worst names, which every panel below highlights
    ranked: list[str] = []
    worst: list[str] = []
    outliers: dict = {}
    if metric_cols and not df.empty:
        with guard("Outlier detection", errors, out_desc):
            outliers = detect_outliers(df, metric_cols)
            scores = deviation_scores(df, metric_cols)
            finite = np.where(np.isfinite(scores), scores, -np.inf)
            ranked = [str(df.iloc[i, 0]) for i in np.argsort(-finite)]
            worst = [name for name in ranked if name in outliers][:3]

    box_panels: list[dict] = []
    dropped: list[str] = []
    if not df.empty and metric_cols:
        ordered_cols: list[str] = []
        with guard("Metric ordering", errors, out_desc):
            _, ordered_cols = group_metrics(metric_cols)
        with guard("Deviation strip", errors, out_desc):
            strip, dropped = build_deviation_strip(df, ordered_cols)
            _save("strip", "strip", strip)
        with guard("Boxplots", errors, out_desc):
            for i, (box_title, fig) in enumerate(build_grouped_boxes(df, ordered_cols)):
                fname = derivative_path("", "nirs", ".html", datatype="figures",
                                        desc=f"box{i}").name
                h = _save_figure_html(fig, fig_dir / fname, extra_js=_STRIP_CLICK_JS)
                box_panels.append({
                    "src": f"figures/{fname}", "h": h,
                    "w": int(getattr(fig.layout, "width", None) or 300), "title": box_title,
                })

    if dropped:
        shown = ", ".join(dropped[:8])
        more = f", and {len(dropped) - 8} more" if len(dropped) > 8 else ""
        note(notes, out_desc,
             f"{len(dropped)} metrics hold the same value on every run, so there is no "
             f"deviation to plot and the overview leaves them out ({shown}{more}). They are "
             f"still in the table.")

    _warn_on_mixed_windows(full_rows)
    with guard("Windowed grid", errors, out_desc):
        _save("windows", "windows", build_window_grid(full_rows, highlight=worst))
    with guard("Condition panels", errors, out_desc):
        _save("conditions", "conditions", build_condition_panels(full_rows, highlight=worst))
    with guard("Condition matrix", errors, out_desc):
        _save("conditionmatrix", "conditionmatrix",
              build_condition_matrix(full_rows, order=ranked))

    # the coupling scalars are pinned to SCI_WINDOW_S so they stay comparable across runs,
    # while every windowed panel follows the run's own QC window. Silence means the two agree
    windows = {float(v) for v in (r.get("qc_window_s") for r in full_rows)
               if isinstance(v, (int, float))}
    if windows - {SCI_WINDOW_S}:
        note(notes, out_desc, section_note(
            "caveat.window_mismatch", pinned=SCI_WINDOW_S,
            windows=", ".join(f"{w:g}" for w in sorted(windows))))

    conditions = condition_names(full_rows)
    if not conditions:
        note(notes, out_desc,
             "no run carries conditions, so the per-condition panels are absent; the "
             "windowed panel keeps the time axis")

    html = render(
        template_name,
        **page_vars(
            title=title,
            heading=title,
            nav_meta=[("runs", len(df)), ("metrics", len(metric_cols))],
        ),
        **footer_vars(scope=out_desc, errors=errors, notes=notes,
                     versions=collect_software_versions()),
        title=title,
        n_rows=len(df),
        n_metrics=len(metric_cols),
        summary_meta=_summary_meta(df, metric_cols, descs or []),
        headline_rows=_headline_rows(df) if not df.empty else [],
        small_cohort=0 < len(df) < SMALL_COHORT_N,
        score_threshold=SCORE_THRESHOLD,
        smooth_s=SMOOTH_S,
        sci_window_s=SCI_WINDOW_S,
        conditions=conditions,
        worst_runs=worst,
        figure_paths=figure_paths,
        box_panels=box_panels,
        tsv_name=tsv_path.name,
        table_columns=list(df.columns),
        table_rows=df.values.tolist(),
        outliers=outliers,
    )
    html_path = output_dir / derivative_path("", "report", ".html", desc=out_desc).name
    html_path.write_text(html, encoding="utf-8")
    logger.info("group HTML -> %s", html_path)
    return html_path


def _build_group(
    output_dir: Path,
    entity_glob: str,
    out_desc: str,
    title: str,
    template_name: str = "group_report.html.j2",
) -> Path:
    """Glob SQM JSONs and render group report."""
    df, full_rows, descs = _collect_sqm(output_dir, entity_glob)
    return _render_group(output_dir, out_desc, title, df, full_rows, descs, template_name)


def rows_to_dataframe(full_rows: list[dict]) -> pd.DataFrame:
    """Scalar-only table from full SQM records, one row per run, columns sorted."""
    if not full_rows:
        return pd.DataFrame(columns=["bids_name"])
    scalar_rows = [{"bids_name": r["bids_name"], **_scalars(r)} for r in full_rows]
    cols = ["bids_name"] + sorted({k for r in scalar_rows for k in r if k != "bids_name"})
    return pd.DataFrame(scalar_rows, columns=cols)


def build_group_raw_report(output_dir: Path) -> Path:
    """Aggregate all sub-XX/nirs/...desc-sqm_qc.json into desc-subjects_{qc.tsv,report.html}."""
    return _build_group(
        output_dir, entity_glob="sub-*",
        out_desc="subjects",
        title="Cohort QC (individual subjects)",
    )

