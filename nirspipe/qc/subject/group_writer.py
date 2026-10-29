"""Cohort-level QC aggregation: glob per-subject (or per-group) SQM JSONs into
TSV + an HTML viewer with heatmap / boxplots / sortable table / outlier panel."""

from __future__ import annotations

import json
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from nirspipe.qc.boilerplate import collect_software_versions
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.boilerplate.vocabulary import (
    COHORT_ID_COLUMN, COHORT_SECTIONS, CONDITION_COLUMNS, CONDITION_METRICS, CONDITION_WINDOW,
    format_metric, metric_label, record_key_summary,
)
from nirspipe.qc.metrics.coupling import SCI_WINDOW_S, labels_look_swapped
from nirspipe.qc.metrics.imu import IMU_QUANTITIES
from nirspipe.qc.common.figure_io import _save_figure_html
from nirspipe.qc.figures.subject.group_figures import (
    SCORE_THRESHOLD,
    SMOOTH_S,
    _split_column,
    build_condition_matrix,
    build_condition_panels,
    build_deviation_strip,
    build_fc_distributions,
    build_fc_movement,
    build_grouped_boxes,
    build_window_grid,
    check_only,
    condition_names,
    detect_outliers,
    deviation_scores,
    group_metrics,
)
from nirspipe.qc.common.report_shell import (
    footer_vars,
    guard,
    note,
    page_vars,
    render,
)
from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import entity_of
from nirspipe.io.naming import derivative_path
from nirspipe.qc.common.channel_table import (
    CHANNEL_METRICS_SUFFIXES, long_kept_pairs, optode_rejections, pair_status, region_coverage,
)
from nirspipe.qc.subject.record_io import channel_columns, channel_frame, read_record
from nirspipe.qc.subject.sqm_record import (
    OPTIONAL_SECTIONS,
    POST_BANDPASS_HAEMO_STAGES,
    PRE_BANDPASS_HAEMO_STAGE,
    SECTIONS,
    RECORD_SUFFIXES,
    SQM_DESCS,
    fill_skipped_long_sections,
    raw_verdict_view,
)
from nirspipe.utils import fisher_r_to_z, load_toml, pair_of, pair_order
from nirspipe.utils.logging import get_logger
from nirspipe.io.tables import read_table, write_tsv

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
# the outlier rule each need a middle to sit in. The panels still render, carrying a line that
# says what they cannot tell the reader at this cohort size.
SMALL_COHORT_N = 5

# What the summary reports, in this order. Each is read off the whole-channel-set column of
# the earliest stage the cohort carries, that being the one every record has.
_HEADLINE_METRICS = ("sci_win_mean", "psp_mean", "snr_mean", "cv_mean",
                     "channel_retention_rate", "gvtd_pct_above_thresh")
_STAGE_ORDER = ("raw", "motion", "motion_post", "preproc", "filtered", "resampled", "errts")

# How a record's desc reads in the summary.
_DESC_LABELS = {"sqm": "pipeline", "sqmraw": "raw QC"}



def _warn_on_mixed_windows(rows: list) -> None:
    """A time x subject heatmap only reads straight if every subject was binned the same way."""
    lengths = {r.get("qc_window_s") for r in rows}
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


def _column_dictionary(columns) -> dict:
    """The table's sidecar: each ``section_metric`` column described from the vocabulary."""
    sections = sorted((*SECTIONS, *OPTIONAL_SECTIONS), key=len, reverse=True)
    out: dict = {}
    for column in columns:
        if column == "bids_name":
            out[column] = {"Description": COHORT_ID_COLUMN}
            continue
        section = next((s for s in sections if column.startswith(f"{s}_")), None)
        key = column[len(section) + 1:] if section else column
        text = record_key_summary(key)
        if not text:
            continue
        entry = {}
        if label := metric_label(key, fallback=""):
            entry["LongName"] = f"{label}, {section}" if section else label
        entry["Description"] = " ".join(filter(None, (text, COHORT_SECTIONS.get(section))))
        out[column] = entry
    return out


# a condition entry's channel sets, as the cohort table's section suffixes
_CONDITION_SETS = {"all": "", "long": "_long", "short": "_short"}
_CONDITION_FAMILIES = ("od_by_set", "motion_by_set", "haemo_by_set")
# the stage a haemoglobin key's suffix names, where the run's own value of it sits
_HAEMO_STAGE_SUFFIXES = (("_prereg", "filtered"), ("_postreg", "errts"), ("_errts", "errts"))


def _condition_column(family: str, key: str, row: dict) -> "tuple[str, str]":
    """The section and metric a condition's value is named by: those of its run's value.

    ::

      od_by_set sci_win_mean                    -> raw, sci_win_mean
      motion_by_set gvtd_filt_mean              -> motion_post, or raw without a correction
      haemo_by_set gcor_hbo_postreg (pipeline)  -> errts, gcor_hbo
      haemo_by_set hbo_hbr_corr_mean_post (raw) -> rawhaemo_post, hbo_hbr_corr_mean
    """
    if family == "od_by_set":
        return "raw", key
    if family == "motion_by_set":
        if key.startswith("motion_corrected"):
            return "motion", key
        # spikes are found on the recording before correction, GVTD on the corrected copy
        if key.startswith("spike"):
            return "raw", key
        return ("motion_post" if "motion_post" in row else "raw"), key
    base = "preproc" if "preproc" in row else "rawhaemo"
    for suffix, stage in (*_HAEMO_STAGE_SUFFIXES, ("_post", f"{base}_post")):
        if key.endswith(suffix):
            return stage, key[: -len(suffix)]
    return base, key


def condition_rows(row: dict) -> list[dict]:
    """One row per condition of a run, its values under the cohort table's column names."""
    out = []
    for condition, entry in (row.get("by_condition") or {}).items():
        start, stop = entry.get("window_s") or (None, None)
        flat = {"bids_name": row["bids_name"], "condition": condition, "onset": start,
                "duration": None if start is None else stop - start,
                "kept_s": entry.get("kept_s"), "counted_windows": entry.get("counted_windows")}
        for family in _CONDITION_FAMILIES:
            for set_name, values in (entry.get(family) or {}).items():
                for key, value in (values or {}).items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        section, metric = _condition_column(family, key, row)
                        flat[f"{section}{_CONDITION_SETS[set_name]}_{metric}"] = value
        # the rest of the flat scalars repeat the families above, bar these
        scalars = entry.get("scalars") or {}
        flat.update({f"imu_{k}": v for k, v in scalars.items()
                     if k.startswith(tuple(IMU_QUANTITIES)) and isinstance(v, (int, float))})
        flat.update({f"raw_{k}": scalars[k] for k in ("n_long_channels", "n_short_channels")
                     if isinstance(scalars.get(k), (int, float))})
        out.append(flat)
    return out


def _write_conditions_table(output_dir: Path, full_rows: list[dict]) -> "Path | None":
    """``desc-conditions_qc.tsv``, one row per run and condition; None when no run has one."""
    rows = [r for row in full_rows for r in condition_rows(row)]
    if not rows:
        return None
    table = pd.DataFrame(rows)
    front = list(CONDITION_COLUMNS)
    table = table[front + sorted(c for c in table.columns if c not in front)]
    path = output_dir / derivative_path("", "qc", ".tsv", desc="conditions").name
    write_tsv(table, path)
    sidecar = {c: {"Description": CONDITION_COLUMNS[c]} for c in front}
    for column, entry in _column_dictionary(table.columns[len(front):]).items():
        metric = next((m for m in CONDITION_METRICS if column.endswith(f"_{m}")), None)
        text = CONDITION_METRICS[metric] if metric else entry["Description"]
        sidecar[column] = {**entry, "Description": f"{text} {CONDITION_WINDOW}"}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    logger.info("conditions TSV -> %s (rows=%d)", path, len(table))
    return path


def _bids_name_from_sqm_path(path: Path) -> str:
    for desc in SQM_DESCS:
        if (name := path.name.removesuffix(RECORD_SUFFIXES[desc])) != path.name:
            return name
    return path.stem


def _sqm_row(bids_name: str, sqm: dict) -> dict:
    """One group-table row, in the sectioned record's column vocabulary.

    The ``_long`` sections an all-long montage skipped are filled in first, or a cohort
    compared on a ``_long`` column would drop those runs without saying so.
    """
    row = {"bids_name": bids_name, **fill_skipped_long_sections(sqm)}
    # the record keeps the series in a `windowed` section and the heatmaps below look
    # them up by their bare names. Lifting here means one shape reaches the panels;
    # the section stays as written on disk
    windowed = row.pop("windowed", None)
    if isinstance(windowed, dict):
        row.update(windowed)
    return row


def _collect_sqm(
    output_dir: Path, entity_glob: str,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Glob SQM JSONs under output_dir matching entity prefix (e.g. 'sub-*' or 'group-*').

    One row per run, never one per file: a run that was processed by both the pipeline and
    raw QC has two records, and the sectioned one wins because it is a superset.

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
            sqm = read_record(sqm_path)
        except StageError:
            raise
        except Exception as exc:
            logger.warning("skip %s: %s", sqm_path, exc)
            continue
        full_rows.append(_sqm_row(bids_name, sqm))
        descs.add(desc)

    return (rows_to_dataframe(full_rows), full_rows,
            [d for d in SQM_DESCS if d in descs])


def _write_channel_table(output_dir: Path, out_desc: str, full_rows: list[dict]) -> "Path | None":
    """Every run's channel table stacked under a ``bids_name`` column; None when no run has one."""
    frames = [channel_frame(row["per_channel"]).assign(bids_name=row["bids_name"])
              for row in full_rows if isinstance(row.get("per_channel"), dict)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    table = table[["bids_name", *(c for c in table.columns if c != "bids_name")]]
    path = output_dir / derivative_path("", "nirsmap", ".tsv", desc=out_desc).name
    write_tsv(table, path)
    sidecar = {"bids_name": {"Description": COHORT_ID_COLUMN}, **channel_columns(table)}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    logger.info("channel TSV -> %s (rows=%d)", path, len(table))
    return path


def _headline_rows(df: pd.DataFrame) -> list[dict]:
    """One row per headline metric: the cohort's median and its range.

    ::

        Mean SCI (10 s)   0.950   0.940 to 0.960

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
            "label": metric_label(metric),
            "median": format_metric(metric, float(np.median(finite))),
            "range": f"{format_metric(metric, finite.min())} to "
                     f"{format_metric(metric, finite.max())}",
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


def unfinished_subjects(output_dir: Path) -> list[dict]:
    """Runs whose last invocation failed or never wrote its outcome, one per run record, raw QC's too.

    ::

        sub-01/logs/sub-01.toml        [outcome] status = "failed"   ->  listed, with its error
        sub-02/logs/sub-02.toml        [outcome] status = "success"  ->  left out
        sub-03/logs/sub-03_ses-2.toml  no [outcome] (killed)         ->  listed as sub-03_ses-2
    """
    rows: list[dict] = []
    for path in sorted(output_dir.glob("sub-*/logs/sub-*.toml")):
        try:
            record = load_toml(path)
        except Exception:
            logger.warning("unreadable run record %s", path, exc_info=True)
            continue
        execution = record.get("execution", {})
        # a dry run stops before processing, so it has nothing to finish
        if execution.get("dry_run"):
            continue
        outcome = record.get("outcome", {})
        status = outcome.get("status", "unfinished")
        if status == "success":
            continue
        rows.append({
            "subject": path.stem,
            "status": status,
            "error": outcome.get("error"),
            "started": execution.get("run_timestamp"),
        })
    return rows


def _channel_tables(output_dir: Path) -> "dict[str, Path]":
    """Each run's channel table by run label, the pipeline's where raw QC wrote one too."""
    tables: dict[str, Path] = {}
    for suffix in CHANNEL_METRICS_SUFFIXES:
        for path in sorted(output_dir.glob(f"sub-*/**/nirs/*{suffix}")):
            tables.setdefault(path.name[:-len(suffix)], path)
    return tables


def _task_of(label: str) -> str:
    """The cohort column a run falls in: its task, every acquisition of it together."""
    return f"task-{entity_of(label, 'task')}"


def channels_kept(output_dir: Path) -> dict:
    """How many subjects kept each source-detector pair, per task: ``{"tasks", "rows"}``, empty without a channel table.

    ::

      sub-01 keeps S1_D1 in run 1 only, sub-02 rejects it  ->  S1_D1  task-rest  1/2

    A subject keeps a pair when any of its runs of the task did, sessions included; the
    count is over the subjects whose runs of the task list the pair.
    """
    listed: dict[str, dict[str, set]] = {}
    kept: dict[str, dict[str, set]] = {}
    separation: dict[str, str] = {}
    for label, path in sorted(_channel_tables(output_dir).items()):
        task = _task_of(label)
        subject = entity_of(label, "sub")
        for pair, status in pair_status(path).items():
            separation.setdefault(pair, status["separation"])
            listed.setdefault(task, {}).setdefault(pair, set()).add(subject)
            if not status["is_bad"]:
                kept.setdefault(task, {}).setdefault(pair, set()).add(subject)
    if not listed:
        return {}
    tasks = sorted(listed)
    rows = [{"pair": pair, "separation": sep,
             "cells": [f"{len(kept.get(task, {}).get(pair, ()))}/{len(listed[task][pair])}"
                       if pair in listed[task] else None for task in tasks]}
            for pair, sep in separation.items()]
    return {"tasks": tasks, "rows": rows}


def swapped_label_tasks(full_rows: list[dict]) -> list[dict]:
    """Each task whose runs mostly read as if the two wavelength labels were swapped.

    ``[{"task": "task-rest", "runs": [names], "n": runs with a verdict}]``, a task listed when
    more than half of its runs with a verdict read swapped (:func:`labels_look_swapped`). A
    label error belongs to a device or a converter, so it shows in most runs; one run alone
    can read swapped by chance.
    """
    by_task: dict[str, dict] = {}
    for row in full_rows:
        long_set = row.get("raw_long") or {}
        swapped = labels_look_swapped(long_set.get("cardiac_ratio"),
                                      long_set.get("cardiac_ratio_swapped"))
        if swapped is None:
            continue
        task = by_task.setdefault(_task_of(row["bids_name"]),
                                  {"runs": [], "n": 0})
        task["n"] += 1
        if swapped:
            task["runs"].append(row["bids_name"])
    return [{"task": name, "runs": t["runs"], "n": t["n"]} for name, t in sorted(by_task.items())
            if len(t["runs"]) > t["n"] / 2]


def _optode_order(name: str) -> tuple:
    """Sources before detectors, then by number: S2 before S10."""
    digits = name[1:]
    return (name[:1].upper() != "S", int(digits) if digits.isdigit() else 0, name)


def optode_losses(output_dir: Path) -> dict:
    """Each optode's long channels rejected over the runs of each task, pooled: ``{"tasks", "rows"}``, empty without a channel table.

    ::

      run 1 rejects S1_D1 and S1_D2, run 2 rejects neither  ->  S1  task-rest  50% (2/4)

    Short channels are left out, as on each run's optode map.
    """
    counts: dict[str, dict[str, list[int]]] = {}
    for label, path in sorted(_channel_tables(output_dir).items()):
        for optode, (bad, listed) in optode_rejections(pair_status(path)).items():
            total = counts.setdefault(_task_of(label), {}).setdefault(optode, [0, 0])
            total[0] += bad
            total[1] += listed
    if not counts:
        return {}
    tasks = sorted(counts)
    optodes = sorted({o for by_optode in counts.values() for o in by_optode}, key=_optode_order)

    def cell(task: str, optode: str) -> "str | None":
        if optode not in counts[task]:
            return None
        bad, listed = counts[task][optode]
        return f"{bad / listed:.0%} ({bad}/{listed})"

    return {"tasks": tasks,
            "rows": [{"optode": o, "cells": [cell(t, o) for t in tasks]} for o in optodes]}


def region_losses(output_dir: Path, roi_map: "dict[str, list[str]] | None") -> dict:
    """The runs of each task in which an ``--roi-mapping`` region kept no long channel: ``{"tasks", "rows"}``, empty without a map or a channel table.

    ::

      run 1 keeps 0 of L's 2 long channels, run 2 keeps 1  ->  L  task-rest  1/2

    A run whose recording has none of the region's long channels counts as keeping none,
    since the region drops out of its region outputs either way. Each cell carries the
    runs it counts under ``runs``.
    """
    if not roi_map:
        return {}
    lost: dict[str, dict[str, list[str]]] = {}
    runs: dict[str, int] = {}
    for label, path in sorted(_channel_tables(output_dir).items()):
        task = _task_of(label)
        runs[task] = runs.get(task, 0) + 1
        by_region = lost.setdefault(task, {})
        for row in region_coverage(pair_status(path), roi_map):
            by_region.setdefault(row["region"], [])
            if row["kept"] == 0:
                by_region[row["region"]].append(label)
    if not runs:
        return {}
    tasks = sorted(runs)
    regions = list(next(iter(lost.values())))
    return {"tasks": tasks,
            "rows": [{"region": region,
                      "cells": [{"text": f"{len(lost[t][region])}/{runs[t]}",
                                 "runs": lost[t][region]} for t in tasks]}
                     for region in regions]}


def _relmat(path: Path) -> pd.DataFrame:
    return read_table(path).set_index("channel")


def fc_distributions(output_dir: Path) -> dict:
    """Each run's channel-pair r over its kept long channels, before and after the regression, per task.

    ``{task: {"runs": [label, ...], "values": {(chromophore, "before"|"after"): [r per run]}}}``,
    read from the residual's channel matrix and the one written beside it on the series the
    regression started from (``_desc-<stage>_relmat.tsv``); a run without the second ran no
    regression and is left out. Long and kept are read off the run's channel table.
    """
    tables = _channel_tables(output_dir)
    out: dict = {}
    for after_path in sorted(output_dir.glob("sub-*/**/nirs/*_stat-pearson_relmat.tsv")):
        name = after_path.name
        if entity_of(name, "seg") or entity_of(name, "agg"):
            continue
        label, chroma = name[:name.index("_chromo-")], entity_of(name, "chromo")
        before_path = next(iter(sorted(after_path.parent.glob(
            f"{label}_chromo-{chroma}_stat-pearson_desc-*_relmat.tsv"))), None)
        if before_path is None or label not in tables:
            continue
        kept = long_kept_pairs(pair_status(tables[label]))
        after, before = _relmat(after_path), _relmat(before_path)
        names = [c for c in after.index if pair_of(c) in kept and c in before.index]
        if len(names) < 2:
            continue
        upper = np.triu_indices(len(names), 1)
        task = out.setdefault(_task_of(label), {"runs": [], "values": {}})
        if label not in task["runs"]:
            task["runs"].append(label)
        for side, matrix in (("before", before), ("after", after)):
            r = matrix.loc[names, names].to_numpy(dtype=float)[upper]
            task["values"].setdefault((chroma, side), []).append(r[np.isfinite(r)])
    return out


# a correlation across fewer runs than this says nothing of a connection; ours
FC_MOVEMENT_MIN_RUNS = 5

FC_MOVEMENT_COLUMNS = {
    "task": "The task whose runs were pooled, acquisitions together.",
    "chromophore": "The chromophore of the connection.",
    "channel_a": "One end of the connection, a long source-detector pair.",
    "channel_b": "The other end.",
    "n_runs": "Runs of the task that kept both channels, the runs the r is taken over.",
    "r": "Pearson r across those runs between each run's head movement (band-passed GVTD "
         "averaged over its long channels, gvtd_filt_mean) and the Fisher z of the "
         "connection's FC. Descriptive, no test.",
    "distance_mm": "Distance between the two channels in the runs' template space, the median "
                   "over the runs; blank when no run placed them.",
}


def _positions(nirs_dir: Path, label: str) -> "dict[str, np.ndarray]":
    """Each pair's position in the run's template space; empty when the run has none."""
    path = next(iter(sorted(nirs_dir.glob(f"{label}_space-*_channels.tsv"))), None)
    if path is None:
        return {}
    table = read_table(path)
    return {pair_of(name): np.array([x, y, z], float)
            for name, x, y, z in zip(table["name"], table["x"], table["y"], table["z"])}


def fc_against_movement(output_dir: Path, full_rows: list[dict]) -> pd.DataFrame:
    """Each connection's r, across the runs of a task, between a run's head movement and its FC.

    One row per task, chromophore and pair of kept long channels that at least
    :data:`FC_MOVEMENT_MIN_RUNS` runs share; the columns are :data:`FC_MOVEMENT_COLUMNS`.
    Movement is each record's ``gvtd_filt_mean`` on its long set, FC the run's channel matrix
    after the regression. Runs pool by task, connections match by channel name.

    ::

      3 runs keep S1_D1 and S2_D2, movement 1, 2, 3, FC z 0.1, 0.2, 0.3  ->  r 1.0
    """
    movement = {row["bids_name"]: value for row in full_rows
                if (value := raw_verdict_view(row).get("gvtd_filt_mean")) is not None}
    tables = _channel_tables(output_dir)
    pooled: dict = {}
    for path in sorted(output_dir.glob("sub-*/**/nirs/*_stat-pearson_relmat.tsv")):
        name = path.name
        if entity_of(name, "seg") or entity_of(name, "agg"):
            continue
        label, chroma = name[:name.index("_chromo-")], entity_of(name, "chromo")
        if label not in movement or label not in tables:
            continue
        kept = long_kept_pairs(pair_status(tables[label]))
        fc = _relmat(path)
        names = [c for c in fc.index if pair_of(c) in kept]
        values = fisher_r_to_z(fc.loc[names, names].to_numpy(dtype=float))
        positions = _positions(path.parent, label)
        task = _task_of(label)
        for i, j in zip(*np.triu_indices(len(names), 1)):
            if not np.isfinite(values[i, j]):
                continue
            a, b = sorted((pair_of(names[i]), pair_of(names[j])), key=pair_order)
            distance = (float(np.linalg.norm(positions[a] - positions[b]))
                        if a in positions and b in positions else np.nan)
            pooled.setdefault((task, chroma, a, b), []).append(
                (movement[label], values[i, j], distance))
    rows = []
    for (task, chroma, a, b), runs in pooled.items():
        moved, z, distance = (np.array(v, float) for v in zip(*runs))
        if len(runs) < FC_MOVEMENT_MIN_RUNS or np.ptp(moved) == 0 or np.ptp(z) == 0:
            continue
        rows.append({"task": task, "chromophore": chroma, "channel_a": a, "channel_b": b,
                     "n_runs": len(runs), "r": float(np.corrcoef(moved, z)[0, 1]),
                     "distance_mm": (float(np.nanmedian(distance))
                                     if np.isfinite(distance).any() else np.nan)})
    return pd.DataFrame(rows, columns=list(FC_MOVEMENT_COLUMNS))


def _write_fc_movement(output_dir: Path, table: pd.DataFrame) -> "Path | None":
    if table.empty:
        return None
    path = output_dir / derivative_path("", "qc", ".tsv", desc="fcmovement").name
    write_tsv(table, path)
    path.with_suffix(".json").write_text(json.dumps(
        {c: {"Description": d} for c, d in FC_MOVEMENT_COLUMNS.items()}, indent=2) + "\n",
        encoding="utf-8")
    logger.info("FC against movement -> %s (rows=%d)", path, len(table))
    return path


def _render_group(
    output_dir: Path,
    out_desc: str,
    title: str,
    df: pd.DataFrame,
    full_rows: list[dict],
    descs: list[str] | None = None,
    template_name: str = "group_report.html.j2",
    unfinished: list[dict] | None = None,
    roi_map: "dict[str, list[str]] | None" = None,
    dyad_page: "str | None" = None,
) -> Path:
    """Render TSV + HTML for an already-collected group of SQM rows.

    Every panel is built under its own guard, so a metric one cohort cannot plot costs that
    panel and not the report, as in every other report builder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    notes: list[str] = []

    tsv_path = output_dir / derivative_path("", "qc", ".tsv", desc=out_desc).name
    write_tsv(df, tsv_path)
    tsv_path.with_suffix(".json").write_text(
        json.dumps(_column_dictionary(df.columns), indent=2) + "\n", encoding="utf-8")
    logger.info("group TSV  -> %s (rows=%d, cols=%d)", tsv_path, len(df), len(df.columns))
    channel_path = _write_channel_table(output_dir, out_desc, full_rows)
    with guard("conditions table", errors, out_desc):
        _write_conditions_table(output_dir, full_rows)

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

    # the columns the charts draw, settings and check-only metrics left out; the outlier
    # score is taken over the same ones, so a run is never called out on a number not shown
    ordered_cols: list[str] = []
    if metric_cols:
        with guard("Metric ordering", errors, out_desc):
            _, ordered_cols = group_metrics(metric_cols)

    # one number per run, and the few worst names, which every panel below highlights
    ranked: list[str] = []
    worst: list[str] = []
    outliers: dict = {}
    if ordered_cols and not df.empty:
        with guard("Outlier detection", errors, out_desc):
            outliers = detect_outliers(df, ordered_cols)
            scores = deviation_scores(df, ordered_cols)
            finite = np.where(np.isfinite(scores), scores, -np.inf)
            ranked = [str(df.iloc[i, 0]) for i in np.argsort(-finite)]
            worst = [name for name in ranked if name in outliers][:3]

    box_panels: list[dict] = []
    dropped: list[str] = []
    if not df.empty and metric_cols:
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

    kept: dict = {}
    with guard("Channels kept", errors, out_desc):
        kept = channels_kept(output_dir)
    optodes: dict = {}
    with guard("Optodes", errors, out_desc):
        optodes = optode_losses(output_dir)
    regions: dict = {}
    with guard("Regions", errors, out_desc):
        regions = region_losses(output_dir, roi_map)
    with guard("Connectivity distributions", errors, out_desc):
        _save("fcdistribution", "fcdistribution", build_fc_distributions(fc_distributions(output_dir)))
    fc_movement = pd.DataFrame()
    with guard("Connectivity against movement", errors, out_desc):
        fc_movement = fc_against_movement(output_dir, full_rows)
        _write_fc_movement(output_dir, fc_movement)
        _save("fcmovement", "fcmovement", build_fc_movement(fc_movement))

    for task in swapped_label_tasks(full_rows):
        msg = section_note("caveat.labels_swapped", task=task["task"], k=len(task["runs"]),
                           n=task["n"], runs=", ".join(task["runs"]))
        notes.append(msg)
        logger.warning(msg)

    conditions = condition_names(full_rows)
    if not conditions:
        note(notes, out_desc,
             "no run carries conditions, so the per-condition panels are absent; the "
             "windowed panel keeps the time axis")

    # the TSV keeps every column; the page leaves out what is only there to be checked
    shown_cols = [c for c in df.columns if not check_only(c)]
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
        summary_meta=_summary_meta(df, metric_cols, descs or [])
                     + ([("Subjects not finished", len(unfinished))] if unfinished else []),
        unfinished=unfinished or [],
        headline_rows=_headline_rows(df) if not df.empty else [],
        small_cohort=0 < len(df) < SMALL_COHORT_N,
        score_threshold=SCORE_THRESHOLD,
        fc_movement_min_runs=FC_MOVEMENT_MIN_RUNS,
        smooth_s=SMOOTH_S,
        sci_window_s=SCI_WINDOW_S,
        conditions=conditions,
        worst_runs=worst,
        figure_paths=figure_paths,
        box_panels=box_panels,
        tsv_name=tsv_path.name,
        channel_tsv_name=channel_path.name if channel_path else None,
        table_columns=shown_cols,
        table_rows=df[shown_cols].values.tolist(),
        outliers=outliers,
        kept=kept,
        optodes=optodes,
        regions=regions,
        dyad_page=dyad_page,
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
    unfinished: list[dict] | None = None,
    roi_map: "dict[str, list[str]] | None" = None,
    dyad_page: "str | None" = None,
) -> Path | None:
    """Glob SQM JSONs and render group report; None when there are none."""
    df, full_rows, descs = _collect_sqm(output_dir, entity_glob)
    if df.empty:
        return None
    return _render_group(output_dir, out_desc, title, df, full_rows, descs, template_name,
                         unfinished, roi_map, dyad_page)


def rows_to_dataframe(full_rows: list[dict]) -> pd.DataFrame:
    """Scalar-only table from full SQM records, one row per run, columns sorted."""
    if not full_rows:
        return pd.DataFrame(columns=["bids_name"])
    scalar_rows = [{"bids_name": r["bids_name"], **_scalars(r)} for r in full_rows]
    cols = ["bids_name"] + sorted({k for r in scalar_rows for k in r if k != "bids_name"})
    return pd.DataFrame(scalar_rows, columns=cols)


def dyad_page(output_dir: Path) -> "str | None":
    """The dyad cohort page's name when the tree holds dyad records, which build that page next."""
    suffix = RECORD_SUFFIXES["sqm"]
    held = next(chain(output_dir.glob(f"group-*/nirs/*{suffix}"),
                      output_dir.glob(f"group-*/ses-*/nirs/*{suffix}")), None)
    return derivative_path("", "report", ".html", desc="groups").name if held else None


def build_group_raw_report(output_dir: Path,
                           roi_map: "dict[str, list[str]] | None" = None) -> Path | None:
    """Aggregate all sub-XX/nirs/...desc-sqm_qc.json into desc-subjects_{qc.tsv,report.html}."""
    return _build_group(
        output_dir, entity_glob="sub-*",
        out_desc="subjects",
        title="Cohort QC (individual subjects)",
        unfinished=unfinished_subjects(output_dir),
        roi_map=roi_map,
        dyad_page=dyad_page(output_dir),
    )

