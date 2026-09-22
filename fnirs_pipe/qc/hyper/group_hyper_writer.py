"""Cohort-level hyperscanning QC: every dyad in a tree on one page.

Assembled from what the dyads already wrote, never from anything held in memory, so the page
can be rebuilt for a tree produced weeks ago and a dyad the run skipped is simply absent
rather than stale. Each dyad contributes its quality record
(``group-*/nirs/<label>_desc-sqm_nirs.json``) and, where the run wrote one, its usable-time
table (``<label>_hyper-usable.tsv``).

What it reports is what a dyad has and a subject cannot:

- the **shared** usable time, a channel pair being usable only while it is coupled in both
  members at once, split into the asymmetric loss and the shared one
- where that time went, per channel pair and per condition
- each window's coherence against its own surrogate null, on the rank scale, since a raw
  coherence is not comparable between windows of different length

Deliberately not here: SCI, PSP, CV, retention and motion distributions across the cohort.
Those are ``fnirs-qc cohort``'s, measured per subject, and a second copy of them here
would be the same numbers under a heading that implies they are about the dyad.

Nor group statistics. The distribution of a percentile across dyads is here to find the dyad
that does not look like the others, not to decide whether synchrony is significant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.common.figure_io import _save_figure_html
from fnirs_pipe.qc.figures.hyper.group_hyper_figures import (
    build_condition_dials, build_null_strip, build_pair_field, build_usable_bars,
    cohort_order,
)
from fnirs_pipe.qc.metrics.hyper import NULL_ALPHA_PCT
from fnirs_pipe.qc.subject.sqm_record import RECORD_SUFFIX, record_label
from fnirs_pipe.qc.common.report_shell import footer_vars, guard, note, page_vars, render
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.group_hyper_writer")

# Below this many dyads a cohort has no middle to measure a dyad against: the median and the
# outlier rule both need one. The panels still draw, carrying the line that says so.
SMALL_COHORT_N = 5

# What the summary reports, in this order, and how each prints.
_HEADLINE = (
    ("usable_window_frac", "Shared usable time", "{:.0%}"),
    ("one_member_frac", "Lost to one member", "{:.0%}"),
    ("neither_frac", "Lost to both", "{:.0%}"),
    ("usable_stretch_s", "Median usable stretch", "{:.0f} s"),
    ("percentile_median", "Coherence percentile", "{:.0f}"),
)

_COLUMNS = (
    ("label", "Dyad", "{}"),
    ("usable_window_frac", "Usable", "{:.0%}"),
    ("one_member_frac", "One member", "{:.0%}"),
    ("neither_frac", "Neither", "{:.0%}"),
    ("usable_stretch_s", "Median stretch", "{:.0f} s"),
    ("usable_pairs_mean", "Usable pairs", "{:.1f}"),
    ("n_long_pairs", "Long pairs", "{:.0f}"),
    ("aligned_duration_s", "Aligned", "{:.0f} s"),
    ("max_offset_s", "Max offset", "{:.2f} s"),
    ("offsets_equal", "Offsets equal", "{}"),
    ("pct_all_good", "Good in both", "{:.0f}%"),
    ("percentile_median", "Median pct", "{:.0f}"),
    ("n_above", "Above null", "{:.0f}"),
)

_USABLE_KEYS = ("n_long_pairs", "n_windows", "usable_pairs_mean", "usable_window_frac",
                "one_member_frac", "neither_frac", "usable_stretch_s")



def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("unreadable record, its dyad goes unlisted: %s (%s)", path.name, exc)
        return {}


def _read_tsv(path: Path) -> "pd.DataFrame | None":
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="\t")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("unreadable table, its panel loses a row: %s (%s)", path.name, exc)
        return None


def _shares(table: "pd.DataFrame | None", key: str) -> dict:
    """``{pair: share}`` or ``{condition: share}``, averaged over the other column."""
    if table is None or table.empty or key not in table.columns:
        return {}
    return {str(k): float(v) for k, v in table.groupby(key)["usable_frac"].mean().items()}


def _offsets_equal(record: dict) -> "bool | None":
    """Whether both members were cropped by the same amount, off the alignment block.

    Two members with the same offset are the hard case rather than the easy one: the columns
    are wrong together, so nothing looks asymmetric if a member's own clock has been mixed up
    with the dyad's. It is reported because it says which of the two a reader is looking at,
    not because unequal offsets are a fault.
    """
    offsets = (record.get("alignment") or {}).get("align_offset_s")
    if not isinstance(offsets, dict) or not offsets:
        return None
    values = [v for v in offsets.values() if v is not None]
    return len({round(float(v), 3) for v in values}) == 1 if values else None


def collect_rows(output_dir: Path) -> list[dict]:
    """One row per dyad-task under ``output_dir``, for the panels and the table.

    The label a row carries is the BIDS stem of its record, so a dyad recorded on two tasks
    is two rows and neither is silently the other.
    """
    rows: list[dict] = []
    for record_path in sorted(output_dir.glob(f"group-*/nirs/*{RECORD_SUFFIX}")):
        record = _read_json(record_path)
        if not record or record.get("step") != "hyper_sqm":
            continue
        label = record_label(record_path)
        group_dir = record_path.parent.parent
        table = _read_tsv(record_path.parent / f"{label}_hyper-usable.tsv")
        screening = record.get("screening") or {}
        percentile = {name: float(v["percentile"])
                      for name, v in (screening.get("windows") or {}).items()
                      if v.get("percentile") is not None}

        report = group_dir / f"{label}_desc-hyperraw_nirs.html"
        index = group_dir / f"{group_dir.name}_index.html"
        href = next((f"{group_dir.name}/{p.name}" for p in (report, index) if p.exists()),
                    None)

        rows.append({
            "label": label,
            "href": href,
            "usable": {k: record[k] for k in _USABLE_KEYS if k in record},
            "by_pair": _shares(table, "pair"),
            "by_cond": _shares(table, "condition"),
            "percentile": percentile,
            "alpha": screening.get("alpha", NULL_ALPHA_PCT),
            "record": record,
        })
    return rows


def _flat(row: dict) -> dict:
    """Every scalar one table row prints, the panels and the headline reading the same dict."""
    record = row["record"]
    values = list(row["percentile"].values())
    return {
        "label": row["label"],
        "href": row["href"],
        **row["usable"],
        "aligned_duration_s": record.get("aligned_duration_s"),
        "max_offset_s": record.get("max_offset_s"),
        "offsets_equal": _offsets_equal(record),
        "pct_all_good": record.get("pct_all_good"),
        "percentile_median": round(float(np.median(values)), 1) if values else None,
        "n_above": sum(v >= row["alpha"] for v in values) if values else None,
    }


def _format(value, fmt: str) -> "str | None":
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def _headline_rows(flats: list[dict]) -> list[dict]:
    out = []
    for key, label, fmt in _HEADLINE:
        values = [f[key] for f in flats if f.get(key) is not None]
        if not values:
            continue
        out.append({"label": label,
                    "median": _format(float(np.median(values)), fmt),
                    "range": f"{_format(min(values), fmt)} to {_format(max(values), fmt)}"})
    return out


def build_group_hyper_report(output_dir: Path) -> "Path | None":
    """Render ``cohort_hyper_nirs.html`` over every dyad in a derivatives tree.

    Returns the path, or None when the tree holds no dyad record, which is what a tree that
    has only seen the per-subject pipeline looks like.
    """
    output_dir = Path(output_dir)
    rows = collect_rows(output_dir)
    if not rows:
        logger.warning("no dyad records under %s; run `fnirs-qc hyper-raw` first", output_dir)
        return None

    errors: list[str] = []
    notes: list[str] = []
    scope = output_dir.name
    order = cohort_order(rows)
    flats = [_flat(row) for row in rows]
    by_label = {f["label"]: f for f in flats}

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict = {}
    builders = (("usable_bars", build_usable_bars), ("pair_field", build_pair_field),
                ("condition_dials", build_condition_dials), ("null_strip", build_null_strip))
    for name, builder in builders:
        with guard(f"{name} panel", errors, scope):
            fig = builder(rows, order)
            if fig is None:
                continue
            fname = f"cohort_hyper_{name}.html"
            figure_paths[name] = {"src": f"figures/{fname}",
                                  "h": _save_figure_html(fig, fig_dir / fname)}

    if "null_strip" not in figure_paths:
        note(notes, scope, "no dyad carries a screening verdict, so the null panel is "
                           "empty; it needs records written by a run that measured the "
                           "surrogate null")
    if "pair_field" not in figure_paths:
        note(notes, scope, "no usable-time table found beside the dyad records, so the "
                           "panels that split the usable time are empty")

    tsv_path = output_dir / "cohort_hyper_nirs.tsv"
    pd.DataFrame([{k: v for k, v in f.items() if k != "href"} for f in flats]).to_csv(
        tsv_path, sep="\t", index=False)

    def cell(flat: dict, key: str, fmt: str) -> "str | None":
        text = _format(flat.get(key), fmt)
        if key == "label" and flat.get("href"):
            return f'<a href="{flat["href"]}">{text}</a>'
        return text

    versions = collect_software_versions()
    html = render(
        "group_hyper_report.html.j2",
        **page_vars(title=f"fnirs-pipe cohort QC, hyper ({output_dir.name})",
                    heading="fnirs-pipe Cohort QC (hyperscanning groups)",
                    nav_meta=[("dyads", len(rows))]),
        **footer_vars(versions=versions, errors=errors, notes=notes),
        n_rows=len(rows),
        small_cohort=len(rows) < SMALL_COHORT_N,
        no_shared_pairs="pair_field" not in figure_paths,
        alpha=rows[0]["alpha"],
        summary_meta=[
            ("Dyads", len(rows)),
            ("Tree", output_dir.name),
            ("Alpha", f"top {100 - float(rows[0]['alpha']):.0f}% of the null"),
            ("Table", f'<a href="{tsv_path.name}" download>{tsv_path.name}</a>'),
            ("fnirs-pipe", f"v{versions.get('fnirs-pipe', 'n/a')}"),
        ],
        headline_rows=_headline_rows(flats),
        figure_paths=figure_paths,
        table_columns=[label for _key, label, _fmt in _COLUMNS],
        table_rows=[[cell(by_label[label], key, fmt) for key, _label, fmt in _COLUMNS]
                    for label in order],
    )

    out_path = output_dir / "cohort_hyper_nirs.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("cohort hyper report saved: %s (%d dyads)", out_path, len(rows))
    return out_path
