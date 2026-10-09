"""Dyad-level landing page: one row per analysed window, linking to that window's report.

Every hyperscanning figure belongs to one window of one task, so this is an index rather
than a report. What it adds on top of the links is the one thing no single page can show:
the conditions side by side on the number the analysis exists to produce.

It reads the tables the run already wrote rather than anything held in memory, which is
what lets it serve both orders the pipeline can be driven in. A recording preprocessed
whole gives one task with a ``condition`` column, so the rows are that task's windows; a
tree cropped per condition first gives one task per condition and no such column, so the
rows are the tasks. Either way the page can be rebuilt for a tree produced weeks ago, and
a tree holding both is listed as what it is: the two are different analyses of the same
recording, and the ``kind`` column says which a row came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.io.tables import read_tsv_or_none
from fnirs_pipe.io.naming import report_name, derivative_path, parse_path
from fnirs_pipe.qc.common.figure_io import _pair_fname, figure_namer, pair_slug
from fnirs_pipe.qc.common.report_shell import (
    OUTLIER_Z, footer_vars, outlier_flags, page_vars, render)
from fnirs_pipe.utils import fisher_r_to_z
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.io.derivatives import entity_of, group_label

logger = get_logger("qc.hyper_index")

# what an unreadable table costs this page, for the warning
_LOST = "its rows go unlisted"

_CHROMA_LABEL = {"hbo": "HbO", "hbr": "HbR"}

# What a value has to beat to be counted past its own null. The same 95 the null tables'
# `null_p95` column is drawn at, so the count and that column say one thing.
NULL_PERCENTILE = 95

# Other products of a task, as (link text, path relative to group_dir). Only the ones on
# disk reach the page; the window page a row already links is not repeated here.
_ARTEFACTS = (
    ("raw QC",     "{raw_report}"),
    ("provenance", "figures/{provenance}"),
    ("coherence",  "{nirs}/{coherence}"),
    ("null",       "{nirs}/{null}"),
    ("ISC pairs",  "{nirs}/{isc}"),
)


def _table(stem: str, **entities) -> str:
    """One of a dyad's tables by name, from the ``group-<id>_task-<task>`` stem it shares.

    ::

      _table("group-G1_task-rest", statistic="wtc")
        -> "group-G1_task-rest_stat-wtc_relmat.tsv"

    Built through the namer rather than spelled out, so this page's links and the writers
    that produce those files cannot be renamed apart.
    """
    return derivative_path("", "relmat", ".tsv",
                           group=entity_of(stem, "group"),
                           session=entity_of(stem, "ses"),
                           task=entity_of(stem, "task"), **entities).name


def _band(path: Path) -> str:
    """The band a table was averaged over, off its sidecar, for the column heading.

    Empty when the sidecar is missing rather than guessed: a coherence value means nothing
    without the band, so a heading that named one it could not read would be worse than a
    heading that names none.
    """
    try:
        params = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        lo, hi = params["parameters"]["band_fmin"], params["parameters"]["band_fmax"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return ""
    return f"{lo:.3g}–{hi:.3g} Hz"


def _pairings(*frames) -> list:
    """The member pairings a group's tables carry, in the order they were written.

    Read off the tables rather than off a member list: the index is rebuilt for trees it did
    not produce, and a pairing that failed has no rows and belongs on no row of the page. A
    table from before the columns existed reports one unnamed pairing, which is what a dyad
    always was.
    """
    for df in frames:
        if df is None or not {"sub1", "sub2"}.issubset(getattr(df, "columns", [])):
            continue
        seen = dict.fromkeys(zip(df["sub1"].astype(str), df["sub2"].astype(str)))
        if seen:
            return list(seen)
    return [None]


def _crossed_rows(df: "pd.DataFrame") -> bool:
    """Whether a band or null table holds pairings off the diagonal, which a crossed run writes."""
    return "label2" in df.columns and bool((df["label"] != df["label2"]).any())


def _mean_by_chroma(df: "pd.DataFrame | None", column: str,
                    where: "tuple[str, str] | None" = None,
                    pair: "tuple[str, str] | None" = None, crossed: bool = False) -> dict:
    """``{chromophore: mean of column}``, over one condition's rows when ``where`` is given.

    ::

      _mean_by_chroma(wtcbycond, "coherence", ("condition", "task1"))
      -> {"hbo": 0.25, "hbr": 0.24}

    Homologous pairs by default; ``crossed`` takes every pairing instead, and is empty for an
    uncrossed table. The two are different quantities and get a column each.
    """
    if df is None or column not in df.columns:
        return {}
    if pair is not None and {"sub1", "sub2"}.issubset(df.columns):
        df = df[(df["sub1"] == pair[0]) & (df["sub2"] == pair[1])]
    if where is not None:
        key, value = where
        if key not in df.columns:
            return {}
        df = df[df[key] == value]
    if crossed:
        if not _crossed_rows(df):
            return {}
    elif "label2" in df.columns:
        df = df[df["label"] == df["label2"]]
    if df.empty or "chromophore" not in df.columns:
        return {}
    return {str(c): float(v) for c, v in df.groupby("chromophore")[column].mean().items()}


def _past_null(df: "pd.DataFrame | None", where: "tuple[str, str] | None" = None,
               pair: "tuple[str, str] | None" = None, crossed: bool = False) -> dict:
    """``{chromophore: (pairs past their null, pairs measured)}`` for one window.

    ::

      _past_null(null_table, ("condition", "task1")) -> {"hbo": (2, 10), "hbr": (0, 10)}

    Coherence has a floor that moves with the window, so two windows' raw values do not
    compare and neither is readable on its own. Each channel pair's own surrogate draws are
    the scale that makes them both, and this is that scale reduced to the one number a table
    cell holds: how many of the dyad's pairs the real value beat the draws for.

    Over the homologous pairs by default, every pairing with ``crossed``, as
    :func:`_mean_by_chroma` splits them.
    """
    if df is None or "percentile" not in df.columns:
        return {}
    if pair is not None and {"sub1", "sub2"}.issubset(df.columns):
        df = df[(df["sub1"] == pair[0]) & (df["sub2"] == pair[1])]
    if where is not None:
        key, value = where
        if key not in df.columns:
            return {}
        df = df[df[key] == value]
    if crossed:
        if not _crossed_rows(df):
            return {}
    elif "label2" in df.columns:
        df = df[df["label"] == df["label2"]]
    if df.empty or "chromophore" not in df.columns:
        return {}
    out = {}
    for chroma, part in df.groupby("chromophore"):
        ranks = part["percentile"].dropna()
        if not ranks.empty:
            out[str(chroma)] = (int((ranks >= NULL_PERCENTILE).sum()), int(ranks.size))
    return out


def _isc_mean(nirs_dir: Path, stem: str, label: "str | None" = None,
              slug: str = "", crossed: bool = False) -> dict:
    """``{chromophore: mean ISC}`` from the two ISC matrices of one window, in Fisher z.

    The diagonal by default, a channel against the other member's copy of it; ``crossed``
    every cell, and nothing for an uncrossed run, whose off-diagonal is blank. NaN cells are
    the channels a member lost and are skipped. Averaged in z as the ROI ISC is.

    ``label`` reads a condition's own matrices, written under the ``cond-`` entity its page
    takes. A window analysed before per-condition ISC existed has none, and the row shows a
    dash rather than the run's number.

    ``slug`` picks one pairing's matrices out of a group that wrote several.
    """
    out: dict = {}
    for chroma in ("hbo", "hbr"):
        df = read_tsv_or_none(nirs_dir / _table(
            stem, pairing=slug.lstrip("_") or None, chromophore=chroma,
            condition=_pair_fname(label) if label else None, statistic="isc"), _LOST)
        if df is None or df.empty:
            continue
        values = df.set_index(df.columns[0]).to_numpy(dtype=float)
        if values.shape[0] != values.shape[1]:
            continue
        off = ~np.eye(values.shape[0], dtype=bool)
        if crossed and not np.isfinite(values[off]).any():
            continue
        cells = values if crossed else values.diagonal()
        z = fisher_r_to_z(cells[np.isfinite(cells)])
        if z.size:
            out[chroma] = float(np.tanh(z.mean()))
    return out


def _links(group_dir: Path, stem: str, nirs: str = "nirs") -> list[dict[str, str]]:
    return [{"text": text, "href": rel}
            for text, template in _ARTEFACTS
            if (group_dir / (rel := template.format(
                nirs=nirs,
                raw_report=report_name(stem, desc="raw"),
                provenance=figure_namer(stem)("provenance", extension=".png"),
                coherence=_table(stem, statistic="wtc"),
                null=_table(stem, nulldist="phase", statistic="wtc"),
                isc=_table(stem, statistic="isc")))).exists()]


def _tasks(nirs_dir: Path, group_id: str) -> "list[str]":
    """The tasks this dyad has a whole-run coherence table for, in filename order.

    Matched on the entities rather than by a regex over the name: the whole-run table is the
    one carrying neither a condition nor a null, which a glob cannot say.
    """
    tasks = set()
    for path in nirs_dir.glob("group-*_stat-wtc_relmat.tsv"):
        entities = parse_path(path.name)
        if entities.get("group") != group_id or not entities.get("task"):
            continue
        if {"condition", "nulldist", "aggregation", "desc"} & set(entities):
            continue
        tasks.add(entities["task"])
    return sorted(tasks)


def _nirs_dirs(group_dir: Path) -> "list[tuple[str | None, Path]]":
    """``[(session, nirs_dir)]``: ``group-G/nirs`` and each ``group-G/ses-S/nirs``."""
    found = [(None, group_dir / "nirs")]
    found += [(d.parent.name.removeprefix("ses-"), d)
              for d in sorted(group_dir.glob("ses-*/nirs")) if d.is_dir()]
    return found


def collect_rows(group_dir: Path, group_id: str) -> "list[dict]":
    """One row per analysed window under ``group_dir``, for the index table.

    A task contributes its whole-run row and then one row per condition found in its
    ``cond-all`` table. The conditions come in the order that table lists them, which
    is the order the windows were found in the recording, so the rows read down the session
    rather than alphabetically. A group recorded per session lists each session's tasks.
    """
    rows: list[dict] = []

    for session, nirs_dir in _nirs_dirs(group_dir):
        rows += _session_rows(group_dir, group_id, session, nirs_dir)

    # a window is marked against the dyad's other windows, so this waits until every row is
    # in hand. Flagged per chromophore, the two being separate measurements
    for chroma in ("hbo", "hbr"):
        for key, flag_key in (("coherence", "flagged"), ("coherence_crossed", "flagged_crossed")):
            flags = outlier_flags([r[key].get(chroma) for r in rows])
            for row, flagged in zip(rows, flags):
                row.setdefault(flag_key, {})[chroma] = flagged
    return rows


def _session_rows(group_dir: Path, group_id: str, session: "str | None",
                  nirs_dir: Path) -> "list[dict]":
    rows: list[dict] = []
    nirs = nirs_dir.relative_to(group_dir).as_posix()
    for task in _tasks(nirs_dir, group_id):
        stem = group_label(group_id, task, session)
        whole = read_tsv_or_none(nirs_dir / _table(stem, statistic="wtc"), _LOST)
        bycond = read_tsv_or_none(nirs_dir / _table(stem, condition="all", statistic="wtc"),
                                  _LOST)
        # written only when the run drew a null; a tree without one keeps the column empty
        whole_null = read_tsv_or_none(nirs_dir / _table(stem, nulldist="phase", statistic="wtc"),
                                      _LOST)
        bycond_null = read_tsv_or_none(nirs_dir / _table(stem, condition="all",
                                                  nulldist="phase", statistic="wtc"), _LOST)
        # every inter-brain number is of two members, so a group of three contributes three
        # rows per window, one per pairing, rather than one row averaging across them
        pairings = _pairings(whole, bycond)

        def _row(label: "str | None", pair, where=None) -> dict:
            source = whole if where is None else bycond
            null = whole_null if where is None else bycond_null
            slug = pair_slug(pair, len(pairings))
            report = group_dir / report_name(
                stem, condition=_pair_fname(label) if label else None,
                pairing=slug.lstrip("_") or None)
            return {
                "session": session,
                "task": task,
                "condition": label,
                "pair": " × ".join(pair) if pair and len(pairings) > 1 else None,
                "kind": "whole run" if label is None else "window",
                "href": report.name if report.exists() else None,
                # the task's other products hang off its whole-run row, being the task's and
                # not one window's; the raw report among them, which nothing else links
                "links": _links(group_dir, stem, nirs) if label is None else [],
                "coherence": _mean_by_chroma(source, "coherence", where, pair),
                "coherence_crossed": _mean_by_chroma(source, "coherence", where, pair,
                                                     crossed=True),
                "past_null": _past_null(null, where, pair),
                "past_null_crossed": _past_null(null, where, pair, crossed=True),
                "valid_frac": _mean_by_chroma(source, "n_valid_frac", where, pair),
                "isc": _isc_mean(nirs_dir, stem, label, slug),
                "isc_crossed": _isc_mean(nirs_dir, stem, label, slug, crossed=True),
                "window": _window_of(
                    nirs_dir / _table(stem, condition="all", statistic="wtc"), label),
            }

        for pair in pairings:
            rows.append(_row(None, pair))
            if bycond is not None and "condition" in bycond.columns:
                for label in list(dict.fromkeys(bycond["condition"].astype(str))):
                    rows.append(_row(label, pair, ("condition", label)))
    return rows


def _window_of(bycond_path: Path, label: "str | None") -> str:
    """``"300.0-1200.0 s"`` for one condition, off the table's sidecar, or ""."""
    if label is None:
        return ""
    try:
        spans = (json.loads(bycond_path.with_suffix(".json").read_text(encoding="utf-8"))
                 ["parameters"]["condition_windows_s"])
        lo, hi = spans[label]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ""
    # `+ 0.0` so a window starting at the recording's own zero does not print as "-0":
    # `condition_windows` subtracts the trigger offset, which lands on a negative zero
    return f"{lo + 0.0:.0f}–{hi + 0.0:.0f} s"


def write_hyper_index(
    group_dir: Path,
    group_id: str,
    subject_ids: "list[str] | None" = None,
    run_command: str = "",
) -> "Path | None":
    """Render ``group-<id>_index.html`` over the dyad's report pages.

    Returns None when the dyad has no coherence table, which is what a directory holding
    only a raw report looks like.
    """
    rows = collect_rows(group_dir, group_id)
    if not rows:
        logger.warning("group-%s | no coherence table found, index not written", group_id)
        return None

    chroma = [c for c in ("hbo", "hbr")
              if any(c in row["coherence"] for row in rows)]
    first = rows[0]
    band = _band(group_dir / (f"ses-{first['session']}/nirs" if first["session"] else "nirs")
                 / _table(group_label(group_id, first["task"], first["session"]),
                          statistic="wtc"))

    html = render(
        "hyper_index.html.j2",
        **page_vars(
            title=f"fnirs-pipe hyper  ·  group-{group_id}",
            heading=f"fnirs-pipe hyper  ·  group-{group_id}",
        ),
        **footer_vars(versions=collect_software_versions()),
        group_id=group_id,
        subject_ids=subject_ids or [],
        rows=rows,
        has_pairs=any(row.get("pair") for row in rows),
        has_sessions=any(row["session"] for row in rows),
        chroma=chroma,
        chroma_labels={c: _CHROMA_LABEL[c] for c in chroma},
        band=band,
        n_tasks=len({row["task"] for row in rows}),
        outlier_z=OUTLIER_Z,
        null_percentile=NULL_PERCENTILE,
        has_null=any(row["past_null"] or row["past_null_crossed"] for row in rows),
        run_command=run_command,
    )
    out_path = group_dir / report_name(f"group-{group_id}", desc="index")
    out_path.write_text(html, encoding="utf-8")
    logger.info("group-%s | dyad index saved: %s", group_id, out_path)
    return out_path
