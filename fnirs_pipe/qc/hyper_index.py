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
import re
import statistics
from pathlib import Path

import pandas as pd

from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.figure_io import _pair_fname
from fnirs_pipe.qc.report_shell import footer_vars, page_vars, render, stylesheet
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_index")

# How far from the other rows a window has to sit before its cell is marked. The subject
# index's own threshold, so a flag means the same thing on both pages.
_OUTLIER_Z = 3.5

_CHROMA_LABEL = {"hbo": "HbO", "hbr": "HbR"}


def _read_tsv(path: Path) -> "pd.DataFrame | None":
    try:
        return pd.read_csv(path, sep="\t")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("unreadable table, its rows go unlisted: %s (%s)", path.name, exc)
        return None


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


def _mean_by_chroma(df: "pd.DataFrame | None", column: str,
                    where: "tuple[str, str] | None" = None) -> dict:
    """``{chromophore: mean of column}``, over one condition's rows when ``where`` is given.

    ::

      _mean_by_chroma(wtcbycond, "coherence", ("condition", "game1"))
      -> {"hbo": 0.2538, "hbr": 0.2483}

    Homologous pairs only. A crossed table holds every channel against every other, and the
    mean over all of those is dominated by pairings of unrelated sites: it is a different
    quantity from the homologous mean and moves differently between conditions, so mixing
    the two down one column would make a crossed run and an uncrossed one incomparable.
    """
    if df is None or column not in df.columns:
        return {}
    if where is not None:
        key, value = where
        if key not in df.columns:
            return {}
        df = df[df[key] == value]
    if "label2" in df.columns:
        df = df[df["label"] == df["label2"]]
    if df.empty or "chromophore" not in df.columns:
        return {}
    return {str(c): float(v) for c, v in df.groupby("chromophore")[column].mean().items()}


def _isc_mean(nirs_dir: Path, stem: str, label: "str | None" = None) -> dict:
    """``{chromophore: mean same-channel ISC}`` from the two ISC matrices of one window.

    The diagonal, which is a channel against the other member's copy of the same channel.
    The off-diagonal is every site against every other and belongs to the connectogram, not
    to a single number. NaN cells are the channels a member lost and are skipped.

    ``label`` reads a condition's own matrices, written under the ``desc-`` entity its page
    takes. A window analysed before per-condition ISC existed has none, and the row shows a
    dash rather than the run's number.
    """
    out: dict = {}
    desc = f"_desc-{_pair_fname(label)}" if label else ""
    for chroma in ("hbo", "hbr"):
        df = _read_tsv(nirs_dir / f"{stem}{desc}_hyper-isc-{chroma}.tsv")
        if df is None or df.empty:
            continue
        values = df.set_index(df.columns[0]).to_numpy(dtype=float)
        if values.shape[0] != values.shape[1]:
            continue
        diagonal = pd.Series(values.diagonal()).dropna()
        if not diagonal.empty:
            out[chroma] = float(diagonal.mean())
    return out


def _outlier_flags(values: "list[float | None]") -> "list[bool]":
    """Which rows sit apart from the others on one metric.

    Scaled by the median absolute deviation, so the row being looked for cannot widen the
    scale meant to catch it. Under four rows there is nothing to compare against, which is
    why a two-condition design gets no flags at all.

    ``[0.26, 0.25, 0.26, 0.41, 0.27] -> [False, False, False, True, False]``
    """
    present = [v for v in values if v is not None]
    if len(present) < 4:
        return [False] * len(values)
    median = statistics.median(present)
    deviations = [abs(v - median) for v in present]
    scale = statistics.median(deviations) * 1.4826
    if scale == 0:
        scale = sum(deviations) / len(deviations)
    if scale == 0:
        return [False] * len(values)
    return [v is not None and abs(v - median) / scale >= _OUTLIER_Z for v in values]


def _tasks(nirs_dir: Path, group_id: str) -> "list[str]":
    """The tasks this dyad has a whole-run coherence table for, in filename order."""
    pattern = re.compile(rf"group-{re.escape(group_id)}_task-([A-Za-z0-9]+)_hyper-wtc\.tsv")
    return sorted({m.group(1) for p in nirs_dir.glob("*_hyper-wtc.tsv")
                   if (m := pattern.fullmatch(p.name))})


def collect_rows(group_dir: Path, group_id: str) -> "list[dict]":
    """One row per analysed window under ``group_dir``, for the index table.

    A task contributes its whole-run row and then one row per condition found in its
    ``hyper-wtcbycond.tsv``. The conditions come in the order that table lists them, which
    is the order the windows were found in the recording, so the rows read down the session
    rather than alphabetically.
    """
    nirs_dir = group_dir / "nirs"
    rows: list[dict] = []

    for task in _tasks(nirs_dir, group_id):
        stem = f"group-{group_id}_task-{task}"
        whole = _read_tsv(nirs_dir / f"{stem}_hyper-wtc.tsv")
        bycond = _read_tsv(nirs_dir / f"{stem}_hyper-wtcbycond.tsv")

        def _row(label: "str | None", where=None) -> dict:
            source = whole if where is None else bycond
            desc = "hyperpost" if label is None else f"{_pair_fname(label)}_hyperpost"
            report = group_dir / f"{stem}_desc-{desc}_nirs.html"
            return {
                "task": task,
                "condition": label,
                "kind": "whole run" if label is None else "window",
                "href": report.name if report.exists() else None,
                "coherence": _mean_by_chroma(source, "coherence", where),
                "valid_frac": _mean_by_chroma(source, "n_valid_frac", where),
                "isc": _isc_mean(nirs_dir, stem, label),
                "window": _window_of(nirs_dir / f"{stem}_hyper-wtcbycond.tsv", label),
            }

        rows.append(_row(None))
        if bycond is not None and "condition" in bycond.columns:
            for label in list(dict.fromkeys(bycond["condition"].astype(str))):
                rows.append(_row(label, ("condition", label)))

    # a window is marked against the dyad's other windows, so this waits until every row is
    # in hand. Flagged per chromophore, the two being separate measurements
    for chroma in ("hbo", "hbr"):
        flags = _outlier_flags([r["coherence"].get(chroma) for r in rows])
        for row, flagged in zip(rows, flags):
            row.setdefault("flagged", {})[chroma] = flagged
    return rows


def _window_of(bycond_path: Path, label: "str | None") -> str:
    """``"521.3-1421.3 s"`` for one condition, off the table's sidecar, or ""."""
    if label is None:
        return ""
    try:
        spans = (json.loads(bycond_path.with_suffix(".json").read_text(encoding="utf-8"))
                 ["parameters"]["condition_windows_s"])
        lo, hi = spans[label]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return ""
    return f"{lo:.0f}–{hi:.0f} s"


def write_hyper_index(
    group_dir: Path,
    group_id: str,
    subject_ids: "list[str] | None" = None,
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
    band = _band(group_dir / "nirs" / f"group-{group_id}_task-{rows[0]['task']}_hyper-wtc.tsv")

    html = render(
        "hyper_index.html.j2",
        **page_vars(
            title=f"fnirs-pipe hyper — group-{group_id}",
            heading=f"fnirs-pipe hyper — group-{group_id}",
            css=stylesheet("subject.css"),
        ),
        **footer_vars(versions=collect_software_versions()),
        group_id=group_id,
        subject_ids=subject_ids or [],
        rows=rows,
        chroma=chroma,
        chroma_labels={c: _CHROMA_LABEL[c] for c in chroma},
        band=band,
        n_tasks=len({row["task"] for row in rows}),
        outlier_z=_OUTLIER_Z,
    )
    out_path = group_dir / f"group-{group_id}_index.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("group-%s | dyad index saved: %s", group_id, out_path)
    return out_path
