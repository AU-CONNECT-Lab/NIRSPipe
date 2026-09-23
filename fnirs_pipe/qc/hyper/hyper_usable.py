"""The dyad's shared usable time, as the numbers a cohort reads rather than as a carpet.

A cell is one channel pair in one window, and it is usable only while the pair is coupled in
**both** members at that moment. That intersection is what
:func:`~fnirs_pipe.qc.metrics.hyper.dyad_status` computes; this reduces it to what a
report over twenty dyads can carry: a handful of scalars for the record, and one row per
(pair, condition) for the panels.

Two layers on purpose. The record holds one number per dyad, the way every other quality
record does, and a vector as long as the montage goes beside it in a table, the way the
per-channel SCI scores already do.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.qc.metrics.hyper import dyad_status
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe import __version__

logger = get_logger("qc.hyper_usable")


def _long_rows(grid: dict) -> list[int]:
    """Row indices of ``grid`` that are long pairs, the ones an inter-brain metric uses."""
    long_pairs = set(grid.get("long_pairs") or grid["pairs"])
    return [i for i, name in enumerate(grid["pairs"]) if name in long_pairs]


def _window_s(grid: dict) -> float:
    t = np.asarray(grid["t"], dtype=float)
    return float(np.median(np.diff(t))) if t.size > 1 else 0.0


def _longest_run(flags: np.ndarray) -> int:
    """Longest run of True. ``[T, T, F, T] -> 2``."""
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def usable_scalars(grid: dict, subject_ids: list[str]) -> dict:
    """The dyad-level numbers a cohort table and its outlier rule read.

    ::

      -> {"n_long_pairs": 14, "usable_pairs_mean": 11.8, "usable_window_frac": 0.82,
          "one_member_frac": 0.11, "neither_frac": 0.07, "usable_stretch_s": 170.0,
          "n_windows": 120}

    The three fractions sum to 1 and are the split no subject-level record can make.
    ``one_member_frac`` is an asymmetric loss, one cap or one electrode, and ``neither_frac``
    a shared one, both members out at the same moment; two dyads with the same usable share
    and different middles are two different problems.

    ``usable_stretch_s`` is the median over pairs of that pair's longest unbroken usable run.
    A dyad keeping 60% of its cells in one block and one keeping 60% scattered over two
    hundred single-window holes read the same on the fractions and are not the same
    recording: a coherence window needs contiguous data, and the second dyad has none long
    enough to hold one.
    """
    rows = _long_rows(grid)
    if not rows:
        return {}
    status = dyad_status(grid, subject_ids)[rows]
    usable = status == 2
    window_s = _window_s(grid)
    return {
        "n_long_pairs": len(rows),
        "n_windows": int(status.shape[1]),
        "usable_pairs_mean": round(float(usable.sum(axis=0).mean()), 2),
        "usable_window_frac": round(float(usable.mean()), 4),
        "one_member_frac": round(float((status == 1).mean()), 4),
        "neither_frac": round(float((status == 0).mean()), 4),
        "usable_stretch_s": round(
            float(np.median([_longest_run(row) for row in usable])) * window_s, 1),
    }


def usable_table(grid: dict, subject_ids: list[str],
                 conditions: "dict[str, tuple[float, float]] | None" = None) -> pd.DataFrame:
    """One row per (pair, condition): the three shares inside that block.

    Columns: ``pair``, ``condition``, ``usable_frac``, ``one_member_frac``,
    ``neither_frac``, ``n_windows``. A dyad with no annotated blocks gets one row per pair
    under the condition ``whole run``, so the table has the same shape either way and a
    cohort panel does not have to know which it is reading.

    The windows are on the dyad's aligned clock, :func:`coupled_grid` having already put
    every member on it; a condition span measured on a member's own clock would select the
    wrong columns here by that member's crop offset.
    """
    rows = _long_rows(grid)
    if not rows:
        return pd.DataFrame()
    t = np.asarray(grid["t"], dtype=float)
    status = dyad_status(grid, subject_ids)[rows]
    pairs = [grid["pairs"][i] for i in rows]
    spans = dict(conditions or {}) or {"whole run": (float(t.min()), float(t.max()) + 1)}

    out = []
    for i, pair in enumerate(pairs):
        for name, (start, stop) in spans.items():
            keep = (t >= start) & (t < stop)
            if not keep.any():
                continue
            cells = status[i, keep]
            out.append({
                "pair": pair, "condition": str(name),
                "usable_frac": round(float((cells == 2).mean()), 4),
                "one_member_frac": round(float((cells == 1).mean()), 4),
                "neither_frac": round(float((cells == 0).mean()), 4),
                "n_windows": int(keep.sum()),
            })
    return pd.DataFrame(out)


def write_usable_table(path: Path, grid: dict, subject_ids: list[str],
                       conditions: "dict[str, tuple[float, float]] | None",
                       sources: "list[str] | None" = None, **params) -> "Path | None":
    """Write the usable-time table with its sidecar. None when there is nothing to write."""
    from fnirs_pipe.io.derivatives import write_sidecar_json

    table = usable_table(grid, subject_ids, conditions)
    if table.empty:
        return None
    table.to_csv(path, sep="\t", index=False)
    write_sidecar_json(path, {
        "pipeline_version": __version__,
        "step": "hyper_usable",
        "Sources": sources or [],
        "parameters": {"window_s": round(_window_s(grid), 3),
                       "n_long_pairs": len(_long_rows(grid)), **params},
    })
    logger.info("usable-time table -> %s", path)
    return path
