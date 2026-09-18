"""Circular connectivity (connectogram) figure.

::

  fc_connectogram():    within-subject FC connectogram from a channel × channel DataFrame.
                        HbO and HbR are drawn as separate circles, side by side.

The circle itself is :mod:`fnirs_pipe.qc.figures.common.circle_map`, shared with the
inter-brain connectogram in ``hyper_post_figures``: one subject's channels against each
other and two members' against each other are different matrices and the same picture.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.figures.common.circle_map import (
    SOURCE_GAP, bezier, circle_axes, node_arc, radial_label, ring_angles,
)
from fnirs_pipe.qc.figures.common.matrix_map import CORRELATION_SCALE, scale_color
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.connectogram")

_CIRCLE_PX = 470
_CHORD_WIDTH = 1.6


def _source_groups(ch_names: list[str]) -> dict[str, str]:
    """Auto-group channels by source label (S1, S2, …) from names like 'S1_D1 hbo'."""
    groups: dict[str, str] = {}
    for ch in ch_names:
        m = re.match(r"(S\d+)", ch)
        groups[ch] = m.group(1) if m else "other"
    return groups


def _keep_edges(mat: np.ndarray, threshold: float,
                n_lines: "int | None") -> np.ndarray:
    """Blank every edge the circle should not draw, as NaN rather than as zero.

    ::

      [[0, .1, .8], ...] at threshold .3 -> [[nan, nan, .8], ...]

    NaN rather than zero because a zero is a value: it lands mid-colour-scale and draws a
    near-white chord, which is the haze that used to sit under every real one. The matrix is
    one subject against itself, so the diagonal is a channel against itself and every edge
    appears twice; ``n_lines`` is a count of edges, not of cells.
    """
    out = np.array(mat, dtype=float, copy=True)
    np.fill_diagonal(out, np.nan)
    if n_lines is None:
        out[np.abs(out) < threshold] = np.nan
        return out
    finite = np.abs(out[np.isfinite(out)])
    if finite.size > n_lines * 2:
        out[np.abs(out) < np.sort(finite)[-n_lines * 2]] = np.nan
    return out


def _ordered(names: list[str], groups: dict[str, str]) -> "tuple[list[str], list[int]]":
    """Channels sorted into their source blocks, and the size of each block."""
    order = list(dict.fromkeys(groups.get(c, "other") for c in names))
    ranked = sorted(names, key=lambda c: (order.index(groups.get(c, "other")), c))
    counts = [sum(1 for c in ranked if groups.get(c, "other") == g) for g in order]
    return ranked, counts


def _circle(fig, mat: np.ndarray, names: list[str], groups: dict[str, str],
            threshold: float, n_lines: "int | None", col: int) -> int:
    """One chromophore's circle into a subplot. Returns the number of chords drawn."""
    ranked, counts = _ordered(names, groups)
    idx = [names.index(c) for c in ranked]
    kept = _keep_edges(mat[np.ix_(idx, idx)], threshold, n_lines)

    ang = np.deg2rad(ring_angles(counts, gap=SOURCE_GAP))
    xy = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    step = float(ang[1] - ang[0]) if len(ang) > 1 else np.deg2rad(20.0)

    for k, name in enumerate(ranked):
        node_arc(fig, ang[k], step * 0.42, name, 1, col)
        radial_label(fig, ang[k], 1.06, name.rsplit(" ", 1)[0], 1, col, size=7)

    # the upper triangle alone: the matrix is symmetric, so drawing both halves lays every
    # chord over itself and doubles the file for no second reading
    pairs = [(i, j) for i in range(len(ranked)) for j in range(i + 1, len(ranked))
             if np.isfinite(kept[i, j])]
    # weakest first, so a strong connection is never drawn under a weak one
    pairs.sort(key=lambda ij: abs(kept[ij]))
    for i, j in pairs:
        value = float(kept[i, j])
        x, y = bezier(xy[i], xy[j])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", showlegend=False,
            line=dict(color=scale_color(value, CORRELATION_SCALE, -1.0, 1.0),
                      width=_CHORD_WIDTH),
            hovertemplate=(f"{ranked[i]} x {ranked[j]}<br>r = {value:.3f}<extra></extra>"),
        ), row=1, col=col)
    circle_axes(fig, 1, col)
    return len(pairs)


def fc_connectogram(
    fc_df: pd.DataFrame,
    fc_hbr_df: "pd.DataFrame | None" = None,
    groups: "dict[str, str] | None" = None,
    threshold: float = 0.3,
    n_lines: "int | None" = None,
    title: str = "FC connectogram",
) -> "go.Figure":
    """One connectogram per chromophore present, side by side.

    Nodes are the channels, blocked by source with blank circle between blocks, and a chord
    is one channel pair coloured by its correlation on the report's own scale. **Nodes carry
    no colour of their own**: the blocks are already separated by position, so hue there
    would repeat what the gaps say and compete with the chords, which are what the figure is
    for.

    Args:
        fc_df:      Square channel x channel Pearson r DataFrame (from compute_fc).
                    compute_fc returns one matrix per chromophore, so this is the HbO one;
                    a single matrix holding both is also accepted and split by suffix.
        fc_hbr_df:  The HbR matrix, when the two are supplied separately.
        groups:     Optional mapping channel -> group label. Defaults to the source label
                    (S1, S2, ...). Rejected channels are left off the circle: compute_fc
                    blanks their row, so they carry no edge and a node kept for one would
                    claim the ring has a channel it did not measure.
        threshold:  Minimum ``|r|`` to draw a connection. Weaker edges are hidden.
        n_lines:    If set, draw only the top-N strongest connections (overrides threshold).
        title:      Base title; the chromophore names the panel.
    """
    subsets: list[tuple[list[str], np.ndarray, str]] = []
    for frame in (fc_df, fc_hbr_df):
        if frame is None or frame.empty:
            continue
        all_ch = frame.columns.tolist()
        fc_mat = frame.to_numpy(dtype=float).copy()
        drawable = ~np.all(np.isnan(fc_mat), axis=1)
        np.fill_diagonal(fc_mat, 0.0)
        for suffix, label in ((" hbo", "HbO"), (" hbr", "HbR")):
            names = [c for i, c in enumerate(all_ch) if c.endswith(suffix) and drawable[i]]
            if not names:
                continue
            idx = [all_ch.index(c) for c in names]
            subsets.append((names, fc_mat[np.ix_(idx, idx)], label))

    if not subsets:
        raise ValueError("fc_df has no recognised HbO/HbR channels")
    if groups is None:
        groups = _source_groups([c for names, _, _ in subsets for c in names])

    rule = (f"top {n_lines} connections" if n_lines is not None
            else f"|r| >= {threshold:g}")
    # wide, because the radial labels run outside the axis and plotly does not clip an
    # annotation to its subplot: too little here and one circle's labels land on the next
    fig = make_subplots(rows=1, cols=len(subsets), horizontal_spacing=0.22,
                        subplot_titles=[lab for _, _, lab in subsets])
    drawn = 0
    for col, (names, mat, _label) in enumerate(subsets, start=1):
        drawn += _circle(fig, mat, names, groups, threshold, n_lines, col)

    # the one trace that carries the scale: the chords are individual lines and none of them
    # can show a colour bar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=CORRELATION_SCALE, cmin=-1.0, cmax=1.0, showscale=True,
                    color=[0], opacity=0,
                    colorbar=dict(title=dict(text="Pearson r", side="right",
                                             font=dict(size=10)),
                                  thickness=12, len=0.7, tickfont=dict(size=9),
                                  tickvals=[-1, -0.5, 0, 0.5, 1])),
    ), row=1, col=1)

    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(
        height=_CIRCLE_PX + 110, plot_bgcolor="white", showlegend=False,
        margin=dict(l=56, r=40, t=76, b=20),
        title=dict(text=f"{title} ({rule}, {drawn} drawn)", x=0.01, font=dict(size=13)))
    return fig
