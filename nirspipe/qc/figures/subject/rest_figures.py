"""Resting-state QC figures.

rest_channel_panel():   the run's channels in one panel, FC matrices over ALFF and fALFF.
alff_brain_figure():    mALFF projected onto the cortex, as the GLM activation is.
alff_flat_figure():     mALFF on the optode flat map, for optodes no reading placed.
fc_roi_matrix_figure(): connectivity ROI by ROI instead of channel by channel.
fc_seed_topo_figure():  seed-to-whole-brain correlations drawn on the optode flat map.

The panel answers how much and with whom, the brain and the flat map answer where. The seed
map stays on the shared head in :mod:`nirspipe.qc.figures.common.head_map`, where each
channel keeps its own place next to the seed's.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from plotly.subplots import make_subplots

from nirspipe.pipeline.chromophores import chroma_channels, chroma_of
from nirspipe.qc.common.figure_io import fig_png_b64, stack_png_b64
from nirspipe.utils import pair_of, pair_order
from nirspipe.utils.logging import get_logger

from nirspipe.qc.figures.common._brain_utils import RENDER_LOCK
from nirspipe.qc.figures.common._cortex import project_to_cortex, render_cortex
from nirspipe.qc.figures.common._surface_overlay import coverage_rgba
from nirspipe.qc.figures.common._utils import CHROMA_STYLE
from nirspipe.qc.figures.common.head_map import (
    BLANK_COLOR, head_axes, head_geometry, head_glyph, head_ground,
)
from nirspipe.qc.figures.common.matrix_map import (
    CORRELATION_SCALE, cell_values, kept_axes,
)
from nirspipe.qc.figures.common.motion_panel import _roi_order, roi_colours

logger = get_logger("qc.figures.rest")

_MEAN_LINE_COLOR = "#555555"


# The blocks a rest panel's channels are laid out in, and what each is called. Same split and
# same order as the per-channel table, so a reader moving between them finds the same groups.
_SEP_BLOCKS = (("long", "long"), ("mid", "neither range"), ("short", "short"))
# Blank x positions between blocks, so the grouping needs no rule drawn through the panel.
_BLOCK_GAP = 1.4
_STRIP_MARK, _CONNECTOR, _REJECTED_INK = 8, "#c3cad1", "#b9c2cb"
# The panel's two bands, in pixels: the matrices are square and need the room, a strip does
# not, and a strip added later must not shrink the matrices.
_PANEL_MATRIX_PX, _PANEL_STRIP_PX = 560, 230
# the ROI colour strip beside a matrix, in cells left of (and above) its first row
_ROI_STRIP = (-1.35, -0.8)


def _pair_blocks(pairs, raw, sep_bands):
    """The panel's channel order: ``(pairs, x positions, block spans)``.

    Grouped by separation and, inside a group, by source then detector number, the order
    most montages are recorded in and so the per-channel table's.

    Without ``raw`` there is no separation to block by and the pairs are one list, which is
    what a caller holding only the tables can draw.
    """
    groups = {}
    if raw is not None:
        from nirspipe.qc.metrics import long_short_channels
        long_names, short_names = long_short_channels(raw, sep_bands)
        of = {pair_of(c): "long" for c in long_names}
        of.update({pair_of(c): "short" for c in short_names})
        for p in pairs:
            groups.setdefault(of.get(p, "mid"), []).append(p)
        blocks = [(key, label) for key, label in _SEP_BLOCKS if groups.get(key)]
    else:
        groups, blocks = {"long": list(pairs)}, [("long", "")]

    order, xs, spans, x = [], [], [], 0.0
    for key, label in blocks:
        members = sorted(groups.get(key, []), key=pair_order)
        if not members:
            continue
        start = x
        for p in members:
            order.append(p)
            xs.append(x)
            x += 1.0
        spans.append((start, x - 1.0, label))
        x += _BLOCK_GAP
    return order, np.asarray(xs, dtype=float), spans


def _roi_blocks(pairs, roi_map):
    """The panel's channel order by ROI: ``(pairs, x positions, block spans, groups)``.

    ``groups`` is :func:`_roi_order`'s ``[(ROI, first row, last row), ...]``; a pair no ROI
    claims, a short channel among them, goes last under "unassigned".
    """
    idx, groups = _roi_order(pairs, roi_map)
    order = [pairs[i] for i in idx]
    xs, spans, x = [], [], 0.0
    for label, i0, i1 in groups:
        start = x
        for _ in range(i0, i1 + 1):
            xs.append(x)
            x += 1.0
        spans.append((start, x - 1.0, label))
        x += _BLOCK_GAP
    return order, np.asarray(xs, dtype=float), spans, groups


def _panel_matrix(fig, frame, order, chromo, col, show_bar, lim, groups, colours):
    """One chromophore's full channel matrix into the panel's top row, on the panel's order.

    The axes are categorical, which places category i at i, so the ROI rules and strips are
    drawn in cell units.
    """
    names = [f"{p} {chromo}" for p in order]
    # to_numpy can hand back a read-only view, and the matrix is blanked in place
    mat = np.array(frame.reindex(index=names, columns=names).to_numpy(dtype=float), copy=True)
    np.fill_diagonal(mat, np.nan)
    n = len(order)
    fig.add_trace(go.Heatmap(
        z=mat.astype(np.float32), x=order, y=order, zmin=-lim, zmax=lim,
        colorscale=CORRELATION_SCALE, showscale=show_bar,
        colorbar=dict(title=dict(text="Pearson r", side="right", font=dict(size=10)),
                      thickness=12, len=0.46, y=1.0, yanchor="top", tickfont=dict(size=9)),
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
    ), row=1, col=col)
    xref, yref = ("x", "y") if col == 1 else (f"x{col}", f"y{col}")
    rule = dict(color="#555", width=0.8)
    for _label, i0, _i1 in groups[1:]:
        fig.add_shape(type="line", x0=i0 - 0.5, x1=i0 - 0.5, y0=-0.5, y1=n - 0.5,
                      xref=xref, yref=yref, line=rule)
        fig.add_shape(type="line", y0=i0 - 0.5, y1=i0 - 0.5, x0=-0.5, x1=n - 0.5,
                      xref=xref, yref=yref, line=rule)
    for label, i0, i1 in groups:
        for x0, x1, y0, y1 in ((*_ROI_STRIP, i0 - 0.5, i1 + 0.5), (i0 - 0.5, i1 + 0.5, *_ROI_STRIP)):
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref=xref, yref=yref,
                          fillcolor=colours[label], line=dict(width=0))
    fig.add_shape(type="rect", x0=-0.5, x1=n - 0.5, y0=-0.5, y1=n - 0.5,
                  xref=xref, yref=yref, line=rule)
    edge = (_ROI_STRIP[0] if groups else -0.5) - 0.1
    ticks = dict(type="category", tickfont=dict(size=7), showgrid=False, zeroline=False,
                 ticks="")
    # `constrain="domain"` on **both** axes, or the one without it pads its range instead of
    # shrinking, the square ends up centred in a wider cell, and that axis's tick labels stay
    # out at the cell's edge with a gap between them and the matrix
    fig.update_xaxes(range=[edge, n - 0.5], tickangle=-90, constrain="domain",
                     **ticks, row=1, col=col)
    fig.update_yaxes(range=[n - 0.5, edge], scaleanchor=xref, scaleratio=1,
                     constrain="domain", **ticks, row=1, col=col)


def _panel_strip(fig, values, order, xs, measure, label, row, col, legend):
    """One measure's row: a marker per chromophore per pair, a pair's markers joined."""
    def value(pair, chromo):
        got = values.get(pair, {}).get(chromo)
        return float(got[measure]) if got is not None and measure in got else np.nan

    chromas = [c for c in CHROMA_STYLE if any(c in v for v in values.values())]
    seg_x, seg_y = [], []
    for xi, p in zip(xs, order):
        ys = [y for y in (value(p, c) for c in chromas) if np.isfinite(y)]
        if len(ys) > 1:
            seg_x += [xi, xi, None]
            seg_y += [min(ys), max(ys), None]
    fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", hoverinfo="skip",
                             line=dict(color=_CONNECTOR, width=1.5), showlegend=False),
                  row=row, col=col)

    drawn = np.zeros(len(order), dtype=bool)
    seen = []
    for chromo in chromas:
        name, colour = CHROMA_STYLE[chromo]
        y = np.array([value(p, chromo) for p in order])
        ok = np.isfinite(y)
        drawn |= ok
        seen.append(y[ok])
        fig.add_trace(go.Scatter(
            x=xs[ok], y=y[ok], mode="markers", name=name, showlegend=legend,
            customdata=[order[i] for i in np.flatnonzero(ok)],
            marker=dict(size=_STRIP_MARK, color=colour, line=dict(width=0.8, color="white")),
            hovertemplate="%{customdata}<br>" + label + " = %{y:.3g}<extra></extra>",
        ), row=row, col=col)
        if ok.any():
            # the run's own channel mean, the reference a marker is read against
            fig.add_hline(y=float(np.mean(y[ok])),
                          line=dict(color=colour, width=1, dash="dash"),
                          opacity=0.5, row=row, col=col)

    measured = np.concatenate(seen) if any(len(v) for v in seen) else np.zeros(1)
    lo, hi = float(np.min(measured)), float(np.max(measured))
    span = (hi - lo) or abs(hi) or 1.0

    if (~drawn).any():
        # a channel with no value keeps its position, below the measured range so it never
        # sits on a real near-zero value; "no value", since the panel cannot tell why
        idx = np.flatnonzero(~drawn)
        fig.add_trace(go.Scatter(
            x=xs[idx], y=np.full(len(idx), lo - 0.17 * span), mode="markers",
            name="no value", showlegend=legend, customdata=[order[i] for i in idx],
            marker=dict(size=_STRIP_MARK, symbol="circle-open", color=_REJECTED_INK,
                        line=dict(color=_REJECTED_INK, width=1.4)),
            hovertemplate="%{customdata}<br>no value<extra></extra>",
        ), row=row, col=col)

    fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=9),
                     showgrid=True, gridcolor="#f2f2f2", zeroline=False,
                     range=[lo - 0.28 * span, hi + 0.12 * span],
                     row=row, col=col)


def rest_channel_panel(
    fc: "dict[str, pd.DataFrame]",
    alff_df=None,
    raw=None,
    sep_bands=None,
    title: str = "Resting state per channel: connectivity, amplitude, spectral share",
    roi_map: "dict | None" = None,
):
    """The run's channels in one panel: FC on top, ALFF and fALFF below.

    Built in the shape of the HbO-HbR correlation panel: matrices side by side on one colour
    scale, and a per-pair strip under them. Here the matrices are the chromophores in ``fc``
    ({chromophore: channel matrix}) rather than two stages, and the strips carry amplitude
    rather than a correlation.

    The matrices are drawn whole, diagonal blank, on a scale symmetric about 0 that reaches
    the largest |r| any of them holds, so a run whose correlations are all positive still
    spans the red half.

    **The rows share one channel order and one set of blocks**, so a channel is in the same
    relative place in all of them: by ROI when ``roi_map`` ({ROI: [channels]}) is given, with
    a rule between ROIs and a colour strip beside each matrix, and by separation otherwise.
    They do not share an x *scale*: the matrices are narrower than the strips, so a column
    and a position correspond as labels and as blocks, not pixel for pixel.

    ``alff_df`` may be None, which is a run that computed no amplitude, and the panel is then
    the matrices alone. None overall when there is no FC to draw. A channel with no
    correlation in any matrix is left off every row (:func:`kept_axes`).
    """
    frames = {c: fc[c] for c in CHROMA_STYLE
              if fc.get(c) is not None and not fc[c].empty}
    if not frames:
        return None

    pairs = sorted({pair_of(c) for f in frames.values() for c in f.columns}, key=pair_order)
    measured = []
    for chromo, frame in frames.items():
        names = [f"{p} {chromo}" for p in pairs]
        mat = np.array(frame.reindex(index=names, columns=names).to_numpy(dtype=float),
                       copy=True)
        np.fill_diagonal(mat, np.nan)
        measured.append(kept_axes([mat])[0])
    pairs = [p for p, kept in zip(pairs, np.any(measured, axis=0)) if kept]
    if roi_map:
        order, xs, spans, groups = _roi_blocks(pairs, roi_map)
        colours = roi_colours(roi_map)
    else:
        (order, xs, spans), groups, colours = _pair_blocks(pairs, raw, sep_bands), [], {}
    if not order:
        return None
    lim = 0.0
    for chromo, frame in frames.items():
        names = [f"{p} {chromo}" for p in order]
        mat = frame.reindex(index=names, columns=names).to_numpy(dtype=float)
        off = mat[~np.eye(len(names), dtype=bool)]
        if np.isfinite(off).any():
            lim = max(lim, float(np.nanmax(np.abs(off))))
    lim = lim or 1.0

    values = {}
    if alff_df is not None:
        for _, row in alff_df.iterrows():
            name = str(row["channel"])
            if " " in name:
                pair, chromo = name.rsplit(" ", 1)
                values.setdefault(pair, {})[chromo] = row

    measures = [(m, lab) for m, lab in (("alff", "ALFF (M)"), ("falff", "fALFF"))
                if alff_df is not None and m in alff_df.columns]
    # the strips sit side by side under the matrices, so the panel is one grid and not a
    # matrix block with a tail
    n_rows = 2 if measures else 1
    n_cols = max(2, len(frames))
    height = _PANEL_MATRIX_PX + (_PANEL_STRIP_PX if measures else 0) + 150
    heights = [_PANEL_MATRIX_PX] + ([_PANEL_STRIP_PX] if measures else [])
    heights = [h / sum(heights) for h in heights]
    titles = [CHROMA_STYLE[c][0] for c in frames] + [""] * (n_cols - len(frames))
    fig = make_subplots(
        rows=n_rows, cols=n_cols, row_heights=heights, vertical_spacing=0.10,
        horizontal_spacing=0.09,
        subplot_titles=titles + [lab for _, lab in measures])

    for col, (chromo, frame) in enumerate(frames.items(), start=1):
        _panel_matrix(fig, frame, order, chromo, col, (col == len(frames)), lim, groups,
                      colours)
    for label, _i0, _i1 in groups:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                 marker=dict(symbol="square", size=11, color=colours[label]),
                                 hoverinfo="skip"))

    for col, (measure, label) in enumerate(measures, start=1):
        _panel_strip(fig, values, order, xs, measure, label, 2, col, legend=(col == 1))
        fig.update_xaxes(tickmode="array", tickvals=xs, ticktext=order, tickangle=-90,
                         tickfont=dict(size=7), showgrid=False, zeroline=False,
                         row=2, col=col)

    if measures and len(spans) > 1:
        # inside the strip, not above it: the subplot title is up there naming the measure,
        # and a heading at the same height reads as part of it
        for col in range(1, len(measures) + 1):
            for xa, xb, name in spans:
                fig.add_annotation(x=(xa + xb) / 2, y=0.99, yref="y domain", yanchor="top",
                                   text=name, showarrow=False,
                                   font=dict(color="#9aa5af", size=8), row=2, col=col)

    # a cell neither a matrix nor a strip fills is left blank rather than drawn as empty axes
    for row, filled in ((1, len(frames)), (2, len(measures) if measures else n_cols)):
        for col in range(filled + 1, n_cols + 1):
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

    names = {label for label, _ in CHROMA_STYLE.values()} | {lab for _, lab in measures}
    for note in fig.layout.annotations:
        if note.text in names:
            note.font = dict(size=12, color="#34495e")
    fig.update_layout(
        height=height, plot_bgcolor="white", margin=dict(l=76, r=30, t=118, b=104),
        # clear of the matrix titles under it
        legend=dict(orientation="h", yanchor="bottom", y=1.045, x=0.0, font=dict(size=10)),
        title=dict(text=title, x=0.01, y=0.985, font=dict(size=13)))
    return fig


# An ROI matrix is a handful of cells, so it is sized to be read rather than scanned.
_ROI_CELL_PX, _ROI_MIN_PX, _ROI_MAX_PX = 74, 240, 520


def fc_roi_matrix_figure(
    fc_roi: "dict[str, pd.DataFrame]",
    title: str = "ROI-to-ROI Functional Connectivity (Pearson r)",
) -> "go.Figure | None":
    """The ROI x ROI FC heatmaps, or None if there is nothing to draw.

    ``fc_roi`` is {chromophore: ROI x ROI frame}, as :func:`compute_fc_roi` returns it. The
    channel matrices' rule in :func:`rest_channel_panel`: symmetric about 0, out to the
    largest |r| either panel holds. A handful of ROIs means every label fits, so
    unlike those this one labels and prints every cell. The diagonal is blanked: an ROI's
    correlation with itself is 1 by construction and says nothing. An ROI with no value is
    left off both axes (:func:`kept_axes`).
    """
    panels = []
    for chromo, (label, _) in CHROMA_STYLE.items():
        frame = fc_roi.get(chromo)
        if frame is None or frame.empty:
            continue
        mat = frame.to_numpy(dtype=float).copy()
        np.fill_diagonal(mat, np.nan)
        keep = kept_axes([mat])[0]
        if not keep.any():
            continue
        mat = mat[np.ix_(keep, keep)]
        # symmetric, as the channel matrices are, so the same half is drawn
        mat[np.triu_indices(len(mat), k=1)] = np.nan
        panels.append(([n for n, k in zip(frame.index.tolist(), keep) if k], mat,
                       f"ROI FC - {label}"))
    if not panels:
        return None
    lim = max((float(np.nanmax(np.abs(m))) for _, m, _ in panels if np.isfinite(m).any()),
              default=0.0) or 1.0

    side = float(np.clip(max(len(n) for n, _, _ in panels) * _ROI_CELL_PX,
                         _ROI_MIN_PX, _ROI_MAX_PX))
    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.14,
                        subplot_titles=[lab for _, _, lab in panels])
    for col, (names, mat, _label) in enumerate(panels, start=1):
        fig.add_trace(go.Heatmap(
            z=mat.astype(np.float32), x=names, y=names, zmin=-lim, zmax=lim,
            colorscale=CORRELATION_SCALE, showscale=(col == len(panels)),
            colorbar=dict(title=dict(text="Pearson r", side="right", font=dict(size=10)),
                          thickness=12, len=0.8, tickfont=dict(size=9)),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
        ), row=1, col=col)
        cell_values(fig, mat, names, names, cmap=CORRELATION_SCALE, vmin=-lim, vmax=lim,
                    row=1, col=col)
        axes = dict(tickfont=dict(size=9), showgrid=False, zeroline=False, ticks="",
                    showline=False, constrain="domain")
        # square cells, and `constrain` on both axes shrinks whichever is not binding rather
        # than padding its range, so the labels stay against the matrix either way
        fig.update_xaxes(tickangle=-45, **axes, row=1, col=col)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{col if col > 1 else ''}",
                         scaleratio=1, **axes, row=1, col=col)

    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=side + 150, plot_bgcolor="white", showlegend=False,
                      margin=dict(l=90, r=40, t=62, b=90),
                      title=dict(text=title, x=0.01, font=dict(size=13)))
    return fig


def _head_for(raw: mne.io.Raw, sep_bands) -> "dict | None":
    """The long-channel flat head this run's maps are drawn on, or None with no positions.

    Short channels are left out of every map in this module: they measure extracerebral
    signal.
    """
    from nirspipe.qc.metrics import long_short_channels

    long_names, _ = long_short_channels(raw, sep_bands)
    pairs = sorted({pair_of(ch) for ch in (long_names or raw.ch_names)})
    geo = head_geometry(raw, pairs)
    if geo is None or "long" not in geo:
        return None
    return geo


def _values_for(geo: dict, lookup, chromo: str) -> np.ndarray:
    """One value per pair the head draws, NaN where ``lookup`` has nothing for it."""
    return np.array([lookup(f"{name} {chromo}") for name in geo["long"]["names"]], dtype=float)


# the gap between rows of heads, wide enough to hold a horizontal colour bar and its ticks
_ROW_GAP_PX = 66
# tall enough that a head, which the grid anchors square, is still legible at report width
_HEAD_PX = 250
_MARGIN_T, _MARGIN_B = 62, 54


# in pixels rather than as a share of the figure, so a head keeps its size however many rows
def _grid_px(n_rows: int) -> int:
    return n_rows * _HEAD_PX + (n_rows - 1) * _ROW_GAP_PX


def _row_gap(n_rows: int) -> float:
    return _ROW_GAP_PX / _grid_px(n_rows) if n_rows > 1 else 0.0


_COL_GAP = 0.02


def _head_grid(n_rows: int, n_cols: int, titles: list[str], geo: dict, empty=()):
    """An empty grid of heads with the outline and skeleton under each panel but ``empty``."""
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=_COL_GAP, vertical_spacing=_row_gap(n_rows))
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            if (r, c) not in empty:
                head_ground(fig, geo, r, c)
    return fig


def _finish_head_grid(fig, geo, n_rows, n_cols, title):
    """Square axes, the run's title, and room under the grid for the bars."""
    head_axes(fig, {"run": geo}, n_rows, n_cols)
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=_grid_px(n_rows) + _MARGIN_T + _MARGIN_B, plot_bgcolor="white",
                      showlegend=False, margin=dict(l=40, r=40, t=_MARGIN_T, b=_MARGIN_B),
                      title=dict(text=title, x=0.01, font=dict(size=13)))
    return fig


def _column_centre(c: int, n_cols: int) -> float:
    width = (1 - (n_cols - 1) * _COL_GAP) / n_cols
    return (c - 1) * (width + _COL_GAP) + width / 2


def flat_value_maps(raw: mne.io.Raw, rows: "list[dict]", title: str, colorscale,
                    sep_bands=None) -> "go.Figure | None":
    """Channel values on the optode flat map, a row of heads per entry, or None without positions.

    Each row is ``{"panels": [(title, chromophore, {channel: value}), ...], "scale": (vmin,
    vmax) or None, "bar": label}``: a scale is shared by the row under one bar, None gives
    every head its own range and bar. A channel with no value is grey, a rejected one faded.
    """
    geo = _head_for(raw, sep_bands)
    if geo is None or not rows:
        return None
    n_rows, n_cols = len(rows), max(len(row["panels"]) for row in rows)
    titles = [row["panels"][c][0] if c < len(row["panels"]) else ""
              for row in rows for c in range(n_cols)]
    empty = {(r, c) for r, row in enumerate(rows, start=1)
             for c in range(len(row["panels"]) + 1, n_cols + 1)}
    fig = _head_grid(n_rows, n_cols, titles, geo, empty=empty)
    dim = {pair_of(ch) for ch in raw.info["bads"]}
    total = _grid_px(n_rows)
    for r, row in enumerate(rows, start=1):
        # in the gap under its row, which is sized for a horizontal bar
        bar_y = 1 - (r * _HEAD_PX + (r - 1) * _ROW_GAP_PX + 10) / total
        for c, (_, chromo, by_channel) in enumerate(row["panels"], start=1):
            values = _values_for(geo, lambda ch, d=by_channel: d.get(ch, np.nan), chromo)
            shown = values[np.isfinite(values)]
            vmin, vmax = row["scale"] or ((float(shown.min()), float(shown.max()))
                                          if shown.size else (0.0, 1.0))
            own_bar = row["scale"] is None
            bar = ({"orientation": "h", "len": 0.8 / n_cols if own_bar else 0.28,
                    "thickness": 9, "x": _column_centre(c, n_cols) if own_bar else 0.46,
                    "xanchor": "center", "y": bar_y, "yanchor": "top",
                    "tickfont": {"size": 9}, "title": {"side": "right"}}
                   if own_bar or c == 1 else False)
            head_glyph(fig, geo, "long", values, r, c, row["bar"], vmin, vmax, bar=bar,
                       colorscale=colorscale, dim=dim, blank_color=BLANK_COLOR)
    return _finish_head_grid(fig, geo, n_rows, n_cols, title)


def alff_flat_figure(raw: mne.io.Raw, alff_df: pd.DataFrame, sep_bands=None) -> "go.Figure | None":
    """mALFF on the optode flat map, a head per chromophore on its own range: the stand-in for :func:`alff_brain_figure` when the optodes could not be placed."""
    measure = "malff" if "malff" in alff_df.columns else "alff"
    values = alff_df.set_index("channel")[measure].astype(float)
    bads = set(raw.info["bads"])
    panels = []
    for chroma in (c for c in CHROMA_STYLE if c in {chroma_of(ch) for ch in values.index}):
        kept = {ch: float(v) for ch, v in values.items()
                if chroma_of(ch) == chroma and ch not in bads and np.isfinite(v)}
        if kept:
            panels.append((CHROMA_STYLE[chroma][0], chroma, kept))
    if not panels:
        return None
    label = "mALFF (1 = mean channel)" if measure == "malff" else "ALFF (M)"
    return flat_value_maps(raw, [{"panels": panels, "scale": None, "bar": label}],
                           f"{label} on the optode flat map", "Viridis", sep_bands)


def short_share_figure(raw: mne.io.Raw, shares: "dict[str, float]",
                       sep_bands=None) -> "go.Figure | None":
    """Each long channel's variance the short-channel regressors carried, a head per chromophore on one 0 to 1 scale."""
    panels = [(CHROMA_STYLE[c][0], c, {ch: v for ch, v in shares.items() if chroma_of(ch) == c})
              for c in CHROMA_STYLE if any(chroma_of(ch) == c for ch in shares)]
    if not panels:
        return None
    return flat_value_maps(raw, [{"panels": panels, "scale": (0.0, 1.0), "bar": "Share"}],
                           "Variance carried by the short-channel regressors", "Viridis",
                           sep_bands)


# seeds per row before the grid wraps: four heads still fit the report's width at full size
_SEED_COLS = 4


def fc_seed_topo_figure(
    raw: mne.io.Raw,
    seeds: "dict[str, pd.DataFrame]",
    title: str = "Seed-to-whole-brain connectivity (Pearson r)",
    sep_bands=None,
) -> "go.Figure | None":
    """One flat map per seed ROI and chromophore, or None if the montage has no positions.

    compute_fc_seed returns an ROI x channel frame per chromophore, and ``seeds`` holds them
    as {chromophore: frame}. Each channel is a bar of discs along its source-to-detector path,
    coloured by that seed's correlation with it, on the same reversed RdBu / +-1 scale
    fc_matrix_figure uses so the two figures can be read against each other.

    Three states are distinguishable:

    - an ordinary channel, coloured by r;
    - a channel with no value, grey. That is a channel **inside the seed**, whose correlation
      is inflated by construction, and grey says "no claim made here" where a blue channel
      would say "no connection";
    - a **rejected** channel, drawn faded. Whether it also has a value depends on the frame;
      either way the fading says the channel was excluded upstream.
    """
    panels = [(seeds[c], c) for c in CHROMA_STYLE
              if seeds.get(c) is not None and not seeds[c].empty]
    if not panels:
        return None
    geo = _head_for(raw, sep_bands)
    if geo is None:
        logger.warning("seed topography skipped: montage carries no optode positions")
        return None

    rois = list(dict.fromkeys([r for frame, _ in panels for r in frame.index]))
    dim = {pair_of(ch) for ch in raw.info["bads"]}
    # a seed per column, a chromophore per row; past _SEED_COLS seeds the next ones wrap
    # into another row per chromophore, so a seed's maps always sit one above the other
    n_cols = min(len(rois), _SEED_COLS)
    chunks = [rois[k:k + n_cols] for k in range(0, len(rois), n_cols)]
    cells = {}
    for k, chunk in enumerate(chunks):
        for p, (frame, chromo) in enumerate(panels):
            for c, roi in enumerate(chunk, start=1):
                cells[(k * len(panels) + p + 1, c)] = (roi, frame, chromo)
    n_rows = len(chunks) * len(panels)
    grid = [(r, c) for r in range(1, n_rows + 1) for c in range(1, n_cols + 1)]
    titles = [f"{cells[rc][0]} - {CHROMA_STYLE[cells[rc][2]][0]}" if rc in cells else ""
              for rc in grid]
    fig = _head_grid(n_rows, n_cols, titles, geo, empty={rc for rc in grid if rc not in cells})

    first = True
    for (i, j), (roi, frame, chromo) in cells.items():
        if roi not in frame.index:
            continue
        row = frame.loc[roi]
        values = _values_for(geo, lambda ch, r=row: float(r[ch]) if ch in r.index
                             else np.nan, chromo)
        # one bar for the whole figure: every panel is on the same fixed +-1 scale, so
        # a bar per row would be the same bar drawn again
        head_glyph(fig, geo, "long", values, i, j, "Pearson r", -1.0, 1.0,
                   bar={"orientation": "h", "len": 0.28, "thickness": 9,
                        "x": 0.46, "xanchor": "center", "y": -10 / _grid_px(n_rows),
                        "yanchor": "top",
                        "tickfont": {"size": 9}, "title": {"side": "right"}}
                       if first else False,
                   colorscale=CORRELATION_SCALE, dim=dim, blank_color=BLANK_COLOR)
        first = False
    return _finish_head_grid(fig, geo, n_rows, n_cols, title)


# (title, azimuth, elevation): both sides, the top and the front. An fMRI surface plot's
# medial views are left out, since no optode reaches the medial cortex.
_BRAIN_VIEWS = (("Left", 180, 90), ("Right", 0, 90), ("Superior", 0, 0), ("Frontal", 90, 90))
# about a view's panel width at 300 dpi, so the render is not upsampled
_BRAIN_VIEW_PX = (1400, 1200)


def alff_brain_figure(raw: mne.io.Raw, alff_df: pd.DataFrame, sep_bands=None) -> "str | None":
    """mALFF on the template's cortex in four views, a block per chromophore the table holds, as a base64 PNG.

    The long channels that were kept and carry a value are projected as the GLM activation
    is, and each block's scale runs from its lowest value to its highest. None when no
    channel has one.
    """
    from nirspipe.qc.metrics import long_short_channels

    measure = "malff" if "malff" in alff_df.columns else "alff"
    values = alff_df.set_index("channel")[measure].astype(float)
    long_names, _ = long_short_channels(raw, sep_bands)
    long_pairs = {pair_of(ch) for ch in (long_names or raw.ch_names)}
    held = {chroma_of(ch) for ch in values.index}
    blocks = []
    for chroma in (c for c in CHROMA_STYLE if c in held):
        drawn = chroma_channels(raw, chroma)
        bads = set(drawn.info["bads"])
        names = [ch for ch in drawn.ch_names if ch not in bads and pair_of(ch) in long_pairs
                 and np.isfinite(values.get(ch, np.nan))]
        if names:
            blocks.append(_alff_brain_block(drawn.pick(names), values.loc[names].to_numpy(),
                                            measure, CHROMA_STYLE[chroma][0]))
    return stack_png_b64(blocks) if blocks else None


def _alff_brain_block(drawn: mne.io.Raw, vals: np.ndarray, measure: str, label: str) -> str:
    vmin, vmax = float(vals.min()), float(vals.max())
    # a column of ones alongside, so each vertex knows how much of it the channels cover
    stc, subjects_dir = project_to_cortex(drawn, np.column_stack([vals, np.ones_like(vals)]))
    with RENDER_LOCK:
        images = render_cortex(
            stc, lambda curv, s: coverage_rgba(curv, s[:, 0], s[:, 1], vmin, vmax),
            subjects_dir, _BRAIN_VIEW_PX,
            [{"azimuth": az, "elevation": el} for _, az, el in _BRAIN_VIEWS])

    fig = plt.figure(figsize=(9.2, 8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.02, hspace=0.08)
    for k, ((view, _, _), img) in enumerate(zip(_BRAIN_VIEWS, images)):
        ax = fig.add_subplot(grid[k // 2, k % 2])
        ax.imshow(img)
        ax.set_title(view, fontsize=11)
        ax.axis("off")
    bar = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap="viridis"),
                       cax=fig.add_subplot(grid[:, 2]))
    bar.set_label(f"mALFF (1 = mean channel), {label}" if measure == "malff"
                  else f"ALFF (M), {label}", fontsize=11)
    bar.outline.set_linewidth(0.5)
    return fig_png_b64(fig, pad_inches=0.1)
