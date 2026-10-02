"""Resting-state QC figures.

rest_channel_panel():   the run's channels in one panel, FC matrices over ALFF and fALFF.
alff_topo_figure():     amplitude and spectral share drawn on the optode flat map.
fc_roi_matrix_figure(): connectivity ROI by ROI instead of channel by channel.
fc_seed_topo_figure():  seed-to-whole-brain correlations drawn on the optode flat map.

The panel answers how much and with whom, the flat maps answer where. Both flat maps are
built on the shared head in :mod:`fnirs_pipe.qc.figures.common.head_map` rather than on their
own projection, so a channel sits where the report's other head figures put it.

# TODO: project ALFF/fALFF onto a brain surface (not just the flat map) via mne_nirs when
# head coordinates are available.
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.utils import pair_of
from fnirs_pipe.utils.logging import get_logger

from fnirs_pipe.qc.figures.common._utils import HBO_COLOR, HBR_COLOR
from fnirs_pipe.qc.figures.common.head_map import (
    BLANK_COLOR, head_axes, head_geometry, head_glyph, head_ground,
)
from fnirs_pipe.qc.figures.common.matrix_map import (
    CORRELATION_SCALE, cell_values, matrix_ground,
)

logger = get_logger("qc.figures.rest")

_HBO_COLOR = HBO_COLOR
_HBR_COLOR = HBR_COLOR
_MEAN_LINE_COLOR = "#555555"

# an unsigned magnitude, so one hue ramped rather than a diverging pair
_ALFF_SCALE = "Viridis"


# The blocks a rest panel's channels are laid out in, and what each is called. Same split and
# same order as the per-channel table, so a reader moving between them finds the same groups.
_SEP_BLOCKS = (("long", "long"), ("mid", "neither range"), ("short", "short"))
# Blank x positions between blocks, so the grouping needs no rule drawn through the panel.
_BLOCK_GAP = 1.4
_STRIP_MARK, _CONNECTOR, _REJECTED_INK = 8, "#c3cad1", "#b9c2cb"
# The panel's two bands, in pixels: the matrices are square and need the room, a strip does
# not, and a strip added later must not shrink the matrices.
_PANEL_MATRIX_PX, _PANEL_STRIP_PX = 560, 230


def _pair_blocks(pairs, raw, sep_bands):
    """The panel's channel order: ``(pairs, x positions, block spans)``.

    Grouped by separation and sorted by name inside a group, which is the per-channel
    table's order.

    Without ``raw`` there is no separation to block by and the pairs are one list, which is
    what a caller holding only the tables can draw.
    """
    groups = {}
    if raw is not None:
        from fnirs_pipe.qc.metrics import long_short_channels
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
        members = sorted(groups.get(key, []))
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


def _panel_matrix(fig, frame, order, chromo, col, show_bar):
    """One chromophore's channel matrix into the panel's top row, on the panel's order."""
    names = [f"{p} {chromo}" for p in order]
    # to_numpy can hand back a read-only view, and the matrix is blanked in place
    mat = np.array(frame.reindex(index=names, columns=names).to_numpy(dtype=float), copy=True)
    np.fill_diagonal(mat, np.nan)
    # Lower triangle only, as the HbO-HbR correlation panel draws it: the matrix is symmetric
    mat[np.triu_indices(len(order), k=1)] = np.nan
    matrix_ground(fig, len(order), len(order), 1, col, triangle=True)
    fig.add_trace(go.Heatmap(
        z=mat.astype(np.float32), x=order, y=order, zmin=-1.0, zmax=1.0,
        colorscale=CORRELATION_SCALE, showscale=show_bar,
        colorbar=dict(title=dict(text="Pearson r", side="right", font=dict(size=10)),
                      thickness=12, len=0.46, y=1.0, yanchor="top",
                      tickfont=dict(size=9), tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
    ), row=1, col=col)
    # `constrain="domain"` on **both** axes, or the one without it pads its range instead of
    # shrinking, the square ends up centred in a wider cell, and that axis's tick labels stay
    # out at the cell's edge with a gap between them and the matrix
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=7), showgrid=False, ticks="",
                     constrain="domain", row=1, col=col)
    fig.update_yaxes(autorange="reversed", scaleanchor=f"x{col if col > 1 else ''}",
                     scaleratio=1, constrain="domain", tickfont=dict(size=7),
                     showgrid=False, ticks="", row=1, col=col)


def _panel_strip(fig, values, order, xs, measure, label, row, col, legend):
    """One measure's row: a marker per chromophore per pair, the two joined."""
    def value(pair, chromo):
        got = values.get(pair, {}).get(chromo)
        return float(got[measure]) if got is not None and measure in got else np.nan

    seg_x, seg_y = [], []
    for xi, p in zip(xs, order):
        a, b = value(p, "hbo"), value(p, "hbr")
        if np.isfinite(a) and np.isfinite(b):
            seg_x += [xi, xi, None]
            seg_y += [a, b, None]
    fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode="lines", hoverinfo="skip",
                             line=dict(color=_CONNECTOR, width=1.5), showlegend=False),
                  row=row, col=col)

    drawn = np.zeros(len(order), dtype=bool)
    seen = []
    for chromo, colour, name in (("hbo", _HBO_COLOR, "HbO"), ("hbr", _HBR_COLOR, "HbR")):
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
    fc_df,
    fc_hbr_df=None,
    alff_df=None,
    raw=None,
    sep_bands=None,
    title: str = "Resting state per channel: connectivity, amplitude, spectral share",
):
    """The run's channels in one panel: FC on top, ALFF and fALFF below.

    Built in the shape of the HbO-HbR correlation panel: two matrices side by side on one
    colour scale, and a per-pair strip under them. Here the two matrices are the two
    chromophores rather than two stages, and the strips carry amplitude rather than a
    correlation.

    **The rows share one channel order and one set of separation blocks**, so a channel is in
    the same relative place in all of them. They do not share an x *scale*: the matrices are
    half-width and the strips full-width, so a column and a position correspond as labels and
    as blocks, not pixel for pixel.

    ``alff_df`` may be None, which is a run that computed no amplitude, and the panel is then
    the matrices alone. None overall when there is no FC to draw.
    """
    frames = {c: f for c, f in (("hbo", fc_df), ("hbr", fc_hbr_df))
              if f is not None and not f.empty}
    if not frames:
        return None

    pairs = sorted({pair_of(c) for f in frames.values() for c in f.columns})
    order, xs, spans = _pair_blocks(pairs, raw, sep_bands)
    if not order:
        return None

    values = {}
    if alff_df is not None:
        for _, row in alff_df.iterrows():
            name = str(row["channel"])
            if " " in name:
                pair, chromo = name.rsplit(" ", 1)
                values.setdefault(pair, {})[chromo] = row

    measures = [(m, lab) for m, lab in (("alff", "ALFF (M)"), ("falff", "fALFF"))
                if alff_df is not None and m in alff_df.columns]
    # the two strips sit side by side under the two matrices, so the panel is one grid and
    # not a matrix block with a tail
    n_rows = 2 if measures else 1
    height = _PANEL_MATRIX_PX + (_PANEL_STRIP_PX if measures else 0) + 150
    heights = [_PANEL_MATRIX_PX] + ([_PANEL_STRIP_PX] if measures else [])
    heights = [h / sum(heights) for h in heights]
    fig = make_subplots(
        rows=n_rows, cols=2, row_heights=heights, vertical_spacing=0.10,
        horizontal_spacing=0.09,
        subplot_titles=["HbO", "HbR"] + [lab for _, lab in measures])

    for col, chromo in enumerate(("hbo", "hbr"), start=1):
        frame = frames.get(chromo)
        if frame is not None:
            _panel_matrix(fig, frame, order, chromo, col, show_bar=(col == len(frames)))

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

    for note in fig.layout.annotations:
        if note.text in ("HbO", "HbR") or note.text in {lab for _, lab in measures}:
            note.font = dict(size=12, color="#34495e")
    fig.update_layout(
        height=height, plot_bgcolor="white", margin=dict(l=76, r=30, t=96, b=104),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, x=0.0, font=dict(size=10)),
        title=dict(text=title, x=0.01, y=0.985, font=dict(size=13)))
    return fig


# An ROI matrix is a handful of cells, so it is sized to be read rather than scanned.
_ROI_CELL_PX, _ROI_MIN_PX, _ROI_MAX_PX = 74, 240, 520


def fc_roi_matrix_figure(
    fc_roi: "dict[str, pd.DataFrame]",
    title: str = "ROI-to-ROI Functional Connectivity (Pearson r)",
) -> "go.Figure | None":
    """The ROI x ROI FC heatmaps, or None if there is nothing to draw.

    ``fc_roi`` is {chromophore: ROI x ROI frame}, as :func:`compute_fc_roi` returns it. Same
    scale as the channel matrices in :func:`rest_channel_panel`, so the ROI view and the
    channel view can be read against each other. A handful of ROIs means every label fits, so
    unlike those this one labels and prints every cell. The diagonal is blanked: an ROI's
    correlation with itself is 1 by construction and says nothing.
    """
    panels = []
    for chromo, label in (("hbo", "HbO"), ("hbr", "HbR")):
        frame = fc_roi.get(chromo)
        if frame is None or frame.empty:
            continue
        mat = frame.to_numpy(dtype=float).copy()
        np.fill_diagonal(mat, np.nan)
        # symmetric, as the channel matrices are, so the same half is drawn
        mat[np.triu_indices(len(mat), k=1)] = np.nan
        panels.append((frame.index.tolist(), mat, f"ROI FC - {label}"))
    if not panels:
        return None

    side = float(np.clip(max(len(n) for n, _, _ in panels) * _ROI_CELL_PX,
                         _ROI_MIN_PX, _ROI_MAX_PX))
    fig = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.14,
                        subplot_titles=[lab for _, _, lab in panels])
    for col, (names, mat, _label) in enumerate(panels, start=1):
        matrix_ground(fig, len(names), len(names), 1, col, triangle=True)
        fig.add_trace(go.Heatmap(
            z=mat.astype(np.float32), x=names, y=names, zmin=-1.0, zmax=1.0,
            colorscale=CORRELATION_SCALE, showscale=(col == len(panels)),
            colorbar=dict(title=dict(text="Pearson r", side="right", font=dict(size=10)),
                          thickness=12, len=0.8, tickfont=dict(size=9),
                          tickvals=[-1, -0.5, 0, 0.5, 1]),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.3f}<extra></extra>",
        ), row=1, col=col)
        cell_values(fig, mat, names, names, cmap=CORRELATION_SCALE, vmin=-1.0, vmax=1.0,
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
    from fnirs_pipe.qc.metrics import long_short_channels

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
_ROW_GAP = 0.14
# tall enough that a head, which the grid anchors square, is still legible at report width
_HEAD_PX = 250


def _head_grid(n_rows: int, n_cols: int, titles: list[str], geo: dict):
    """An empty grid of heads with the outline and skeleton already under each panel."""
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.02, vertical_spacing=_ROW_GAP)
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            head_ground(fig, geo, r, c)
    return fig


def _finish_head_grid(fig, geo, n_rows, n_cols, title):
    """Square axes, the run's title, and room under the grid for the bars."""
    head_axes(fig, {"run": geo}, n_rows, n_cols)
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=_HEAD_PX * n_rows + 90, plot_bgcolor="white",
                      showlegend=False, margin=dict(l=40, r=40, t=62, b=54),
                      title=dict(text=title, x=0.01, font=dict(size=13)))
    return fig


def fc_seed_topo_figure(
    raw: mne.io.Raw,
    seed_df: pd.DataFrame,
    seed_hbr_df: pd.DataFrame | None = None,
    title: str = "Seed-to-whole-brain connectivity (Pearson r)",
    sep_bands=None,
) -> "go.Figure | None":
    """One flat map per seed ROI, or None if the montage has no positions.

    compute_fc_seed returns an ROI x channel frame per chromophore, so HbO arrives in seed_df
    and HbR in seed_hbr_df. Each channel is a bar of discs along its source-to-detector path,
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
    panels = [(f, lab) for f, lab in ((seed_df, "HbO"), (seed_hbr_df, "HbR"))
              if f is not None and not f.empty]
    if not panels:
        return None
    geo = _head_for(raw, sep_bands)
    if geo is None:
        logger.warning("seed topography skipped: montage carries no optode positions")
        return None

    rois = list(dict.fromkeys([r for frame, _ in panels for r in frame.index]))
    dim = {pair_of(ch) for ch in raw.info["bads"]}
    n_rows, n_cols = len(rois), len(panels)
    titles = [f"{roi} - {lab}" for roi in rois for _, lab in panels]
    fig = _head_grid(n_rows, n_cols, titles, geo)

    for i, roi in enumerate(rois, start=1):
        for j, (frame, label) in enumerate(panels, start=1):
            if roi not in frame.index:
                continue
            row = frame.loc[roi]
            chromo = label.lower()
            values = _values_for(geo, lambda ch, r=row: float(r[ch]) if ch in r.index
                                 else np.nan, chromo)
            # one bar for the whole figure: every panel is on the same fixed +-1 scale, so
            # a bar per row would be the same bar drawn again
            head_glyph(fig, geo, "long", values, i, j, "Pearson r", -1.0, 1.0,
                       bar={"orientation": "h", "len": 0.28, "thickness": 9,
                            "x": 0.46, "xanchor": "center", "y": -0.05, "yanchor": "top",
                            "tickfont": {"size": 9}, "title": {"side": "right"}}
                           if (i == 1 and j == n_cols) else False,
                       colorscale=CORRELATION_SCALE, dim=dim, blank_color=BLANK_COLOR)
    return _finish_head_grid(fig, geo, n_rows, n_cols, title)


# What each row draws, best column first, and the measured column its hover names. mALFF
# is standardised within each chromophore, so HbO and HbR share one scale; the measured
# amplitude is one hover away.
_ALFF_ROWS = ((("malff", "alff"), "alff"), (("falff",), None))
_ALFF_LABELS = {"malff": "mALFF", "alff": "ALFF", "falff": "fALFF"}


def _alff_rows(alff_df: pd.DataFrame) -> "list[tuple[str, str, str | None]]":
    """(column, label, measured column) per row, dropping a row the frame cannot fill.

    ``alff`` stands in where a frame carries no ``malff``, which only a hand-built one does.
    Its hover names nothing extra, the colour already being the measured number.
    """
    out = []
    for candidates, measured in _ALFF_ROWS:
        col = next((c for c in candidates if c in alff_df.columns), None)
        if col:
            out.append((col, _ALFF_LABELS[col], measured if measured != col else None))
    return out


def _row_colorbar(row: int, n_rows: int) -> dict:
    """Colorbar placement for one row of a head grid: horizontal, under the row it describes.

    Not beside the row: the grid anchors each head square inside a much wider cell, so a
    vertical bar at a cell's right edge floats in the gap between two heads. The row gap is
    what the head circle leaves free.
    """
    cell = (1.0 - _ROW_GAP * (n_rows - 1)) / n_rows
    return {"orientation": "h", "len": 0.3, "thickness": 9,
            "x": 0.5, "xanchor": "center",
            "y": 1.0 - row * cell - (row - 1) * _ROW_GAP - 0.02, "yanchor": "top",
            # beside the bar, not above it: a title over a horizontal bar grows the block
            # upward and runs into the subplot title of the row below
            "tickfont": {"size": 9}, "title": {"side": "right"}}


def alff_topo_figure(
    raw: mne.io.Raw,
    alff_df: pd.DataFrame,
    title: str = "ALFF and fALFF on the optode layout",
    sep_bands=None,
) -> "go.Figure | None":
    """Low-frequency amplitude drawn on the flat map, or None with no positions.

    One disc per channel, at its source-detector midpoint, because the value is a property
    of a place rather than of a path.

    Both rows are dimensionless, so a row's two chromophores share one scale and one bar and
    can be read against each other. The rows keep their own: fALFF is a share of the whole
    spectrum and mALFF a multiple of the chromophore's own mean.

    A rejected channel is drawn grey rather than dropped, so the montage stays complete and
    the gap reads as a rejection, and its value is kept out of the scale. Grey alone, not
    grey and faded: for this figure a rejected channel is exactly the channel with no value.
    """
    geo = _head_for(raw, sep_bands)
    if geo is None:
        logger.warning("ALFF topography skipped: montage carries no optode positions")
        return None

    values = {str(row["channel"]): row for _, row in alff_df.iterrows()}
    cols = [(chromo, lab) for chromo, lab in (("hbo", "HbO"), ("hbr", "HbR"))
            if any(ch.endswith(f" {chromo}") for ch in values)]
    rows = _alff_rows(alff_df)
    if not cols or not rows:
        return None
    dim = {pair_of(ch) for ch in raw.info["bads"]}
    blank = [n in dim for n in geo["long"]["names"]]

    def column(measure: str, chromo: str) -> np.ndarray:
        vals = _values_for(
            geo, lambda ch, m=measure: float(values[ch][m]) if ch in values else np.nan,
            chromo)
        # compute_alff already blanks a rejected channel, but the frame is an argument and
        # may not have come from it; the recording's own bads are the authority, and a
        # rejected channel that kept its value would still set the scale below
        vals[blank] = np.nan
        return vals

    n_rows, n_cols = len(rows), len(cols)
    titles = [f"{label} - {lab}" for _, label, _ in rows for _, lab in cols]
    fig = _head_grid(n_rows, n_cols, titles, geo)

    for i, (measure, label, raw_key) in enumerate(rows, start=1):
        by_col = [column(measure, chromo) for chromo, _ in cols]
        # the scale comes from the channels that have a value: one rejected channel with a
        # runaway amplitude would flatten every real difference into one colour
        good = np.concatenate([v[np.isfinite(v)] for v in by_col])
        if not len(good):
            continue
        for j, ((chromo, _lab), vals) in enumerate(zip(cols, by_col), start=1):
            measured = column(raw_key, chromo) if raw_key else None
            head_glyph(fig, geo, "long", vals, i, j, label,
                       float(good.min()), float(good.max()),
                       bar=_row_colorbar(i, n_rows) if j == n_cols else False,
                       colorscale=_ALFF_SCALE, blank_color=BLANK_COLOR,
                       fmt=".2f", mark="disc",
                       customdata=measured,
                       hover_tail="" if measured is None else "<br>%{customdata:.3g} M")
    return _finish_head_grid(fig, geo, n_rows, n_cols, title)
