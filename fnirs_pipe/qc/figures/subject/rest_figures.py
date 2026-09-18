"""Resting-state QC figures.

alff_falff_figure():    per-channel ALFF and fALFF bar charts (HbO / HbR colour-coded).
alff_topo_figure():     the same two measures drawn on the optode flat map.
fc_matrix_figure():     functional connectivity heatmaps, HbO and HbR as separate subplots.
fc_roi_matrix_figure(): the same, ROI by ROI instead of channel by channel.
fc_seed_topo_figure():  seed-to-whole-brain correlations drawn on the optode flat map.

The two flat maps are built on the shared head in :mod:`fnirs_pipe.qc.figures.common.head_map`
rather than on their own projection, so a channel sits where the report's other head figures
put it.

# TODO: project ALFF/fALFF onto a brain surface (not just the flat map) via mne_nirs when
# head coordinates are available.
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.utils.logging import get_logger

from fnirs_pipe.qc.figures.common._utils import HBO_COLOR, HBR_COLOR
from fnirs_pipe.qc.figures.common.head_map import (
    BLANK_COLOR, head_axes, head_geometry, head_glyph, head_ground,
)

logger = get_logger("qc.figures.rest")

_HBO_COLOR = HBO_COLOR
_HBR_COLOR = HBR_COLOR
_MEAN_LINE_COLOR = "#555555"

# Plotly's RdBu runs red to blue, so it is reversed to put red at r = +1, as the correlation
# panel does. One colour means one r wherever a correlation is drawn in this report.
_FC_SCALE = "RdBu_r"
# an unsigned magnitude, so one hue ramped rather than a diverging pair: the middle of ALFF
# is not a neutral value the way r = 0 is
_ALFF_SCALE = "Viridis"


def _pair_of(ch: str) -> str:
    """"S1_D1 hbo" -> "S1_D1"; the head draws pairs, the frames are keyed by channel."""
    return ch.split(" ")[0]


def alff_falff_figure(
    alff_df: pd.DataFrame,
    title: str = "ALFF and fALFF per Channel",
) -> str:
    """Return base64 PNG of ALFF / fALFF bar charts.

    alff_df must have columns: channel, alff, falff.
    HbO channels (name ends with ' hbo') are drawn in red; HbR in blue.
    Rejected channels arrive NaN and so have no bar; a grey stripe marks where they sat, and
    the mean lines are over the good channels only.
    """
    channels = alff_df["channel"].tolist()
    alff_vals = alff_df["alff"].to_numpy(dtype=float)
    falff_vals = alff_df["falff"].to_numpy(dtype=float)

    is_hbo = np.array([ch.endswith(" hbo") for ch in channels])
    is_bad = (alff_df["bad"].to_numpy(dtype=bool) if "bad" in alff_df.columns
              else np.zeros(len(channels), dtype=bool))
    colors = [_HBO_COLOR if h else _HBR_COLOR for h in is_hbo]
    x = np.arange(len(channels))

    fig, (ax_a, ax_f) = plt.subplots(
        2, 1,
        figsize=(max(6.0, len(channels) * 0.22), 7),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.12)

    for ax, vals, ylabel, row_title in (
        (ax_a, alff_vals,  "ALFF",  "ALFF"),
        (ax_f, falff_vals, "fALFF", "fALFF"),
    ):
        ax.bar(x, vals, color=colors, edgecolor="none", alpha=0.85)

        # a rejected channel's value is NaN, so its bar is missing rather than zero; the
        # stripe says the gap is a rejection and not a channel that measured nothing
        for i in np.flatnonzero(is_bad):
            ax.axvspan(i - 0.5, i + 0.5, color="#eeeeee", lw=0, zorder=0)

        def _mean(mask):
            sel = vals[mask]
            return float(np.nanmean(sel)) if np.isfinite(sel).any() else None

        hbo_mean = _mean(is_hbo) if is_hbo.any() else None
        hbr_mean = _mean(~is_hbo) if (~is_hbo).any() else None
        if hbo_mean is not None:
            # ALFF is ~1e-8 and fALFF ~1e-2, so a fixed number of decimals reads 0.0000
            # on one of the two panels whichever number is chosen
            ax.axhline(hbo_mean, color=_HBO_COLOR, lw=1.0, ls="--", alpha=0.7,
                       label=f"HbO mean = {hbo_mean:.4g}")
        if hbr_mean is not None:
            ax.axhline(hbr_mean, color=_HBR_COLOR, lw=1.0, ls="--", alpha=0.7,
                       label=f"HbR mean = {hbr_mean:.4g}")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(row_title, fontsize=10, pad=4)
        ax.legend(fontsize=7, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)

    step = max(1, len(channels) // 30)
    ax_f.set_xticks(x[::step])
    ax_f.set_xticklabels(
        [channels[i] for i in range(0, len(channels), step)],
        rotation=45, ha="right", fontsize=6,
    )
    ax_f.set_xlabel("Channel", fontsize=9)

    # shared legend for colour meaning
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_HBO_COLOR, label="HbO"),
        Patch(facecolor=_HBR_COLOR, label="HbR"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False,
               bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(title, fontsize=11, y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fc_matrix_figure(
    fc_df: pd.DataFrame,
    fc_hbr_df: pd.DataFrame | None = None,
    title: str = "Functional Connectivity (Pearson r)",
) -> str:
    """Return base64 PNG of FC heatmaps, one panel per chromophore present.

    compute_fc returns one matrix per chromophore, so HbO arrives in fc_df and HbR in
    fc_hbr_df. A single matrix holding both is also accepted and split by channel suffix.
    Diagonal is set to NaN so self-correlations are not shown.

    Blank cells are grey. A rejected channel arrives from compute_fc already NaN, so it shows
    as a full grey row and column; the diagonal is the one-cell grey line through the middle.
    """
    panels: list[tuple[list[str], np.ndarray, str]] = []
    for frame in (fc_df, fc_hbr_df):
        if frame is None or frame.empty:
            continue
        names = frame.columns.tolist()
        mat = frame.to_numpy(dtype=float).copy()
        np.fill_diagonal(mat, np.nan)
        for suffix, label in ((" hbo", "HbO"), (" hbr", "HbR")):
            idx = [i for i, c in enumerate(names) if c.endswith(suffix)]
            if idx:
                panels.append(([names[i] for i in idx], mat[np.ix_(idx, idx)], label))

    if not panels:
        raise ValueError("fc_df has no recognised HbO/HbR channels")

    def _square_size(n: int) -> float:
        return max(3.0, min(n * 0.14, 8.0))

    sizes = [_square_size(len(names)) for names, _, _ in panels]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(sum(sizes) + 2.0, max(sizes)),
        gridspec_kw={"width_ratios": sizes},
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.4)

    for ax, (names, mat, label) in zip(axes[0], panels):
        im = ax.imshow(mat, aspect="equal", cmap=cmap, vmin=-1, vmax=1,
                       interpolation="nearest")
        n = len(names)
        step = max(1, n // 20)
        idxs = list(range(0, n, step))
        ax.set_xticks(idxs)
        ax.set_xticklabels([names[i] for i in idxs], fontsize=6,
                           rotation=45, ha="right")
        ax.set_yticks(idxs)
        ax.set_yticklabels([names[i] for i in idxs], fontsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r", pad=0.02)
        ax.set_title(f"FC — {label}", fontsize=10, pad=6)

    fig.suptitle(title, fontsize=11, y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fc_roi_matrix_figure(
    fc_roi: "dict[str, pd.DataFrame]",
    title: str = "ROI-to-ROI Functional Connectivity (Pearson r)",
) -> str | None:
    """Return base64 PNG of the ROI x ROI FC heatmaps, or None if there is nothing to draw.

    ``fc_roi`` is {chromophore: ROI x ROI frame}, as :func:`compute_fc_roi` returns it. Same
    RdBu_r / +-1 scale as :func:`fc_matrix_figure`, so the ROI view and the channel view can
    be read against each other. A handful of ROIs means every label fits, so unlike that
    figure this one labels and annotates every cell. The diagonal is blanked: an ROI's
    correlation with itself is 1 by construction and says nothing.
    """
    panels = [(fc_roi.get(c), lab) for c, lab in (("hbo", "HbO"), ("hbr", "HbR"))]
    panels = [(f, lab) for f, lab in panels if f is not None and not f.empty]
    if not panels:
        return None

    fig, axes = plt.subplots(
        1, len(panels), figsize=(len(panels) * 4.2 + 1.0, 4.0), squeeze=False,
    )
    fig.subplots_adjust(wspace=0.45)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")

    for ax, (frame, label) in zip(axes[0], panels):
        names = frame.index.tolist()
        mat = frame.to_numpy(dtype=float).copy()
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(mat, aspect="equal", cmap=cmap, vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                if i == j or not np.isfinite(mat[i, j]):
                    continue
                # white on the saturated ends, black in the pale middle
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(mat[i, j]) > 0.6 else "black")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r", pad=0.02)
        ax.set_title(f"ROI FC - {label}", fontsize=10, pad=6)

    fig.suptitle(title, fontsize=11, y=1.02)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _head_for(raw: mne.io.Raw, sep_bands) -> "dict | None":
    """The long-channel flat head this run's maps are drawn on, or None with no positions.

    Short channels are deliberately left out of every map in this module: they measure
    extracerebral signal, so neither a connectivity claim nor a low-frequency amplitude is
    about the cortex there, and including them would set a shared colour scale from signal
    nobody is asking about.
    """
    from fnirs_pipe.qc.metrics import long_short_channels

    long_names, _ = long_short_channels(raw, sep_bands)
    pairs = sorted({_pair_of(ch) for ch in (long_names or raw.ch_names)})
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

    Three states are distinguishable on purpose, because confusing them is the mistake this
    figure exists to avoid:

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
    dim = {_pair_of(ch) for ch in raw.info["bads"]}
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
                       colorscale=_FC_SCALE, dim=dim, blank_color=BLANK_COLOR)
    return _finish_head_grid(fig, geo, n_rows, n_cols, title)


# What each row draws, best column first, and the measured column its hover names. mALFF
# rather than ALFF because the raw amplitude is in molar at 1e-8, which no reader has a
# calibration for, and because it is standardised within each chromophore: that is what lets
# HbO and HbR share one scale without HbR, the smaller of the two by roughly a factor, coming
# out a single dark colour. The measured amplitude is still one hover away.
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

    Beside the row is where it belongs but not where it can go: the grid anchors each head
    square inside a much wider cell, so a vertical bar at a cell's right edge floats in the
    gap between two heads and reads as belonging to neither. The row gap is what the head
    circle leaves free; inside the cell the circle fills the height and a bar there crosses
    its lower arc.
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

    The bar chart in :func:`alff_falff_figure` orders channels by name, which puts no two
    neighbours side by side; low-frequency amplitude is a spatial claim, and this is the
    view it can be read as one in.

    One disc per channel, at its source-detector midpoint, because the value is a property
    of a place rather than of a path. The segment idiom belongs to
    :func:`fc_seed_topo_figure`, where the value really is a claim about a pair of points;
    borrowed here it chained neighbouring channels into polylines that meant nothing.

    Both rows are dimensionless, so a row's two chromophores share one scale and one bar and
    can be read against each other. The rows keep their own: fALFF is a share of the whole
    spectrum and mALFF a multiple of the chromophore's own mean, and nothing is gained by
    putting them on one axis.

    A rejected channel is drawn grey rather than dropped, so the montage stays complete and
    the gap reads as a rejection, and its value is kept out of the scale. Grey alone, not
    grey and faded: for this figure a rejected channel is exactly the channel with no value,
    and a disc carrying both marks is too faint to find.
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
    dim = {_pair_of(ch) for ch in raw.info["bads"]}
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
