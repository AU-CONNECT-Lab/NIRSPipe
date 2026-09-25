"""Motion annotation figures: carpet + GVTD combined, bad segment zoom, per-channel detail.

The OD/GVTD carpet is the pre-processing (raw) motion view; post-denoising uses carpet_compare_figure.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.figures.common._utils import LONG_COLOR, SHORT_COLOR, UNCLASSIFIED_COLOR
from fnirs_pipe.qc.figures.common._utils import decimate as _decimate
from fnirs_pipe.qc.figures.common._utils import line_xy as _line_xy
from fnirs_pipe.qc.metrics import (
    GVTD_MOTION_BAND, GVTD_N_STD, _motion_band_diff, gvtd_threshold, gvtd_timetrace,
)
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.motion_panel")

_MAX_PTS = 4000

_ZOOM_COLORS = ["#e74c3c", "#2980b9", "#27ae60"]

_LW = 1.5          # data traces, thick enough to read at report width
_LW_RULE = 1.2     # threshold rules, thinner than the data they judge

# Muted colours. One green for everything the correction owns: the corrected trace in
# both figures and the footprint strip. Amber shades spikes, dark red draws the threshold.
_GVTD_LINE = "#8494a6"
_GVTD_AFTER = "#55a868"
_DERIVATIVE = "#8172b3"
_OD_BEFORE = "#adb5bd"
_THRESH_RULE = "#c44e52"
_SPIKE_FILL = "rgba(245,158,11,0.20)"
_CORRECTED = "#55a868"

_SPIKE_LABEL = "derivative spikes (≥10% ch)"

# ---- Row geometry ----
# Both motion figures carry a GVTD row and a correction strip, and the two figures are read
# against each other, so those rows get a pixel height here instead of a share of two
# different figure totals.
_GVTD_ROW_PX = 84
_STRIP_ROW_PX = 26


def _px_rows(heights_px: list[int], vertical_spacing: float, chrome_px: int):
    """Row-height fractions and a figure height that make ``heights_px`` land as real pixels.

    ``make_subplots`` takes row heights as fractions of what is left of the plotting area
    after ``vertical_spacing`` has eaten its gaps, so a figure height of
    ``sum(heights_px) + chrome`` renders every row a little short of what it asked for. This
    inflates the plotting area by the gaps first::

        _px_rows([26, 84, 218], 0.04, 100)  ->  ([0.08, 0.26, 0.66], 457)

    ``chrome_px`` is the non-plot furniture: top and bottom margins, plus any legend or
    x-axis title that sits outside them.
    """
    total = sum(heights_px)
    plot_px = total / (1.0 - vertical_spacing * (len(heights_px) - 1))
    return [h / total for h in heights_px], int(round(plot_px)) + chrome_px


# display cap for a GVTD line; well above the carpet's column cap, bounded by the screen
_LINE_MAX_PTS = 10000


def _maxpool_xy(t: np.ndarray, y: np.ndarray, max_pts: int = _LINE_MAX_PTS):
    """Downsample a trace to <= max_pts by taking the max in each bin (keeps spike heights).

    The timestamp kept for each bin is the one the maximum was found at, not the bin's first
    sample, so a peak stays at the time it happened instead of sliding up to a bin edge::

        t = [0, 1, 2, 3], y = [0, 5, 0, 0], step 2  ->  t = [1, 2], y = [5, 0]

    Spacing therefore becomes slightly uneven, which a line plot does not care about.
    Display only: reported GVTD scalars are computed full-res elsewhere and are unaffected.
    """
    n = len(y)
    if n <= max_pts:
        return t, y
    step = n // max_pts
    m = (n // step) * step
    bins = y[:m].reshape(-1, step)
    peak = bins.argmax(axis=1)
    y_ds = bins[np.arange(bins.shape[0]), peak]
    t_ds = t[:m].reshape(-1, step)[np.arange(bins.shape[0]), peak]
    return t_ds, y_ds


def _span_polygons(spans, y0: float, y1: float):
    """Every span as its own closed rectangle in one ``fill="toself"`` trace.

    Vertices are the span edges exactly as given, and the ``None`` between rectangles keeps
    them separate, so neighbouring spans are never bridged into one block::

        [(1.0, 2.0), (7.0, 0.5)] -> x = [1,1,3,3,None, 7,7,7.5,7.5,None]

    Returns two flat lists (x, y), or two ``[None]`` placeholders for an empty span list.
    """
    xs: list = []
    ys: list = []
    for onset, duration in (spans or []):
        x0, x1 = float(onset), float(onset) + float(duration)
        xs += [x0, x0, x1, x1, None]
        ys += [y0, y1, y1, y0, None]
    return (xs or [None]), (ys or [None])


def _tighten_strip(fig, gap: float = 0.006) -> None:
    """Sit the correction strip directly on the panel below it.

    ``make_subplots`` spaces every row equally, which leaves a strip one tenth the height of
    its neighbours floating well clear of the trace it annotates. This moves the first row's
    domain down until it nearly touches the second, without changing its height.
    """
    strip, panel = fig.layout.yaxis, fig.layout.yaxis2
    height = strip.domain[1] - strip.domain[0]
    bottom = min(panel.domain[1] + gap, 1.0 - height)
    strip.domain = (bottom, bottom + height)


def _matched_od_after(
    raw_after: "mne.io.Raw | None",
    ch_names: list[str],
    shape: tuple,
    sfreq: float,
) -> "np.ndarray | None":
    """The motion-corrected recording as OD over ``ch_names``, or None if it does not line up.

    A before/after panel only says something if both sides describe the same thing, so this
    returns data on an exact match and None on anything else, which the caller draws as the
    plain single-trace figure::

        ch_names = [S1_D1 760, S1_D1 850];  after has both, 10 Hz, same n_times  -> (2, N)
        ch_names = [S1_D1 760, S1_D1 850];  after has only S1_D1 760             -> None

    A near miss is worse than nothing here: GVTD over a different channel set would show as
    an effect of the correction. The corrected file is already OD, so the conversion is only
    for the case where a caller hands over intensity.
    """
    if raw_after is None:
        return None
    try:
        od = (raw_after if is_optical_density(raw_after)
              else mne.preprocessing.nirs.optical_density(raw_after.copy()))
        if not set(ch_names) <= set(od.ch_names):            # same channels
            logger.warning("corrected file is missing channels the panel draws; "
                           "before/after comparison dropped")
            return None
        if abs(float(od.info["sfreq"]) - sfreq) > 1e-6:      # same rate
            logger.warning("corrected file is at a different sampling rate; "
                           "before/after comparison dropped")
            return None
        data = od.get_data(picks=ch_names)
        if data.shape != shape:                              # same duration
            logger.warning("corrected file has a different length; "
                           "before/after comparison dropped")
            return None
        return data
    except Exception:
        logger.warning("corrected file unusable; before/after comparison dropped",
                       exc_info=True)
        return None


# ---- GVTD rows ----
# The trace colours stay with before/after, so the separation class is carried by the row
# title and the carpet's side bar instead. A set with no separation meaning ("all", or a
# montage that could not be split) keeps the trace's own grey.
_SET_COLORS = {"long": LONG_COLOR, "short": SHORT_COLOR, "unclassified": UNCLASSIFIED_COLOR}

# Headroom over the tallest sample, leaving the top of each row for the two stat lines.
_GVTD_HEADROOM = 1.45
# y cap percentile, so one spike cannot flatten the rest; each row's maximum is in its label
_GVTD_CAP_PCTL = 99.5

# names the annotation holding a GVTD row's numbers, so a condition view can find it
GVTD_STAT_SLOT = "gvtd-stat-"


def _gvtd_row_label(name: str, n_ch: int) -> str:
    """Row title: the channel set large, its size and band small beside it.

    ``_gvtd_row_label("long", 28)`` -> ``GVTD long   28 ch · 0.01-0.5 Hz``. GVTD is an RMS
    across channels, so which channels went in changes every value on the row and the
    threshold with them.
    """
    return (f"<b>GVTD {name}</b>  <span style='font-size:9px;color:#8b95a1'>"
            f"{n_ch} ch · 0.01–0.5 Hz</span>")


def gvtd_y_top(traces: "list[np.ndarray]", thresholds: "list[float | None]") -> float:
    """The top every GVTD row on a carpet panel shares, over whatever stretch it is given.

    ::

      gvtd_y_top([long_before, long_after, short_before, short_after], [4.8e-4, 3.4e-4])
      -> 0.0127

    A percentile rather than a maximum, so the top 0.5% may run off the canvas. Never below a
    row's threshold, since the line a row is judged against has to stay visible.

    One number for every row, since long and short are the same unit.

    Public because a per-condition view recomputes it over that condition's window.

    ``build_motion_detail_figure`` deliberately draws the same series against its own
    maximum instead, its GVTD row being read as a pair with the derivative row under it.
    """
    caps = ([float(np.nanpercentile(np.concatenate(traces), _GVTD_CAP_PCTL))]
            if traces else [])
    top = max(caps + [t for t in thresholds if t is not None] or [1.0])
    return top * _GVTD_HEADROOM


def _gvtd_stat_label(g: np.ndarray, thresh: "float | None", prefix: str = "") -> str:
    """One line of run-level numbers for a GVTD row, e.g. ``max 1.9e-03 · … · 7.1% above``.

    ``thresh`` is dropped from a corrected row's line, since it is the uncorrected run's
    threshold.
    """
    if g.size == 0:
        return ""
    parts = [f"max {g.max():.2e}", f"mean {g.mean():.2e}"]
    if thresh is not None:
        if not prefix:
            parts.append(f"thresh {thresh:.2e}")
        parts.append(f"{float(np.mean(g > thresh)) * 100:.1f}% above")
    return (prefix + " · " if prefix else "") + " · ".join(parts)


def _blocked_carpet(z, z_after, blocks):
    """The carpet's y labels and ``[(set name, first row, last row), ...]`` for its blocks.

    The images pass through untouched: the blocks sit directly on top of each other and the
    seam between them is a thin drawn rule (see :func:`_carpet_band_marks`)::

        blocks [("long", [a, b]), ("short", [c])]
        -> labels [a, b, c], spans [("long", 0, 1), ("short", 2, 2)]
    """
    labels: list[str] = []
    spans = []
    for name, chs in blocks:
        spans.append((name, len(labels), len(labels) + len(chs) - 1))
        labels += list(chs)
    return z, z_after, labels, spans


# ---- Carpet ----
# Both the subject panel and the dyad panel draw this image, so the scale, the clip and the
# column cap live here once; grey, so a dyad page matches the member's own carpet.
CARPET_MAX_PTS = 2000
CARPET_Z = 3.0


def carpet_coloraxis(z_threshold: float = CARPET_Z, y: float = 0.35) -> dict:
    """The ``coloraxis`` every carpet is drawn on, so no view invents its own scale."""
    return dict(colorscale="Greys", cmin=-z_threshold, cmax=z_threshold,
                colorbar=dict(title="Z-score", thickness=10, len=0.45, y=y))


def carpet_z(
    data: np.ndarray,
    times: np.ndarray,
    z_threshold: float = CARPET_Z,
    stats: "tuple[np.ndarray, np.ndarray] | None" = None,
):
    """Decimated, per-channel z-scored optical density, ready for a ``Heatmap``.

    ::

        carpet_z(od[:, :39611], times)  ->  (z (44, 1980), t (1980,), (mean, std))

    Returns the per-channel mean and SD alongside, so a second carpet of the same recording
    after a correction can be z-scored by the *uncorrected* numbers: rescaling it by its own
    SD would divide out the shrinkage the comparison exists to show. Pass them back as
    ``stats`` to do that. 2 dp because the colour scale cannot resolve more.
    """
    step = max(1, data.shape[1] // CARPET_MAX_PTS)
    carpet = data[:, ::step]
    if stats is None:
        mean = carpet.mean(axis=1, keepdims=True)
        std = carpet.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
    else:
        mean, std = stats
    z = np.round(np.clip((carpet - mean) / std, -z_threshold, z_threshold), 2)
    return z, times[:data.shape[1]:step], (mean, std)


def add_carpet(fig, row: int, z: np.ndarray, t: np.ndarray, labels: list[str],
               spans: list) -> None:
    """One carpet image with its channel-set side bars, on the figure's shared coloraxis.

    The channel names stay in the hover rather than on the axis, where a full montage would
    be an unreadable stack, and the axis is reversed so row 0 of ``z`` is the top row.
    """
    fig.add_trace(go.Heatmap(
        z=z, x=t, y=labels, coloraxis="coloraxis",
        hovertemplate="%{y}<br>t=%{x:.1f}s<br>z=%{z:.2f}<extra></extra>",
        showlegend=False,
    ), row=row, col=1)
    _carpet_band_marks(fig, row, spans, len(labels))
    fig.update_yaxes(showticklabels=False, autorange="reversed", row=row, col=1)


# The division between two channel-set blocks: a rule across the image and a matching break
# in the side bar, kept to a couple of pixels so the blocks read as one carpet.
_SEAM = 0.005


def _carpet_band_marks(fig, row: int, spans: list, n_rows: int) -> None:
    """A colour bar down the left edge of a carpet, one per channel-set block, plus the seam.

    Placed in y-domain fractions rather than by channel name: the axis is categorical and
    reversed, so row ``i`` of ``n`` runs from ``1 - (i+1)/n`` to ``1 - i/n`` up from the
    bottom. The bar carries no text: it takes its colour from the GVTD row that averaged
    those channels, which is directly above and names itself. It sits hard against the
    image's left edge. Skipped for a single block, where the bars would
    all be one colour and tell nothing apart.
    """
    if len(spans) < 2:
        return
    last = len(spans) - 1
    for k, (name, i0, i1) in enumerate(spans):
        y0 = 1.0 - (i1 + 1) / n_rows + (_SEAM / 2 if k < last else 0.0)
        y1 = 1.0 - i0 / n_rows - (_SEAM / 2 if k else 0.0)
        fig.add_shape(type="rect", xref="x domain", yref="y domain",
                      x0=-0.009, x1=-0.002, y0=y0, y1=y1,
                      fillcolor=_SET_COLORS.get(name, _GVTD_LINE), line=dict(width=0),
                      row=row, col=1)
        if k:
            edge = y1 + _SEAM / 2
            fig.add_shape(type="line", xref="x domain", yref="y domain",
                          x0=0, x1=1, y0=edge, y1=edge,
                          line=dict(color="white", width=2), row=row, col=1)


def carpet_gvtd_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    z_threshold: float = 3.0,
    corrected_segments: "list[tuple[float, float]] | None" = None,
    spike_segments: "dict[str, list] | list[tuple[float, float]] | None" = None,
    raw_after: "mne.io.Raw | None" = None,
    channel_set: str | None = None,
    blocks: "list[tuple[str, list[str]]] | None" = None,
) -> go.Figure:
    """Motion-band GVTD + per-channel z-scored OD carpet, on one shared time axis.

    Only the 0.01-0.5 Hz GVTD is drawn, since the unfiltered trace is dominated by the
    cardiac component; ``gvtd_mean`` and ``gvtd_p95`` still report it.
    ``corrected_segments`` (motion-correction footprint) is a bar on a thin strip directly
    above the trace, so what the correction touched sits against what it was aimed at.
    ``spike_segments`` shades the GVTD panel behind the trace, and is either one span list
    for the canonical row or ``{set name: spans}`` so each row shades what was found on its
    own channels. The test behind a span is ">= 10% of these channels spiking", so a long-set
    span is not a statement about the short ones and the two are never shared. Both are drawn
    span by span with no merging,
    since the gap between two spans is the claim that nothing happened there. Neither is
    derived from the trace: spikes are a per-channel robust outlier test on the same
    band-limited derivative, so shading without a peak over the threshold, or the reverse,
    is a real disagreement between a per-channel and a cross-channel reading rather than a
    drawing error.

    Given ``raw_after``, the motion-corrected recording, every GVTD row carries a second
    trace and a second carpet is stacked under the first, so the figure answers whether the
    correction removed what it was there to remove.

    Everything the comparison is read against stays fixed to the uncorrected side: the
    threshold line, and the per-channel mean and SD both carpets are z-scored by.
    Re-deriving either from the corrected data would hide a correction that shrank the
    signal. A ``raw_after`` that does not cover the same channels for the same duration at
    the same rate is dropped rather than drawn (see ``_matched_od_after``).

    ``blocks`` is ``[(set name, channel names), ...]`` with the canonical set first, normally
    ``[("long", ...), ("short", ...)]`` from :func:`gvtd_channel_blocks`. Each gets its own
    GVTD row and its own block of carpet rows, so short-channel quality can be read off the
    same panel without entering the number the verdict is taken from. The sets stay separate
    traces rather than one trace over the union, and the rows share a y range (see
    :func:`gvtd_y_top`). Only the canonical row carries the
    ``segments`` labels, since it is the one the reported scalars come from.
    Omitting ``blocks`` draws the single row ``channel_set`` names, over ``ch_names``.
    """
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy())
    sfreq = float(raw_od.info["sfreq"])

    # one block unless the caller split the montage; either way the carpet keeps the block
    # order, so the rows under a GVTD row are the channels that row averaged
    present = set(ch_names)
    blocks = [(name, [c for c in names if c in present])
              for name, names in (blocks or [(channel_set or "all", list(ch_names))])]
    blocks = [(name, names) for name, names in blocks if names]
    if not blocks:
        blocks = [(channel_set or "all", list(ch_names))]

    ordered = [c for _, names in blocks for c in names]
    od_data, times = raw_od.get_data(picks=ordered, return_times=True)
    od_after = _matched_od_after(raw_after, ordered, od_data.shape, sfreq)
    has_after = od_after is not None

    # GVTD full-res for the threshold/metric; plotted trace is max-pooled for display only.
    # 0.01-0.5 Hz motion band, the band the threshold is set on.
    t_gvtd = times[1:]
    rows: list[dict] = []
    start = 0
    for name, names in blocks:
        rows_slice = slice(start, start + len(names))
        start += len(names)
        gvtd_filt = gvtd_timetrace(od_data[rows_slice], sfreq, *GVTD_MOTION_BAND)
        t_ds, g_ds = _maxpool_xy(t_gvtd, gvtd_filt)
        after, after_ds = None, None
        if has_after:
            after = gvtd_timetrace(od_after[rows_slice], sfreq, *GVTD_MOTION_BAND)
            after_ds = _maxpool_xy(t_gvtd, after)
        rows.append({
            "name": name, "n_ch": len(names), "gvtd": gvtd_filt, "after": after,
            "t_ds": t_ds, "g_ds": g_ds, "after_ds": after_ds,
            "thresh": gvtd_threshold(gvtd_filt, n_std=GVTD_N_STD),
        })

    # carpet: every block's channels, both wavelengths, z-scored by the uncorrected side
    data_z, t_carpet, stats = carpet_z(od_data, times, z_threshold)
    data_z_after = (None if not has_after else
                    carpet_z(od_after, times, z_threshold, stats=stats)[0])
    # a blank row between blocks, so where one set ends is visible without counting channels.
    # the labels are spaces because they still have to be distinct categories on the y axis
    data_z, data_z_after, carpet_rows, band_spans = _blocked_carpet(
        data_z, data_z_after, blocks)

    n_ch      = len(carpet_rows)
    n_carpets = 2 if has_after else 1
    # the strip is the correction's own row, so it is there only when there is a correction
    # to draw; prep-raw runs before any and would otherwise get a labelled empty band
    has_strip = bool(corrected_segments)
    n_rows    = int(has_strip) + len(rows) + n_carpets
    # 6 px a channel, capped: the carpet is read as a texture, not row by row
    carpet_px = int(max(150, min(n_ch * 6, 380)))
    heights   = (([_STRIP_ROW_PX] if has_strip else []) + [_GVTD_ROW_PX] * len(rows)
                 + [carpet_px] * n_carpets)
    vspace    = 0.03
    # chrome: the margins, the legend and the shared x-axis title
    row_heights, total_px = _px_rows(heights, vspace, chrome_px=130)
    strip_row = 1 if has_strip else None
    gvtd_row  = 2 if has_strip else 1

    # only the carpets take a subplot title, in plotly's own styling so they match the
    # titles on every other figure in the report; the GVTD rows name themselves inside their
    # panels, where a title would sit on the row above at this spacing
    carpet_titles = ["before", "after (corrected)"] if has_after else ["carpet"]
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=vspace,
        subplot_titles=[""] * (n_rows - n_carpets) + carpet_titles,
    )
    # plotly sets a title's baseline on its panel's top edge; lift it just clear of the
    # image, which starts at that edge, and it stays with its own carpet rather than
    # drifting into the middle of the gap
    for ann in fig.layout.annotations:
        ann.yshift = 3

    if has_strip:
        xs, ys = _span_polygons(corrected_segments, 0.30, 0.70)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", mode="lines", name="corrected",
            fillcolor=_CORRECTED, line=dict(width=0),
            hovertemplate="corrected: %{x:.1f}s<extra></extra>",
        ), row=strip_row, col=1)

    traces = [r["g_ds"] for r in rows if r["g_ds"].size]
    traces += [r["after_ds"][1] for r in rows
               if r["after_ds"] is not None and np.size(r["after_ds"][1])]
    y_top = gvtd_y_top(traces, [r["thresh"] for r in rows])

    spike_legend_drawn = False
    for i, r in enumerate(rows):
        row_i = gvtd_row + i
        colour = _SET_COLORS.get(r["name"], _GVTD_LINE)
        spans = (spike_segments.get(r["name"]) if isinstance(spike_segments, dict)
                 else (spike_segments if i == 0 else None))
        if spans:
            xs, ys = _span_polygons(spans, 0.0, y_top)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, fill="toself", mode="lines", name=_SPIKE_LABEL,
                fillcolor=_SPIKE_FILL, line=dict(width=0), hoverinfo="skip",
                legendgroup="spikes", showlegend=not spike_legend_drawn,
            ), row=row_i, col=1)
            spike_legend_drawn = True

        fig.add_trace(go.Scatter(
            x=r["t_ds"], y=r["g_ds"], mode="lines",
            name="before" if has_after else "GVTD", legendgroup="before",
            showlegend=(i == 0), line=dict(color=_GVTD_LINE, width=_LW),
            hovertemplate="t=%{x:.1f}s<br>before=%{y:.3e}<extra></extra>",
        ), row=row_i, col=1)

        if r["after_ds"] is not None:
            t_post_ds, g_post_ds = r["after_ds"]
            fig.add_trace(go.Scatter(
                x=t_post_ds, y=g_post_ds, mode="lines", name="after",
                legendgroup="after", showlegend=(i == 0),
                line=dict(color=_GVTD_AFTER, width=_LW),
                hovertemplate="t=%{x:.1f}s<br>after=%{y:.3e}<extra></extra>",
            ), row=row_i, col=1)

        if r["thresh"] is not None:
            # each row's own threshold, from its own distribution; the corrected trace is
            # judged against the uncorrected one, which is the yardstick the % motion in the
            # metrics table is counted against
            fig.add_hline(
                y=r["thresh"], row=row_i, col=1,
                line=dict(color=_THRESH_RULE, width=_LW_RULE, dash="dash"),
            )

        fig.update_yaxes(range=[0, y_top], tickfont=dict(size=8),
                         gridcolor="#eef1f4", zeroline=False, row=row_i, col=1)
        fig.add_annotation(
            x=0.004, xref="x domain", y=0.99, yref="y domain",
            text=_gvtd_row_label(r["name"], r["n_ch"]), showarrow=False,
            xanchor="left", yanchor="top", font=dict(size=13, color=colour),
            row=row_i, col=1,
        )
        # named, so a per-condition view can rewrite the line over its own window rather
        # than leave the run's maximum printed beside an axis that no longer reaches it
        fig.add_annotation(
            x=0.998, xref="x domain", y=0.99, yref="y domain",
            text=_gvtd_stat_label(r["gvtd"], r["thresh"]), showarrow=False,
            xanchor="right", yanchor="top", font=dict(size=9, color=_GVTD_LINE),
            name=f"{GVTD_STAT_SLOT}{r['name']}", row=row_i, col=1,
        )
        if r["after_ds"] is not None:
            fig.add_annotation(
                x=0.998, xref="x domain", y=0.80, yref="y domain",
                text=_gvtd_stat_label(r["after"], r["thresh"], prefix="corrected"),
                showarrow=False, xanchor="right", yanchor="top",
                font=dict(size=9, color=_GVTD_AFTER),
                name=f"{GVTD_STAT_SLOT}{r['name']}-after", row=row_i, col=1,
            )

    carpet_top = gvtd_row + len(rows)
    for i, z in enumerate([data_z, data_z_after][:n_carpets]):
        add_carpet(fig, carpet_top + i, z, t_carpet, carpet_rows, band_spans)

    if has_strip:
        fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False,
                         zeroline=False, row=strip_row, col=1)
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    fig.update_xaxes(range=[float(times[0]), float(times[-1])])
    if has_strip:
        _tighten_strip(fig)

    fig.update_layout(
        height=total_px,
        coloraxis=carpet_coloraxis(z_threshold),
        margin=dict(l=52, r=20, t=34, b=45),
        hovermode="x unified",
        legend=dict(font=dict(size=9), orientation="h",
                    x=1, xanchor="right", y=1.0, yanchor="bottom"),
        plot_bgcolor="white",
    )
    return fig


def bad_segment_zoom_figure(
    raw_after: mne.io.Raw,
    bad_segments: list[tuple[float, float]],
    ch_names: list[str],
    raw_before: "mne.io.Raw | None" = None,
    n_top: int = 10,
    padding: float = 15.0,
) -> "str | None":
    """Zoom into top-N bad segments; show before (optional) and after correction.

    Returns a base64-encoded PNG string, or None if there are no segments.
    Each column = one segment; rows = before (if raw_before given) / after.
    """
    if not bad_segments or not ch_names:
        return None

    sorted_segs = sorted(bad_segments, key=lambda s: s[1], reverse=True)[:n_top]
    n_segs = len(sorted_segs)
    rep_chs = ch_names[:min(3, len(ch_names))]

    n_rows = 2 if raw_before is not None else 1
    row_labels = ["Before", "After"] if n_rows == 2 else ["After"]
    raws = ([raw_before, raw_after] if n_rows == 2 else [raw_after])

    fig, axes = plt.subplots(
        n_rows, n_segs,
        figsize=(max(3 * n_segs, 8), 2.5 * n_rows),
        squeeze=False,
        sharey="row",
    )
    fig.subplots_adjust(hspace=0.12, wspace=0.08)

    t_total = raw_after.times[-1]
    for col_idx, (onset, duration) in enumerate(sorted_segs):
        tmin = max(0.0, onset - padding)
        tmax = min(t_total, onset + duration + padding)

        for row_idx, raw_obj in enumerate(raws):
            ax = axes[row_idx][col_idx]
            try:
                ch_idx = [raw_obj.ch_names.index(c) for c in rep_chs
                          if c in raw_obj.ch_names]
                if not ch_idx:
                    ax.set_visible(False)
                    continue
                data, times = raw_obj.get_data(picks=ch_idx, return_times=True)
                mask = (times >= tmin) & (times <= tmax)
                t_seg = times[mask]
                for i, ts in enumerate(data[:, mask]):
                    std = ts.std()
                    y = (ts - ts.mean()) / std if std > 0 else ts - ts.mean()
                    ax.plot(t_seg, y + i * 3,
                            lw=0.8, color=_ZOOM_COLORS[i % len(_ZOOM_COLORS)],
                            label=rep_chs[i] if col_idx == 0 else None)
                ax.axvspan(onset, onset + duration,
                           color="#e74c3c", alpha=0.15, zorder=0)
                ax.axvline(onset, color="#e74c3c", lw=0.6, ls="--")
                ax.axvline(onset + duration, color="#e74c3c", lw=0.6, ls="--")
            except Exception:
                ax.set_visible(False)
                continue

            ax.set_yticks([])
            ax.set_xlim(tmin, tmax)
            if row_idx == 0:
                ax.set_title(f"t={onset:.1f}s  +{duration:.1f}s", fontsize=7, pad=2)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)

    ax0 = axes[0][0]
    if ax0.get_visible() and ax0.get_lines():
        ax0.legend(fontsize=6, loc="upper left", framealpha=0.7)

    fig.suptitle(
        f"Bad segment zoom: top {n_segs} by duration (red = artifact window)",
        fontsize=9, y=1.01,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_motion_detail_figure(
    raw_od_before: mne.io.Raw,
    raw_od_after: mne.io.Raw,
    ch_name: str,
    segments: "dict | None" = None,
    max_pts: int = _LINE_MAX_PTS,
    corrected_segments: "list[tuple[float, float]] | None" = None,
    spike_segments: "list[tuple[float, float]] | None" = None,
    gvtd_picks: "list[str] | None" = None,
    gvtd_set: str | None = None,
) -> go.Figure:
    """4-row per-channel motion figure: GVTD, this channel's derivative, before/after OD, band strip.

    Both inputs must be in OD space (output of optical_density()). The bottom strip shows
    the motion-correction footprint and spike timepoints, never overlapping the traces.

    Row 2 is this channel's band-limited |dOD/dt|, the signal the spike marks in the strip are
    detected on, so a peak there and a mark below it refer to the same event. It is a
    per-channel view of the spike detector, not a per-channel GVTD: GVTD is defined across
    channels :footcite:`Sherafati2020` and has no single-channel form. Both rows are
    band-limited to ``GVTD_MOTION_BAND`` before differencing, since the unfiltered derivative
    is dominated by the ~1 Hz cardiac component and shows pulse rather than movement.

    ``gvtd_picks`` is the channel set row 1 averages and ``gvtd_set`` names it on the panel.
    The caller passes the separation class ``ch_name`` itself belongs to, because rows 1 and 2
    are read as a pair and the two classes are not on one scale. ``spike_segments`` has to
    come from the same class for the same reason. Left unset, row 1 covers every channel,
    which matches no other panel in the report.

    References
    ----------
    .. footbibliography::
    """
    # GVTD + threshold on full-res OD so they match the reported gvtd_filt_* metrics; the
    # plotted trace is max-pooled afterwards (display only), like the carpet figure.
    od_full, t_full = raw_od_before.get_data(return_times=True)
    full_sfreq = float(raw_od_before.info["sfreq"])
    present = set(raw_od_before.ch_names)
    gvtd_names = [c for c in (gvtd_picks or raw_od_before.ch_names) if c in present]
    gvtd_filt_full = gvtd_timetrace(
        raw_od_before.get_data(picks=gvtd_names) if gvtd_names else od_full,
        full_sfreq, *GVTD_MOTION_BAND)
    motion_thresh  = gvtd_threshold(gvtd_filt_full, n_std=GVTD_N_STD)
    t_gvtd, gvtd_filt = _maxpool_xy(t_full[1:], gvtd_filt_full, max_pts)

    # this channel's |dOD/dt| on full-res OD, through the derivative the spike marks come from
    # differenced before pooling: decimating first would alias the cardiac band back in
    if ch_name in raw_od_before.ch_names:
        ch_idx   = raw_od_before.ch_names.index(ch_name)
        tvd_full = np.abs(_motion_band_diff(od_full[[ch_idx]], full_sfreq)[0])
    else:
        tvd_full = np.zeros(len(t_full) - 1)
    t_tvd, tvd = _maxpool_xy(t_full[1:], tvd_full, max_pts)

    def _get_ch(raw, ch):
        idx = raw.ch_names.index(ch)
        d, t = raw.get_data(picks=[idx], return_times=True)
        d, t = _decimate(d, t, max_pts)
        return t, d[0]

    t_b, y_b = _get_ch(raw_od_before, ch_name)
    t_a, y_a = _get_ch(raw_od_after,  ch_name)

    # same shape as the carpet panel: the correction footprint on a strip over the traces,
    # the spikes shaded behind the two derivative rows they were detected on
    has_strip = bool(corrected_segments)
    strip_row = 1 if has_strip else None
    gvtd_row  = 2 if has_strip else 1
    tvd_row, od_row = gvtd_row + 1, gvtd_row + 2
    # pixel rows, so the GVTD row is the same height here as in carpet_gvtd_figure; the
    # derivative row matches it, since the two are read as a pair
    heights = ([_STRIP_ROW_PX] if has_strip else []) + [_GVTD_ROW_PX, _GVTD_ROW_PX, 218]
    vspace  = 0.04
    row_heights, total_px = _px_rows(heights, vspace, chrome_px=100)  # margins t=60, b=40
    fig = make_subplots(
        rows=len(heights), cols=1,
        shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=vspace,
    )

    if has_strip:
        xs, ys = _span_polygons(corrected_segments, 0.30, 0.70)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", mode="lines", name="corrected",
            fillcolor=_CORRECTED, line=dict(width=0),
            hovertemplate="corrected: %{x:.1f}s<extra></extra>",
        ), row=strip_row, col=1)

    # polygons rather than one vrect per span: a busy channel carries a few hundred of them
    # max, not the carpet's percentile: read as a pair with the derivative row below
    gvtd_top = float(np.nanmax(gvtd_filt)) * 1.1 if len(gvtd_filt) else 1.0
    tvd_top  = float(np.nanmax(tvd)) * 1.1 if len(tvd) else 1.0
    for row, top in ((gvtd_row, gvtd_top), (tvd_row, tvd_top)):
        if not spike_segments:
            break
        xs, ys = _span_polygons(spike_segments, 0.0, top)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", mode="lines", name=_SPIKE_LABEL,
            fillcolor=_SPIKE_FILL, line=dict(width=0), hoverinfo="skip",
            showlegend=row == gvtd_row,
        ), row=row, col=1)

    fig.add_trace(go.Scatter(
        **_line_xy(t_gvtd, gvtd_filt), mode="lines",
        line=dict(color=_GVTD_LINE, width=_LW), name="GVTD 0.01–0.5 Hz",
    ), row=gvtd_row, col=1)
    if motion_thresh is not None:
        fig.add_shape(
            type="line", x0=0, x1=1, xref="x domain",
            y0=motion_thresh, y1=motion_thresh, yref="y",
            line=dict(dash="dash", color=_THRESH_RULE, width=_LW_RULE),
            row=gvtd_row, col=1,
        )
        fig.add_annotation(
            x=1, xref="x domain", y=motion_thresh, yref="y",
            text=f"thresh={motion_thresh:.4f}", showarrow=False,
            font=dict(size=8, color=_THRESH_RULE), xanchor="right", yanchor="bottom",
            row=gvtd_row, col=1,
        )

    fig.add_trace(go.Scatter(
        **_line_xy(t_tvd, tvd), mode="lines",
        line=dict(color=_DERIVATIVE, width=_LW), name="|dOD/dt|",
    ), row=tvd_row, col=1)

    fig.add_trace(go.Scatter(
        **_line_xy(t_b, y_b), mode="lines",
        line=dict(color=_OD_BEFORE, width=_LW), name="Before",
    ), row=od_row, col=1)
    fig.add_trace(go.Scatter(
        **_line_xy(t_a, y_a), mode="lines",
        line=dict(color=_GVTD_AFTER, width=_LW), name="After",
    ), row=od_row, col=1)

    if segments:
        seg_fills = [
            "rgba(231,76,60,0.12)", "rgba(243,156,18,0.12)",
            "rgba(155,89,182,0.12)", "rgba(26,188,156,0.12)",
        ]
        for k, (label, spans) in enumerate(segments.items()):
            fill = seg_fills[k % len(seg_fills)]
            for j, (onset, dur) in enumerate(spans):
                ann = dict(annotation_text=label, annotation_position="top left",
                           annotation_font_size=7) if j == 0 else {}
                for row in (gvtd_row, tvd_row, od_row):
                    fig.add_vrect(
                        x0=onset, x1=onset + dur,
                        fillcolor=fill, line_width=0, layer="below",
                        row=row, col=1,
                        **(ann if row == gvtd_row else {}),
                    )

    if has_strip:
        fig.update_yaxes(title_text="corrected", title_font_size=7, range=[0, 1],
                         showticklabels=False, showgrid=False, zeroline=False,
                         row=strip_row, col=1)
    fig.update_yaxes(title_text="GVTD", tickfont=dict(size=7), range=[0, gvtd_top],
                     row=gvtd_row, col=1)
    fig.update_yaxes(title_text="|dOD/dt|", tickfont=dict(size=7), range=[0, tvd_top],
                     row=tvd_row, col=1)
    fig.update_yaxes(title_text="OD", tickfont=dict(size=7), row=od_row, col=1)
    # what each row is, written inside it: these rows are short enough that a title over one
    # lands on the panel above, and an axis title long enough to say it runs past the row.
    # Row 1 is named the way the carpet panel names its rows.
    for row, label, size, colour in (
        (gvtd_row,
         _gvtd_row_label(gvtd_set or "all", len(gvtd_names) or len(raw_od_before.ch_names)),
         12, _SET_COLORS.get(gvtd_set, _GVTD_LINE)),
        (tvd_row, f"{ch_name}, 0.01–0.5 Hz", 8, "#8b95a1"),
        (od_row, "before / after", 8, "#8b95a1"),
    ):
        fig.add_annotation(
            x=0.004, xref="x domain", y=0.97, yref="y domain",
            text=label, showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=size, color=colour), row=row, col=1,
        )
    fig.update_xaxes(title_text="Time (s)", gridcolor="#eee", row=od_row, col=1)
    if has_strip:
        _tighten_strip(fig)
    fig.update_layout(
        title_text=ch_name, height=total_px,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(font=dict(size=9), orientation="h",
                    x=1, xanchor="right", y=1.0, yanchor="bottom"),
    )
    return fig
