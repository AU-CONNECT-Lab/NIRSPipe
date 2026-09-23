"""Figure builders for the hyperscanning post-processing QC report."""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.hyper_post")


def _log_freq_ticks(freqs: np.ndarray) -> "tuple[list[float], list[float]]":
    """(labelled ticks, unlabelled ticks) for a WTC frequency axis, in Hz.

    e.g. a 0.004-0.2 Hz axis gives ([0.01, 0.1], [0.005 ... 0.09]): the decades carry the
    labels and every intermediate digit gets a bare tick.

    Named rather than left to a locator, which over a range of this width labels all nine
    digits of every decade: 0.008 and 0.009 then sit a fifth as far apart as 0.1 and 0.2 and
    the text collides. One list drives every figure in the report, so two of them cannot
    disagree about what a tick means.
    """
    lo, hi = float(np.min(freqs)), float(np.max(freqs))
    if not (lo > 0 and hi > lo):
        return [], []

    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    steps = [m * 10.0 ** d for d in decades for m in range(1, 10)]
    major = [f for f in steps if f == 10.0 ** round(np.log10(f)) and lo <= f <= hi]
    # a band narrower than two decades carries at most one of them, which is not an axis:
    # fall back to the 1-2-5 pattern, and to the band's own ends when even that is empty.
    # Still one rule the whole way down
    if len(major) < 2:
        major = [f for f in steps if round(f / 10.0 ** np.floor(np.log10(f))) in (1, 2, 5)
                 and lo <= f <= hi]
    if len(major) < 2:
        major = list(np.geomspace(lo, hi, 3))
    return major, [f for f in steps if lo <= f <= hi and f not in major]


def _freq_label(f: float, superscript: str = "html") -> str:
    """"0.01" as "10^-2" where it is a whole power of ten, else three significant digits."""
    exponent = np.log10(f)
    if abs(exponent - round(exponent)) >= 1e-9:
        return f"{f:.3g}"
    e = int(round(exponent))
    return f"10<sup>{e}</sup>" if superscript == "html" else f"$10^{{{e}}}$"


def _apply_log_freq_axis(ax, freqs: np.ndarray) -> None:
    """Put :func:`_log_freq_ticks` on an axis, low frequency at the top.

    Reversed so frequency increases downward, which is how the wavelet coherence figures in
    the literature are drawn.
    """
    major, minor = _log_freq_ticks(freqs)
    ax.set_yscale("log")
    ax.set_ylim(float(np.max(freqs)), float(np.min(freqs)))   # frequency increases downward
    if not major:
        return
    ax.yaxis.set_major_locator(FixedLocator(major))
    # 10^-1 beside 0.02 and 0.05 reads as two different scales, and the 1-2-5 fallback axis
    # is where that happens: one of its ticks is a whole power of ten and the rest are not
    on_decades = all(abs(np.log10(f) - round(np.log10(f))) < 1e-9 for f in major)
    ax.yaxis.set_major_formatter(FixedFormatter(
        [_freq_label(f, "tex") if on_decades else f"{f:.3g}" for f in major]))
    ax.yaxis.set_minor_locator(FixedLocator(minor))
    ax.yaxis.set_minor_formatter(NullFormatter())


# Display threshold for phase arrows when no Monte Carlo level was computed; not a test.
ARROW_MIN_COHERENCE = 0.5


def _clears(wtc_data: dict, arrow_min: float = ARROW_MIN_COHERENCE) -> str:
    """What the caption says an arrow had to clear, named after where the level came from.

    ::

      a map carrying the phase-scrambled null's level -> "the phase-scrambled null"

    Three sources and three wordings, because two of them are tests and the third is not: a
    caption reading "the Monte Carlo level" over arrows drawn at a flat display threshold
    would be claiming a test nobody ran.
    """
    if wtc_data.get("sig") is None:
        return f"{arrow_min:g}"
    return ("the phase-scrambled null" if wtc_data.get("sig_source") == "null"
            else "the Monte Carlo level")


def _arrow_mask(wtc_arr, sig, freqs, freq_coi, arrow_min: float = ARROW_MIN_COHERENCE):
    """Where a phase arrow is worth drawing: inside the cone, and above the noise.

    ::

      a 51 x 3962 map -> a boolean of the same shape, usually a few percent True

    Two conditions, and both matter. **Inside the cone**, because a coefficient built against
    the padding has a phase built against the padding too, and the old figure drew those
    arrows at the same weight as the rest. **Above the level**, the Monte Carlo one when
    ``--wtc-significance`` produced it and ``arrow_min`` otherwise, because the relative
    phase of two uncorrelated series is a uniformly random direction and a field of those
    reads as structure to the eye.
    """
    inside = freqs[:, None] >= freq_coi[None, :]
    if sig is not None and len(np.asarray(sig)) == wtc_arr.shape[0]:
        above = wtc_arr >= np.asarray(sig, dtype=float)[:, None]
    else:
        above = wtc_arr >= float(arrow_min)
    return inside & above


def build_wtc_channel(
    wtc_data: dict,
    freqs: np.ndarray,
    times: np.ndarray,
    pair_label: str,
    markers_list: list[dict],
    cond_colors: dict[str, str],
    site_label: str = "",
    arrow_min: float = ARROW_MIN_COHERENCE,
) -> "str | None":
    """WTC map for one channel or ROI pair, as a base64 PNG: time x log-frequency, colour 0-1.

    ::

      wtc_data: {"wtc": ndarray(n_freqs, n_times), "coi": ndarray(n_times),
                 "phase": ndarray(n_freqs, n_times), "sig": ndarray(n_freqs) | None}

    Frequency runs downward on a log axis labelled at the decades. ``site_label`` names
    the pairing in the title, ``"S1_D1"`` for a homologous one and ``"S1_D1 × S2_D2"`` for a
    crossed one, which is what the pair of selectors above the panel picks. Four things are
    drawn on top of the map:

    - the **phase arrows**, thinned onto a coarse grid by :func:`_phase_arrows`. Right is in
      phase, left antiphase, up means the first member leads by a quarter cycle. A quiver
      field is the half of a coherence map that says which brain led. Plotly draws one too,
      through ``figure_factory.create_quiver``, so the arrows are not what keeps this panel a
      PNG; its arrowheads are laid out in data coordinates and would skew on a log axis.
    - the region **outside the cone of influence**, washed out rather than only bounded by
      the dashed line. Those cells are coefficients padded against the record's edges, near 1
      whatever the data did, so a reader who takes them for signal reads the ends of every
      recording as strongly coupled.
    - the **significance contour**, where coherence beats the Monte Carlo level, when one was
      computed.
    - one **span bar per block** above the axes with a line at each end of it, and one
      legend entry per condition in the top right. A block design repeats a condition, so a
      label on every bar printed the same word once per block. The bar runs the block's
      actual length, so the gaps between blocks are visible:
      a recording is continuous and its untasked stretches are data like any other, which a
      set of onset lines alone made look like block boundaries. The interactive version
      shaded each block on the map instead, under an opaque heatmap, so nothing showed;
      moving the span outside the axes is that information back where it cannot be covered.
    """
    if wtc_data is None or len(freqs) == 0 or len(times) == 0:
        return None

    wtc_arr = np.asarray(wtc_data["wtc"], dtype=float)
    coi     = np.asarray(wtc_data["coi"], dtype=float)
    sig     = wtc_data.get("sig")
    freqs, times = np.asarray(freqs, dtype=float), np.asarray(times, dtype=float)

    fig, ax = plt.subplots(figsize=WTC_FIGSIZE)
    mesh = ax.pcolormesh(times, freqs, wtc_arr, cmap="viridis", vmin=0, vmax=1,
                         shading="nearest")
    _apply_log_freq_axis(ax, freqs)

    freq_coi = _freq_coi(coi, freqs)
    ax.fill_between(times, freq_coi, freqs.min(), color="white", alpha=0.45, lw=0, zorder=2)
    ax.plot(times, freq_coi, color="white", lw=1.3, ls="--", zorder=3)

    if sig is not None and len(sig) == wtc_arr.shape[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = wtc_arr / np.asarray(sig, dtype=float)[:, None]
        ax.contour(times, freqs, ratio, levels=[1.0], colors="black", linewidths=1.1,
                   zorder=4)

    # one span bar per block above the axes, and a line at each end of it on the map itself.
    # The bar says where the block is and the lines say where it starts and stops; without
    # the closing line a reader inside the heatmap has to look up at the bar to find the end.
    # A line is a claim about one moment, so an end past the right edge is dropped rather
    # than drawn at the edge, the same rule the onset already followed. The bar is cut there
    # instead, being about where the block sits rather than how long it is
    drawn_conditions: dict[str, str] = {}
    for m in markers_list:
        onset, duration = float(m["onset"]), float(m["duration"])
        if duration <= 0.1 or not (times[0] <= onset <= times[-1]):
            continue
        colour = cond_colors.get(m["description"], "#f39c12")
        offset = onset + duration
        ax.axvline(onset, color=colour, lw=1.1, ls=":", zorder=5)
        if times[0] <= offset <= times[-1]:
            ax.axvline(offset, color=colour, lw=1.1, ls=":", zorder=5)
        ax.plot([onset, min(offset, float(times[-1]))], [1.012, 1.012],
                transform=ax.get_xaxis_transform(), color=colour, lw=3.0,
                solid_capstyle="butt", clip_on=False, zorder=5)
        drawn_conditions.setdefault(m["description"], colour)

    # this panel is full width, so it takes more arrows and much smaller ones than the
    # thumbnails elsewhere do
    _phase_arrows(ax, times, freqs, wtc_data.get("phase"),
                  n_time=34, n_freq=13, scale=WTC_QUIVER_SCALE, width=0.0022,
                  mask=_arrow_mask(wtc_arr, sig, freqs, freq_coi, arrow_min))

    lead = (pair_label.split("×")[0].strip() or "the first member"
            if pair_label else "the first member")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.tick_params(labelsize=8)

    # ---- title left, condition legend right, both on the strip above the span bars ----
    # A block design repeats a condition, and naming every bar printed "game1" eight times.
    # One key in the corner names each colour once; the bars keep the colour and drop the text
    legend_rows = 1
    if drawn_conditions:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=c, lw=3.0, solid_capstyle="butt")
                   for c in drawn_conditions.values()]
        n_col = min(len(handles), 6)
        legend_rows = -(-len(handles) // n_col)
        ax.legend(handles, list(drawn_conditions), loc="lower right",
                  bbox_to_anchor=(1.0, 1.045), ncol=n_col, frameon=False, fontsize=7.5,
                  handlelength=1.1, handletextpad=0.45, columnspacing=1.3,
                  labelspacing=0.3, borderpad=0.0, borderaxespad=0.0)
    heading = f"{site_label}   {pair_label}".strip() if site_label else pair_label
    ax.set_title(heading, fontsize=10, loc="left", pad=20 + 10 * (legend_rows - 1))
    # the live twin of this panel has no frame, and a framed map beside an unframed one
    # reads as two different kinds of figure
    for spine in ax.spines.values():
        spine.set_visible(False)
    flat_colorbar(fig, mesh, ax, "WTC", pad=0.015)
    # Two lines, and this is about the figure's width rather than about the wording: the
    # crop on the way out takes the widest thing drawn, and on one line this caption is wider
    # than the map and its colorbar together, which left the saved image with a sixth of its
    # width in white to the right of the bar.
    caption = (f"arrows: right = in phase, left = antiphase, up = {lead} leads by a quarter "
               f"cycle,\ndrawn only where coherence clears "
               f"{_clears(wtc_data, arrow_min)}"
               "    washed-out band: outside the cone of influence")
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=8, color="#444444",
             linespacing=1.5)
    return _png_b64(fig)


# Height only. The panel has no width of its own: it fills whatever the report's iframe is,
# which it can do because the arrows are measured in pixels rather than in data.
_INTERACTIVE_PLOT_H = 430

# The cross panel's height, the same way. Two square panels side by side, so this is what
# decides how large a square gets: the width is the page's, and the aspect constraint
# letterboxes whatever the height does not use. At 550 a matrix came out around 400 px on a
# wide monitor with a third of the panel blank to either side of it.
PANEL_HEIGHT = 720

# Arrow length in pixels. The still's is a fifty-second of its axes, which on a report page
# about 1600 px wide comes out near this; it is a pixel length here rather than a share of
# the panel because an annotation's tail is offset in pixels.
_INTERACTIVE_ARROW_PX = 23.0

# Name every arrow carries, so the resize hook can find them among the figure's annotations
# and a per-condition view can swap the whole set without disturbing the caption.
_ARROW_NAME = "wtcarrow"

# Wide and flat, and this is a layout number rather than a taste one. The page caps the
# still at the height of the live ROI map under it, so on any monitor wide enough the aspect
# alone decides how much of the column is left blank beside it: with the colorbar and labels
# these 5 inches per inch come out near 3.1, which fills the width of a 1920 page at that
# cap. At the 3.06 this started from the panel was half again as tall as the map below it and
# at 4 it left a sixth of the page white.
WTC_FIGSIZE = (15.0, 3.0)
WTC_QUIVER_SCALE = 52.0

# ---- why the live panel's lines are drawn about twice their matplotlib widths ----
# The still's widths are in points and it is written at 300 dpi, so a 1.1 pt line lands as
# 4.6 px in the file and, once the page has scaled that file to its own width, as about
# 2.4 CSS px. Plotly is handed CSS pixels directly, so the same 1.1 draws half as heavy a
# line. Every width below is the still's own, doubled, which is what makes the two panels
# look like one pair rather than a drawing and a sketch of it.
_LIVE_LINE_SCALE = 2.2

# Height from width, and arrow length from the plot area, on every draw and every resize.
# It runs again after a relayout because that is when a condition view swaps in its own
# arrows, and each pass renormalises whatever length it finds rather than scaling what is
# there, so running twice is the same as running once. The one-pixel guard is what stops
# the relayout it makes from calling it forever.
def _arrow_annotations(wtc_data: dict, freqs: np.ndarray, times: np.ndarray,
                       freq_coi: np.ndarray,
                       arrow_min: float = ARROW_MIN_COHERENCE) -> list[dict]:
    """The phase field of one map as Plotly annotations, thinned onto the same grid.

    ::

      a 51 x 3962 map -> about 150 arrows, one annotation each

    The live twin of :func:`_phase_arrows`, and a function rather than a block inside the
    figure because a per-condition view of that figure needs its own set: the grid spans
    whatever time axis it is given, so a 300 s condition read off a 1800 s run would show
    the five columns of the run's grid that happen to land inside it.

    Inset off the edges, which the matplotlib panel does not need: its quiver is clipped at
    the axes and an annotation is not, so an arrow on the outermost row would hang its tail
    over the tick labels.
    """
    def _grid(n: int, count: int) -> np.ndarray:
        return np.unique(np.linspace(0.03, 0.97, min(count, n)) * (n - 1)).astype(int)

    wtc_arr = np.asarray(wtc_data["wtc"], dtype=float)
    sig     = wtc_data.get("sig")
    phase   = wtc_data.get("phase")
    if phase is None or np.asarray(phase).shape != wtc_arr.shape:
        return []

    fi = _grid(len(freqs), 13)
    ti = _grid(len(times), 34)
    keep = _arrow_mask(wtc_arr, sig, freqs, freq_coi, arrow_min)[np.ix_(fi, ti)]
    if not keep.any():
        return []

    angle = np.asarray(phase, dtype=float)[np.ix_(fi, ti)]
    arrows = []
    for r, f_i in enumerate(fi):
        for c, t_i in enumerate(ti):
            if not keep[r, c]:
                continue
            a = float(angle[r, c])
            arrows.append(dict(
                # a log axis takes an annotation's coordinate in log10, not in Hz. Given in
                # Hz every arrow lands off the plot and is clipped away with no error at
                # all, which is how this was drawing nothing
                x=float(times[t_i]), y=float(np.log10(freqs[f_i])),
                ax=-_INTERACTIVE_ARROW_PX * np.cos(a),
                ay=_INTERACTIVE_ARROW_PX * np.sin(a),
                axref="pixel", ayref="pixel", text="", showarrow=True, name=_ARROW_NAME,
                arrowhead=2, arrowsize=0.8, arrowcolor="black",
                arrowwidth=1.1 * _LIVE_LINE_SCALE,
            ))
    return arrows


def _freq_coi(coi: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """The cone as the lowest frequency each time point can still be measured at."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.asarray(coi, dtype=float) > 1e-10,
                       1.0 / np.asarray(coi, dtype=float), freqs.max())
    return np.clip(out, freqs.min(), freqs.max())


def wtc_condition_views(fig, spans, wtc_data: dict, freqs: np.ndarray, times: np.ndarray,
                        arrow_min: float = ARROW_MIN_COHERENCE) -> "dict | None":
    """Each condition's view of one whole-run coherence map, keyed by slug for a page to pick.

    ::

      wtc_condition_views(fig, [("video", 300.0, 600.0)], data, freqs, times)
      -> {"video": {"x": [300.0, 600.0], "annotations": [...]}}

    What the file then serves is the run at its own URL and each condition at
    ``…/map.html#video``, which is the arrangement the subject report's time-axis figures
    already use. It is sound here for the same reason the tables are: a condition is read
    out of the whole-run transform and never cut from it, so its map *is* a slice of this
    one, down to the cone. A run transformed per condition is a different figure and gets
    its own file; the caller decides, since it is the one that knows which route ran.

    Only the arrows are rebuilt. They sit on a grid spread over whatever span they were
    drawn for, so a narrowed view of the run's set would show a handful of columns; each
    condition carries its own set and the caption is left alone.
    """
    if not spans or fig is None:
        return None
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    fixed = [a.to_plotly_json() for a in (fig.layout.annotations or ())
             if a.name != _ARROW_NAME]

    out = {}
    from fnirs_pipe.qc.common.figure_io import _pair_fname
    for label, t0, t1 in spans:
        keep = (times >= float(t0)) & (times <= float(t1))
        if not keep.any():
            continue
        cut = {"wtc": np.asarray(wtc_data["wtc"])[:, keep],
               "phase": (None if wtc_data.get("phase") is None
                         else np.asarray(wtc_data["phase"])[:, keep]),
               "sig": wtc_data.get("sig"),
               "sig_source": wtc_data.get("sig_source")}
        coi = _freq_coi(np.asarray(wtc_data["coi"])[keep], freqs)
        out[_pair_fname(label)] = {
            "x": [float(t0), float(t1)],
            "annotations": fixed + _arrow_annotations(cut, freqs, times[keep], coi,
                                                      arrow_min),
        }
    return out or None


def build_wtc_map_interactive(
    wtc_data: dict,
    freqs: np.ndarray,
    times: np.ndarray,
    pair_label: str,
    markers_list: list[dict],
    cond_colors: dict[str, str],
    site_label: str = "",
    arrow_min: float = ARROW_MIN_COHERENCE,
):
    """:func:`build_wtc_channel` as a Plotly figure, for the panels worth zooming into.

    ::

      the same wtc_data -> go.Figure, about 3 MB of standalone HTML

    Same map, same cone, same arrow rule, same span bars and legend. What it adds is hover
    (time, frequency and coherence per cell) and zoom, which is the whole reason the ROI
    panel takes this and the channel panels stay PNG: a per-channel map costs the same 3 MB
    and there are 2352 of them on a 14-channel crossed dyad against 192 ROI maps.

    **Each arrow is an annotation anchored in data with its tail offset in pixels**
    (``axref="pixel"``), which is what ``angles="uv"`` gives the matplotlib panel: the head
    sits on its grid point and the direction is a screen direction, so a relative phase of a
    quarter cycle draws a quarter turn however wide the frame ends up. ``ff.create_quiver``
    would work too and is one trace instead of N, but it lays the barb and both head strokes
    out in *data* coordinates, and on axes of seconds against log-Hz that shears every
    arrowhead and makes the angle depend on the rendered width. An arrow costs about 180
    bytes against the map's three megabytes, so the trace count is not worth the distortion.
    Plotly's ``ay`` grows downward, which is the sign that makes ``sin`` point up.

    No ``zsmooth``. Plotly would interpolate the cells, and the moire the pixel grid makes of
    a map this wide would go with it, but so would the real structure: at the fast end of the
    band the coherence decorrelates in about ten seconds, and those stripes are the data.
    """
    if wtc_data is None or len(freqs) == 0 or len(times) == 0:
        return None

    wtc_arr = np.asarray(wtc_data["wtc"], dtype=float)
    coi     = np.asarray(wtc_data["coi"], dtype=float)
    sig     = wtc_data.get("sig")
    freqs, times = np.asarray(freqs, dtype=float), np.asarray(times, dtype=float)

    lf0, lf1 = float(np.log10(freqs).min()), float(np.log10(freqs).max())
    t0, t1 = float(times[0]), float(times[-1])
    # a single time point or a single scale has no axis to draw on
    if t1 <= t0 or lf1 <= lf0:
        return None

    fig = go.Figure()
    # three decimals: the payload is one number per cell written out as text, and no page in
    # this package reads a coherence past the third place
    fig.add_trace(go.Heatmap(
        z=np.round(wtc_arr, 3), x=times, y=freqs,
        colorscale="Viridis", zmin=0.0, zmax=1.0,
        colorbar=dict(title="WTC", thickness=14, len=0.92, x=1.005),
        hovertemplate="t = %{x:.0f} s<br>f = %{y:.4g} Hz<br>WTC = %{z:.3f}<extra></extra>",
    ))

    freq_coi = _freq_coi(coi, freqs)
    fig.add_trace(go.Scatter(
        x=np.concatenate([times, times[::-1]]),
        y=np.concatenate([freq_coi, np.full_like(times, freqs.min())]),
        fill="toself", fillcolor="rgba(255,255,255,0.45)", line=dict(width=0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=times, y=freq_coi, mode="lines",
        line=dict(color="white", width=1.3 * _LIVE_LINE_SCALE, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))

    if sig is not None and len(np.asarray(sig)) == wtc_arr.shape[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = wtc_arr / np.asarray(sig, dtype=float)[:, None]
        fig.add_trace(go.Contour(
            z=ratio, x=times, y=freqs, showscale=False, hoverinfo="skip",
            contours=dict(start=1.0, end=1.0, size=1.0, coloring="none"),
            line=dict(color="black", width=1.1 * _LIVE_LINE_SCALE),
        ))

    arrows = _arrow_annotations(wtc_data, freqs, times, freq_coi, arrow_min)

    # ---- one span bar per block, one legend entry per condition ----
    shapes, seen = [], {}
    for m in markers_list:
        onset, duration = float(m["onset"]), float(m["duration"])
        if duration <= 0.1 or not (t0 <= onset <= t1):
            continue
        colour = cond_colors.get(m["description"], "#f39c12")
        shapes.append(dict(type="line", xref="x", yref="paper", y0=1.012, y1=1.012,
                           x0=onset, x1=min(onset + duration, t1),
                           line=dict(color=colour, width=3.0 * _LIVE_LINE_SCALE)))
        for edge in (onset, onset + duration):
            if t0 <= edge <= t1:
                # against the paper rather than the axis: the line means "the full height",
                # and a log axis takes its own coordinates in log10
                shapes.append(dict(type="line", xref="x", yref="paper", x0=edge, x1=edge,
                                   y0=0.0, y1=1.0,
                                   line=dict(color=colour, dash="dot",
                                             width=1.1 * _LIVE_LINE_SCALE)))
        seen.setdefault(m["description"], colour)
    for label, colour in seen.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label,
                                 line=dict(color=colour, width=3.0 * _LIVE_LINE_SCALE),
                                 hoverinfo="skip"))

    major, minor = _log_freq_ticks(freqs)
    on_decades = all(abs(np.log10(f) - round(np.log10(f))) < 1e-9 for f in major)
    lead = (pair_label.split("×")[0].strip() or "the first member"
            if pair_label else "the first member")
    heading = f"{site_label}   {pair_label}".strip() if site_label else pair_label
    clears = _clears(wtc_data, arrow_min)
    fig.update_layout(
        height=_INTERACTIVE_PLOT_H + 150, autosize=True,
        margin=dict(l=70, r=80, t=90, b=95),
        title=dict(text=heading, x=0.0, xanchor="left", y=0.965, font=dict(size=15)),
        legend=dict(orientation="h", x=1.0, xanchor="right", y=1.035, yanchor="bottom",
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        shapes=shapes, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Time (s)", range=[t0, t1], showgrid=False, zeroline=False),
        # a log axis takes its range in log10 units. Given explicitly so the outermost row
        # does not hang half a cell past the washed band, which reads as an uncut stripe
        yaxis=dict(title="Frequency (Hz)", type="log", range=[lf1, lf0],
                   tickmode="array", tickvals=major,
                   ticktext=[_freq_label(f, "html") if on_decades else f"{f:.3g}"
                             for f in major],
                   showgrid=False, zeroline=False),
        annotations=arrows + [dict(
            xref="paper", yref="paper", x=0.0, y=-0.155, xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=11, color="#444444"),
            text=("arrows: right = in phase, left = antiphase, up = "
                  f"{lead} leads by a quarter cycle, drawn only where coherence clears "
                  f"{clears}"
                  "&nbsp;&nbsp;&nbsp;&nbsp;washed-out band: outside the cone of influence"),
        )],
    )
    fig.update_yaxes(minor=dict(tickvals=minor, showgrid=False))
    return fig


def flat_colorbar(fig, mappable, ax, label: str, **kwargs):
    """A colorbar with no black box around it. Every bar in this report goes through it.

    ::

      flat_colorbar(fig, mesh, ax, "WTC", pad=0.015)

    The live ROI panel is Plotly and draws an unframed bar; the stills are matplotlib and
    drew a framed one, so the same scale appeared twice on one page in two liveries. One
    look, and the choice is made in one place.
    """
    bar = fig.colorbar(mappable, ax=ax, label=label, **kwargs)
    bar.outline.set_visible(False)
    return bar


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# Node colours for the two members, on every circle this report draws: purple and green
# rather than a red and a blue, which on these pages mean HbO and HbR.
NODE_COLORS = ("#8e44ad", "#27ae60")

# One scale per quantity, each over its own full range. A correlation is signed and takes a
# The scales, the blank cell and the printed cell value are the report's, not this figure's;
# see figures.common.matrix_map for which quantity takes which and why
from fnirs_pipe.qc.figures.common.matrix_map import (  # noqa: E402
    BLANK_CELL, COHERENCE_SCALE, CORRELATION_SCALE, cell_values as _cell_values,
    matrix_ground, scale_color as _arc_color,
)
# The circle itself is the report's, not this figure's: one subject's channels against each
# other and two members' against each other are the same picture on different matrices
from fnirs_pipe.qc.figures.common.circle_map import (  # noqa: E402
    DYAD_GAP as _CIRCLE_GAP, bezier as _bezier, circle_axes, node_arc, radial_label,
    ring_angles,
)


# Half-width of the connectogram's axes, with the nodes on the unit circle: what is left
# over is the ring the radial labels are written into.
_CIRCLE_SPAN = 1.16

# The colorbar every cross-brain panel carries, minus its title. One layout, so a figure of
# two panels and a figure of one put their scale in the same place.
_COLORBAR = {"thickness": 13, "len": 0.72, "x": 1.0, "y": 0.46}


def _matrix_panel(fig, z, row_labels, col_labels, subject_ids, *, cmap, vmin, vmax,
                  value_label, row, col, colorbar=None):
    """One cross-brain matrix into a subplot: the blank ground, the cells, their values.

    ``colorbar`` is the bar's layout for the one panel that carries it; every other panel
    passes None and draws none, which is what puts several panels of a figure on one scale.
    """
    sub1 = subject_ids[0] if subject_ids else "Sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "Sub2"

    # a blank cell here is a site one member lost
    matrix_ground(fig, len(row_labels), len(col_labels), row, col)
    fig.add_trace(go.Heatmap(
        z=z, x=list(col_labels), y=list(row_labels), colorscale=cmap,
        zmin=vmin, zmax=vmax, showscale=colorbar is not None, colorbar=colorbar or {},
        hovertemplate=(f"{sub1} %{{y}} × {sub2} %{{x}}"
                       f"<br>{value_label} = %{{z:.3f}}<extra></extra>"),
    ), row=row, col=col)
    _cell_values(fig, z, row_labels, col_labels, cmap=cmap, vmin=vmin, vmax=vmax,
                 row=row, col=col)


def _circle_traces(fig, z, row_labels, col_labels, subject_ids, *,
                   cmap, vmin, vmax, value_label, keep, row, col):
    """Draw the connectogram into one subplot: node arcs, labels, and one chord per pairing.

    ``keep`` is the boolean of which cells get a chord, already decided by the caller. Every
    chord runs from a node on the left semicircle to one on the right, because the matrix is
    one brain against the other and has no within-brain cell to draw.
    """
    n = len(row_labels)
    ang = np.deg2rad(ring_angles([n, n], gap=_CIRCLE_GAP))
    xy = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    # the nodes are evenly spaced, so any two neighbours inside one member give the pitch
    step = float(ang[1] - ang[0]) if n > 1 else np.deg2rad(20.0)

    # ---- one thick arc per node, and its label outside it ----
    sub1 = subject_ids[0] if subject_ids else "Sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "Sub2"
    labels = list(row_labels) + list(col_labels)

    for k, name in enumerate(labels):
        side = 0 if k < n else 1
        node_arc(fig, ang[k], step * 0.42, f"{(sub1, sub2)[side]}<br>{name}", row, col,
                 colour=NODE_COLORS[side])
        radial_label(fig, ang[k], 1.06, name, row, col)

    # ---- the chords, weakest first so the strong ones are not drawn under them ----
    pairs = [(i, j) for i in range(n) for j in range(n) if keep[i, j]]
    pairs.sort(key=lambda ij: abs(z[ij]))
    for i, j in pairs:
        value = float(z[i, j])
        x, y = _bezier(xy[i], xy[n + j])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=_arc_color(value, cmap, vmin, vmax), width=1.6),
            showlegend=False,
            hovertemplate=(f"{sub1} {row_labels[i]} × {sub2} {col_labels[j]}"
                           f"<br>{value_label} = {value:.3f}<extra></extra>"),
        ), row=row, col=col)
    return len(pairs)


# Share of pairings a connectogram keeps when there is no null to rank them against. A
# display cut, and the subtitle says so: it draws the same number of chords whatever the data
# did, which is exactly what a test must not do.
ARC_FALLBACK_QUANTILE = 0.90


def _arc_rule(
    z: np.ndarray,
    value_label: str,
    threshold: "float | None",
    level: "np.ndarray | None",
    quantile: "float | None",
) -> "tuple[np.ndarray, str]":
    """Which cells get a chord, and the sentence over the circle saying why.

    ::

      level given      -> cells above their own null          "above its own null (95th pct)"
      threshold given  -> cells over one number               "|Pearson r| >= 0.3"
      quantile given   -> the strongest share of them         "strongest 10% (display cut)"

    Three tiers in that order of preference, which is the order they deserve to be believed
    in. A per-cell ``level`` out of a surrogate distribution is a test; a fixed
    ``threshold`` is a number somebody chose, and the scale it has to be chosen on moves with
    the preprocessing, so it wins only when asked for explicitly; a ``quantile`` of the matrix
    itself keeps the strongest share whatever they are worth and is labelled as the display
    cut it is. With none of the three, every finite pairing is drawn.

    The rule string uses the ">=" character rather than the HTML entity: plotly's text parser
    takes the tags it knows, and an escaped entity in a subplot title fails the whole render.
    """
    finite = np.isfinite(z)
    if threshold is not None:
        return finite & (np.abs(z) >= float(threshold)), f"|{value_label}| ≥ {threshold:g}"
    if level is not None:
        level = np.asarray(level, dtype=float)
        keep = finite & np.isfinite(level) & (np.abs(z) >= level)
        return keep, f"above its own null ({int(keep.sum())} of {int(finite.sum())})"
    if quantile is not None and finite.any():
        cut = float(np.nanquantile(np.abs(z[finite]), quantile))
        share = int(round((1.0 - quantile) * 100))
        return finite & (np.abs(z) >= cut), f"strongest {share}% (display cut, not a test)"
    return finite, f"all {int(finite.sum())} pairings"


def _cross_matrix_figure(
    panels: "list[tuple[str, np.ndarray]]",
    labels: list[str],
    subject_ids: list[str],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    value_label: str,
    title: str,
    kind: str,
):
    """Several cross-brain matrices side by side on one colour scale, one per chromophore.

    ::

      [("HbO", 14 x 14), ("HbR", 14 x 14)] -> [ HbO heatmap | HbR heatmap ], one colorbar

    Shared by the coherence matrices and the ROI ISC one, which differ only in scale and
    wording. Rows are the first member's sites and columns the second's, so the diagonal is
    the homologous pairing and no cell is within-brain. Each panel's subtitle carries its
    own grand mean, which is the number the two chromophores are compared on.

    ``panels`` is ``[(display name, matrix), ...]``, already filtered to the ones worth
    drawing. The last panel carries the colorbar and the rest draw none, which is what puts
    them all on one scale.
    """
    sub1 = subject_ids[0] if subject_ids else "sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "sub2"

    fig = make_subplots(
        rows=1, cols=len(panels), horizontal_spacing=0.06,
        subplot_titles=[f"{name}   (grand mean {float(np.nanmean(z)):.3f})"
                        for name, z in panels],
    )

    for i, (_, z) in enumerate(panels, start=1):
        _matrix_panel(fig, z, labels, labels, subject_ids,
                      cmap=cmap, vmin=vmin, vmax=vmax, value_label=value_label,
                      row=1, col=i,
                      colorbar=(_COLORBAR | {"title": value_label}
                                if i == len(panels) else None))
        # Ranges given rather than left to autorange, which pads a heatmap by a fraction of
        # a cell on every side and leaves a white margin around the grid. A cell spans half a
        # step either side of its index, so these two are the grid's own extent.
        fig.update_xaxes(title_text=f"{sub2} {kind}".strip(), side="bottom", tickangle=-90,
                         range=[-0.5, len(labels) - 0.5], showgrid=False, zeroline=False,
                         constrain="domain", row=1, col=i)
        # the first row at the top, which is how the table it stands for is read. The y title
        # goes on the left panel alone: both rows are the same member and printing it twice
        # reads as two different axes
        fig.update_yaxes(title_text=(f"{sub1} {kind}".strip() if i == 1 else ""),
                         range=[len(labels) - 0.5, -0.5], showgrid=False, zeroline=False,
                         scaleanchor=f"x{'' if i == 1 else i}", constrain="domain",
                         row=1, col=i)

    fig.update_layout(
        height=PANEL_HEIGHT, autosize=True,
        margin=dict(l=70, r=90, t=84, b=70),
        title=dict(text=title, x=0.5, xanchor="center", y=0.975, font=dict(size=14)),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    for note in fig.layout.annotations[:len(panels)]:
        note.font.size = 11
        note.font.color = "#444444"
    return fig


def build_cross_panel(
    z: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    subject_ids: list[str],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    value_label: str,
    matrix_title: str,
    suptitle: str,
    arc_threshold: "float | None",
    arc_level: "np.ndarray | None" = None,
    arc_quantile: "float | None" = None,
    kind: str = "",
):
    """A cross-brain matrix as two live panels: the heatmap, and the same numbers as a circle.

    ::

      a 14 x 14 of ISC values -> [ matrix with every cell printed | connectogram ]

    The two panels answer different questions off one set of numbers, which is why both are
    on the page. **The matrix is the record**: every cell carries its value, blanks included,
    so a pairing can be looked up. **The circle is the shape**: an eye reads 196 printed
    numbers as a texture, and the chords say which sites the strong pairings actually land
    on.

    The coherence matrices are not drawn by this. They come in a pair, HbO against HbR, and
    that comparison is what the panel is for there, so :func:`build_wtc_cross_matrix` spends
    both halves of its width on the two heatmaps instead.

    Both rows and columns are the *montage*, row for the first member and column for the
    second, so cell (i, j) is one member's site i against the other's site j and the diagonal
    is the homologous pairing. There is no within-brain cell anywhere in it, which is why
    every chord on the circle crosses the middle.

    Live rather than a still, and the whole reason is the circle: a chord is drawn between
    two nodes and thirty of them cross, so on a PNG the only way to read one is to trace it
    to both ends and hope the labels are legible. Hovering says which pairing it is and what
    it is worth. The heatmap is live for the same reason, a channel montage printing two
    hundred cells that are easier to hover than to read.

    The scale is the caller's and both panels share it, so there is one colorbar: a
    correlation takes :data:`CORRELATION_SCALE` over -1 to 1.

    Which pairings get a chord is :func:`_arc_rule`. It changes no number, and the subtitle
    over the circle says which of its rules drew them.
    """
    z = np.asarray(z, dtype=float)
    if z.size == 0 or not len(row_labels):
        return None

    sub1 = subject_ids[0] if subject_ids else "Sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "Sub2"

    keep, rule = _arc_rule(z, value_label, arc_threshold, arc_level, arc_quantile)

    # both panels are squares centred in their half, so the space between them is whatever
    # the square constraint leaves over rather than a gap set here
    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                        horizontal_spacing=0.03,
                        subplot_titles=(matrix_title, rule))

    _matrix_panel(fig, z, row_labels, col_labels, subject_ids, cmap=cmap, vmin=vmin,
                  vmax=vmax, value_label=value_label, row=1, col=1,
                  colorbar=_COLORBAR | {"title": value_label})

    _circle_traces(fig, z, row_labels, col_labels, subject_ids, cmap=cmap, vmin=vmin,
                   vmax=vmax, value_label=value_label, keep=keep, row=1, col=2)

    axis_title = lambda who: f"{who} {kind}".strip()
    fig.update_layout(
        height=PANEL_HEIGHT, autosize=True,
        margin=dict(l=70, r=90, t=80, b=70),
        title=dict(text=suptitle, x=0.5, xanchor="center", y=0.975, font=dict(size=14)),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        # Ranges given rather than left to autorange, which pads a heatmap by a fraction of
        # a cell on every side and leaves a white margin around the grid. A cell spans half a
        # step either side of its index, so these two are the grid's own extent.
        xaxis=dict(title=axis_title(sub2), side="bottom", tickangle=-90,
                   range=[-0.5, len(col_labels) - 0.5],
                   showgrid=False, zeroline=False, constrain="domain"),
        # the first row at the top, which is how the table it stands for is read
        yaxis=dict(title=axis_title(sub1), range=[len(row_labels) - 0.5, -0.5],
                   showgrid=False, zeroline=False, scaleanchor="x", constrain="domain"),
        # The same span on both, so the constraint letterboxes the subplot instead of
        # stretching the circle into an ellipse. The span is what sets the circle's size in
        # the panel: the nodes sit at radius 1, so at 1.38 the circle used less than three
        # quarters of the height the matrix beside it used all of. The labels are drawn in
        # points rather than data units and do not shrink with it, so this is as close as the
        # longest of them can come to the panel edge.
        xaxis2=dict(visible=False, range=[-_CIRCLE_SPAN, _CIRCLE_SPAN], constrain="domain"),
        yaxis2=dict(visible=False, range=[-_CIRCLE_SPAN, _CIRCLE_SPAN], scaleanchor="x2",
                    scaleratio=1, constrain="domain"),
    )
    for note in fig.layout.annotations[:2]:
        note.font.size = 11
        note.font.color = "#444444"
    return fig


def build_wtc_cross_matrix(
    band_dfs: dict,
    labels: list[str],
    subject_ids: list[str],
    band_fmin: float,
    band_fmax: float,
    kind: str = "channel",
):
    """Band-mean coherence for every pairing across the two brains, one heatmap per chromophore.

    ::

      {"HbO": frame, "HbR": frame} -> [ HbO matrix | HbR matrix ], one colour scale

    Rows are sub1's sites, columns sub2's, so cell (i, j) is sub1's site i against sub2's
    site j and the diagonal is the homologous pairing the rest of the report shows. This is
    the whole point of ``--wtc-channel-cross``: the crossed pairs are computed and written to
    the TSV, and without this figure the only ones anybody looks at are the n on the diagonal.

    **The two chromophores share the figure and the scale.** HbO and HbR are two parallel
    passes, never mixed and never averaged, and what they are both run for is the check that
    a coupling shows in each; side by side on one colorbar is that check, where one above the
    other on two colorbars was two results a reader had to hold in their head. The panels
    were a heatmap and a connectogram of the same numbers until 2026-09-12, and the circle is
    the half that went: a coherence grid is small enough that the heatmap already carries its
    shape, and the second chromophore is the comparison worth the width. :func:`build_isc_panel`
    keeps its circle, having a null to rank the pairings by; the ROI ISC matrices have none
    and are laid out like this one.

    ``band_dfs`` maps a display name to that chromophore's band-mean frame. A frame that is
    None, carries no ``label2`` column, or has no finite cell is left out rather than drawn
    empty, so a run of one chromophore gets one panel. ``labels`` is the montage rather than
    whatever the frames happen to carry, so a dyad that lost a channel still gets a matrix of
    the same shape as one that did not, and the blanks say which sites went.

    Read cell by cell the off-diagonal is exploratory: single pairings are noisy and a
    correction over n**2 of them leaves little. The structure is what it is for.
    """
    sub1 = subject_ids[0] if subject_ids else "sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "sub2"

    panels: list[tuple[str, np.ndarray]] = []
    for name, band_df in (band_dfs or {}).items():
        if band_df is None or "label2" not in getattr(band_df, "columns", []):
            continue
        has_pair = {"sub1", "sub2"}.issubset(band_df.columns)
        # keyed by the pairing as well as the sites: the frame is long over every pairing
        # in the group, and a key of sites alone collapses them onto one cell, the last row
        # read winning, under axes naming only the first two members
        lookup = {(r.sub1, r.sub2, r.label, r.label2): r.coherence
                  for r in band_df.itertuples()} if has_pair else {
                  (r.label, r.label2): r.coherence for r in band_df.itertuples()}
        z = np.array([[lookup.get((sub1, sub2, row, col) if has_pair else (row, col), np.nan)
                       for col in labels] for row in labels], dtype=float)
        if np.isfinite(z).any():
            panels.append((str(name), z))
    if not panels:
        return None

    return _cross_matrix_figure(
        panels, labels, subject_ids,
        cmap=COHERENCE_SCALE, vmin=0, vmax=1, value_label="coherence",
        title=(f"Inter-brain coherence, {kind} × {kind}, "
               f"band {band_fmin:.3g}-{band_fmax:.3g} Hz  —  {sub1} × {sub2}"),
        kind=kind)


def _phase_arrows(ax, times: np.ndarray, freqs: np.ndarray, phase, n_time: int = 14,
                  n_freq: int = 10, scale: float = 22, width: float = 0.006,
                  mask=None) -> None:
    """Draw the relative-phase field over a coherence map, thinned to a readable grid.

    Right means in phase, left antiphase, and up means the row's subject leads by a quarter
    cycle. The map carries one phase per pixel, tens of thousands of them, so it is sampled
    onto a coarse grid; the arrows are directions and not magnitudes, so every one is the
    same length and drawing fewer loses nothing.

    ``scale`` and ``width`` are quiver's, in axes-relative units, so they have to be given
    per panel rather than fixed here: the defaults suit a thumbnail, and the same numbers on
    a full-width single map draw arrows tall enough to hide the map under them. Larger
    ``scale`` is shorter arrows.

    ``mask`` keeps only the grid points it marks True, which is how :func:`_arrow_mask` stops
    the field being drawn over noise and over padding. Sampled on the same coarse grid as the
    phase, so a point is kept on its own cell's verdict and not its neighbours'.
    """
    if phase is None:
        return
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (len(freqs), len(times)):
        return

    fi = np.unique(np.linspace(0, len(freqs) - 1, min(n_freq, len(freqs))).astype(int))
    ti = np.unique(np.linspace(0, len(times) - 1, min(n_time, len(times))).astype(int))
    grid_t, grid_f = np.meshgrid(times[ti], freqs[fi])
    angle = phase[np.ix_(fi, ti)]
    u, v = np.cos(angle), np.sin(angle)
    if mask is not None:
        keep = np.asarray(mask)[np.ix_(fi, ti)]
        if not keep.any():
            return
        # NaN is quiver's own way of skipping an arrow, and it keeps the grid rectangular
        u = np.where(keep, u, np.nan)
        v = np.where(keep, v, np.nan)
    ax.quiver(grid_t, grid_f, u, v,
              color="black", scale=scale, width=width, headwidth=4, headlength=5,
              pivot="mid", zorder=3)


def build_isc_panel(
    isc_mat: np.ndarray,
    ch_names: list[str],
    subject_ids: list[str],
    ch_type: str = "hbo",
    isc_threshold: "float | None" = None,
    arc_level: "np.ndarray | None" = None,
):
    """The 2-panel ISC summary: ISC matrix | connectogram.

    The panel itself is :func:`build_cross_panel`. What stays here is the scale a correlation
    is read on and the wording of its titles.

    Args:
        isc_mat:       n × n ISC matrix from compute_isc().
        ch_names:      Channel labels (n,), without type suffix.
        subject_ids:   [sub1_id, sub2_id, ...].
        ch_type:       "hbo" or "hbr", shown in titles.
        isc_threshold: Absolute ``|ISC|`` a pairing must clear for a chord. None, the
                       default, leaves the choice to :func:`_arc_rule`.
        arc_level:     Per-cell level out of the phase-scrambled null, when one was drawn.
    """
    if isc_mat is None or len(ch_names) == 0:
        return None

    sub1_label = subject_ids[0] if subject_ids else "Sub1"
    sub2_label = subject_ids[1] if len(subject_ids) > 1 else "Sub2"
    type_label = ch_type.upper()
    return build_cross_panel(
        np.asarray(isc_mat, dtype=float), list(ch_names), list(ch_names), subject_ids,
        cmap=CORRELATION_SCALE, vmin=-1, vmax=1, value_label="Pearson r",
        matrix_title=f"ISC matrix ({type_label})",
        suptitle=f"Inter-brain Synchrony ({type_label})  —  {sub1_label} × {sub2_label}",
        arc_threshold=isc_threshold,
        arc_level=arc_level,
        arc_quantile=ARC_FALLBACK_QUANTILE,
    )


def build_isc_roi_matrix(isc_by_chroma: dict, subject_ids: list[str]):
    """ISC at the ROI level, one heatmap per chromophore on a single correlation scale.

    ::

      {"HbO": (4 x 4 r, ["L", "R", ...]), "HbR": (...)} -> [ HbO matrix | HbR matrix ]

    The ROI counterpart of :func:`build_isc_panel`, and it drops the connectogram that one
    keeps. A handful of regions is small enough that the heatmap already carries the shape,
    and the ROI matrices have no phase-scrambled null, so a chord rule would have nothing to
    rank pairings by. Side by side on one scale is the HbO against HbR check instead, the
    reading :func:`build_wtc_cross_matrix` is laid out for.

    ``isc_by_chroma`` maps a display name to ``(matrix, roi labels)``. An entry that is
    None, has no finite cell, or does not match the axis the first panel set is left out
    rather than drawn empty, so a run of one chromophore gets one panel.
    """
    panels: list[tuple[str, np.ndarray]] = []
    labels: list[str] = []
    for name, got in (isc_by_chroma or {}).items():
        mat, roi_names = got or (None, None)
        if mat is None or not roi_names:
            continue
        z = np.asarray(mat, dtype=float)
        if not np.isfinite(z).any():
            continue
        labels = labels or list(roi_names)
        if z.shape != (len(labels), len(labels)):
            continue
        panels.append((str(name), z))
    if not panels:
        return None

    sub1 = subject_ids[0] if subject_ids else "sub1"
    sub2 = subject_ids[1] if len(subject_ids) > 1 else "sub2"
    return _cross_matrix_figure(
        panels, labels, subject_ids,
        cmap=CORRELATION_SCALE, vmin=-1, vmax=1, value_label="Pearson r",
        title=f"Inter-brain synchrony, ROI \u00d7 ROI  \u2014  {sub1} \u00d7 {sub2}",
        kind="ROI")
