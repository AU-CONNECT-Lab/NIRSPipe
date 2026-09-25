"""Shared color constants and helpers used across QC figure modules."""

import numpy as np
import mne
import plotly.graph_objects as go

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.utils")

HBO_COLOR      = "#e74c3c"
HBR_COLOR      = "#3498db"
HBO_MEAN_COLOR = "#c0392b"   # darker HbO for bold mean line
HBR_MEAN_COLOR = "#11629e"   # darker HbR for bold mean line

# By source-detector separation, the split every raw-level view uses.
LONG_COLOR  = "#f37d7d"
SHORT_COLOR = "#78b1f2"
# channels the long and short ranges both leave out; see channel_table._neither_range_title
UNCLASSIFIED_COLOR = "#95a5a6"

# Tick and axis text, set explicitly so the CLI and the interface render it the same colour.
AXIS_TEXT_COLOR = "#2a3f5f"

# Welch segment length for the single-stage PSD panels; the multi-stage figure keeps MNE's
# longer default, whose finer bin its sub-0.2 Hz cutoffs need.
PSD_NFFT = 256

# Qualitative palette for condition/trigger colors (cycled by index).
CONDITION_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


# A marker with no duration is an instant, and the bar drawn for it would be invisible;
# below this many seconds a block is drawn as a tick too, or it renders as a smear the
# reader cannot tell from a line.
_MIN_BLOCK_S = 1e-3


def timeline_row_traces(events, y, color, name, showlegend, hover_tail=""):
    """One row of an event timeline: a bar per timed block, a tick per instant event.

    ::

      timeline_row_traces([{"onset": 10, "duration": 30}], 1, "#e74c3c", "video", True)
      -> [Bar spanning 10-40 s at y=1]

    Both shapes can appear on one row, since a run can mix a timed block with an instant cue,
    so each event picks its own by its duration and the two traces share a legend entry.
    """
    blocks = [e for e in events if float(e.get("duration") or 0) > _MIN_BLOCK_S]
    ticks  = [e for e in events if float(e.get("duration") or 0) <= _MIN_BLOCK_S]
    traces = []
    if blocks:
        onsets = [float(e["onset"]) for e in blocks]
        spans  = [float(e["duration"]) for e in blocks]
        traces.append(go.Bar(
            x=spans, y=[y] * len(blocks), base=onsets, orientation="h", width=0.38,
            marker=dict(color=color, opacity=0.9, cornerradius=3, line=dict(width=0)),
            name=name, legendgroup=name, showlegend=showlegend,
            customdata=[[o, d, o + d] for o, d in zip(onsets, spans)],
            hovertemplate=(f"<b>{name}</b><br>onset: %{{customdata[0]:.2f}} s"
                           f"<br>duration: %{{customdata[1]:.2f}} s"
                           f"<br>ends: %{{customdata[2]:.2f}} s{hover_tail}<extra></extra>"),
        ))
    if ticks:
        traces.append(go.Scatter(
            x=[float(e["onset"]) for e in ticks], y=[y] * len(ticks), mode="markers",
            marker=dict(symbol="line-ns-open", size=14, color=color,
                        line=dict(width=1.6, color=color)),
            name=name, legendgroup=name, showlegend=showlegend and not blocks,
            hovertemplate=f"<b>{name}</b><br>t=%{{x:.2f}} s{hover_tail}<extra></extra>",
        ))
    return traces


# Shared styling for the event timelines; the alternating band ties a mark to its row label.
TIMELINE_BAND_COLOR = "#f7f8fa"
TIMELINE_ROW_PX = 36
# a block narrower than this share of the recording has no room for its length printed
# inside it; the number is still on hover
_LABEL_MIN_FRAC = 0.07


def timeline_row_bands(n_rows: int) -> list:
    """Alternating row backgrounds for an event timeline, as layout shapes."""
    return [dict(type="rect", xref="paper", yref="y", layer="below",
                 x0=0, x1=1, y0=i - 0.5, y1=i + 0.5,
                 fillcolor=TIMELINE_BAND_COLOR, line=dict(width=0))
            for i in range(n_rows) if i % 2]


def block_duration_labels(events, y, span: float) -> list:
    """The length printed on each block wide enough to hold it, as layout annotations.

    ::

      block_duration_labels([{"onset": 10, "duration": 300}], 1, 3900)
      -> [annotation "300 s" centred on the block at y=1]
    """
    out = []
    for e in events:
        dur = float(e.get("duration") or 0)
        if dur > _MIN_BLOCK_S and span > 0 and dur / span > _LABEL_MIN_FRAC:
            out.append(dict(x=float(e["onset"]) + dur / 2, y=y, text=f"{dur:g} s",
                            showarrow=False, font=dict(size=10, color="white")))
    return out


def timeline_axes(row_labels: "list[str] | None"):
    """The x and y axis dicts both event timelines share."""
    xaxis = dict(title="Time (s)", gridcolor="#edf0f3", zeroline=False, showline=True,
                 linecolor="#d5dbe1", ticks="outside", tickcolor="#d5dbe1", ticklen=4)
    n = len(row_labels or ())
    yaxis = dict(tickvals=list(range(n)), ticktext=list(row_labels or ()), showgrid=False,
                 zeroline=False, tickfont=dict(size=11), range=[n - 0.5, -0.5])
    return xaxis, yaxis


def decimate(arr: np.ndarray, times: np.ndarray, max_pts: int):
    """Uniformly subsample columns of arr (and times) to at most max_pts for display."""
    if len(times) <= max_pts:
        return arr, times
    step = max(1, len(times) // max_pts)
    return arr[:, ::step], times[::step]


def line_xy(times: np.ndarray, values: np.ndarray) -> dict:
    """Scatter x/y kwargs for a long trace, sized for the file it is written to.

    Two things, both invisible on screen::

        line_xy([0, .5, 1], [3, 4, 5])   -> {"x0": 0.0, "dx": 0.5, "y": array([3., 4., 5.])}
        line_xy([0, .5, 9], [3, 4, 5])   -> {"x": array([0., .5, 9.]), "y": array([3., 4., 5.])}

    Plotly writes a numpy array as base64 and a Python list as one full-precision decimal per
    element, so the arrays have to reach it as arrays: ``0.6881280000000001`` is eighteen
    characters for a number that occupies eight bytes. A uniformly sampled x is then not sent
    at all, ``x0``/``dx`` saying the same thing in two numbers.

    The uniformity test is what keeps this honest. ``decimate`` strides, so its timestamps
    pass; ``_maxpool_xy`` keeps the timestamp each bin's peak was found at, so a peak sits
    where it happened rather than on a bin edge, and those fail the test and keep their x.

    Values go out as float32, which is a display cast and not a measurement one. Timestamps
    stay float64, since those are what the uniformity test and the peak positions are read off.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=np.float32)
    if t.size > 2:
        d = np.diff(t)
        if np.allclose(d, d[0], rtol=1e-6, atol=0.0):
            return {"x0": float(t[0]), "dx": float(d[0]), "y": y}
    return {"x": t, "y": y}


def physio_bands(cardiac=None, resp=None):
    """PSD annotation bands as (name, f_lo, f_hi). Single source so every PSD figure agrees.

    Mayer is a fixed visual reference (no metric counterpart). ``cardiac`` / ``resp`` take
    the CLI ``(l_freq, h_freq)`` so the shading matches the bands the metrics integrate over;
    pass None to omit that band entirely rather than guess a default (e.g. the hyper pipeline
    has no respiration band).
    """
    bands = [("Mayer", 0.07, 0.13)]
    if resp is not None:
        bands.append(("Resp", *resp))
    if cardiac is not None:
        bands.append(("Cardiac", *cardiac))
    return bands


# colours for the physiological band annotations (the frequencies come from physio_bands)
BAND_COLORS = {
    "Mayer":   "rgba(46,204,113,0.12)",
    "Resp":    "rgba(241,196,15,0.10)",
    "Cardiac": "rgba(231,76,60,0.10)",
}


def add_band_shading(fig, fmax: float, bands: list, rows: "int | None" = None) -> None:
    """Shade and label the physiological bands on a PSD figure.

    One definition so the multi-stage PSD panel and the per-channel PSD mark the same bands
    the same way. ``rows`` is the subplot row count for a figure made with
    make_subplots, or None for a plain single-axis figure, which is the only difference
    between the two::

        add_band_shading(fig, 2.0, physio_bands(cardiac=(0.7, 1.5)))          # plain
        add_band_shading(fig, 2.0, physio_bands((0.7, 1.5), (0.2, 0.4)), 2)   # 2 subplots

    Bands starting past ``fmax`` are dropped rather than clamped: a stripe pinned to the
    right edge would claim the band is in view when it is off the plot.
    """
    targets = [{}] if rows is None else [{"row": row, "col": 1} for row in range(1, rows + 1)]
    for name, x0, x1 in bands:
        if x0 > fmax:
            continue
        for target in targets:
            fig.add_vrect(x0=x0, x1=min(x1, fmax),
                          fillcolor=BAND_COLORS.get(name, "rgba(120,120,120,0.10)"),
                          line_width=0, layer="below", **target)

    for name, x0, x1 in bands:
        label_x = (x0 + min(x1, fmax)) / 2
        if label_x > fmax:
            continue
        for i, _ in enumerate(targets):
            ax = "" if (rows is None or i == 0) else str(i + 1)
            fig.add_annotation(
                x=label_x, y=1.0,
                xref=f"x{ax}", yref=f"y{ax} domain",
                text=name, showarrow=False,
                font=dict(size=8, color="#555"),
                textangle=-90, xanchor="center", yanchor="top",
            )


def head_outline(ax, xs, ys):
    """Draw the head circle, ears and nose around a set of optode x/y, and set the limits.

    One definition so every flat map in the report has the same head. The circle is centred
    on the optode bounding box, not on the origin, because montage coordinates are in head
    space and a frontal-only cap sits well off centre::

        xs = [-0.05, 0.05], ys = [0.0, 0.08]  ->  centre (0, 0.04), radius 0.047

    Returns ``(cx, cy, r)`` so the caller can place anything else relative to the head.
    """
    import matplotlib.patches as mpatches

    cx =(max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    # 1.18 leaves the outermost optode just inside the scalp rather than on it
    r = max(max(abs(x - cx) for x in xs), max(abs(y - cy) for y in ys)) * 1.18

    ax.add_patch(mpatches.Circle((cx, cy), r, fill=False, edgecolor="#aaa", lw=1.5, zorder=0))
    nw, nh = r * 0.06, r * 0.10
    ax.fill([cx - nw, cx, cx + nw, cx - nw],
            [cy + r - nh * 0.3, cy + r + nh, cy + r - nh * 0.3, cy + r - nh * 0.3],
            color="#ddd", edgecolor="#aaa", lw=1, zorder=0)
    for ex in (cx - r, cx + r):
        ax.add_patch(mpatches.Ellipse((ex, cy), r * 0.14, r * 0.24,
                                      fc="#ddd", ec="#aaa", lw=1, zorder=0))

    # bounds relative to the head so the ears and nose are never clipped
    pad = r * 0.25
    ax.set_xlim(cx - r - r * 0.14 - pad, cx + r + r * 0.14 + pad)
    ax.set_ylim(cy - r - pad, cy + r + r * 0.12 + pad)
    return cx, cy, r


# ---- Epoching gate ----

def chunk_annotations(raw, chunk_duration: "float | None"):
    """A copy of ``raw`` with each long annotation cut into ``chunk_duration``-long trials.

    MNE's Epochs is a 3-D array, so every trial has to be the same length, and a block design
    marks one long annotation per block. Cutting the annotations rather than the events means
    every figure, the event timeline and the per-trial scoring all see the same trials.

    A 240 s block chunked at 25 s -> 9 annotations of 25 s, and the 15 s that do not fill a
    chunk are dropped, which is the rule MNE applies. Annotations already shorter than a
    chunk are left alone, and ``BAD_`` marks are censoring spans rather than stimuli and are
    never cut.

    ``None`` returns ``raw`` itself, so a caller can pass the option straight through.
    """
    if not chunk_duration or chunk_duration <= 0:
        return raw

    onsets, durations, descriptions = [], [], []
    for ann in raw.annotations:
        onset, duration = float(ann["onset"]), float(ann["duration"])
        desc = str(ann["description"])
        if desc.upper().startswith("BAD") or duration < 2 * chunk_duration:
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(desc)
            continue
        # np.arange from the onset, keeping only the pieces a whole chunk fits in
        for start in np.arange(onset, onset + duration, chunk_duration):
            if onset + duration - start < chunk_duration:
                break
            onsets.append(float(start))
            durations.append(float(chunk_duration))
            descriptions.append(desc)

    out = raw.copy()
    out.set_annotations(mne.Annotations(
        onsets, durations, descriptions,
        orig_time=raw.annotations.orig_time))
    logger.info("annotations chunked at %.1f s: %d -> %d",
                chunk_duration, len(raw.annotations), len(onsets))
    return out


# ---- Optode geometry ----
# Read off the channel locs and projected the way MNE flattens sensors, so an optode
# lands in the same frame as the channel midpoints.

def _topomap_project(xyz: np.ndarray, sphere: np.ndarray) -> np.ndarray:
    """Flatten 3-D points the way MNE flattens sensors for a topomap.

    Azimuthal-equidistant projection about the fitted head sphere: translate to the sphere
    origin, read (azimuth, polar) as (angle, radius), scale radians back to metres. MNE only
    exposes this for channel positions, so optodes are projected here with the same sphere and
    land in the same frame as the channel midpoints returned by ``_get_pos_outlines``.

    e.g. a point on the sphere equator, 90 deg from the vertex, maps onto the head circle.
    """
    from mne.transforms import _cart_to_sph, _pol_to_cart

    sph = _cart_to_sph(np.asarray(xyz, dtype=float) - sphere[:3])
    out = _pol_to_cart(sph[:, 1:][:, ::-1])
    out *= sph[:, [0]] / (np.pi / 2.0)
    return out + sphere[:2]


def _optode_positions(chs, ch_names) -> tuple[dict, dict, list[tuple[str, str]]]:
    """Named source / detector coordinates read off the fNIRS channel locs.

    A channel "S1_D2 760" carries its source in ``loc[3:6]`` and its detector in ``loc[6:9]``
    (``loc[:3]`` is the midpoint, not the detector), so it contributes {"S1": xyz} to the
    sources, {"D2": xyz} to the detectors and ("S1", "D2") to the pairs; repeats collapse.
    """
    src: dict[str, np.ndarray] = {}
    det: dict[str, np.ndarray] = {}
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ch, name in zip(chs, ch_names):
        pair = name.split(" ")[0]
        if "_" not in pair:
            continue
        s_name, d_name = pair.split("_")[:2]
        s_xyz, d_xyz = np.asarray(ch["loc"][3:6]), np.asarray(ch["loc"][6:9])
        if not (np.any(s_xyz) or np.any(d_xyz)):
            continue
        src.setdefault(s_name, s_xyz)
        det.setdefault(d_name, d_xyz)
        if (s_name, d_name) not in seen:
            seen.add((s_name, d_name))
            pairs.append((s_name, d_name))
    return src, det, pairs
