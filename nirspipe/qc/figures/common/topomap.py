"""Channel-wise scalp maps of the evoked response, drawn as coloured source-detector paths.

Each channel is painted along the path between its own source and detector, so the map shows
what was measured and nothing else. An interpolated map (MNE's ``plot_topomap``) would fill
the scalp between channels with a field the optodes did not measure.

Short channels get their own row on the same colour scale as the long ones, so a scalp
response shows as a short row as strongly coloured as the long row above it.
"""

from __future__ import annotations

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nirspipe.qc.metrics._helpers import epochable_events
from nirspipe.utils import is_marker
from nirspipe.utils.logging import get_logger
from nirspipe.qc.figures.common._utils import _optode_positions, _topomap_project

logger = get_logger("qc.figures")

# ---- Glyph geometry ----
_SAMPLES     = 26      # markers laid along a long channel to read as a continuous bar
_TRIM        = 0.16    # fraction of the path left bare at each end, so optodes stay visible
# marker sizes in pixels, set for the smallest head this grid draws; the dyad heads share them
_LONG_SIZE   = 7
_SHORT_SIZE  = 12      # a short channel is too stubby to read as a path; draw one disc

# ---- Time axis ----
_FRAME_STEP  = 1.0
_FRAME_STOP  = 20.0    # past the canonical response; frames beyond this carry nothing
_OPEN_AT     = 6.0     # canonical HbO peak, so the figure opens on the informative frame

# ---- Colour ----
# a percentile, not the max, so one stray condition cannot wash the rest out to white
_SCALE_PCT   = 99.5
# below this a condition is single-trial noise: drawn, but not consulted for the range
_SCALE_MIN_TRIALS = 3


def _condition_evokeds(
    raw_haemo: mne.io.Raw, epoch_tmin: float, epoch_tmax: float,
) -> "dict[str, mne.Evoked]":
    """Condition -> evoked, epoched on the (non-BAD) annotations. Empty dict if there are none."""
    if not any(is_marker(a["description"]) for a in raw_haemo.annotations):
        return {}
    events, event_id = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
    if len(events) == 0:
        return {}
    epochs = mne.Epochs(
        raw_haemo, events, event_id, tmin=epoch_tmin, tmax=epoch_tmax,
        baseline=(epoch_tmin, 0), preload=True, verbose=False,
    )
    out = {}
    for cond in event_id:
        try:
            out[str(cond)] = epochs[cond].average()
        except Exception:
            continue
    return out


def _projected_optodes(info: mne.Info) -> "tuple[dict, list, dict] | None":
    """Optode 2-D coordinates, channel pairs and the head outline, in MNE's topomap frame.

    Uses the same azimuthal projection and fitted sphere MNE uses for a real topomap, so this
    figure and the optode layout describe the same head. Falls back to raw x/y when the
    montage carries no sphere, which loses the head outline but keeps the relative geometry.

    e.g. a two-pair montage yields ``({"S1": (x, y), "D1": ..., "D2": ...},
    [("S1", "D1"), ("S1", "D2")], {"head": (xs, ys), ...})``.
    """
    picks = list(mne.pick_types(info, meg=False, fnirs=True, exclude=[]))
    if not picks:
        return None
    chs   = [info["chs"][i] for i in picks]
    names = [info["ch_names"][i] for i in picks]

    outlines = {}
    project = None
    try:
        from mne.utils import _check_sphere
        from mne.viz.topomap import _get_pos_outlines
        sphere = _check_sphere(None, info)
        _, outlines = _get_pos_outlines(info, picks, sphere)
        project = lambda xyz: _topomap_project(xyz, sphere)  # noqa: E731
    except Exception as exc:
        logger.warning("head projection unavailable, plotting flat x/y: %s", exc)
        project = lambda xyz: np.asarray(xyz, dtype=float)[:, :2]  # noqa: E731

    src, det, pairs = _optode_positions(chs, names)
    opt_xy: dict = {}
    for group in (src, det):
        if group:
            opt_xy.update(zip(group, project(np.array(list(group.values())))))
    pairs = [(s, d) for s, d in pairs if s in opt_xy and d in opt_xy]
    if not pairs:
        return None
    return opt_xy, pairs, outlines


def _glyph_points(pairs, opt_xy, short: bool):
    """Marker coordinates for one row's glyphs, plus the pair label each marker belongs to.

    A long channel becomes ``_SAMPLES`` markers strung along the source-detector path; a short
    one becomes a single disc at its midpoint. Both return flat arrays so a whole row is one
    Plotly trace and a frame updates it by swapping one colour array.
    """
    xs, ys, labels, per_pair = [], [], [], []
    for s, d in pairs:
        a, b = opt_xy[s], opt_xy[d]
        if short:
            f = np.array([0.5])
        else:
            f = np.linspace(_TRIM, 1.0 - _TRIM, _SAMPLES)
        xs += list(a[0] + (b[0] - a[0]) * f)
        ys += list(a[1] + (b[1] - a[1]) * f)
        labels += [f"{s}_{d}"] * len(f)
        per_pair.append(len(f))
    return np.array(xs), np.array(ys), labels, per_pair


def _pair_values(evoked, pairs, per_pair, chromo: str, t: float) -> np.ndarray:
    """Each pair's value at time ``t``, repeated to match its marker count. NaN when absent."""
    i = int(np.argmin(np.abs(evoked.times - t)))
    out = []
    for (s, d), n in zip(pairs, per_pair):
        name = f"{s}_{d} {chromo}"
        v = (float(evoked.data[evoked.ch_names.index(name), i]) * 1e6
             if name in evoked.ch_names else np.nan)
        out += [v] * n
    return np.array(out)


def evoked_channel_map_figure(
    raw_haemo: mne.io.Raw,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    sep_bands=None,
) -> "go.Figure | None":
    """Evoked response per channel, one head per condition, with a time slider.

    Rows are chromophore x separation (HbO long, HbO short, HbR long, HbR short), columns are
    conditions, and the slider steps the whole grid through the epoch window a second at a
    time. Long and short share a colour scale within a chromophore.

    ``sep_bands`` is this run's separations from :func:`separation_bands`; every caller in one
    run has to pass the same value or the rows would describe a different montage than the
    regression used. None takes the package defaults.

    None when the run has no events, no usable optode positions, or no channel in either
    separation band.
    """
    from nirspipe.qc.metrics import long_short_channels

    try:
        evokeds = _condition_evokeds(raw_haemo, epoch_tmin, epoch_tmax)
    except Exception as exc:
        logger.warning("evoked channel map epoching failed: %s", exc)
        return None
    if not evokeds:
        return None

    geom = _projected_optodes(raw_haemo.info)
    if geom is None:
        return None
    opt_xy, all_pairs, outlines = geom

    long_names, short_names = long_short_channels(raw_haemo, sep_bands)
    scoped = {}
    for scope, names in (("long", set(long_names)), ("short", set(short_names))):
        kept = [(s, d) for s, d in all_pairs
                if any(f"{s}_{d} {c}" in names for c in ("hbo", "hbr"))]
        if kept:
            scoped[scope] = kept
    if not scoped:
        logger.warning("no channel falls in either separation band; skipping channel map")
        return None

    chromos = [c for c in ("hbo", "hbr")
               if len(mne.pick_types(raw_haemo.info, fnirs=c, exclude=[]))]
    rows = [(c, scope) for c in chromos for scope in ("long", "short") if scope in scoped]
    conds = list(evokeds)
    if not rows or not conds:
        return None

    lo, hi = next(iter(evokeds.values())).times[[0, -1]]
    times = np.arange(0.0, min(_FRAME_STOP, float(hi)) + 1e-9, _FRAME_STEP)
    times = times[(times >= lo) & (times <= hi)]
    if not len(times):
        times = np.array([float(np.clip(0.0, lo, hi))])
    open_at = int(np.argmin(np.abs(times - _OPEN_AT)))

    geometry = {
        scope: (_glyph_points(pairs, opt_xy, scope == "short"), pairs)
        for scope, pairs in scoped.items()
    }

    def row_values(chromo, scope, cond, t):
        (_, _, _, per_pair), pairs = geometry[scope]
        return _pair_values(evokeds[cond], pairs, per_pair, chromo, t)

    # one scale per chromophore, spanning both separations
    # conditions under _SCALE_MIN_TRIALS are drawn but left out of the range, as the grand mean
    n_trials = {c: int(getattr(evokeds[c], "nave", 0) or 0) for c in conds}
    scaling = [c for c in conds if n_trials[c] >= _SCALE_MIN_TRIALS] or list(conds)

    vlim = {}
    for chromo in chromos:
        vals = np.concatenate([row_values(chromo, scope, cond, t)
                               for scope in scoped
                               for cond in scaling for t in times
                               ] or [np.array([0.0])])
        v = float(np.nanpercentile(np.abs(vals), _SCALE_PCT)) if np.any(np.isfinite(vals)) else 0.0
        vlim[chromo] = v or 1.0

    # each short panel's peak as a share of the long peak, over the whole window so the
    # verdict does not change as the slider moves
    verdicts = {}
    for chromo, scope in rows:
        if scope != "short" or "long" not in scoped:
            continue
        for cond in conds:
            peak_s = _window_peak(row_values, chromo, "short", cond, times)
            peak_l = _window_peak(row_values, chromo, "long", cond, times)
            if peak_l and peak_s is not None:
                verdicts[(chromo, cond)] = peak_s / peak_l

    return _assemble(rows, conds, times, open_at, geometry, outlines, opt_xy,
                     row_values, vlim, chromos, verdicts, scaling)


def _window_peak(row_values, chromo, scope, cond, times) -> "float | None":
    """Largest absolute value this row reaches anywhere in the window, None if all NaN."""
    vals = np.concatenate([np.abs(row_values(chromo, scope, cond, t)) for t in times])
    return None if not np.any(np.isfinite(vals)) else float(np.nanmax(vals))


# ---- Layout ----
_MARGIN   = dict(l=56, r=86, t=46, b=70)
_H_SPACE  = 0.02
_V_SPACE  = 0.05
# the canvas follows the panel, not the other way round, so few columns never mean huge heads
_FIG_W_MAX   = 1100
_PANEL_W_MAX = 200


def _outline_traces(fig, outlines, row, col):
    for key in ("head", "nose", "ear_left", "ear_right"):
        if key not in outlines:
            continue
        ox, oy = outlines[key]
        fig.add_trace(go.Scatter(x=list(ox), y=list(oy), mode="lines",
                                 line=dict(color="#c8c8c8", width=1.2),
                                 hoverinfo="skip", showlegend=False), row=row, col=col)


def _skeleton(pairs, opt_xy):
    """Grey source-detector lines under the glyphs, so a channel with no value still shows."""
    xs, ys = [], []
    for s, d in pairs:
        xs += [opt_xy[s][0], opt_xy[d][0], None]
        ys += [opt_xy[s][1], opt_xy[d][1], None]
    return xs, ys


def _assemble(rows, conds, times, open_at, geometry, outlines, opt_xy,
              row_values, vlim, chromos, verdicts, scaling) -> go.Figure:
    """Build the subplot grid, the frames and the slider from the per-row value function."""
    n_rows, n_cols = len(rows), len(conds)
    titles = [c if c in scaling or len(scaling) == len(conds)
              else f"{c}<br><span style='font-size:9px'>off the shared scale</span>"
              for c in conds]
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=titles + [""] * (n_cols * (n_rows - 1)),
                        horizontal_spacing=_H_SPACE, vertical_spacing=_V_SPACE)

    if outlines:
        ox = np.concatenate([outlines[k][0] for k in
                             ("head", "nose", "ear_left", "ear_right") if k in outlines])
        oy = np.concatenate([outlines[k][1] for k in
                             ("head", "nose", "ear_left", "ear_right") if k in outlines])
    else:
        ox = np.array([p[0] for p in opt_xy.values()])
        oy = np.array([p[1] for p in opt_xy.values()])
    pad = 0.03 * float(ox.max() - ox.min() or 1.0)
    xr = [float(ox.min()) - pad, float(ox.max()) + pad]
    yr = [float(oy.min()) - pad, float(oy.max()) + pad]

    glyph_idx = []
    for ri, (chromo, scope) in enumerate(rows, start=1):
        (gx, gy, labels, _), pairs = geometry[scope]
        skel_x, skel_y = _skeleton(pairs, opt_xy)
        size = _SHORT_SIZE if scope == "short" else _LONG_SIZE
        axis = "coloraxis" if chromo == chromos[0] else "coloraxis2"
        for ci, cond in enumerate(conds, start=1):
            _outline_traces(fig, outlines, ri, ci)
            fig.add_trace(go.Scatter(x=skel_x, y=skel_y, mode="lines",
                                     line=dict(color="#ececec", width=1),
                                     hoverinfo="skip", showlegend=False), row=ri, col=ci)
            fig.add_trace(go.Scatter(
                x=gx, y=gy, mode="markers", text=labels,
                marker=dict(size=size, coloraxis=axis, line=dict(width=0),
                            color=row_values(chromo, scope, cond, times[open_at])),
                hovertemplate="%{text}<br>%{marker.color:.2f} µmol/L<extra></extra>",
                showlegend=False), row=ri, col=ci)
            glyph_idx.append(len(fig.data) - 1)
            fig.update_xaxes(visible=False, range=xr, row=ri, col=ci)
            # the report renders figures responsive, so a wider container would stretch the
            # head into an ellipse; the anchor keeps it round and spends the slack as margin
            n = (ri - 1) * n_cols + ci
            fig.update_yaxes(visible=False, range=yr, row=ri, col=ci,
                             scaleanchor="x" if n == 1 else f"x{n}", scaleratio=1)

    fig.frames = [
        go.Frame(name=f"{t:.0f}",
                 data=[go.Scatter(marker=dict(color=row_values(chromo, scope, cond, t)))
                       for chromo, scope in rows for cond in conds],
                 traces=glyph_idx)
        for t in times
    ]

    # the canvas is sized so a panel lands at the head's own aspect; scaleanchor would honour
    # the aspect too but leaves the slack as dead space inside each panel
    spread = 1 - _H_SPACE * (n_cols - 1)
    panel_w = min(_PANEL_W_MAX,
                  (_FIG_W_MAX - _MARGIN["l"] - _MARGIN["r"]) * spread / n_cols)
    panel_h = panel_w * (yr[1] - yr[0]) / (xr[1] - xr[0])
    fig_w = round(panel_w * n_cols / spread + _MARGIN["l"] + _MARGIN["r"])
    fig_h = round(panel_h * n_rows / (1 - _V_SPACE * max(n_rows - 1, 1)) +
                  _MARGIN["t"] + _MARGIN["b"])

    coloraxes = {}
    for i, chromo in enumerate(chromos):
        span = [ri for ri, (c, _) in enumerate(rows) if c == chromo]
        if not span:
            continue
        top    = fig.layout[_yaxis_key(span[0], n_cols)].domain[1]
        bottom = fig.layout[_yaxis_key(span[-1], n_cols)].domain[0]
        key = "coloraxis" if i == 0 else f"coloraxis{i + 1}"
        coloraxes[key] = dict(
            colorscale="RdBu_r", cmin=-vlim[chromo], cmax=vlim[chromo],
            colorbar=dict(title=f"{chromo.upper()}<br>µmol/L", x=1.01,
                          y=(top + bottom) / 2, len=max(top - bottom, 0.12),
                          thickness=12, tickfont=dict(size=9)),
        )

    fig.update_layout(
        height=fig_h, width=fig_w, plot_bgcolor="white", paper_bgcolor="white",
        margin=_MARGIN, **coloraxes,
        sliders=[dict(active=open_at, x=0.06, len=0.88, y=-0.02, pad=dict(t=6),
                      currentvalue=dict(prefix="t = ", suffix=" s from onset",
                                        font=dict(size=12)),
                      steps=[dict(method="animate", label=f"{t:.0f}",
                                  args=[[f"{t:.0f}"],
                                        dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))])
                             for t in times])],
    )
    for ri, (chromo, scope) in enumerate(rows):
        dom = fig.layout[_yaxis_key(ri, n_cols)].domain
        fig.add_annotation(x=0, xref="paper", y=(dom[0] + dom[1]) / 2, yref="paper",
                           text=f"<b>{chromo.upper()}</b><br>{scope}", showarrow=False,
                           xanchor="right", xshift=-8, font=dict(size=11), align="right")
        for ci, cond in enumerate(conds):
            share = verdicts.get((chromo, cond)) if scope == "short" else None
            if share is None:
                continue
            xdom = fig.layout[_xaxis_key(ri, ci, n_cols)].domain
            fig.add_annotation(
                x=(xdom[0] + xdom[1]) / 2, xref="paper", y=dom[1], yref="paper",
                text=f"scalp {share:.0%} of brain peak", showarrow=False,
                # an auto anchor is left or right for a column off the middle, not centre
                xanchor="center", yanchor="bottom", yshift=2,
                font=dict(size=10, color="#c0392b" if share >= 0.5 else "#7f8c8d"))
    fig.for_each_annotation(
        lambda a: a.update(font=dict(size=12)) if a.text in conds else None)
    return fig


def _xaxis_key(row_idx: int, col_idx: int, n_cols: int) -> str:
    """Layout key of the x axis at (row, col), both 0-based: xaxis, xaxis2, ..."""
    n = row_idx * n_cols + col_idx + 1
    return "xaxis" if n == 1 else f"xaxis{n}"


def _yaxis_key(row_idx: int, n_cols: int) -> str:
    """Layout key of the first column's y axis on row ``row_idx`` (0-based): yaxis, yaxis4, ..."""
    n = row_idx * n_cols + 1
    return "yaxis" if n == 1 else f"yaxis{n}"
