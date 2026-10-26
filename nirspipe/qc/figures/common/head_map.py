"""The flat head every figure that draws a value per channel is built on.

One projection, one outline, two marks. Which mark is right depends on what the number is:

- a **bar** along the source-to-detector path, for a value that is a claim about that path,
  such as a correlation between a seed and this channel. A short channel, too stubby to read
  as a path, becomes a single larger disc inside a grey ring.
- a **disc** at the channel's midpoint, for a value that is a property of one place, such as
  an amplitude. Bars are wrong for those: they meet end to end wherever two channels share
  an optode, so a montage of independent measurements reads as one connected polyline.

Two views built on it cannot drift into describing different heads.

Built on :mod:`nirspipe.qc.figures.common.topomap`'s projection, which is MNE's own, so
these heads and a real topomap put a channel in the same place. What varies between callers
is the colour scale and which channels have a value; the geometry never does.
"""

from __future__ import annotations

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nirspipe.qc.figures.common.topomap import (
    _LONG_SIZE, _SHORT_SIZE, _glyph_points, _projected_optodes,
)

# one disc per channel, drawn larger than the discs a bar is strung from
_DISC_SIZE = 15
from nirspipe.qc.metrics.windowed import SCREEN_WINDOW_S
from nirspipe.utils.logging import get_logger

logger = get_logger("qc.figures.head_map")

# a channel the caller has no value for; grey, since a scale end would read as a real extreme
BLANK_COLOR = "#cfd6dc"
DIM_OPACITY = 0.3

_OUTLINE_KEYS = ("head", "nose", "ear_left", "ear_right")


def head_geometry(raw: mne.io.Raw, pairs: list[str]) -> "dict | None":
    """Glyph coordinates and the head outline for one recording, keyed by separation.

    ``pairs`` names the long source-detector pairs as ``"S1_D1"``; everything else the
    montage carries is short. A caller that wants only the long channels drawn passes them
    here and then draws the ``"long"`` scope alone.

    None when the montage has no usable optode positions.
    """
    got = _projected_optodes(raw)
    if got is None:
        return None
    opt_xy, all_pairs, outlines = got
    long_set = set(pairs)
    out = {"outlines": {k: outlines[k] for k in _OUTLINE_KEYS if k in outlines}}
    for scope, sel in (("long", [(a, b) for a, b in all_pairs if f"{a}_{b}" in long_set]),
                       ("short", [(a, b) for a, b in all_pairs
                                  if f"{a}_{b}" not in long_set])):
        if not sel:
            continue
        gx, gy, labels, per_pair = _glyph_points(sel, opt_xy, short=(scope == "short"))
        mid = np.array([[(opt_xy[a][0] + opt_xy[b][0]) / 2,
                         (opt_xy[a][1] + opt_xy[b][1]) / 2] for a, b in sel])
        out[scope] = {"gx": gx, "gy": gy, "labels": labels, "per_pair": per_pair,
                      "mid_x": mid[:, 0], "mid_y": mid[:, 1],
                      "names": [f"{a}_{b}" for a, b in sel],
                      "skel": [(opt_xy[a][:2], opt_xy[b][:2]) for a, b in sel]}
    return out if ("long" in out or "short" in out) else None


def head_ground(fig, geo: dict, row: int, col: int) -> None:
    """The outline and the bare source-detector skeleton, under the glyphs."""
    for key in _OUTLINE_KEYS:
        xy = geo["outlines"].get(key)
        if xy is not None:
            fig.add_trace(go.Scatter(x=np.asarray(xy[0]), y=np.asarray(xy[1]), mode="lines",
                                     line=dict(color="#c9d2da", width=1.2),
                                     hoverinfo="skip", showlegend=False), row=row, col=col)
    sx, sy = [], []
    for scope in ("long", "short"):
        for a, b in geo.get(scope, {}).get("skel", []):
            sx += [a[0], b[0], None]
            sy += [a[1], b[1], None]
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color="#ececec", width=1),
                             hoverinfo="skip", showlegend=False), row=row, col=col)


def _expand(values, per_pair) -> np.ndarray:
    """One entry per marker from one entry per channel."""
    return np.repeat(np.asarray(values, dtype=float), per_pair)


def _marker(g: dict, keep, size: float, short: bool, disc: bool = False,
            **marker) -> go.Scatter:
    """One Scatter over the markers ``keep`` selects, with whatever ring its mark carries."""
    ring = 1.2 if disc else (1.6 if short else 0)
    gx, gy = np.asarray(g["gx"]), np.asarray(g["gy"])
    labels = g["labels"]
    if keep is not None:
        gx, gy = gx[keep], gy[keep]
        labels = [lab for lab, k in zip(labels, keep) if k]
    return go.Scatter(
        x=gx, y=gy, mode="markers", text=labels,
        # grey ring on short-channel markers, white on discs so neighbours do not merge
        marker=dict(size=size, symbol="circle",
                    line=dict(width=ring, color="white" if ring and disc else "#98a4ae"),
                    **marker),
        showlegend=False)


def head_glyph(fig, geo, scope, values, row, col, title, cmin, cmax, bar, colorscale,
               sid: str = "", dim=None, blank_color: "str | None" = None,
               fmt: str = ".3f", mark: str = "bar", hover_tail: str = "",
               customdata=None) -> int:
    """One scope's channels drawn on the head, coloured by ``values``.

    ``values`` carries one number per channel in ``geo[scope]["names"]`` order, and every
    marker of a channel takes that channel's colour.

    ``mark`` is "bar" or "disc"; see this module's own docstring for which a number wants.
    With bars, long and short stay on one colour scale and the **shape** separates them.

    ``customdata`` gives one extra value per channel and ``hover_tail`` a template fragment
    to print it with, for a figure whose colour is a derived number and whose reader still
    wants the measured one.

    ``bar`` is True for the trace that shows the colour bar, or a dict of colorbar settings
    where a grid needs one bar per panel rather than one for the whole figure.

    Two optional states, both drawn under the coloured markers so a value is never hidden by
    a channel that has none:

    - ``blank_color`` draws the channels whose value is not finite in that flat colour. A
      caller with no number for a channel then gets a grey channel rather than a gap.
    - ``dim`` names the channels to draw at reduced opacity. They keep their colour, because
      they do carry a real value; what the dimming means is the caller's to say.

    Without either, every marker goes into one trace, non-finite ones included, so a caller
    that updates colours frame by frame can hand back an array of the same length it passed.

    Returns the index of the coloured trace, which is what such a frame updates.
    """
    g = geo[scope]
    disc = mark == "disc"
    if disc:
        # one marker per channel, so the per-channel arrays need no expanding at all
        g = {**g, "gx": g["mid_x"], "gy": g["mid_y"], "labels": g["names"],
             "per_pair": np.ones(len(g["names"]), dtype=int)}
    short = scope == "short"
    size = _DISC_SIZE if disc else (_SHORT_SIZE if short else _LONG_SIZE)
    marker_values = _expand(values, g["per_pair"])
    extra = None if customdata is None else _expand(customdata, g["per_pair"])

    def _coloured(keep, opacity, show_bar) -> int:
        colorbar = dict(title=dict(text=title, side="right", font=dict(size=10)),
                        thickness=12, len=0.72, tickfont=dict(size=9))
        if isinstance(show_bar, dict):
            # title is nested, so a plain update would drop the text and leave a bar with
            # only its side set
            over = dict(show_bar)
            colorbar["title"] = {**colorbar["title"], **over.pop("title", {})}
            colorbar.update(over)
        trace = _marker(
            g, keep, size, short, disc, opacity=opacity,
            color=(marker_values if keep is None else marker_values[keep]).astype(np.float32),
            colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=bool(show_bar),
            colorbar=colorbar,
        ).update(
            # the member in every bubble, since the row label is off at the edge
            hovertemplate=(f"<b>{sid}</b><br>" if sid else "")
                          + "%{text}<br>%{marker.color:" + fmt + "}" + hover_tail
                          + "<extra></extra>",
        )
        if extra is not None:
            trace.update(customdata=extra if keep is None else extra[keep])
        fig.add_trace(trace, row=row, col=col)
        return len(fig.data) - 1

    if blank_color is None and not dim:
        return _coloured(None, 1.0, bar)

    finite = np.isfinite(marker_values)
    dimmed = np.zeros(len(marker_values), dtype=bool)
    if dim:
        dim_set = set(dim)
        dimmed = _expand([n in dim_set for n in g["names"]], g["per_pair"]).astype(bool)

    if blank_color is not None:
        for keep, opacity in ((~finite & dimmed, DIM_OPACITY), (~finite & ~dimmed, 1.0)):
            if keep.any():
                fig.add_trace(
                    _marker(g, keep, size, short, disc, color=blank_color, opacity=opacity)
                    .update(hovertemplate="%{text}<br>no value<extra></extra>"),
                    row=row, col=col)

    index = -1
    for keep, opacity, show_bar in ((finite & dimmed, DIM_OPACITY, False),
                                    (finite & ~dimmed, 1.0, bar)):
        if keep.any():
            got = _coloured(keep, opacity, show_bar)
            index = got if opacity == 1.0 else index
    return index


def head_axes(fig, geo_by_sub: dict, n_rows: int, n_cols: int,
              row_labels: "list[str] | None" = None) -> None:
    """One square, unlabelled panel per head, all sharing the widest outline's bounds."""
    xs, ys = [], []
    for geo in geo_by_sub.values():
        # a montage drawn without a head is bounded by its own optodes
        drawn = list(geo["outlines"].values()) or [
            (geo[s]["gx"], geo[s]["gy"]) for s in ("long", "short") if s in geo]
        for xy in drawn:
            xs += list(np.asarray(xy[0]))
            ys += list(np.asarray(xy[1]))
    pad = 0.04 * ((max(xs) - min(xs)) or 1.0)
    xr, yr = [min(xs) - pad, max(xs) + pad], [min(ys) - pad, max(ys) + pad]
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            n = (r - 1) * n_cols + c
            fig.update_xaxes(visible=False, range=xr, row=r, col=c)
            # the anchor keeps the head round in a responsive container; stripped rather than
            # `visible=False`, which would also drop the title that names the row
            fig.update_yaxes(range=yr, row=r, col=c, showticklabels=False, showgrid=False,
                             zeroline=False, showline=False, ticks="",
                             scaleanchor="x" if n == 1 else f"x{n}", scaleratio=1)
    if row_labels:
        for r, label in enumerate(row_labels, start=1):
            fig.update_yaxes(title_text=label, title_font=dict(size=11, color="#34495e"),
                             row=r, col=1)


# ---- The slider: a recording's coupling on the head, window by window ----

# the channel grids' fail, near and pass colours, so a head reads like the grid beside it
HEAD_SCALE = [[0.0, "#F8786E"], [0.5, "#FFD966"], [1.0, "#C5E0B3"]]


def pair_values(geo, scope, by_pair: dict, reduce_fn) -> "np.ndarray":
    """One value per channel this head draws, NaN where ``by_pair`` has no such pair."""
    return np.array([reduce_fn(by_pair[name]) if name in by_pair else np.nan
                     for name in geo[scope]["names"]])


def _blank_twin(fig, i: int, sid: str, row: int, col: int) -> int:
    """A grey copy of marker trace ``i`` for the markers with no value; returns its index.

    The slider swaps one colour array per frame, so the markers with no value cannot be a
    trace of their own: both traces keep every marker and :func:`_split_sizes` hides each
    marker in the trace it does not belong to.
    """
    src = fig.data[i]
    fig.add_trace(go.Scatter(
        x=src.x, y=src.y, mode="markers", text=src.text, showlegend=False,
        marker=dict(color=BLANK_COLOR, symbol=src.marker.symbol, line=dict(width=0)),
        hovertemplate=f"<b>{sid}</b><br>%{{text}}<br>no value<extra></extra>",
    ), row=row, col=col)
    return len(fig.data) - 1


def _split_sizes(values: np.ndarray, size: float) -> "tuple[np.ndarray, np.ndarray]":
    """Marker sizes for a coloured trace and its grey twin, so each marker shows in one."""
    finite = np.isfinite(values)
    return np.where(finite, size, 0.0), np.where(finite, 0.0, size)


def _passing_words(values: np.ndarray, per_pair) -> np.ndarray:
    """``yes`` or ``no`` per marker, the hover of a row whose colour is a pass or a fail."""
    return np.repeat(np.where(np.asarray(values) >= 0.5, "yes", "no"), per_pair)


def _slider_rows(grid: dict, series_by_sub: dict, subject_ids: list[str],
                 cutoffs: dict) -> list[tuple]:
    """The slider's numbers, left to right: (label, colour bar title, {sid: {pair: series}},
    cmin, cmax, colour scale).

    Passing is the mask itself; SCI and PSP are scaled with their line at mid-scale, the rule
    the subject report's SCI/PSP strip follows. PSP with no line spans its own values on one
    hue, as that strip draws it; SCI with no line, or a number with no data, is left out.
    """
    rows = []
    ok = grid.get("ok") or {}
    passing = {sid: dict(zip(grid["pairs"], np.asarray(ok[sid], dtype=float)))
               for sid in subject_ids if sid in ok}
    if passing:
        rows.append(("Passing", "Passing", passing, 0.0, 1.0, HEAD_SCALE))
    window = f"<br>({SCREEN_WINDOW_S:g} s)"
    for label, line in (("SCI", cutoffs.get("sci")), ("PSP", cutoffs.get("psp"))):
        key = f"per_pair_{label.lower()}"
        values = {sid: series_by_sub[sid][key] for sid in subject_ids
                  if (series_by_sub.get(sid) or {}).get(key) is not None}
        if not values or (line is None and label == "SCI"):
            continue
        if line is None:
            cmin, cmax = _series_range(values)
            rows.append((label, label + window, values, cmin, cmax, "Viridis"))
            continue
        line = float(line)
        cmin, cmax = (2 * line - 1.0, 1.0) if label == "SCI" else (0.0, 2 * line)
        rows.append((label, label + window, values, cmin, cmax, HEAD_SCALE))
    return rows


def _series_range(values: dict) -> "tuple[float | None, float | None]":
    """Smallest and largest finite value over ``{sid: {pair: series}}``."""
    found = [np.asarray(s, dtype=float) for by_pair in values.values()
             for s in by_pair.values()]
    finite = np.concatenate([a[np.isfinite(a)] for a in found]) if found else np.array([])
    return (float(finite.min()), float(finite.max())) if finite.size else (None, None)


def build_head_slider(
    geo_by_sub: dict,
    subject_ids: list[str],
    grid: dict,
    series_by_sub: dict,
    conditions: "dict | list | None" = None,
    cutoffs: "dict[str, float] | None" = None,
    step: int = 3,
) -> "go.Figure | None":
    """A row of heads per member, or the one subject's, the slider running the whole recording.

    The instantaneous view, in columns read left to right: whether the window passed, the
    mask the screening counts, then SCI and PSP, the numbers that build it, each on a scale
    centred on its line in ``cutoffs`` where it has one. Every frame names the block it lands in, so a position on the
    slider says what was being done and not only when. ``conditions`` is ``{name: (t0, t1)}``,
    or a list of ``(name, (t0, t1))`` where a condition recurs.

    ``step`` decimates the frames. Each one carries a colour per marker per head, so the page
    grows with the frame count.
    """
    if not geo_by_sub:
        return None
    t = np.asarray(grid["t"], dtype=float)
    frames_at = list(range(0, len(t), max(1, step)))
    if not frames_at:
        return None
    spans = list(conditions.items()) if isinstance(conditions, dict) else list(conditions or [])
    rows = _slider_rows(grid, series_by_sub, subject_ids, cutoffs or {})
    if not rows:
        return None

    def task_at(when: float) -> str:
        for name, (a, b) in spans:
            if a <= when <= b:
                return name
        return "between blocks"

    def caption(k: int) -> dict:
        """The block the slider sits in, as the figure's title.

        A frame that hands back ``layout.annotations`` overwrites the layout's own array
        **by index**, so a one-element caption lands on subplot title 0 and the real caption,
        sitting at index 2, never updates: the first head loses its name and the caption is
        frozen at the first frame. A title has no index to collide with.
        """
        # a run without events has no blocks to be between
        where = f"<b>{task_at(t[k])}</b> &nbsp;" if spans else ""
        return dict(text=f"{where}<span style='color:#8a949e'>t = {t[k]:.0f} s</span>",
                    x=0.5, xanchor="center", y=0.98, yanchor="top",
                    font=dict(size=13, color="#34495e"))

    # members in rows, as the per-block heads above put them, so one subject is one row
    n_rows, n_cols = len(subject_ids), len(rows)
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[row[0] for row in rows] + [""] * (n_cols * (n_rows - 1)),
                        horizontal_spacing=0.07, vertical_spacing=0.05)
    drawn: list[tuple] = []
    for mi, (label, title, values, cmin, cmax, scale) in enumerate(rows, start=1):
        # one colour bar per number, right of its column and as tall as the members; the
        # column title already names it
        x_dom = fig.get_subplot(1, mi).xaxis.domain
        top = fig.get_subplot(1, mi).yaxis.domain[1]
        bottom = fig.get_subplot(n_rows, mi).yaxis.domain[0]
        bar = dict(x=x_dom[1] + 0.004, xanchor="left", y=(top + bottom) / 2, yanchor="middle",
                   len=0.8 * (top - bottom), title=dict(text=""))
        if label == "Passing":
            bar.update(tickvals=[0, 1], ticktext=["no", "yes"])
        for ri, sid in enumerate(subject_ids, start=1):
            geo = geo_by_sub.get(sid)
            if geo is None or sid not in values:
                continue
            head_ground(fig, geo, ri, mi)
            for scope in ("long", "short"):
                if scope not in geo:
                    continue
                per_pair = geo[scope]["per_pair"]
                vals = pair_values(geo, scope, values[sid], lambda v, k=frames_at[0]: v[k])
                i = head_glyph(fig, geo, scope, vals, ri, mi, title, cmin, cmax,
                                bar=bar if (ri == n_rows and scope == "long") else False,
                                colorscale=scale, sid=sid)
                if label == "Passing":
                    fig.data[i].update(
                        customdata=_passing_words(vals, per_pair),
                        hovertemplate=f"<b>{sid}</b><br>%{{text}}<br>passing: "
                                      "%{customdata}<extra></extra>")
                j = _blank_twin(fig, i, sid, ri, mi)
                size = float(fig.data[i].marker.size)
                shown, blank = _split_sizes(np.repeat(vals, per_pair), size)
                fig.data[i].marker.size, fig.data[j].marker.size = shown, blank
                drawn.append((mi, sid, scope, i, j, size))
    if not drawn:
        return None

    def frame_data(k: int) -> list:
        data = []
        for mi, sid, scope, _i, _j, size in drawn:
            label, values = rows[mi - 1][0], rows[mi - 1][2]
            per_pair = geo_by_sub[sid][scope]["per_pair"]
            per_channel = pair_values(geo_by_sub[sid], scope, values[sid],
                                       lambda v, kk=k: v[kk])
            markers = np.repeat(per_channel, per_pair)
            shown, blank = _split_sizes(markers, size)
            coloured = go.Scatter(marker=dict(color=markers.astype(np.float32), size=shown))
            if label == "Passing":
                coloured.customdata = _passing_words(per_channel, per_pair)
            data += [coloured, go.Scatter(marker=dict(size=blank))]
        return data

    fig.frames = [
        go.Frame(name=f"{t[k]:.0f}", data=frame_data(k),
                 traces=[x for *_r, i, j, _z in drawn for x in (i, j)],
                 layout=dict(title=caption(k)))
        for k in frames_at
    ]
    head_axes(fig, geo_by_sub, n_rows, n_cols, row_labels=list(subject_ids))
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(
        height=280 * n_rows + 154, plot_bgcolor="white", showlegend=False,
        # the left margin holds the row names, as on the per-block heads
        margin=dict(l=84, r=78, t=78, b=76),
        title=caption(frames_at[0]),
        sliders=[dict(active=0, y=0, yanchor="top", pad=dict(t=30, b=4),
                      currentvalue=dict(visible=False), font=dict(size=9),
                      steps=[dict(method="animate", label=fr.name,
                                  args=[[fr.name],
                                        dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))])
                             for fr in fig.frames])])
    return fig
