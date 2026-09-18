"""The flat head every figure that draws a value per channel is built on.

One projection, one outline, one glyph convention: a long channel is a bar of small discs
along its source-to-detector path, a short one a single larger disc inside a grey ring. A
reader who has learned one of these figures can read the next one, and two views cannot
drift into describing different heads.

Built on :mod:`fnirs_pipe.qc.figures.common.topomap`'s projection, which is MNE's own, so
these heads and a real topomap put a channel in the same place. What varies between callers
is the colour scale and which channels have a value; the geometry never does.
"""

from __future__ import annotations

import mne
import numpy as np
import plotly.graph_objects as go

from fnirs_pipe.qc.figures.common.topomap import _LONG_SIZE, _SHORT_SIZE
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.head_map")

# a channel the caller has no value for. Grey rather than an end of the colour scale, which
# would read as a measured extreme instead of as an absence
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
    from fnirs_pipe.qc.figures.common.topomap import _glyph_points, _projected_optodes

    got = _projected_optodes(raw.info)
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
        out[scope] = {"gx": gx, "gy": gy, "labels": labels, "per_pair": per_pair,
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


def _marker(g: dict, keep, size: float, short: bool, **marker) -> go.Scatter:
    """One Scatter over the markers ``keep`` selects, with the ring short channels carry."""
    gx, gy = np.asarray(g["gx"]), np.asarray(g["gy"])
    labels = g["labels"]
    if keep is not None:
        gx, gy = gx[keep], gy[keep]
        labels = [lab for lab, k in zip(labels, keep) if k]
    return go.Scatter(
        x=gx, y=gy, mode="markers", text=labels,
        # a grey ring, which reads as a separate object against both the head and the bars
        # while leaving the fill on the shared colour scale. White disappears into the page
        # and near-black fights the fill for attention.
        marker=dict(size=size, symbol="circle",
                    line=dict(width=1.6 if short else 0, color="#98a4ae"), **marker),
        showlegend=False)


def head_glyph(fig, geo, scope, values, row, col, title, cmin, cmax, bar, colorscale,
               sid: str = "", dim=None, blank_color: "str | None" = None,
               fmt: str = ".3f") -> int:
    """One scope's channels drawn on the head, coloured by ``values``.

    ``values`` carries one number per channel in ``geo[scope]["names"]`` order, and every
    marker of a channel takes that channel's colour. Long and short stay on one colour
    scale, so a short channel reads as a contamination check rather than as a second map;
    the **shape** is what separates them, which is why one figure can carry both.

    ``bar`` is True for the trace that shows the colour bar, or a dict of colorbar settings
    where a grid needs one bar per panel rather than one for the whole figure.

    Two optional states, both drawn under the coloured markers so a value is never hidden by
    a channel that has none:

    - ``blank_color`` draws the channels whose value is not finite in that flat colour. A
      caller with no number for a channel then gets a grey channel rather than a gap, which
      is what keeps the montage readable as a montage.
    - ``dim`` names the channels to draw at reduced opacity. They keep their colour, because
      they do carry a real value; what the dimming means is the caller's to say.

    Without either, every marker goes into one trace, non-finite ones included, so a caller
    that updates colours frame by frame can hand back an array of the same length it passed.

    Returns the index of the coloured trace, which is what such a frame updates.
    """
    g = geo[scope]
    short = scope == "short"
    size = _SHORT_SIZE if short else _LONG_SIZE
    marker_values = _expand(values, g["per_pair"])

    def _coloured(keep, opacity, show_bar) -> int:
        colorbar = dict(title=dict(text=title, side="right", font=dict(size=10)),
                        thickness=12, len=0.72, tickfont=dict(size=9))
        if isinstance(show_bar, dict):
            colorbar.update(show_bar)
        fig.add_trace(_marker(
            g, keep, size, short, opacity=opacity,
            color=(marker_values if keep is None else marker_values[keep]).astype(np.float32),
            colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=bool(show_bar),
            colorbar=colorbar,
        ).update(
            # the member in every bubble: a grid of heads is read by pointing at one, and
            # the row label is off at the edge by then
            hovertemplate=(f"<b>{sid}</b><br>" if sid else "")
                          + "%{text}<br>%{marker.color:" + fmt + "}<extra></extra>",
        ), row=row, col=col)
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
                    _marker(g, keep, size, short, color=blank_color, opacity=opacity)
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
        for xy in geo["outlines"].values():
            xs += list(np.asarray(xy[0]))
            ys += list(np.asarray(xy[1]))
    pad = 0.04 * ((max(xs) - min(xs)) or 1.0)
    xr, yr = [min(xs) - pad, max(xs) + pad], [min(ys) - pad, max(ys) + pad]
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            n = (r - 1) * n_cols + c
            fig.update_xaxes(visible=False, range=xr, row=r, col=c)
            # the report renders figures responsive, so a wider container would stretch the
            # head into an ellipse; the anchor keeps it round and spends the slack as margin.
            # The axis stays visible with everything stripped rather than `visible=False`,
            # which would take the title with it, and the title is what names the row.
            fig.update_yaxes(range=yr, row=r, col=c, showticklabels=False, showgrid=False,
                             zeroline=False, showline=False, ticks="",
                             scaleanchor="x" if n == 1 else f"x{n}", scaleratio=1)
    if row_labels:
        for r, label in enumerate(row_labels, start=1):
            fig.update_yaxes(title_text=label, title_font=dict(size=11, color="#34495e"),
                             row=r, col=1)
