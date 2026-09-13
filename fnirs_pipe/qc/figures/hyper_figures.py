"""Plotly figure builders for the hyperscanning group-level raw QC report."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.figure_io import extract_markers as _extract_markers
from fnirs_pipe.qc.figures._utils import (CONDITION_PALETTE, PSD_NFFT,
                                          TIMELINE_ROW_PX,
                                          decimate as _decimate, physio_bands, timeline_axes,
                                          timeline_row_bands, timeline_row_traces)
# Imported rather than restated: these heads are the subject report's channel map with a
# different quantity on them, and a reader who learned one reads the other. Two copies of
# the pair would let one report's bars thicken while the other's stayed put.
from fnirs_pipe.qc.figures.topomap import (
    _LONG_SIZE as _HEAD_SIZE, _SHORT_SIZE as _HEAD_SHORT_SIZE,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.hyper")

_MAX_TS_PTS = 4000

_SUB_COLORS   = ["#8e44ad", "#e67e22", "#16a085", "#f39c12",
                  "#2c3e50", "#1abc9c", "#c0392b", "#34495e"]
_COND_PALETTE = CONDITION_PALETTE
_COND_DASHES  = ["solid", "dash", "dot", "dashdot", "longdash"]

_GOOD_COLOR = "#C5E0B3"
_MIX_COLOR  = "#FFD966"
_NA_COLOR   = "#D3D3D3"
_BAD_COLOR  = "#F8786E"

# colors for the physiological band annotations (frequencies come from physio_bands)
_PSD_BAND_COLORS = {
    "Mayer":   "rgba(52,152,219,0.10)",
    "Resp":    "rgba(39,174,96,0.08)",
    "Cardiac": "rgba(231,76,60,0.08)",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _cond_colors(descriptions: list[str]) -> dict[str, str]:
    return {d: _COND_PALETTE[i % len(_COND_PALETTE)] for i, d in enumerate(descriptions)}


def _psd_band_shapes(cardiac=None) -> tuple[list[dict], list[dict]]:
    # resp omitted: the hyper raw pipeline has no respiration band input
    bands = physio_bands(cardiac=cardiac, resp=None)
    shapes = [dict(type="rect", xref="x", yref="paper",
                   x0=x0, x1=x1, y0=0, y1=1,
                   fillcolor=_PSD_BAND_COLORS.get(name, "rgba(120,120,120,0.08)"),
                   line=dict(width=0)) for name, x0, x1 in bands]
    annots = [dict(x=(x0 + x1) / 2, y=0.97, xref="x", yref="paper",
                   text=name, showarrow=False,
                   font=dict(size=8, color="#666")) for name, x0, x1 in bands]
    return shapes, annots


def sci_of(sqm_data: dict, sid: str) -> dict:
    """The per-channel SCI a dyad page prints, for one member.

    The **windowed** estimate, which is the one this project reads: a drift shared by both
    wavelengths lifts the whole-run number, and the two can disagree about which channel
    coupled better. The whole-run scores stand in only for a record written before the
    windowed pass existed, so an old tree degrades to the number it has rather than to an
    empty grid.
    """
    member = sqm_data.get(sid) or {}
    return member.get("sci_win_per_channel") or member.get("sci_per_channel") or {}


def _rejected_pairs(sqm_data: dict, sid: str) -> "set[str] | None":
    """The S-D pairs one member's screening rejected, or None when nothing recorded it.

    ::

      ["S1_D1 760", "S1_D1 850"] -> {"S1_D1"}

    Rejection is stored per wavelength, since that is what prep marks on the recording, and
    every lookup on a dyad page is per pair. The empty set and None are different answers:
    a member that lost no channel has an empty list in its record, a member whose record
    was never written has no list at all, and only the second is unknown.
    """
    bad = sqm_data.get(sid, {}).get("bad_channels")
    if bad is None:
        return None
    return {str(ch).rsplit(" ", 1)[0] for ch in bad}


def _ch_kept_by_member(
    ch_pair: str,
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> list[bool | None]:
    """Whether each member kept this channel pair: the screening's verdict, per member.

    ::

      -> [True, False]   # kept by the first member, rejected by the second

    **The verdict, not a threshold on SCI.** These three colours are labelled good / mixed /
    bad on every panel that draws them, and "good" has one meaning in this package: the
    channel survived screening. Screening is `good_frac`, the share of 10 s windows in which
    SCI and PSP both cleared their lines, so a channel can be dropped at an SCI of 0.96 and
    kept at a lower one. Colouring by ``sci_threshold`` instead, which is what this did, put
    a green channel under a rejected one and called both "good".

    ``sci_threshold`` stays the fallback and nothing else: a record with no rejection list at
    all is the one case with no verdict to draw, and an all-grey montage says less than the
    number that record does carry. The subject report makes the same split and shows both,
    rejection first and SCI as the grading underneath it.
    """
    result: list[bool | None] = []
    for sid in subject_ids:
        rejected = _rejected_pairs(sqm_data, sid)
        if rejected is not None:
            result.append(ch_pair not in rejected)
            continue
        sci_d = sci_of(sqm_data, sid)
        val = sci_d.get(f"{ch_pair} hbo") or sci_d.get(ch_pair)
        result.append(None if val is None else float(val) >= sci_threshold)
    return result


# ---------------------------------------------------------------------------
# Shared data layer: the screening grid, on the dyad's clock
# ---------------------------------------------------------------------------

def coupled_grid(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    offsets: dict[str, float],
    duration_s: "float | None" = None,
) -> "dict | None":
    """The screening verdict per long pair per window, for every member, on one clock.

    ::

      -> {"t": [...], "pairs": ["S1_D1", ..., "S1_D8", ...],
          "long_pairs": ["S1_D1", ...], "ok": {"sub-01": mask, "sub-02": mask}}

    with each ``mask`` a ``pair x window`` boolean, True where that member's SCI and PSP both
    cleared their lines in that window. The three dyad panels that read it (the usable-time
    carpet, the two head figures) then cannot disagree about which window was good.

    **Every pair, with the long ones named separately.** The carpet is a long-channel picture
    because the dyad measures run on long channels, but a head draws the whole montage, and a
    grid holding only the long set handed the short markers a NaN each: eight discs with no
    colour and no meaning on it.

    Two things this does that a per-panel version kept getting wrong. The stored matrices are
    on each member's **own** clock, so the window centres are shifted by that member's crop
    offset, and a dyad with unequal offsets would otherwise compare window *k* of one against
    window *k* of the other. And they cover the **whole** recording while the dyad exists only
    on the aligned span, so windows outside it are dropped rather than drawn past the ends of
    the shared clock.

    ``sqm_data`` carries ``screen_windows`` whichever command produced it: the dyad raw pass
    keeps the grid it screened by, and a record read from disk is re-masked at that run's own
    lines. Neither is re-measured here, so the shading and the verdict are one measurement.

    None when a member has no grid, or when the members were not screened on the same one.
    """
    grids, masks = {}, {}
    for sid in subject_ids:
        member = sqm_data.get(sid) or {}
        grid = member.get("screen_windows") or {}
        mask, centers = grid.get("mask"), grid.get("centers")
        if mask is None or centers is None or not len(centers):
            logger.warning("no screening grid for %s; the dyad grid is empty", sid)
            return None
        grids[sid] = np.asarray(centers, dtype=float) - float(offsets.get(sid, 0.0))
        masks[sid] = np.asarray(mask, dtype=bool)

    ref = grids[subject_ids[0]]
    for sid, t in grids.items():
        if t.shape != ref.shape or not np.allclose(t, ref, atol=1.0):
            logger.warning("%s was screened on a different window grid; skipping the dyad "
                           "grid rather than comparing windows that are not the same window",
                           sid)
            return None

    keep = ref >= 0
    if duration_s is not None:
        keep &= ref <= float(duration_s)

    pairs: list[str] = []
    long_pairs: list[str] = []
    rows = {sid: [] for sid in subject_ids}
    for sid in subject_ids:
        member = sqm_data.get(sid) or {}
        order = list((member.get("screen_windows") or {}).get("channel_order") or [])
        long_names = {k.rsplit(" ", 1)[0]
                      for k in (member.get("per_channel_long") or {})
                      .get("sci_per_channel", {})}
        if not order:
            return None
        by_pair: dict[str, list[int]] = {}
        for i, name in enumerate(order):
            by_pair.setdefault(name.rsplit(" ", 1)[0], []).append(i)
        if not pairs:
            pairs = list(by_pair)
            long_pairs = [p for p in pairs if p in long_names] or list(pairs)
        for pair in pairs:
            idx = by_pair.get(pair)
            # a pair one member lacks is not usable by the dyad at any moment
            rows[sid].append(masks[sid][idx].all(axis=0) if idx
                             else np.zeros(masks[sid].shape[1], dtype=bool))

    return {"t": ref[keep],
            "pairs": pairs,
            "long_pairs": long_pairs,
            "ok": {sid: np.array(rows[sid])[:, keep] for sid in subject_ids}}


def dyad_status(grid: dict, subject_ids: list[str]) -> "np.ndarray":
    """``pair x window``: 2 coupled in every member, 1 in some, 0 in none."""
    stack = np.array([grid["ok"][sid] for sid in subject_ids])
    return np.where(stack.all(axis=0), 2, np.where(stack.any(axis=0), 1, 0))


def member_series(sqm_data: dict, sid: str, grid: dict, offset: float) -> dict:
    """One member's long-channel means per window, on the dyad's clock.

    ::

      -> {"t": (388,), "sci": (388,), "psp": (388,), "gvtd": (388,),
          "per_pair_sci": {"S1_D1": (388,), ...}}

    The three series are averaged over the same rows the carpet folds into pairs, off the
    same matrices its mask came from, so a dip in a line and a hole under it are one
    measurement and not two. GVTD is absent unless the member's record carried it; see
    :func:`coupled_grid`.

    ``per_pair_sci`` keeps every pair separately, short ones included, because the head
    figures colour one marker per channel and draw both separations.
    """
    sw = (sqm_data.get(sid) or {}).get("screen_windows") or {}
    order = list(sw.get("channel_order") or [])
    if not order or sw.get("centers") is None:
        return {}
    long_pairs = set(grid.get("long_pairs") or grid["pairs"])
    rows = [i for i, name in enumerate(order) if name.rsplit(" ", 1)[0] in long_pairs]
    t = np.asarray(sw["centers"], dtype=float) - float(offset)
    keep = np.isin(np.round(t, 2), np.round(np.asarray(grid["t"], dtype=float), 2))
    out = {"t": t[keep]}
    for key in ("sci", "psp"):
        m = sw.get(key)
        if m is not None and rows:
            out[key] = np.asarray(m, dtype=float)[rows][:, keep].mean(axis=0)
    gvtd = sw.get("gvtd")
    if gvtd is not None and len(gvtd) == len(t):
        out["gvtd"] = np.asarray(gvtd, dtype=float)[keep]

    sci_m = sw.get("sci")
    if sci_m is not None:
        sci_m = np.asarray(sci_m, dtype=float)[:, keep]
        by_pair: dict[str, list[int]] = {}
        for i, name in enumerate(order):
            by_pair.setdefault(name.rsplit(" ", 1)[0], []).append(i)
        # a pair is one SCI, stored once per wavelength; the mean over its rows is that
        # number and not an average of two different ones
        out["per_pair_sci"] = {pair: sci_m[idx].mean(axis=0)
                               for pair, idx in by_pair.items()}
    return out


# ---------------------------------------------------------------------------
# Figure: the conditions, before and after alignment
# ---------------------------------------------------------------------------

def build_alignment_timeline(
    raw_raws: "dict[str, mne.io.Raw] | None",
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
) -> "go.Figure | None":
    """Every member's annotated blocks, on its own clock above and the shared one below.

    Alignment is a claim about where the blocks sit, so the blocks are the picture and the
    millisecond residual is a number for the table beside it. The second row is what every
    later panel reads; a block that does not line up there is a trigger that landed late in
    one member, or two machines whose rates pulled them apart across the run.

    One row per member in each panel. None when nothing is annotated.
    """
    rows = [("Each recording on its own clock", raw_raws),
            ("After alignment, on the shared clock", aligned_raws)]
    rows = [(title, raws) for title, raws in rows if raws]
    if not rows:
        return None

    descs = list(dict.fromkeys(
        b["desc"] for _title, raws in rows for sid in subject_ids
        if sid in raws for b in _blocks(raws[sid])))
    if not descs:
        return None
    colours = _cond_colors(descs)

    fig = make_subplots(rows=len(rows), cols=1, vertical_spacing=0.22,
                        subplot_titles=[t for t, _r in rows])
    seen: set[str] = set()
    for r, (_title, raws) in enumerate(rows, start=1):
        for yi, sid in enumerate(subject_ids):
            for b in _blocks(raws.get(sid)):
                show = b["desc"] not in seen
                seen.add(b["desc"])
                fig.add_trace(go.Bar(
                    x=[max(b["duration"], 1.0)], y=[yi], base=[b["onset"]],
                    orientation="h", width=0.42,
                    marker=dict(color=colours[b["desc"]], opacity=0.9, cornerradius=3,
                                line=dict(width=0)),
                    name=b["desc"], legendgroup=b["desc"], showlegend=show,
                    hovertemplate=(f"<b>{b['desc']}</b><br>{sid}<br>"
                                   "onset %{base:.2f} s<br>"
                                   f"duration {b['duration']:.2f} s<extra></extra>"),
                ), row=r, col=1)
        fig.update_yaxes(tickvals=list(range(len(subject_ids))), ticktext=subject_ids,
                         range=[len(subject_ids) - 0.4, -0.6], showgrid=False,
                         tickfont=dict(size=10), row=r, col=1)
        fig.update_xaxes(gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9),
                         row=r, col=1)
    fig.update_xaxes(title_text="Time (s)", title_font=dict(size=10), row=len(rows), col=1)
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=170 * len(rows), plot_bgcolor="white", barmode="overlay",
                      margin=dict(l=96, r=24, t=70, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


def _blocks(raw: "mne.io.Raw | None") -> list[dict]:
    """Annotated blocks on the recording's own zero, BAD spans dropped.

    ``crop`` moves ``first_samp`` and leaves annotation onsets on the original clock, so the
    shared-clock time is ``onset - first_time``. Without that subtraction an aligned timeline
    is drawn identical to the unaligned one, which is the one way this figure can look right
    and be wrong.
    """
    if raw is None:
        return []
    t0 = float(raw.first_time)
    return [{"desc": str(a["description"]), "onset": float(a["onset"]) - t0,
             "duration": float(a["duration"] or 0.0)}
            for a in raw.annotations
            if not str(a["description"]).upper().startswith("BAD")]


# ---------------------------------------------------------------------------
# Figure: shared usable time
# ---------------------------------------------------------------------------

_SERIES_ROWS = (("sci", "SCI (10 s)", "sci"), ("psp", "PSP (10 s)", "psp"),
                ("gvtd", "GVTD", None))
_LEAD_COLOURS = ("#3498db", "#e67e22", "#16a085", "#8e44ad")


def build_usable_time(
    grid: dict,
    subject_ids: list[str],
    series: dict[str, dict],
    conditions: "dict[str, tuple[float, float]] | None" = None,
    cutoffs: "dict[str, float] | None" = None,
) -> "go.Figure | None":
    """Which pairs the dyad could use, when, and what went wrong where it could not.

    A channel is usable by the dyad only while it is coupled in **both** members at the same
    moment, so the dyad's usable time is the intersection of the two and not the smaller of
    them: two members can each keep 13 of 14 pairs and share only 12, if the pair they each
    lose is a different one. That intersection is the carpet at the bottom.

    Over it, each member's long-channel means for the metrics the mask is made of, so the
    panel says why a pair went and not only that it did. A condition bar names the blocks and
    dotted rules carry their edges down through the series.

    None when the grid holds no pair.
    """
    pairs = list(grid.get("long_pairs") or grid["pairs"])
    if not pairs:
        return None
    t = np.asarray(grid["t"], dtype=float)
    # the carpet is a long-channel picture: every dyad measure runs on long channels, and a
    # short row here would be read as coverage the analysis could have used
    keep = [i for i, name in enumerate(grid["pairs"]) if name in set(pairs)]
    status = dyad_status(grid, subject_ids)[keep]
    lost = (status != 2).mean(axis=1)
    order = np.argsort(lost)[::-1]
    labels = [f"{pairs[i]}   {lost[i] * 100:.0f}%" if lost[i] else pairs[i] for i in order]

    have = [(key, label) for key, label, _need in _SERIES_ROWS
            if any(key in (series.get(sid) or {}) for sid in subject_ids)]
    conditions = conditions or {}
    bar_rows = 1 if conditions else 0
    n_rows = bar_rows + len(have) + 1
    bar_h, series_h, carpet_h = 34, 86, 26 * len(pairs)
    total = bar_h * bar_rows + series_h * len(have) + carpet_h
    heights = ([bar_h / total] * bar_rows + [series_h / total] * len(have)
               + [carpet_h / total])
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=heights)

    if bar_rows:
        colours = _cond_colors(list(conditions))
        for name, (a, b) in conditions.items():
            fig.add_trace(go.Bar(
                x=[b - a], y=[0], base=[a], orientation="h", width=0.55,
                marker=dict(color=colours[name], opacity=0.9, cornerradius=3,
                            line=dict(width=0)),
                name=name, legendgroup=name, showlegend=True,
                hovertemplate=f"<b>{name}</b><br>%{{base:.0f}} to {b:.0f} s<extra></extra>",
            ), row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[-0.5, 0.5],
                         row=1, col=1)

    lines = dict(zip(subject_ids, _LEAD_COLOURS))
    for r, (key, label) in enumerate(have, start=bar_rows + 1):
        for sid in subject_ids:
            s = series.get(sid) or {}
            if key not in s:
                continue
            fig.add_trace(go.Scatter(
                x=np.asarray(s["t"]), y=np.asarray(s[key]), mode="lines", name=sid,
                legendgroup=sid, showlegend=(r == bar_rows + 1),
                line=dict(color=lines[sid], width=1.4), opacity=0.9,
                hovertemplate=f"t=%{{x:.0f}}s<br>{label} %{{y:.4g}}<extra></extra>",
            ), row=r, col=1)
        line = (cutoffs or {}).get(key)
        if line is not None:
            fig.add_hline(y=float(line), line_color="#adb5bd", line_width=1,
                          line_dash="dot", row=r, col=1)
        fig.update_yaxes(title_text=label, title_font=dict(size=9.5), row=r, col=1)

    fig.add_trace(go.Heatmap(
        z=status[order], x=t, y=labels,
        colorscale=[[0, _BAD_COLOR], [0.33, _BAD_COLOR], [0.33, _MIX_COLOR],
                    [0.66, _MIX_COLOR], [0.66, _GOOD_COLOR], [1, _GOOD_COLOR]],
        zmin=0, zmax=2, showscale=False, ygap=2,
        hovertemplate="<b>%{y}</b><br>t=%{x:.0f}s<br>%{z:.0f} member(s) coupled<extra></extra>",
    ), row=n_rows, col=1)
    # the key a discrete heatmap cannot carry itself
    for colour, lab in ((_GOOD_COLOR, "coupled in both"), (_MIX_COLOR, "one member only"),
                        (_BAD_COLOR, "neither")):
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name=lab,
                             marker=dict(color=colour, line=dict(width=0)),
                             showlegend=True, hoverinfo="skip"), row=n_rows, col=1)

    for a, b in conditions.values():
        for edge in (a, b):
            fig.add_vline(x=edge, line_color="#e3e8ec", line_width=1, line_dash="dot")

    fig.update_xaxes(gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=9.5),
                     row=n_rows, col=1)
    fig.update_xaxes(title_text="Time on the shared clock (s)", title_font=dict(size=10),
                     row=n_rows, col=1)
    fig.update_layout(height=96 + total, plot_bgcolor="white", barmode="overlay",
                      margin=dict(l=112, r=24, t=48, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


# ---------------------------------------------------------------------------
# Figure: where on the head
# ---------------------------------------------------------------------------

_HEAD_SCALE = [[0.0, _BAD_COLOR], [0.5, _MIX_COLOR], [1.0, _GOOD_COLOR]]


def head_geometry(raw: mne.io.Raw, pairs: list[str]) -> "dict | None":
    """Glyph coordinates and the head outline for one member, keyed by separation.

    Reuses the subject report's channel map projection, so a dyad head and a subject head are
    the same head and a reader who learned one can read the other. ``pairs`` names the long
    S-D pairs; everything else the montage carries is short.

    None when the montage has no usable optode positions.
    """
    from fnirs_pipe.qc.figures.topomap import _glyph_points, _projected_optodes

    got = _projected_optodes(raw.info)
    if got is None:
        return None
    opt_xy, all_pairs, outlines = got
    long_set = set(pairs)
    out = {"outlines": {k: outlines[k] for k in
                        ("head", "nose", "ear_left", "ear_right") if k in outlines}}
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


def _head_ground(fig, geo: dict, row: int, col: int) -> None:
    """The outline and the bare source-detector skeleton, under the glyphs."""
    for key in ("head", "nose", "ear_left", "ear_right"):
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


def _head_glyph(fig, geo, scope, values, row, col, title, cmin, cmax, bar) -> int:
    """One trace per separation, every channel's markers carried in one colour array.

    Long and short keep one colour scale, as the subject report's channel map does, so a
    short channel reads as a contamination check rather than a second map. That report can
    give them a row of their own; one head cannot, so the **shape** carries the distinction:
    a long channel is a bar of small discs along its path, a short one a single larger disc
    inside a dark ring.
    """
    g = geo[scope]
    short = scope == "short"
    fig.add_trace(go.Scatter(
        x=g["gx"], y=g["gy"], mode="markers", text=g["labels"],
        marker=dict(size=_HEAD_SHORT_SIZE if short else _HEAD_SIZE,
                    symbol="circle",
                    color=np.repeat(values, g["per_pair"]).astype(np.float32),
                    colorscale=_HEAD_SCALE, cmin=cmin, cmax=cmax, showscale=bar,
                    # a dark ring, which reads as a separate object against both the head
                    # and the bars while leaving the fill on the shared colour scale. White
                    # was tried and disappears into the page.
                    line=dict(width=1.6 if short else 0, color="#34495e"),
                    colorbar=dict(title=dict(text=title, side="right", font=dict(size=10)),
                                  thickness=12, len=0.72, tickfont=dict(size=9))),
        hovertemplate="%{text}<br>%{marker.color:.3f}<extra></extra>",
        showlegend=False), row=row, col=col)
    return len(fig.data) - 1


def _head_axes(fig, geo_by_sub: dict, n_rows: int, n_cols: int) -> None:
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
            # head into an ellipse; the anchor keeps it round and spends the slack as margin
            fig.update_yaxes(visible=False, range=yr, row=r, col=c,
                             scaleanchor="x" if n == 1 else f"x{n}", scaleratio=1)


def _pair_values(geo, scope, by_pair: dict, reduce_fn) -> "np.ndarray":
    """One value per channel this head draws, NaN where the dyad grid has no such pair."""
    return np.array([reduce_fn(by_pair[name]) if name in by_pair else np.nan
                     for name in geo[scope]["names"]])


def build_head_by_condition(
    geo_by_sub: dict,
    subject_ids: list[str],
    grid: dict,
    conditions: dict,
) -> "go.Figure | None":
    """One head per member per block, coloured by the share of its windows that coupled.

    The aggregated view. It answers which part of whose cap went and in which block, which a
    single instant cannot: a reader would otherwise drag a slider and average in their head.
    Complements the usable-time carpet, which is channel by time with no geometry.
    """
    names = list(conditions)
    if not names or not geo_by_sub:
        return None
    t = np.asarray(grid["t"], dtype=float)
    fig = make_subplots(rows=len(subject_ids), cols=len(names),
                        subplot_titles=names + [""] * (len(names) * (len(subject_ids) - 1)),
                        horizontal_spacing=0.015, vertical_spacing=0.04)
    for ri, sid in enumerate(subject_ids, start=1):
        geo = geo_by_sub.get(sid)
        if geo is None:
            continue
        by_pair = dict(zip(grid["pairs"], np.asarray(grid["ok"][sid], dtype=float)))
        for ci, cond in enumerate(names, start=1):
            a, b = conditions[cond]
            sel = (t >= a) & (t <= b)
            _head_ground(fig, geo, ri, ci)
            for scope in ("long", "short"):
                if scope not in geo:
                    continue
                vals = _pair_values(geo, scope, by_pair, lambda v, s=sel: v[s].mean())
                _head_glyph(fig, geo, scope, vals, ri, ci, "Coupled<br>windows", 0.0, 1.0,
                            bar=(ri == 1 and ci == len(names) and scope == "long"))
        fig.add_annotation(x=0, y=0.5, xref="paper",
                           yref=f"y{(ri - 1) * len(names) + 1}", text=sid,
                           showarrow=False, xanchor="right", xshift=-8,
                           font=dict(size=10, color="#6c757d"))
    _head_axes(fig, geo_by_sub, len(subject_ids), len(names))
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    # a head per member per block, so the grid is as wide as the blocks and only as tall as
    # the members; the height is what decides how big each head is drawn
    fig.update_layout(height=230 * len(subject_ids) + 46, plot_bgcolor="white",
                      showlegend=False, margin=dict(l=88, r=78, t=46, b=14))
    return fig


def build_head_slider(
    geo_by_sub: dict,
    subject_ids: list[str],
    grid: dict,
    series_by_sub: dict,
    conditions: "dict | None" = None,
    step: int = 3,
    cmin: float = 0.6,
    cmax: float = 1.0,
) -> "go.Figure | None":
    """One head per member, the slider running the whole recording on the shared clock.

    The instantaneous view, coloured by that window's SCI rather than by a share. Every frame
    names the block it lands in, so a position on the slider says what the dyad was doing and
    not only when; without that the reader has the carpet's x axis and no way to place it.

    ``step`` decimates the frames. Each one carries a colour per marker per member, so the
    page grows with the frame count and a ten-second grid is three times the file a
    thirty-second one is for no reading a QC pass makes.
    """
    if not geo_by_sub:
        return None
    t = np.asarray(grid["t"], dtype=float)
    frames_at = list(range(0, len(t), max(1, step)))
    if not frames_at:
        return None
    conditions = conditions or {}

    def task_at(when: float) -> str:
        for name, (a, b) in conditions.items():
            if a <= when <= b:
                return name
        return "between blocks"

    def caption(k: int) -> list[dict]:
        return [dict(x=0.5, y=1.10, xref="paper", yref="paper", showarrow=False,
                     text=f"<b>{task_at(t[k])}</b>"
                          f"<span style='color:#8a949e'> &nbsp;t = {t[k]:.0f} s</span>",
                     font=dict(size=12, color="#34495e"))]

    fig = make_subplots(rows=1, cols=len(subject_ids), subplot_titles=list(subject_ids),
                        horizontal_spacing=0.04)
    drawn: list[tuple] = []
    for ci, sid in enumerate(subject_ids, start=1):
        geo = geo_by_sub.get(sid)
        sci = (series_by_sub.get(sid) or {}).get("per_pair_sci")
        if geo is None or sci is None:
            continue
        _head_ground(fig, geo, 1, ci)
        for scope in ("long", "short"):
            if scope not in geo:
                continue
            vals = _pair_values(geo, scope, sci, lambda v, k=frames_at[0]: v[k])
            i = _head_glyph(fig, geo, scope, vals, 1, ci, "SCI<br>(10 s)", cmin, cmax,
                            bar=(ci == len(subject_ids) and scope == "long"))
            drawn.append((sid, scope, i))
    if not drawn:
        return None

    fig.frames = [
        go.Frame(name=f"{t[k]:.0f}",
                 data=[go.Scatter(marker=dict(color=np.repeat(
                     _pair_values(geo_by_sub[sid], scope,
                                  series_by_sub[sid]["per_pair_sci"],
                                  lambda v, kk=k: v[kk]),
                     geo_by_sub[sid][scope]["per_pair"]).astype(np.float32)))
                       for sid, scope, _i in drawn],
                 traces=[i for _s, _sc, i in drawn],
                 layout=dict(annotations=caption(k)))
        for k in frames_at
    ]
    _head_axes(fig, geo_by_sub, 1, len(subject_ids))
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(
        height=420, plot_bgcolor="white", showlegend=False,
        margin=dict(l=20, r=78, t=62, b=76),
        annotations=list(fig.layout.annotations) + caption(frames_at[0]),
        sliders=[dict(active=0, y=0, yanchor="top", pad=dict(t=30, b=4),
                      currentvalue=dict(visible=False), font=dict(size=9),
                      steps=[dict(method="animate", label=fr.name,
                                  args=[[fr.name],
                                        dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))])
                             for fr in fig.frames])])
    return fig


# ---------------------------------------------------------------------------
# Figure: screening synchrony against its null
# ---------------------------------------------------------------------------

# The conventional line a surrogate test is read at. One number, because the figure shades
# it and the table flags against it and the two must not drift.
NULL_ALPHA_PCT = 95.0


def build_screening_strip(coherence_df: "pd.DataFrame") -> "go.Figure | None":
    """Each window's coherence as its rank inside its own surrogate null, one row per window.

    **Raw coherence cannot share an axis across windows.** The estimator's floor sits near
    1/(number of Welch segments), and that count falls with the window, so on one recording
    the floor moves by an order of magnitude between a 300 s block and the whole run: 0.03 is
    unremarkable in one window and out of reach in another. A value's percentile inside the
    null drawn for *that* window is the quantity that is comparable, and it puts every window
    on one axis with one line to clear.

    One pale dot per channel, a diamond for the channel mean, and the top 5% shaded. The
    channel mean is the number to read: at the iteration counts a QC pass can afford, a
    per-channel percentile is a noisy rank and the count of channels over the line moves with
    the draw while the window's own verdict does not.

    Expects the frame :func:`~fnirs_pipe.pipeline.synchrony.screening_coherence` returns.
    None when it is empty.
    """
    if coherence_df is None or coherence_df.empty:
        return None
    windows = list(dict.fromkeys(coherence_df["window"]))
    rows = list(reversed(windows))
    jitter = np.random.default_rng(0)
    # each block in the colour panel 2 and 3 give it, so a reader carries one mapping across
    # the page; the whole run is not a block and stays neutral
    blocks = [w for w in windows if w != "whole run"]
    colours = {**_cond_colors(blocks), "whole run": "#7f8c8d"}

    fig = go.Figure()
    fig.add_vrect(x0=NULL_ALPHA_PCT, x1=100, fillcolor="#3498db", opacity=0.07, line_width=0)
    fig.add_vline(x=NULL_ALPHA_PCT, line_color="#adb5bd", line_width=1, line_dash="dot")
    for i, name in enumerate(rows):
        sub = coherence_df[coherence_df["window"] == name]
        pct = sub["percentile"].to_numpy(dtype=float)
        colour = colours.get(name, "#7f8c8d")
        fig.add_trace(go.Scatter(
            x=pct, y=i + jitter.uniform(-0.13, 0.13, len(pct)), mode="markers",
            name=str(name), legendgroup=str(name), showlegend=False,
            customdata=sub["ch_name"].tolist(),
            # saturated only above the line: a channel that cleared its null is the one a
            # reader is looking for, and the rest are the spread it has to be read against
            marker=dict(size=8,
                        color=[colour if p >= NULL_ALPHA_PCT else "#d7dde2" for p in pct],
                        opacity=[0.95 if p >= NULL_ALPHA_PCT else 0.75 for p in pct],
                        line=dict(width=0.6, color="#fff")),
            hovertemplate=("<b>%{customdata}</b><br>" + str(name)
                           + "<br>%{x:.1f}th percentile of its null<extra></extra>"),
        ))
    # the window's own rank, not the mean of its channels' ranks: averaging fourteen noisy
    # ranks is a weaker statement than pooling the channels and ranking once
    means = [float(coherence_df[coherence_df["window"] == n]["window_percentile"].iloc[0])
             for n in rows]
    fig.add_trace(go.Scatter(
        x=means, y=list(range(len(rows))), mode="markers", name="channel mean",
        customdata=rows, showlegend=False,
        marker=dict(size=13, symbol="diamond",
                    color=[colours.get(n, "#7f8c8d") for n in rows],
                    line=dict(width=1.2, color="#fff")),
        hovertemplate="%{customdata}<br>mean at the %{x:.1f}th percentile<extra></extra>"))

    fig.update_xaxes(title_text="Percentile inside its own pseudo-dyad null",
                     title_font=dict(size=10), range=[-2, 102], dtick=25,
                     gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=rows, showgrid=False,
                     range=[-0.6, len(rows) - 0.4], tickfont=dict(size=9))
    fig.update_layout(height=110 + 40 * len(rows), plot_bgcolor="white",
                      margin=dict(l=110, r=24, t=52, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


def screening_summary(coherence_df: "pd.DataFrame") -> dict:
    """The dyad-level numbers the summary prints, off the same frame the strip draws.

    ::

      -> {"windows": {"conversation": {"percentile": 100.0, "coherence": 0.224, ...}},
          "above": ["conversation"], "alpha": 95.0}

    ``mean_coherence`` stays in the record because a reader wants the measured value, but the
    **percentile is what grades it**: the old summary coloured the raw number against 0.3 and
    0.1, which on a normal recording sits at the null floor and printed red for a dyad with
    nothing wrong with it.
    """
    if coherence_df is None or coherence_df.empty:
        return {}
    out: dict = {"alpha": NULL_ALPHA_PCT, "windows": {}}
    for name in dict.fromkeys(coherence_df["window"]):
        sub = coherence_df[coherence_df["window"] == name]
        out["windows"][str(name)] = {
            "coherence": round(float(sub["coherence"].mean()), 4),
            "null_mean": round(float(sub["null_mean"].mean()), 4),
            "percentile": round(float(sub["window_percentile"].iloc[0]), 1),
            "n_channels_above": int((sub["percentile"] >= NULL_ALPHA_PCT).sum()),
            "n_channels": int(len(sub)),
            "n_seg": int(sub["n_seg"].iloc[0]),
            "window_s": float(sub["window_s"].iloc[0]),
        }
    out["above"] = [k for k, v in out["windows"].items()
                    if v["percentile"] >= NULL_ALPHA_PCT]
    return out


# ---------------------------------------------------------------------------
# Figure: trigger timeline
# ---------------------------------------------------------------------------

def build_trigger_timeline(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    xlabel: str = "Time (s) [aligned]",
) -> go.Figure | None:
    per_sub: dict[str, list[dict]] = {}
    all_descs: list[str] = []
    for sid in subject_ids:
        raw = aligned_raws.get(sid)
        mkrs = _extract_markers(raw) if raw else []
        per_sub[sid] = mkrs
        for m in mkrs:
            if m["description"] not in all_descs:
                all_descs.append(m["description"])

    if not any(per_sub.values()):
        return None

    colors = _cond_colors(all_descs)
    traces: list[go.BaseTraceType] = []
    seen: set[str] = set()

    for sub_idx, sid in enumerate(subject_ids):
        by_desc: dict[str, list[dict]] = {}
        for m in per_sub[sid]:
            by_desc.setdefault(m["description"], []).append(m)
        for desc, events in by_desc.items():
            traces += timeline_row_traces(events, sub_idx, colors.get(desc, "#999"),
                                          desc, desc not in seen, f"<br>{sid}")
            seen.add(desc)

    # the legend stays here, unlike the single-recording timeline: a row is a member and
    # the conditions sharing it are told apart by colour alone
    xaxis, yaxis = timeline_axes([f"sub-{s}" for s in subject_ids])
    xaxis["title"] = xlabel
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=xaxis, yaxis=yaxis,
            shapes=timeline_row_bands(len(subject_ids)),
            plot_bgcolor="white", paper_bgcolor="white",
            # overlay, or plotly groups each subject's bars and shifts them off their row
            barmode="overlay",
            height=max(110, len(subject_ids) * TIMELINE_ROW_PX + 88),
            margin=dict(l=80, r=20, t=10, b=44),
            legend=dict(font=dict(size=9), orientation="h", y=-0.35),
            hovermode="closest",
        ),
    )


# ---------------------------------------------------------------------------
# Figure: HbO + HbR signal overlay — 2-row subplot, channel switching via select
# ---------------------------------------------------------------------------

def build_signal_overlay(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    markers_list: list[dict],
    cond_colors_: dict[str, str],
) -> go.Figure | None:
    if not aligned_raws:
        return None
    ref_raw = aligned_raws.get(subject_ids[0])
    if ref_raw is None:
        return None
    hbo_picks = mne.pick_types(ref_raw.info, fnirs="hbo")
    if not len(hbo_picks):
        return None

    ch_pairs = [ref_raw.ch_names[p].rsplit(" ", 1)[0] for p in hbo_picks]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.06,
        subplot_titles=["HbO", "HbR"],
    )

    # HbO traces (row 1) then HbR traces (row 2): n_ch * n_subs each
    for row_idx, hb_type in enumerate(["hbo", "hbr"], start=1):
        for ch_idx, pair in enumerate(ch_pairs):
            ch_name = f"{pair} {hb_type}"
            for sub_idx, sid in enumerate(subject_ids):
                raw = aligned_raws.get(sid)
                if raw is None or ch_name not in raw.ch_names:
                    t_vals, y_vals = [], []
                else:
                    pick = raw.ch_names.index(ch_name)
                    arr, times = _decimate(
                        raw.get_data(picks=[pick]), raw.times, _MAX_TS_PTS
                    )
                    t_vals = times.tolist()
                    y_vals = (arr[0] * 1e6).tolist()
                fig.add_trace(go.Scatter(
                    x=t_vals, y=y_vals,
                    name=sid, mode="lines",
                    line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.3),
                    visible=(ch_idx == 0),
                    showlegend=False,
                    legendgroup=sid,
                ), row=row_idx, col=1)

    # Ghost traces for stable subject legend
    for sub_idx, sid in enumerate(subject_ids):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=sid,
            line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.5),
            showlegend=True, visible=True,
            legendgroup=sid,
        ), row=1, col=1)

    mkr_shapes = []
    for m in markers_list:
        color = cond_colors_.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            mkr_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(width=0), layer="below",
            ))

    fig.update_layout(
        shapes=mkr_shapes,
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=60, r=15, t=30, b=40),
        legend=dict(font=dict(size=9)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_xaxes(title_text="Time (s) [aligned]", row=2, col=1)
    fig.update_yaxes(title_text="µmol/L", gridcolor="#eeeeee")

    return fig


def build_signal_overlay_pair(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    pair: str,
    markers_list: list[dict],
    cond_colors_: dict[str, str],
) -> go.Figure | None:
    """Single-channel-pair signal overlay (multi-subject), for iframe per-channel pages."""
    if not aligned_raws:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.06,
        subplot_titles=["HbO", "HbR"],
    )

    for row_idx, hb_type in enumerate(["hbo", "hbr"], start=1):
        ch_name = f"{pair} {hb_type}"
        for sub_idx, sid in enumerate(subject_ids):
            raw = aligned_raws.get(sid)
            if raw is None or ch_name not in raw.ch_names:
                continue
            pick = raw.ch_names.index(ch_name)
            arr, times = _decimate(
                raw.get_data(picks=[pick]), raw.times, _MAX_TS_PTS
            )
            fig.add_trace(go.Scatter(
                x=times.tolist(), y=(arr[0] * 1e6).tolist(),
                name=sid, mode="lines",
                line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.3),
                showlegend=(row_idx == 1),
                legendgroup=sid,
            ), row=row_idx, col=1)

    mkr_shapes = []
    for m in markers_list:
        color = cond_colors_.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            mkr_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(width=0), layer="below",
            ))

    fig.update_layout(
        shapes=mkr_shapes,
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=60, r=15, t=30, b=40),
        legend=dict(font=dict(size=9)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_xaxes(title_text="Time (s) [aligned]", row=2, col=1)
    fig.update_yaxes(title_text="µmol/L", gridcolor="#eeeeee")
    return fig


# ---------------------------------------------------------------------------
# Figure: per-channel PSD (HbO, multi-subject)
# ---------------------------------------------------------------------------

def build_psd(
    aligned_raws: dict[str, mne.io.Raw],
    ch_pair: str,
    subject_ids: list[str],
    cardiac: "tuple[float, float] | None" = None,
) -> go.Figure | None:
    hbo_name = f"{ch_pair} hbo"
    traces: list[go.BaseTraceType] = []

    for sub_idx, sid in enumerate(subject_ids):
        raw = aligned_raws.get(sid)
        if raw is None or hbo_name not in raw.ch_names:
            continue
        pick = raw.ch_names.index(hbo_name)
        arr = raw.get_data(picks=[pick])
        sfreq = raw.info["sfreq"]
        psds, freqs = mne.time_frequency.psd_array_welch(
            arr, sfreq, n_fft=min(PSD_NFFT, arr.shape[1]), verbose=False)
        psd = psds[0]
        fmax = min(2.0, float(sfreq / 2))
        mask = freqs <= fmax
        traces.append(go.Scatter(
            x=freqs[mask].tolist(), y=psd[mask].tolist(),
            name=sid, mode="lines",
            line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.8),
        ))

    if not traces:
        return None

    shapes, annots = _psd_band_shapes(cardiac)
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title="Frequency (Hz)", range=[0, 2], gridcolor="#eeeeee"),
            yaxis=dict(title="Power (HbO)", type="log", gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=200, margin=dict(l=60, r=15, t=22, b=38),
            legend=dict(font=dict(size=9), orientation="h",
                        x=1, xanchor="right", y=1.0, yanchor="bottom"),
            shapes=shapes, annotations=annots,
        ),
    )


# ---------------------------------------------------------------------------
# Figure: channel quality summary — group status, channels on x-axis
# ---------------------------------------------------------------------------

def build_channel_summary(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> go.Figure | None:
    ch_set: set[str] = set()
    for sid in subject_ids:
        for k in sci_of(sqm_data, sid):
            ch_set.add(k.rsplit(" ", 1)[0] if " " in k else k)

    if not ch_set:
        return None

    ch_names = sorted(ch_set)
    colors, hover_texts = [], []

    for ch in ch_names:
        statuses = _ch_kept_by_member(ch, sqm_data, subject_ids, sci_threshold)
        known    = [s for s in statuses if s is not None]
        sci_vals = []
        for sid in subject_ids:
            sci_d = sci_of(sqm_data, sid)
            val   = sci_d.get(f"{ch} hbo") or sci_d.get(ch)
            sci_vals.append(f"{sid}: {val:.3f}" if val is not None else f"{sid}: N/A")

        if not known:
            c, status = _NA_COLOR, "N/A"
        elif all(known):
            c, status = _GOOD_COLOR, "all good"
        elif not any(known):
            c, status = _BAD_COLOR, "all bad"
        else:
            c, status = _MIX_COLOR, "mixed"

        colors.append(c)
        hover_texts.append(f"<b>{ch}</b> ({status})<br>" + "<br>".join(sci_vals))

    traces: list[go.BaseTraceType] = [
        go.Scatter(
            x=ch_names, y=[0] * len(ch_names),
            mode="markers",
            marker=dict(symbol="square", size=16, color=colors,
                        line=dict(width=0.8, color="#bbb")),
            hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )
    ]
    for color, label in [(_GOOD_COLOR, "All good"), (_MIX_COLOR, "Mixed"),
                          (_BAD_COLOR, "All bad"), (_NA_COLOR, "N/A")]:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color=color,
                        line=dict(width=1, color="#bbb")),
            name=label, showlegend=True,
        ))

    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(tickfont=dict(size=8), showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=[-1, 1]),
            plot_bgcolor="white", paper_bgcolor="white",
            height=120, margin=dict(l=20, r=20, t=8, b=60),
            legend=dict(font=dict(size=9), orientation="h", y=-0.55),
        ),
    )


# ---------------------------------------------------------------------------
# Compute: group-level SQM scalars
# ---------------------------------------------------------------------------

def compute_hyper_sqm(
    sqm_data: dict[str, dict],
    coherence_df: pd.DataFrame,
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    subject_ids: list[str],
    sci_threshold: float,
) -> dict:
    ch_set: set[str] = set()
    for sid in subject_ids:
        for k in sci_of(sqm_data, sid):
            ch_set.add(k.rsplit(" ", 1)[0] if " " in k else k)

    n_all_good = n_mixed = n_all_bad = n_unknown = 0
    for ch in ch_set:
        statuses = _ch_kept_by_member(ch, sqm_data, subject_ids, sci_threshold)
        known = [s for s in statuses if s is not None]
        if not known:
            n_unknown += 1
        elif all(known):
            n_all_good += 1
        elif not any(known):
            n_all_bad += 1
        else:
            n_mixed += 1

    n_total  = len(ch_set)
    pct_good = round(n_all_good / n_total * 100, 1) if n_total > 0 else None

    mean_coherence = peak_coherence = peak_coherence_channel = None
    if not coherence_df.empty:
        mean_coherence         = round(float(coherence_df["coherence"].mean()), 3)
        ch_mean                = coherence_df.groupby("ch_name")["coherence"].mean()
        peak_coherence_channel = str(ch_mean.idxmax())
        peak_coherence         = round(float(ch_mean.max()), 3)

    aligned_duration_s = None
    if aligned_raws:
        aligned_duration_s = round(float(next(iter(aligned_raws.values())).times[-1]), 1)

    max_offset_s = round(max(offsets.values()), 3) if offsets else None

    return dict(
        n_all_good=n_all_good, n_mixed=n_mixed,
        n_all_bad=n_all_bad, n_unknown=n_unknown, n_total=n_total,
        pct_all_good=pct_good,
        mean_coherence=mean_coherence, peak_coherence=peak_coherence,
        peak_coherence_channel=peak_coherence_channel,
        aligned_duration_s=aligned_duration_s, max_offset_s=max_offset_s,
    )
