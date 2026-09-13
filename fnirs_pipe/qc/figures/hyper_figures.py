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


def _lighter(hex_color: str, factor: float = 0.45) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


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


def _group_color(statuses: list[bool | None]) -> str:
    known = [s for s in statuses if s is not None]
    if not known:
        return _NA_COLOR
    if all(known):
        return _GOOD_COLOR
    if not any(known):
        return _NA_COLOR
    return _MIX_COLOR


def _hover_sci(pair: str, sqm_data: dict, subject_ids: list[str]) -> str:
    """One channel pair's line per member: the verdict, and the SCI behind it.

    Both, because they answer different questions and the marker can only carry one colour.
    A channel rejected at a high SCI is the case worth being able to see, and it reads as a
    mistake unless the two numbers sit together.
    """
    lines = [f"<b>{pair}</b>"]
    for sid in subject_ids:
        sci_d = sci_of(sqm_data, sid)
        val = sci_d.get(f"{pair} hbo") or sci_d.get(pair)
        sci = f"SCI (10 s) = {val:.3f}" if val is not None else "SCI n/a"
        rejected = _rejected_pairs(sqm_data, sid)
        verdict = ("" if rejected is None
                   else " &bull; rejected" if pair in rejected else " &bull; kept")
        lines.append(f"{sid}: {sci}{verdict}")
    return "<br>".join(lines)


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

      -> {"t": [...], "pairs": ["S1_D1", ...], "ok": {"sub-01": mask, "sub-02": mask}}

    with each ``mask`` a ``pair x window`` boolean, True where that member's SCI and PSP both
    cleared their lines in that window. The three dyad panels that read it (the usable-time
    carpet, the two head figures) then cannot disagree about which window was good.

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

    pairs, rows = [], {sid: [] for sid in subject_ids}
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
            pairs = [p for p in by_pair if p in long_names] or list(by_pair)
        for pair in pairs:
            idx = by_pair.get(pair)
            # a pair one member lacks is not usable by the dyad at any moment
            rows[sid].append(masks[sid][idx].all(axis=0) if idx
                             else np.zeros(masks[sid].shape[1], dtype=bool))

    return {"t": ref[keep],
            "pairs": pairs,
            "ok": {sid: np.array(rows[sid])[:, keep] for sid in subject_ids}}


def dyad_status(grid: dict, subject_ids: list[str]) -> "np.ndarray":
    """``pair x window``: 2 coupled in every member, 1 in some, 0 in none."""
    stack = np.array([grid["ok"][sid] for sid in subject_ids])
    return np.where(stack.all(axis=0), 2, np.where(stack.any(axis=0), 1, 0))


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
