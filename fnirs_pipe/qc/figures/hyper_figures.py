"""Plotly figure builders for the hyperscanning group-level raw QC report."""

from __future__ import annotations

from itertools import combinations

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import coherence

from fnirs_pipe.qc.figure_io import extract_markers as _extract_markers
from fnirs_pipe.qc.figures._brain_utils import mni_trans
from fnirs_pipe.qc.figures._utils import (CONDITION_PALETTE, decimate as _decimate,
                                          epochable_events, physio_bands)
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


def _ch_sci_status(
    ch_pair: str,
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> list[bool | None]:
    result = []
    for sid in subject_ids:
        sci_d = sqm_data.get(sid, {}).get("sci_per_channel", {})
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
    lines = [f"<b>{pair}</b>"]
    for sid in subject_ids:
        sci_d = sqm_data.get(sid, {}).get("sci_per_channel", {})
        val = sci_d.get(f"{pair} hbo") or sci_d.get(pair)
        lines.append(f"{sid}: SCI = {val:.3f}" if val is not None else f"{sid}: N/A")
    return "<br>".join(lines)


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
            color = colors.get(desc, "#999")
            traces.append(go.Scatter(
                x=[e["onset"] for e in events],
                y=[sub_idx] * len(events),
                mode="markers",
                marker=dict(symbol="line-ns-open", size=18, color=color,
                            line=dict(width=2.5, color=color)),
                name=desc,
                legendgroup=desc,
                showlegend=(desc not in seen),
                hovertemplate=(
                    f"<b>{desc}</b><br>onset: %{{x:.2f}} s"
                    f"<br>{sid}<extra></extra>"
                ),
            ))
            seen.add(desc)

    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title=xlabel, gridcolor="#eeeeee"),
            yaxis=dict(tickvals=list(range(len(subject_ids))),
                       ticktext=[f"sub-{s}" for s in subject_ids],
                       autorange="reversed", gridcolor="#eeeeee",
                       tickfont=dict(size=10)),
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(110, len(subject_ids) * 52 + 60),
            margin=dict(l=80, r=15, t=8, b=40),
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
        # match the metric-side PSD (MNE compute_psd default Welch, n_fft=256)
        psds, freqs = mne.time_frequency.psd_array_welch(
            arr, sfreq, n_fft=min(256, arr.shape[1]), verbose=False)
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
            height=200, margin=dict(l=60, r=15, t=8, b=38),
            legend=dict(font=dict(size=9)),
            shapes=shapes, annotations=annots,
        ),
    )


# ---------------------------------------------------------------------------
# Figure: per-channel mean epoch comparison
# ---------------------------------------------------------------------------

def build_epoch(
    aligned_raws: dict[str, mne.io.Raw],
    ch_pair: str,
    subject_ids: list[str],
    tmin: float = -5.0,
    tmax: float = 25.0,
) -> go.Figure | None:
    hbo_name = f"{ch_pair} hbo"
    first_raw = next(
        (aligned_raws[s] for s in subject_ids if s in aligned_raws), None
    )
    if first_raw is None:
        return None

    all_descs = list(dict.fromkeys(
        a["description"] for a in first_raw.annotations
        if not str(a["description"]).upper().startswith("BAD")
    ))
    if not all_descs:
        return None

    traces: list[go.BaseTraceType] = []
    for sub_idx, sid in enumerate(subject_ids):
        raw = aligned_raws.get(sid)
        if raw is None or hbo_name not in raw.ch_names:
            continue
        pick = raw.ch_names.index(hbo_name)
        try:
            events_mne, event_id = epochable_events(raw, tmin, tmax)
            if len(events_mne) == 0:
                continue
            epochs = mne.Epochs(
                raw, events_mne, event_id,
                tmin=tmin, tmax=tmax, picks=[pick],
                baseline=(tmin, 0), preload=True, verbose=False,
            )
        except Exception as exc:
            logger.warning("epoch failed sub-%s %s: %s", sid, ch_pair, exc)
            continue

        sub_color = _SUB_COLORS[sub_idx % len(_SUB_COLORS)]
        for ci, desc in enumerate(all_descs):
            if desc not in event_id:
                continue
            try:
                ep = epochs[desc].get_data()
                if ep.shape[0] == 0:
                    continue
                traces.append(go.Scatter(
                    x=epochs.times.tolist(),
                    y=(ep[:, 0, :].mean(axis=0) * 1e6).tolist(),
                    name=f"{sid} {desc} (n={ep.shape[0]})",
                    mode="lines",
                    line=dict(color=sub_color, width=1.8,
                              dash=_COND_DASHES[ci % len(_COND_DASHES)]),
                    legendgroup=sid,
                ))
            except Exception as exc:
                logger.warning("epoch cond %s sub-%s %s: %s", desc, sid, ch_pair, exc)

    if not traces:
        return None

    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title="Time rel. onset (s)", gridcolor="#eeeeee",
                       zerolinecolor="#cccccc"),
            yaxis=dict(title="HbO (µmol/L)", gridcolor="#eeeeee"),
            shapes=[dict(type="line", xref="x", yref="paper",
                         x0=0, x1=0, y0=0, y1=1,
                         line=dict(color="#7f8c8d", width=1, dash="dash"))],
            annotations=[dict(x=0, y=1.0, xref="x", yref="paper",
                              text="onset", showarrow=False,
                              font=dict(size=8, color="#7f8c8d"), xanchor="left")],
            plot_bgcolor="white", paper_bgcolor="white",
            height=200, margin=dict(l=60, r=15, t=8, b=38),
            legend=dict(font=dict(size=8), tracegroupgap=4),
        ),
    )


# ---------------------------------------------------------------------------
# Figure: 2D optode layout with multi-subject SCI colouring
# ---------------------------------------------------------------------------

def build_layout_2d(
    aligned_raws: dict[str, mne.io.Raw],
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> go.Figure | None:
    ref_raw = aligned_raws.get(subject_ids[0]) if subject_ids else None
    if ref_raw is None:
        return None

    hbo_picks = mne.pick_types(ref_raw.info, fnirs="hbo")
    if not len(hbo_picks):
        return None

    chs      = ref_raw.info["chs"]
    ch_names = ref_raw.ch_names
    locs     = np.array([chs[p]["loc"][:3] for p in hbo_picks])
    if not np.any(np.abs(locs) > 1e-4):
        logger.warning("build_layout_2d: no optode positions found (all near-zero); skipping")
        return None

    pair_names  = [ch_names[p].rsplit(" ", 1)[0] for p in hbo_picks]
    colors      = [_group_color(_ch_sci_status(p, sqm_data, subject_ids, sci_threshold))
                   for p in pair_names]
    hover_texts = [_hover_sci(p, sqm_data, subject_ids) for p in pair_names]
    x_mm = (locs[:, 0] * 1000).tolist()
    y_mm = (locs[:, 1] * 1000).tolist()

    seen: set = set()
    lines_x, lines_y = [], []
    for pick in hbo_picks:
        ch  = chs[pick]
        src = tuple(round(v, 6) for v in ch["loc"][3:6])
        det = tuple(round(v, 6) for v in ch["loc"][:3])
        key = src + det
        if key in seen or not (any(src) or any(det)):
            continue
        seen.add(key)
        lines_x += [src[0] * 1000, det[0] * 1000, None]
        lines_y += [src[1] * 1000, det[1] * 1000, None]

    traces: list[go.BaseTraceType] = []
    if lines_x:
        traces.append(go.Scatter(
            x=lines_x, y=lines_y, mode="lines",
            line=dict(color="#b0bec5", width=1),
            showlegend=False, hoverinfo="skip",
        ))
    traces.append(go.Scatter(
        x=x_mm, y=y_mm, mode="markers+text",
        marker=dict(size=11, color=colors, line=dict(width=1, color="#888")),
        text=pair_names, textposition="top center", textfont=dict(size=7),
        hovertext=hover_texts, hoverinfo="text",
        showlegend=False,
    ))
    for color, label in [(_GOOD_COLOR, "All good"), (_MIX_COLOR, "Mixed"),
                          (_NA_COLOR, "All bad / N/A")]:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, line=dict(width=1, color="#888")),
            name=label, showlegend=True,
        ))

    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title="x (mm)", gridcolor="#eeeeee", scaleanchor="y"),
            yaxis=dict(title="y (mm)", gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=320, margin=dict(l=50, r=15, t=8, b=40),
            legend=dict(font=dict(size=9)),
        ),
    )


# ---------------------------------------------------------------------------
# Figure: 3D channel positions (fsaverage brain)
# ---------------------------------------------------------------------------

def build_layout_3d(
    aligned_raws: dict[str, mne.io.Raw],
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> go.Figure | None:
    ref_raw = aligned_raws.get(subject_ids[0]) if subject_ids else None
    if ref_raw is None:
        return None

    hbo_picks = mne.pick_types(ref_raw.info, fnirs="hbo")
    if not len(hbo_picks):
        return None

    chs      = ref_raw.info["chs"]
    ch_names = ref_raw.ch_names
    locs     = np.array([chs[p]["loc"][:3] for p in hbo_picks])
    if not np.any(np.abs(locs) > 1e-4):
        logger.warning("build_layout_3d: no optode positions found; skipping")
        return None

    pair_names = [ch_names[p].rsplit(" ", 1)[0] for p in hbo_picks]

    trans     = mni_trans(ref_raw.info)
    coords_mm = mne.transforms.apply_trans(trans, locs) * 1000

    seen3: set = set()
    lx, ly, lz = [], [], []
    for pick in hbo_picks:
        ch  = chs[pick]
        src = ch["loc"][3:6]
        det = ch["loc"][:3]
        key = tuple(round(float(v), 5) for v in np.concatenate([src, det]))
        if key in seen3 or not (np.any(src) or np.any(det)):
            continue
        seen3.add(key)
        src_m = mne.transforms.apply_trans(trans, src.reshape(1, 3))[0] * 1000
        det_m = mne.transforms.apply_trans(trans, det.reshape(1, 3))[0] * 1000
        lx += [float(src_m[0]), float(det_m[0]), None]
        ly += [float(src_m[1]), float(det_m[1]), None]
        lz += [float(src_m[2]), float(det_m[2]), None]

    ch_colors   = [_group_color(_ch_sci_status(p, sqm_data, subject_ids, sci_threshold))
                   for p in pair_names]
    hover_texts = [_hover_sci(p, sqm_data, subject_ids) for p in pair_names]

    traces_3d: list[go.BaseTraceType] = []
    try:
        from nilearn import datasets as nl_ds, surface as surf
        fsavg5 = nl_ds.fetch_surf_fsaverage(mesh="fsaverage5")
        for key in ("pial_left", "pial_right"):
            verts, faces = surf.load_surf_mesh(fsavg5[key])
            traces_3d.append(go.Mesh3d(
                x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
                i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
                color="#e3e3e3", opacity=0.55, flatshading=False,
                lighting=dict(ambient=0.85, diffuse=0.5, specular=0.4, fresnel=0.6),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo="skip", showlegend=False,
            ))
    except Exception as exc:
        logger.warning("brain mesh skipped: %s", exc)

    if lx:
        traces_3d.append(go.Scatter3d(
            x=lx, y=ly, z=lz, mode="lines",
            line=dict(color="#95a5a6", width=4),
            hoverinfo="skip", showlegend=False,
        ))

    traces_3d.append(go.Scatter3d(
        x=coords_mm[:, 0].tolist(), y=coords_mm[:, 1].tolist(),
        z=coords_mm[:, 2].tolist(), mode="markers",
        marker=dict(size=7, color=ch_colors, opacity=0.92,
                    line=dict(width=0.5, color="#333")),
        hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>",
        name="Channels", showlegend=True,
    ))

    return go.Figure(
        data=traces_3d,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=0, y=-1.9, z=0.6)),
                bgcolor="#ffffff",
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#ffffff", height=380,
            legend=dict(x=0.01, y=0.99, font=dict(size=9)),
        ),
    )


# ---------------------------------------------------------------------------
# Figure: inter-subject coherence bar chart (mean per channel, sorted)
# ---------------------------------------------------------------------------

def build_coherence_bar(coherence_df: pd.DataFrame) -> go.Figure | None:
    if coherence_df.empty:
        return None

    pairs = (
        coherence_df.drop_duplicates(["sub1", "sub2"])[["sub1", "sub2"]]
        .values.tolist()
    )
    ch_mean   = coherence_df.groupby("ch_name")["coherence"].mean().sort_values(ascending=True)
    ch_sorted = ch_mean.index.tolist()

    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    traces: list[go.BaseTraceType] = []
    for pi, (sub1, sub2) in enumerate(pairs):
        sub_df = coherence_df[
            (coherence_df["sub1"] == sub1) & (coherence_df["sub2"] == sub2)
        ]
        ch_map = dict(zip(sub_df["ch_name"], sub_df["coherence"]))
        traces.append(go.Bar(
            y=ch_sorted, x=[ch_map.get(ch) for ch in ch_sorted],
            name=f"{sub1}–{sub2}", orientation="h",
            marker=dict(color=palette[pi % len(palette)], opacity=0.78),
        ))

    return go.Figure(
        data=traces,
        layout=go.Layout(
            barmode="group",
            xaxis=dict(title="Coherence", range=[0, 1], gridcolor="#eeeeee"),
            yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(280, len(ch_sorted) * 20 + 80),
            margin=dict(l=80, r=20, t=8, b=40),
            legend=dict(font=dict(size=9)),
            bargap=0.25, bargroupgap=0.05,
        ),
    )


# ---------------------------------------------------------------------------
# Compute: windowed coherence (time x channel DataFrame)
# ---------------------------------------------------------------------------

def compute_windowed_coherence(
    aligned_raws: dict[str, mne.io.Raw],
    fmin: float,
    fmax: float,
    window_s: float = 30.0,
    step_s: float = 5.0,
) -> pd.DataFrame:
    """Sliding-window pairwise coherence per long HbO channel.

    Channels are matched across participants by S-D label, the way the whole-record
    coherence and WTC match them; a label one participant lacks keeps its row with NaN
    coherence, so the heatmap shows the gap rather than shifting its neighbours into it.

    Returns DataFrame with columns: t_center, ch_name, sub1, sub2, coherence.
    """
    from fnirs_pipe.pipeline.synchrony import _long_hbo_by_label, _shared_sfreq

    subject_ids = list(aligned_raws.keys())
    if len(subject_ids) < 2:
        return pd.DataFrame()

    sfreq     = _shared_sfreq(aligned_raws)
    win_samp  = int(window_s * sfreq)
    step_samp = int(step_s * sfreq)
    n_times   = min(raw.n_times for raw in aligned_raws.values())
    if win_samp >= n_times:
        return pd.DataFrame()

    starts = list(range(0, n_times - win_samp + 1, step_samp))
    # freq_req: minimum nperseg so that at least one bin falls within [fmin, fmax]
    # capped at win_samp//2 so scipy coherence has >=2 segments per window
    freq_req = max(32, int(np.ceil(sfreq / fmax)))
    nperseg  = min(win_samp // 2, freq_req)

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        raw1, raw2 = aligned_raws[sub1], aligned_raws[sub2]
        map1, map2 = _long_hbo_by_label(raw1), _long_hbo_by_label(raw2)
        data1  = raw1.get_data(picks=list(map1.values()))
        data2  = raw2.get_data(picks=list(map2.values()))
        row_of = {label: i for i, label in enumerate(map2)}
        # (row in data1, row in data2 or None, label), resolved once for every window
        matched = [(i, row_of.get(label), label) for i, label in enumerate(map1)]
        for label in (lbl for _, j, lbl in matched if j is None):
            logger.warning("windowed coherence: %s has no %s, leaving that row blank",
                           sub2, label)

        for start in starts:
            end      = start + win_samp
            t_center = round((start + win_samp / 2) / sfreq, 2)
            for i, j, label in matched:
                if j is None:
                    mean_coh = float("nan")
                else:
                    freqs, coh = coherence(
                        data1[i, start:end], data2[j, start:end],
                        fs=sfreq, nperseg=nperseg,
                    )
                    mask     = (freqs >= fmin) & (freqs <= fmax)
                    mean_coh = float(np.mean(coh[mask])) if mask.any() else float("nan")
                rows.append(dict(
                    t_center=t_center,
                    ch_name=label,
                    sub1=sub1, sub2=sub2,
                    coherence=mean_coh,
                ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure: windowed coherence heatmap (time x channel)
# ---------------------------------------------------------------------------

def build_coherence_timeseries(windowed_df: pd.DataFrame) -> go.Figure | None:
    if windowed_df.empty:
        return None

    pairs = windowed_df.drop_duplicates(["sub1", "sub2"])[["sub1", "sub2"]].values.tolist()

    ch_mean_order = (
        windowed_df.groupby("ch_name")["coherence"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    traces: list[go.BaseTraceType] = []
    buttons: list[dict] = []
    for pi, (sub1, sub2) in enumerate(pairs):
        pair_df = windowed_df[
            (windowed_df["sub1"] == sub1) & (windowed_df["sub2"] == sub2)
        ]
        pivot = pair_df.pivot_table(
            index="ch_name", columns="t_center", values="coherence", aggfunc="mean"
        ).reindex(ch_mean_order)

        traces.append(go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.values.tolist(),
            colorscale="Blues", zmin=0, zmax=1,
            colorbar=dict(title="Coherence", tickformat=".2f", len=0.8),
            hovertemplate="Ch: %{y}<br>Time: %{x:.1f} s<br>Coherence: %{z:.3f}<extra></extra>",
            visible=(pi == 0),
        ))
        buttons.append(dict(
            label=f"{sub1}–{sub2}", method="update",
            args=[{"visible": [j == pi for j in range(len(pairs))]},
                  {"title.text": f"Windowed Coherence: {sub1}–{sub2}"}],
        ))

    layout_kwargs: dict = dict(
        xaxis=dict(title="Time (s)", gridcolor="#eeeeee"),
        yaxis=dict(title="Channel", tickfont=dict(size=8)),
        plot_bgcolor="white", paper_bgcolor="white",
        height=max(280, len(ch_mean_order) * 14 + 80),
        margin=dict(l=90, r=20, t=40, b=40),
    )
    if len(pairs) > 1:
        layout_kwargs["updatemenus"] = [dict(
            type="buttons", buttons=buttons,
            direction="right", showactive=True,
            x=0, y=1.12, xanchor="left", yanchor="top",
            font=dict(size=11),
        )]

    return go.Figure(data=traces, layout=go.Layout(**layout_kwargs))


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
        for k in sqm_data.get(sid, {}).get("sci_per_channel", {}):
            ch_set.add(k.rsplit(" ", 1)[0] if " " in k else k)

    if not ch_set:
        return None

    ch_names = sorted(ch_set)
    colors, hover_texts = [], []

    for ch in ch_names:
        statuses = _ch_sci_status(ch, sqm_data, subject_ids, sci_threshold)
        known    = [s for s in statuses if s is not None]
        sci_vals = []
        for sid in subject_ids:
            sci_d = sqm_data.get(sid, {}).get("sci_per_channel", {})
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
        for k in sqm_data.get(sid, {}).get("sci_per_channel", {}):
            ch_set.add(k.rsplit(" ", 1)[0] if " " in k else k)

    n_all_good = n_mixed = n_all_bad = n_unknown = 0
    for ch in ch_set:
        statuses = _ch_sci_status(ch, sqm_data, subject_ids, sci_threshold)
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
