"""Figure builders for the hyperscanning post-processing QC report."""

from __future__ import annotations

import numpy as np
import mne
import pandas as pd
import plotly.graph_objects as go

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.hyper_post")


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_wtc_channel(
    wtc_data: dict,
    freqs: np.ndarray,
    times: np.ndarray,
    pair_label: str,
    markers_list: list[dict],
    cond_colors: dict[str, str],
) -> go.Figure | None:
    """WTC heatmap for one channel pair: time × log-frequency, colour = coherence [0–1].

    wtc_data: {"wtc": ndarray(n_freqs, n_times), "coi": ndarray(n_times)}
    COI boundary drawn as a white dashed line; regions below it may be edge-affected.
    """
    if wtc_data is None or len(freqs) == 0 or len(times) == 0:
        return None

    wtc_arr = wtc_data["wtc"]   # (n_freqs, n_times)
    coi     = wtc_data["coi"]   # (n_times) — max reliable period in seconds

    # COI → minimum reliable frequency at each time point
    with np.errstate(divide="ignore", invalid="ignore"):
        freq_coi = np.where(coi > 1e-10, 1.0 / coi, freqs.max())
    freq_coi = np.clip(freq_coi, float(freqs.min()), float(freqs.max()))

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=times.tolist(),
        y=freqs.tolist(),
        z=np.round(wtc_arr, 3).tolist(),
        zmin=0, zmax=1,
        colorscale="Viridis",
        colorbar=dict(title="WTC", thickness=12, len=0.6, y=0.5),
        hovertemplate="t=%{x:.1f}s<br>f=%{y:.4f}Hz<br>WTC=%{z:.3f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=times.tolist(),
        y=freq_coi.tolist(),
        mode="lines",
        line=dict(color="white", width=1.5, dash="dot"),
        name="COI",
        showlegend=True,
        hoverinfo="skip",
    ))

    shapes = []
    for m in markers_list:
        color = cond_colors.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(width=0), layer="below",
            ))

    fig.update_layout(
        shapes=shapes,
        xaxis=dict(title=f"Time (s)  [{pair_label}]", gridcolor="#444"),
        yaxis=dict(title="Frequency (Hz)", type="log", gridcolor="#444"),
        height=300,
        margin=dict(l=60, r=80, t=10, b=40),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(font=dict(size=9)),
    )
    return fig


# TODO (optional): ROI-level WTC — average HbO within each anatomical ROI
# (roi_map: {"PFC_left": ["S1-D1", ...], ...}), then compute WTC on the
# averaged signal. Requires roi_map passed from CLI --roi-mapping.
def build_wtc_roi(
    aligned_raws: dict[str, mne.io.Raw],
    roi_map: dict[str, list[str]],
    subject_ids: list[str],
    markers_list: list[dict],
    cond_colors: dict[str, str],
    fmin: float = 0.004,
    fmax: float = 0.20,
) -> dict[str, go.Figure]:
    raise NotImplementedError


def build_isc_matrix(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
) -> go.Figure | None:
    """Inter-brain correlation heatmap (Sub1 channels × Sub2 channels, Pearson r)."""
    raise NotImplementedError

