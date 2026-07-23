"""Figure builders for the hyperscanning post-processing QC report."""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import mne
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

    wtc_data: {"wtc": ndarray(n_freqs, n_times), "coi": ndarray(n_times),
               "sig": ndarray(n_freqs) | None}
    COI boundary drawn as a white dashed line; regions below it may be edge-affected.
    When "sig" is present, a black contour outlines where coherence exceeds the
    Monte Carlo significance level (WTC / sig > 1).
    """
    if wtc_data is None or len(freqs) == 0 or len(times) == 0:
        return None

    wtc_arr = wtc_data["wtc"]   # (n_freqs, n_times)
    coi     = wtc_data["coi"]   # (n_times) — max reliable period in seconds
    sig     = wtc_data.get("sig")  # (n_freqs) per-frequency significance level, or None

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

    # Significance contour: outline where coherence beats the Monte Carlo level (WTC / sig > 1).
    if sig is not None and len(sig) == wtc_arr.shape[0]:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = wtc_arr / np.asarray(sig, dtype=float)[:, None]
        fig.add_trace(go.Contour(
            x=times.tolist(),
            y=freqs.tolist(),
            z=ratio.tolist(),
            contours=dict(type="constraint", operation=">", value=1.0),
            line=dict(color="black", width=1.2),
            fillcolor="rgba(0,0,0,0)",
            showscale=False,
            name="p<0.05",
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


def compute_isc(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    bad_channels: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Compute inter-brain Pearson r matrix (n_ch × n_ch).

    matrix[i, j] = Pearson r between sub1_ch_i and sub2_ch_j.
    Diagonal = same-channel ISC.

    Args:
        ch_type: "hbo" or "hbr".
        bad_channels: {subject_id: [bad channel names]}. sub1's bad channels blank
            the matching rows, sub2's blank the columns (set to NaN).
    """
    if len(subject_ids) < 2:
        return None, None
    raw1 = aligned_raws.get(subject_ids[0])
    raw2 = aligned_raws.get(subject_ids[1])
    if raw1 is None or raw2 is None:
        return None, None

    picks1 = mne.pick_types(raw1.info, fnirs=ch_type)
    picks2 = mne.pick_types(raw2.info, fnirs=ch_type)
    n = min(len(picks1), len(picks2))
    if n == 0:
        return None, None

    data1 = raw1.get_data(picks=picks1[:n])
    data2 = raw2.get_data(picks=picks2[:n])
    ch_names = [raw1.ch_names[picks1[i]].rsplit(" ", 1)[0] for i in range(n)]

    def _zscore(x: np.ndarray) -> np.ndarray:
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std[std < 1e-12] = 1.0
        return (x - mu) / std

    d1 = _zscore(data1)
    d2 = _zscore(data2)
    isc_mat = (d1 @ d2.T) / d1.shape[1]
    np.clip(isc_mat, -1.0, 1.0, out=isc_mat)

    if bad_channels:
        def _bad_idx(sub_id: str) -> list[int]:
            bad_pairs = {c.rsplit(" ", 1)[0] for c in bad_channels.get(sub_id, [])}
            return [i for i, ch in enumerate(ch_names) if ch in bad_pairs]
        bad_rows = _bad_idx(subject_ids[0])
        bad_cols = _bad_idx(subject_ids[1])
        if bad_rows:
            isc_mat[bad_rows, :] = np.nan
        if bad_cols:
            isc_mat[:, bad_cols] = np.nan

    return isc_mat, ch_names


def build_isc_panel(
    isc_mat: np.ndarray,
    ch_names: list[str],
    subject_ids: list[str],
    ch_type: str = "hbo",
    isc_threshold: float = 0.3,
) -> str:
    """Return base64 PNG of 2-panel ISC summary: ISC matrix | connectogram.

    Args:
        isc_mat:       n × n ISC matrix from compute_isc().
        ch_names:      Channel labels (n,), without type suffix.
        subject_ids:   [sub1_id, sub2_id, ...].
        ch_type:       "hbo" or "hbr", shown in titles.
        isc_threshold: Minimum |ISC| arc threshold forwarded to connectogram.
    """
    from PIL import Image
    from fnirs_pipe.qc.figures.connectogram import isc_connectogram as _isc_conn

    if isc_mat is None or len(ch_names) == 0:
        return ""

    n          = len(ch_names)
    isc_diag   = np.diag(isc_mat)
    sub1_label = subject_ids[0] if subject_ids else "Sub1"
    sub2_label = subject_ids[1] if len(subject_ids) > 1 else "Sub2"
    type_label = ch_type.upper()

    # height driven by matrix size so it can be square; cap to [5, 9]
    sq = float(np.clip(n * 0.20, 5.0, 9.0))
    fig = plt.figure(figsize=(sq * 2.4, sq + 1.0))
    gs  = fig.add_gridspec(1, 2, width_ratios=[2, 2], wspace=0.35)
    ax_matrix = fig.add_subplot(gs[0])
    ax_circle = fig.add_subplot(gs[1])

    # left: channel × channel ISC heatmap (bad channels masked to grey)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d0d0d0")
    im = ax_matrix.imshow(isc_mat, cmap=cmap, vmin=-1, vmax=1,
                           aspect="equal", interpolation="nearest")
    step = max(1, n // 20)
    idxs = list(range(0, n, step))
    ax_matrix.set_xticks(idxs)
    ax_matrix.set_xticklabels([ch_names[i] for i in idxs],
                               fontsize=6, rotation=45, ha="right")
    ax_matrix.set_yticks(idxs)
    ax_matrix.set_yticklabels([ch_names[i] for i in idxs], fontsize=6)
    ax_matrix.set_xlabel(sub2_label, fontsize=8)
    ax_matrix.set_ylabel(sub1_label, fontsize=8)
    ax_matrix.set_title(f"ISC matrix ({type_label})", fontsize=9, pad=4)
    plt.colorbar(im, ax=ax_matrix, shrink=0.75, label="Pearson r", pad=0.02)

    # right: connectogram embedded as image
    try:
        b64 = _isc_conn(
            np.nan_to_num(isc_mat, nan=0.0), ch_names,
            (sub1_label, sub2_label),
            threshold=isc_threshold,
            title=f"ISC {type_label}",
        )
        circle_bytes = base64.b64decode(b64)
        circle_img   = np.array(Image.open(io.BytesIO(circle_bytes)).convert("RGB"))
        ax_circle.imshow(circle_img)
        ax_circle.axis("off")
    except Exception as exc:
        logger.debug("ISC connectogram embed failed: %s", exc)
        ax_circle.text(0.5, 0.5, f"Connectogram\nunavailable\n{exc}",
                       ha="center", va="center",
                       transform=ax_circle.transAxes, fontsize=8, color="#888")
        ax_circle.axis("off")

    fig.suptitle(
        f"Inter-brain Synchrony ({type_label})  —  {sub1_label} × {sub2_label}",
        fontsize=10, y=1.01,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
