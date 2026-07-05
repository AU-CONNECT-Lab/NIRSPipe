"""Motion annotation figures: carpet + GVTD combined, bad segment zoom, per-channel detail.

The OD/GVTD carpet is the pre-processing (raw) motion view; post-denoising uses carpet_compare_figure.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_MAX_PTS = 4000


def _decimate(arr: np.ndarray, times: np.ndarray, max_pts: int):
    if len(times) <= max_pts:
        return arr, times
    step = max(1, len(times) // max_pts)
    return arr[:, ::step], times[::step]

_ZOOM_COLORS = ["#e74c3c", "#2980b9", "#27ae60"]


def carpet_gvtd_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    segments: "dict | None" = None,
    z_threshold: float = 3.0,
) -> str:
    """GVTD trace (top) + per-channel z-scored OD carpet (bottom, all channels), shared x-axis."""
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy())
    od_data, times = raw_od.get_data(picks=ch_names, return_times=True)

    # GVTD = sqrt(mean_ch(diff_t(OD)^2)), full-res (spikes are single samples, matches the metric); only carpet decimated
    gvtd   = np.sqrt(np.mean(np.diff(od_data, axis=1) ** 2, axis=0))
    t_gvtd = times[1:]
    p95    = float(np.percentile(gvtd, 95))

    # carpet: all channels (both wavelengths are positively correlated, safe in one z-scored image);
    # matches GVTD's channel set. decimate columns for display only
    carpet   = od_data
    MAX_PTS = 2000
    if carpet.shape[1] > MAX_PTS:
        carpet = carpet[:, :: carpet.shape[1] // MAX_PTS]
    # per-channel z-score: (OD - mean_t) / std_t, clipped to ±z_threshold
    mean     = carpet.mean(axis=1, keepdims=True)
    std      = carpet.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    data_z   = np.clip((carpet - mean) / std, -z_threshold, z_threshold)

    n_ch     = carpet.shape[0]
    carpet_h = max(1.5, n_ch * 0.09)
    fig = plt.figure(figsize=(12, carpet_h + 1.5))
    gs  = fig.add_gridspec(
        2, 2,
        height_ratios=[1.5, carpet_h],
        width_ratios=[1, 0.025],
        hspace=0.0, wspace=0.05,
    )
    ax_g = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[1, 0], sharex=ax_g)
    cax  = fig.add_subplot(gs[1, 1])

    ax_g.plot(t_gvtd, gvtd, lw=1.0, color="#2c3e50")
    ax_g.set_xlim(times[0], times[-1])
    ax_g.axhline(p95, ls="--", lw=0.8, color="#e74c3c", label=f"p95={p95:.4f}")
    ax_g.set_ylabel("GVTD", fontsize=8)
    ax_g.legend(fontsize=7, loc="upper right", framealpha=0.6)
    ax_g.tick_params(labelsize=7)
    for sp in ax_g.spines.values():
        sp.set_visible(False)

    im = ax_c.imshow(
        data_z, aspect="auto", cmap="gray_r",
        vmin=-z_threshold, vmax=z_threshold,
        extent=[times[0], times[-1], n_ch - 0.5, -0.5],
        interpolation="nearest",
    )
    ax_c.set_yticks([])
    ax_c.set_xlabel("Time (s)", fontsize=9)
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, cax=cax, label="Z-score")

    if segments:
        for k, (label, spans) in enumerate(segments.items()):
            for j, (onset, duration) in enumerate(spans):
                for ax in (ax_g, ax_c):
                    ax.axvspan(onset, onset + duration,
                               color="#e74c3c", alpha=0.15, zorder=2)
                if j == 0:
                    ax_g.text(
                        onset + duration / 2, 1.02, label,
                        ha="center", va="bottom", fontsize=6, color="#e74c3c",
                        transform=ax_g.get_xaxis_transform(),
                    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def bad_segment_zoom_figure(
    raw_after: mne.io.Raw,
    bad_segments: list[tuple[float, float]],
    ch_names: list[str],
    raw_before: "mne.io.Raw | None" = None,
    n_top: int = 10,
    padding: float = 15.0,
) -> "str | None":
    """Zoom into top-N bad segments; show before (optional) and after correction.

    Returns a base64-encoded PNG string, or None if there are no segments.
    Each column = one segment; rows = before (if raw_before given) / after.
    """
    if not bad_segments or not ch_names:
        return None

    sorted_segs = sorted(bad_segments, key=lambda s: s[1], reverse=True)[:n_top]
    n_segs = len(sorted_segs)
    rep_chs = ch_names[:min(3, len(ch_names))]

    n_rows = 2 if raw_before is not None else 1
    row_labels = ["Before", "After"] if n_rows == 2 else ["After"]
    raws = ([raw_before, raw_after] if n_rows == 2 else [raw_after])

    fig, axes = plt.subplots(
        n_rows, n_segs,
        figsize=(max(3 * n_segs, 8), 2.5 * n_rows),
        squeeze=False,
        sharey="row",
    )
    fig.subplots_adjust(hspace=0.12, wspace=0.08)

    t_total = raw_after.times[-1]
    for col_idx, (onset, duration) in enumerate(sorted_segs):
        tmin = max(0.0, onset - padding)
        tmax = min(t_total, onset + duration + padding)

        for row_idx, raw_obj in enumerate(raws):
            ax = axes[row_idx][col_idx]
            try:
                ch_idx = [raw_obj.ch_names.index(c) for c in rep_chs
                          if c in raw_obj.ch_names]
                if not ch_idx:
                    ax.set_visible(False)
                    continue
                data, times = raw_obj.get_data(picks=ch_idx, return_times=True)
                mask = (times >= tmin) & (times <= tmax)
                t_seg = times[mask]
                for i, ts in enumerate(data[:, mask]):
                    std = ts.std()
                    y = (ts - ts.mean()) / std if std > 0 else ts - ts.mean()
                    ax.plot(t_seg, y + i * 3,
                            lw=0.8, color=_ZOOM_COLORS[i % len(_ZOOM_COLORS)],
                            label=rep_chs[i] if col_idx == 0 else None)
                ax.axvspan(onset, onset + duration,
                           color="#e74c3c", alpha=0.15, zorder=0)
                ax.axvline(onset, color="#e74c3c", lw=0.6, ls="--")
                ax.axvline(onset + duration, color="#e74c3c", lw=0.6, ls="--")
            except Exception:
                ax.set_visible(False)
                continue

            ax.set_yticks([])
            ax.set_xlim(tmin, tmax)
            if row_idx == 0:
                ax.set_title(f"t={onset:.1f}s  +{duration:.1f}s", fontsize=7, pad=2)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)

    ax0 = axes[0][0]
    if ax0.get_visible() and ax0.get_lines():
        ax0.legend(fontsize=6, loc="upper left", framealpha=0.7)

    fig.suptitle(
        f"Bad segment zoom — top {n_segs} by duration (red = artifact window)",
        fontsize=9, y=1.01,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_motion_detail_figure(
    raw_od_before: mne.io.Raw,
    raw_od_after: mne.io.Raw,
    ch_name: str,
    segments: "dict | None" = None,
    max_pts: int = 4000,
) -> go.Figure:
    """3-row per-channel motion figure: GVTD (global) + TVD + before/after OD overlaid.

    Both inputs must be in OD space (output of optical_density()).
    """
    od_data, od_times = raw_od_before.get_data(return_times=True)
    od_data, od_times = _decimate(od_data, od_times, max_pts)

    diff_all = np.diff(od_data, axis=1)
    gvtd     = np.sqrt(np.mean(diff_all ** 2, axis=0))    # GVTD = sqrt(mean_ch(diff_t(OD)^2))
    t_gvtd   = od_times[1:].tolist()
    p95      = float(np.percentile(gvtd, 95))

    if ch_name in raw_od_before.ch_names:
        ch_idx = raw_od_before.ch_names.index(ch_name)
        tvd = np.abs(diff_all[ch_idx]).tolist()    # TVD = |diff_t(OD)| for this channel
    else:
        tvd = np.zeros(len(t_gvtd)).tolist()

    def _get_ch(raw, ch):
        idx = raw.ch_names.index(ch)
        d, t = raw.get_data(picks=[idx], return_times=True)
        d, t = _decimate(d, t, max_pts)
        return t.tolist(), d[0].tolist()

    t_b, y_b = _get_ch(raw_od_before, ch_name)
    t_a, y_a = _get_ch(raw_od_after,  ch_name)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.2, 0.2, 0.6],
        vertical_spacing=0.04,
        subplot_titles=["GVTD (global)", f"TVD — {ch_name}", f"{ch_name}  before / after"],
    )

    fig.add_trace(go.Scatter(
        x=t_gvtd, y=gvtd.tolist(), mode="lines",
        line=dict(color="#2c3e50", width=1.0), name="GVTD",
    ), row=1, col=1)
    fig.add_shape(
        type="line", x0=0, x1=1, xref="x domain",
        y0=p95, y1=p95, yref="y",
        line=dict(dash="dash", color="#e74c3c", width=0.8),
        row=1, col=1,
    )
    fig.add_annotation(
        x=1, xref="x domain", y=p95, yref="y",
        text=f"p95={p95:.4f}", showarrow=False,
        font=dict(size=7), xanchor="right", yanchor="bottom",
        row=1, col=1,
    )

    fig.add_trace(go.Scatter(
        x=t_gvtd, y=tvd, mode="lines",
        line=dict(color="#8e44ad", width=1.0), name="TVD",
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t_b, y=y_b, mode="lines",
        line=dict(color="#95a5a6", width=1.0), name="Before", opacity=0.7,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=t_a, y=y_a, mode="lines",
        line=dict(color="#2980b9", width=1.2), name="After",
    ), row=3, col=1)

    if segments:
        seg_fills = [
            "rgba(231,76,60,0.12)", "rgba(243,156,18,0.12)",
            "rgba(155,89,182,0.12)", "rgba(26,188,156,0.12)",
        ]
        for k, (label, spans) in enumerate(segments.items()):
            fill = seg_fills[k % len(seg_fills)]
            for j, (onset, dur) in enumerate(spans):
                ann = dict(annotation_text=label, annotation_position="top left",
                           annotation_font_size=7) if j == 0 else {}
                for row in (1, 2, 3):
                    fig.add_vrect(
                        x0=onset, x1=onset + dur,
                        fillcolor=fill, line_width=0, layer="below",
                        row=row, col=1,
                        **(ann if row == 1 else {}),
                    )

    fig.update_yaxes(title_text="GVTD",   tickfont=dict(size=7), row=1, col=1)
    fig.update_yaxes(title_text="TVD",    tickfont=dict(size=7), row=2, col=1)
    fig.update_yaxes(title_text="OD",     tickfont=dict(size=7), row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", gridcolor="#eee", row=3, col=1)
    fig.update_layout(
        title_text=ch_name, height=520,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig
