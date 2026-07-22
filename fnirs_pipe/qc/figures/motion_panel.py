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

from fnirs_pipe.qc.figures._utils import decimate as _decimate
from fnirs_pipe.qc.quantitative_metrics import GVTD_MOTION_BAND, gvtd_threshold, gvtd_timetrace

_MAX_PTS = 4000

_ZOOM_COLORS = ["#e74c3c", "#2980b9", "#27ae60"]


def _maxpool_xy(t: np.ndarray, y: np.ndarray, max_pts: int = 2000):
    """Downsample a trace to <= max_pts by taking the max in each bin (keeps spike heights).

    Display only: reported GVTD scalars are computed full-res elsewhere and are unaffected.
    """
    n = len(y)
    if n <= max_pts:
        return t, y
    step = n // max_pts
    m = (n // step) * step
    y_ds = y[:m].reshape(-1, step).max(axis=1)
    t_ds = t[:m].reshape(-1, step)[:, 0]
    return t_ds, y_ds


def carpet_gvtd_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    segments: "dict | None" = None,
    z_threshold: float = 3.0,
    corrected_segments: "list[tuple[float, float]] | None" = None,
    spike_segments: "list[tuple[float, float]] | None" = None,
) -> str:
    """Raw GVTD + filtered GVTD + per-channel z-scored OD carpet, shared x-axis.

    The two GVTD traces sit in separate stacked panels (they overlap badly on one axis).
    ``corrected_segments`` (motion-correction footprint) and ``spike_segments`` are drawn
    as short bands along the carpet's bottom edge, distinct from the full-height red
    ``segments`` windows.
    """
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy())
    od_data, times = raw_od.get_data(picks=ch_names, return_times=True)

    # GVTD full-res for the threshold/metric; plotted trace is max-pooled for display only.
    # filt = 0.01-0.5 Hz motion band (used for the threshold).
    sfreq     = float(raw_od.info["sfreq"])
    gvtd      = gvtd_timetrace(od_data, sfreq)
    gvtd_filt = gvtd_timetrace(od_data, sfreq, *GVTD_MOTION_BAND)
    t_gvtd    = times[1:]
    motion_thresh = gvtd_threshold(gvtd_filt, n_std=3.0)
    t_raw_ds,  gvtd_ds      = _maxpool_xy(t_gvtd, gvtd)
    t_filt_ds, gvtd_filt_ds = _maxpool_xy(t_gvtd, gvtd_filt)

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
    fig = plt.figure(figsize=(12, carpet_h + 3.0))
    gs  = fig.add_gridspec(
        4, 2,
        height_ratios=[1.0, 1.0, carpet_h, 0.5],
        width_ratios=[1, 0.025],
        hspace=0.0, wspace=0.05,
    )
    ax_graw  = fig.add_subplot(gs[0, 0])
    ax_gfilt = fig.add_subplot(gs[1, 0], sharex=ax_graw)
    ax_c     = fig.add_subplot(gs[2, 0], sharex=ax_graw)
    ax_b     = fig.add_subplot(gs[3, 0], sharex=ax_graw)  # dedicated band strip, never overlaps carpet
    cax      = fig.add_subplot(gs[2, 1])

    ax_graw.plot(t_raw_ds, gvtd_ds, lw=0.5, color="#b0b0b0")
    ax_graw.set_xlim(times[0], times[-1])
    ax_graw.set_ylabel("GVTD\nraw", fontsize=8)
    ax_graw.tick_params(labelsize=7, labelbottom=False)

    ax_gfilt.plot(t_filt_ds, gvtd_filt_ds, lw=0.6, color="#2c3e50")
    if motion_thresh is not None:
        ax_gfilt.axhline(motion_thresh, ls="--", lw=0.8, color="#e74c3c",
                         label=f"thresh={motion_thresh:.4f}")
        ax_gfilt.legend(fontsize=7, loc="upper right", framealpha=0.6)
    ax_gfilt.set_ylabel("GVTD\n0.01–0.5 Hz", fontsize=8)
    ax_gfilt.tick_params(labelsize=7, labelbottom=False)
    for ax in (ax_graw, ax_gfilt):
        for sp in ax.spines.values():
            sp.set_visible(False)

    im = ax_c.imshow(
        data_z, aspect="auto", cmap="gray_r",
        vmin=-z_threshold, vmax=z_threshold,
        extent=[times[0], times[-1], n_ch - 0.5, -0.5],
        interpolation="nearest",
    )
    ax_c.set_yticks([])
    ax_c.tick_params(labelbottom=False)  # x-axis lives on the band strip below
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, cax=cax, label="Z-score")

    if segments:
        for k, (label, spans) in enumerate(segments.items()):
            for j, (onset, duration) in enumerate(spans):
                for ax in (ax_graw, ax_gfilt, ax_c):
                    ax.axvspan(onset, onset + duration,
                               color="#e74c3c", alpha=0.15, zorder=2)
                if j == 0:
                    ax_graw.text(
                        onset + duration / 2, 1.02, label,
                        ha="center", va="bottom", fontsize=6, color="#e74c3c",
                        transform=ax_graw.get_xaxis_transform(),
                    )

    # dedicated strip below the carpet: motion-correction footprint + spike timepoints
    ax_b.set_ylim(0, 1)
    ax_b.set_yticks([])
    ax_b.set_xlabel("Time (s)", fontsize=9)
    ax_b.tick_params(labelsize=7)
    for sp in ax_b.spines.values():
        sp.set_visible(False)

    def _band(spans, ymin, ymax, color, label):
        for onset, duration in (spans or []):
            ax_b.axvspan(onset, onset + duration, ymin=ymin, ymax=ymax, color=color, alpha=0.9)
        ax_b.text(-0.006, (ymin + ymax) / 2, label, transform=ax_b.transAxes,
                  fontsize=6, color=color, va="center", ha="right")

    _band(corrected_segments, 0.05, 0.45, "#16a085", "corrected")
    _band(spike_segments, 0.55, 0.95, "#e67e22", "spikes")

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
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def build_motion_detail_figure(
    raw_od_before: mne.io.Raw,
    raw_od_after: mne.io.Raw,
    ch_name: str,
    segments: "dict | None" = None,
    max_pts: int = 4000,
    corrected_segments: "list[tuple[float, float]] | None" = None,
    spike_segments: "list[tuple[float, float]] | None" = None,
) -> go.Figure:
    """4-row per-channel motion figure: GVTD (global) + TVD + before/after OD + band strip.

    Both inputs must be in OD space (output of optical_density()). The bottom strip shows
    the (global) motion-correction footprint and spike timepoints, never overlapping the traces.
    """
    # GVTD + threshold on full-res OD so they match the reported gvtd_* metrics; the plotted
    # GVTD traces are decimated afterwards (display only), like the carpet figure.
    od_full, t_full = raw_od_before.get_data(return_times=True)
    full_sfreq = float(raw_od_before.info["sfreq"])
    gvtd_full      = gvtd_timetrace(od_full, full_sfreq)                     # canonical (unfiltered)
    gvtd_filt_full = gvtd_timetrace(od_full, full_sfreq, *GVTD_MOTION_BAND)  # motion-band
    motion_thresh  = gvtd_threshold(gvtd_filt_full, n_std=3.0)
    gvtd_arr, t_gvtd_arr = _decimate(gvtd_full[np.newaxis], t_full[1:], max_pts)
    gvtd_filt_arr, _     = _decimate(gvtd_filt_full[np.newaxis], t_full[1:], max_pts)
    gvtd, gvtd_filt = gvtd_arr[0], gvtd_filt_arr[0]
    t_gvtd = t_gvtd_arr.tolist()

    # decimated OD (display resolution) for this channel's TVD and before/after traces
    od_data, od_times = _decimate(od_full, t_full, max_pts)
    diff_all = np.diff(od_data, axis=1)
    t_tvd = od_times[1:].tolist()

    if ch_name in raw_od_before.ch_names:
        ch_idx = raw_od_before.ch_names.index(ch_name)
        tvd = np.abs(diff_all[ch_idx]).tolist()    # TVD = |diff_t(OD)| for this channel
    else:
        tvd = np.zeros(len(t_tvd)).tolist()

    def _get_ch(raw, ch):
        idx = raw.ch_names.index(ch)
        d, t = raw.get_data(picks=[idx], return_times=True)
        d, t = _decimate(d, t, max_pts)
        return t.tolist(), d[0].tolist()

    t_b, y_b = _get_ch(raw_od_before, ch_name)
    t_a, y_a = _get_ch(raw_od_after,  ch_name)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.19, 0.19, 0.5, 0.12],
        vertical_spacing=0.04,
        subplot_titles=["GVTD (global)", f"TVD — {ch_name}", f"{ch_name}  before / after", "motion bands"],
    )

    fig.add_trace(go.Scatter(
        x=t_gvtd, y=gvtd.tolist(), mode="lines",
        line=dict(color="#b0b0b0", width=0.4), name="GVTD raw",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_gvtd, y=gvtd_filt.tolist(), mode="lines",
        line=dict(color="#2c3e50", width=0.6), name="GVTD 0.01–0.5 Hz",
    ), row=1, col=1)
    if motion_thresh is not None:
        fig.add_shape(
            type="line", x0=0, x1=1, xref="x domain",
            y0=motion_thresh, y1=motion_thresh, yref="y",
            line=dict(dash="dash", color="#e74c3c", width=0.8),
            row=1, col=1,
        )
        fig.add_annotation(
            x=1, xref="x domain", y=motion_thresh, yref="y",
            text=f"thresh={motion_thresh:.4f}", showarrow=False,
            font=dict(size=7), xanchor="right", yanchor="bottom",
            row=1, col=1,
        )

    fig.add_trace(go.Scatter(
        x=t_tvd, y=tvd, mode="lines",
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

    # bottom strip: motion-correction footprint (teal) + spike timepoints (orange)
    def _detail_band(spans, y0, y1, color):
        for onset, dur in (spans or []):
            fig.add_shape(type="rect", x0=onset, x1=onset + dur, y0=y0, y1=y1,
                          fillcolor=color, line_width=0, layer="above", row=4, col=1)
    _detail_band(corrected_segments, 0.05, 0.45, "#16a085")
    _detail_band(spike_segments, 0.55, 0.95, "#e67e22")

    fig.update_yaxes(title_text="GVTD",   tickfont=dict(size=7), row=1, col=1)
    fig.update_yaxes(title_text="TVD",    tickfont=dict(size=7), row=2, col=1)
    fig.update_yaxes(title_text="OD",     tickfont=dict(size=7), row=3, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, row=4, col=1)
    fig.update_xaxes(title_text="Time (s)", gridcolor="#eee", row=4, col=1)
    fig.update_layout(
        title_text=ch_name, height=560,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig
