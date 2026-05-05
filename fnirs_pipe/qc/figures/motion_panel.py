"""Motion annotation figures: carpet plot + time series assembly."""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .sci_psp_panel import timeseries_figure, _SEGMENT_PALETTE

_BAR_COLORS = [
    "#e74c3c", "#e67e22", "#9b59b6", "#1abc9c",
    "#2980b9", "#f39c12", "#27ae60", "#c0392b",
]

def nirs_carpet_figure(data, channel_names, times, title="fNIRS Carpet Plot", z_threshold=3.0):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    std[std == 0] = 1.0
    data_z = (data - mean) / std

    fig = go.Figure(data=go.Heatmap(
        z=data_z,
        x=times,
        y=channel_names,
        colorscale='Gray',
        zmin=-z_threshold,
        zmax=z_threshold,
        colorbar=dict(title="Z-score"),
        hovertemplate="Time: %{x:.2f}s<br>Ch: %{y}<br>Z: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Time (s)", showgrid=False),
        yaxis=dict(title="Channels", autorange="reversed", showgrid=False),
        plot_bgcolor="black",
        width=900,
        height=600,
    )
    return fig

# Do not delete the following function; it may be useful for future interactive versions of the carpet plot.
# def broken_barh_figure(
#     segments: dict[str, list[tuple[float, float]]],
#     total_duration: float,
#     title: str = "Annotations",
#     row_color: str | None = None,
# ) -> go.Figure:
#     fig = go.Figure()
#     for k, (label, spans) in enumerate(segments.items()):
#         color = row_color if row_color else _BAR_COLORS[k % len(_BAR_COLORS)]
#         onsets    = [s[0] for s in spans]
#         durations = [s[1] for s in spans]
#         fig.add_trace(go.Bar(
#             y=[label] * len(spans),
#             x=durations,
#             base=onsets,
#             orientation="h",
#             name=label,
#             marker_color=color,
#             marker_line_width=0,
#             hovertemplate="onset=%{base:.2f}s  dur=%{x:.2f}s<extra>" + label + "</extra>",
#         ))
#     fig.update_layout(
#         title=title,
#         barmode="overlay",
#         xaxis=dict(title="Time (s)", range=[0, total_duration]),
#         yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
#         plot_bgcolor="#f0f0f0",
#         paper_bgcolor="white",
#         bargap=0.25,
#         showlegend=False,
#     )
#     return fig

def _gvtd_trace(data: np.ndarray, times: np.ndarray) -> tuple[go.Scatter, float]:
    diff = np.diff(data, axis=1)
    gvtd = np.sqrt(np.mean(diff ** 2, axis=0))
    t_gvtd = times[1:]
    p95 = float(np.percentile(gvtd, 95))
    trace = go.Scatter(
        x=t_gvtd.tolist(), y=gvtd.tolist(),
        mode="lines",
        line=dict(width=1.2, color="#2c3e50"),
        name="GVTD",
        showlegend=True,
        hovertemplate="t=%{x:.2f}s<br>GVTD=%{y:.4f}<extra></extra>",
    )
    return trace, p95


def motion_correction_panel(
    raw: mne.io.Raw,
    ch_names: list[str],
    colors: list[str],
    segments: dict[str, list[tuple[float, float]]],
    title: str = "Motion Correction",
) -> go.Figure:
    """GVTD (top) + normalised time series (bottom). Carpet is separate."""
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy())
    od_data, od_times = raw_od.get_data(picks=ch_names, return_times=True)
    fig_ts = timeseries_figure(raw, ch_names, colors, segments=segments)
    gvtd_trace, gvtd_p95 = _gvtd_trace(od_data, od_times)

    n_ch = len(ch_names)
    combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.18, 0.82],
        vertical_spacing=0.04,
        subplot_titles=["GVTD", "Time series (z-normalised)"],
    )

    combined.add_trace(gvtd_trace, row=1, col=1)
    combined.add_shape(
        type="line",
        x0=0, x1=1, xref="x domain",
        y0=gvtd_p95, y1=gvtd_p95, yref="y",
        line=dict(dash="dash", color="#e74c3c", width=1),
        row=1, col=1,
    )
    combined.add_annotation(
        x=1, xref="x domain", y=gvtd_p95, yref="y",
        text=f"p95={gvtd_p95:.4f}", showarrow=False,
        font=dict(size=8), xanchor="right", yanchor="bottom",
        row=1, col=1,
    )

    for trace in fig_ts.data:
        combined.add_trace(trace, row=2, col=1)

    for k, (label, spans) in enumerate(segments.items()):
        fill = _SEGMENT_PALETTE[k % len(_SEGMENT_PALETTE)]
        for j, (onset, duration) in enumerate(spans):
            kw = dict(annotation_text=label, annotation_position="top left",
                      annotation_font_size=8) if j == 0 else {}
            for row in (1, 2):
                combined.add_vrect(x0=onset, x1=onset + duration,
                                   fillcolor=fill, line_width=0, layer="below",
                                   row=row, col=1, **kw if row == 2 else {})

    combined.update_yaxes(title_text="GVTD", tickfont=dict(size=8), row=1, col=1)
    combined.update_yaxes(
        tickvals=[i * 3 for i in range(n_ch)],
        ticktext=ch_names,
        autorange="reversed",
        tickfont=dict(size=8),
        row=2, col=1,
    )
    combined.update_xaxes(title_text="Time (s)", row=2, col=1)
    combined.update_layout(
        title=title,
        height=max(400, n_ch * 20 + 150),
        margin=dict(l=120, r=30, t=80, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=9)),
    )
    return combined


def carpet_static_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    segments: "dict | None" = None,
    z_threshold: float = 3.0,
    title: str = "Carpet plot (z-score per channel)",
) -> str:
    """Render carpet plot as base64 PNG using matplotlib.

    Static output keeps file size small (avoids large Plotly heatmap JSON).
    """
    data, times = raw.get_data(picks=ch_names, return_times=True)

    MAX_PTS = 2000
    if data.shape[1] > MAX_PTS:
        step = data.shape[1] // MAX_PTS
        data = data[:, ::step]
        times = times[::step]

    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    data_z = np.clip((data - mean) / std, -z_threshold, z_threshold)

    n_ch = len(ch_names)
    fig_h = max(1.5, n_ch * 0.09)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    im = ax.imshow(
        data_z, aspect="auto", cmap="gray_r",
        vmin=-z_threshold, vmax=z_threshold,
        extent=[times[0], times[-1], n_ch - 0.5, -0.5],
        interpolation="nearest",
    )

    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_title(title, fontsize=10, pad=4)

    if segments:
        for k, (label, spans) in enumerate(segments.items()):
            for j, (onset, duration) in enumerate(spans):
                ax.axvspan(onset, onset + duration,
                           color="#e74c3c", alpha=0.18, zorder=2)
                if j == 0:
                    ax.text(onset + duration / 2, -0.8, label,
                            ha="center", va="top", fontsize=6,
                            color="#e74c3c", transform=ax.get_xaxis_transform())

    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax, label="Z-score", shrink=0.7, pad=0.01, aspect=30)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


_ZOOM_COLORS = ["#e74c3c", "#2980b9", "#27ae60"]


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
    n_rep = len(rep_chs)

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
                ax.set_title(
                    f"t={onset:.1f}s  +{duration:.1f}s",
                    fontsize=7, pad=2,
                )
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=8)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=7)
            ax.tick_params(labelsize=6)

    # legend on first visible axes
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
