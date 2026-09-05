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
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.motion_panel")

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


def _matched_od_after(
    raw_after: "mne.io.Raw | None",
    ch_names: list[str],
    shape: tuple,
    sfreq: float,
) -> "np.ndarray | None":
    """The motion-corrected recording as OD over ``ch_names``, or None if it does not line up.

    A before/after panel only says something if both sides describe the same thing, so this
    returns data on an exact match and None on anything else, which the caller draws as the
    plain single-trace figure::

        ch_names = [S1_D1 760, S1_D1 850];  after has both, 10 Hz, same n_times  -> (2, N)
        ch_names = [S1_D1 760, S1_D1 850];  after has only S1_D1 760             -> None

    A near miss is worse than nothing here: GVTD over a different channel set differs
    several-fold on one recording, and the panel would show that as an effect of the
    correction. The corrected file is already OD, so the conversion is only for the case
    where a caller hands over intensity.
    """
    if raw_after is None:
        return None
    try:
        od = (raw_after if is_optical_density(raw_after)
              else mne.preprocessing.nirs.optical_density(raw_after.copy()))
        if not set(ch_names) <= set(od.ch_names):            # same channels
            logger.warning("corrected file is missing channels the panel draws; "
                           "before/after comparison dropped")
            return None
        if abs(float(od.info["sfreq"]) - sfreq) > 1e-6:      # same rate
            logger.warning("corrected file is at a different sampling rate; "
                           "before/after comparison dropped")
            return None
        data = od.get_data(picks=ch_names)
        if data.shape != shape:                              # same duration
            logger.warning("corrected file has a different length; "
                           "before/after comparison dropped")
            return None
        return data
    except Exception:
        logger.warning("corrected file unusable; before/after comparison dropped",
                       exc_info=True)
        return None


def carpet_gvtd_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    segments: "dict | None" = None,
    z_threshold: float = 3.0,
    corrected_segments: "list[tuple[float, float]] | None" = None,
    spike_segments: "list[tuple[float, float]] | None" = None,
    raw_after: "mne.io.Raw | None" = None,
) -> go.Figure:
    """Raw GVTD + filtered GVTD + per-channel z-scored OD carpet, on one shared time axis.

    The two GVTD traces sit in separate stacked panels (they overlap badly on one axis).
    ``corrected_segments`` (motion-correction footprint) and ``spike_segments`` are drawn
    as short bands on a strip under the carpet, distinct from the full-height red
    ``segments`` windows.

    Given ``raw_after``, the motion-corrected recording, each GVTD panel carries a second
    trace and a second carpet is stacked under the first, so the figure answers whether the
    correction removed what it was there to remove. The two traces land on top of each other
    wherever the correction changed nothing, which for the unfiltered trace is nearly
    everywhere; they share a legend group per side, so clicking ``before`` or ``after`` in
    the legend pulls that side out of both panels at once and separates them.

    Everything the comparison is read against stays fixed to the uncorrected side: the
    threshold line, and the per-channel mean and SD both carpets are z-scored by.
    Re-deriving either from the corrected data would rescale the very panel that is supposed
    to show the improvement, and a correction that shrank the signal would come out looking
    unchanged. A ``raw_after`` that does not cover the same channels for the same duration at
    the same rate is dropped rather than drawn (see ``_matched_od_after``).
    """
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy())
    od_data, times = raw_od.get_data(picks=ch_names, return_times=True)
    sfreq = float(raw_od.info["sfreq"])
    od_after = _matched_od_after(raw_after, ch_names, od_data.shape, sfreq)
    has_after = od_after is not None

    # GVTD full-res for the threshold/metric; plotted trace is max-pooled for display only.
    # filt = 0.01-0.5 Hz motion band (used for the threshold).
    gvtd      = gvtd_timetrace(od_data, sfreq)
    gvtd_filt = gvtd_timetrace(od_data, sfreq, *GVTD_MOTION_BAND)
    t_gvtd    = times[1:]
    motion_thresh = gvtd_threshold(gvtd_filt, n_std=3.0)
    t_raw_ds,  gvtd_ds      = _maxpool_xy(t_gvtd, gvtd)
    t_filt_ds, gvtd_filt_ds = _maxpool_xy(t_gvtd, gvtd_filt)
    if has_after:
        _, gvtd_post_ds      = _maxpool_xy(t_gvtd, gvtd_timetrace(od_after, sfreq))
        _, gvtd_filt_post_ds = _maxpool_xy(
            t_gvtd, gvtd_timetrace(od_after, sfreq, *GVTD_MOTION_BAND))

    # carpet: all channels (both wavelengths are positively correlated, safe in one z-scored
    # image); matches GVTD's channel set. decimate columns for display only
    MAX_PTS = 2000
    step     = max(1, od_data.shape[1] // MAX_PTS)
    carpet   = od_data[:, ::step]
    t_carpet = times[::step]
    # one scale for both carpets, taken from the uncorrected side: z-scoring the corrected
    # data by its own SD would divide out the very shrinkage the panel is there to show
    mean = carpet.mean(axis=1, keepdims=True)
    std  = carpet.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    # 2 dp: the colour scale cannot resolve more, and the z values are most of the payload
    # of the saved HTML, so rounding them keeps the file a fraction of the size
    data_z = np.round(np.clip((carpet - mean) / std, -z_threshold, z_threshold), 2)
    data_z_after = (None if not has_after else
                    np.round(np.clip((od_after[:, ::step] - mean) / std,
                                     -z_threshold, z_threshold), 2))

    n_ch      = carpet.shape[0]
    n_carpets = 2 if has_after else 1
    n_rows    = 2 + n_carpets + 1  # 2 GVTD panels, the carpets, the annotation strip
    gvtd_px   = 110
    carpet_px = int(max(200, min(n_ch * 13, 700)))
    band_px   = 60
    heights   = [gvtd_px, gvtd_px] + [carpet_px] * n_carpets + [band_px]
    total_px  = sum(heights) + 120  # margins and the shared x-axis title
    band_row  = n_rows

    titles = ["GVTD (raw)", "GVTD (0.01\u20130.5 Hz)"]
    titles += (["Carpet (before)", "Carpet (after)"] if has_after else ["Carpet"])
    titles += [""]
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=[h / sum(heights) for h in heights],
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    def _gvtd_traces(row, y_before, y_after, t, color_before):
        """One GVTD panel: the uncorrected trace, plus the corrected one when there is one.

        ``legendgroup`` is the side rather than the panel, so one click in the legend hides
        that side in every panel at once. Only the first panel contributes legend entries;
        the second would otherwise duplicate them.
        """
        show = row == 1
        if y_after is not None:
            fig.add_trace(go.Scatter(
                x=t, y=y_after, mode="lines", name="after",
                legendgroup="after", showlegend=show,
                line=dict(color="#16a085", width=1.4),
                hovertemplate="t=%{x:.1f}s<br>after=%{y:.3e}<extra></extra>",
            ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=y_before, mode="lines",
            name="before" if y_after is not None else "GVTD",
            legendgroup="before", showlegend=show,
            line=dict(color=color_before, width=1.0),
            hovertemplate="t=%{x:.1f}s<br>before=%{y:.3e}<extra></extra>",
        ), row=row, col=1)

    _gvtd_traces(1, gvtd_ds, gvtd_post_ds if has_after else None, t_raw_ds, "#5d6d7e")
    _gvtd_traces(2, gvtd_filt_ds, gvtd_filt_post_ds if has_after else None,
                 t_filt_ds, "#2c3e50")

    if motion_thresh is not None:
        # the uncorrected recording's threshold, kept for the corrected trace as well: it is
        # the yardstick the % motion in the metrics table is counted against
        fig.add_hline(
            y=motion_thresh, row=2, col=1,
            line=dict(color="#e74c3c", width=1, dash="dash"),
            annotation_text=f"thresh={motion_thresh:.3e}",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#e74c3c"),
        )

    for i, z in enumerate([data_z, data_z_after][:n_carpets]):
        fig.add_trace(go.Heatmap(
            z=z, x=t_carpet, y=ch_names,
            coloraxis="coloraxis",  # one scale and one bar for both carpets
            hovertemplate="%{y}<br>t=%{x:.1f}s<br>z=%{z:.2f}<extra></extra>",
            showlegend=False,
        ), row=3 + i, col=1)

    def _band(spans, y, color, label):
        xs, ys = [], []
        for onset, duration in (spans or []):
            xs += [onset, onset + duration, None]
            ys += [y, y, None]
        fig.add_trace(go.Scatter(
            x=xs or [None], y=ys or [None], mode="lines", name=label,
            line=dict(color=color, width=9),
            hovertemplate=label + ": %{x:.1f}s<extra></extra>",
        ), row=band_row, col=1)

    _band(corrected_segments, 0.3, "#16a085", "corrected")
    _band(spike_segments, 0.7, "#e67e22", "spikes")

    if segments:
        for label, spans in segments.items():
            for onset, duration in spans:
                fig.add_vrect(
                    x0=onset, x1=onset + duration,
                    fillcolor="#e74c3c", opacity=0.15, line_width=0,
                    layer="below", row="all", col=1,
                )
                fig.add_annotation(
                    x=onset + duration / 2, y=1.0, yref="y domain", row=1, col=1,
                    text=label, showarrow=False, yanchor="bottom",
                    font=dict(size=9, color="#e74c3c"),
                )

    fig.update_yaxes(title_text="GVTD", title_font_size=9, row=1, col=1)
    fig.update_yaxes(title_text="GVTD", title_font_size=9, row=2, col=1)
    for i in range(n_carpets):
        # the channel names stay in the hover, where they are readable; on the axis a full
        # montage would be an unreadable stack
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=3 + i, col=1)
    fig.update_yaxes(showticklabels=False, range=[0, 1], row=band_row, col=1)
    fig.update_xaxes(title_text="Time (s)", row=band_row, col=1)
    fig.update_xaxes(range=[float(times[0]), float(times[-1])])

    fig.update_layout(
        height=total_px,
        coloraxis=dict(
            colorscale="Greys", cmin=-z_threshold, cmax=z_threshold,
            colorbar=dict(title="Z-score", thickness=10, len=0.5, y=0.4),
        ),
        margin=dict(l=60, r=20, t=40, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="white",
    )
    for ann in fig.layout.annotations:
        if ann.text in titles:
            ann.font.size = 10
    return fig


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
