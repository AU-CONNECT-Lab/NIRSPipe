"""Plotly figure builders for the interactive raw fNIRS QC viewer."""

from __future__ import annotations

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.utils import pair_of
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.utils.logging import get_logger

from fnirs_pipe.qc.figures.common._brain_utils import mni_trans
from fnirs_pipe.qc.figures.common._utils import (
    BAND_COLORS, CONDITION_PALETTE, HBO_COLOR, HBR_COLOR, _hex_to_rgba,
    LONG_COLOR, PSD_NFFT, SHORT_COLOR, UNCLASSIFIED_COLOR,
    TIMELINE_ROW_PX, block_duration_labels,
    decimate as _decimate, line_xy,
    _optode_positions, _topomap_project,
    physio_bands, timeline_axes, timeline_row_bands,
    timeline_row_traces,
)
from fnirs_pipe.qc.metrics._helpers import Bands, epochable_events, long_short_channels
from fnirs_pipe.qc.common.channel_table import _neither_range_title
from fnirs_pipe.qc.common.figure_io import extract_markers

logger = get_logger("qc.figures")

_PSD_FMAX = 2.0

# A condition needs this many epochs before its row is drawn as an average. One epoch is
# that block's own trace, not a mean of anything, and the panel is headed "grand mean".
# The same floor the report's per-trial panels take, for the same reason.
EPOCH_MIN_TRIALS = 2




def _ch_colors(raw: mne.io.Raw, sep_bands: "Bands | None" = None) -> list[str]:
    """Long, short or unclassified per channel, by the split every report uses."""
    long_names, short_names = map(set, long_short_channels(raw, sep_bands))
    return [_hex_to_rgba(LONG_COLOR if ch in long_names
                         else SHORT_COLOR if ch in short_names
                         else UNCLASSIFIED_COLOR, 0.78)
            for ch in raw.ch_names]


def psd_layout(height: int = 220, cardiac=None, resp=None) -> dict:
    """Layout for a single-panel PSD, with the physiological bands shaded.

    ``cardiac`` and ``resp`` are the run's own ``(l_freq, h_freq)``, not constant edges.
    Passing None omits that band, and the colours and
    frequencies both come from the same place the multi-stage PSD figure reads them from.
    """
    bands = physio_bands(cardiac, resp)
    shapes = [dict(type="rect", xref="x", yref="paper",
                   x0=x0, x1=min(x1, _PSD_FMAX), y0=0, y1=1,
                   fillcolor=BAND_COLORS.get(name, "rgba(120,120,120,0.10)"),
                   line=dict(width=0))
              for name, x0, x1 in bands if x0 <= _PSD_FMAX]
    annotations = [dict(x=(x0 + min(x1, _PSD_FMAX)) / 2, y=0.97, xref="x", yref="paper",
                        text=name, showarrow=False,
                        font=dict(size=8, color="#666"))
                   for name, x0, x1 in bands if x0 <= _PSD_FMAX]
    return dict(
        xaxis=dict(title="Frequency (Hz)", range=[0, _PSD_FMAX], gridcolor="#eeeeee"),
        yaxis=dict(title="Power", type="log", gridcolor="#eeeeee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=height, margin=dict(l=60, r=15, t=22, b=38),
        legend=dict(font=dict(size=9), orientation="h",
                    x=1, xanchor="right", y=1.0, yanchor="bottom"),
        shapes=shapes, annotations=annotations,
    )


SCI_WARN_RATIO = 0.625   # amber band as a share of the run's line; 0.5 at the 0.8 default


def sci_color(sci: float | None, threshold: float = SCI_PASS,
              rejected: bool | None = None) -> str:
    """Red if the channel was rejected; otherwise green above the run's SCI line, amber near it.

    ::

      sci_color(0.96, 0.80)                  -> green
      sci_color(0.96, 0.80, rejected=True)   -> red

    ``rejected`` outranks SCI because the two are not the same question. Screening counts how
    many windows a channel was coupled in, so a channel whose whole-run SCI is 0.96 can still
    be loose through most of a condition and be dropped.

    Passing None keeps the SCI-only ladder, for the callers that have no verdict to hand.
    """
    if rejected:
        return "#e74c3c"
    if sci is None:
        return "#aaa"
    if sci >= threshold:
        return "#27ae60"
    if sci >= SCI_WARN_RATIO * threshold:
        return "#f39c12"
    return "#e74c3c"


def sci_legend(threshold: float = SCI_PASS) -> str:
    """Caption for :func:`sci_color`, from the same two numbers it colours by."""
    warn = SCI_WARN_RATIO * threshold
    return (f"red = rejected, else SCI (green ≥ {threshold:.2f} / yellow ≥ {warn:.2f} "
            f"/ red < {warn:.2f})")


def condition_colors(markers: list[dict]) -> dict[str, str]:
    descs = list(dict.fromkeys(m["description"] for m in markers))
    return {d: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, d in enumerate(descs)}


def build_ts_figure(
    raw: mne.io.Raw,
    markers: list[dict],
    bad_channels: set[str],
    max_ts_pts: int,
    sep_bands: "Bands | None" = None,
) -> tuple[go.Figure, list[dict], dict[str, str], list[dict], float, float]:
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    if len(picks) == 0:
        picks = list(range(len(raw.ch_names)))

    data, times = raw.get_data(picks=picks, return_times=True)
    data, times = _decimate(data, times, max_ts_pts)
    times_list = times.tolist()

    all_colors = _ch_colors(raw, sep_bands)
    cond_colors_ = condition_colors(markers)

    traces = []
    band_shapes = []
    for i, pick in enumerate(picks):
        ch = raw.ch_names[pick]
        arr = data[i]
        std = float(np.std(arr))
        normed = ((arr - arr.mean()) / std).tolist() if std > 0 else (arr - arr.mean()).tolist()
        shifted = [v + i * 3 for v in normed]
        color = "#b2bec3" if ch in bad_channels else (all_colors[pick] if pick < len(all_colors) else "#aaa")
        traces.append(go.Scatter(
            x=times_list, y=shifted, name=ch, mode="lines",
            line=dict(width=0.9, color=color),
            customdata=[ch] * len(times_list),
            hovertemplate="<b>%{customdata}</b><extra></extra>",
        ))
        band_shapes.append(dict(
            type="rect", xref="paper", yref="y", layer="below",
            x0=0, x1=1, y0=i * 3 - 1.5, y1=i * 3 + 1.5,
            fillcolor="rgba(0,0,0,0.025)" if i % 2 == 0 else "rgba(0,0,0,0)",
            line=dict(width=0),
        ))

    mk_shapes = []
    for m in markers:
        color = cond_colors_.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            mk_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.09),
                line=dict(width=0), layer="below",
            ))

    n = len(picks)
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title="Time (s)", gridcolor="#eeeeee", zerolinecolor="#cccccc"),
            # one tick per trace, so the name sits on the trace instead of in a legend
            yaxis=dict(
                tickmode="array",
                tickvals=[i * 3 for i in range(n)],
                ticktext=[raw.ch_names[p] for p in picks],
                tickfont=dict(size=8),
                showgrid=False, zeroline=False,
            ),
            plot_bgcolor="white", paper_bgcolor="white",
            shapes=band_shapes + mk_shapes,
            height=max(380, n * 22 + 80),
            margin=dict(l=110, r=15, t=12, b=40),
            showlegend=False,
            hovermode="closest",
        ),
    )
    marker_data = [{**m, "color": cond_colors_.get(m["description"], "#f39c12")} for m in markers]
    return fig, marker_data, cond_colors_, band_shapes, float(times[0]), float(times[-1])


def build_channel_figure(
    raw_haemo: mne.io.Raw,
    markers: list[dict],
    ch_pair: str,
    max_ts_pts: int,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
    epoch: bool = True,
) -> tuple[go.Figure, go.Figure | None, go.Figure | None]:
    """One channel pair as up to three panels: HbO/HbR over time, its PSD, its epoch average.

    ``cardiac`` and ``resp`` are the run's own band edges, taken from the CLI, and shade the
    PSD panel the way the multi-stage PSD figure shades its rows; passing None omits that
    band rather than guessing a default.

    ``epoch=False`` returns None for the third panel, for a caller with a denoised view of
    the same channel elsewhere.
    """
    hbo_name = f"{ch_pair} hbo"
    hbr_name = f"{ch_pair} hbr"
    haemo_names = raw_haemo.ch_names

    if hbo_name not in haemo_names or hbr_name not in haemo_names:
        raise ValueError(f"channel pair {ch_pair!r} not found")

    hbo_pick = haemo_names.index(hbo_name)
    hbr_pick = haemo_names.index(hbr_name)
    # after a crop `times` restarts at 0 while `markers` stay absolute; shapes want absolute
    # and set_annotations below wants relative, so both need t0, in opposite directions
    t0 = float(raw_haemo.first_time)
    haemo_data, times = raw_haemo.get_data(picks=[hbo_pick, hbr_pick], return_times=True)
    haemo_data, times_d = _decimate(haemo_data, times, max_ts_pts)
    times_list = (times_d + t0).tolist()
    hbo = (haemo_data[0] * 1e6).tolist()
    hbr = (haemo_data[1] * 1e6).tolist()

    cond_colors_ = condition_colors(markers)

    def _mk_shapes():
        shapes = []
        for m in markers:
            color = cond_colors_.get(m["description"], "#f39c12")
            dur = m["duration"] if m["duration"] > 0.1 else 0.5
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + dur, y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.18),
                line=dict(width=0), layer="below",
            ))
        return shapes

    detail_fig = go.Figure(
        data=[
            go.Scatter(x=times_list, y=hbo, name="HbO", mode="lines",
                       line=dict(color=HBO_COLOR, width=1.5),
                       fill="tozeroy", fillcolor="rgba(231,76,60,0.06)"),
            go.Scatter(x=times_list, y=hbr, name="HbR", mode="lines",
                       line=dict(color=HBR_COLOR, width=1.5),
                       fill="tozeroy", fillcolor="rgba(52,152,219,0.06)"),
        ],
        layout=go.Layout(
            xaxis=dict(title="Time (s)", gridcolor="#eeeeee", zerolinecolor="#cccccc"),
            yaxis=dict(title="Conc. (µmol/L)", gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=160, margin=dict(l=55, r=15, t=22, b=38),
            legend=dict(font=dict(size=9), orientation="h",
                        x=1, xanchor="right", y=1.0, yanchor="bottom"),
            shapes=_mk_shapes(),
        ),
    )

    psd_fig = None
    try:
        from mne.time_frequency import psd_array_welch
        raw_arr = raw_haemo.get_data(picks=[hbo_pick, hbr_pick])
        sfreq = raw_haemo.info["sfreq"]
        psds, freqs = psd_array_welch(raw_arr, sfreq,
                                      n_fft=min(PSD_NFFT, raw_arr.shape[1]), verbose=False)
        psd_hbo, psd_hbr = psds[0], psds[1]
        fmax = min(2.0, sfreq / 2)
        mask = freqs <= fmax
        psd_fig = go.Figure(
            data=[
                go.Scatter(**line_xy(freqs[mask], psd_hbo[mask]),
                           name="HbO", mode="lines", line=dict(color=HBO_COLOR, width=2)),
                go.Scatter(**line_xy(freqs[mask], psd_hbr[mask]),
                           name="HbR", mode="lines", line=dict(color=HBR_COLOR, width=2)),
            ],
            layout=go.Layout(**psd_layout(cardiac=cardiac, resp=resp)),
        )
    except Exception as exc:
        logger.warning("channel PSD failed for %s: %s", ch_pair, exc)

    epoch_fig = None
    if epoch and markers:
        try:
            # relative onsets: an absolute one lands a second first_time past the data
            anns = mne.Annotations(
                onset=[m["onset"] - t0 for m in markers],
                duration=[m["duration"] for m in markers],
                description=[m["description"] for m in markers],
            )
            raw_copy = raw_haemo.copy().set_annotations(anns)
            events_mne, event_id = epochable_events(raw_copy, epoch_tmin, epoch_tmax)
            if len(events_mne) > 0:
                epochs = mne.Epochs(
                    raw_copy, events_mne, event_id,
                    tmin=epoch_tmin, tmax=epoch_tmax,
                    picks=[hbo_pick, hbr_pick],
                    baseline=(epoch_tmin, 0),
                    preload=True, verbose=False,
                )
                # one panel per condition, HbO/HbR in their usual red and blue; a condition
                # with one epoch is not an average
                panels = []
                for cond in event_id:
                    try:
                        ep_subset = epochs[cond]
                    except Exception:
                        continue
                    if len(ep_subset) < EPOCH_MIN_TRIALS:
                        continue
                    panels.append((cond, ep_subset.get_data()))
                if panels:
                    # stacked, one row per condition, like the grand mean
                    n = len(panels)
                    epoch_fig = make_subplots(
                        rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[f"{c} (n={e.shape[0]})" for c, e in panels],
                    )
                    for i, (_cond, ep) in enumerate(panels, start=1):
                        for row_i, color, label in ((0, HBO_COLOR, "HbO"),
                                                    (1, HBR_COLOR, "HbR")):
                            epoch_fig.add_trace(go.Scatter(
                                x=epochs.times.tolist(),
                                y=(ep[:, row_i, :].mean(axis=0) * 1e6).tolist(),
                                name=label, mode="lines", legendgroup=label,
                                showlegend=(i == 1),
                                line=dict(color=color, width=2),
                            ), row=i, col=1)
                        epoch_fig.add_vline(
                            x=0, line=dict(color="#7f8c8d", width=1, dash="dash"),
                            row=i, col=1)
                        # one scale over the rows, or each condition autoscales to itself
                        epoch_fig.update_yaxes(
                            title_text="Conc. (µmol/L)", gridcolor="#eeeeee",
                            row=i, col=1, **({} if i == 1 else {"matches": "y"}))
                    epoch_fig.update_xaxes(title_text="Time rel. onset (s)",
                                           gridcolor="#eeeeee",
                                           zerolinecolor="#cccccc", row=n, col=1)
                    epoch_fig.update_annotations(font_size=9)
                    epoch_fig.update_layout(
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=50 + 130 * n, margin=dict(l=60, r=15, t=42, b=42),
                        legend=dict(font=dict(size=8), orientation="h",
                                    x=1, xanchor="right", y=1.0, yanchor="bottom"),
                    )
        except Exception as exc:
            logger.warning("epoch preview failed for %s: %s", ch_pair, exc)

    return detail_fig, psd_fig, epoch_fig




_HEAD_PAD = 0.02   # fraction of the outline's own extent left around it


def _head_outline_shapes(outlines: dict | None) -> list[dict]:
    """MNE head / nose / ear polylines as plotly paths, drawn under the montage."""
    if not outlines:
        return []
    shapes = []
    for key in ("head", "nose", "ear_left", "ear_right"):
        xy = outlines.get(key)
        if xy is None:
            continue
        pts = " L ".join(f"{x},{y}" for x, y in zip(*xy))
        shapes.append(dict(type="path", path=f"M {pts}", xref="x", yref="y",
                           layer="below", line=dict(color="#b7c0c9", width=1.6)))
    return shapes


def _head_extent(outlines: dict | None, ys: "list[float]") -> "list[float] | None":
    """Vertical range that just holds the head outline, or None to leave autorange alone.

    Plotly pads an autorange by a fixed share of the span, and with `scaleanchor` the head
    is only ever as large as the height allows, so that padding is the difference between a
    head that fills the panel and one ringed by white. Falls back to the channel positions
    when the montage has no outline to measure.
    """
    if not outlines:
        return None
    vals = [float(v) for key in ("head", "nose", "ear_left", "ear_right")
            if outlines.get(key) is not None for v in outlines[key][1]]
    vals += list(ys)
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * _HEAD_PAD
    return [lo - pad, hi + pad]


def build_layout_figure(
    raw: mne.io.Raw,
    bad_channels: set[str],
    sci_scores: dict[str, float],
    sep_bands: "Bands | None" = None,
    sci_threshold: float = SCI_PASS,
) -> tuple[go.Figure | None, go.Figure | None]:
    chs = raw.info["chs"]
    ch_names = raw.ch_names
    colors = _ch_colors(raw, sep_bands)

    ch_locs = np.array([ch["loc"][:3] for ch in chs])
    has_positions = np.any(ch_locs != 0)

    fig_2d = fig_3d = None

    if has_positions:
        # bads stay in, or the grey they are drawn in below never reaches them
        picks_2d = (list(mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[]))
                    or list(range(len(chs))))
        names_2d = [ch_names[i] for i in picks_2d]
        chs_2d   = [chs[i] for i in picks_2d]

        sphere = pos2d = outlines = None
        try:
            from mne.utils import _check_sphere
            from mne.viz.topomap import _get_pos_outlines
            sphere = _check_sphere(None, raw.info)
            pos2d, outlines = _get_pos_outlines(raw.info, picks_2d, sphere)
        except Exception as exc:
            logger.warning("head projection unavailable, plotting flat x/y: %s", exc)

        flat = pos2d is None
        if flat:
            pos2d = np.array([ch["loc"][:3] for ch in chs_2d])[:, :2]

        def _project(xyz):
            arr = np.asarray(xyz, dtype=float)
            return arr[:, :2] if flat else _topomap_project(arr, sphere)

        src_xyz, det_xyz, pairs = _optode_positions(chs_2d, names_2d)
        opt_xy: dict[str, np.ndarray] = {}
        for group in (src_xyz, det_xyz):
            if group:
                opt_xy.update(zip(group, _project(np.array(list(group.values())))))

        lines_x, lines_y = [], []
        for s_name, d_name in pairs:
            if s_name in opt_xy and d_name in opt_xy:
                lines_x += [float(opt_xy[s_name][0]), float(opt_xy[d_name][0]), None]
                lines_y += [float(opt_xy[s_name][1]), float(opt_xy[d_name][1]), None]

        marker_colors_2d = [
            "#949e9f" if n in bad_channels
            else sci_color(sci_scores.get(n), sci_threshold)
            for n in names_2d
        ]

        def _optode_trace(names, color, symbol):
            return go.Scatter(
                x=[float(opt_xy[n][0]) for n in names],
                y=[float(opt_xy[n][1]) for n in names],
                mode="markers+text", text=list(names),
                textposition="top center", textfont=dict(size=8, color="#5d6d7e"),
                marker=dict(size=7, color=color, symbol=symbol,
                            line=dict(width=0.8, color="#ffffff")),
                hovertemplate="<b>%{text}</b><extra></extra>", showlegend=False,
            )

        # trace 1 has to stay the channel scatter: the viewer and the Dash app both index it there
        fig_2d = go.Figure(
            data=[
                go.Scatter(x=lines_x, y=lines_y, mode="lines",
                           line=dict(color="#bdc3c7", width=2),
                           hoverinfo="skip", showlegend=False),
                go.Scatter(x=pos2d[:, 0].tolist(), y=pos2d[:, 1].tolist(), mode="markers",
                           marker=dict(size=10, color=marker_colors_2d, opacity=0.9,
                                       line=dict(width=0.8, color="#555")),
                           customdata=names_2d,
                           hovertemplate="<b>%{customdata}</b><extra></extra>",
                           showlegend=False),
                _optode_trace([n for n in src_xyz if n in opt_xy], "#c0392b", "square"),
                _optode_trace([n for n in det_xyz if n in opt_xy], "#2980b9", "circle"),
            ],
            layout=go.Layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, scaleanchor="x",
                           range=_head_extent(outlines, pos2d[:, 1].tolist())),
                plot_bgcolor="white", paper_bgcolor="white",
                height=700, margin=dict(l=4, r=4, t=4, b=4),
                shapes=_head_outline_shapes(outlines),
                hovermode="closest",
            ),
        )

        trans     = mni_trans(raw.info)
        coords_mm = mne.transforms.apply_trans(trans, ch_locs) * 1000

        seen3: set = set()
        lx, ly, lz = [], [], []
        for ch in chs:
            # loc[:3] is the channel midpoint; the source and detector follow it
            src, det = ch["loc"][3:6], ch["loc"][6:9]
            key = tuple(round(float(v), 5) for v in np.concatenate([src, det]))
            if key in seen3 or not (np.any(src) or np.any(det)):
                continue
            seen3.add(key)
            src_m = mne.transforms.apply_trans(trans, src.reshape(1, 3))[0] * 1000
            det_m = mne.transforms.apply_trans(trans, det.reshape(1, 3))[0] * 1000
            lx += [float(src_m[0]), float(det_m[0]), None]
            ly += [float(src_m[1]), float(det_m[1]), None]
            lz += [float(src_m[2]), float(det_m[2]), None]

        traces_3d = []
        try:
            from nilearn import datasets, surface as surf
            fsavg5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
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

        ch_colors_3d = [
            "#7f8c8d" if n in bad_channels else colors[i]
            for i, n in enumerate(ch_names)
        ]
        traces_3d += [
            go.Scatter3d(
                x=coords_mm[:, 0].tolist(), y=coords_mm[:, 1].tolist(),
                z=coords_mm[:, 2].tolist(), mode="markers",
                marker=dict(size=7, color=ch_colors_3d, opacity=0.92,
                            line=dict(width=0.5, color="#333")),
                customdata=ch_names,
                hovertemplate="<b>%{customdata}</b><br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>",
                name="Channels", showlegend=True,
            ),
            go.Scatter3d(x=[], y=[], z=[], mode="markers",
                         marker=dict(size=14, color="#f39c12", opacity=1.0,
                                     line=dict(width=1.5, color="#e67e22")),
                         hoverinfo="skip", showlegend=False, name="_hl"),
        ]
        fig_3d = go.Figure(
            data=traces_3d,
            layout=go.Layout(
                scene=dict(
                    xaxis=dict(visible=False), yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    camera=dict(eye=dict(x=0, y=-1.9, z=0.6)), bgcolor="#ffffff",
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="#ffffff", height=700,
                legend=dict(x=0.01, y=0.99, font=dict(size=9)),
            ),
        )

    return fig_2d, fig_3d


def _psd_groups(
    raw_od: mne.io.Raw, ch_names: list[str], sep_bands=None,
) -> list[tuple[str, np.ndarray, dict]]:
    """Partition the PSD rows by source-detector separation, long first.

    Short channels sit on a much shorter photon path, so the two get a mean each. The split
    is the one every other raw-level view uses, read from ``long_short_channels`` so this
    figure cannot disagree with the per-channel table or the quality heatmap about which
    channel is which.

    e.g. 30 channels of which 6 are short ->
    [("Long channels", <24 rows>, ...), ("Short channels", <6 rows>, ...)]

    A montage whose separations fall in neither range gets a third curve rather than being
    folded into "long". Returns one unnamed mean when there is no usable split at all.
    """
    from fnirs_pipe.qc.metrics import long_short_channels

    try:
        long_names, short_names = long_short_channels(raw_od, sep_bands)
    except Exception:
        long_names, short_names = [], []
    if not long_names and not short_names:
        return [("Mean (OD)", np.arange(len(ch_names)), dict(color="#2980b9", width=2.5))]

    long_set, short_set = set(long_names), set(short_names)
    groups = [
        ("Long channels",  np.array([c in long_set for c in ch_names]),
         dict(color=LONG_COLOR, width=2.5)),
        ("Short channels", np.array([c in short_set for c in ch_names]),
         dict(color=SHORT_COLOR, width=2.5)),
        (_neither_range_title(),
         np.array([c not in long_set and c not in short_set for c in ch_names]),
         dict(color=UNCLASSIFIED_COLOR, width=2, dash="dash")),
    ]
    return [(name, np.flatnonzero(sel), line) for name, sel, line in groups if sel.any()]


def build_psd_mean_figure(
    raw: mne.io.Raw, cardiac=None, resp=None, sep_bands=None,
) -> go.Figure | None:
    """Channel spectra in optical density, one mean per separation group, channels behind.

    Nothing here is a quality verdict: the panel shows what the spectrum looks like, and
    the screening result is the quality heatmap's and the per-channel table's job.
    """
    try:
        from mne.time_frequency import psd_array_welch
        picks = mne.pick_types(raw.info, meg=False, fnirs=True)
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        data = raw_od.get_data(picks=picks)
        sfreq = raw_od.info["sfreq"]
        psds, freqs = psd_array_welch(data, sfreq,
                                      n_fft=min(PSD_NFFT, data.shape[1]), verbose=False)
        fmax = min(_PSD_FMAX, sfreq / 2)
        mask = freqs <= fmax
        freqs_masked = freqs[mask]
        band = psds[:, mask]
        groups = _psd_groups(raw_od, [raw_od.ch_names[i] for i in picks], sep_bands)

        # ---- Individual channels behind, one bold mean per group ----
        # neutral grey, so a group's own colour is what stands out against it
        traces = [
            go.Scatter(**line_xy(freqs_masked, ch_psd), mode="lines",
                       line=dict(width=0.6, color="rgba(140,140,140,0.15)"),
                       showlegend=False, hoverinfo="skip")
            for ch_psd in band
        ]
        for name, rows, line in groups:
            traces.append(go.Scatter(
                **line_xy(freqs_masked, band[rows].mean(axis=0)), mode="lines",
                name=f"{name} (n={len(rows)})", line=dict(**line),
                hovertemplate=f"{name}<br>%{{x:.3f}} Hz<br>%{{y:.3g}}<extra></extra>",
            ))
        return go.Figure(data=traces,
                         layout=go.Layout(**psd_layout(cardiac=cardiac, resp=resp)))
    except Exception as exc:
        logger.warning("mean PSD failed: %s", exc)
        return None




def _trial_image_data(
    raw_haemo: mne.io.Raw, picks: "list[int]", epoch_tmin: float, epoch_tmax: float,
) -> "list[tuple[str, np.ndarray, np.ndarray, list[str] | None]] | None":
    """Epoch on (non-BAD) events, average over picks -> [(label, (n_trials, n_times) µM, times, rows)].

    One entry per condition, e.g. two conditions with 20 trials each give
    [("all conditions", (40, n_times)), ("rest", (20, ...)), ("task", (20, ...))].
    The pooled entry leads so a condition with few trials can be read against it, and is the
    only one carrying ``rows``, the condition each of its rows came from.
    """
    if not any(not str(a["description"]).upper().startswith("BAD") for a in raw_haemo.annotations):
        return None
    if not picks:
        return None
    try:
        events, event_id = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
        if len(events) == 0:
            return None
        epochs = mne.Epochs(
            raw_haemo, events, event_id, tmin=epoch_tmin, tmax=epoch_tmax,
            picks=picks, baseline=(epoch_tmin, 0), preload=True, verbose=False,
        )
        data = epochs.get_data()  # (n_trials, n_picks, n_times)
        if data.shape[0] == 0:
            return None
        out: "list[tuple[str, np.ndarray, np.ndarray, list[str] | None]]" = []
        pooled = len(event_id) > 1
        if pooled:
            # the pooled panel stacks conditions in event order, so without the name of each
            # row a reader cannot tell which row belongs to which condition
            by_code = {v: k for k, v in event_id.items()}
            rows = [str(by_code.get(int(c), "")) for c in epochs.events[:, 2]]
            # average over channels → (n_trials, n_times)
            out.append(("all conditions", data.mean(axis=1) * 1e6, epochs.times, rows))
        for cond in event_id:
            try:
                cond_data = epochs[cond].get_data()
            except Exception:
                continue
            if cond_data.shape[0] == 0:
                continue
            # a one-row heatmap is a colour strip above a copy of its own average; the
            # pooled panel already carries that trial, named
            if cond_data.shape[0] < 2 and pooled:
                continue
            out.append((str(cond), cond_data.mean(axis=1) * 1e6, epochs.times, None))
        return out or None
    except Exception:
        return None


def _smooth_trials(data: np.ndarray, trial_smooth: int) -> np.ndarray:
    if trial_smooth > 1 and data.shape[0] > 2 * trial_smooth:
        from scipy.ndimage import uniform_filter1d  # moving average across adjacent trials
        return uniform_filter1d(data, size=trial_smooth, axis=0, mode="nearest")
    return data


def _trial_image_plot(
    data: np.ndarray, times: np.ndarray, title: str, zmax: float,
    row_labels: "list[str] | None" = None,
) -> "go.Figure":
    """Trial x time heatmap + trial average. ``data`` is already trial-smoothed."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06,
        subplot_titles=[title, "trial average"],
    )
    fig.add_trace(go.Heatmap(
        z=data, x=times.tolist(), colorscale="RdBu_r", zmid=0, zmin=-zmax, zmax=zmax,
        # no zsmooth: it would blend adjacent rows, which are separate trials or conditions
        colorbar=dict(title="µM", len=0.7, y=0.62),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=times.tolist(), y=data.mean(axis=0).tolist(),
        line=dict(color="#c0392b", width=1.5),
    ), row=2, col=1)
    fig.add_vline(x=0.0, line=dict(color="#333", width=1, dash="dash"))
    fig.update_yaxes(title_text="trial", row=1, col=1)
    if row_labels and len(row_labels) <= _TRIAL_ROW_LABEL_MAX:
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(row_labels))),
                         ticktext=row_labels, tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(title_text="µM", row=2, col=1)
    fig.update_xaxes(title_text="Time from onset (s)", row=2, col=1)
    fig.update_layout(
        height=480, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _auto_trial_smooth(n_trials: int) -> int:
    return max(1, n_trials // 15)


# above this every row label would be drawn and they collide; the axis stays numeric instead
_TRIAL_ROW_LABEL_MAX = 12


def _n_trials(n: int) -> str:
    return "1 trial" if n == 1 else f"{n} trials"


def _trial_image_figures(
    res: "list[tuple[str, np.ndarray, np.ndarray, list[str] | None]]",
    title_for: "callable",
    trial_smooth: "int | None",
) -> "list[go.Figure]":
    """One figure per condition, on a colour scale shared across them so panels compare.

    Smoothing runs first: the scale is taken from what is actually drawn, and an unsmoothed
    single-trial spike would otherwise wash out every panel.
    """
    panels = []
    for label, data, times, rows in res:
        sm = trial_smooth if trial_smooth is not None else _auto_trial_smooth(data.shape[0])
        panels.append((label, _smooth_trials(data, sm), times, rows))
    zmax = max(
        (float(np.nanpercentile(np.abs(d), 97)) for _, d, _, _ in panels),
        default=0.0,
    ) or 1.0
    return [
        _trial_image_plot(data, times, title_for(label, data.shape[0]), zmax, rows)
        for label, data, times, rows in panels
    ]


def build_trial_image_figure(
    raw_haemo: mne.io.Raw,
    ch_name: str,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    trial_smooth: "int | None" = None,
) -> "list[go.Figure] | None":
    """Trial image: one HbO channel's epochs stacked as a trial x time heatmap + trial average.

    Rows = stimulus repetitions, x = time from onset, colour = baseline-corrected HbO,
    optionally smoothed across adjacent trials to reveal the slow response.
    The un-averaged companion to the block average. One figure per condition, pooled panel
    first. None if no (non-BAD) events / channel absent.
    """
    if ch_name not in raw_haemo.ch_names:
        return None
    res = _trial_image_data(raw_haemo, [raw_haemo.ch_names.index(ch_name)], epoch_tmin, epoch_tmax)
    if res is None:
        return None
    return _trial_image_figures(
        res, lambda label, n: f"Trial image: {ch_name} / {label} ({_n_trials(n)})", trial_smooth)


def _roi_picks(raw_haemo: mne.io.Raw, channels: "list[str]") -> "list[int]":
    """ROI members as HbO channel indices. Names or S-D pair labels both resolve."""
    hbo = {c for c in raw_haemo.ch_names if c.endswith(" hbo")}
    return [raw_haemo.ch_names.index(c if c in hbo else f"{c} hbo")
            for c in channels if (c in hbo or f"{c} hbo" in hbo)]


# ---- Trial images, one condition window at a time ----

def _trial_image_by_span(
    raw_haemo: mne.io.Raw,
    picks: "list[int]",
    spans: "list[tuple[str, float, float]]",
    epoch_tmin: float,
    epoch_tmax: float,
) -> "dict[str, tuple[np.ndarray, np.ndarray, list[str]]] | None":
    """Epoch once, then give each condition window the trials whose onset falls inside it.

    ``spans`` is ``[(label, t0, t1)]`` on the data axis, the windows the quality record was
    written against. A trial belongs to the window its *onset* sits in, which is how the
    record assigns everything else. Its epoch window is free to run past the edge of the
    condition and usually does, since the response outlasts the event.

    Example: blocks ``("task", 0, 120)`` and ``("rest", 120, 240)`` with six 8 s trials
    inside each give ``{"task": (6, n_times), "rest": (6, n_times)}``.

    Returns None for a recording with nothing to epoch, and omits a window that caught no
    trial rather than giving it an empty entry.

    The annotation that *defines* a window is not one of its trials, which is why the test
    on ``t0`` is strict. On a blocked event-related design the block is itself an event,
    sitting at ``t0``, so counting it would put a row describing the whole block beside the
    trials inside it. A genuine trial starting in the same sample as its block is lost with
    it: the two are indistinguishable from the annotations.
    """
    if not picks or not spans:
        return None
    try:
        events, event_id = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
        if len(events) == 0:
            return None
        epochs = mne.Epochs(
            raw_haemo, events, event_id, tmin=epoch_tmin, tmax=epoch_tmax,
            picks=picks, baseline=(epoch_tmin, 0), preload=True, verbose=False,
        )
        data = epochs.get_data()  # (n_trials, n_picks, n_times)
        if data.shape[0] == 0:
            return None
        by_code = {v: k for k, v in event_id.items()}
        sfreq = float(raw_haemo.info["sfreq"])
        # the data axis, the one condition_windows measures on; an event's sample index is
        # against the original recording and a cropped input puts the two a first_time apart
        onsets = (epochs.events[:, 0] - raw_haemo.first_samp) / sfreq
        descs = [str(by_code.get(int(c), "")) for c in epochs.events[:, 2]]
        # an event lands on a sample and a window bound does not, so the two agree only to
        # within half a sample
        eps = 0.5 / sfreq
        out: "dict[str, tuple[np.ndarray, np.ndarray, list[str]]]" = {}
        for label, t0, t1 in spans:
            keep = [i for i, onset in enumerate(onsets) if t0 + eps < onset < t1 - eps]
            if not keep:
                continue
            out[str(label)] = (data[keep].mean(axis=1) * 1e6, epochs.times,
                               [descs[i] for i in keep])
        return out or None
    except Exception:
        return None


def _condition_trial_images(
    raw_haemo: mne.io.Raw,
    picks: "list[int]",
    spans: "list[tuple[str, float, float]]",
    epoch_tmin: float,
    epoch_tmax: float,
    title_for: "callable",
    trial_smooth: "int | None",
    min_trials: int,
) -> "dict[str, list[go.Figure]] | None":
    """One figure per condition window, on a colour scale shared across the windows.

    Shared because the panels land on separate pages, one per condition, and are read against
    each other. This is the run's own trial image split by window instead of by event name.
    """
    res = _trial_image_by_span(raw_haemo, picks, spans, epoch_tmin, epoch_tmax)
    if res is None:
        return None
    kept = [(label, data, times, rows) for label, (data, times, rows) in res.items()
            if data.shape[0] >= min_trials]
    if not kept:
        return None
    figs = _trial_image_figures(kept, title_for, trial_smooth)
    return {label: [fig] for (label, *_), fig in zip(kept, figs)}


def build_trial_image_by_condition(
    raw_haemo: mne.io.Raw,
    ch_name: str,
    spans: "list[tuple[str, float, float]]",
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    trial_smooth: "int | None" = None,
    min_trials: int = 2,
) -> "dict[str, list[go.Figure]] | None":
    """One HbO channel's trial image, split by condition window rather than by event name.

    ``min_trials`` is why a block design gets nothing: its condition window holds the one
    annotation that defines it, and a one-row heatmap is a colour strip above a copy of its
    own average.
    """
    if ch_name not in raw_haemo.ch_names:
        return None
    return _condition_trial_images(
        raw_haemo, [raw_haemo.ch_names.index(ch_name)], spans, epoch_tmin, epoch_tmax,
        lambda label, n: f"Trial image: {ch_name} / {label} ({_n_trials(n)})",
        trial_smooth, min_trials)


def build_roi_trial_image_by_condition(
    raw_haemo: mne.io.Raw,
    roi_name: str,
    channels: "list[str]",
    spans: "list[tuple[str, float, float]]",
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    trial_smooth: "int | None" = None,
    min_trials: int = 2,
) -> "dict[str, list[go.Figure]] | None":
    """The ROI trial image split by condition window. Channels averaged first, as the run's is."""
    picks = _roi_picks(raw_haemo, channels)
    if not picks:
        return None
    return _condition_trial_images(
        raw_haemo, picks, spans, epoch_tmin, epoch_tmax,
        lambda label, n: (f"Trial image: ROI {roi_name} / {label} "
                          f"({len(picks)} ch, {_n_trials(n)})"),
        trial_smooth, min_trials)


def build_roi_trial_image_figure(
    raw_haemo: mne.io.Raw,
    roi_name: str,
    channels: "list[str]",
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    trial_smooth: "int | None" = None,
) -> "list[go.Figure] | None":
    """ROI trial image: average the ROI's HbO channels first (higher SNR), then stack trials.

    ``channels`` are channel names or S-D pair labels; matched to their HbO channels.
    One figure per condition, pooled panel first.
    """
    picks = _roi_picks(raw_haemo, channels)
    if not picks:
        return None
    res = _trial_image_data(raw_haemo, picks, epoch_tmin, epoch_tmax)
    if res is None:
        return None
    return _trial_image_figures(
        res,
        lambda label, n: f"Trial image: ROI {roi_name} / {label} ({len(picks)} ch, {_n_trials(n)})",
        trial_smooth,
    )


def build_epoch_preview_figure(
    raw_haemo: mne.io.Raw,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    sep_bands=None,
) -> go.Figure | None:
    """Grand mean per condition: every good long channel averaged, the good short ones dotted.

    The short channels are drawn rather than dropped, as the scalp reference for the
    long-channel mean.

    ``sep_bands`` is this run's separations from :func:`separation_bands`; None takes the
    package defaults.
    """
    markers = extract_markers(raw_haemo)
    if not markers:
        return None

    try:
        events_mne, event_id = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
        if len(events_mne) == 0:
            return None

        epochs = mne.Epochs(
            raw_haemo, events_mne, event_id,
            tmin=epoch_tmin, tmax=epoch_tmax,
            baseline=(epoch_tmin, 0),
            preload=True, verbose=False,
        )
        from fnirs_pipe.qc.metrics import long_short_channels
        long_names, short_names = long_short_channels(raw_haemo, sep_bands)

        bads = set(raw_haemo.info["bads"])

        def _picks(names, chromo):
            keep = {n for n in names if n.endswith(chromo) and n not in bads}
            return [i for i, n in enumerate(epochs.ch_names) if n in keep]

        # every channel in one figure when the montage has no short channels to split off
        traces = []
        for chromo, color in (("hbo", HBO_COLOR), ("hbr", HBR_COLOR)):
            lp, sp = _picks(long_names, chromo), _picks(short_names, chromo)
            if not lp and not sp:
                lp = list(mne.pick_types(epochs.info, fnirs=chromo))
                traces.append((lp, color, chromo.upper(), "solid"))
                continue
            if lp:
                label = f"{chromo.upper()} long" if sp else chromo.upper()
                traces.append((lp, color, label, "solid"))
            if sp:
                traces.append((sp, color, f"{chromo.upper()} short (scalp)", "dot"))

        # one panel per condition, so HbO and HbR can keep the red/blue they carry in every
        # other panel
        panels = []
        for cond in event_id:
            try:
                ep = epochs[cond].get_data()
            except Exception:
                continue
            if not ep.size:
                continue
            panels.append((cond, ep))
        if not panels:
            return None

        # stacked, one row per condition, rather than side by side
        n = len(panels)
        scaling = _scale_setting_conditions(panels)
        titles = [f"{c} (n={e.shape[0]})" +
                  ("" if (c, e) in scaling or len(scaling) == n else ", off the shared scale")
                  for c, e in panels]
        fig = make_subplots(
            rows=n, cols=1, shared_xaxes=True, vertical_spacing=min(0.06, 0.9 / n),
            subplot_titles=titles,
        )
        # the task's own length, per condition, so the curve reads against when it stopped
        by_cond: "dict[str, list[float]]" = {}
        for m in markers:
            if float(m["duration"]) > 0:
                by_cond.setdefault(m["description"], []).append(float(m["duration"]))

        yrange = _shared_yrange(scaling, traces)

        for i, (_cond, ep) in enumerate(panels, start=1):
            for picks, color, label, dash in traces:
                if not len(picks):
                    continue
                fig.add_trace(go.Scatter(
                    x=epochs.times.tolist(),
                    y=(ep[:, picks, :].mean(axis=(0, 1)) * 1e6).tolist(),
                    name=label, mode="lines", legendgroup=label,
                    showlegend=(i == 1),
                    line=dict(color=color, width=2, dash=dash),
                ), row=i, col=1)
            # clamped to the window: an unclamped shape drives the autorange
            block = float(np.median(by_cond.get(_cond, []))) if by_cond.get(_cond) else 0.0
            if block > 0:
                fig.add_vrect(x0=0, x1=min(block, epoch_tmax), line_width=0,
                              fillcolor="#f1c40f", opacity=0.16, layer="below",
                              row=i, col=1)
            fig.add_hline(y=0, line=dict(color="#cccccc", width=1), row=i, col=1)
            fig.add_vline(x=0, line=dict(color="#7f8c8d", width=1, dash="dash"),
                          row=i, col=1)
            # one y scale for every row, or each condition autoscales to itself
            fig.update_yaxes(title_text="Conc. (µmol/L)", gridcolor="#eeeeee",
                             row=i, col=1,
                             **({"range": yrange} if i == 1 else {"matches": "y"}))
        fig.update_xaxes(range=[epoch_tmin, epoch_tmax])
        fig.update_xaxes(title_text="Time rel. onset (s)", gridcolor="#eeeeee",
                         zerolinecolor="#cccccc", row=n, col=1)
        fig.update_annotations(font_size=10)
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=60 + 260 * n, margin=dict(l=60, r=15, t=46, b=44),
            # top right rather than centred: a centred legend sits on the first row's title
            legend=dict(font=dict(size=9), orientation="h",
                        x=1, xanchor="right", y=1.0, yanchor="bottom"),
        )
        return fig
    except Exception as exc:
        logger.warning("epoch preview failed: %s", exc)
        return None


_SCALE_MIN_TRIALS = 3


def _scale_setting_conditions(panels):
    """Conditions allowed to set the shared y scale, which is not every condition.

    A stray trigger leaves a condition of one or two trials whose average is single-trial
    noise. Below ``_SCALE_MIN_TRIALS`` a condition is still drawn, just not consulted for the
    range.

    e.g. panels of 30, 30 and 2 trials -> the two 30-trial ones. When nothing clears the bar
    (a block design with one block per condition) every panel is kept.
    """
    kept = [p for p in panels if p[1].shape[0] >= _SCALE_MIN_TRIALS]
    return kept or list(panels)


def _shared_yrange(scaling, traces, pad: float = 0.08):
    """Symmetric-padded range over the traces the scale-setting conditions actually draw."""
    lo, hi = np.inf, -np.inf
    for _cond, ep in scaling:
        for picks, _c, _l, _d in traces:
            if not len(picks):
                continue
            y = ep[:, picks, :].mean(axis=(0, 1)) * 1e6
            lo, hi = min(lo, float(np.nanmin(y))), max(hi, float(np.nanmax(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    margin = (hi - lo) * pad
    return [lo - margin, hi + margin]


def build_trigger_timeline_single(
    markers: list[dict],
    cond_colors: dict[str, str],
) -> go.Figure | None:
    if not markers:
        return None

    all_descs: list[str] = []
    by_desc: dict[str, list[dict]] = {}
    for m in markers:
        desc = m["description"]
        if desc not in all_descs:
            all_descs.append(desc)
        by_desc.setdefault(desc, []).append(m)

    span = max(m["onset"] + m["duration"] for m in markers) - min(m["onset"] for m in markers)

    traces, anns = [], []
    # Summary row at y=0: all conditions overlaid with their own colours, unlabelled -- a
    # length printed there would sit on whichever condition happened to be drawn last
    for desc in all_descs:
        traces += timeline_row_traces(by_desc[desc], 0, cond_colors.get(desc, "#999"),
                                      desc, False)

    # Per-condition rows starting at y=1, each block carrying its length
    for desc_idx, desc in enumerate(all_descs, start=1):
        traces += timeline_row_traces(by_desc[desc], desc_idx, cond_colors.get(desc, "#999"),
                                      desc, False)
        anns += block_duration_labels(by_desc[desc], desc_idx, span)

    rows = ["(all)"] + all_descs
    xaxis, yaxis = timeline_axes(rows)
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=xaxis, yaxis=yaxis,
            shapes=timeline_row_bands(len(rows)), annotations=anns,
            plot_bgcolor="white", paper_bgcolor="white",
            # overlay, or plotly groups the per-condition bars and shifts each row off its
            # own tick
            barmode="overlay",
            height=max(120, len(rows) * TIMELINE_ROW_PX + 60),
            margin=dict(l=110, r=20, t=14, b=44),
            hovermode="closest",
            # the rows are named on the axis, so a legend would repeat them
            showlegend=False,
        ),
    )


def _topo_layers(
    raw_haemo: mne.io.Raw,
    markers: list[dict],
    picks: "list[int]",
    max_ts_pts: int,
    epoch_tmin: float,
    epoch_tmax: float,
) -> "list[dict]":
    """One drawing layer per condition, each holding that condition's evoked trace per channel.

    e.g. two conditions -> [{"label": "rest", "times": [...], "by_ch": {"S1_D1 hbo": [...], ...}},
    {"label": "task", ...}]. Falls back to a single layer of the continuous (decimated) signal
    when the run has no usable events, which is what a resting-state run gets.
    """
    usable = [m for m in markers if not str(m["description"]).upper().startswith("BAD")]
    if usable:
        try:
            anns = mne.Annotations(
                onset=[m["onset"] for m in usable],
                duration=[m["duration"] for m in usable],
                description=[m["description"] for m in usable],
            )
            raw_copy = raw_haemo.copy().set_annotations(anns)
            events, event_id = epochable_events(raw_copy, epoch_tmin, epoch_tmax)
            if len(events) > 0:
                epochs = mne.Epochs(
                    raw_copy, events, event_id, tmin=epoch_tmin, tmax=epoch_tmax,
                    picks=picks, baseline=(epoch_tmin, 0), preload=True, verbose=False,
                )
                cond_colors_ = condition_colors(usable)
                single = len(event_id) == 1
                layers = []
                for ci, cond in enumerate(event_id):
                    try:
                        ev = epochs[cond].average()
                    except Exception:
                        continue
                    base = cond_colors_.get(cond, CONDITION_PALETTE[ci % len(CONDITION_PALETTE)])
                    layers.append({
                        "label": str(cond),
                        "n": len(epochs[cond]),
                        "times": ev.times.tolist(),
                        "by_ch": {name: (row * 1e6).tolist()
                                  for name, row in zip(ev.ch_names, ev.data)},
                        # a single condition keeps the familiar HbO/HbR colours; several are
                        # told apart by condition instead, or the cells become unreadable
                        "hbo_color": HBO_COLOR if single else base,
                        "hbr_color": HBR_COLOR if single else base,
                    })
                if layers:
                    return layers
        except Exception as exc:
            logger.warning("evoked topo epoching failed, falling back to continuous: %s", exc)

    data, times = raw_haemo.get_data(picks=picks, return_times=True)
    data, times = _decimate(data, times, max_ts_pts)
    names = [raw_haemo.ch_names[i] for i in picks]
    return [{
        "label": "",
        "n": 0,
        "times": times.tolist(),
        "by_ch": {name: (row * 1e6).tolist() for name, row in zip(names, data)},
        "hbo_color": HBO_COLOR,
        "hbr_color": HBR_COLOR,
    }]


# ---- Topo cell placement ----

_TOPO_INSET = 0.03          # keep cells off the head outline
_TOPO_LABEL_GAP = 0.014     # paper units reserved above a cell for its S-D label
_TOPO_DISC_R = 0.46         # cell centres stay within this radius of the head outline


def _spread_topo_cells(norm01: "np.ndarray", hw: float, hh: float) -> "np.ndarray":
    """Snap cell centres onto a grid of non-overlapping slots, nearest free slot wins.

    Each cell is its own plotly subplot, and overlapping subplot domains swallow each
    other's clicks: whichever is drawn last takes the whole overlap, so a channel hidden
    underneath can never be selected. Short channels make this the common case, since they
    sit almost exactly on top of the long channel sharing their source.

    e.g. two channels normalised to (0.51, 0.30) and (0.52, 0.30) land in the same slot;
    the second is pushed to the neighbouring slot instead of on top of the first.
    """
    slot_w = 2 * hw + 0.006
    slot_h = 2 * hh + _TOPO_LABEL_GAP
    lo, hi = _TOPO_INSET, 1.0 - _TOPO_INSET
    ncols = max(1, int((hi - lo) // slot_w))
    nrows = max(1, int((hi - lo) // slot_h))
    cell_w = (hi - lo) / ncols
    cell_h = (hi - lo) / nrows

    def centre(r, c):
        return lo + (c + 0.5) * cell_w, hi - (r + 0.5) * cell_h

    # corner slots of the square fall outside the head outline; drop them unless the
    # montage needs more slots than the disc holds
    inside = {
        (r, c) for r in range(nrows) for c in range(ncols)
        if sum((v - 0.5) ** 2 for v in centre(r, c)) <= _TOPO_DISC_R ** 2
    }
    allowed = inside if len(inside) >= len(norm01) else None

    # rings of slots around the target, nearest in paper distance first
    reach = max(nrows, ncols)
    offsets = sorted(
        ((dr, dc) for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)),
        key=lambda d: (d[0] * cell_h) ** 2 + (d[1] * cell_w) ** 2,
    )

    taken: set = set()
    out = np.empty_like(norm01, dtype=float)
    # place top-left first so the fallback drift is downward/rightward and stays readable
    for i in np.lexsort((norm01[:, 0], -norm01[:, 1])):
        c0 = int(np.clip(round(norm01[i, 0] * ncols - 0.5), 0, ncols - 1))
        r0 = int(np.clip(round((1.0 - norm01[i, 1]) * nrows - 0.5), 0, nrows - 1))
        for dr, dc in offsets:
            r, c = r0 + dr, c0 + dc
            if not (0 <= r < nrows and 0 <= c < ncols) or (r, c) in taken:
                continue
            if allowed is not None and (r, c) not in allowed:
                continue
            taken.add((r, c))
            out[i] = centre(r, c)
            break
        else:
            out[i] = centre(r0, c0)
    return out


def build_evoked_topo_figure(
    raw_haemo: mne.io.Raw,
    markers: list[dict],
    max_ts_pts: int = 2000,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
) -> go.Figure | None:
    """Per-channel evoked HbO/HbR laid out at the channel's position on the head.

    One cell per S-D pair, one trace pair per condition. Without usable events the cells
    show the continuous signal instead, so a resting-state run still gets the layout view.
    """
    # channel order, matching get_channel_pairs; a name sort disagrees on multi-digit indices
    hbo_entries = [(i, ch) for i, ch in enumerate(raw_haemo.ch_names) if ch.endswith(" hbo")]
    if not hbo_entries:
        return None

    pairs = [pair_of(ch) for _, ch in hbo_entries]
    n = len(pairs)

    locs = np.array([raw_haemo.info["chs"][i]["loc"][:2] for i, _ in hbo_entries])
    if np.any(locs != 0):
        lo, hi = locs.min(axis=0), locs.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        norm01 = (locs - lo) / span
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        norm01 = np.array([
            [(i % ncols + 0.5) / ncols, (i // ncols + 0.5) / nrows]
            for i in range(n)
        ])

    picks = [i for i, _ in hbo_entries] + [
        raw_haemo.ch_names.index(f"{p} hbr")
        for p in pairs
        if f"{p} hbr" in raw_haemo.ch_names
    ]
    layers = _topo_layers(raw_haemo, markers, picks, max_ts_pts, epoch_tmin, epoch_tmax)
    evoked_mode = bool(layers[0]["label"])
    x_title = "Time from onset (s)" if evoked_mode else "Time (s)"

    _H, _ML, _MR, _MT, _MB = 700, 10, 10, 12, 8

    hw, hh = 0.048, 0.023
    norm = _spread_topo_cells(norm01, hw, hh)
    box_shapes = []
    onset_shapes = []
    fig = go.Figure()

    for pi, pair in enumerate(pairs):
        cx, cy = float(norm[pi, 0]), float(norm[pi, 1])
        x0, x1 = max(0.0, cx - hw), min(1.0, cx + hw)
        y0, y1 = max(0.0, cy - hh), min(1.0, cy + hh)
        ai = pi + 1
        xk = "xaxis" if ai == 1 else f"xaxis{ai}"
        yk = "yaxis" if ai == 1 else f"yaxis{ai}"
        xr = "x"    if ai == 1 else f"x{ai}"
        yr = "y"    if ai == 1 else f"y{ai}"
        fig.update_layout(**{
            xk: dict(domain=[x0, x1], showticklabels=False, showgrid=False,
                     zeroline=False, anchor=yr),
            yk: dict(domain=[y0, y1], showticklabels=False, showgrid=False,
                     zeroline=False, anchor=xr),
        })
        # name lets a caller find a cell's box without relying on shape order
        box_shapes.append(dict(
            type="rect", name=pair, xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#ccc", width=0.8),
            fillcolor="rgba(255,255,255,0.80)",
            layer="below",
        ))

        if evoked_mode:
            onset_shapes.append(dict(
                type="line", xref=xr, yref=f"{yr} domain",
                x0=0, x1=0, y0=0, y1=1,
                line=dict(color="#999", width=0.8, dash="dash"),
            ))

        first = (pi == 0)
        for layer in layers:
            label, times_list = layer["label"], layer["times"]
            suffix = f" {label}" if label else ""
            n_txt = f" (n={layer['n']})" if layer["n"] else ""
            for chromo, color, dash in (
                ("hbo", layer["hbo_color"], "solid"),
                ("hbr", layer["hbr_color"], "dot"),
            ):
                y = layer["by_ch"].get(f"{pair} {chromo}")
                if y is None:
                    continue
                fig.add_trace(go.Scatter(
                    x=times_list, y=y,
                    name=f"{chromo.upper()}{suffix}{n_txt}" if first else None,
                    mode="lines",
                    line=dict(color=color, width=1.0, dash=dash),
                    xaxis=xr, yaxis=yr,
                    customdata=[pair] * len(times_list),
                    legendgroup=f"{chromo}{suffix}", showlegend=first,
                    hovertemplate=(
                        f"<b>{pair}</b> %{{x:.1f}}s %{{y:.2f}} µM"
                        f"<extra>{chromo.upper()}{suffix}</extra>"
                    ),
                ))

        fig.add_annotation(
            x=(x0 + x1) / 2, y=y1 + 0.003,
            xref="paper", yref="paper",
            text=pair, showarrow=False,
            font=dict(size=7, color="#444"),
            xanchor="center", yanchor="bottom",
        )

    head_shapes = [
        dict(type="circle",
             xref="paper", yref="paper",
             x0=0.01, y0=0.01, x1=0.99, y1=0.99,
             line=dict(color="#bbb", width=2),
             fillcolor="rgba(245,245,245,0.45)",
             layer="below"),
        dict(type="path",
             path="M 0.455,0.982 L 0.500,1.018 L 0.545,0.982",
             xref="paper", yref="paper",
             line=dict(color="#bbb", width=2),
             layer="below"),
        dict(type="path",
             path="M 0.010,0.560 Q -0.028,0.500 0.010,0.440",
             xref="paper", yref="paper",
             line=dict(color="#bbb", width=2),
             layer="below"),
        dict(type="path",
             path="M 0.990,0.560 Q 1.028,0.500 0.990,0.440",
             xref="paper", yref="paper",
             line=dict(color="#bbb", width=2),
             layer="below"),
    ]

    _W = _H - _MT - _MB + _ML + _MR  # keep plot area square (680×680)
    fig.update_layout(
        width=_W,
        height=_H,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        hoverdistance=-1,   # the default 20px cutoff leaves dead spots inside a cell
        margin=dict(l=_ML, r=_MR, t=_MT, b=_MB),
        shapes=box_shapes + onset_shapes + head_shapes,
        legend=dict(x=0.99, y=0.99, xanchor="right",
                    font=dict(size=9), bgcolor="rgba(255,255,255,0.75)",
                    title=dict(text="HbO / HbR", font=dict(size=9))),
    )
    span = layers[0]["times"]
    if span:
        fig.add_annotation(
            x=0.01, y=0.0, xref="paper", yref="paper",
            text=f"{x_title}: {span[0]:.0f} to {span[-1]:.0f}",
            showarrow=False, font=dict(size=9, color="#666"),
            xanchor="left", yanchor="bottom",
        )
    return fig
