"""Plotly figure builders for the interactive raw fNIRS QC viewer."""

from __future__ import annotations

import os

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures")

_PSD_BANDS = [
    {"name": "Mayer",   "x0": 0.07, "x1": 0.13, "color": "rgba(52,152,219,0.10)"},
    {"name": "Resp",    "x0": 0.15, "x1": 0.40, "color": "rgba(39,174,96,0.08)"},
    {"name": "Cardiac", "x0": 0.70, "x1": 1.50, "color": "rgba(231,76,60,0.08)"},
]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _ch_colors(raw: mne.io.Raw, short_thresh: float) -> list[str]:
    picks = list(range(len(raw.ch_names)))
    try:
        dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
        return ["#78b1f2" if d <= short_thresh else "rgba(243,125,125,0.78)" for d in dists]
    except Exception:
        return ["rgba(243,125,125,0.78)"] * len(raw.ch_names)


def _decimate(arr: np.ndarray, times: np.ndarray, max_pts: int):
    if len(times) <= max_pts:
        return arr, times
    step = max(1, len(times) // max_pts)
    return arr[:, ::step], times[::step]


def psd_layout(height: int = 220) -> dict:
    shapes = [dict(type="rect", xref="x", yref="paper",
                   x0=b["x0"], x1=b["x1"], y0=0, y1=1,
                   fillcolor=b["color"], line=dict(width=0)) for b in _PSD_BANDS]
    annotations = [dict(x=(b["x0"] + b["x1"]) / 2, y=0.97, xref="x", yref="paper",
                        text=b["name"], showarrow=False,
                        font=dict(size=8, color="#666")) for b in _PSD_BANDS]
    return dict(
        xaxis=dict(title="Frequency (Hz)", range=[0, 2], gridcolor="#eeeeee"),
        yaxis=dict(title="Power", type="log", gridcolor="#eeeeee"),
        plot_bgcolor="white", paper_bgcolor="white",
        height=height, margin=dict(l=60, r=15, t=8, b=38),
        legend=dict(font=dict(size=9)),
        shapes=shapes, annotations=annotations,
    )


def sci_color(sci: float | None) -> str:
    if sci is None:
        return "#aaa"
    if sci >= 0.75:
        return "#27ae60"
    if sci >= 0.5:
        return "#f39c12"
    return "#e74c3c"


def condition_colors(markers: list[dict]) -> dict[str, str]:
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    ]
    descs = list(dict.fromkeys(m["description"] for m in markers))
    return {d: palette[i % len(palette)] for i, d in enumerate(descs)}


def build_ts_figure(
    raw: mne.io.Raw,
    markers: list[dict],
    bad_channels: set[str],
    max_ts_pts: int,
    short_thresh: float,
) -> tuple[go.Figure, list[dict], dict[str, str], list[dict], float, float]:
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    if len(picks) == 0:
        picks = list(range(len(raw.ch_names)))

    data, times = raw.get_data(picks=picks, return_times=True)
    data, times = _decimate(data, times, max_ts_pts)
    times_list = times.tolist()

    all_colors = _ch_colors(raw, short_thresh)
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
            yaxis=dict(showticklabels=False, gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            shapes=band_shapes + mk_shapes,
            height=max(380, n * 22 + 80),
            margin=dict(l=55, r=15, t=12, b=40),
            legend=dict(font=dict(size=8), tracegroupgap=0),
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
    epoch_tmin: float,
    epoch_tmax: float,
) -> tuple[go.Figure, go.Figure | None, go.Figure | None]:
    hbo_name = f"{ch_pair} hbo"
    hbr_name = f"{ch_pair} hbr"
    haemo_names = raw_haemo.ch_names

    if hbo_name not in haemo_names or hbr_name not in haemo_names:
        raise ValueError(f"channel pair {ch_pair!r} not found")

    hbo_pick = haemo_names.index(hbo_name)
    hbr_pick = haemo_names.index(hbr_name)
    haemo_data, times = raw_haemo.get_data(picks=[hbo_pick, hbr_pick], return_times=True)
    haemo_data, times_d = _decimate(haemo_data, times, max_ts_pts)
    times_list = times_d.tolist()
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
                       line=dict(color="#e74c3c", width=1.5),
                       fill="tozeroy", fillcolor="rgba(231,76,60,0.06)"),
            go.Scatter(x=times_list, y=hbr, name="HbR", mode="lines",
                       line=dict(color="#3498db", width=1.5),
                       fill="tozeroy", fillcolor="rgba(52,152,219,0.06)"),
        ],
        layout=go.Layout(
            xaxis=dict(title="Time (s)", gridcolor="#eeeeee", zerolinecolor="#cccccc"),
            yaxis=dict(title="Conc. (µmol/L)", gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=160, margin=dict(l=55, r=15, t=8, b=38),
            legend=dict(font=dict(size=9), orientation="h", y=1.12),
            shapes=_mk_shapes(),
        ),
    )

    psd_fig = None
    try:
        from scipy.signal import welch
        raw_arr = raw_haemo.get_data(picks=[hbo_pick, hbr_pick])
        sfreq = raw_haemo.info["sfreq"]
        nperseg = min(512, max(64, raw_arr.shape[1] // 4))
        freqs, psd_hbo = welch(raw_arr[0], fs=sfreq, nperseg=nperseg)
        _, psd_hbr     = welch(raw_arr[1], fs=sfreq, nperseg=nperseg)
        fmax = min(2.0, sfreq / 2)
        mask = freqs <= fmax
        psd_fig = go.Figure(
            data=[
                go.Scatter(x=freqs[mask].tolist(), y=psd_hbo[mask].tolist(),
                           name="HbO", mode="lines", line=dict(color="#e74c3c", width=2)),
                go.Scatter(x=freqs[mask].tolist(), y=psd_hbr[mask].tolist(),
                           name="HbR", mode="lines", line=dict(color="#3498db", width=2)),
            ],
            layout=go.Layout(**psd_layout()),
        )
    except Exception as exc:
        logger.warning("channel PSD failed for %s: %s", ch_pair, exc)

    epoch_fig = None
    if markers:
        try:
            colors10 = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                        "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
            anns = mne.Annotations(
                onset=[m["onset"] for m in markers],
                duration=[m["duration"] for m in markers],
                description=[m["description"] for m in markers],
            )
            raw_copy = raw_haemo.copy().set_annotations(anns)
            events_mne, event_id = mne.events_from_annotations(raw_copy, verbose=False)
            if len(events_mne) > 0:
                epochs = mne.Epochs(
                    raw_copy, events_mne, event_id,
                    tmin=epoch_tmin, tmax=epoch_tmax,
                    picks=[hbo_pick, hbr_pick],
                    baseline=(epoch_tmin, 0),
                    preload=True, verbose=False,
                )
                epoch_traces = []
                for ci, (cond, _) in enumerate(event_id.items()):
                    try:
                        ep_subset = epochs[cond]
                        if len(ep_subset) == 0:
                            continue
                        ep = ep_subset.get_data()
                        base_color = cond_colors_.get(cond, colors10[ci % len(colors10)])
                        epoch_traces += [
                            go.Scatter(x=epochs.times.tolist(),
                                       y=(ep[:, 0, :].mean(axis=0) * 1e6).tolist(),
                                       name=f"{cond} HbO (n={ep.shape[0]})", mode="lines",
                                       line=dict(color=base_color, width=2)),
                            go.Scatter(x=epochs.times.tolist(),
                                       y=(ep[:, 1, :].mean(axis=0) * 1e6).tolist(),
                                       name=f"{cond} HbR", mode="lines",
                                       line=dict(color=base_color, width=1.2, dash="dot")),
                        ]
                    except Exception:
                        pass
                epoch_fig = go.Figure(
                    data=epoch_traces,
                    layout=go.Layout(
                        xaxis=dict(title="Time rel. onset (s)", gridcolor="#eeeeee",
                                   zerolinecolor="#cccccc"),
                        yaxis=dict(title="Conc. (µmol/L)", gridcolor="#eeeeee"),
                        shapes=[dict(type="line", xref="x", yref="paper",
                                     x0=0, x1=0, y0=0, y1=1,
                                     line=dict(color="#7f8c8d", width=1, dash="dash"))],
                        annotations=[dict(x=0, y=1.0, xref="x", yref="paper",
                                          text="onset", showarrow=False,
                                          font=dict(size=8, color="#7f8c8d"),
                                          xanchor="left")],
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=220, margin=dict(l=60, r=15, t=8, b=38),
                        legend=dict(font=dict(size=8), tracegroupgap=0),
                    ),
                )
        except Exception as exc:
            logger.warning("epoch preview failed for %s: %s", ch_pair, exc)

    return detail_fig, psd_fig, epoch_fig


def build_layout_figure(
    raw: mne.io.Raw,
    bad_channels: set[str],
    sci_scores: dict[str, float],
    short_thresh: float,
) -> tuple[go.Figure | None, go.Figure | None]:
    chs = raw.info["chs"]
    ch_names = raw.ch_names
    colors = _ch_colors(raw, short_thresh)

    ch_locs = np.array([ch["loc"][:3] for ch in chs])
    has_positions = np.any(ch_locs != 0)

    fig_2d = fig_3d = None

    if has_positions:
        x = (ch_locs[:, 0] * 1000).tolist()
        y = (ch_locs[:, 1] * 1000).tolist()
        seen: set = set()
        lines_x, lines_y = [], []
        for ch in chs:
            src = tuple(round(v, 6) for v in ch["loc"][3:6])
            det = tuple(round(v, 6) for v in ch["loc"][:3])
            key = src + det
            if key in seen or not (any(src) or any(det)):
                continue
            seen.add(key)
            lines_x += [src[0] * 1000, det[0] * 1000, None]
            lines_y += [src[1] * 1000, det[1] * 1000, None]

        marker_colors_2d = [
            "#949e9f" if n in bad_channels else sci_color(sci_scores.get(n))
            for n in ch_names
        ]
        fig_2d = go.Figure(
            data=[
                go.Scatter(x=lines_x, y=lines_y, mode="lines",
                           line=dict(color="#bdc3c7", width=2),
                           hoverinfo="skip", showlegend=False),
                go.Scatter(x=x, y=y, mode="markers+text",
                           text=[n.split(" ")[0] for n in ch_names],
                           textposition="top center", textfont=dict(size=7),
                           marker=dict(size=10, color=marker_colors_2d, opacity=0.9,
                                       line=dict(width=0.8, color="#555")),
                           customdata=ch_names,
                           hovertemplate="<b>%{customdata}</b><extra></extra>",
                           showlegend=False),
            ],
            layout=go.Layout(
                xaxis=dict(showgrid=False, zeroline=False, title="x (mm)"),
                yaxis=dict(showgrid=False, zeroline=False, title="y (mm)", scaleanchor="x"),
                plot_bgcolor="#f8f9fa", paper_bgcolor="white",
                height=700, margin=dict(l=40, r=15, t=10, b=40),
                hovermode="closest",
            ),
        )

        coords_mm = ch_locs * 1000
        trans = None
        try:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            trans = mne.read_trans(os.path.join(fs_dir, "bem", "fsaverage-trans.fif"))
            coords_mm = mne.transforms.apply_trans(trans, ch_locs) * 1000
        except Exception:
            pass

        seen3: set = set()
        lx, ly, lz = [], [], []
        for ch in chs:
            src, det = ch["loc"][:3], ch["loc"][3:6]
            key = tuple(round(float(v), 5) for v in np.concatenate([src, det]))
            if key in seen3 or not (np.any(src) or np.any(det)):
                continue
            seen3.add(key)
            try:
                src_m = mne.transforms.apply_trans(trans, src.reshape(1, 3))[0] * 1000
                det_m = mne.transforms.apply_trans(trans, det.reshape(1, 3))[0] * 1000
            except Exception:
                src_m, det_m = src * 1000, det * 1000
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


def build_psd_mean_figure(raw: mne.io.Raw) -> go.Figure | None:
    try:
        from scipy.signal import welch
        picks = mne.pick_types(raw.info, meg=False, fnirs=True)
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        data = raw_od.get_data(picks=picks)
        sfreq = raw_od.info["sfreq"]
        nperseg = min(512, max(64, data.shape[1] // 4))
        freqs, psds = welch(data, fs=sfreq, nperseg=nperseg)
        fmax = min(2.0, sfreq / 2)
        mask = freqs <= fmax
        freqs_list = freqs[mask].tolist()
        traces = [
            go.Scatter(x=freqs_list, y=ch_psd, mode="lines",
                       line=dict(width=0.6, color="rgba(100,150,200,0.18)"),
                       showlegend=False, hoverinfo="skip")
            for ch_psd in psds[:, mask].tolist()
        ]
        traces.append(go.Scatter(
            x=freqs_list, y=psds[:, mask].mean(axis=0).tolist(),
            name="Mean (OD)", mode="lines", line=dict(width=2.5, color="#2980b9"),
        ))
        return go.Figure(data=traces, layout=go.Layout(**psd_layout()))
    except Exception as exc:
        logger.warning("mean PSD failed: %s", exc)
        return None


def build_sci_psp_figure(
    sci_scores: dict[str, float],
    psp_per_channel: dict[str, float],
    bad_channels: set[str],
    sci_threshold: float = 0.75,
    psp_threshold: float = 0.1,
    sci_matrix: np.ndarray | None = None,
    sci_win_times: np.ndarray | None = None,
    psp_matrix: np.ndarray | None = None,
    psp_win_times: np.ndarray | None = None,
) -> go.Figure:
    from .sci_psp_panel import lollipop_scores_figure as _lollipop, _SPACING

    ch_names = list(sci_scores.keys())
    sci_arr  = np.array([sci_scores.get(ch, 0.0) for ch in ch_names])
    has_psp  = bool(psp_per_channel)
    psp_arr  = np.array([psp_per_channel.get(ch, 0.0) for ch in ch_names]) if has_psp else None

    has_matrices = (
        sci_matrix is not None and sci_win_times is not None
        and psp_matrix is not None and psp_win_times is not None
        and has_psp
    )

    if not has_matrices:
        sci_colors = [
            "#e74c3c" if ch in bad_channels
            else ("#27ae60" if sci_scores.get(ch, 0.0) >= sci_threshold else "#f39c12")
            for ch in ch_names
        ]
        if not has_psp:
            return _lollipop(ch_names, sci_arr, sci_colors, sci_threshold, "SCI")
        psp_colors = ["#27ae60" if psp_per_channel.get(ch, 0.0) >= psp_threshold
                      else "#e74c3c" for ch in ch_names]
        fig_sci = _lollipop(ch_names, sci_arr, sci_colors, sci_threshold, "SCI")
        fig_psp = _lollipop(ch_names, psp_arr,  psp_colors, psp_threshold, "PSP")
        n_ch = len(ch_names)
        combined = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                 horizontal_spacing=0.08, subplot_titles=["SCI", "PSP"])
        for t in fig_sci.data: combined.add_trace(t, row=1, col=1)
        for t in fig_psp.data: combined.add_trace(t, row=1, col=2)
        combined.update_yaxes(tickvals=[i * _SPACING for i in range(n_ch)],
                               ticktext=ch_names, autorange="reversed",
                               tickfont=dict(size=9), row=1, col=1)
        combined.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
        combined.update_layout(height=min(max(300, n_ch * 14 + 100), 700),
                                margin=dict(l=110, r=30, t=40, b=40),
                                plot_bgcolor="white", paper_bgcolor="white")
        return combined

    ch_sorted  = ch_names
    sci_sorted = sci_arr
    psp_sorted = psp_arr
    sci_mat_s  = sci_matrix
    psp_mat_s  = psp_matrix
    wt_sci     = np.asarray(sci_win_times).ravel()
    wt_psp     = np.asarray(psp_win_times).ravel()

    sci_colors = [
        "#e74c3c" if ch in bad_channels
        else ("#27ae60" if sci_scores.get(ch, 0.0) >= sci_threshold else "#f39c12")
        for ch in ch_sorted
    ]
    psp_colors = ["#27ae60" if psp_per_channel.get(ch, 0.0) >= psp_threshold
                  else "#e74c3c" for ch in ch_sorted]

    n_ch   = len(ch_sorted)
    row_h  = min(max(220, n_ch * 14 + 80), 480)

    fig = make_subplots(
        rows=2, cols=2,
        shared_yaxes=True,
        column_widths=[0.875, 0.125],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.06,
        horizontal_spacing=0.02,
        subplot_titles=["SCI (windowed)", "Mean SCI", "PSP (windowed)", "Mean PSP"],
    )

    fig.add_trace(go.Heatmap(
        z=sci_mat_s, x=wt_sci.tolist(), y=ch_sorted,
        colorscale="RdYlGn", zmid=sci_threshold,
        colorbar=dict(title="SCI", thickness=10, len=0.44, y=0.78, x=1.01),
        hovertemplate="Ch: %{y}<br>t=%{x:.1f}s<br>SCI=%{z:.3f}<extra></extra>",
        name="SCI",
    ), row=1, col=1)

    sx, sy = [], []
    for ch, v in zip(ch_sorted, sci_sorted):
        sx += [0.0, float(v), None]; sy += [ch, ch, None]
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                             line=dict(color="#aaa", width=1.2),
                             showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=sci_sorted.tolist(), y=ch_sorted, mode="markers",
                             marker=dict(size=7, color=sci_colors,
                                         line=dict(width=0.5, color="#333")),
                             showlegend=False,
                             hovertemplate="%{y}: %{x:.3f}<extra></extra>"), row=1, col=2)
    fig.add_vline(x=sci_threshold, line_dash="dash", line_color="#888",
                  line_width=1, row=1, col=2)

    fig.add_trace(go.Heatmap(
        z=psp_mat_s, x=wt_psp.tolist(), y=ch_sorted,
        colorscale="RdYlGn", zmid=psp_threshold,
        colorbar=dict(title="PSP", thickness=10, len=0.44, y=0.22, x=1.01),
        hovertemplate="Ch: %{y}<br>t=%{x:.1f}s<br>PSP=%{z:.3f}<extra></extra>",
        name="PSP",
    ), row=2, col=1)

    px_, py_ = [], []
    for ch, v in zip(ch_sorted, psp_sorted):
        px_ += [0.0, float(v), None]; py_ += [ch, ch, None]
    fig.add_trace(go.Scatter(x=px_, y=py_, mode="lines",
                             line=dict(color="#aaa", width=1.2),
                             showlegend=False, hoverinfo="skip"), row=2, col=2)
    fig.add_trace(go.Scatter(x=psp_sorted.tolist(), y=ch_sorted, mode="markers",
                             marker=dict(size=7, color=psp_colors,
                                         line=dict(width=0.5, color="#333")),
                             showlegend=False,
                             hovertemplate="%{y}: %{x:.3f}<extra></extra>"), row=2, col=2)
    fig.add_vline(x=psp_threshold, line_dash="dash", line_color="#888",
                  line_width=1, row=2, col=2)

    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9), row=1, col=1)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=9), row=2, col=1)
    fig.update_yaxes(autorange="reversed", showticklabels=False, row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", gridcolor="#eee", row=2, col=1)
    fig.update_xaxes(title_text="Score", row=2, col=2)
    fig.update_layout(
        height=row_h * 2 + 80,
        margin=dict(l=120, r=110, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig




def build_epoch_preview_figure(
    raw_haemo: mne.io.Raw,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
) -> go.Figure | None:
    anns = raw_haemo.annotations
    markers = [
        {"onset": float(a["onset"]), "duration": float(a["duration"]),
         "description": str(a["description"])}
        for a in anns
        if not str(a["description"]).upper().startswith("BAD")
    ]
    if not markers:
        return None

    cond_colors_ = condition_colors(markers)
    colors10 = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    try:
        events_mne, event_id = mne.events_from_annotations(raw_haemo, verbose=False)
        if len(events_mne) == 0:
            return None

        epochs = mne.Epochs(
            raw_haemo, events_mne, event_id,
            tmin=epoch_tmin, tmax=epoch_tmax,
            baseline=(epoch_tmin, 0),
            preload=True, verbose=False,
        )
        hbo_picks = mne.pick_types(epochs.info, fnirs="hbo")
        hbr_picks = mne.pick_types(epochs.info, fnirs="hbr")

        traces = []
        for ci, cond in enumerate(event_id):
            try:
                ep = epochs[cond].get_data()
                base_color = cond_colors_.get(cond, colors10[ci % len(colors10)])
                if len(hbo_picks) > 0:
                    grand_hbo = ep[:, hbo_picks, :].mean(axis=(0, 1)) * 1e6
                    traces.append(go.Scatter(
                        x=epochs.times.tolist(), y=grand_hbo.tolist(),
                        name=f"{cond} HbO (n={ep.shape[0]})", mode="lines",
                        line=dict(color=base_color, width=2),
                    ))
                if len(hbr_picks) > 0:
                    grand_hbr = ep[:, hbr_picks, :].mean(axis=(0, 1)) * 1e6
                    traces.append(go.Scatter(
                        x=epochs.times.tolist(), y=grand_hbr.tolist(),
                        name=f"{cond} HbR", mode="lines",
                        line=dict(color=base_color, width=1.5, dash="dot"),
                    ))
            except Exception:
                pass

        if not traces:
            return None

        return go.Figure(
            data=traces,
            layout=go.Layout(
                xaxis=dict(title="Time rel. onset (s)", gridcolor="#eeeeee",
                           zerolinecolor="#cccccc"),
                yaxis=dict(title="Conc. (µmol/L)", gridcolor="#eeeeee"),
                shapes=[dict(type="line", xref="x", yref="paper",
                             x0=0, x1=0, y0=0, y1=1,
                             line=dict(color="#7f8c8d", width=1, dash="dash"))],
                annotations=[dict(x=0, y=1.0, xref="x", yref="paper",
                                  text="onset", showarrow=False,
                                  font=dict(size=8, color="#7f8c8d"),
                                  xanchor="left")],
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(l=60, r=15, t=20, b=38),
                legend=dict(font=dict(size=9)),
            ),
        )
    except Exception as exc:
        logger.warning("epoch preview failed: %s", exc)
        return None


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

    traces = []

    # Summary row at y=0: all conditions overlaid with their own colours
    for desc in all_descs:
        color = cond_colors.get(desc, "#999")
        onsets = [e["onset"] for e in by_desc[desc]]
        traces.append(go.Scatter(
            x=onsets,
            y=[0] * len(onsets),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=16, color=color,
                        line=dict(width=2.0, color=color)),
            name=desc,
            showlegend=False,
            hovertemplate=f"<b>{desc}</b><br>t=%{{x:.2f}} s<extra></extra>",
        ))

    # Per-condition rows starting at y=1
    for desc_idx, desc in enumerate(all_descs, start=1):
        color = cond_colors.get(desc, "#999")
        onsets = [e["onset"] for e in by_desc[desc]]
        traces.append(go.Scatter(
            x=onsets,
            y=[desc_idx] * len(onsets),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=16, color=color,
                        line=dict(width=2.0, color=color)),
            name=desc,
            showlegend=False,
            hovertemplate=f"<b>{desc}</b><br>t=%{{x:.2f}} s<extra></extra>",
        ))

    n = len(all_descs)
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title="Time (s)", gridcolor="#eeeeee"),
            yaxis=dict(tickvals=[0] + list(range(1, n + 1)),
                       ticktext=["(all)"] + all_descs,
                       autorange="reversed", gridcolor="#eeeeee",
                       tickfont=dict(size=10)),
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(80, (n + 1) * 40 + 50),
            margin=dict(l=120, r=15, t=8, b=38),
            hovermode="closest",
            showlegend=False,
        ),
    )


def build_evoked_topo_figure(
    raw_haemo: mne.io.Raw,
    markers: list[dict],
    max_ts_pts: int = 2000,
) -> go.Figure | None:
    hbo_entries = sorted(
        [(i, ch) for i, ch in enumerate(raw_haemo.ch_names) if ch.endswith(" hbo")],
        key=lambda x: x[1].rsplit(" ", 1)[0],
    )
    if not hbo_entries:
        return None

    pairs = [ch.rsplit(" ", 1)[0] for _, ch in hbo_entries]
    n = len(pairs)

    locs = np.array([raw_haemo.info["chs"][i]["loc"][:2] for i, _ in hbo_entries])
    if np.any(locs != 0):
        lo, hi = locs.min(axis=0), locs.max(axis=0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        norm = (locs - lo) / span * 0.82 + 0.06
    else:
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
        norm = np.array([
            [(i % ncols + 0.5) / ncols * 0.82 + 0.06,
             (i // ncols + 0.5) / nrows * 0.82 + 0.06]
            for i in range(n)
        ])

    _HBO_COLOR = "#e74c3c"
    _HBR_COLOR = "#3498db"

    picks = [i for i, _ in hbo_entries] + [
        raw_haemo.ch_names.index(f"{p} hbr")
        for p in pairs
        if f"{p} hbr" in raw_haemo.ch_names
    ]
    data, times = raw_haemo.get_data(picks=picks, return_times=True)
    data, times = _decimate(data, times, max_ts_pts)
    times_list = times.tolist()
    ch_names_picked = [raw_haemo.ch_names[i] for i in picks]

    _H, _ML, _MR, _MT, _MB = 700, 10, 10, 12, 8

    hw, hh = 0.050, 0.025
    box_shapes = []
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
        box_shapes.append(dict(
            type="rect", xref="paper", yref="paper",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="#ccc", width=0.8),
            fillcolor="rgba(255,255,255,0.80)",
            layer="below",
        ))

        first = (pi == 0)
        try:
            hbo_row = ch_names_picked.index(f"{pair} hbo")
            hbo_y = (data[hbo_row] * 1e6).tolist()
            fig.add_trace(go.Scatter(
                x=times_list, y=hbo_y,
                name="HbO", mode="lines",
                line=dict(color=_HBO_COLOR, width=1.0),
                xaxis=xr, yaxis=yr,
                customdata=[pair] * len(times_list),
                legendgroup="hbo", showlegend=first,
                hovertemplate=f"<b>{pair}</b> %{{x:.1f}}s %{{y:.2f}} µM<extra>HbO</extra>",
            ))
        except (ValueError, IndexError):
            pass

        try:
            hbr_row = ch_names_picked.index(f"{pair} hbr")
            hbr_y = (data[hbr_row] * 1e6).tolist()
            fig.add_trace(go.Scatter(
                x=times_list, y=hbr_y,
                name="HbR", mode="lines",
                line=dict(color=_HBR_COLOR, width=1.0),
                xaxis=xr, yaxis=yr,
                customdata=[pair] * len(times_list),
                legendgroup="hbr", showlegend=first,
                hovertemplate=f"<b>{pair}</b> %{{x:.1f}}s %{{y:.2f}} µM<extra>HbR</extra>",
            ))
        except (ValueError, IndexError):
            pass

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

    fig.update_layout(
        height=_H,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="white",
        margin=dict(l=_ML, r=_MR, t=_MT, b=_MB),
        shapes=box_shapes + head_shapes,
        legend=dict(x=0.99, y=0.99, xanchor="right",
                    font=dict(size=9), bgcolor="rgba(255,255,255,0.75)",
                    title=dict(text="HbO / HbR", font=dict(size=9))),
    )
    return fig
