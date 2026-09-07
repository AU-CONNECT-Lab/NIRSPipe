"""Plotly figure builders for the interactive raw fNIRS QC viewer."""

from __future__ import annotations

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.metrics import PSP_PASS, SCI_PASS
from fnirs_pipe.utils.logging import get_logger

from ._brain_utils import mni_trans
from ._utils import (BAND_COLORS, CONDITION_PALETTE, HBO_COLOR, HBR_COLOR,
                     decimate as _decimate, epochable_events, physio_bands)

logger = get_logger("qc.figures")

_PSD_FMAX = 2.0


def _window_centers(win_times) -> np.ndarray:
    """[start, end] window pairs -> one centre per window; already-1-D input passes through.

    e.g. [(0.0, 10.1), (10.1, 20.2)] -> [5.05, 15.15]. Flattening instead would hand a
    heatmap twice as many x values as it has columns, and the extras are silently dropped,
    which compresses the plotted time axis to half the recording.
    """
    a = np.asarray(win_times, dtype=float)
    return a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a.ravel()


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


def psd_layout(height: int = 220, cardiac=None, resp=None) -> dict:
    """Layout for a single-panel PSD, with the physiological bands shaded.

    ``cardiac`` and ``resp`` are the run's own ``(l_freq, h_freq)``, which is the point: the
    band edges used to be constants here, so a study of infants (cardiac near 2 Hz) got a
    stripe drawn over the adult band and a reader checking whether a channel carries a pulse
    was looking at the wrong place. Passing None omits that band, and the colours and
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
    descs = list(dict.fromkeys(m["description"] for m in markers))
    return {d: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, d in enumerate(descs)}


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
            # one tick per trace, so the name sits on the trace instead of in a legend
            # whose order is the reverse of the stacking and whose spacing is unrelated
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
    epoch_tmin: float,
    epoch_tmax: float,
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
) -> tuple[go.Figure, go.Figure | None, go.Figure | None]:
    """One channel pair as three panels: HbO/HbR over time, its PSD, its epoch average.

    ``cardiac`` and ``resp`` are the run's own band edges, taken from the CLI, and shade the
    PSD panel the way the multi-stage PSD figure shades its rows. Without them the panel is
    a bare spectrum, and a reader judging whether a channel carries a pulse has to hold the
    band edges in their head; passing None omits that band rather than guessing a default.
    """
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
            height=160, margin=dict(l=55, r=15, t=8, b=38),
            legend=dict(font=dict(size=9), orientation="h", y=1.12),
            shapes=_mk_shapes(),
        ),
    )

    psd_fig = None
    try:
        from mne.time_frequency import psd_array_welch
        raw_arr = raw_haemo.get_data(picks=[hbo_pick, hbr_pick])
        sfreq = raw_haemo.info["sfreq"]
        # match the metric-side PSD (MNE compute_psd default Welch, n_fft=256)
        psds, freqs = psd_array_welch(raw_arr, sfreq, n_fft=min(256, raw_arr.shape[1]), verbose=False)
        psd_hbo, psd_hbr = psds[0], psds[1]
        fmax = min(2.0, sfreq / 2)
        mask = freqs <= fmax
        psd_fig = go.Figure(
            data=[
                go.Scatter(x=freqs[mask].tolist(), y=psd_hbo[mask].tolist(),
                           name="HbO", mode="lines", line=dict(color=HBO_COLOR, width=2)),
                go.Scatter(x=freqs[mask].tolist(), y=psd_hbr[mask].tolist(),
                           name="HbR", mode="lines", line=dict(color=HBR_COLOR, width=2)),
            ],
            layout=go.Layout(**psd_layout(cardiac=cardiac, resp=resp)),
        )
    except Exception as exc:
        logger.warning("channel PSD failed for %s: %s", ch_pair, exc)

    epoch_fig = None
    if markers:
        try:
            colors10 = CONDITION_PALETTE
            anns = mne.Annotations(
                onset=[m["onset"] for m in markers],
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


def _topomap_project(xyz: np.ndarray, sphere: np.ndarray) -> np.ndarray:
    """Flatten 3-D points the way MNE flattens sensors for a topomap.

    Azimuthal-equidistant projection about the fitted head sphere: translate to the sphere
    origin, read (azimuth, polar) as (angle, radius), scale radians back to metres. MNE only
    exposes this for channel positions, so optodes are projected here with the same sphere and
    land in the same frame as the channel midpoints returned by ``_get_pos_outlines``.

    e.g. a point on the sphere equator, 90 deg from the vertex, maps onto the head circle.
    """
    from mne.transforms import _cart_to_sph, _pol_to_cart

    sph = _cart_to_sph(np.asarray(xyz, dtype=float) - sphere[:3])
    out = _pol_to_cart(sph[:, 1:][:, ::-1])
    out *= sph[:, [0]] / (np.pi / 2.0)
    return out + sphere[:2]


def _optode_positions(chs, ch_names) -> tuple[dict, dict, list[tuple[str, str]]]:
    """Named source / detector coordinates read off the fNIRS channel locs.

    A channel "S1_D2 760" carries its source in ``loc[3:6]`` and its detector in ``loc[6:9]``
    (``loc[:3]`` is the midpoint, not the detector), so it contributes {"S1": xyz} to the
    sources, {"D2": xyz} to the detectors and ("S1", "D2") to the pairs; repeats collapse.
    """
    src: dict[str, np.ndarray] = {}
    det: dict[str, np.ndarray] = {}
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for ch, name in zip(chs, ch_names):
        pair = name.split(" ")[0]
        if "_" not in pair:
            continue
        s_name, d_name = pair.split("_")[:2]
        s_xyz, d_xyz = np.asarray(ch["loc"][3:6]), np.asarray(ch["loc"][6:9])
        if not (np.any(s_xyz) or np.any(d_xyz)):
            continue
        src.setdefault(s_name, s_xyz)
        det.setdefault(d_name, d_xyz)
        if (s_name, d_name) not in seen:
            seen.add((s_name, d_name))
            pairs.append((s_name, d_name))
    return src, det, pairs


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
    short_thresh: float,
) -> tuple[go.Figure | None, go.Figure | None]:
    chs = raw.info["chs"]
    ch_names = raw.ch_names
    colors = _ch_colors(raw, short_thresh)

    ch_locs = np.array([ch["loc"][:3] for ch in chs])
    has_positions = np.any(ch_locs != 0)

    fig_2d = fig_3d = None

    if has_positions:
        picks_2d = list(mne.pick_types(raw.info, meg=False, fnirs=True)) or list(range(len(chs)))
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
            "#949e9f" if n in bad_channels else sci_color(sci_scores.get(n))
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
            src, det = ch["loc"][:3], ch["loc"][3:6]
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


def build_psd_mean_figure(raw: mne.io.Raw, cardiac=None, resp=None) -> go.Figure | None:
    try:
        from mne.time_frequency import psd_array_welch
        picks = mne.pick_types(raw.info, meg=False, fnirs=True)
        if len(picks) == 0:
            picks = list(range(len(raw.ch_names)))
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        data = raw_od.get_data(picks=picks)
        sfreq = raw_od.info["sfreq"]
        # match the metric-side PSD (MNE compute_psd default Welch, n_fft=256)
        psds, freqs = psd_array_welch(data, sfreq, n_fft=min(256, data.shape[1]), verbose=False)
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
        return go.Figure(data=traces,
                         layout=go.Layout(**psd_layout(cardiac=cardiac, resp=resp)))
    except Exception as exc:
        logger.warning("mean PSD failed: %s", exc)
        return None


def build_sci_psp_figure(
    sci_scores: dict[str, float],
    psp_per_channel: dict[str, float],
    bad_channels: set[str],
    sci_threshold: float = SCI_PASS,
    psp_threshold: float = PSP_PASS,
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
    wt_sci     = _window_centers(sci_win_times)
    wt_psp     = _window_centers(psp_win_times)

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
        subplot_titles=["SCI (windowed)", "Mean SCI", "PSP (windowed)", "Mean PSP (10 s)"],
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




def _trial_image_data(
    raw_haemo: mne.io.Raw, picks: "list[int]", epoch_tmin: float, epoch_tmax: float,
) -> "list[tuple[str, np.ndarray, np.ndarray]] | None":
    """Epoch on (non-BAD) events, average over picks -> [(label, (n_trials, n_times) µM, times)].

    One entry per condition, e.g. two conditions with 20 trials each give
    [("all conditions", (40, n_times)), ("rest", (20, ...)), ("task", (20, ...))].
    The pooled entry leads so a condition with few trials can be read against it.
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
        out: "list[tuple[str, np.ndarray, np.ndarray]]" = []
        if len(event_id) > 1:
            # average over channels → (n_trials, n_times)
            out.append(("all conditions", data.mean(axis=1) * 1e6, epochs.times))
        for cond in event_id:
            try:
                cond_data = epochs[cond].get_data()
            except Exception:
                continue
            if cond_data.shape[0] == 0:
                continue
            out.append((str(cond), cond_data.mean(axis=1) * 1e6, epochs.times))
        return out or None
    except Exception:
        return None


def _smooth_trials(data: np.ndarray, trial_smooth: int) -> np.ndarray:
    if trial_smooth > 1 and data.shape[0] > 2 * trial_smooth:
        from scipy.ndimage import uniform_filter1d  # moving average across adjacent trials
        return uniform_filter1d(data, size=trial_smooth, axis=0, mode="nearest")
    return data


def _trial_image_plot(data: np.ndarray, times: np.ndarray, title: str, zmax: float) -> "go.Figure":
    """Trial x time heatmap + trial average. ``data`` is already trial-smoothed."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06,
        subplot_titles=[title, "trial average"],
    )
    fig.add_trace(go.Heatmap(
        z=data, x=times.tolist(), colorscale="RdBu_r", zmid=0, zmin=-zmax, zmax=zmax,
        zsmooth="best",  # bilinear interpolation → continuous look (fNIRS has few, wide time cells)
        colorbar=dict(title="µM", len=0.7, y=0.62),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=times.tolist(), y=data.mean(axis=0).tolist(),
        line=dict(color="#c0392b", width=1.5),
    ), row=2, col=1)
    fig.add_vline(x=0.0, line=dict(color="#333", width=1, dash="dash"))
    fig.update_yaxes(title_text="trial", row=1, col=1)
    fig.update_yaxes(title_text="µM", row=2, col=1)
    fig.update_xaxes(title_text="Time from onset (s)", row=2, col=1)
    fig.update_layout(
        height=480, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def _auto_trial_smooth(n_trials: int) -> int:
    return max(1, n_trials // 15)


def _trial_image_figures(
    res: "list[tuple[str, np.ndarray, np.ndarray]]",
    title_for: "callable",
    trial_smooth: "int | None",
) -> "list[go.Figure]":
    """One figure per condition, on a colour scale shared across them so panels compare.

    Smoothing runs first: the scale is taken from what is actually drawn, and an unsmoothed
    single-trial spike would otherwise wash out every panel.
    """
    panels = []
    for label, data, times in res:
        sm = trial_smooth if trial_smooth is not None else _auto_trial_smooth(data.shape[0])
        panels.append((label, _smooth_trials(data, sm), times))
    zmax = max(
        (float(np.nanpercentile(np.abs(d), 97)) for _, d, _ in panels),
        default=0.0,
    ) or 1.0
    return [
        _trial_image_plot(data, times, title_for(label, data.shape[0]), zmax)
        for label, data, times in panels
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
        res, lambda label, n: f"Trial image — {ch_name} / {label} ({n} trials)", trial_smooth)


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
    hbo = {c for c in raw_haemo.ch_names if c.endswith(" hbo")}
    picks = [raw_haemo.ch_names.index(c if c in hbo else f"{c} hbo")
             for c in channels if (c in hbo or f"{c} hbo" in hbo)]
    if not picks:
        return None
    res = _trial_image_data(raw_haemo, picks, epoch_tmin, epoch_tmax)
    if res is None:
        return None
    return _trial_image_figures(
        res,
        lambda label, n: f"Trial image — ROI {roi_name} / {label} ({len(picks)} ch, {n} trials)",
        trial_smooth,
    )


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
    colors10 = CONDITION_PALETTE
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
            showlegend=True,
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
            height=max(120, (n + 1) * 60 + 60),
            margin=dict(l=120, r=100, t=8, b=38),
            hovermode="closest",
            showlegend=True,
            legend=dict(x=1.01, y=1.0, xanchor="left",
                        font=dict(size=9), itemsizing="constant"),
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

    pairs = [ch.rsplit(" ", 1)[0] for _, ch in hbo_entries]
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
