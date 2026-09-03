"""Static 3D three-view brain figure with channel links coloured by quality.

A channel is the source-detector segment, so quality is drawn on the link, not on
a sphere at the channel midpoint. Colours follow the 2-D optode flat map so the
two panels of the combined figure read the same way.

Three camera perspectives (frontal, left lateral, superior) assembled into a
single base64 PNG (no Plotly wrapper — avoids large go.Image JSON overhead).
brain_viewer.py is intentionally not touched.
"""

import base64
import io as _io
import re

import mne
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image as _PILImage

from fnirs_pipe.utils.logging import get_logger
from ._brain_utils import CAMERAS, VIEW_LABELS, load_mesh_traces, to_mni

logger = get_logger("qc.figures.brain_views")

_GOOD_COLOR = "#27ae60"
_MID_COLOR  = "#f39c12"
_BAD_COLOR  = "#e74c3c"
_NA_COLOR   = "#95a5a6"
# red source / blue detector is the field convention, and matches the 2-D flat map;
# it re-uses the quality colours, but optodes are dots and channels are lines
_SRC_COLOR, _SRC_EDGE = "#e74c3c", "#922b21"
_DET_COLOR, _DET_EDGE = "#2980b9", "#1a5276"

_CH_RE = re.compile(r"(S\d+)[_\s]+(D\d+)", re.IGNORECASE)


def _trim_white(arr: np.ndarray, pad: int = 6, threshold: int = 252) -> np.ndarray:
    """Crop near-white columns only; preserve row count to keep vertical alignment."""
    mask = np.any(arr[:, :, :3] < threshold, axis=2)
    cols = np.where(np.any(mask, axis=0))[0]
    if not cols.size:
        return arr
    c0, c1 = cols[0], cols[-1]
    return arr[:, max(0, c0 - pad): c1 + pad + 1]


def _link_color(sci: float | None, good: bool | None) -> str:
    """SCI thresholds first (matches the flat map), pass/fail as fallback."""
    if sci is not None and not np.isnan(sci):
        if sci >= 0.75:
            return _GOOD_COLOR
        return _MID_COLOR if sci >= 0.5 else _BAD_COLOR
    if good is not None:
        return _GOOD_COLOR if good else _BAD_COLOR
    return _NA_COLOR


def _lookup_sci(sci_scores: dict, name: str) -> float | None:
    """SCI for a channel, trying the chromophore-suffixed names of its base."""
    base = name.split(" ")[0]
    for key in (name, base + " hbo", base + " hbr"):
        v = sci_scores.get(key)
        if v is not None:
            return float(v)
    return None


def _collect_pairs(raw: mne.io.Raw, sci_scores: dict, good_by_base: dict) -> tuple[dict, dict, dict]:
    """One entry per S-D pair: endpoints in head space, mean SCI, pass/fail.

    e.g. channels "S1_D1 760"/"S1_D1 850" collapse to pair "S1_D1" whose colour
    comes from the mean of their two SCI values.
    """
    ends: dict[str, tuple] = {}
    scis: dict[str, list[float]] = {}
    goods: dict[str, bool] = {}

    for idx in mne.pick_types(raw.info, fnirs=True):
        ch   = raw.info["chs"][idx]
        name = raw.info["ch_names"][idx]
        m = _CH_RE.search(name)
        if m is None:
            continue
        src = np.asarray(ch["loc"][3:6], dtype=float)
        det = np.asarray(ch["loc"][6:9], dtype=float)
        if np.isnan(src).any() or np.isnan(det).any():
            continue
        if not (np.any(src) or np.any(det)):
            continue

        pair_id = f"{m.group(1).upper()}_{m.group(2).upper()}"
        ends[pair_id] = (src, det)

        sci = _lookup_sci(sci_scores, name)
        if sci is not None:
            scis.setdefault(pair_id, []).append(sci)
        good = good_by_base.get(name.split(" ")[0])
        if good is not None:
            goods[pair_id] = goods.get(pair_id, True) and good

    return ends, scis, goods


def _link_traces(raw: mne.io.Raw, sci_scores: dict, good_by_base: dict) -> list[go.Scatter3d]:
    """S-D segments grouped into one trace per quality colour, plus optode dots."""
    ends, scis, goods = _collect_pairs(raw, sci_scores, good_by_base)
    if not ends:
        return []

    pair_ids = list(ends)
    flat = np.vstack([np.vstack(ends[p]) for p in pair_ids])
    mni  = to_mni(flat, raw.info)

    groups: dict[str, dict[str, list]] = {}
    sources: dict[str, np.ndarray] = {}
    detectors: dict[str, np.ndarray] = {}
    for i, pair_id in enumerate(pair_ids):
        s, d = mni[2 * i], mni[2 * i + 1]
        vals = scis.get(pair_id)
        sci  = float(np.mean(vals)) if vals else None
        color = _link_color(sci, goods.get(pair_id))
        g = groups.setdefault(color, {"x": [], "y": [], "z": [], "t": []})
        g["x"] += [float(s[0]), float(d[0]), None]
        g["y"] += [float(s[1]), float(d[1]), None]
        g["z"] += [float(s[2]), float(d[2]), None]
        label = f"{pair_id}<br>SCI = " + (f"{sci:.3f}" if sci is not None else "N/A")
        g["t"] += [label, label, None]

        src_id, det_id = pair_id.split("_")
        sources.setdefault(src_id, s)
        detectors.setdefault(det_id, d)

    traces = [
        go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"], mode="lines",
            line=dict(color=color, width=6),
            text=g["t"], hoverinfo="text", showlegend=False,
        )
        for color, g in groups.items()
    ]
    optode_specs = (
        (sources,   _SRC_COLOR, _SRC_EDGE, 5),
        (detectors, _DET_COLOR, _DET_EDGE, 4),
    )
    for optodes, color, edge, size in optode_specs:
        if not optodes:
            continue
        pos = np.vstack(list(optodes.values()))
        traces.append(go.Scatter3d(
            x=pos[:, 0].tolist(), y=pos[:, 1].tolist(), z=pos[:, 2].tolist(),
            mode="markers",
            marker=dict(size=size, color=color, opacity=1.0,
                        line=dict(width=1.0, color=edge)),
            text=list(optodes), hoverinfo="text", showlegend=False,
        ))
    return traces


def _channel_marker_trace(ch_names, coords_mni, good_mask) -> go.Scatter3d:
    """Fallback when optode positions are unavailable: dots at channel midpoints."""
    return go.Scatter3d(
        x=coords_mni[:, 0].tolist(),
        y=coords_mni[:, 1].tolist(),
        z=coords_mni[:, 2].tolist(),
        mode="markers",
        marker=dict(size=6, color=[_GOOD_COLOR if g else _BAD_COLOR for g in good_mask],
                    opacity=0.95, line=dict(width=0.5, color="#333")),
        text=ch_names,
        hovertemplate="<b>%{text}</b><br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>",
        showlegend=False,
    )


def _build_3d_scene(mesh_traces, data_traces) -> go.Figure:
    """Single-panel 3D figure used for static rendering."""
    fig = go.Figure()
    for mt in mesh_traces:
        fig.add_trace(mt)
    for tr in data_traces:
        fig.add_trace(tr)
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), bgcolor="#ffffff",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#ffffff",
        showlegend=False,
    )
    return fig


def quality_brain_views(
    ch_names: list[str],
    coords_head: np.ndarray,
    good_mask: np.ndarray,
    raw: mne.io.Raw | None = None,
    sci_scores: dict[str, float] | None = None,
) -> str | None:
    """Render 3-view brain figure and return as base64 PNG string, or None on failure.

    Avoids embedding go.Image traces in Plotly (which inflates HTML size significantly).
    """
    mesh_traces = load_mesh_traces()

    good_by_base = {n.split(" ")[0]: bool(g) for n, g in zip(ch_names, good_mask)}
    data_traces: list = []
    if raw is not None:
        try:
            data_traces = _link_traces(raw, sci_scores or {}, good_by_base)
        except Exception as exc:
            logger.warning("channel links failed: %s", exc)
    if not data_traces:
        data_traces = [_channel_marker_trace(
            ch_names,
            to_mni(coords_head, raw.info if raw is not None else None),
            good_mask,
        )]

    fig_3d = _build_3d_scene(mesh_traces, data_traces)

    scene_axis = dict(visible=False, showbackground=False, showgrid=False,
                      showspikes=False, showline=False, zeroline=False)
    scene_base = dict(xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis,
                      bgcolor="#ffffff")
    imgs = []
    for cam, label in zip(CAMERAS, VIEW_LABELS):
        fig_3d.update_layout(scene=dict(**scene_base, camera=cam),
                             title=label, title_x=0.5,
                             title_font=dict(size=11))
        try:
            png = pio.to_image(fig_3d, format="png", width=700, height=560, scale=2)
            arr = np.array(_PILImage.open(_io.BytesIO(png)))
            imgs.append(arr)
        except Exception as exc:
            logger.warning("Static render failed (%s); panel skipped", exc)

    if not imgs:
        return None

    trimmed = [_trim_white(a) for a in imgs]
    h = min(a.shape[0] for a in trimmed)
    combined = np.hstack([a[:h, :] for a in trimmed])
    pil = _PILImage.fromarray(combined.astype(np.uint8))
    buf = _io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
