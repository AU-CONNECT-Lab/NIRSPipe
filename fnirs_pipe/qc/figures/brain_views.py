"""Static 3D three-view brain figure with channel spheres coloured by quality.

Three camera perspectives (frontal, left lateral, superior) assembled into a
single base64 PNG (no Plotly wrapper — avoids large go.Image JSON overhead).
brain_viewer.py is intentionally not touched.
"""

import base64
import io as _io
import os

import mne
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image as _PILImage

from fnirs_pipe.utils.logging import get_logger
from ._brain_utils import CAMERAS, VIEW_LABELS, load_mesh_traces, to_mni

logger = get_logger("qc.figures.brain_views")

_GOOD_COLOR = "#27ae60"
_BAD_COLOR  = "#e74c3c"


def _trim_white(arr: np.ndarray, pad: int = 6, threshold: int = 252) -> np.ndarray:
    """Crop near-white columns only; preserve row count to keep vertical alignment."""
    mask = np.any(arr[:, :, :3] < threshold, axis=2)
    cols = np.where(np.any(mask, axis=0))[0]
    if not cols.size:
        return arr
    c0, c1 = cols[0], cols[-1]
    return arr[:, max(0, c0 - pad): c1 + pad + 1]


def _build_3d_scene(ch_names, coords_mni, colors, mesh_traces, sd_line_trace) -> go.Figure:
    """Single-panel 3D figure used for static rendering."""
    fig = go.Figure()
    for mt in mesh_traces:
        fig.add_trace(mt)
    if sd_line_trace is not None:
        fig.add_trace(sd_line_trace)
    fig.add_trace(go.Scatter3d(
        x=coords_mni[:, 0].tolist(),
        y=coords_mni[:, 1].tolist(),
        z=coords_mni[:, 2].tolist(),
        mode="markers",
        marker=dict(size=6, color=colors, opacity=0.95,
                    line=dict(width=0.5, color="#333")),
        text=ch_names,
        hovertemplate="<b>%{text}</b><br>%{x:.1f}, %{y:.1f}, %{z:.1f} mm<extra></extra>",
        showlegend=False,
    ))
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
) -> str | None:
    """Render 3-view brain figure and return as base64 PNG string, or None on failure.

    Avoids embedding go.Image traces in Plotly (which inflates HTML size significantly).
    """
    coords_mni = to_mni(coords_head)
    colors = [_GOOD_COLOR if g else _BAD_COLOR for g in good_mask]

    mesh_traces = load_mesh_traces()
    sd_line_trace = None
    if raw is not None:
        try:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            trans = mne.read_trans(os.path.join(fs_dir, "bem", "fsaverage-trans.fif"))
            seen: set = set()
            lx, ly, lz = [], [], []
            for ch in raw.info["chs"]:
                src, det = ch["loc"][:3], ch["loc"][3:6]
                key = tuple(round(float(v), 5) for v in np.concatenate([src, det]))
                if key in seen or not (np.any(src) or np.any(det)):
                    continue
                seen.add(key)
                s = mne.transforms.apply_trans(trans, src[None])[0] * 1000
                d = mne.transforms.apply_trans(trans, det[None])[0] * 1000
                lx += [float(s[0]), float(d[0]), None]
                ly += [float(s[1]), float(d[1]), None]
                lz += [float(s[2]), float(d[2]), None]
            if lx:
                sd_line_trace = go.Scatter3d(
                    x=lx, y=ly, z=lz, mode="lines",
                    line=dict(color="#95a5a6", width=3),
                    hoverinfo="skip", showlegend=False,
                )
        except Exception as exc:
            logger.warning("S-D lines failed: %s", exc)

    fig_3d = _build_3d_scene(ch_names, coords_mni, colors, mesh_traces, sd_line_trace)

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
