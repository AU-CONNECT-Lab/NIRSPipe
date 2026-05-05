"""Shared brain-rendering helpers for brain_views and glm_figures."""

import os

import mne
import numpy as np
import plotly.graph_objects as go

from fnirs_pipe.utils.logging import get_logger

_logger = get_logger("qc.figures.brain_utils")

CAMERAS = [
    dict(eye=dict(x=0.0,  y=-2.0, z=0.5)),   # frontal
    dict(eye=dict(x=-2.0, y=0.0,  z=0.3)),   # left lateral
    dict(eye=dict(x=0.0,  y=0.3,  z=2.0)),   # superior
]
VIEW_LABELS = ["Frontal", "Left Lateral", "Superior"]


def load_mesh_traces() -> list[go.Mesh3d]:
    """Load fsaverage5 pial surface as glass-style Mesh3d traces."""
    try:
        from nilearn import datasets, surface as surf
        fsavg5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        traces = []
        for key in ("pial_left", "pial_right"):
            verts, faces = surf.load_surf_mesh(fsavg5[key])
            traces.append(go.Mesh3d(
                x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
                i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
                color="#e3e3e3", opacity=0.45, flatshading=False,
                lighting=dict(ambient=0.85, diffuse=0.5, specular=0.3, fresnel=0.5),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo="skip", showlegend=False,
            ))
        return traces
    except Exception as exc:
        _logger.warning("fsaverage5 surface unavailable: %s", exc)
        return []


def to_mni(coords_head: np.ndarray) -> np.ndarray:
    """Transform head-space coordinates (m) to MNI (mm)."""
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        trans = mne.read_trans(os.path.join(fs_dir, "bem", "fsaverage-trans.fif"))
        return mne.transforms.apply_trans(trans, coords_head) * 1000
    except Exception as exc:
        _logger.warning("head→MNI transform failed (%s); using raw coords", exc)
        return coords_head * 1000


def render_3views(
    ch_scatter: go.Scatter3d,
    mesh_traces: list,
    width: int = 520,
    height: int = 460,
    scale: float = 2,
) -> list:
    """Render a Scatter3d channel figure at 3 camera angles to image arrays."""
    import io as _io
    try:
        import plotly.io as pio
        from PIL import Image as PILImage
    except ImportError as exc:
        _logger.warning("static render unavailable: %s", exc)
        return [None, None, None]

    fig = go.Figure()
    for mt in mesh_traces:
        fig.add_trace(mt)
    fig.add_trace(ch_scatter)

    scene_axis = dict(visible=False)
    scene_base = dict(xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis,
                      bgcolor="#ffffff")
    imgs = []
    for cam in CAMERAS:
        fig.update_layout(
            scene=dict(**scene_base, camera=cam),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#ffffff",
            showlegend=False,
        )
        try:
            png = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
            arr = np.array(PILImage.open(_io.BytesIO(png)))
            imgs.append(arr)
        except Exception as exc:
            _logger.warning("static render failed (%s)", exc)
            imgs.append(None)
    return imgs
