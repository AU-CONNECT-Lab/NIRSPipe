"""Shared brain-rendering helpers for brain_views and glm_figures."""

import os
from functools import lru_cache

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


def load_mesh_traces(opacity: float = 1.0) -> list[go.Mesh3d]:
    """Load fsaverage5 pial surface as Mesh3d traces, opaque by default.

    An opaque surface hides whatever sits behind it, which is the point: optodes on
    the far side of the head no longer show through and crowd the near-side ones.
    """
    try:
        from nilearn import datasets, surface as surf
        fsavg5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        traces = []
        for key in ("pial_left", "pial_right"):
            verts, faces = surf.load_surf_mesh(fsavg5[key])
            traces.append(go.Mesh3d(
                x=verts[:, 0].tolist(), y=verts[:, 1].tolist(), z=verts[:, 2].tolist(),
                i=faces[:, 0].tolist(), j=faces[:, 1].tolist(), k=faces[:, 2].tolist(),
                color="#e8e8e8", opacity=opacity, flatshading=False,
                lighting=dict(ambient=0.5, diffuse=0.75, specular=0.12, fresnel=0.15),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo="skip", showlegend=False,
            ))
        return traces
    except Exception as exc:
        _logger.warning("fsaverage5 surface unavailable: %s", exc)
        return []


# ---- Optode coordinate frames ----
#
# SNIRF does not record which space its optode coordinates live in. Some montages are in
# MNE head coordinates, others come straight from an acquisition template already in
# MNI. Applying head->MRI to the latter shifts the whole cap off the brain.

_HEAD_FID_TOL = 0.010   # m; how far a fiducial may stray from its head-frame axis


@lru_cache(maxsize=1)
def _head_to_mri() -> np.ndarray:
    """fsaverage head->MRI (surface RAS) as a 4x4, or identity if fsaverage is missing."""
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        trans = mne.read_trans(os.path.join(fs_dir, "bem", "fsaverage-trans.fif"))
        return np.asarray(trans["trans"], float)
    except Exception as exc:
        _logger.warning("fsaverage head->MRI transform unavailable (%s); using raw coords", exc)
        return np.eye(4)


def _fiducials_are_head_like(info) -> bool:
    """True when LPA/RPA sit on the x axis and the nasion on the y axis, as head coords do."""
    from mne._fiff.constants import FIFF

    fids = {d["ident"]: np.asarray(d["r"], float)
            for d in (info["dig"] or []) if d["kind"] == FIFF.FIFFV_POINT_CARDINAL}
    lpa = fids.get(FIFF.FIFFV_POINT_LPA)
    nas = fids.get(FIFF.FIFFV_POINT_NASION)
    rpa = fids.get(FIFF.FIFFV_POINT_RPA)
    if lpa is None or nas is None or rpa is None:
        return False
    off_axis = [lpa[1], lpa[2], rpa[1], rpa[2], nas[0], nas[2]]
    return max(abs(float(v)) for v in off_axis) < _HEAD_FID_TOL


def mni_trans(info) -> np.ndarray:
    """4x4 taking this recording's optode coordinates (m) to fsaverage surface RAS (m).

    Falls back to the shape of the fiducials when coord_frame is UNKNOWN, and to no
    transform when even those are missing.
    """
    from mne._fiff.constants import FIFF

    chs = info["chs"]
    frame = chs[0]["coord_frame"] if len(chs) else FIFF.FIFFV_COORD_UNKNOWN
    if frame == FIFF.FIFFV_COORD_MRI:
        return np.eye(4)
    if frame == FIFF.FIFFV_COORD_HEAD:
        return _head_to_mri()
    if _fiducials_are_head_like(info):
        return _head_to_mri()
    _logger.info("optode frame unknown and fiducials are not head-like; "
                 "treating coordinates as MNI and applying no transform")
    return np.eye(4)


def to_mni(coords: np.ndarray, info=None) -> np.ndarray:
    """Transform optode coordinates (m) to fsaverage surface RAS (mm).

    Without `info` the coordinates are assumed to be in head space.
    """
    trans = mni_trans(info) if info is not None else _head_to_mri()
    return mne.transforms.apply_trans(trans, coords) * 1000


def to_head(coords: np.ndarray, info) -> np.ndarray:
    """Inverse of `to_mni`, for MNE routines that take `trans="fsaverage"`."""
    mri = mne.transforms.apply_trans(mni_trans(info), coords)
    return mne.transforms.apply_trans(np.linalg.inv(_head_to_mri()), mri)


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
