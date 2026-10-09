"""Shared brain-rendering helpers for brain_views and glm_figures."""

import os
import threading
from functools import lru_cache

import mne
import numpy as np

from fnirs_pipe.utils.logging import get_logger

_logger = get_logger("qc.figures.brain_utils")

# (direction from the brain's centre to the camera, the image's up), in fsaverage RAS
# (+x right, +y anterior, +z up); superior puts the nose up, as the optode flat map does
CAMERAS = [
    ((0.0,  2.0, 0.5), (0, 0, 1)),    # frontal
    ((-2.0, 0.0, 0.3), (0, 0, 1)),    # left lateral
    ((0.0,  0.3, 2.0), (0, 1, 0)),    # superior
]
VIEW_LABELS = ["Frontal", "Left Lateral", "Superior"]

# VTK cannot render from two threads at once, and subjects run in threads
RENDER_LOCK = threading.Lock()


def load_brain_meshes() -> list:
    """fsaverage5 pial surface, left and right, as pyvista meshes in mm."""
    import pyvista as pv
    from nilearn import datasets, surface as surf

    fsavg5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    meshes = []
    for key in ("pial_left", "pial_right"):
        verts, faces = surf.load_surf_mesh(fsavg5[key])
        cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
        meshes.append(pv.PolyData(np.asarray(verts, float), cells))
    return meshes


# ---- Optode coordinate frames ----
#
# SNIRF does not record which space its optode coordinates live in. Some montages are in
# MNE head coordinates, others come straight from an acquisition template already in
# MNI. Applying head->MRI to the latter shifts the whole cap off the brain.

_HEAD_FID_TOL = 0.010   # m; how far a fiducial may stray from its head-frame axis


@lru_cache(maxsize=1)
def _head_to_mri() -> np.ndarray:
    """fsaverage head->MRI (surface RAS) as a 4x4; raises when fsaverage cannot be loaded."""
    try:
        fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
        trans = mne.read_trans(os.path.join(fs_dir, "bem", "fsaverage-trans.fif"))
    except Exception as exc:
        raise RuntimeError(
            f"fsaverage is needed to place the optodes on the brain and could not be loaded "
            f"({exc}); run mne.datasets.fetch_fsaverage() once on a machine with internet "
            f"access") from exc
    return np.asarray(trans["trans"], float)


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
