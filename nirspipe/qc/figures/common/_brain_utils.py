"""Shared brain-rendering helpers for brain_views and glm_figures."""

import threading

import mne
import numpy as np

from nirspipe.io.optode_space import head_to_mri, placement_of
from nirspipe.utils.logging import get_logger

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


# ---- Optode placement ----
#
# Decided once per run by nirspipe.io.optode_space and recorded beside it. A run no reading
# places is drawn as its coordinates stand, which claims no anatomy.

def to_mni(coords: np.ndarray, raw: "mne.io.Raw | None" = None) -> np.ndarray:
    """Optode coordinates (m) on fsaverage surface RAS (mm), by the run's placement.

    Without `raw` the coordinates are taken to be in head space.
    """
    if raw is None:
        return mne.transforms.apply_trans(head_to_mri(), coords) * 1000
    placement = placement_of(raw)
    if placement.placed:
        return placement.apply(coords)
    _logger.info("optodes not placed on fsaverage (%s); drawing the coordinates as they are",
                 placement.note)
    return np.asarray(coords, float) * 1000


def to_head(coords: np.ndarray, raw: "mne.io.Raw") -> np.ndarray:
    """Optode coordinates (m) in fsaverage's head frame (m), for MNE routines that take `trans="fsaverage"`."""
    return mne.transforms.apply_trans(np.linalg.inv(head_to_mri()), to_mni(coords, raw) / 1000)
