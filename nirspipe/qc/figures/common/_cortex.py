"""Channel values projected onto the fsaverage cortex and rendered off-screen, without Qt."""

import os
from typing import Callable

import mne
import numpy as np

from nirspipe.qc.figures.common._brain_utils import to_head
from nirspipe.qc.figures.common._surface_overlay import smoothing_matrix, vertex_normals


def project_to_cortex(raw: mne.io.Raw, columns: np.ndarray) -> tuple:
    """``(stc, subjects_dir)``: each column (a row per channel of ``raw``) spread over the
    fsaverage pial surface within 3 cm of the channels, as ``plot_glm_surface_projection`` does.

    ``weighted`` mode gives each vertex its sensors' weights summing to one, so a column of
    ones projects to the share of a vertex the channels cover.
    """
    from mne import EvokedArray, read_source_spaces
    from mne._fiff.constants import FIFF
    from mne.source_estimate import stc_near_sensors
    from mne.utils import get_subjects_dir

    ea = EvokedArray(np.asarray(columns, float).reshape(len(raw.ch_names), -1), raw.info.copy())
    # trans="fsaverage" makes stc_near_sensors read loc as head coords, so move the
    # optodes into that frame rather than just relabelling them
    for idx in range(len(ea.ch_names)):
        loc = ea.info["chs"][idx]["loc"]
        loc[:9] = to_head(loc[:9].reshape(3, 3), raw).reshape(9)
        ea.info["chs"][idx]["coord_frame"] = FIFF.FIFFV_COORD_HEAD

    subjects_dir = get_subjects_dir(raise_error=True)
    src = read_source_spaces(os.path.join(subjects_dir, "fsaverage", "bem",
                                          "fsaverage-ico-5-src.fif"))
    stc = stc_near_sensors(
        evoked=ea, picks=np.arange(len(ea.ch_names)), subject="fsaverage", trans="fsaverage",
        distance=0.03, mode="weighted", surface="pial",
        subjects_dir=subjects_dir, src=src, project=True, verbose=False,
    )
    return stc, subjects_dir


def mne_lighting(plotter) -> None:
    """MNE's three camera lights in place of the headlight: below, upper left, upper right."""
    lights = list(plotter.renderer.GetLights())
    lights[0].SetSwitch(False)
    placed = ((0, -45), (-60, 30), (60, 30))
    for i, light in enumerate(lights[1:]):
        az, el = placed[i] if i < len(placed) else (0.0, 0.0)
        theta, phi = np.deg2rad(az), np.deg2rad(90.0 - el)
        light.SetPosition(np.sin(theta) * np.sin(phi), np.cos(phi), np.cos(theta) * np.sin(phi))
        light.SetIntensity(0.7 if i < len(placed) else 0.0)
        light.SetSwitch(i < len(placed))
        light.SetColor(1.0, 1.0, 1.0)


def render_cortex(stc: mne.SourceEstimate,
                  colour: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  subjects_dir, size: tuple[int, int], view_configs: list[dict]) -> list[np.ndarray]:
    """One RGB image per view of ``stc`` on the fsaverage pial surface, as ``stc.plot`` draws it.

    ``colour(curvature, smoothed)`` turns a hemisphere's curvature and its data, spread from
    the source vertices over 10 smoothing steps (``(n_vertices, n_columns)``), into RGBA.
    A view is ``{"azimuth", "elevation"}`` in MNE's convention.
    """
    import pyvista as pv
    from nibabel.freesurfer import read_morph_data

    surf_dir = os.path.join(subjects_dir, "fsaverage", "surf")
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    try:
        plotter.set_background("white")
        for hemi, vertices, data in (("lh", stc.vertices[0], stc.lh_data),
                                     ("rh", stc.vertices[1], stc.rh_data)):
            coords, faces = mne.read_surface(os.path.join(surf_dir, f"{hemi}.pial"))
            curv = read_morph_data(os.path.join(surf_dir, f"{hemi}.curv"))
            smooth = smoothing_matrix(faces, len(coords), vertices, steps=10)

            mesh = pv.PolyData(coords, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
            mesh.point_data["Normals"] = vertex_normals(coords, faces)
            mesh.GetPointData().SetActiveNormals("Normals")
            mesh.point_data["Data"] = colour(curv, smooth @ data)
            actor = plotter.add_mesh(mesh, scalars="Data", rgba=True, reset_camera=False)
            actor.prop.interpolation = "phong"

        mne_lighting(plotter)
        focal = np.asarray(plotter.center, float)
        images = []
        for cfg in view_configs:
            # MNE's convention: elevation is the polar angle from +z; camera 480 mm from origin
            az, el = np.deg2rad(cfg["azimuth"]), np.deg2rad(cfg["elevation"])
            position = 480 * np.array([np.sin(el) * np.cos(az), np.sin(el) * np.sin(az),
                                       np.cos(el)])
            up = (0, 0, 1) if 5 <= abs(cfg["elevation"]) <= 175 else (0, 1, 0)
            plotter.camera_position = [position, focal, up]
            plotter.render()
            images.append(np.asarray(plotter.screenshot(return_img=True))[:, :, :3])
        return images
    finally:
        plotter.close()
