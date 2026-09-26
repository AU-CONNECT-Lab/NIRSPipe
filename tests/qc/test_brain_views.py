"""The off-screen 3D brain views: each camera faces the side its label names, and a render
that cannot happen is reported rather than left blank."""

import base64
import io
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import pytest
from PIL import Image

from fnirs_pipe.qc.figures.common._brain_utils import VIEW_LABELS, load_mesh_traces
from fnirs_pipe.qc.figures.subject import brain_views as bv

NASION, LPA, RPA = [0.0, 0.085, -0.035], [-0.081, -0.029, -0.041], [0.084, -0.029, -0.041]
PURE = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0),
        "magenta": (255, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255)}


def _shows(img: np.ndarray, name: str) -> bool:
    """True when the panel holds a clear patch of this saturated colour, shading allowed."""
    on = np.array(PURE[name]) > 0
    rgb = img[:, :, :3].astype(int)
    hit = np.all(rgb[:, :, on] > 120, axis=2) & np.all(rgb[:, :, ~on] < 70, axis=2)
    return hit.sum() > 50


def _montage(sci_value: float = 0.95):
    names = ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"]
    info = mne.create_info(names, 10.0, ch_types=["fnirs_cw_amplitude"] * 4)
    for ch, wl in zip(info["chs"], [760.0, 850.0, 760.0, 850.0]):
        ch["loc"][9] = wl
    raw = mne.io.RawArray(np.zeros((4, 100)), info, verbose=False)
    # on the scalp over the left motor strip, outside the pial surface
    pos = {"S1": [-0.05, 0.02, 0.10], "D1": [-0.02, 0.02, 0.11],
           "S2": [-0.05, -0.02, 0.10], "D2": [-0.02, -0.02, 0.11]}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head",
                                                  nasion=NASION, lpa=LPA, rpa=RPA),
                    verbose="error")
    sci = dict.fromkeys(names, sci_value)
    return raw, sci


def test_each_camera_faces_the_pole_its_label_names():
    brain = load_mesh_traces()
    verts = np.vstack([m.points for m in brain])
    poles = {  # label -> (the pole it must show, the pole it must hide)
        "Frontal":      (("red", verts[verts[:, 1].argmax()]), ("blue", verts[verts[:, 1].argmin()])),
        "Left Lateral": (("green", verts[verts[:, 0].argmin()]), ("magenta", verts[verts[:, 0].argmax()])),
        "Superior":     (("yellow", verts[verts[:, 2].argmax()]), ("cyan", verts[verts[:, 2].argmin()])),
    }
    markers = [(bv._spheres(p[None, :], 6.0), PURE[c]) for pair in poles.values() for c, p in pair]
    plotter = bv._build_3d_scene(brain, markers)
    try:
        imgs = dict(zip(VIEW_LABELS, bv._render_views(plotter)))
    finally:
        plotter.close()

    for label, ((near, _), (far, _)) in poles.items():
        assert _shows(imgs[label], near), f"{label} view does not show the {near} marker"
        assert not _shows(imgs[label], far), f"{label} view shows the {far} marker behind it"


def _green_pixels(sci_value: float, good: bool) -> int:
    raw, sci = _montage(sci_value)
    b64 = bv.quality_brain_views(list(sci), None, np.full(4, good), raw=raw, sci_scores=sci)
    img = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")).astype(int)
    return int(np.all(np.abs(img - (0x27, 0xae, 0x60)) < 40, axis=2).sum())


def test_links_are_drawn_in_their_quality_colour():
    assert _green_pixels(0.95, good=True) > 50
    assert _green_pixels(0.10, good=False) == 0


def test_a_failed_render_raises_instead_of_returning_a_blank(monkeypatch):
    def broken():
        raise RuntimeError("no surface")

    monkeypatch.setattr(bv, "load_mesh_traces", broken)
    raw, sci = _montage()
    with pytest.raises(RuntimeError, match="no surface"):
        bv.quality_brain_views(list(sci), None, np.ones(4, bool), raw=raw, sci_scores=sci)


def test_subjects_rendering_in_parallel_threads_each_get_their_figure():
    raw, sci = _montage()

    def one(_):
        return bv.quality_brain_views(list(sci), None, np.ones(4, bool), raw=raw, sci_scores=sci)

    with ThreadPoolExecutor(max_workers=3) as pool:
        figures = list(pool.map(one, range(3)))
    assert len(figures) == 3 and all(figures)
