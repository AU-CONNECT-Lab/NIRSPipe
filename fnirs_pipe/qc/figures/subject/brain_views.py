"""Static 3D three-view brain figure with channel links coloured by quality.

A channel is the source-detector segment, so quality is drawn on the link, not on
a sphere at the channel midpoint. Colours follow the 2-D optode flat map so the
two panels of the combined figure read the same way.

Three camera perspectives (frontal, left lateral, superior) rendered off-screen and
assembled into a single base64 PNG.
"""

import base64
import io as _io
import re

import mne
import numpy as np
from PIL import Image as _PILImage

from fnirs_pipe.utils import pair_of
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.qc.figures.common._brain_utils import (
    CAMERAS, RENDER_LOCK, VIEW_LABELS, load_brain_meshes, to_mni,
)
from fnirs_pipe.qc.figures.subject.raw_figures import SCI_WARN_RATIO

logger = get_logger("qc.figures.brain_views")

_GOOD_COLOR = "#27ae60"
_MID_COLOR  = "#f39c12"
_BAD_COLOR  = "#e74c3c"
_NA_COLOR   = "#95a5a6"
# red source / blue detector, matching the 2-D flat map;
# it re-uses the quality colours, but optodes are dots and channels are lines
_SRC_COLOR = "#e74c3c"
_DET_COLOR = "#2980b9"
# drawn sizes, in mm on the fsaverage surface
_LINK_RADIUS = 1.0
_SRC_RADIUS, _DET_RADIUS, _MARKER_RADIUS = 3.0, 2.5, 3.0
_VIEW_SIZE = (1400, 1120)

_CH_RE = re.compile(r"(S\d+)[_\s]+(D\d+)", re.IGNORECASE)


def _trim_white(arr: np.ndarray, pad: int = 6, threshold: int = 252) -> np.ndarray:
    """Crop near-white columns only; preserve row count to keep vertical alignment."""
    mask = np.any(arr[:, :, :3] < threshold, axis=2)
    cols = np.where(np.any(mask, axis=0))[0]
    if not cols.size:
        return arr
    c0, c1 = cols[0], cols[-1]
    return arr[:, max(0, c0 - pad): c1 + pad + 1]


def _link_color(sci: float | None, good: bool | None, threshold: float) -> str:
    """The screening verdict first, then the SCI ladder (which matches the flat map).

    ``good`` is the verdict and SCI only grades under it. Channels are screened on how many
    windows they were coupled in, so a channel at SCI 0.96 can be dropped and must not be
    drawn green. See ``sci_color``, which carries the same rule for the flat map beside this
    one.
    """
    if good is False:
        return _BAD_COLOR
    if sci is not None and not np.isnan(sci):
        if sci >= threshold:
            return _GOOD_COLOR
        return _MID_COLOR if sci >= SCI_WARN_RATIO * threshold else _BAD_COLOR
    if good is not None:
        return _GOOD_COLOR if good else _BAD_COLOR
    return _NA_COLOR


def _lookup_sci(sci_scores: dict, name: str) -> float | None:
    """SCI for a channel, trying the chromophore-suffixed names of its base."""
    base = pair_of(name)
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
        good = good_by_base.get(pair_of(name))
        if good is not None:
            goods[pair_id] = goods.get(pair_id, True) and good

    return ends, scis, goods


def _spheres(points: np.ndarray, radius: float):
    import pyvista as pv

    return pv.PolyData(np.asarray(points, float)).glyph(
        geom=pv.Sphere(radius=radius), scale=False, orient=False)


def _link_meshes(raw: mne.io.Raw, sci_scores: dict, good_by_base: dict,
                 sci_threshold: float) -> list[tuple]:
    """S-D segments as one tube mesh per quality colour, plus optode spheres.

    Returns ``(mesh, colour)`` pairs.
    """
    import pyvista as pv

    ends, scis, goods = _collect_pairs(raw, sci_scores, good_by_base)
    if not ends:
        return []

    pair_ids = list(ends)
    flat = np.vstack([np.vstack(ends[p]) for p in pair_ids])
    mni  = to_mni(flat, raw.info)

    groups: dict[str, list[np.ndarray]] = {}
    sources: dict[str, np.ndarray] = {}
    detectors: dict[str, np.ndarray] = {}
    for i, pair_id in enumerate(pair_ids):
        s, d = mni[2 * i], mni[2 * i + 1]
        vals = scis.get(pair_id)
        sci  = float(np.mean(vals)) if vals else None
        color = _link_color(sci, goods.get(pair_id), sci_threshold)
        groups.setdefault(color, []).extend([s, d])

        src_id, det_id = pair_id.split("_")
        sources.setdefault(src_id, s)
        detectors.setdefault(det_id, d)

    meshes = []
    for color, points in groups.items():
        n = len(points) // 2
        lines = np.column_stack([np.full(n, 2), np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)])
        segments = pv.PolyData(np.vstack(points), lines=lines.ravel())
        meshes.append((segments.tube(radius=_LINK_RADIUS), color))
    for optodes, color, radius in ((sources, _SRC_COLOR, _SRC_RADIUS),
                                   (detectors, _DET_COLOR, _DET_RADIUS)):
        if optodes:
            meshes.append((_spheres(np.vstack(list(optodes.values())), radius), color))
    return meshes


def _channel_marker_meshes(ch_names, coords_mni, good_mask) -> list[tuple]:
    """Fallback when optode positions are unavailable: spheres at channel midpoints."""
    good_mask = np.asarray(good_mask, bool)
    return [(_spheres(coords_mni[mask], _MARKER_RADIUS), color)
            for mask, color in ((good_mask, _GOOD_COLOR), (~good_mask, _BAD_COLOR))
            if mask.any()]


def _build_3d_scene(brain_meshes, data_meshes):
    """Off-screen plotter holding the brain surface and the ``(mesh, colour)`` data."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=list(_VIEW_SIZE))
    plotter.set_background("white")
    # the light kit's placement, but white: its key light is warm and tints the grey yellow
    for light in plotter.renderer.GetLights():
        light.SetColor(1.0, 1.0, 1.0)
    for mesh in brain_meshes:
        plotter.add_mesh(mesh, color="#e8e8e8", smooth_shading=True,
                         ambient=0.3, diffuse=0.75, specular=0.0)
    for mesh, color in data_meshes:
        plotter.add_mesh(mesh, color=color, smooth_shading=True)
    return plotter


def _render_views(plotter) -> list[np.ndarray]:
    """One RGB screenshot per camera in ``CAMERAS``, each titled with its view label."""
    center = np.asarray(plotter.center, float)
    imgs = []
    for (direction, up), label in zip(CAMERAS, VIEW_LABELS):
        eye = np.asarray(direction, float)
        # any distance works: reset_camera keeps the direction and refits the scene
        plotter.camera_position = [center + 500 * eye / np.linalg.norm(eye), center, up]
        plotter.reset_camera()
        title = plotter.add_text(label, position="upper_edge", font_size=14, color="black")
        imgs.append(np.asarray(plotter.screenshot(return_img=True))[:, :, :3])
        plotter.remove_actor(title)
    return imgs


def quality_brain_views(
    ch_names: list[str],
    coords_head: np.ndarray,
    good_mask: np.ndarray,
    raw: mne.io.Raw | None = None,
    sci_scores: dict[str, float] | None = None,
    *,
    sci_threshold: float,
) -> str:
    """Render 3-view brain figure and return as base64 PNG string.

    A failed render raises, so the report lists it rather than leaving the panel blank.
    """
    good_by_base = {pair_of(n): bool(g) for n, g in zip(ch_names, good_mask)}
    data_meshes: list = []
    if raw is not None:
        try:
            data_meshes = _link_meshes(raw, sci_scores or {}, good_by_base,
                                       sci_threshold)
        except Exception as exc:
            logger.warning("channel links failed: %s", exc)
    if not data_meshes:
        data_meshes = _channel_marker_meshes(
            ch_names,
            to_mni(coords_head, raw.info if raw is not None else None),
            good_mask,
        )

    with RENDER_LOCK:
        plotter = _build_3d_scene(load_brain_meshes(), data_meshes)
        try:
            imgs = _render_views(plotter)
        finally:
            plotter.close()

    trimmed = [_trim_white(a) for a in imgs]
    h = min(a.shape[0] for a in trimmed)
    combined = np.hstack([a[:h, :] for a in trimmed])
    pil = _PILImage.fromarray(combined.astype(np.uint8))
    buf = _io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
