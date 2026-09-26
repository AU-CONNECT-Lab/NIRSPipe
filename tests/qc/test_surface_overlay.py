"""The activation colours are computed in the package, and must stay what MNE's ``stc.plot``
would draw from the same numbers. The comparisons import MNE internals, so they skip rather
than fail once MNE moves them; the package itself never imports them."""

import os

import mne
import numpy as np
import pytest

from fnirs_pipe.qc.figures.common._surface_overlay import (
    activation_rgba, diverging_lut, smoothing_matrix, vertex_normals,
)

LIMITS = [(0.0, 0.5, 1.0), (0.5, 0.75, 1.0), (0.2, 0.3, 2.0), (0.0, 0.0, 1.0), (0.0, 1.0, 1.0),
          (0.0, 1.2e-7, 2.4e-7)]


def _mne(module: str, name: str):
    try:
        return getattr(__import__(module, fromlist=[name]), name)
    except (ImportError, AttributeError):
        pytest.skip(f"{module}.{name} is gone from this MNE")


@pytest.fixture(scope="module")
def sphere():
    import pyvista as pv
    mesh = pv.Icosphere(nsub=3)
    return np.asarray(mesh.points, float), mesh.regular_faces.astype(np.int64)


@pytest.fixture(scope="module")
def fsaverage_lh():
    path = mne.utils.get_subjects_dir()
    surf = None if path is None else os.path.join(path, "fsaverage", "surf", "lh.pial")
    if surf is None or not os.path.isfile(surf):
        pytest.skip("fsaverage is not on this machine")
    return mne.read_surface(surf)


# ---- behaviour, on the package's own terms ----

def test_the_table_hides_values_near_zero_and_shows_the_extremes():
    lut = diverging_lut(0.0, 0.5, 1.0)
    assert lut.shape == (256, 4) and lut.dtype == np.uint8
    assert lut[127, 3] < 5 and lut[128, 3] < 5
    assert lut[0, 3] == lut[-1, 3] == 255
    assert lut[0, 2] > lut[0, 0] and lut[-1, 0] > lut[-1, 2]    # blue below, red above


def test_a_degenerate_scale_is_refused():
    with pytest.raises(ValueError):
        diverging_lut(1.0, 1.0, 1.0)


def test_smoothing_averages_so_a_constant_stays_constant(sphere):
    coords, faces = sphere
    src = np.arange(0, len(coords), 4)
    smooth = smoothing_matrix(faces, len(coords), src, steps=5)
    spread = smooth @ np.full(len(src), 3.0)
    reached = np.asarray(smooth.sum(axis=1)).ravel() > 0
    assert np.allclose(spread[reached], 3.0) and reached.all()


# ---- agreement with MNE ----

@pytest.mark.parametrize("fmin, fmid, fmax", LIMITS)
def test_the_colour_table_matches_mne(fmin, fmid, fmax):
    calculate_lut = _mne("mne.viz._brain.colormap", "calculate_lut")
    want = np.round(calculate_lut("RdBu_r", alpha=1.0, fmin=fmin, fmid=fmid, fmax=fmax,
                                  center=0.0, transparent=True) * 255).astype(np.uint8)
    np.testing.assert_array_equal(diverging_lut(fmin, fmid, fmax), want)


def test_the_normals_match_mne(fsaverage_lh):
    complete_surface_info = _mne("mne.surface", "complete_surface_info")
    coords, faces = fsaverage_lh
    want = complete_surface_info(dict(rr=coords, tris=faces), copy=True, verbose=False,
                                 do_neighbor_tri=False)["nn"]
    np.testing.assert_allclose(vertex_normals(coords, faces), want, atol=1e-12)


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_the_smoothing_matches_mne_on_a_small_mesh(sphere, steps):
    hemi_morph = _mne("mne.morph", "_hemi_morph")
    coords, faces = sphere
    src = np.arange(0, len(coords), 7)
    want = hemi_morph(faces, np.arange(len(coords)), src, steps, maps=None, warn=False)
    got = smoothing_matrix(faces, len(coords), src, steps)
    assert abs(got - want).max() < 1e-12


def test_the_smoothing_matches_mne_on_fsaverage(fsaverage_lh):
    hemi_morph = _mne("mne.morph", "_hemi_morph")
    coords, faces = fsaverage_lh
    src = np.arange(10242)                    # the ico-5 source space the GLM projects onto
    with mne.utils.use_log_level(False):
        want = hemi_morph(faces, np.arange(len(coords)), src, 10, maps=None, warn=False)
    got = smoothing_matrix(faces, len(coords), src, 10)
    assert abs(got - want).max() < 1e-12


@pytest.mark.parametrize("fmin, fmid, fmax", LIMITS[:3])
def test_the_composited_colours_match_mne(fmin, fmid, fmax):
    layered = _mne("mne.viz._3d_overlay", "_LayeredMesh")
    rng = np.random.default_rng(0)
    n = 5000
    curv = rng.normal(size=n)
    values = rng.uniform(-1.3 * fmax, 1.3 * fmax, size=n)    # past both ends too

    calculate_lut = _mne("mne.viz._brain.colormap", "calculate_lut")
    ctable = np.round(calculate_lut("RdBu_r", alpha=1.0, fmin=fmin, fmid=fmid, fmax=fmax,
                                    center=0.0, transparent=True) * 255).astype(np.uint8)
    mesh = layered(renderer=None, vertices=np.zeros((n, 3)), triangles=None, normals=None)
    mesh.add_overlay(scalars=(curv > 0).astype(np.int64), colormap="Greys", rng=[-1, 2],
                     opacity=1.0, name="curv")
    mesh.add_overlay(scalars=values, colormap=ctable, rng=[-fmax, fmax], opacity=None,
                     name="data")
    np.testing.assert_allclose(activation_rgba(curv, values, fmin, fmid, fmax),
                               mesh._current_colors, atol=1e-12)
