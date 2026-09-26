"""The GLM activation render: betas drawn in the diverging scale on the fsaverage surface,
off-screen and from worker threads, with no Qt window involved."""

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fnirs_pipe.qc.figures.subject import glm_figures

VIEWS = [{"azimuth": 180, "elevation": 90}, {"azimuth": 0, "elevation": 0},
         {"azimuth": 90, "elevation": 90}]


@pytest.fixture(scope="module")
def subjects_dir():
    path = mne.utils.get_subjects_dir()
    if path is None or not os.path.isfile(os.path.join(path, "fsaverage", "surf", "lh.pial")):
        pytest.skip("fsaverage is not on this machine")
    return str(path)


def _stc(lh_value: float, rh_value: float) -> mne.SourceEstimate:
    n = 10242
    data = np.r_[np.full(n, lh_value), np.full(n, rh_value)][:, None]
    return mne.SourceEstimate(data, vertices=[np.arange(n), np.arange(n)], tmin=0, tstep=1,
                              subject="fsaverage")


def _count(img: np.ndarray, rgb) -> int:
    return int(np.all(np.abs(img.astype(int) - rgb) < 45, axis=2).sum())


def test_a_positive_hemisphere_is_red_and_a_negative_one_blue(subjects_dir):
    clim = dict(kind="value", pos_lims=(0.0, 0.5, 1.0))
    lateral, dorsal, frontal = glm_figures._render_activation(
        _stc(1.0, -1.0), clim, subjects_dir, (400, 350), VIEWS)

    assert lateral.shape == (350, 400, 3)
    red, blue = (103, 0, 31), (5, 48, 97)   # the two ends of RdBu_r
    assert _count(lateral, red) > 500 and _count(lateral, blue) == 0
    assert _count(dorsal, red) > 500 and _count(dorsal, blue) > 500


def test_values_under_the_lower_limit_show_the_bare_cortex(subjects_dir):
    clim = dict(kind="value", pos_lims=(0.5, 0.75, 1.0))
    lateral, _, _ = glm_figures._render_activation(
        _stc(0.1, 0.1), clim, subjects_dir, (400, 350), VIEWS)
    brain = lateral[np.any(lateral < 250, axis=2)]
    # transparent overlay: only the grey curvature is left, so no pixel carries a hue
    assert np.ptp(brain.astype(int), axis=1).max() < 12


def _haemo():
    names = [f"S{s}_D{d} {c}" for s, d in ((1, 1), (2, 2)) for c in ("hbo", "hbr")]
    info = mne.create_info(names, 10.0, ch_types=["hbo", "hbr", "hbo", "hbr"])
    raw = mne.io.RawArray(np.zeros((4, 100)), info, verbose=False)
    pos = {"S1": [-0.05, 0.02, 0.10], "D1": [-0.02, 0.02, 0.11],
           "S2": [0.05, 0.02, 0.10], "D2": [0.02, 0.02, 0.11]}
    fid = dict(nasion=[0.0, 0.085, -0.035], lpa=[-0.081, -0.029, -0.041],
               rpa=[0.084, -0.029, -0.041])
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head", **fid),
                    verbose="error")
    return raw


def test_conditions_render_from_parallel_threads(subjects_dir):
    raw = _haemo()
    df = pd.DataFrame({"ch_name": ["S1_D1 hbo", "S2_D2 hbo"], "Coef.": [2e-7, -2e-7]})
    clim = glm_figures._shared_clim({"x": df}, raw)

    def one(_):
        return glm_figures._save_glm_brain(raw, df, clim, size=(300, 260), title="x")

    with ThreadPoolExecutor(max_workers=2) as pool:
        figures = list(pool.map(one, range(2)))
    assert all(figures)
    img = np.array(Image.open(io.BytesIO(base64.b64decode(figures[0]))).convert("RGB"))
    assert img.shape[1] > img.shape[0]


def test_a_failed_condition_render_is_listed_on_the_run_page(mini_bids, tmp_path, monkeypatch):
    from fnirs_pipe.cli.run import main

    def broken(*a, **k):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(glm_figures, "_render_activation", broken)
    out = tmp_path / "out"
    main([str(mini_bids), str(out), "participant", "--participant-label", "01",
          "--task-label", "tapping", "--skip-bids-validation", "--dpf", "6",
          "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.1", "--resp-h-freq", "0.4", "--mode", "glm",
          "--hrf-model", "glover", "--drift-model", "cosine", "--drift-high-pass", "0.01",
          "--stim-dur", "5", "--noise-model", "ols"])
    pages = "".join(p.read_text(encoding="utf-8") for p in (out / "sub-01").glob("*_report.html"))
    assert "GLM activation panel (tapping): renderer unavailable" in pages
