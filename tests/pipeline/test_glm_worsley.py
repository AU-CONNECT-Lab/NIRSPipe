"""`--noise-model worsley`: the bias matrix, the corrected estimate, and the fall back to auto.

The matrix is held to fmristat's dense formula (``fmrilm.m``, n x n matrices) rather than to
the blocked basis algebra the package uses; the estimate is held to a planted MA(2) noise,
whose autocovariances vanish past lag 2, so the corrected ones must come back unbiased.
"""

import argparse

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import solve_toeplitz

from nirspipe.pipeline import glm_worsley
from nirspipe.pipeline.glm import fit_glm, run_glm_pipeline
from nirspipe.pipeline.glm_censored import fit_censored
from nirspipe.pipeline.glm_worsley import (bias_matrix, fit_worsley, lag_products, levinson,
                                           worsley_order)
from nirspipe.qc.boilerplate.vocabulary import template_slots
from nirspipe.utils.lineage import stamp
from tests._synth import synth_raw

FIT = dict(stim_dur=5.0, hrf_model="spm", drift_model="polynomial",
           high_pass=None, drift_order=1, fir_delays=None)


def _design(n):
    t = np.arange(n)
    box = ((t // 15) % 2).astype(float)
    return np.column_stack([np.ones(n), t / n, np.cos(2 * np.pi * t / 45), box])


def _dense_m(design, keep, order):
    """fmrilm.m's M with R the kept rows' residual-forming matrix, zero elsewhere."""
    n = len(design)
    r = np.zeros((n, n))
    xk = design[keep]
    r[np.ix_(keep, keep)] = np.eye(keep.sum()) - xk @ np.linalg.pinv(xk)
    m = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            di = (np.eye(n, k=i) + np.eye(n, k=-i)) / (1 + (i == 0))
            dj = (np.eye(n, k=j) + np.eye(n, k=-j)) / (1 + (j == 0))
            m[i, j] = np.trace(r @ di @ r @ dj) / (1 + (i > 0))
    return m


GAP = np.r_[np.ones(40, bool), np.zeros(15, bool), np.ones(35, bool)]


@pytest.mark.parametrize("keep", [np.ones(90, bool), GAP], ids=["whole", "censored"])
def test_the_bias_matrix_is_fmristats(keep):
    design = _design(90)
    assert_allclose(bias_matrix(design, keep, 4), _dense_m(design, keep, 4), atol=1e-9)


def test_lag_products_count_kept_pairs_only():
    assert_allclose(lag_products(np.array([1.0, 2, 3, 4]), np.ones(4, bool), 2), [30, 20, 11])
    keep = np.array([True, False, True, True])
    assert_allclose(lag_products(np.array([1.0, 2, 3, 4]), keep, 2), [26, 12, 3])


def test_levinson_solves_yule_walker_and_refuses_what_is_not_positive_definite():
    assert_allclose(levinson(np.array([1.0, 0.5, 0.25]), 2), [0.5, 0.0], atol=1e-15)
    acov = np.array([2.0, 1.1, 0.4, -0.1])
    assert_allclose(levinson(acov, 3), solve_toeplitz(acov[:3], acov[1:]), rtol=1e-12)
    assert levinson(np.array([1.0, 0.9, 0.1]), 2) is None
    assert levinson(np.array([0.0, 0.0]), 1) is None


@pytest.mark.parametrize("keep", [np.ones(240, bool),
                                  np.r_[np.ones(100, bool), np.zeros(30, bool),
                                        np.ones(110, bool)]], ids=["whole", "censored"])
def test_the_corrected_autocovariances_are_unbiased_where_the_raw_ones_are_not(keep):
    """MA(2) noise, b = (0.6, 0.3): autocovariances (1.45, 0.78, 0.3) and zero beyond lag 2."""
    rng = np.random.default_rng(0)
    n, order, b1, b2 = len(keep), 2, 0.6, 0.3
    truth = np.array([1 + b1 ** 2 + b2 ** 2, b1 + b1 * b2, b2])
    t = np.arange(n)
    design = np.column_stack([np.ones(n)] + [np.cos(np.pi * k * (t + 0.5) / n)
                                             for k in range(1, 9)])
    m = bias_matrix(design, keep, order)
    pairs = np.array([keep.sum()] + [(keep[:-k] & keep[k:]).sum() for k in (1, 2)])
    corrected, raw = [], []
    for _ in range(2000):
        e = rng.standard_normal(n + 2)
        y = e[2:] + b1 * e[1:-1] + b2 * e[:-2]
        theta = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
        c = lag_products(y - design @ theta, keep, order)
        corrected.append(np.linalg.solve(m, c))
        raw.append(c / pairs)
    corrected, raw = np.array(corrected), np.array(raw)
    se = corrected.std(axis=0) / np.sqrt(len(corrected))
    assert np.all(np.abs(corrected.mean(axis=0) - truth) < 4 * se)
    assert np.all(np.abs(raw.mean(axis=0) - truth) > 10 * se)


def test_the_order_follows_auto_unless_named():
    assert worsley_order("worsley", 7.81) == 31
    assert worsley_order("worsley5", 7.81) == 5


@pytest.mark.parametrize("value", ["worsley", "worsley12"])
def test_the_cli_takes_it(value):
    from nirspipe.cli.run import _noise_model
    assert _noise_model(value) == value


@pytest.mark.parametrize("value", ["worsley0", "Worsley", "worsley_", "worsley1.5"])
def test_the_cli_refuses_a_near_miss(value):
    from nirspipe.cli.run import _noise_model
    with pytest.raises(argparse.ArgumentTypeError):
        _noise_model(value)


@pytest.fixture(scope="module")
def haemo():
    raw = synth_raw("01", "tapping", duration=200.0, n_long_pairs=3,
                    bad_pair=None, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return stamp(mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0),
                 stage="preproc", step="beer_lambert")


@pytest.fixture(scope="module")
def design(haemo):
    return run_glm_pipeline(haemo, noise_model="ols", **FIT)[2]


def _same_fit(left, right, channels):
    for ch in channels:
        assert_allclose(np.ravel(left.data[ch].theta), np.ravel(right.data[ch].theta),
                        rtol=1e-8, atol=1e-14)
        assert_allclose(left.data[ch].dispersion, right.data[ch].dispersion, rtol=1e-8)


def test_a_channel_with_no_positive_definite_correction_is_fitted_as_auto(haemo, design,
                                                                          monkeypatch):
    monkeypatch.setattr(glm_worsley, "worsley_rho", lambda *args: None)
    est, fallback = fit_worsley(haemo, design, "worsley")
    assert fallback == haemo.ch_names
    _same_fit(est, fit_glm(haemo, design, "auto"), haemo.ch_names)


def test_the_censored_fall_back_is_the_censored_auto_fit(haemo, design, monkeypatch):
    monkeypatch.setattr(glm_worsley, "worsley_rho", lambda *args: None)
    keep = np.ones(haemo.n_times, bool)
    keep[300:500] = False
    est, _ = fit_worsley(haemo, design, "worsley", keep=keep)
    _same_fit(est, fit_censored(haemo, design, keep, "auto"), haemo.ch_names)


def test_the_corrected_fit_differs_from_auto(haemo, design):
    est, fallback = fit_worsley(haemo, design, "worsley")
    auto = fit_glm(haemo, design, "auto")
    assert fallback == []
    assert not np.allclose(est.data[haemo.ch_names[0]].theta, auto.data[haemo.ch_names[0]].theta,
                           rtol=1e-6, atol=0)


def test_the_run_records_the_model_and_says_it_in_the_methods(haemo, tmp_path):
    import json

    nirs = tmp_path / "sub-01" / "nirs"
    nirs.mkdir(parents=True)
    run_glm_pipeline(haemo, noise_model="worsley", output_dir=str(nirs),
                     source_path=str(nirs / "sub-01_task-tapping_desc-preproc_nirs.snirf"), **FIT)
    side = json.loads(next(nirs.glob("*desc-glm*.json")).read_text(encoding="utf-8"))
    assert side["parameters"]["noise_model"] == "worsley"
    assert side["parameters"]["worsley_fitted_as_auto"] == []
    text = template_slots("glm", {"noise_model": "worsley",
                                  "worsley_fitted_as_auto": ["S1_D1 hbo"]})["noise_model"]
    assert "bias correction of Worsley et al. (2002)" in text
    assert "four times the sampling rate" in text and "(1 channel(s)" in text
