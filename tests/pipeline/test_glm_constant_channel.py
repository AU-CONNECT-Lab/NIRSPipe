"""A channel holding one value throughout (prep zeroes a wholly non-finite one) is fitted by
OLS under every AR noise model, and the other channels fit as they would without it."""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.qc.boilerplate.vocabulary import template_slots
from nirspipe.utils.lineage import lineage_of, stamp
from tests._synth import synth_raw

FIT = dict(stim_dur=5.0, hrf_model="spm", drift_model="polynomial",
           high_pass=None, drift_order=1, fir_delays=None)
DEAD = ["S1_D1 hbo", "S1_D1 hbr"]


def _haemo(censored: bool) -> mne.io.Raw:
    raw = synth_raw("01", "tapping", duration=120.0, n_long_pairs=3, short_channels=False,
                    bad_pair=None, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = stamp(mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0),
                  stage="preproc", step="beer_lambert")
    if censored:
        haemo.annotations.append(50.0, 10.0, "BAD_manual")
    return haemo


def _with_dead_pair(haemo: mne.io.Raw) -> mne.io.Raw:
    dead = haemo.copy()
    dead._data[[dead.ch_names.index(ch) for ch in DEAD]] = 0.0
    dead.info["bads"] = list(DEAD)
    return dead


@pytest.mark.parametrize("censored", [False, True], ids=["whole", "censored"])
@pytest.mark.parametrize("noise_model", ["ar1", "auto", "ar_irls", "worsley"])
def test_a_constant_channel_is_fitted_by_ols_and_leaves_the_others_as_they_were(noise_model,
                                                                                 censored):
    haemo = _haemo(censored)
    _, est, _, resid = run_glm_pipeline(_with_dead_pair(haemo), noise_model=noise_model, **FIT)
    _, alone, _, alone_resid = run_glm_pipeline(haemo.copy().drop_channels(DEAD),
                                                noise_model=noise_model, **FIT)

    assert lineage_of(resid).params["constant_fitted_as_ols"] == DEAD
    for ch in DEAD:
        assert not hasattr(est.data[ch].model, "rho")
        assert_allclose(resid.get_data(picks=[ch]), 0.0, atol=0)
    for ch in alone.ch_names:
        assert_allclose(np.ravel(est.data[ch].theta), np.ravel(alone.data[ch].theta),
                        rtol=1e-10, atol=0)
    assert_allclose(resid.get_data(picks=alone.ch_names), alone_resid.get_data(), rtol=1e-10,
                    atol=1e-18)


def test_ols_records_nothing_and_the_methods_name_the_fallback():
    _, _, _, resid = run_glm_pipeline(_with_dead_pair(_haemo(False)), noise_model="ols", **FIT)
    assert "constant_fitted_as_ols" not in lineage_of(resid).params

    text = template_slots("glm", {"noise_model": "auto",
                                  "constant_fitted_as_ols": DEAD})["noise_model"]
    assert text.startswith("an autoregressive noise model whose order is four times")
    assert "; 2 channel(s) holding one value throughout" in text
    assert "fitted by ordinary least squares" in text
    assert "ordinary least squares" not in template_slots(
        "glm", {"noise_model": "auto"})["noise_model"]
