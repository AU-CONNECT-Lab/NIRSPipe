"""`--noise-model` must actually reach nilearn and actually prewhiten.

Nothing else checks this. The model name is threaded from the CLI through `PostConfig` into
`run_glm_pipeline`, and if it were dropped anywhere along the way every run would silently
fit OLS while the record claimed AR(1), which changes the standard errors of every beta
without changing a single fitted value.

Durbin-Watson is the measurement that shows it: whitening is supposed to remove the serial
correlation the residuals carry, driving DW from wherever it started toward 2.

The trap this encodes, and the reason the attribute matters more than the number: DW on
`.residuals` does **not** improve between `ols` and `ar1`, so a test written against that
attribute would have shown a plausible-looking number that proved nothing.
`.whitened_residuals` is where prewhitening is visible.

`.residuals` is not the unwhitened residual either, which is the second trap and the one
that matters for a written file. nilearn builds it as `Y - whitened_design @ theta`, mixing the
two spaces, so under an AR model it is neither residual; the last test here pins the stage
file against `Y - design @ theta` instead. Under `ols` the two agree, because whitening is
the identity there.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from nirspipe.pipeline.glm import run_glm_pipeline
from nirspipe.utils.lineage import stamp
from tests._synth import synth_raw

FIT = dict(stim_dur=5.0, hrf_model="spm", drift_model="polynomial",
           high_pass=None, drift_order=1, fir_delays=None)


def _durbin_watson(residuals) -> float:
    x = np.asarray(residuals).ravel()
    return float(np.sum(np.diff(x) ** 2) / np.sum(x ** 2))


@pytest.fixture(scope="module")
def haemo():
    # the synthetic cardiac and respiration oscillations are what make the residuals serially
    # correlated, so there is something for prewhitening to remove
    raw = synth_raw("01", "rest", duration=200.0, n_long_pairs=4,
                    bad_pair=None, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return stamp(mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0),
                 stage="preproc", step="beer_lambert")


@pytest.fixture(scope="module")
def fits(haemo):
    return {model: run_glm_pipeline(haemo, noise_model=model, **FIT)[1]
            for model in ("ols", "ar1", "ar2")}


def _mean_dw(glm_est, attribute: str) -> float:
    return float(np.mean([_durbin_watson(getattr(glm_est.data[ch], attribute))
                          for ch in glm_est.data]))


# ---- ols leaves the residuals alone ----

def test_ols_whitening_is_the_identity(fits):
    # nothing to whiten with, so the two arrays must be the same array's worth of numbers.
    # If this ever diverges, some default noise model is being substituted underneath
    est = fits["ols"]
    for ch in est.data:
        assert_allclose(est.data[ch].whitened_residuals,
                        est.data[ch].residuals, rtol=1e-12,
                        err_msg=f"{ch} was whitened under ols")


# ---- ar1 / ar2 do not ----

@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_an_autoregressive_model_changes_the_residuals(fits, model):
    est = fits[model]
    differs = [ch for ch in est.data
               if not np.allclose(est.data[ch].whitened_residuals, est.data[ch].residuals)]
    assert len(differs) == len(est.data), f"{model} left some channels unwhitened"


@pytest.mark.parametrize("model", ["ar1", "ar2"])
def test_whitening_moves_durbin_watson_toward_two(fits, model):
    est = fits[model]
    before = _mean_dw(est, "residuals")
    after = _mean_dw(est, "whitened_residuals")
    assert abs(after - 2.0) < abs(before - 2.0), (
        f"{model}: DW went from {before:.3f} to {after:.3f}, no closer to 2")
    assert 1.5 < after < 2.5, f"{model}: whitened DW {after:.3f} is not near 2"


def test_the_unwhitened_residuals_are_the_trap(fits):
    # they are serially correlated under every model, including the two that whiten, because
    # whitening does not write back to `.residuals`. A test reading this attribute would
    # have concluded that ar1 changes nothing
    values = {model: _mean_dw(est, "residuals") for model, est in fits.items()}
    for model, dw in values.items():
        assert dw < 1.5, f"{model}: unwhitened DW {dw:.3f} was expected to stay far from 2"
    # not bit-identical: GLS moves the betas slightly, so the residuals move with them.
    # The point is the scale of the difference, four orders of magnitude below the change
    # whitening produces on the other attribute
    assert_allclose(values["ar1"], values["ols"], rtol=1e-3)


def test_the_written_residual_is_in_the_datas_own_space(haemo):
    """`desc-errts` must be `Y - design @ theta`, not nilearn's mixed-space `.residuals`.

    Under ar1 nilearn subtracts the whitened design's fit, which is roughly the fit
    differenced, so the confounds stay in the file. The exact comparison is what makes this
    a guard: how large the discrepancy grows depends on the autocorrelation and on how much
    the design explains, and this fixture is mild on both counts, so an assertion on the
    size would pass here and prove nothing.
    """
    for model in ("ols", "ar1"):
        _, est, dm, resid = run_glm_pipeline(haemo, noise_model=model, **FIT)
        design = np.asarray(dm)
        # the residual is written back by position, so a reordering would scramble channels
        assert list(est.ch_names) == list(resid.ch_names)
        for ch in est.ch_names:
            res = est.data[ch]
            expected = (np.asarray(res.Y) - design @ np.asarray(res.theta)).ravel()
            assert_allclose(resid.get_data(picks=[ch]).ravel(), expected, rtol=1e-9,
                            err_msg=f"{model} {ch}: written residual is not y - X@theta")


def test_ols_is_the_negative_control_on_the_whitened_residuals(fits):
    # the same measurement that lands near 2 for ar1/ar2 must not land there for ols, or the
    # assertion above would be satisfied by something other than prewhitening
    assert _mean_dw(fits["ols"], "whitened_residuals") < 1.5


# ---- the model name is what selects the behaviour ----

def test_the_three_models_give_three_different_fits(fits):
    whitened = {model: _mean_dw(est, "whitened_residuals") for model, est in fits.items()}
    assert len({round(v, 6) for v in whitened.values()}) == 3, (
        f"two noise models produced the same whitening: {whitened}")
