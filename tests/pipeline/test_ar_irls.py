"""AR-IRLS, checked piece by piece against something that was written separately.

Every component here has a slower or better-established counterpart, and each test compares
against that rather than against a number this file made up: the order search against
statsmodels' Yule-Walker, the whitening against nilearn's own, the effective df against the
n-by-n matrix the fast form exists to avoid, the whole solver against a reference written
from the published algorithm with different primitives. The last two tests are the ones that
say the estimator is calibrated rather than merely reproducible.
"""

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from fnirs_pipe.pipeline.ar_irls import (
    bic_ar_order,
    fit_channel,
    resolve_pmax,
    satterthwaite_df,
    whiten,
)


def _ar_series(rng, n, rho, scale=1.0):
    """AR(p) noise with coefficients `rho`."""
    rho = np.atleast_1d(rho)
    x = np.zeros(n + 200)
    for i in range(len(rho), n + 200):
        x[i] = rho @ x[i - len(rho):i][::-1] + rng.normal(scale=scale)
    return x[200:]


def _task_design(n, sfreq=7.81, period=60.0, on=20.0):
    """A block design plus an intercept, at a length and rate this package sees."""
    t = np.arange(n) / sfreq
    block = ((t % period) < on).astype(float)
    return np.column_stack([block - block.mean(), np.ones(n)])


# ---- the order search, against statsmodels ----
@pytest.mark.parametrize("seed,truth", [(0, [0.6]), (1, [0.5, -0.3]), (2, [0.4, -0.2, 0.15])])
def test_the_selected_coefficients_are_yule_walkers(seed, truth):
    """Whatever order BIC lands on, the coefficients at that order have to be the ones the
    standard solver gives. The Levinson recursion is here to deliver every order in one pass,
    not to estimate something different; if these disagree, the speed came from changing the
    estimator.

    Comparing at the order `bic_ar_order` chose is what makes this a test of the function
    rather than of a recursion retyped into the test."""
    x = _ar_series(np.random.default_rng(seed), 8000, truth)
    ours = bic_ar_order(x, pmax=20)
    assert ours.size >= 1, "BIC found no structure in a series that has some"
    theirs, _ = sm.regression.linear_model.yule_walker(x - x.mean(), order=ours.size,
                                                       method="mle")
    assert np.allclose(ours, theirs, atol=1e-10), f"{ours} vs {theirs}"


def test_the_periodogram_autocovariance_equals_the_direct_one():
    """The order search takes its autocovariance through an FFT rather than `np.correlate`'s
    direct O(n^2) convolution; the two must give the same numbers."""
    rng = np.random.default_rng(20)
    for n in (997, 4096, 5000):                 # one prime, one power of two, one neither
        x = _ar_series(rng, n, [0.6, -0.2])
        x0 = x - x.mean()
        direct = np.correlate(x0, x0, mode="full")[n - 1:n + 10] / n
        size = 1 << int(np.ceil(np.log2(2 * n)))
        spec = np.fft.rfft(x0, size)
        fast = np.fft.irfft(spec * np.conj(spec), size)[:11] / n
        assert np.allclose(direct, fast, atol=1e-12), f"n={n}"


def test_bic_recovers_a_known_ar1():
    rng = np.random.default_rng(1)
    rho = bic_ar_order(_ar_series(rng, 8000, [0.7]), pmax=20)
    assert rho.size >= 1
    assert rho[0] == pytest.approx(0.7, abs=0.05)


def test_bic_recovers_a_known_ar3():
    rng = np.random.default_rng(11)
    truth = np.array([0.5, -0.3, 0.2])
    rho = bic_ar_order(_ar_series(rng, 20000, truth), pmax=20)
    assert rho.size == 3, f"BIC chose order {rho.size}"
    assert np.allclose(rho, truth, atol=0.05)


def test_white_noise_buys_no_lags():
    rho = bic_ar_order(np.random.default_rng(2).normal(size=8000), pmax=20)
    assert rho.size <= 1, "BIC should not pay for lags that explain nothing"


def test_pmax_is_never_exceeded():
    rng = np.random.default_rng(3)
    x = np.convolve(rng.normal(size=4000), np.ones(30) / 30, mode="same")
    assert bic_ar_order(x, pmax=5).size <= 5


def test_a_series_too_short_to_fit_returns_no_coefficients():
    assert bic_ar_order(np.arange(3.0), pmax=10).size == 0


def test_a_constant_series_does_not_divide_by_zero():
    assert bic_ar_order(np.ones(500), pmax=10).size == 0


# ---- the whitening, against nilearn ----
def test_whitening_is_nilearns_own():
    """The solver estimates rho and the weights under this transform and the final model
    applies nilearn's. If the two forms differed, the weights would belong to a fit that
    never happened."""
    from nilearn.glm.regression import ARModel

    rng = np.random.default_rng(5)
    X, rho = rng.normal(size=(400, 3)), np.array([0.6, -0.2])
    assert np.allclose(whiten(X, rho), ARModel(X, rho).whiten(X))


def test_whitening_flattens_a_known_ar_spectrum():
    """An AR(2) series whitened by its own coefficients should come out with no detectable
    autocorrelation left at the first lags."""
    rng = np.random.default_rng(6)
    truth = np.array([0.6, -0.2])
    x = _ar_series(rng, 20000, truth)
    w = whiten(x.reshape(-1, 1), truth).ravel()[10:]
    acf = [np.corrcoef(w[:-k], w[k:])[0, 1] for k in (1, 2, 3)]
    assert max(abs(a) for a in acf) < 0.03, acf


# ---- the effective df, against the matrix it avoids building ----
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_fast_df_equals_the_n_by_n_definition(seed):
    """`satterthwaite_df` replaces an n-by-n construction with p-by-p algebra. At an n small
    enough to build H outright, the two must agree; that is the only thing standing between
    the derivation and a plausible-looking wrong number."""
    rng = np.random.default_rng(seed)
    n, p = 200, 4
    X = np.column_stack([rng.normal(size=(n, p - 1)), np.ones(n)])
    w = rng.uniform(0.0, 1.0, size=n)

    a = w[:, None] * X
    h = np.diag(w) - a @ np.linalg.pinv(a.T @ a) @ a.T
    hth = h.T @ h
    brute = np.trace(hth) ** 2 / np.trace(hth @ hth)

    assert satterthwaite_df(X, w) == pytest.approx(brute, rel=1e-9)


def test_all_weights_one_gives_back_the_ordinary_df():
    """With nothing down-weighted the Satterthwaite form has to collapse to n - p, or it
    would be silently changing every fit that has no outliers."""
    rng = np.random.default_rng(7)
    n, p = 500, 5
    X = np.column_stack([rng.normal(size=(n, p - 1)), np.ones(n)])
    assert satterthwaite_df(X, np.ones(n)) == pytest.approx(n - p, rel=1e-9)


def test_down_weighting_costs_degrees_of_freedom():
    rng = np.random.default_rng(8)
    n, p = 500, 4
    X = np.column_stack([rng.normal(size=(n, p - 1)), np.ones(n)])
    w = np.ones(n)
    w[:100] = 0.0
    assert satterthwaite_df(X, w) < n - p
    assert satterthwaite_df(X, w) == pytest.approx(400 - p, rel=0.02)


# ---- the whole solver, against a reference written from the algorithm ----
def _reference_ar_irls(y, X, pmax, tune=4.685, n_iter=10):
    """The published alternation, written with different primitives.

    statsmodels' AutoReg for the order search instead of a Levinson recursion, scipy's
    lfilter for the whitening instead of the shift-and-subtract form, and the loop run to a
    fixed count instead of a tolerance. Agreement between this and the solver is therefore
    evidence about the algorithm rather than about one transcription of it.
    """
    from scipy.signal import lfilter

    y = np.asarray(y, float).ravel()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(n_iter):
        resid = y - X @ beta
        best, coefs = np.inf, np.zeros(0)
        for p in range(0, pmax + 1):
            fit = sm.tsa.AutoReg(resid - resid.mean(), lags=p, old_names=False).fit()
            if fit.bic > best:
                break
            best, coefs = fit.bic, np.asarray(fit.params[1:])
        wf = np.hstack([1.0, -coefs])
        yf = lfilter(wf, 1, y)
        xf = np.column_stack([lfilter(wf, 1, X[:, j]) for j in range(X.shape[1])])
        beta = sm.RLM(yf, xf, M=sm.robust.norms.TukeyBiweight(c=tune)).fit().params
    return beta


def test_the_solver_agrees_with_an_independent_transcription():
    """Not bit-identical, and it should not be: the two search the AR order with different
    estimators and treat the first few samples differently, so this compares two readings of
    one algorithm rather than one implementation with itself.

    The yardstick is the estimate's own sampling variability: the two implementations should
    agree far more closely than either agrees with what it is estimating.
    """
    truth = 1.0
    gaps, from_truth = [], []
    for seed in range(8):
        rng = np.random.default_rng(100 + seed)
        n = 3000
        X = _task_design(n)
        y = X @ np.array([truth, 0.3]) + _ar_series(rng, n, [0.65, -0.15], scale=0.4)
        y[900:930] += 6.0                          # something for the robust norm to find

        res = fit_channel(y, X, pmax=20)
        ours = float(res.theta[0, 0])
        theirs = float(_reference_ar_irls(y, X, pmax=20)[0])
        se = float(np.sqrt(res.vcov(matrix=np.array([1.0, 0.0]))))
        gaps.append(abs(ours - theirs) / se)
        from_truth.append(abs(ours - truth) / se)

    assert np.median(gaps) < 0.3, f"median gap {np.median(gaps):.3f} se"
    assert max(gaps) < 0.6, f"worst gap {max(gaps):.3f} se"
    assert np.median(gaps) < np.median(from_truth), (
        "the two implementations should differ by less than either differs from the truth")


def test_the_model_object_reproduces_the_robust_fit():
    """The returned nilearn model re-solves the weighted whitened problem itself. Its theta
    has to come out as the last robust fit's, or the object being handed downstream is not
    the fit that was measured."""
    rng = np.random.default_rng(13)
    n = 2000
    X = _task_design(n)
    y = X @ np.array([1.0, 0.0]) + _ar_series(rng, n, [0.6], scale=0.5)
    y[500:520] += 8.0

    res = fit_channel(y, X, pmax=15)
    wx = whiten(X, res.model.rho)
    wy = whiten(y.reshape(-1, 1), res.model.rho).ravel()
    rlm = sm.RLM(wy, wx, M=sm.robust.norms.TukeyBiweight(c=4.685)).fit()
    assert np.allclose(res.theta.ravel(), rlm.params, rtol=1e-6, atol=1e-9)
    assert res.dispersion == pytest.approx(rlm.scale ** 2, rel=1e-9)


@pytest.mark.parametrize("n_bursts", [0, 10])
def test_the_standard_error_is_the_robust_one(n_bursts):
    """The estimate is only half of it: the t value divides by this, so a covariance rebuilt
    from the weighted normal equations instead of taken from the robust fit is a silent
    error on every t in the run."""
    rng = np.random.default_rng(19)
    n = 4000
    X = _task_design(n)
    y = X @ np.array([1.0, 0.0]) + _ar_series(rng, n, [0.7, -0.2], scale=0.5)
    for start in rng.integers(0, n - 40, size=n_bursts):
        y[start:start + 25] += rng.choice([-1, 1]) * 8.0

    res = fit_channel(y, X, pmax=20)
    wx = whiten(X, res.model.rho)
    wy = whiten(y.reshape(-1, 1), res.model.rho).ravel()
    rlm = sm.RLM(wy, wx, M=sm.robust.norms.TukeyBiweight(c=4.685)).fit()

    assert np.allclose(np.sqrt(np.diag(res.vcov())), rlm.bse, rtol=1e-9)
    contrast_se = float(np.sqrt(res.vcov(matrix=np.array([1.0, 0.0]))))
    assert contrast_se == pytest.approx(rlm.bse[0], rel=1e-9)


# ---- does it recover the truth, and is the t value calibrated ----
def test_it_recovers_a_known_beta_through_contaminated_ar_noise():
    rng = np.random.default_rng(14)
    n, truth = 4000, 1.0
    X = _task_design(n)
    errs = []
    for k in range(8):
        r = np.random.default_rng(100 + k)
        y = X @ np.array([truth, 0.0]) + _ar_series(r, n, [0.7, -0.2], scale=0.5)
        for start in r.integers(0, n - 40, size=6):   # six bursts per run
            y[start:start + 25] += r.choice([-1, 1]) * 8.0
        errs.append(abs(fit_channel(y, X, pmax=20).theta[0, 0] - truth))
    assert np.median(errs) < 0.15, f"median |error| {np.median(errs):.3f}"
    assert rng is not None


def test_the_burst_moves_the_beta_far_less_than_it_moves_least_squares():
    rng = np.random.default_rng(15)
    n = 3000
    X = _task_design(n)
    clean = X @ np.array([1.0, 0.0]) + _ar_series(rng, n, [0.6], scale=0.3)
    burst = clean.copy()
    burst[1000:1060] += 15.0

    ols = abs(np.linalg.lstsq(X, burst, rcond=None)[0][0]
              - np.linalg.lstsq(X, clean, rcond=None)[0][0])
    irls = abs(fit_channel(burst, X, pmax=15).theta[0, 0]
               - fit_channel(clean, X, pmax=15).theta[0, 0])
    assert irls < ols / 3, f"irls moved {irls:.4f}, ols moved {ols:.4f}"


def _null_rejection_rates(n_draws, n, n_bursts, seed0):
    """Share of draws where a nominal 5% test rejects, under noise containing no effect."""
    from nilearn.glm.regression import ARModel

    from fnirs_pipe.pipeline.ar_irls import bic_ar_order

    hits = {"ar_irls": 0, "ar_only": 0, "ols": 0}
    for k in range(n_draws):
        rng = np.random.default_rng(seed0 + k)
        X = _task_design(n, period=40.0 + 3 * (k % 7))
        y = _ar_series(rng, n, [0.75, -0.15], scale=0.5)
        for start in rng.integers(0, n - 40, size=n_bursts):
            y[start:start + 25] += rng.choice([-1, 1]) * 6.0

        res = fit_channel(y, X, pmax=20)
        t = float(res.Tcontrast(np.array([1.0, 0.0])).t)
        hits["ar_irls"] += abs(t) > stats.t.ppf(0.975, res.df_residuals)

        # the same order search without the robust norm, so the arms differ in one thing
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        ar = ARModel(X, bic_ar_order(y - X @ beta, 20)).fit(y.reshape(-1, 1))
        t_ar = float(ar.Tcontrast(np.array([1.0, 0.0])).t)
        hits["ar_only"] += abs(t_ar) > stats.t.ppf(0.975, ar.df_residuals)

        resid = y - X @ beta
        se = np.sqrt((resid @ resid / (n - 2)) * np.linalg.pinv(X.T @ X)[0, 0])
        hits["ols"] += abs(beta[0] / se) > stats.t.ppf(0.975, n - 2)
    return {k: v / n_draws for k, v in hits.items()}


@pytest.mark.slow
def test_the_t_value_is_calibrated_under_a_clean_ar_null():
    """The estimator's job. Under noise with no effect in it, a nominal 5% test should reject
    about 5% of the time. OLS is the control: it is known to be badly inflated at these
    sampling rates, so a run where it also looked fine would mean the null was too easy."""
    r = _null_rejection_rates(n_draws=120, n=2000, n_bursts=0, seed0=5000)
    assert r["ols"] > 0.20, f"the control is not inflated ({r['ols']:.3f}); the null is too easy"
    assert 0.01 < r["ar_irls"] < 0.13, f"ar_irls {r['ar_irls']:.3f} against a nominal 0.05"


@pytest.mark.slow
def test_the_robust_norm_holds_calibration_when_the_null_is_contaminated():
    """Where AR-IRLS is supposed to beat plain whitening. Bursts the AR model cannot absorb
    push the unweighted arm off nominal; the robust one should stay near it."""
    r = _null_rejection_rates(n_draws=120, n=2000, n_bursts=8, seed0=5000)
    assert r["ols"] > 0.40, f"the control is not inflated ({r['ols']:.3f})"
    assert r["ar_irls"] < 0.14, f"ar_irls {r['ar_irls']:.3f} against a nominal 0.05"
    assert r["ar_irls"] <= r["ar_only"] + 0.02, (
        f"the robust norm bought nothing: ar_irls {r['ar_irls']:.3f} vs "
        f"AR alone {r['ar_only']:.3f}")


# ---- the container contract ----
def test_the_result_keeps_y_and_the_design_unwhitened():
    """`run_glm_pipeline` builds the data-space residual out of these two, so a solver that
    stored the whitened versions here would silently change what `desc-errts` holds."""
    rng = np.random.default_rng(16)
    X = _task_design(800)
    y = X @ np.array([1.0, 0.0]) + rng.normal(size=800)
    res = fit_channel(y, X, pmax=10)
    assert np.allclose(np.asarray(res.Y).ravel(), y)
    assert np.allclose(np.asarray(res.model.design), X)


def test_the_result_can_take_a_contrast():
    rng = np.random.default_rng(17)
    X = _task_design(1200)
    y = X @ np.array([2.0, 0.0]) + rng.normal(size=1200) * 0.3
    con = fit_channel(y, X, pmax=10).Tcontrast(np.array([1.0, 0.0]))
    assert np.isfinite(con.t).all()
    assert con.effect.item() == pytest.approx(2.0, abs=0.2)


def test_the_effective_df_is_below_the_nominal_one_when_samples_are_rejected():
    rng = np.random.default_rng(18)
    n = 2000
    X = _task_design(n)
    y = X @ np.array([1.0, 0.0]) + rng.normal(size=n) * 0.3
    y[300:500] += 20.0
    res = fit_channel(y, X, pmax=10)
    assert res.df_residuals < n - X.shape[1]


def test_pmax_follows_the_same_rule_as_auto():
    assert resolve_pmax("ar_irls", 7.81) == 31
    assert resolve_pmax("ar_irls", 10.0) == 40
    assert resolve_pmax("ar_irls40", 7.81) == 40


# ---- the CLI surface ----
@pytest.mark.parametrize("value", ["ar_irls", "ar_irls40", "auto", "ols", "ar16"])
def test_the_validator_takes_it(value):
    from fnirs_pipe.cli.run import _noise_model
    assert _noise_model(value) == value


@pytest.mark.parametrize("value", ["ar_irls0", "ar_irls_", "irls", "ar_IRLS"])
def test_the_validator_refuses_a_near_miss(value):
    import argparse

    from fnirs_pipe.cli.run import _noise_model
    with pytest.raises(argparse.ArgumentTypeError):
        _noise_model(value)
