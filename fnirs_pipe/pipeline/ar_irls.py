"""AR whitening with iteratively reweighted least squares.

An AR noise model divides out the residual's autocorrelation but still fits every time point
with equal weight, so a few seconds of residual motion move the beta as hard as a clean
minute does. AR-IRLS alternates the two: estimate the AR filter from the current residual,
whiten both sides, refit with a robust norm that down-weights the outlying samples, repeat.

Three quantities come out of it and all three reach the t value:

* ``theta``, the robust whitened estimate;
* the robust scale, which replaces the residual mean square as the dispersion;
* an effective degrees of freedom, because a fit that gave several hundred samples a weight
  near zero did not spend the sample count a residual df of ``n - p`` claims it did.

The solver returns an ordinary :mod:`nilearn` regression result, so contrasts, the tidy frame
and the data-space residual all work exactly as they do under ``arN``.
"""

from __future__ import annotations

import numpy as np
from nilearn.glm.regression import ARModel

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.ar_irls")

# Tukey's bisquare tuning constant for 95% efficiency at the Gaussian
DEFAULT_TUNE = 4.685
DEFAULT_MAX_ITER = 10
# relative change in beta below which the alternation has converged
BETA_TOL = 1e-2


# ---- AR order selection ----
def bic_ar_order(resid: np.ndarray, pmax: int) -> np.ndarray:
    r"""AR coefficients of ``resid``, order chosen by BIC over ``0..pmax``.

    Levinson-Durbin gives every order's coefficients and prediction-error variance from one
    pass over the autocovariance, so the whole order search costs one recursion rather than
    ``pmax`` separate fits.

    .. math::

        \mathrm{BIC}(p) = n \log \sigma^2_p + p \log n

    where :math:`\sigma^2_p` is the order-:math:`p` prediction error variance. The
    autocovariance is the biased one (divided by :math:`n`), which makes the coefficients at
    a fixed order the Yule-Walker maximum-likelihood ones.

    Parameters
    ----------
    resid : ndarray, shape (n,)
        Residual series to model.
    pmax : int
        Largest order considered.

    Returns
    -------
    ndarray, shape (p,)
        Coefficients of the selected order, empty when BIC picks 0.
    """
    x = np.asarray(resid, float).ravel()
    x = x - x.mean()
    n = x.size
    pmax = int(min(pmax, n // 2 - 1))
    if pmax < 1:
        return np.zeros(0)

    acov = np.correlate(x, x, mode="full")[n - 1:n + pmax] / n
    if acov[0] <= 0:
        return np.zeros(0)

    # Levinson-Durbin, carrying every order's solution as it climbs
    phi = np.zeros((pmax + 1, pmax + 1))
    sigma = np.full(pmax + 1, np.inf)
    sigma[0] = acov[0]
    reached = pmax
    for p in range(1, pmax + 1):
        k = (acov[p] - phi[1:p, p - 1] @ acov[p - 1:0:-1]) / sigma[p - 1]
        phi[p, p] = k
        phi[1:p, p] = phi[1:p, p - 1] - k * phi[p - 1:0:-1, p - 1]
        sigma[p] = sigma[p - 1] * (1 - k ** 2)
        if sigma[p] <= 0:                      # numerically singular, stop climbing
            reached = p - 1
            break

    orders = np.arange(reached + 1)
    bic = n * np.log(np.maximum(sigma[:reached + 1], np.finfo(float).tiny)) + orders * np.log(n)
    best = int(np.argmin(bic))
    return phi[1:best + 1, best] if best else np.zeros(0)


# ---- effective degrees of freedom ----
def satterthwaite_df(design: np.ndarray, weights: np.ndarray) -> float:
    r"""Satterthwaite effective residual df of a weighted fit.

    With :math:`H = \mathrm{diag}(w) - AMA^\top`, :math:`A = \mathrm{diag}(w)X` and
    :math:`M = (A^\top A)^{+}`,

    .. math::

        \mathrm{df} = \frac{\operatorname{tr}(H^\top H)^2}{\operatorname{tr}((H^\top H)^2)}

    Building :math:`H` costs :math:`n \times n`, which at fNIRS record lengths is tens of
    gigabytes, so every trace below is taken through the :math:`p \times p` matrices and the
    per-row leverages instead. :math:`H` is symmetric, so :math:`H^\top H = H^2` and the two
    traces are of :math:`H^2` and :math:`H^4`; expanding :math:`(D - P)^k` and collecting by
    cyclic equivalence leaves only :math:`\sum_i w_i^k q_i`, :math:`\operatorname{tr}((MB)^2)`
    and :math:`\operatorname{tr}(P^k) = p`, where :math:`q_i` is row :math:`i`'s leverage in
    :math:`A` and :math:`B = A^\top \mathrm{diag}(w) A`.

    Parameters
    ----------
    design : ndarray, shape (n, p)
        The whitened design the robust fit ran on.
    weights : ndarray, shape (n,)
        Robust weights, in ``[0, 1]``.

    Returns
    -------
    float
        Effective residual degrees of freedom.
    """
    w = np.asarray(weights, float).ravel()
    a = w[:, None] * np.asarray(design, float)
    gram = a.T @ a
    minv = np.linalg.pinv(gram)
    rank = float(np.linalg.matrix_rank(gram))

    lev = np.einsum("ij,jk,ik->i", a, minv, a)        # q_i, row leverages in A
    b = a.T @ (w[:, None] * a)                        # B = A' diag(w) A
    tr_mb2 = float(np.trace((minv @ b) @ (minv @ b)))

    s = [float(np.sum(w ** k * lev)) for k in (1, 2, 3)]
    tr_h2 = float(np.sum(w ** 2)) - 2 * s[0] + rank
    tr_h4 = (float(np.sum(w ** 4)) - 4 * s[2] + 4 * s[1] + 2 * tr_mb2 - 4 * s[0] + rank)
    if tr_h4 <= 0:
        return float(w.sum() - rank)
    return tr_h2 ** 2 / tr_h4


# ---- the alternation ----
class _ARIRLSModel(ARModel):
    """``ARModel`` whose whitening also applies the robust weights.

    Folding the weights into ``whiten`` is what keeps ``.Y`` and ``.model.design`` in the
    data's own space: nilearn stores those unwhitened and whitens on the way into the fit,
    so ``desc-errts`` keeps carrying the data-space residual rather than the whitened
    robust-weighted one.
    """

    def __init__(self, design, rho, weights):
        self._sqrt_w = np.sqrt(np.asarray(weights, float)).reshape(-1, 1)
        super().__init__(design, rho)

    def whiten(self, X):
        white = super().whiten(X)
        w = self._sqrt_w if white.ndim > 1 else self._sqrt_w[:, 0]
        return white * w


def whiten(X: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """AR-filter along the first axis, in nilearn's own shift-and-subtract form."""
    out = np.array(X, float, copy=True)
    for i, r in enumerate(rho):
        out[i + 1:] -= r * X[:-(i + 1)]
    return out


def fit_channel(y: np.ndarray, design: np.ndarray, pmax: int,
                tune: float = DEFAULT_TUNE, max_iter: int = DEFAULT_MAX_ITER):
    """Fit one channel by AR-IRLS and return nilearn's regression result.

    Each pass estimates the AR filter from the residual *in the data's own space*, whitens
    both sides with it, and refits with a bisquare robust norm. The loop stops when beta
    moves less than :data:`BETA_TOL` in relative norm.

    Parameters
    ----------
    y : ndarray, shape (n,) or (n, 1)
        One channel's time course.
    design : ndarray, shape (n, p)
        Design matrix, unwhitened.
    pmax : int
        Largest AR order BIC may choose.
    tune : float
        Bisquare tuning constant; smaller is less sensitive to outliers and less efficient
        when there are none.
    max_iter : int
        Cap on the alternation.

    Returns
    -------
    nilearn RegressionResults
        Whose ``theta`` is the AR-IRLS estimate, whose ``Y`` and ``model.design`` are
        unwhitened, and whose ``dispersion`` and ``df_residuals`` are the robust scale and
        the effective df rather than the residual mean square and ``n - p``.
    """
    import statsmodels.api as sm

    y2 = np.asarray(y, float).reshape(-1, 1)
    norm = sm.robust.norms.TukeyBiweight(c=tune)
    beta = np.linalg.lstsq(design, y2, rcond=None)[0]
    rho = np.zeros(0)
    fit = None

    for _ in range(max_iter):
        resid = (y2 - design @ beta).ravel()
        rho = bic_ar_order(resid, pmax)
        wx = whiten(design, rho)
        fit = sm.RLM(whiten(y2, rho).ravel(), wx, M=norm).fit()
        new = fit.params.reshape(-1, 1)
        moved = np.linalg.norm(new - beta) / max(np.linalg.norm(beta), np.finfo(float).tiny)
        beta = new
        if moved < BETA_TOL:
            break

    res = _ARIRLSModel(design, rho, fit.weights).fit(y2)
    # both are plain attributes read at contrast time, so assigning here is what makes the
    # t value the robust one rather than the weighted least-squares one
    res.dispersion = float(fit.scale) ** 2
    res.df_residuals = satterthwaite_df(wx, fit.weights)
    return res


def resolve_pmax(spec: str, sfreq: float) -> int:
    """Largest AR order for an ``ar_irls`` spec: ``ar_irls`` follows ``auto``'s 4x rule.

    resolve_pmax("ar_irls", 7.81) -> 31;  resolve_pmax("ar_irls40", 7.81) -> 40
    """
    suffix = spec[len("ar_irls"):]
    return int(suffix) if suffix else int(round(sfreq * 4))
