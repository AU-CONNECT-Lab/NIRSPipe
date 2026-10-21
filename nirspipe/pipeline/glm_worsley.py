"""Worsley's bias-corrected AR noise model for the GLM (``--noise-model worsley``).

The AR coefficients come from the OLS residual's autocovariances after removing the bias the
fit puts into them, as fmristat's ``fmrilm.m`` does (Worsley et al., 2002): the residual is
the data with the design projected out, which leaves its autocovariances short of the
noise's, most of all at the low frequencies the drift terms absorb. Each lag's sum of
products has expectation :math:`\\sum_k M_{jk} a_k` in the noise autocovariances
:math:`a_k`, with

.. math::

    M_{jk} = \\frac{\\operatorname{tr}(R D_j R D_k)}{1 + [j > 0]}

where :math:`R` forms the residual from the design and :math:`D_j` is the symmetric lag-j
shift (:math:`D_0` the identity), so :math:`a = M^{-1} c` is solved for and its AR
coefficients found by Levinson-Durbin. A channel whose corrected autocovariances are not
positive definite at the order asked for is fitted as ``auto`` would fit it.

On a run with ``BAD_`` spans the residual and the lag products cover kept frames only, and
:math:`R` is the kept rows' residual-forming matrix, zero on the censored ones, so the same
expectation holds.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
import pandas as pd

from nirspipe.utils.logging import get_logger

logger = get_logger("pipeline.glm_worsley")

NOISE_PREFIX = "worsley"
# rows per block when summing the shifted bases, so a long run never holds them all at once
_ROW_BLOCK = 2048
# singular values below this share of the largest leave the design's column space
_RANK_TOL = 1e-10


def worsley_order(noise_model: str, sfreq: float) -> int:
    """``worsley`` -> 4x the sampling rate, as ``auto``; ``worsley5`` -> 5."""
    rest = noise_model[len(NOISE_PREFIX):]
    return int(rest) if rest else int(np.round(4 * sfreq))


def _kept_basis(design: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the design's column space on the kept rows, zero on the rest."""
    u, s, _ = np.linalg.svd(design[keep], full_matrices=False)
    basis = np.zeros((len(keep), int((s > _RANK_TOL * s[0]).sum())))
    basis[keep] = u[:, :basis.shape[1]]
    return basis


def bias_matrix(design: np.ndarray, keep: np.ndarray, order: int) -> np.ndarray:
    """fmristat's ``M`` for lags 0..order, from an orthonormal basis rather than the n x n matrices.

    ``tr(R D_j R D_k)`` with ``R = W - Q Q'`` (``W`` the kept-frame mask, ``Q`` the basis)
    expands to ``tr(W D_j W D_k) - 2 tr(Q' D_j W D_k Q) + tr(A_j A_k)``, ``A_j = Q' D_j Q``;
    the first term is nonzero only for ``j = k``, where it counts the kept frames (lag 0) or
    twice the pairs of kept frames ``j`` apart.
    """
    q = _kept_basis(np.asarray(design, float), keep)
    n, k = q.shape
    w = keep.astype(float)
    cross = np.zeros((order + 1, order + 1))
    a = np.zeros((order + 1, k, k))
    for start in range(0, n, _ROW_BLOCK):
        stop = min(n, start + _ROW_BLOCK)
        rows = np.arange(start, stop)
        shifted = np.zeros((order + 1, stop - start, k))
        shifted[0] = q[start:stop]
        for lag in range(1, order + 1):
            ahead, behind = rows + lag, rows - lag
            fwd, back = ahead < n, behind >= 0
            shifted[lag, fwd] += q[ahead[fwd]]
            shifted[lag, back] += q[behind[back]]
        masked = (shifted * w[start:stop, None]).reshape(order + 1, -1)
        cross += masked @ masked.T
        a += q[start:stop].T @ shifted
    tr_aa = np.einsum("lab,mba->lm", a, a)
    pairs = np.array([w.sum()] + [2.0 * (w[:-lag] * w[lag:]).sum()
                                  for lag in range(1, order + 1)])
    return (np.diag(pairs) - 2 * cross + tr_aa) / (1.0 + (np.arange(order + 1) > 0))[:, None]


def lag_products(resid: np.ndarray, keep: np.ndarray, order: int) -> np.ndarray:
    """``sum_t r[t] r[t + j]`` for j = 0..order, over pairs of kept frames only."""
    r = np.where(keep, np.asarray(resid, float), 0.0)
    return np.array([r @ r] + [r[:-lag] @ r[lag:] for lag in range(1, order + 1)])


def levinson(acov: np.ndarray, order: int) -> "np.ndarray | None":
    """Yule-Walker coefficients from autocovariances by Levinson-Durbin, None when not positive definite.

    ``rho`` is in nilearn's sign convention, ``y[t] - sum(rho[k] * y[t-k-1])`` white::

        acov [1, 0.5, 0.25], order 2  ->  [0.5, 0.0]
    """
    sigma = float(acov[0])
    if not sigma > 0:
        return None
    phi = np.zeros(order)
    for p in range(1, order + 1):
        reflection = (acov[p] - phi[:p - 1] @ acov[p - 1:0:-1]) / sigma
        phi[:p - 1] = phi[:p - 1] - reflection * phi[:p - 1][::-1]
        phi[p - 1] = reflection
        sigma *= 1.0 - reflection ** 2
        if not sigma > 0:
            return None
    return phi


def worsley_rho(resid: np.ndarray, keep: np.ndarray, m: np.ndarray) -> "np.ndarray | None":
    """One channel's bias-corrected AR coefficients, None when they cannot be had at this order."""
    corrected = np.linalg.solve(m, lag_products(resid, keep, m.shape[0] - 1))
    return levinson(corrected, m.shape[0] - 1)


def fit_worsley(haemo: mne.io.BaseRaw, design_matrix: pd.DataFrame, noise_model: str,
                keep: "np.ndarray | None" = None) -> "tuple[Any, list[str]]":
    """Every channel fitted under the bias-corrected AR model, in mne-nirs' results container.

    Returns the results and the channels fitted as ``auto`` instead. Without ``keep`` the fit
    is nilearn's AR model, as ``auto``'s; with it, the censored fit of
    :func:`nirspipe.pipeline.glm_censored.fit_channel`, whitened on the full axis and refitted
    on the kept rows.
    """
    from mne_nirs.statistics._glm_level_first import RegressionResults
    from nilearn.glm.first_level.first_level import _yule_walker
    from nilearn.glm.regression import ARModel, OLSModel

    from nirspipe.pipeline.ar_irls import whiten
    from nirspipe.pipeline.glm_censored import gap_ar_coefficients

    design = design_matrix.values
    sfreq = float(haemo.info["sfreq"])
    order = worsley_order(noise_model, sfreq)
    censored = keep is not None and not keep.all()
    mask = keep if keep is not None else np.ones(len(design), bool)
    m = bias_matrix(design, mask, order)
    data = haemo.get_data()
    results, fallback = {}, []
    for i, ch in enumerate(haemo.ch_names):
        y = data[i].reshape(-1, 1)
        theta = np.linalg.lstsq(design[mask], y[mask], rcond=None)[0]
        resid = (y - design @ theta).ravel()
        rho = worsley_rho(resid, mask, m)
        if rho is None:
            fallback.append(ch)
            rho = (gap_ar_coefficients(resid, mask, order) if censored
                   else _yule_walker(resid[None], order)[0])
        if censored:
            results[ch] = OLSModel(whiten(design, rho)[mask]).fit(whiten(y, rho)[mask])
        else:
            results[ch] = ARModel(design, rho).fit(y)
    if fallback:
        logger.warning("worsley: %d channel(s) fitted as auto, their corrected "
                       "autocovariances not positive definite at AR(%d): %s",
                       len(fallback), order, fallback)
    return RegressionResults(haemo.info, results, design_matrix), fallback
