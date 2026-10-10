"""GLM fits that leave ``BAD_`` frames out of the estimate.

The design is built on the full time axis (HRF convolution and drift basis see continuous
time), the AR filter is estimated from lag windows lying wholly in kept time, both sides are
whitened as continuous series, and only then are the ``BAD_`` rows dropped from the fit.
"""

from __future__ import annotations

import gc
from typing import Any

import mne
import numpy as np
import pandas as pd

from nirspipe.exceptions import StageError
from nirspipe.utils.logging import get_logger
from nirspipe.utils.spans import bad_spans, span_kind

logger = get_logger("pipeline.glm_censored")

# fewest kept frames per regressor a fit is allowed on
MIN_FRAMES_PER_REGRESSOR = 3


def kept_frames(raw: mne.io.BaseRaw) -> np.ndarray:
    """Per frame, whether it lies outside every ``BAD_`` span, of either kind."""
    times = raw.times
    keep = np.ones(len(times), dtype=bool)
    for start, stop, _ in bad_spans(raw):
        keep &= ~((times >= start) & (times < stop))
    return keep


def ar_order(noise_model: str, sfreq: float) -> int:
    """The fixed AR order a noise model names: ols 0, ar1 1, arN N, auto 4x the rate."""
    if noise_model == "ols":
        return 0
    if noise_model == "auto":
        return int(np.round(sfreq * 4))
    if noise_model.startswith("ar") and noise_model[2:].isdigit():
        return int(noise_model[2:])
    raise ValueError(f"no fixed AR order in noise model {noise_model!r}")


def _lag_rows(r: np.ndarray, keep: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """``(X, y)`` regressing each sample on a constant and its ``order`` predecessors, forward
    and backward in time, keeping only rows whose sample and lags all lie in kept time."""
    rows_x, rows_y = [], []
    for series, mask in ((r, keep), (r[::-1], keep[::-1])):
        valid = np.lib.stride_tricks.sliding_window_view(mask, order + 1).all(axis=1)
        target = np.arange(order, len(series))[valid]
        lags = np.stack([series[target - k] for k in range(1, order + 1)], axis=1)
        rows_x.append(np.column_stack([np.ones(len(target)), lags]))
        rows_y.append(series[target])
    return np.vstack(rows_x), np.concatenate(rows_y)


def gap_ar_coefficients(resid: np.ndarray, keep: np.ndarray, order: int,
                        search: bool = False) -> np.ndarray:
    r"""AR coefficients of a residual with gaps, from lag windows wholly in kept time.

    Least squares on forward and backward lag rows with a constant, each row dropped when its
    sample or any of its lags is censored. With ``search`` the order is chosen by BIC over
    ``0..order`` on the rows the largest order allows:

    .. math::

        \mathrm{BIC}(p) = n \log(\mathrm{RSS}_p / n) + p \log n

    Parameters
    ----------
    resid : ndarray, shape (n,)
        Residual in the data's own space, full length.
    keep : ndarray of bool, shape (n,)
        False on censored frames.
    order : int
        The AR order, or with ``search`` the largest one considered.
    search : bool
        Choose the order by BIC instead of fitting ``order`` itself.

    Returns
    -------
    ndarray, shape (p,)
        ``rho`` in nilearn's sign convention, ``y[t] - sum(rho[k] * y[t-k-1])`` white.

    Raises
    ------
    StageError
        When the kept stretches hold too few lag windows to fit the coefficients.
    """
    if order == 0:
        return np.zeros(0)
    X, y = _lag_rows(np.asarray(resid, float), keep, order)
    if len(y) <= order + 1:
        raise StageError(
            f"too few kept stretches longer than {order + 1} frames to estimate an AR({order}) "
            f"noise model around the BAD_ spans ({len(y)} lag rows); choose a lower order "
            f"with --noise-model, or keep longer stretches")
    q, r = np.linalg.qr(X)
    qty = q.T @ y
    p = order
    if search:
        n = len(y)
        rss = float(y @ y) - np.cumsum(qty ** 2)          # rss[k]: constant plus k lags
        rss = np.maximum(rss, np.finfo(float).tiny)
        bic = n * np.log(rss / n) + np.arange(order + 1) * np.log(n)
        p = int(np.argmin(bic))
    coef = np.linalg.solve(r[:p + 1, :p + 1], qty[:p + 1])
    return coef[1:]


def fit_channel(y: np.ndarray, design: np.ndarray, keep: np.ndarray, order: int):
    """One channel: OLS on the kept rows, AR from the gap-aware residual, refit whitened.

    The returned nilearn result is fitted on the whitened kept rows, so its theta, scale,
    df and contrasts are the censored fit's; the data-space residual is taken by the caller
    on the full axis.
    """
    from nilearn.glm.regression import OLSModel

    from nirspipe.pipeline.ar_irls import whiten

    # a column, as run_glm hands nilearn each channel, so MSE and theta keep their shapes
    y = np.asarray(y, float).reshape(-1, 1)
    ols = OLSModel(design[keep]).fit(y[keep])
    if order == 0:
        return ols
    rho = gap_ar_coefficients((y - design @ ols.theta).ravel(), keep, order)
    return OLSModel(whiten(design, rho)[keep]).fit(whiten(y, rho)[keep])


def fit_channel_irls(y: np.ndarray, design: np.ndarray, keep: np.ndarray, pmax: int):
    """AR-IRLS under the same three rules: AR order and coefficients by BIC from lag windows
    in kept time, whitening on the full axis, the robust refit on the kept rows only."""
    import statsmodels.api as sm
    from nilearn.glm.regression import OLSModel

    from nirspipe.pipeline.ar_irls import (BETA_TOL, DEFAULT_MAX_ITER, DEFAULT_TUNE,
                                           satterthwaite_df, whiten)

    y = np.asarray(y, float)
    norm = sm.robust.norms.TukeyBiweight(c=DEFAULT_TUNE)
    beta = np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    fit = rho = None
    for _ in range(DEFAULT_MAX_ITER):
        rho = gap_ar_coefficients(y - design @ beta, keep, pmax, search=True)
        wx = whiten(design, rho)[keep]
        fit = sm.RLM(whiten(y, rho)[keep], wx, M=norm).fit()
        moved = (np.linalg.norm(fit.params - beta)
                 / max(np.linalg.norm(beta), np.finfo(float).tiny))
        beta = fit.params
        if moved < BETA_TOL:
            break
    sqrt_w = np.sqrt(fit.weights)
    res = OLSModel(wx * sqrt_w[:, None]).fit((whiten(y, rho)[keep] * sqrt_w).reshape(-1, 1))
    # as in ar_irls.fit_channel: the robust scale, its cov and the effective df reach the t
    res.dispersion = float(fit.scale) ** 2
    res.cov = np.asarray(fit.cov_params()) / res.dispersion
    res.df_residuals = satterthwaite_df(wx, fit.weights)
    return res


def fit_censored(haemo: mne.io.BaseRaw, design_matrix: pd.DataFrame, keep: np.ndarray,
                 noise_model: str) -> Any:
    """Every channel fitted on its kept rows, in mne-nirs' results container."""
    from mne_nirs.statistics._glm_level_first import RegressionResults

    from nirspipe.pipeline.ar_irls import resolve_pmax

    design = design_matrix.values
    data = haemo.get_data()
    sfreq = float(haemo.info["sfreq"])
    results = {}
    if noise_model.startswith("ar_irls"):
        pmax = resolve_pmax(noise_model, sfreq)
        for i, ch in enumerate(haemo.ch_names):
            results[ch] = fit_channel_irls(data[i], design, keep, pmax)
            gc.collect(0)
    else:
        order = ar_order(noise_model, sfreq)
        for i, ch in enumerate(haemo.ch_names):
            results[ch] = fit_channel(data[i], design, keep, order)
    return RegressionResults(haemo.info, results, design_matrix)


def condition_columns(design_matrix: pd.DataFrame, condition: str) -> list[str]:
    """The design columns nilearn builds for one condition: itself, its derivative and
    dispersion, or its FIR delays."""
    names = {condition, f"{condition}_derivative", f"{condition}_dispersion"}
    return [c for c in design_matrix.columns
            if c in names or (c.startswith(f"{condition}_delay_")
                              and c[len(condition) + 7:].isdigit())]


def silent_conditions(design_matrix: pd.DataFrame, conditions: list[str],
                      keep: np.ndarray) -> list[str]:
    """Conditions whose every column is exactly zero on every kept row."""
    kept = design_matrix[keep]
    return [c for c in conditions
            if (cols := condition_columns(design_matrix, c)) and not kept[cols].to_numpy().any()]


def events_left(events: pd.DataFrame, spans) -> dict[str, dict[str, int]]:
    """Per condition, how many events touch no ``BAD_`` span, out of how many.

    ::

      tap at 10 s (5 s), 50 s (5 s); BAD_ over 48-60 s  ->  {"tap": {"left": 1, "total": 2}}

    A zero-length event is left when its onset lies outside every span.
    """
    out: dict[str, dict[str, int]] = {}
    for trial_type, onset, duration in zip(events["trial_type"], events["onset"],
                                           events["duration"]):
        onset, stop = float(onset), float(onset) + float(duration)
        if stop > onset:
            hit = any(onset < e and stop > s for s, e, _ in spans)
        else:
            hit = any(s <= onset < e for s, e, _ in spans)
        counts = out.setdefault(str(trial_type), {"left": 0, "total": 0})
        counts["total"] += 1
        counts["left"] += not hit
    return out


def censor_summary(raw: mne.io.BaseRaw, keep: np.ndarray) -> dict[str, Any]:
    """What left the fit, by frames: the share per kind, the span count, the kept frames."""
    times = raw.times
    spans = bad_spans(raw)
    by_kind = {kind: np.zeros(len(times), dtype=bool) for kind in ("unselected", "corrupted")}
    for start, stop, desc in spans:
        by_kind[span_kind(desc)] |= (times >= start) & (times < stop)
    return {
        "censored_frac": float(1 - keep.mean()),
        "censored_unselected_frac": float(by_kind["unselected"].mean()),
        "censored_corrupted_frac": float(by_kind["corrupted"].mean()),
        "censored_n_spans": len(spans),
        "kept_frames": int(keep.sum()),
    }


def refuse_too_few_frames(keep: np.ndarray, n_regressors: int, name: str) -> None:
    needed = MIN_FRAMES_PER_REGRESSOR * n_regressors
    if keep.sum() < needed:
        raise StageError(
            f"{name}: {int(keep.sum())} frames lie outside its BAD_ spans and the design holds "
            f"{n_regressors} regressors; a fit needs at least {MIN_FRAMES_PER_REGRESSOR} frames "
            f"per regressor ({needed}).")
