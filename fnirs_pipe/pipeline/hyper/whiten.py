"""Autoregressive prewhitening of the long channels before the wavelet coherence.

Off unless ``--wtc-whiten`` asks for it. The order is fixed in seconds and shared by every
channel of both members, rather than chosen per channel the way the correlation's is, so
the two members of a pair are filtered to the same depth and carry transients of the same
length. It is fitted on the whole aligned record, which every route then windows or crops,
so the real table and both nulls see identically whitened signals.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.io.snirf import long_channel_picks
from fnirs_pipe.pipeline.hyper._helpers import _shared_sfreq
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.whiten")


def autocov(centred: np.ndarray, n_lag: int) -> np.ndarray:
    """Autocovariances of a centred series at lags ``0..n_lag``, each over the full length."""
    n = centred.size
    # one dot product per lag: a full correlation would be quadratic in the record length
    return np.array([centred @ centred] + [centred[:-k] @ centred[k:]
                                           for k in range(1, n_lag + 1)]) / n


def _yule_walker(acov: np.ndarray, order: int) -> "tuple[np.ndarray, float] | None":
    """AR coefficients of ``order`` and the residual variance, from a biased autocovariance.

    None when the Toeplitz system is singular or the fit leaves no positive variance.
    """
    try:
        coef = solve_toeplitz((acov[:order], acov[:order]), acov[1:order + 1])
    except np.linalg.LinAlgError:
        return None
    resid_var = float(acov[0] - coef @ acov[1:order + 1])
    return (coef, resid_var) if resid_var > 0 else None


def ar_whiten_fixed(x: np.ndarray, order: int) -> "np.ndarray | None":
    """Residuals of an order-``order`` Yule-Walker fit, the input's length, or None.

    ::

      a 3900 s trace at 10 Hz, order 100  ->  3900 s of residual, the first 10 s zeroed

    The filter runs from the first sample, so the first ``order`` samples are a startup
    transient; they are set to zero rather than dropped, since dropping them would move every
    later sample off the shared clock.
    None when the row cannot be fitted: non-finite, constant, or a singular system.
    """
    if not np.isfinite(x).all() or x.size <= order:
        return None
    centred = x - x.mean()
    acov = autocov(centred, order)
    if acov[0] <= 0:
        return None
    fit = _yule_walker(acov, order)
    if fit is None:
        return None
    coef, _ = fit
    resid = lfilter(np.r_[1.0, -coef], [1.0], centred)
    resid[:order] = 0.0
    return resid


def whiten_order(raws: dict, order_s: float) -> int:
    """The shared order in samples, ``order_s`` at the members' common rate."""
    return int(round(float(order_s) * _shared_sfreq(raws)))


def whiten_raws(raws: dict, order_s: float, sep_bands=None) -> dict:
    """Copies of ``raws`` whose long HbO and HbR channels hold their AR residuals.

    ::

      {"sub-01": raw, "sub-02": raw}, 10.0  ->  the same two recordings, whitened at AR(100)

    Refuses a record shorter than four times the order, the floor the correlation's fit
    uses, rather than lowering the order for it, since the order is shared. A channel that
    cannot be fitted is marked bad on the copy, which drops it from the coherence the way a
    rejected channel is dropped.
    """
    order = whiten_order(raws, order_s)
    if order < 1:
        return dict(raws)
    out: dict = {}
    for sid, raw in raws.items():
        if raw.n_times < 4 * order:
            raise StageError(
                f"{sid} has {raw.n_times} samples, fewer than four times the AR({order}) that "
                f"--wtc-whiten {order_s:g} asks for. Lower --wtc-whiten or turn it off.")
        copy = raw.copy().load_data()
        failed = []
        for ch_type in ("hbo", "hbr"):
            for idx in long_channel_picks(copy, ch_type, sep_bands=sep_bands):
                resid = ar_whiten_fixed(copy._data[idx], order)
                if resid is None:
                    failed.append(copy.ch_names[idx])
                else:
                    copy._data[idx] = resid
        if failed:
            logger.warning("%s: %d channel(s) could not be whitened at AR(%d) and are left "
                           "out of the coherence: %s", sid, len(failed), order,
                           ", ".join(failed))
            copy.info["bads"] = sorted(set(copy.info["bads"]) | set(failed))
        out[sid] = copy
    return out
