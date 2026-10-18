"""What the noise model left in a GLM's residual: its spectrum against white noise, and its lag-1 autocorrelation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import welch

# ---- constants ----
# Welch segment length; long enough to resolve the slowest block designs
WHITENING_PSD_SEGMENT_S = 200.0
# where a task design's power sits, and where leftover autocorrelation inflates t most
WHITENING_LOWBAND = (0.01, 0.1)


def residual_spectrum(x: np.ndarray, sfreq: float,
                      segment_s: float = WHITENING_PSD_SEGMENT_S) -> tuple[np.ndarray, np.ndarray]:
    """Power of a residual relative to white noise of the same variance, per frequency.

    The series is standardised, its one-sided Welch density taken, and the density
    multiplied by ``sfreq / 2``, which is what white noise of variance 1 has at every
    frequency. So the ratio is 1 where the residual is white, above 1 where correlation
    is left over, below 1 where the whitening overshot.

    Parameters
    ----------
    x : ndarray, shape (n,)
        One channel's residual.
    sfreq : float
        Sampling rate in Hz.
    segment_s : float
        Welch segment length in seconds, capped at the series length.

    Returns
    -------
    freqs : ndarray
        Frequencies in Hz, the zero and Nyquist bins left out: a one-sided density holds
        half the power there, so white noise would read 0.5 on them.
    ratio : ndarray
        Power relative to white, same length as ``freqs``.

    Examples
    --------
    White noise reads about 1 at every frequency; an AR(1) residual with coefficient 0.9
    reads about 19 near zero frequency, ``(1 + 0.9) / (1 - 0.9)``.
    """
    x = np.asarray(x, float)
    x = (x - x.mean()) / x.std()
    nperseg = int(min(len(x), round(segment_s * sfreq)))
    freqs, density = welch(x, fs=sfreq, nperseg=nperseg, detrend=False)
    stop = len(freqs) - 1 if nperseg % 2 == 0 else len(freqs)
    return freqs[1:stop], density[1:stop] * sfreq / 2.0


def lag1_autocorrelation(x: np.ndarray) -> float:
    """Sample autocorrelation at lag 1, about the series mean.

    ``[1, 2, 3, 4]`` -> deviations ``[-1.5, -0.5, 0.5, 1.5]``, lagged products 0.75 - 0.25
    + 0.75 = 1.25 over the sum of squares 5, so 0.25.
    """
    d = np.asarray(x, float) - np.mean(x)
    return float(np.sum(d[1:] * d[:-1]) / np.sum(d * d))


def band_mean(freqs: np.ndarray, ratio: np.ndarray,
              band: tuple[float, float] = WHITENING_LOWBAND) -> float:
    """Mean of ``ratio`` over ``band[0] <= f < band[1]``, NaN when no bin falls inside."""
    inside = (freqs >= band[0]) & (freqs < band[1])
    return float(ratio[inside].mean()) if inside.any() else float("nan")


def whitening_tables(glm_est: Any, residual: np.ndarray, keep: np.ndarray, sfreq: float,
                     chroma: list[str], long_channels: set[str], bads: set[str],
                     self_fitted: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per channel, the residual and whitened residual side by side; then their mean spectra.

    The whitened residual is the fit's own, the series its standard errors assume white:
    for AR-IRLS it carries the robust weights, for a censored fit only the kept frames.
    ``residual`` is the data-space residual, cut to the same kept frames for the
    comparison. The spectra average the long, good channels of each chromophore; a channel
    fitted against a copy of itself has a zero residual and is left out of everything.

    Returns
    -------
    channels : DataFrame
        One row per channel: ``ch_name``, ``Chroma``, ``long``, ``bad``, ``n_samples``,
        ``acf1_residual``, ``acf1_whitened``, ``lowband_residual``, ``lowband_whitened``.
    spectrum : DataFrame
        ``freq`` and, per chromophore present, ``<chroma>_whitened`` and
        ``<chroma>_residual``.
    """
    rows, spectra, freqs = [], {}, None
    for i, ch in enumerate(glm_est.ch_names):
        white = np.asarray(glm_est.data[ch].whitened_residuals, float).ravel()
        resid = np.asarray(residual[i], float)[keep]
        row = {"ch_name": ch, "Chroma": chroma[i], "long": ch in long_channels,
               "bad": ch in bads, "n_samples": len(white)}
        usable = ch not in self_fitted and np.std(white) > 0 and np.std(resid) > 0
        if usable:
            freqs, w_ratio = residual_spectrum(white, sfreq)
            _, r_ratio = residual_spectrum(resid, sfreq)
            row.update(acf1_residual=lag1_autocorrelation(resid),
                       acf1_whitened=lag1_autocorrelation(white),
                       lowband_residual=band_mean(freqs, r_ratio),
                       lowband_whitened=band_mean(freqs, w_ratio))
            if row["long"] and not row["bad"]:
                spectra.setdefault(chroma[i], []).append((w_ratio, r_ratio))
        rows.append(row)
    channels = pd.DataFrame(rows).reindex(columns=[
        "ch_name", "Chroma", "long", "bad", "n_samples", "acf1_residual", "acf1_whitened",
        "lowband_residual", "lowband_whitened"])
    spectrum = pd.DataFrame({"freq": freqs if freqs is not None else []})
    for name, pairs in spectra.items():
        spectrum[f"{name}_whitened"] = np.mean([w for w, _ in pairs], axis=0)
        spectrum[f"{name}_residual"] = np.mean([r for _, r in pairs], axis=0)
    return channels, spectrum
