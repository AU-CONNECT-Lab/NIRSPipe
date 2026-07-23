from __future__ import annotations

import numpy as np
import pandas as pd
import mne
from scipy import signal

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.restingstate")


def compute_alff(raw: mne.io.Raw, low_pass: float, high_pass: float) -> pd.DataFrame:
    r"""ALFF and fALFF per channel from the denoised (errts) bandpassed time series.

    ALFF is the mean spectral amplitude in the low band (Zang 2007); fALFF is that
    band's share of the total spectral amplitude (Zou 2008):

    .. math::

        \text{ALFF} = \frac{1}{|B|} \sum_{f \in B} \sqrt{P(f)}, \qquad
        \text{fALFF} = \frac{\sum_{f \in B} \sqrt{P(f)}}{\sum_{f > 0} \sqrt{P(f)}},

    where :math:`B = [\text{high\_pass}, \text{low\_pass}]` is the low-frequency band and
    :math:`P(f)` is the periodogram power (amplitude is :math:`\sqrt{P}`). fALFF is in
    :math:`[0, 1]`; the DC bin is dropped from the denominator.
    """
    data = raw.get_data()  # (n_channels, n_times)
    fs = raw.info["sfreq"]

    alff_vals  = np.zeros(len(raw.ch_names))
    falff_vals = np.zeros(len(raw.ch_names))

    for i, ch_data in enumerate(data):
        if np.nanstd(ch_data) == 0:
            continue

        ch_demeaned = ch_data - np.nanmean(ch_data)

        freqs, power = signal.periodogram(ch_demeaned, fs, scaling="spectrum")
        power_sqrt   = np.sqrt(power)

        # high_pass is the lower freq bound; low_pass is the upper freq bound
        low_idx  = np.argmin(np.abs(freqs - high_pass))
        high_idx = np.argmin(np.abs(freqs - low_pass))

        alff_vals[i] = np.nanmean(power_sqrt[low_idx:high_idx])  # mean band amplitude

        # fALFF: fraction of total spectral amplitude in the low band (Zou 2008), sum/sum ∈ [0,1]
        total_sum = np.nansum(power_sqrt[1:])  # skip DC
        falff_vals[i] = np.nansum(power_sqrt[low_idx:high_idx]) / total_sum if total_sum > 0 else 0.0

    return pd.DataFrame({
        "channel": raw.ch_names,
        "alff":    alff_vals,
        "falff":   falff_vals,
    })


def compute_fc(raw: mne.io.Raw) -> pd.DataFrame:
    r"""Functional connectivity: full channel-by-channel Pearson correlation matrix.

    .. math::

        \rho_{ij} = \operatorname{corr}(x_i, x_j)

    for every channel pair (diagonal 1). Uses nilearn ConnectivityMeasure.
    """
    from nilearn.connectome import ConnectivityMeasure
    fc = ConnectivityMeasure(kind="correlation", standardize=False).fit_transform([raw.get_data().T])[0]
    return pd.DataFrame(fc, index=raw.ch_names, columns=raw.ch_names)


def fisher_z(fc: pd.DataFrame) -> pd.DataFrame:
    r"""Fisher r-to-z of an FC matrix (diagonal set to 0), for group-level stats.

    .. math::

        z = \operatorname{arctanh}(r) = \tfrac{1}{2} \ln \frac{1 + r}{1 - r}

    Variance-stabilises correlations so they can be averaged / tested across subjects;
    r is clipped just below :math:`\pm 1` to keep perfect correlations from diverging.
    """
    r = fc.to_numpy().clip(-0.999999, 0.999999)  # clip to keep perfect corr from → inf
    z = np.arctanh(r)
    np.fill_diagonal(z, 0.0)
    return pd.DataFrame(z, index=fc.index, columns=fc.columns)


def compute_fc_roi(raw: mne.io.Raw, roi_map: dict[str, list[str]]) -> pd.DataFrame:
    r"""ROI-level FC: average each ROI's HbO channels into one signal, then Pearson corr between ROIs.

    .. math::

        \rho_{AB} = \operatorname{corr}(\bar{x}_A, \bar{x}_B), \qquad
        \bar{x}_A = \frac{1}{|A|} \sum_{c \in A} x_c

    Averages signals first (higher SNR) rather than averaging channel correlations; roi_map is
    {ROI label: [channel names]}, names matched full ("S1_D1 hbo") or by S-D base ("S1_D1").
    """
    from nilearn.connectome import ConnectivityMeasure
    hbo = {c for c in raw.ch_names if c.endswith(" hbo")}
    names, signals = [], []
    for roi, chans in roi_map.items():
        picks = [c if c in hbo else f"{c} hbo" for c in chans]
        picks = [c for c in picks if c in hbo]
        if picks:
            names.append(roi)
            signals.append(raw.get_data(picks=picks).mean(axis=0))
    if len(signals) < 2:
        return pd.DataFrame()
    fc = ConnectivityMeasure(kind="correlation", standardize=False).fit_transform([np.vstack(signals).T])[0]
    return pd.DataFrame(fc, index=names, columns=names)
