from __future__ import annotations

import numpy as np
import pandas as pd
import mne
from scipy import signal

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.restingstate")


def compute_alff(raw: mne.io.Raw, low_pass: float, high_pass: float) -> pd.DataFrame:
    r"""ALFF/fALFF per channel plus cross-channel mALFF/zALFF standardization.

    Expects a broadband (detrended, non-lowpassed) residual: fALFF's denominator spans the
    full spectrum, so a bandpassed input collapses fALFF to ~1. ALFF only reads the band, so
    one broadband signal serves both (Zang 2007; Zou 2008):

    .. math::

        \text{ALFF} = \frac{1}{|B|} \sum_{f \in B} \sqrt{P(f)}, \qquad
        \text{fALFF} = \frac{\sum_{f \in B} \sqrt{P(f)}}{\sum_{f > 0} \sqrt{P(f)}},

        \text{mALFF}_c = \frac{\text{ALFF}_c}{\overline{\text{ALFF}}}, \qquad
        \text{zALFF}_c = \frac{\text{ALFF}_c - \overline{\text{ALFF}}}{\operatorname{sd}(\text{ALFF})},

    where :math:`B = [\text{high\_pass}, \text{low\_pass}]` is the low-frequency band and
    :math:`P(f)` is the periodogram power (amplitude is :math:`\sqrt{P}`). fALFF is in
    :math:`[0, 1]`; the DC bin is dropped from the denominator. mALFF/zALFF standardize the
    scale-dependent ALFF across channels so it is comparable at the group level.

    Notes
    -----
    Rest mode produces two residuals and this function must receive the broadband one.
    Connectivity (FC) wants a bandpassed residual (~0.01-0.08 Hz) so cardiac, respiration and
    Mayer waves are dropped before correlating. ALFF/fALFF want the opposite: fALFF is the
    band's share of the total spectral amplitude, so its denominator needs the full spectrum;
    a bandpassed input removes the out-of-band power and collapses fALFF to ~1 on every
    channel. ALFF is unaffected either way because it only averages the in-band amplitude, so
    a single broadband signal serves both metrics. The pipeline therefore runs the confound
    regression a second time without the low-pass and feeds that residual here (FC keeps the
    bandpassed one). The linear detrend the references apply before the FFT is here supplied
    by the GLM drift regressors, matching the "detrend, no bandpass" recipe of Zang 2007 /
    Zou 2008.
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

    alff_mean = np.nanmean(alff_vals)
    alff_std  = np.nanstd(alff_vals)
    malff_vals = alff_vals / alff_mean if alff_mean != 0 else np.zeros_like(alff_vals)
    zalff_vals = (alff_vals - alff_mean) / alff_std if alff_std != 0 else np.zeros_like(alff_vals)

    return pd.DataFrame({
        "channel": raw.ch_names,
        "alff":    alff_vals,
        "falff":   falff_vals,
        "malff":   malff_vals,
        "zalff":   zalff_vals,
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
