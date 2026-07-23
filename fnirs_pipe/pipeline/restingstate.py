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

        # high_pass is the lower bound, low_pass the upper; inclusive mask keeps both edge bins
        band_mask = (freqs >= high_pass) & (freqs <= low_pass)
        band_amp  = power_sqrt[band_mask]

        alff_vals[i] = np.nanmean(band_amp) if band_amp.size else 0.0  # mean band amplitude

        # fALFF: fraction of total spectral amplitude in the low band (Zou 2008), sum/sum ∈ [0,1]
        total_sum = np.nansum(power_sqrt[1:])  # skip DC
        falff_vals[i] = np.nansum(band_amp) / total_sum if total_sum > 0 else 0.0

    # mALFF/zALFF standardize within each chromophore: HbO and HbR sit on different amplitude
    # scales, so a pooled mean/std would distort both; each chromophore is normalized on its own.
    malff_vals = np.zeros_like(alff_vals)
    zalff_vals = np.zeros_like(alff_vals)
    for suffix in (" hbo", " hbr"):
        idx = np.array([c.endswith(suffix) for c in raw.ch_names])
        if not idx.any():
            continue
        grp_mean = np.nanmean(alff_vals[idx])
        grp_std  = np.nanstd(alff_vals[idx])
        if grp_mean != 0:
            malff_vals[idx] = alff_vals[idx] / grp_mean
        if grp_std != 0:
            zalff_vals[idx] = (alff_vals[idx] - grp_mean) / grp_std

    return pd.DataFrame({
        "channel": raw.ch_names,
        "alff":    alff_vals,
        "falff":   falff_vals,
        "malff":   malff_vals,
        "zalff":   zalff_vals,
    })


def compute_fc(raw: mne.io.Raw, chromophore: str) -> pd.DataFrame:
    r"""Functional connectivity for one chromophore: channel-by-channel Pearson matrix.

    .. math::

        \rho_{ij} = \operatorname{corr}(x_i, x_j)

    over the channels of a single chromophore (``chromophore`` is "hbo" or "hbr"); HbO and HbR
    anti-correlate, so a mixed matrix has no clean meaning and the two are kept separate. Diagonal
    is 1. Empty frame if the chromophore has < 2 channels.
    """
    from nilearn.connectome import ConnectivityMeasure
    picks = [c for c in raw.ch_names if c.endswith(f" {chromophore}")]
    if len(picks) < 2:
        return pd.DataFrame()
    data = raw.get_data(picks=picks)
    fc = ConnectivityMeasure(kind="correlation", standardize=False).fit_transform([data.T])[0]
    return pd.DataFrame(fc, index=picks, columns=picks)


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


def compute_fc_roi(raw: mne.io.Raw, roi_map: dict[str, list[str]], chromophore: str = "hbo") -> pd.DataFrame:
    r"""ROI-level FC for one chromophore: average each ROI's channels, then Pearson corr between ROIs.

    .. math::

        \rho_{AB} = \operatorname{corr}(\bar{x}_A, \bar{x}_B), \qquad
        \bar{x}_A = \frac{1}{|A|} \sum_{c \in A} x_c

    Averages signals first (higher SNR) rather than averaging channel correlations; roi_map is
    {ROI label: [channel names]}, matched by S-D base ("S1_D1"), with any chromophore suffix on
    the map entry replaced by the requested one so the same map serves hbo and hbr.
    """
    from nilearn.connectome import ConnectivityMeasure
    suffix = f" {chromophore}"
    chan_set = {c for c in raw.ch_names if c.endswith(suffix)}
    names, signals = [], []
    for roi, chans in roi_map.items():
        picks = []
        for c in chans:
            base = c[:-4] if (c.endswith(" hbo") or c.endswith(" hbr")) else c
            name = f"{base}{suffix}"
            if name in chan_set:
                picks.append(name)
        if picks:
            names.append(roi)
            signals.append(raw.get_data(picks=picks).mean(axis=0))
    if len(signals) < 2:
        return pd.DataFrame()
    fc = ConnectivityMeasure(kind="correlation", standardize=False).fit_transform([np.vstack(signals).T])[0]
    return pd.DataFrame(fc, index=names, columns=names)
