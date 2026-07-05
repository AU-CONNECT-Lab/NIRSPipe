from __future__ import annotations

import numpy as np
import pandas as pd
import mne
from scipy import signal

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.restingstate")


def compute_alff(raw: mne.io.Raw, low_pass: float, high_pass: float) -> pd.DataFrame:
    """Compute ALFF and fALFF per channel. Input is the denoised (errts) bandpassed time series.

    ALFF = mean band amplitude × SD (Zang 2007); fALFF = Σ band / Σ total amplitude (Zou 2008).
    """
    data = raw.get_data()  # (n_channels, n_times)
    fs = raw.info["sfreq"]

    alff_vals  = np.zeros(len(raw.ch_names))
    falff_vals = np.zeros(len(raw.ch_names))

    for i, ch_data in enumerate(data):
        if np.nanstd(ch_data) == 0:
            continue

        sd_scale = np.nanstd(ch_data)
        ch_norm  = (ch_data - np.nanmean(ch_data)) / sd_scale

        freqs, power = signal.periodogram(ch_norm, fs, scaling="spectrum")
        power_sqrt   = np.sqrt(power)

        # high_pass is the lower freq bound; low_pass is the upper freq bound
        low_idx  = np.argmin(np.abs(freqs - high_pass))
        high_idx = np.argmin(np.abs(freqs - low_pass))

        alff_vals[i] = np.nanmean(power_sqrt[low_idx:high_idx]) * sd_scale  # mean band amplitude × SD

        # fALFF: fraction of total spectral amplitude in the low band (Zou 2008), sum/sum ∈ [0,1]
        total_sum = np.nansum(power_sqrt[1:])  # skip DC
        falff_vals[i] = np.nansum(power_sqrt[low_idx:high_idx]) / total_sum if total_sum > 0 else 0.0

    return pd.DataFrame({
        "channel": raw.ch_names,
        "alff":    alff_vals,
        "falff":   falff_vals,
    })


def compute_fc(raw: mne.io.Raw) -> pd.DataFrame:
    # uses nilearn.connectome.ConnectivityMeasure
    from nilearn.connectome import ConnectivityMeasure
    fc = ConnectivityMeasure(kind="correlation", standardize=False).fit_transform([raw.get_data().T])[0]
    return pd.DataFrame(fc, index=raw.ch_names, columns=raw.ch_names)


def compute_fc_roi(raw: mne.io.Raw, roi_map: dict[str, list[str]]) -> pd.DataFrame:
    """ROI-level FC: average each ROI's HbO channels into one signal, then Pearson corr between ROIs.

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
