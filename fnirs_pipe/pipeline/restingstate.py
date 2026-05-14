from __future__ import annotations

import numpy as np
import pandas as pd
import mne
from scipy import signal

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.restingstate")


def compute_alff(raw: mne.io.Raw, low_pass: float, high_pass: float) -> pd.DataFrame:
    """Compute ALFF and fALFF per channel.

    Input should be the denoised (errts) time series, already bandpass-filtered.
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

        band_amp  = np.nanmean(power_sqrt[low_idx:high_idx])
        total_amp = np.nanmean(power_sqrt[1:])  # skip DC component

        alff_vals[i]  = band_amp * sd_scale
        falff_vals[i] = band_amp / total_amp if total_amp > 0 else 0.0

    return pd.DataFrame({
        "channel": raw.ch_names,
        "alff":    alff_vals,
        "falff":   falff_vals,
    })


def compute_fc(raw: mne.io.Raw) -> pd.DataFrame:
    # uses nilearn.connectome.ConnectivityMeasure
    from nilearn.connectome import ConnectivityMeasure
    fc = ConnectivityMeasure(kind="correlation").fit_transform([raw.get_data().T])[0]
    return pd.DataFrame(fc, index=raw.ch_names, columns=raw.ch_names)
