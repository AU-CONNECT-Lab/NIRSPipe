from __future__ import annotations
from typing import Literal

import mne.io

from fnirs_pipe.utils.lineage import stamp
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.denoise")


def bandpass_filter(
    haemo: mne.io.Raw,
    l_freq: float | None = None,
    h_freq: float | None = None,
) -> mne.io.Raw:
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter
    haemo.filter(l_freq=l_freq, h_freq=h_freq, method="fir", fir_window="hamming")
    return stamp(haemo, stage="filtered", step="bandpass", source=haemo,
                 l_freq=l_freq, h_freq=h_freq)


def resample(haemo: mne.io.Raw, sfreq: float) -> mne.io.Raw:
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
    haemo.resample(sfreq)
    return stamp(haemo, stage="resampled", step="resample", source=haemo, sfreq=sfreq)
