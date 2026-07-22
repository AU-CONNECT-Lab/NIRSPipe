"""Shared color constants and helpers used across QC figure modules."""

import numpy as np

HBO_COLOR      = "#e74c3c"
HBR_COLOR      = "#3498db"
HBO_MEAN_COLOR = "#c0392b"   # darker HbO for bold mean line
HBR_MEAN_COLOR = "#1a5276"   # darker HbR for bold mean line

# Qualitative palette for condition/trigger colors (cycled by index).
CONDITION_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def decimate(arr: np.ndarray, times: np.ndarray, max_pts: int):
    """Uniformly subsample columns of arr (and times) to at most max_pts for display."""
    if len(times) <= max_pts:
        return arr, times
    step = max(1, len(times) // max_pts)
    return arr[:, ::step], times[::step]


def physio_bands(cardiac=None, resp=None):
    """PSD annotation bands as (name, f_lo, f_hi). Single source so every PSD figure agrees.

    Mayer is a fixed visual reference (no metric counterpart). ``cardiac`` / ``resp`` take
    the CLI ``(l_freq, h_freq)`` so the shading matches the bands the metrics integrate over;
    pass None to omit that band entirely rather than guess a default (e.g. the hyper pipeline
    has no respiration band).
    """
    bands = [("Mayer", 0.07, 0.13)]
    if resp is not None:
        bands.append(("Resp", *resp))
    if cardiac is not None:
        bands.append(("Cardiac", *cardiac))
    return bands
