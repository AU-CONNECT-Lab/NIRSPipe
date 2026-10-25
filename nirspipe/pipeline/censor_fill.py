"""Corrupted ``BAD_`` spans filled before the continuous steps (bandpass, resample, regression).

Only corrupted spans are filled; ``BAD_unselected`` is real data and stays as measured. The
filled samples remain under their spans, so every estimate still leaves them out. Frames at
either end that measured no light are held and marked here too, before any OD conversion.
"""

from __future__ import annotations

import mne
import numpy as np
from scipy.interpolate import CubicSpline

from nirspipe.exceptions import StageError
from nirspipe.utils import is_optical_density
from nirspipe.utils.lineage import stamp
from nirspipe.utils.logging import get_logger
from nirspipe.utils.spans import EDGE, add_bad_spans, bad_spans, frames_under, span_kind

logger = get_logger("pipeline.censor_fill")

FILL_METHODS = ("linear", "spline", "lomb")
DEFAULT_CENSOR_FILL = "linear"
# above any analysis band, so the bandpass rather than the fit sets the upper edge
LOMB_FMAX_HZ = 0.25
LOMB_OVERSAMPLE = 4
_LOMB_CHUNK = 64


def corrupted_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """Per sample, whether it lies under a corrupted ``BAD_`` span.

    ::

      10 Hz, BAD_gvtd 1.0-1.3 s, BAD_unselected 0-0.5 s  ->  True at samples 10, 11, 12 only
    """
    return frames_under(raw, [s for s in bad_spans(raw) if span_kind(s[2]) == "corrupted"])


def _lomb_reconstruction(x: np.ndarray, keep: np.ndarray, sfreq: float) -> np.ndarray:
    """Every sample rebuilt from sinusoids least-squares fitted to the kept samples alone.

    One fit per frequency (the Lomb-Scargle form, each with its own time offset), on a grid
    from ``1 / (LOMB_OVERSAMPLE * T)`` up to ``LOMB_FMAX_HZ`` for a recording of length T,
    then rescaled so the kept samples' rebuilt SD matches their measured SD.
    """
    n = x.shape[1]
    t = np.arange(n) / sfreq
    span = n / sfreq
    w_all = (2 * np.pi * np.arange(1, int(LOMB_FMAX_HZ * LOMB_OVERSAMPLE * span) + 1)
             / (LOMB_OVERSAMPLE * span))
    to = t[keep]
    y = x[:, keep]
    mu = y.mean(axis=1, keepdims=True)
    y = y - mu
    rec = np.zeros_like(x, dtype=float)
    # chunked over frequency: one (kept samples x frequencies) block per chunk, not the whole grid
    for i in range(0, len(w_all), _LOMB_CHUNK):
        w = w_all[i:i + _LOMB_CHUNK]
        tau = np.arctan2(np.sin(2 * w * to[:, None]).sum(0),
                         np.cos(2 * w * to[:, None]).sum(0)) / (2 * w)
        c, s = np.cos(w * (to[:, None] - tau)), np.sin(w * (to[:, None] - tau))
        a, b = (y @ c) / (c ** 2).sum(0), (y @ s) / (s ** 2).sum(0)
        phase = w * (t[:, None] - tau)
        rec += a @ np.cos(phase).T + b @ np.sin(phase).T
    rec_sd = rec[:, keep].std(axis=1, keepdims=True)
    scale = np.divide(y.std(axis=1, keepdims=True), rec_sd, out=np.ones_like(rec_sd),
                      where=rec_sd > 0)
    return rec * scale + mu


def fill_values(data: np.ndarray, gap: np.ndarray, method: str, sfreq: float) -> np.ndarray:
    """Replace the samples under ``gap`` from the samples outside it, channel by channel.

    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
        The signal, measured everywhere including under the gap.
    gap : ndarray of bool, shape (n_times,)
        True where a sample is to be replaced. At least one sample must be False.
    method : {"linear", "spline", "lomb"}
        ``"linear"`` joins the kept samples either side of each gap with a straight line.
        ``"spline"`` runs one cubic spline through every kept sample and reads it inside the
        gaps. ``"lomb"`` rebuilds the gaps from a Lomb-Scargle fit to the kept samples
        (sinusoids up to ``LOMB_FMAX_HZ`` at ``LOMB_OVERSAMPLE`` times oversampling, scaled to
        the kept samples' SD). Under ``"linear"`` and ``"spline"`` a gap at either end of the
        recording takes the nearest kept value; nothing is extrapolated.
    sfreq : float
        Sampling rate in Hz, which sets the Lomb-Scargle frequency grid.

    Returns
    -------
    ndarray, shape (n_channels, n_times)
        A copy of ``data`` with only the gap samples changed; every kept sample is returned
        bit for bit.
    """
    if method not in FILL_METHODS:
        raise ValueError(f"unknown fill method {method!r}; expected one of {FILL_METHODS}")
    keep = ~gap
    if not keep.any():
        raise ValueError("no sample lies outside the gap, so there is nothing to fill from")
    idx = np.arange(data.shape[1])
    if method == "lomb":
        filled = _lomb_reconstruction(data, keep, sfreq)
    elif method == "spline" and keep.sum() >= 2:
        filled = CubicSpline(idx[keep], data[:, keep], axis=1, extrapolate=False)(idx)
        first, last = np.flatnonzero(keep)[[0, -1]]
        filled[:, :first], filled[:, last + 1:] = data[:, [first]], data[:, [last]]
    else:
        # np.interp holds the end values past the last kept sample, which is the edge rule
        filled = np.stack([np.interp(idx, idx[keep], row[keep]) for row in data])
    out = data.copy()
    out[:, gap] = filled[:, gap]
    return out


def fill_corrupted(raw: mne.io.BaseRaw, method: str = DEFAULT_CENSOR_FILL) -> mne.io.BaseRaw:
    """Fill the corrupted spans of ``raw`` in place, every channel; the annotations stay.

    ::

      haemo with BAD_gvtd 120-123 s, method "linear"
      ->  the same haemo, 120-123 s a straight line between 119.9 s and 123.0 s, stamped "filled"

    A recording with no corrupted span comes back untouched and unstamped. One whose
    corrupted spans cover every sample is refused.
    """
    gap = corrupted_mask(raw)
    if not gap.any():
        return raw
    if gap.all():
        name = raw.filenames[0].name if raw.filenames and raw.filenames[0] else "the recording"
        raise StageError(f"{name}: no sample lies outside its corrupted BAD_ spans, so there "
                         f"is nothing to fill them from")
    sfreq = float(raw.info["sfreq"])
    logger.info("filling %.1f s under corrupted BAD_ spans (%s)", gap.sum() / sfreq, method)
    raw.apply_function(fill_values, picks="all", channel_wise=False, gap=gap, method=method,
                       sfreq=sfreq)
    return stamp(raw, stage="filled", step="censor_fill", source=raw, censor_fill=method,
                 censor_filled_s=float(gap.sum() / sfreq))


def dark_edge_frames(raw: mne.io.BaseRaw) -> tuple[int, int]:
    """How many frames at the start and at the end measured no light in most channels."""
    x = raw.get_data(picks="fnirs")
    measured = np.isfinite(x) if is_optical_density(raw) else np.isfinite(x) & (x > 0)
    # a channel dark for most of the record would darken every frame, so it has no vote
    voters = measured.mean(axis=1) > 0.5
    if not voters.any():
        return 0, 0
    # more than half of every voter's frames are lit, so at least one frame is too
    lit = measured[voters].mean(axis=0) >= 0.5
    return int(np.argmax(lit)), int(np.argmax(lit[::-1]))


def repair_dark_edges(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, tuple[int, int]]:
    """``raw`` with its dark edge frames held at the nearest lit frame and marked ``BAD_edge``."""
    lead, trail = dark_edge_frames(raw)
    if not (lead or trail):
        return raw, (0, 0)
    raw = raw.copy().load_data()
    sfreq, n = float(raw.info["sfreq"]), raw.n_times
    gap = np.zeros(n, bool)
    gap[:lead] = True
    gap[n - trail:] = True
    picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
    raw._data[picks] = fill_values(raw._data[picks], gap, "linear", sfreq)
    add_bad_spans(raw, [(start / sfreq, count / sfreq)
                        for start, count in ((0, lead), (n - trail, trail)) if count], EDGE)
    return raw, (lead, trail)
