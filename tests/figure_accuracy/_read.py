"""Read a fingerprint back off a trace: which pair it is, and what it carries."""

from __future__ import annotations

import numpy as np

from tests._fingerprint import EVENT_ONSETS, Truth


def xy(trace: dict) -> tuple[np.ndarray, np.ndarray]:
    """A trace's x and y as float arrays, rebuilding x from ``x0``/``dx`` where plotly dropped it."""
    y = np.asarray(trace["y"], dtype=float)
    if trace.get("x") is not None:
        return np.asarray(trace["x"], dtype=float), y
    return trace.get("x0", 0.0) + trace.get("dx", 1.0) * np.arange(len(y)), y


def traces(figure: dict, name: str | None = None, axis: str | None = None) -> list[dict]:
    return [t for t in figure["data"]
            if (name is None or t.get("name") == name)
            and (axis is None or (t.get("xaxis") or "x") == axis)]


def _uniform(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    step = float(np.median(np.diff(x)))
    grid = np.arange(x[0], x[-1], step)
    return np.interp(grid, x, y), 1.0 / step


def share_at(x, y, freq: float, band: tuple[float, float] = (0.02, 0.45)) -> float:
    """Share of the trace's power in ``band`` that sits in the bin nearest ``freq``."""
    signal, sfreq = _uniform(x, y)
    signal = (signal - signal.mean()) * np.hanning(len(signal))
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / sfreq)
    in_band = (freqs > band[0]) & (freqs < band[1])
    nearest = np.argmin(np.abs(freqs - freq))
    return float(power[max(nearest - 1, 0): nearest + 2].sum() / power[in_band].sum())


def which_pair(x, y, truth: Truth) -> str:
    """The long pair whose HbO frequency carries the most of this trace's power."""
    scores = {p.name: share_at(x, y, p.hbo_freq) for p in truth.long_pairs if not p.bad}
    return max(scores, key=scores.get)


def evoked_peak(x, y, window: float = 15.0, baseline: float = 5.0) -> float:
    """Trial-averaged peak over the fingerprint's events, baseline-corrected."""
    signal, sfreq = _uniform(x, y)
    t0 = float(np.asarray(x, dtype=float)[0])
    trials = []
    for onset in EVENT_ONSETS:
        i = int(round((onset - t0) * sfreq))
        b = int(round(baseline * sfreq))
        w = int(round(window * sfreq))
        trials.append(signal[i: i + w] - signal[i - b: i].mean())
    return float(np.mean(trials, axis=0).max())
