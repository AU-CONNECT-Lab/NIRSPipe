"""Bandpass and resample, and the one place in the package a filter is designed."""

from __future__ import annotations

import mne
import mne.io
import numpy as np

from fnirs_pipe.exceptions import FilterDesignError
from fnirs_pipe.utils.lineage import stamp
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.denoise")

# ---- Filter design ----
# One design, used by the data, by the regressors filtered to match it, and by the report, so
# the three cannot drift apart.

FILTER_METHODS = ("iir", "fir")
DEFAULT_FILTER_METHOD = "iir"
DEFAULT_FILTER_ORDER = 4

# h_trans_bandwidth = h_freq * this, for method="fir". A ratio rather than an absolute value
# so the stopband edge stays at 1.5x the cutoff wherever the cutoff is set.
FIR_TRANS_RATIO = 0.5

# taps = this * sfreq / narrowest transition band, for a hamming window (MNE's own factor)
_HAMMING_LENGTH_FACTOR = 3.3


def _fir_transitions(l_freq: float | None, h_freq: float | None) -> tuple[float | None, float | None]:
    """(l_trans_bandwidth, h_trans_bandwidth) for a FIR design; None where that edge is absent.

    Only the low-pass side is overridden; the high-pass is MNE's own formula, since a
    narrower one makes the filter too long to fit a short recording.
    """
    l_trans = min(max(l_freq * 0.25, 2.0), l_freq) if l_freq else None
    h_trans = h_freq * FIR_TRANS_RATIO if h_freq else None
    return l_trans, h_trans


def fir_taps(sfreq: float, l_freq: float | None, h_freq: float | None) -> int:
    """Length in samples of the FIR that `filter_kwargs` would build.

    The narrower of the two transition bands sets it, which is usually the high-pass:
    fir_taps(10.0, 0.05, 0.5) -> 661.
    """
    trans = [t for t in _fir_transitions(l_freq, h_freq) if t]
    if not trans:
        return 0
    taps = int(round(_HAMMING_LENGTH_FACTOR * sfreq / min(trans)))
    return taps + (taps % 2 == 0)  # MNE forces an odd length so the group delay is whole samples


def filter_kwargs(
    sfreq: float,
    n_times: int,
    l_freq: float | None = None,
    h_freq: float | None = None,
    method: str = DEFAULT_FILTER_METHOD,
    order: int = DEFAULT_FILTER_ORDER,
) -> dict:
    """Keyword arguments for one bandpass design, for `Raw.filter` or `mne.filter.filter_data`.

    filter_kwargs(10.17, 3051, 0.02, 0.2) ->
        {"method": "iir", "iir_params": {"order": 4, "ftype": "butter", "output": "sos"}}

    Raises FilterDesignError when a FIR sharp enough to mean what its cutoffs say is longer
    than the recording. MNE only warns there and truncates, which would leave one set of
    nominal parameters meaning different filters on runs of different length.
    """
    if method not in FILTER_METHODS:
        raise FilterDesignError(f"filter method must be one of {FILTER_METHODS}, got {method!r}")

    if method == "iir":
        # sos rather than ba: haemodynamic cutoffs sit at a few thousandths of Nyquist, where
        # the transfer-function form loses precision well before second-order sections do
        return {"method": "iir",
                "iir_params": {"order": order, "ftype": "butter", "output": "sos"}}

    taps = fir_taps(sfreq, l_freq, h_freq)
    if taps > n_times:
        raise FilterDesignError(
            f"a FIR bandpass at {l_freq}-{h_freq} Hz needs {taps} taps ({taps / sfreq:.0f} s) "
            f"but the recording is {n_times} samples ({n_times / sfreq:.0f} s), and MNE would "
            f"truncate it without failing. Use the iir method, or a recording of at least "
            f"{taps / sfreq:.0f} s."
        )
    l_trans, h_trans = _fir_transitions(l_freq, h_freq)
    kwargs: dict = {"method": "fir", "fir_window": "hamming"}
    if l_trans is not None:
        kwargs["l_trans_bandwidth"] = l_trans
    if h_trans is not None:
        kwargs["h_trans_bandwidth"] = h_trans
    return kwargs


def filter_description(
    l_freq: float | None,
    h_freq: float | None,
    method: str = DEFAULT_FILTER_METHOD,
    order: int = DEFAULT_FILTER_ORDER,
) -> str:
    """One sentence naming the filter, for logs and for a methods section.

    filter_description(0.02, 0.2) -> "zero-phase Butterworth bandpass, order 4, 0.02-0.2 Hz"
    """
    if l_freq is not None and h_freq is not None:
        band, edges = "bandpass", f"{l_freq}-{h_freq} Hz"
    elif l_freq is not None:
        band, edges = "high-pass", f"{l_freq} Hz"
    elif h_freq is not None:
        band, edges = "low-pass", f"{h_freq} Hz"
    else:
        return "no filter"
    if method == "iir":
        return f"zero-phase Butterworth {band}, order {order}, {edges}"
    l_trans, h_trans = _fir_transitions(l_freq, h_freq)
    trans = "/".join(f"{t:g}" for t in (l_trans, h_trans) if t is not None)
    return f"hamming-windowed FIR {band}, {edges}, transition {trans} Hz"


def filter_response(
    sfreq: float,
    n_times: int,
    l_freq: float | None = None,
    h_freq: float | None = None,
    method: str = DEFAULT_FILTER_METHOD,
    order: int = DEFAULT_FILTER_ORDER,
    n_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Frequency response of the design `filter_kwargs` returns, as (freqs, dB).

    For plotting the filter itself rather than inferring it from a before/after pair. The
    IIR branch doubles the dB because MNE applies it with filtfilt, so what is drawn is the
    attenuation the data actually receives.
    """
    from scipy.signal import butter, freqz, sosfreqz

    kwargs = filter_kwargs(sfreq, n_times, l_freq, h_freq, method, order)
    if kwargs["method"] == "iir":
        btype = ("bandpass" if l_freq and h_freq else "highpass" if l_freq else "lowpass")
        wn = [l_freq, h_freq] if btype == "bandpass" else (l_freq or h_freq)
        sos = butter(kwargs["iir_params"]["order"], wn, btype=btype, fs=sfreq, output="sos")
        freqs, resp = sosfreqz(sos, worN=n_points, fs=sfreq)
        return freqs, 2 * 20 * np.log10(np.abs(resp) + 1e-30)
    taps = mne.filter.create_filter(
        None, sfreq, l_freq, h_freq, verbose=False,
        **{k: v for k, v in kwargs.items() if k != "method"})
    freqs, resp = freqz(taps, worN=n_points, fs=sfreq)
    return freqs, 20 * np.log10(np.abs(resp) + 1e-30)


def filter_array(
    data: np.ndarray,
    sfreq: float,
    l_freq: float | None = None,
    h_freq: float | None = None,
    method: str = DEFAULT_FILTER_METHOD,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """`bandpass_filter`'s filter applied to a plain (..., n_times) array.

    For signals that have to match the data but do not travel inside the Raw, which is the
    aux regressors: they come from a separate file and would otherwise be regressed against
    data that had been through a filter they never saw.
    """
    kwargs = filter_kwargs(sfreq, data.shape[-1], l_freq, h_freq, method, order)
    return mne.filter.filter_data(data, sfreq, l_freq, h_freq, verbose="error", **kwargs)


def bandpass_filter(
    haemo: mne.io.Raw,
    l_freq: float | None = None,
    h_freq: float | None = None,
    method: str = DEFAULT_FILTER_METHOD,
    order: int = DEFAULT_FILTER_ORDER,
) -> mne.io.Raw:
    kwargs = filter_kwargs(haemo.info["sfreq"], haemo.n_times, l_freq, h_freq, method, order)
    haemo.filter(l_freq=l_freq, h_freq=h_freq, **kwargs)
    # order is meaningless for FIR, so it is not recorded there: a sidecar reader should see
    # what the filter was, not a field that happens to carry a default
    return stamp(haemo, stage="filtered", step="bandpass", source=haemo,
                 l_freq=l_freq, h_freq=h_freq, filter_method=method,
                 filter_order=order if method == "iir" else None,
                 filter_description=filter_description(l_freq, h_freq, method, order))


def resample(haemo: mne.io.Raw, sfreq: float) -> mne.io.Raw:
    haemo.resample(sfreq)
    return stamp(haemo, stage="resampled", step="resample", source=haemo, sfreq=sfreq)
