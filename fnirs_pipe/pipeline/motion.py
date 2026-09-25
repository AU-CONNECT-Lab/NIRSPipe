"""Motion artifact correction for optical density signals.

tddr and wavelet are implemented; spline has no backend yet.
"""

from typing import Literal

import mne
import mne.io
import numpy as np

from fnirs_pipe.utils.lineage import stamp

MotionMethod = Literal["tddr", "wavelet", "spline", "none"]


# ---- Wavelet (Molavi 2012) ----
def _wl_clip_iqr(block: np.ndarray, iqr_factor: float) -> None:
    """Zero motion-artifact outliers in one detail-coefficient block, in place."""
    q25, q75 = np.percentile(block, [25, 75])
    fence = iqr_factor * (q75 - q25)
    # block is a slice view of the caller's array, so this writes through
    block[:] = np.where((block > q75 + fence) | (block < q25 - fence), 0.0, block)


def _wl_filter_coeffs(coeffs, iqr_factor: float, signal_length: int):
    """Zero SWT detail-coefficient outliers, one IQR fence per level."""
    cAf = coeffs[0][0].copy()          # iswt reads only this one approximation
    out = []
    for _, cD in coeffs:
        cDf = cD.copy()
        # one fence per level over the whole time span, never per time window; the slice
        # keeps the padding out of the statistics
        _wl_clip_iqr(cDf[:signal_length], iqr_factor)
        out.append((cAf, cDf))
    return out


def _wavelet_motion_correct(raw_od: mne.io.Raw, wavelet: str = "db2", iqr_factor: float = 1.5,
                            level: int | None = None) -> mne.io.Raw:
    """Wavelet motion correction (Molavi 2012), per channel in OD space.

    Pad to 2^k → remove DC → MAD noise-normalize → SWT → zero detail-coefficient outliers beyond
    Q1/Q3 ± iqr_factor·IQR (per level) → iSWT → denormalize → restore DC and length.

    level is the decomposition depth.
    """
    import pywt
    raw = raw_od.copy()

    def _corr(signal):
        n0 = len(signal)
        padded = np.zeros(2 ** int(np.ceil(np.log2(n0))))
        padded[:n0] = signal
        dc = padded.mean()
        padded = padded - dc
        mad_ds = np.median(np.abs(padded[::2] - np.median(padded[::2])))
        # IQR clipping is scale-invariant, so this normalisation cancels out; kept to
        # stay faithful to the published procedure
        norm_coef = 1.0 / (1.4826 * mad_ds) if mad_ds != 0 else 1.0
        normed = padded * norm_coef
        depth = level if level is not None else int(np.ceil(np.log2(n0))) - 4
        lvl = max(1, min(depth, int(np.log2(len(normed))) - 1))
        coeffs = _wl_filter_coeffs(pywt.swt(normed, wavelet, level=lvl), iqr_factor, n0)
        return (pywt.iswt(coeffs, wavelet) / norm_coef)[:n0] + dc

    raw.apply_function(_corr, channel_wise=True)
    return raw


# ---- Dispatch ----
def correct_motion(raw_od: mne.io.Raw, method: MotionMethod | None = None) -> mne.io.Raw:
    """Apply motion artifact correction to the OD signal.

    Returns corrected raw (modified in-place for tddr).
    Raises NotImplementedError for spline (no backend yet).
    """
    if method is None:
        raise ValueError("--motion-correction is required.")
    if method == "tddr":
        corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    elif method == "wavelet":
        corrected = _wavelet_motion_correct(raw_od)
    elif method == "spline":
        raise NotImplementedError(
            "Spline correction has no backend. Use tddr, wavelet or none. "
            "It is kept as a named method rather than an unknown one so the API says "
            "which it is; the CLI and the GUI do not offer it."
        )
    elif method == "none":
        corrected = raw_od
    else:
        raise ValueError(f"Unknown motion correction method: {method}")
    return stamp(corrected, stage="motcorrected", step="motion_correction",
                 source=raw_od, method=method)
