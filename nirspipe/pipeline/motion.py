"""Motion artifact correction for optical density signals.

tddr, wavelet and tpca are implemented; spline has no backend yet.
"""

from typing import Any, Literal

import mne
import mne.io
import numpy as np

from nirspipe.utils.lineage import stamp

MotionMethod = Literal["tddr", "wavelet", "tpca", "spline", "none"]


# ---- Wavelet (Molavi 2012) ----

WAVELET = "db2"
# detail coefficients this many interquartile ranges beyond the quartiles of their block are zeroed
WAVELET_IQR_FACTOR = 1.5
# a record zero-padded to 2**N samples is decomposed into N - WAVELET_LOWEST_SCALE levels
WAVELET_LOWEST_SCALE = 4


def _wl_transform(x: np.ndarray, n_levels: int, wavelet: str) -> np.ndarray:
    """Shift-invariant DWT: row 0 ends as the approximation, row d + 1 holds level d + 1's details.

    Each level splits every block of row 0 into the DWT of the block and the DWT of its
    one-sample circular shift, so level d + 1 holds 2**(d + 1) blocks, each one phase of the
    whole record decimated 2**(d + 1) times:

      x of 8 samples, n_levels 2 -> wp[1] = [cD | cD_shift]                 2 blocks of 4
                                    wp[2] = [cD | cD_sh | cD | cD_sh]       4 blocks of 2
    """
    import pywt
    wp = np.zeros((n_levels + 1, len(x)))
    wp[0] = x
    for d in range(n_levels):
        blocks = wp[0].reshape(2 ** d, -1)
        cA, cD = pywt.dwt(blocks, wavelet, mode="periodization", axis=1)
        cA_sh, cD_sh = pywt.dwt(np.roll(blocks, 1, axis=1), wavelet, mode="periodization", axis=1)
        wp[0] = np.hstack([cA, cA_sh]).ravel()
        wp[d + 1] = np.hstack([cD, cD_sh]).ravel()
    return wp


def _wl_inverse(wp: np.ndarray, wavelet: str) -> np.ndarray:
    """Inverse of _wl_transform: each block becomes the mean of its two phase reconstructions."""
    import pywt
    approx = wp[0].copy()
    for d in range(wp.shape[0] - 2, -1, -1):
        a = approx.reshape(2 ** d, -1)
        c = wp[d + 1].reshape(2 ** d, -1)
        h = a.shape[1] // 2
        s1 = pywt.idwt(a[:, :h], c[:, :h], wavelet, mode="periodization", axis=1)
        s_sh = pywt.idwt(a[:, h:], c[:, h:], wavelet, mode="periodization", axis=1)
        approx = ((s1 + np.roll(s_sh, -1, axis=1)) / 2).ravel()
    return approx


def _wl_clip_iqr(blocks: np.ndarray, n_stats: int, iqr_factor: float) -> np.ndarray:
    """Zero each row's outliers, the fence taken from the row's first n_stats entries only.

    The tail of a row is padding: kept out of the quartiles, clipped like the rest. Quartiles
    use the Hazen rule (the k-th of n sorted values sits at (k - 0.5) / n).
    """
    q1, q3 = np.quantile(blocks[:, :n_stats], [0.25, 0.75], axis=1, method="hazen", keepdims=True)
    fence = iqr_factor * (q3 - q1)
    return np.where((blocks > q3 + fence) | (blocks < q1 - fence), 0.0, blocks)


def _wl_filter_coeffs(wp: np.ndarray, iqr_factor: float, signal_length: int) -> np.ndarray:
    """Zero detail outliers block by block, one fence per phase, at every level but the coarsest.

    A level-j block holds signal_length // 2**j coefficients of the recording before its padding.
    """
    wp = wp.copy()
    n_stats = signal_length
    for j in range(1, wp.shape[0] - 1):
        n_stats //= 2
        wp[j] = _wl_clip_iqr(wp[j].reshape(2 ** j, -1), n_stats, iqr_factor).ravel()
    return wp


def _wavelet_motion_correct(raw_od: mne.io.Raw, wavelet: str = WAVELET,
                            iqr_factor: float = WAVELET_IQR_FACTOR) -> mne.io.Raw:
    """Wavelet motion correction (Molavi 2012), per channel in OD space.

    Zero-pad to 2**N → remove the mean → _wl_transform into N - WAVELET_LOWEST_SCALE levels →
    _wl_filter_coeffs → _wl_inverse → restore the mean and length. The published noise
    normalisation is left out: IQR clipping is scale-invariant, so it would cancel.
    """
    raw = raw_od.copy()

    def _corr(signal):
        n0 = len(signal)
        n_pow = int(np.ceil(np.log2(n0)))
        padded = np.zeros(2 ** n_pow)
        padded[:n0] = signal
        dc = padded.mean()
        wp = _wl_transform(padded - dc, max(n_pow - WAVELET_LOWEST_SCALE, 0), wavelet)
        return _wl_inverse(_wl_filter_coeffs(wp, iqr_factor, n0), wavelet)[:n0] + dc

    raw.apply_function(_corr, channel_wise=True)
    return raw


# ---- Targeted PCA (Yücel 2014) ----

# a sample is motion when, within TPCA_T_MOTION seconds, a good channel changes by more than
# TPCA_STD_THRESH SDs of its first difference or by more than TPCA_AMP_THRESH in OD
TPCA_T_MOTION = 0.5
TPCA_STD_THRESH = 20.0
TPCA_AMP_THRESH = 5.0
# seconds marked either side of a detected sample
TPCA_T_MASK = 1.0
# share of the motion samples' variance the removed components may carry
TPCA_N_SV = 0.97
TPCA_MAX_ITER = 5


def _tpca_motion_mask(d: np.ndarray, sfreq: float, good: np.ndarray) -> np.ndarray:
    """Motion samples of a time x channels array, one mask shared by every channel.

    Row r of the change matrix is the largest |d[r + lag] - d[r]| over lags 1..n_lag on each
    good channel; a flagged row is widened by n_mask rows either side and marks sample r + 1.
    """
    x = d[:, good]
    mask = np.zeros(len(d), bool)
    if x.shape[1] == 0 or len(x) < 2:
        return mask
    # MATLAB round, half away from zero: 2.5 samples is 3
    n_lag = int(np.floor(TPCA_T_MOTION * sfreq + 0.5))
    n_mask = int(np.floor(TPCA_T_MASK * sfreq + 0.5))
    change = np.zeros((len(x) - 1, x.shape[1]))
    for lag in range(1, min(n_lag, len(x) - 1) + 1):
        change[:len(x) - lag] = np.maximum(change[:len(x) - lag], np.abs(x[lag:] - x[:-lag]))
    sd = np.std(np.diff(x, axis=0), axis=0, ddof=1)
    hit = np.flatnonzero((change > TPCA_STD_THRESH * sd).any(axis=1)
                         | (change.max(axis=1) > TPCA_AMP_THRESH))
    rows = (hit[:, None] + np.arange(-n_mask, n_mask + 1)).ravel()
    rows = rows[(rows >= 0) & (rows < len(x) - 1)]
    mask[rows + 1] = True
    return mask


def _tpca_pca(d: np.ndarray, motion: np.ndarray, good: np.ndarray,
              n_sv: float) -> tuple[np.ndarray, int]:
    """Remove the leading components of the motion samples and stitch the segments back.

    n_sv in (0, 1) is a variance share turned into a component count; otherwise it is the
    count. Each corrected segment is offset to continue from the sample before it, and each
    clean stretch to continue from the segment before it. The segment bookkeeping is kept
    exactly as published, which places a segment one sample early:

      one channel, d = [0 1 2 3 4 5 6], motion on samples 2-4, n_sv 1 -> [0 1 1 1 2 3 4]
    """
    if not motion.any():
        return d, 0
    cols = np.flatnonzero(good)
    y = d[motion][:, cols]
    u, s, _ = np.linalg.svd(y.T @ y)
    share = np.cumsum(s / s.sum())
    n_remove = int(np.count_nonzero(share < n_sv)) if 0 < n_sv < 1 else int(n_sv)
    u = u[:, :n_remove]
    yc = y - y @ u @ u.T

    # 1-based segment edges, as published: a start is the last clean sample, an end the last
    # motion sample
    t_inc = (~motion).astype(int)
    n = len(t_inc)
    starts = np.flatnonzero(np.diff(t_inc) == -1) + 1
    ends = np.flatnonzero(np.diff(t_inc) == 1) + 1
    if len(ends) == 0:
        ends = np.array([n])
    if len(starts) == 0:
        starts = np.array([1])
    if starts[0] > ends[0]:
        starts = np.r_[1, starts]
    if starts[-1] > ends[-1]:
        ends = np.r_[ends, n]
    cum = np.cumsum(ends - starts)

    out = d.copy()
    a, b = starts[0] - 1, ends[0] - 1
    if starts[0] > 1:
        out[a:b, cols] = yc[:cum[0]] - yc[0] + out[a, cols]
    else:
        out[a:b, cols] = yc[:cum[0]] - yc[cum[0] - 1] + out[b - 1, cols]
    for k in range(len(ends) - 1):
        a, b = ends[k] - 2, starts[k + 1]
        out[a:b, cols] = d[a:b, cols] - d[a, cols] + out[a, cols]
        a, b = starts[k + 1] - 1, ends[k + 1] - 1
        out[a:b, cols] = yc[cum[k]:cum[k + 1]] - yc[cum[k]] + out[a, cols]
    if ends[-1] < n:
        a = ends[-1] - 2
        out[a:, cols] = d[a:, cols] - d[a, cols] + out[a, cols]
    return out, n_remove


def _tpca_motion_correct(raw_od: mne.io.Raw) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Targeted PCA motion correction (Yücel 2014), across good channels in OD space.

    Detect with _tpca_motion_mask → _tpca_pca → detect again on the result, until nothing is
    flagged or TPCA_MAX_ITER passes have run. The first pass turns TPCA_N_SV into a component
    count and later passes remove that same count. Bad channels are returned unchanged.
    Returns the corrected raw and what the passes did.
    """
    raw = raw_od.copy()
    d = raw.get_data().T
    good = ~np.isin(raw.ch_names, raw.info["bads"])
    sfreq = float(raw.info["sfreq"])
    motion = _tpca_motion_mask(d, sfreq, good)
    share_before = float(motion.mean())
    n_sv: float = TPCA_N_SV
    removed: list[int] = []
    while motion.any() and len(removed) < TPCA_MAX_ITER:
        d, n_sv = _tpca_pca(d, motion, good, n_sv)
        removed.append(n_sv)
        motion = _tpca_motion_mask(d, sfreq, good)
    raw.apply_function(lambda _: d.T, picks="all", channel_wise=False)
    return raw, {"components_removed": removed, "motion_share_before": share_before,
                 "motion_share_after": float(motion.mean())}


# ---- Dispatch ----
def correct_motion(raw_od: mne.io.Raw, method: MotionMethod | None = None) -> mne.io.Raw:
    """Apply motion artifact correction to the OD signal.

    Returns corrected raw (modified in-place for tddr). tpca also records what its passes did
    on the stamp, under ``tpca``.
    Raises NotImplementedError for spline (no backend yet).
    """
    if method is None:
        raise ValueError("--motion-correction is required.")
    params: dict[str, Any] = {}
    if method == "tddr":
        corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    elif method == "wavelet":
        corrected = _wavelet_motion_correct(raw_od)
    elif method == "tpca":
        corrected, params["tpca"] = _tpca_motion_correct(raw_od)
    elif method == "spline":
        raise NotImplementedError(
            "Spline correction has no backend. Use tddr, wavelet, tpca or none. "
            "It is kept as a named method rather than an unknown one so the API says "
            "which it is; the CLI and the GUI do not offer it."
        )
    elif method == "none":
        corrected = raw_od
    else:
        raise ValueError(f"Unknown motion correction method: {method}")
    return stamp(corrected, stage="motcorrected", step="motion_correction",
                 source=raw_od, method=method, **params)
