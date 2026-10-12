"""Motion artifact correction for optical density signals: tddr, wavelet, tpca, spline, spline_sg,
wavelet_tddr (wavelet, then tddr) and spline_wavelet (spline, then wavelet)."""

from typing import Any, Literal

import mne
import mne.io
import numpy as np
import scipy.sparse as sp
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, freqz, savgol_coeffs, savgol_filter
from scipy.sparse.linalg import spsolve

from nirspipe.utils import pair_of
from nirspipe.utils.lineage import stamp

MotionMethod = Literal["tddr", "wavelet", "tpca", "spline", "spline_sg", "wavelet_tddr",
                       "spline_wavelet", "none"]


def _mround(x: float) -> int:
    """MATLAB round for x >= 0, half away from zero: 2.5 samples is 3."""
    return int(np.floor(x + 0.5))


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


def _motion_mask(x: np.ndarray, sfreq: float, t_motion: float, std_thresh: float,
                 amp_thresh: float, t_mask: float) -> np.ndarray:
    """Motion samples of a time x channels array, one mask for all its columns.

    Row r of the change matrix is the largest |x[r + lag] - x[r]| over lags 1..n_lag on each
    column; a row where any column exceeds std_thresh SDs of its first difference or
    amp_thresh is widened by n_mask rows either side and marks sample r + 1.
    """
    mask = np.zeros(len(x), bool)
    if x.shape[1] == 0 or len(x) < 2:
        return mask
    n_lag = _mround(t_motion * sfreq)
    n_mask = _mround(t_mask * sfreq)
    change = np.zeros((len(x) - 1, x.shape[1]))
    for lag in range(1, min(n_lag, len(x) - 1) + 1):
        change[:len(x) - lag] = np.maximum(change[:len(x) - lag], np.abs(x[lag:] - x[:-lag]))
    sd = np.std(np.diff(x, axis=0), axis=0, ddof=1)
    hit = np.flatnonzero((change > std_thresh * sd).any(axis=1)
                         | (change.max(axis=1) > amp_thresh))
    rows = (hit[:, None] + np.arange(-n_mask, n_mask + 1)).ravel()
    rows = rows[(rows >= 0) & (rows < len(x) - 1)]
    mask[rows + 1] = True
    return mask


def _tpca_motion_mask(d: np.ndarray, sfreq: float, good: np.ndarray) -> np.ndarray:
    """Motion samples of a time x channels array, one mask shared by every good channel."""
    return _motion_mask(d[:, good], sfreq, TPCA_T_MOTION, TPCA_STD_THRESH, TPCA_AMP_THRESH,
                        TPCA_T_MASK)


def _segment_edges(t_inc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1-based edges of the motion segments of a 1/0 inclusion vector, as published.

    A start is the last clean sample before a segment (1 if the record starts in motion), an
    end the segment's last motion sample (n if the record ends in motion):

      t_inc = [1 1 0 0 0 1 1] -> starts [2], ends [5]
    """
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
    return starts, ends


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

    n = len(motion)
    starts, ends = _segment_edges((~motion).astype(int))
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


# ---- Spline (Scholkmann 2010) ----

# a pair's samples are motion when, within SPLINE_T_MOTION seconds, either of its channels
# changes by more than SPLINE_STD_THRESH SDs of its first difference or by more than
# SPLINE_AMP_THRESH in OD
SPLINE_T_MOTION = 0.5
SPLINE_STD_THRESH = 50.0
SPLINE_AMP_THRESH = 5.0
# seconds marked either side of a detected sample
SPLINE_T_MASK = 1.0
# smoothing parameter of the cubic smoothing spline, on time in seconds
SPLINE_P = 0.99
# rejoining averages a whole stretch shorter than SPLINE_DT_SHORT s, SPLINE_DT_SHORT s of one
# shorter than SPLINE_DT_LONG s, and a tenth of a longer one
SPLINE_DT_SHORT = 0.3
SPLINE_DT_LONG = 3.0


def _spline_motion_mask(d: np.ndarray, sfreq: float, ch_names: list[str], good: np.ndarray,
                        t_motion: float = SPLINE_T_MOTION, std_thresh: float = SPLINE_STD_THRESH,
                        amp_thresh: float = SPLINE_AMP_THRESH) -> np.ndarray:
    """Motion samples per channel, time x channels, detected per source-detector pair.

    Both wavelengths of a pair take the pair's mask, and both feed its detection even when
    one is marked bad; a bad channel is never marked.
    """
    pairs = np.array([pair_of(ch) for ch in ch_names])
    mask = np.zeros(d.shape, bool)
    for pair in dict.fromkeys(pairs[good]):
        cols = pairs == pair
        rows = _motion_mask(d[:, cols], sfreq, t_motion, std_thresh, amp_thresh, SPLINE_T_MASK)
        mask[:, cols & good] = rows[:, None]
    return mask


def _smoothing_spline(t: np.ndarray, y: np.ndarray, p: float) -> np.ndarray:
    """Cubic smoothing spline of y over t with natural ends, read back at t.

    f = argmin  p * sum_i (y_i - f(t_i))^2 + (1 - p) * integral f''(t)^2 dt

    Solved by the Reinsch algorithm with a = (1 - p) / p: (R + a Q'Q) g = Q'y, f = y - a Q g,
    where Q (n x n-2) takes second divided differences and R is the tridiagonal Gram matrix
    of the spline's second derivative. Two points or fewer are returned as they are.
    """
    n = len(y)
    if n < 3:
        return y.copy()
    h = np.diff(t)
    inv = 1.0 / h
    q = sp.diags([inv[:-1], -inv[:-1] - inv[1:], inv[1:]], [0, -1, -2], shape=(n, n - 2))
    r = sp.diags([(h[:-1] + h[1:]) / 3, h[1:-1] / 6, h[1:-1] / 6], [0, 1, -1],
                 shape=(n - 2, n - 2))
    a = (1 - p) / p
    g = spsolve((r + a * (q.T @ q)).tocsc(), q.T @ y)
    return y - a * (q @ g)


def _spline_window(length: int, sfreq: float) -> int:
    """Samples averaged at one end of a stretch when it is rejoined to its neighbour."""
    if length < SPLINE_DT_SHORT * sfreq:
        return length
    if length < SPLINE_DT_LONG * sfreq:
        return int(np.floor(SPLINE_DT_SHORT * sfreq))
    return length // 10


def _spline_correct(y: np.ndarray, t_inc: np.ndarray, t: np.ndarray, sfreq: float,
                    p: float = SPLINE_P) -> np.ndarray:
    """One channel's spline correction, with the published segment bookkeeping kept exactly.

    Each motion segment loses its smoothing spline; then each stretch is shifted so the mean
    over a window at its start meets the mean over a window at the end of the stretch
    before it, clean stretches from the raw signal and segments from the spline residual.
    As published, a segment runs from the last clean sample to the last motion sample but
    one, the last clean stretch starts one sample earlier than the others, and a record that
    ends in motion keeps its final sample raw.
    """
    n = len(y)
    starts, ends = _segment_edges(t_inc)
    lengths = ends - starts
    out = y.copy()
    for s, e in zip(starts, ends):
        out[s - 1:e - 1] = y[s - 1:e - 1] - _smoothing_spline(t[s - 1:e - 1], y[s - 1:e - 1], p)

    # first segment: onto the stretch before it, or after it if the record starts in motion
    s, e = starts[0], ends[0]
    w_curr = _spline_window(lengths[0], sfreq)
    if s > 1:
        w_prev = _spline_window(s - 1, sfreq)
        mean_prev = out[s - 1 - w_prev:s - 1].mean()
        mean_curr = out[s - 1:s - 1 + w_curr].mean()
        out[s - 1:e - 1] = out[s - 1:e - 1] - mean_curr + mean_prev
    else:
        w_next = _spline_window((starts[1] if len(starts) > 1 else n) - e + 1, sfreq)
        mean_curr = out[e - 2 - w_curr:e - 2].mean()
        mean_next = out[e - 1:e - 1 + w_next].mean()
        out[s - 1:e - 1] = out[s - 1:e - 1] - mean_curr + mean_next

    for k in range(len(starts) - 1):
        a, b = ends[k], starts[k + 1] - 1
        clean_len = b - a + 1
        w_prev = _spline_window(lengths[k], sfreq)
        w_curr = _spline_window(clean_len, sfreq)
        mean_prev = out[a - 1 - w_prev:a - 1].mean()
        mean_curr = y[a - 1:a - 1 + w_curr].mean()
        out[a - 1:b] = y[a - 1:b] - mean_curr + mean_prev

        a, b = starts[k + 1], ends[k + 1] - 1
        w_prev = _spline_window(clean_len, sfreq)
        w_curr = _spline_window(lengths[k + 1], sfreq)
        mean_prev = out[a - 1 - w_prev:a - 1].mean()
        mean_curr = out[a - 1:a - 1 + w_curr].mean()
        out[a - 1:b] = out[a - 1:b] - mean_curr + mean_prev

    if ends[-1] < n:
        a = ends[-1] - 1
        w_prev = _spline_window(lengths[-1], sfreq)
        w_curr = _spline_window(n - a + 1, sfreq)
        mean_prev = out[a - 1 - w_prev:a - 1].mean()
        mean_curr = y[a - 1:a - 1 + w_curr].mean()
        out[a - 1:] = y[a - 1:] - mean_curr + mean_prev
    return out


def _spline_motion_correct(raw_od: mne.io.Raw, t_motion: float = SPLINE_T_MOTION,
                           std_thresh: float = SPLINE_STD_THRESH,
                           amp_thresh: float = SPLINE_AMP_THRESH) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Spline motion correction (Scholkmann 2010), channel by channel in OD space.

    Detect with _spline_motion_mask, then _spline_correct every good channel with motion.
    Bad channels are returned unchanged. Returns the corrected raw, the share of good-channel
    samples marked as motion, and the number of segments corrected, summed over channels.
    """
    sfreq = float(raw_od.info["sfreq"])
    if np.floor(SPLINE_DT_SHORT * sfreq) < 1:
        raise ValueError(f"Spline correction needs at least {1 / SPLINE_DT_SHORT:.4g} Hz; this "
                         f"recording has {sfreq:g} Hz, at which its rejoining would average "
                         f"over empty windows.")
    raw = raw_od.copy()
    d = raw.get_data().T
    good = ~np.isin(raw.ch_names, raw.info["bads"])
    motion = _spline_motion_mask(d, sfreq, raw.ch_names, good, t_motion, std_thresh, amp_thresh)
    out = d.copy()
    n_segments = 0
    for ch in np.flatnonzero(motion.any(axis=0)):
        t_inc = (~motion[:, ch]).astype(int)
        out[:, ch] = _spline_correct(d[:, ch], t_inc, raw.times, sfreq)
        n_segments += len(_segment_edges(t_inc)[0])
    raw.apply_function(lambda _: out.T, picks="all", channel_wise=False)
    return raw, {"motion_share": float(motion[:, good].mean()) if good.any() else 0.0,
                 "segments": n_segments}


# ---- Spline-SG (Jahani 2018) ----
# the authors' implementation, which departs from the paper's text in five places

# smoothing parameter of the spline step, on time in seconds
SPLINE_SG_P = 0.99
# the cubic Savitzky-Golay frame that ends the correction, in seconds
SPLINE_SG_FRAME = 10.0
SPLINE_SG_ORDER = 3
# zero-phase third-order Butterworth low-pass under the detection and the spline step
SPLINE_SG_LOWPASS = 2.0
# a further low-pass under the detection's moving standard deviation
SPLINE_SG_DETECT_LOWPASS = 0.5
# detection: moving SD over this many seconds, fenced at this many IQRs beyond its quartiles
SPLINE_SG_STD_WINDOW = 1.0
SPLINE_SG_STD_IQR = 2.0
# detection: the signal's gradient fenced at this many IQRs beyond its quartiles
SPLINE_SG_GRAD_IQR = 1.5
# a mark is a baseline shift when it moves the signal more than the median change of the
# clean signal over this many seconds
SPLINE_SG_SHIFT_LAG = 0.5
# a wavelength whose clean stretches have a mean-to-SD ratio below this takes its partner's
# marks; a pair below it on both is left unmarked
SPLINE_SG_SNR_MIN = 3.0
# seconds of edge value padded onto each end around the spline step
SPLINE_SG_PAD = 12.0


def _butter_lowpass(x: np.ndarray, sfreq: float, cutoff: float) -> np.ndarray:
    """Third-order Butterworth low-pass along time, zero phase, padded as MATLAB's filtfilt pads."""
    b, a = butter(3, cutoff * 2 / sfreq)
    return filtfilt(b, a, x, axis=0, padtype="odd", padlen=3 * (len(a) - 1))


def _hazen_fence(x: np.ndarray, factor: float) -> tuple[float, float]:
    q1, q3 = np.quantile(x, [0.25, 0.75], method="hazen")
    return q1 - (q3 - q1) * factor, q3 + (q3 - q1) * factor


def _clean_runs(clean: np.ndarray) -> list[slice]:
    """Maximal runs of True: [T T F T] -> [slice(0, 2), slice(3, 4)]."""
    edges = np.flatnonzero(np.diff(np.r_[0, clean.astype(int), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2])]


def _spline_sg_candidates(s1: np.ndarray, sfreq: float) -> np.ndarray:
    """0-based samples a channel's low-passed signal flags as motion candidates.

    A gradient outlier flags its own sample, a moving-SD outlier the centre of its window. As
    implemented, the gradient is the full convolution with [-1 0 1], so its fence takes n + 2
    values and its first two, -s1[0] and -s1[1], are the signal's own level.
    """
    n = len(s1)
    s2 = _butter_lowpass(s1, sfreq, SPLINE_SG_DETECT_LOWPASS)
    win = _mround(SPLINE_SG_STD_WINDOW * sfreq)
    sd = sliding_window_view(s2, win + 1).std(axis=1, ddof=1)
    low, high = _hazen_fence(sd, SPLINE_SG_STD_IQR)
    by_sd = np.flatnonzero((sd > high) | (sd < low)) + _mround(win / 2)
    grad = np.convolve(s1, [-1, 0, 1])
    low, high = _hazen_fence(grad, SPLINE_SG_GRAD_IQR)
    by_grad = np.flatnonzero((grad[:n] > high) | (grad[:n] < low))
    return np.union1d(by_grad, by_sd)


def _spline_sg_shift_threshold(x: np.ndarray, runs: list[slice], sfreq: float) -> float:
    """Median change over SPLINE_SG_SHIFT_LAG s within the clean stretches, as implemented: a zero
    joins the pool, and a stretch no longer than the lag adds nothing.

      changes 1, 2, 3 -> median of 0 1 2 3 = 1.5
    """
    lag = _mround(SPLINE_SG_SHIFT_LAG * sfreq)
    changes = [np.zeros(1)] + [np.abs(x[r][lag:] - x[r][:-lag]) for r in runs
                               if r.stop - r.start > lag]
    return float(np.quantile(np.concatenate(changes), 0.5, method="hazen"))


def _spline_sg_snr(x: np.ndarray, runs: list[slice], sfreq: float) -> float:
    """Mean |mean| / SD of the clean stretches longer than 3 s, the first and last left out;
    nan with fewer than three such stretches."""
    ratios = [abs(x[r].mean()) / (x[r].std(ddof=1) + np.finfo(float).eps) for r in runs
              if r.stop - r.start > 3 * sfreq]
    return float(np.mean(ratios[1:-1])) if len(ratios) > 2 else np.nan


def _spline_sg_keep(moved: np.ndarray, span: np.ndarray, threshold: float,
                    sfreq: float) -> np.ndarray:
    """Which candidate runs are corrected, as implemented: those moving the signal more than
    threshold, and by span (last sample minus the clean sample before it) those over 0.1 s
    and under 0.5 s, or over 1 s; a run of 0.5 to 1 s is kept by the amplitude rule alone."""
    return ((moved > threshold) | ((span > 0.1 * sfreq) & (span < 0.49999 * sfreq))
            | (span > sfreq))


def _spline_sg_channel_mask(y: np.ndarray, sfreq: float) -> tuple[np.ndarray, float]:
    """One channel's marked samples and its clean-signal ratio for the SNR rule (nan if unusable)."""
    s1 = _butter_lowpass(y, sfreq, SPLINE_SG_LOWPASS)
    candidates = _spline_sg_candidates(s1, sfreq)
    if len(candidates) == 0:
        return np.zeros(len(y), bool), np.nan
    pad = _mround(SPLINE_SG_PAD * sfreq)
    s1 = np.r_[np.full(pad, s1[0]), s1, np.full(pad, s1[-1])]
    sig = np.ones(len(s1))
    sig[candidates + pad] = 0
    starts = np.flatnonzero(np.diff(sig) == -1)          # the clean sample before each run
    ends = np.flatnonzero(np.diff(sig) == 1)             # each run's last sample
    # clean stretches among all samples but the last, which never joins one
    runs = _clean_runs(sig[:-1] == 1)
    threshold = _spline_sg_shift_threshold(_butter_lowpass(s1, sfreq, SPLINE_SG_LOWPASS), runs,
                                           sfreq)
    keep = _spline_sg_keep(np.abs(s1[starts] - s1[ends]), ends - starts, threshold, sfreq)
    marked = np.zeros(len(s1), bool)
    for a, b in zip(starts[keep], ends[keep]):
        marked[a:b + 1] = True
    return marked[pad:len(marked) - pad], _spline_sg_snr(s1, runs, sfreq)


def _wavelength_pairs(ch_names: list[str]) -> list[tuple[int, int]]:
    """Column indices of each source-detector pair, shorter wavelength first."""
    groups: dict[str, list[int]] = {}
    for i, ch in enumerate(ch_names):
        groups.setdefault(pair_of(ch), []).append(i)
    pairs = []
    for pair, cols in groups.items():
        if len(cols) != 2:
            raise ValueError(f"Spline-SG needs two wavelengths per source-detector pair; "
                             f"{pair} has {len(cols)}.")
        pairs.append(tuple(sorted(cols, key=lambda c: float(ch_names[c].rsplit(" ", 1)[1]))))
    return pairs


def _spline_sg_motion_mask(d: np.ndarray, sfreq: float,
                           ch_names: list[str]) -> tuple[np.ndarray, int]:
    """Marked samples per channel, time x channels, after the authors' SNR rule; and how many
    pairs the rule left unmarked because both wavelengths fell below SPLINE_SG_SNR_MIN.

    A channel without a usable ratio is scored on its whole record instead.
    """
    masks, snrs = zip(*(_spline_sg_channel_mask(d[:, c], sfreq) for c in range(d.shape[1])))
    mask = np.column_stack(masks)
    snr = np.array(snrs)
    redo = np.isnan(snr) | (snr == 0)
    snr[redo] = np.abs(d[:, redo].mean(axis=0)) / d[:, redo].std(axis=0, ddof=1)
    n_below = 0
    for a, b in _wavelength_pairs(ch_names):
        if snr[a] < SPLINE_SG_SNR_MIN and snr[b] < SPLINE_SG_SNR_MIN:
            mask[:, [a, b]] = False
            n_below += 1
        elif snr[a] > SPLINE_SG_SNR_MIN and snr[b] < SPLINE_SG_SNR_MIN:
            mask[:, b] = mask[:, a]
        elif snr[a] < SPLINE_SG_SNR_MIN and snr[b] > SPLINE_SG_SNR_MIN:
            mask[:, a] = mask[:, b]
    return mask, n_below


def _spline_sg_motion_correct(raw_od: mne.io.Raw) -> tuple[mne.io.Raw, dict[str, Any]]:
    """Spline-SG motion correction (Jahani 2018) in OD space, as its authors implemented it.

    Mark with _spline_sg_motion_mask; pad SPLINE_SG_PAD s of edge value onto both ends of the
    data and the marks, low-pass at SPLINE_SG_LOWPASS Hz, _spline_correct both wavelengths of
    every pair whose shorter wavelength is good, trim, then smooth every channel, bad ones
    included, with a cubic Savitzky-Golay filter over SPLINE_SG_FRAME s, which low-passes
    the record near 0.1 Hz. Returns the corrected raw and what the marking did.
    """
    sfreq = float(raw_od.info["sfreq"])
    if sfreq <= 2 * SPLINE_SG_LOWPASS:
        raise ValueError(f"Spline-SG needs more than {2 * SPLINE_SG_LOWPASS:g} Hz for its "
                         f"{SPLINE_SG_LOWPASS:g} Hz low-pass; this recording has {sfreq:g} Hz.")
    raw = raw_od.copy()
    d = raw.get_data().T
    good = ~np.isin(raw.ch_names, raw.info["bads"])
    motion, n_below = _spline_sg_motion_mask(d, sfreq, raw.ch_names)

    pad = _mround(SPLINE_SG_PAD * sfreq)
    x = _butter_lowpass(np.pad(d, ((pad, pad), (0, 0)), mode="edge"), sfreq, SPLINE_SG_LOWPASS)
    marked = np.pad(motion, ((pad, pad), (0, 0)), mode="edge")
    t = np.arange(len(x)) / sfreq
    out = x.copy()
    n_segments = 0
    for first, second in _wavelength_pairs(raw.ch_names):
        if not good[first]:
            continue
        for ch in (first, second):
            if marked[:, ch].any():
                t_inc = (~marked[:, ch]).astype(int)
                out[:, ch] = _spline_correct(x[:, ch], t_inc, t, sfreq, p=SPLINE_SG_P)
                n_segments += len(_segment_edges(t_inc)[0])
    out = out[pad:len(out) - pad]
    frame = _mround(SPLINE_SG_FRAME * sfreq)
    frame += frame % 2 == 0
    out = savgol_filter(out, frame, SPLINE_SG_ORDER, axis=0, mode="interp")
    raw.apply_function(lambda _: out.T, picks="all", channel_wise=False)
    return raw, {"motion_share": float(motion[:, good].mean()) if good.any() else 0.0,
                 "segments": n_segments, "pairs_below_snr": n_below}


def spline_sg_cutoff(sfreq: float = 10.0) -> float:
    """-3 dB frequency (Hz) of the Savitzky-Golay step at this sampling rate.

    The frame is fixed in seconds, so the cutoff barely moves with the rate.
    """
    frame = _mround(SPLINE_SG_FRAME * sfreq)
    frame += frame % 2 == 0
    w, h = freqz(savgol_coeffs(frame, SPLINE_SG_ORDER), worN=2 ** 14, fs=sfreq)
    return float(w[np.argmax(np.abs(h) < 1 / np.sqrt(2))])


# ---- Spline + wavelet (Di Lorenzo 2019) ----
# spline with this detection, then the wavelet; smoothing, rejoining and the wavelet itself
# keep spline's and the wavelet's constants

SPLINE_WAVELET_T_MOTION = 1.0
SPLINE_WAVELET_AMP_THRESH = 0.4
# what --motion-std-thresh and --wavelet-iqr default to under this method
SPLINE_WAVELET_STD_THRESH = 13.0
SPLINE_WAVELET_IQR_FACTOR = 0.8


# ---- Dispatch ----

# the methods each user setting reaches, and what each takes when the setting is left unset
STD_THRESH_DEFAULTS = {"spline": SPLINE_STD_THRESH, "spline_wavelet": SPLINE_WAVELET_STD_THRESH}
IQR_FACTOR_DEFAULTS = {"wavelet": WAVELET_IQR_FACTOR, "wavelet_tddr": WAVELET_IQR_FACTOR,
                       "spline_wavelet": SPLINE_WAVELET_IQR_FACTOR}


def motion_settings(method: str | None, std_thresh: float | None = None,
                    iqr_factor: float | None = None) -> dict[str, float]:
    """The settings this method runs with, keyed as its sidecar records them.

    An unset one takes the method's default; one the method does not read is refused:

      spline_wavelet, nothing set -> {"motion_std_thresh": 13.0, "wavelet_iqr": 0.8}
      tddr, nothing set           -> {}
      tddr, iqr_factor 0.5        -> ValueError
    """
    settings: dict[str, float] = {}
    for key, flag, value, defaults in (
            ("motion_std_thresh", "--motion-std-thresh", std_thresh, STD_THRESH_DEFAULTS),
            ("wavelet_iqr", "--wavelet-iqr", iqr_factor, IQR_FACTOR_DEFAULTS)):
        if method in defaults:
            settings[key] = defaults[method] if value is None else float(value)
        elif value is not None:
            raise ValueError(f"{flag} applies to {', '.join(defaults)}; "
                             f"--motion-correction {method} does not read it.")
        if key in settings and not settings[key] > 0:
            raise ValueError(f"{flag} must be positive, not {settings[key]:g}.")
    return settings


# a spline detector that marks more of the good-channel samples than this has likely read
# physiology as motion: its threshold in OD falls as the sampling rate rises
MOTION_SHARE_WARN = 0.10


def overmarked(method: str | None, summary: dict[str, Any] | None) -> bool:
    """Whether a spline detector's recorded summary marks more than MOTION_SHARE_WARN."""
    return (method in STD_THRESH_DEFAULTS and summary is not None
            and summary.get("motion_share", 0.0) > MOTION_SHARE_WARN)


def correct_motion(raw_od: mne.io.Raw, method: MotionMethod | None = None,
                   std_thresh: float | None = None,
                   iqr_factor: float | None = None) -> mne.io.Raw:
    """Apply motion artifact correction to the OD signal.

    Returns corrected raw (modified in-place for tddr). tpca, spline, spline_sg and
    spline_wavelet also record what they did on the stamp, under their own name. std_thresh
    and iqr_factor are resolved by motion_settings.
    """
    if method is None:
        raise ValueError("--motion-correction is required.")
    settings = motion_settings(method, std_thresh, iqr_factor)
    std_thresh, iqr_factor = settings.get("motion_std_thresh"), settings.get("wavelet_iqr")
    params: dict[str, Any] = {}
    if method == "tddr":
        corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    elif method == "wavelet":
        corrected = _wavelet_motion_correct(raw_od, iqr_factor=iqr_factor)
    elif method == "tpca":
        corrected, params["tpca"] = _tpca_motion_correct(raw_od)
    elif method == "spline":
        corrected, params["spline"] = _spline_motion_correct(raw_od, std_thresh=std_thresh)
    elif method == "spline_sg":
        corrected, params["spline_sg"] = _spline_sg_motion_correct(raw_od)
    elif method == "wavelet_tddr":
        corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(
            _wavelet_motion_correct(raw_od, iqr_factor=iqr_factor))
    elif method == "spline_wavelet":
        splined, params["spline_wavelet"] = _spline_motion_correct(
            raw_od, t_motion=SPLINE_WAVELET_T_MOTION, std_thresh=std_thresh,
            amp_thresh=SPLINE_WAVELET_AMP_THRESH)
        corrected = _wavelet_motion_correct(splined, iqr_factor=iqr_factor)
    elif method == "none":
        corrected = raw_od
    else:
        raise ValueError(f"Unknown motion correction method: {method}")
    return stamp(corrected, stage="motcorrected", step="motion_correction",
                 source=raw_od, method=method, **params)
