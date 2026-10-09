"""Inter-subject correlation: the same question as wavelet coherence, in the time domain.

  compute_isc                     One correlation per channel pair between two members,
                                  optionally after autoregressive whitening and a lag
                                  search. The matrix a report draws.
  compute_isc_pairs               The same computation delivered as a long table, with the
                                  scrambled null beside it.
  roi_mean_of_isc                 Groups those correlations into ROIs.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import mne
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from fnirs_pipe.io.snirf import long_channel_picks
from fnirs_pipe.pipeline.hyper._helpers import _shared_sfreq, _zscore_rows, long_axis_over
from fnirs_pipe.pipeline.hyper.surrogate import phase_scramble
from fnirs_pipe.pipeline.hyper.whiten import _yule_walker, autocov
from fnirs_pipe.utils import ROI_MIN_CHANNELS, bare_roi_map, fisher_r_to_z, pair_of
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.isc")


ISC_MAX_AR_ORDER = 32


def _ar_whiten(x: np.ndarray, max_order: int = ISC_MAX_AR_ORDER) -> tuple[np.ndarray, int]:
    r"""Residuals of the best autoregressive fit to ``x``, and the order that won.

    ::

      a slow, strongly autocorrelated trace  ->  a near-white one of the same length, 22

    A haemodynamic trace is heavily autocorrelated: the response is a low-pass filter on
    whatever drove it, so neighbouring samples are near copies and a correlation between two
    such traces has far fewer independent observations than it has samples. Fitting

    .. math::

        x_t = \sum_{k=1}^{p} a_k\, x_{t-k} + e_t

    and keeping :math:`e_t` leaves a series whose own samples are close to independent, so
    the correlation between two of them sits on the scale its sample count implies.

    Coefficients come from the Yule-Walker equations at each order, and ``p`` is the one
    minimising the Bayesian information criterion over ``1..max_order``, which trades the
    variance explained against the coefficients spent. The residual keeps the input's length:
    the filter is applied from the start rather than from sample ``p``, so the first ``p``
    samples are a startup transient the caller drops.

    An all-NaN row, a constant one, or one too short for a single lag comes back unchanged
    at order 0.
    """
    finite = np.isfinite(x)
    if not finite.all() or x.size < 8:
        return x, 0
    centred = x - x.mean()
    n = centred.size
    n_lag = min(int(max_order), n // 4)
    if n_lag < 1:
        return x, 0
    acov = autocov(centred, n_lag)
    if acov[0] <= 0:
        return x, 0

    best_bic, best_coef, best_order = np.inf, None, 0
    for p in range(1, n_lag + 1):
        fit = _yule_walker(acov, p)
        if fit is None:
            break
        coef, resid_var = fit
        bic = n * np.log(resid_var) + p * np.log(n)
        if bic < best_bic:
            best_bic, best_coef, best_order = bic, coef, p
    if best_order == 0:
        return x, 0
    return lfilter(np.r_[1.0, -best_coef], [1.0], x), best_order


def _whiten_rows(data: np.ndarray, max_order: int) -> tuple[np.ndarray, list[int]]:
    """:func:`_ar_whiten` over every row, with the startup transient dropped from all of them.

    Each row gets its own order, so the transient to discard is the longest of them: cutting
    per row would leave the rows on different clocks, and they are about to be correlated
    sample by sample.
    """
    out = np.empty_like(data)
    orders: list[int] = []
    for i, row in enumerate(data):
        out[i], order = _ar_whiten(row, max_order)
        orders.append(order)
    drop = max(orders) if orders else 0
    return (out[:, drop:] if drop else out), orders


def _isc_from_rows(
    data1: np.ndarray, data2: np.ndarray, max_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r of every row of ``data1`` against every row of ``data2``, and at what lag.

    ::

      max_lag 0   ->  r at the same sample, lag 0 everywhere
      max_lag 16  ->  the strongest r within 16 samples either way, and where it was

    With ``max_lag`` the pair is re-correlated at every shift in ``[-max_lag, max_lag]`` and
    the strongest of them kept.

    **Strongest means largest in magnitude, and the sign is kept**, so an anticorrelated
    pairing stays negative rather than becoming a small positive number.

    Lag is in samples and positive means the second member follows the first. Searching
    inflates the value under no coupling, since it is a maximum over many draws.

    Each shift is correlated over its own overlap, so both sides are re-standardised per lag
    rather than once over the whole record.
    """
    n = data1.shape[1]
    if max_lag <= 0:
        isc_mat = (_zscore_rows(data1) @ _zscore_rows(data2).T) / n
        np.clip(isc_mat, -1.0, 1.0, out=isc_mat)
        return isc_mat, np.zeros_like(isc_mat)

    best = best_lag = None
    for shift in range(-int(max_lag), int(max_lag) + 1):
        left  = data1[:, max(0, -shift): n - max(0, shift)]
        right = data2[:, max(0, shift): n - max(0, -shift)]
        at_lag = np.clip((_zscore_rows(left) @ _zscore_rows(right).T) / left.shape[1],
                         -1.0, 1.0)
        if best is None:
            best, best_lag = at_lag, np.full(at_lag.shape, float(shift))
            continue
        # NaN compares False, so a blank cell keeps the NaN it started with
        stronger = np.abs(at_lag) > np.abs(best)
        best = np.where(stronger, at_lag, best)
        best_lag = np.where(stronger, float(shift), best_lag)
    return best, np.where(np.isfinite(best), best_lag, np.nan)


def _isc_rows(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
    band: "tuple[float | None, float | None] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, list[str]] | tuple[None, None, None]":
    """The two members' signals on one montage axis and one clock, ready to correlate.

    One row per axis label per member, NaN where that member has no usable channel
    there, cut to ``window`` if one was asked for. Everything :func:`compute_isc` and
    :func:`compute_isc_pairs` disagree about happens after this.
    """
    if len(subject_ids) < 2:
        return None, None, None
    raw1 = aligned_raws.get(subject_ids[0])
    raw2 = aligned_raws.get(subject_ids[1])
    if raw1 is None or raw2 is None:
        return None, None, None
    # the same refusal WTC makes: at two rates sample i is not the same moment in both members
    _shared_sfreq({subject_ids[0]: raw1, subject_ids[1]: raw2})

    def _by_label(raw: mne.io.Raw) -> dict[str, int]:
        """{label: index} over what this member kept, bads dropped: what gets correlated."""
        return {pair_of(raw.ch_names[p]): p
                for p in long_channel_picks(raw, ch_type, sep_bands=sep_bands)}

    # the axis is the montage, the maps are what survived: one shape, blanks where a channel
    # went. Same axis rule as the crossed WTC matrix.
    ch_names = long_axis_over([raw1, raw2], ch_type, sep_bands)
    map1, map2 = _by_label(raw1), _by_label(raw2)
    if not ch_names:
        return None, None, None

    # alignment trims the pair to a common length, but nothing here depends on that having run
    n_times = min(raw1.n_times, raw2.n_times)

    def _rows(raw: mne.io.Raw, by_label: dict[str, int], who: str) -> np.ndarray:
        """One row per axis label, NaN for a label this subject has no usable channel at."""
        out = np.full((len(ch_names), n_times), np.nan)
        have = [c for c in ch_names if c in by_label]
        if have:
            out[[ch_names.index(c) for c in have]] = \
                raw.get_data(picks=[by_label[c] for c in have])[:, :n_times]
        if (blank := [c for c in ch_names if c not in by_label]):
            logger.warning("ISC (%s): %s has no usable %s, leaving those blank",
                           ch_type, who, ", ".join(blank))
        return out

    data1 = _rows(raw1, map1, subject_ids[0])
    data2 = _rows(raw2, map2, subject_ids[1])

    # before the cut, so the window inherits the whole record's edges rather than its own
    if band:
        sfreq_hz = float(raw1.info["sfreq"])
        data1 = _band_limit(data1, sfreq_hz, band)
        data2 = _band_limit(data2, sfreq_hz, band)

    if window is not None:
        sfreq = float(raw1.info["sfreq"])
        first = max(0, int(round(float(window[0]) * sfreq)))
        last  = min(n_times, int(round(float(window[1]) * sfreq)))
        # two samples is the least a correlation can be computed from at all
        if last - first < 2:
            logger.warning("ISC (%s): window %.1f-%.1f s holds %d sample(s) of %d, "
                           "no correlation computed",
                           ch_type, window[0], window[1], max(0, last - first), n_times)
            return None, None, None
        data1, data2 = data1[:, first:last], data2[:, first:last]

    return data1, data2, ch_names


def compute_isc(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
    whiten: int = 0,
    max_lag_s: float = 0.0,
    band: "tuple[float | None, float | None] | None" = None,
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Compute inter-brain Pearson r matrix (n_ch × n_ch) over long channels.

    matrix[i, j] = Pearson r between sub1_ch_i and sub2_ch_j.
    Diagonal = same-channel ISC.

    ``window`` restricts it to ``(tstart, tstop)`` on the aligned clock, which is how a
    condition gets a correlation of its own. Unlike the wavelet coherence this really is a
    cut and not a slice of a whole-record computation: a correlation has no frequency axis
    and ``band`` is applied over the whole record before the cut (see :func:`_band_limit`),
    so a window carries no edge that the whole record would not have had. Both sides are z-scored inside the window, because the correlation over a
    stretch is against that stretch's mean, not the recording's. The wavelet coherence is
    windowed instead; see :func:`~fnirs_pipe.pipeline.hyper.wtc.window_result`.

    Both axes are the *montage's* long channels, rejected ones included, so every dyad's
    matrix has one shape and a group analysis can stack them however their rejections
    differ. That is the convention
    :func:`fnirs_pipe.pipeline.restingstate.compute_fc` follows for the same reason. A
    rejected channel of sub1 leaves a blank row and one of sub2 a blank column -- never
    both, since sub1's copy of a channel is not needed to correlate sub2's against
    everything else.

    Position is not a safe key: a participant with one more rejected channel than the other
    shifts every channel after it, so column j would hold a different pair than its label
    claims. Everything here is looked up by S-D label.

    Rejections arrive on ``raw.info["bads"]``, which is where
    :func:`fnirs_pipe.pipeline.hyper.load_group_haemo` puts them and the only place
    the WTC path reads them from.

    Raises ValueError if the members were recorded at different sampling rates, which is
    the refusal WTC makes: alignment equalises duration, not rate.

    ``whiten`` is the largest autoregressive order :func:`_ar_whiten` may spend on each
    channel before the correlation, 0 to correlate the signals themselves. Whitening shrinks
    r, so a whitened matrix and an unwhitened one are not comparable and the sidecar records
    which was written.

    Args:
        ch_type: "hbo" or "hbr".
    """
    data1, data2, ch_names = _isc_rows(aligned_raws, subject_ids, ch_type, sep_bands,
                                       window, band)
    if ch_names is None:
        return None, None
    max_lag = _lag_samples(aligned_raws, subject_ids, max_lag_s)
    return _isc_matrix(data1, data2, whiten, max_lag)[0], ch_names


def _band_limit(data: np.ndarray, sfreq: float, band) -> np.ndarray:
    """Band-limit the rows that carry data, leaving all-NaN rows alone.

    ISC has no frequency axis of its own: it reads whatever band the preprocessing left. This
    is what lets it be put on the same band as a WTC band mean.

    Filtering happens before the window is cut, so a 300 s condition carries no edge its
    whole recording did not have. One row of 4000 samples at 10 Hz, band (0.06, 0.15) ->
    the same row with everything outside those cutoffs removed.
    """
    if not band:
        return data
    lo, hi = band
    if lo is None and hi is None:
        return data
    from fnirs_pipe.pipeline.denoise import filter_array

    out = data.copy()
    usable = ~np.isnan(data).any(axis=1)
    if usable.any():
        out[usable] = filter_array(data[usable], sfreq, lo, hi)
    return out


def _lag_samples(aligned_raws: dict, subject_ids: list[str], max_lag_s: float) -> int:
    """``max_lag_s`` seconds as whole samples of the rate both members share."""
    if not max_lag_s:
        return 0
    sfreq = _shared_sfreq({sid: aligned_raws[sid] for sid in subject_ids[:2]})
    return max(0, int(round(float(max_lag_s) * sfreq)))


def compute_isc_pairs(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
    whiten: int = 0,
    max_lag_s: float = 0.0,
    n_null: int = 0,
    seed: int | None = None,
    band: "tuple[float | None, float | None] | None" = None,
    cross: bool = True,
    on_draws: "Callable[[pd.DataFrame], None] | None" = None,
) -> "tuple[np.ndarray, list[str], pd.DataFrame, np.ndarray | None] | tuple[None, None, None, None]":
    """The ISC matrix and the same numbers as one row per channel pair, ranked against a null.

    ::

      a 3 x 3 matrix  ->  9 rows of sub1, sub2, label, label2, r, r_z, ar_order, ar_order2

    The matrix is what the report draws; the frame is what a group analysis reads, and it is
    the form the extra columns fit in. ``r_z`` is the Fisher r-to-z of ``r``. ``ar_order``
    and ``ar_order2`` are what each side's channel was whitened at, and are absent when
    ``whiten`` is 0.

    ``n_null`` phase-scrambles the second member and recomputes; scrambling preserves each
    signal's own power spectrum and so its autocorrelation. It adds ``null_mean``,
    ``null_sd``, ``null_p95`` and ``percentile``, the share of a cell's surrogate draws its
    real value beat. All four are on ``|r|``, since a surrogate is as likely to land either
    side of zero.

    With whitening and the null both on, the null is drawn through the whitening too.

    ``on_draws`` receives every surrogate as rows, one per pair and draw, signed ``r`` and
    ``r_z``, so a test that averages before ranking can be rebuilt from them.

    ``cross`` False keeps each channel against the other member's copy of it only, as an
    uncrossed coherence does: the matrix keeps its shape with every off-diagonal cell NaN,
    and the frame holds the diagonal's rows.
    """
    data1, data2, ch_names = _isc_rows(aligned_raws, subject_ids, ch_type, sep_bands,
                                       window, band)
    if ch_names is None:
        return None, None, None, None

    max_lag = _lag_samples(aligned_raws, subject_ids, max_lag_s)
    isc_mat, orders1, orders2, lags = _isc_matrix(data1, data2, whiten, max_lag)
    off = None if cross else ~np.eye(len(ch_names), dtype=bool)
    if off is not None:
        isc_mat[off] = np.nan
        lags[off] = np.nan
    sfreq = _shared_sfreq({sid: aligned_raws[sid] for sid in subject_ids[:2]})
    sub1, sub2 = subject_ids[0], subject_ids[1]

    rows = [{"sub1": sub1, "sub2": sub2, "label": a, "label2": b,
             "r": float(isc_mat[i, j])}
            for i, a in enumerate(ch_names) for j, b in enumerate(ch_names)
            if cross or i == j]
    frame = pd.DataFrame(rows)
    frame.insert(frame.columns.get_loc("r") + 1, "r_z", fisher_r_to_z(frame["r"]))
    index = {name: i for i, name in enumerate(ch_names)}
    if orders1 is not None:
        frame["ar_order"]  = frame["label"].map(lambda c: orders1[index[c]])
        frame["ar_order2"] = frame["label2"].map(lambda c: orders2[index[c]])
    if max_lag:
        ij = (frame["label"].map(index).to_numpy(), frame["label2"].map(index).to_numpy())
        frame["lag_s"] = lags[ij] / sfreq

    null_level = None
    if n_null > 0:
        draws = _isc_null_draws(data1, data2, whiten, max_lag, n_null, seed)
        if off is not None:
            draws[:, off] = np.nan
        null_level = _add_isc_null_columns(frame, isc_mat, ch_names, draws)
        if on_draws is not None:
            on_draws(_draw_rows(frame, ch_names, draws))
    return isc_mat, ch_names, frame, null_level


def _draw_rows(frame: pd.DataFrame, ch_names: list[str], draws: np.ndarray) -> pd.DataFrame:
    """The surrogate matrices as long rows on ``frame``'s pairs.

    ::

      draws (3, 2, 2), frame with 4 pairs  ->  12 rows, draw 0..2
    """
    index = {name: i for i, name in enumerate(ch_names)}
    ij = (frame["label"].map(index).to_numpy(), frame["label2"].map(index).to_numpy())
    keys = frame[["sub1", "sub2", "label", "label2"]]
    out = pd.concat([keys.assign(draw=i, r=draw[ij]) for i, draw in enumerate(draws)],
                    ignore_index=True)
    out["r_z"] = fisher_r_to_z(out["r"])
    return out


def _isc_null_draws(
    data1: np.ndarray, data2: np.ndarray, whiten: int, max_lag: int, n_iter: int,
    seed: int | None,
) -> np.ndarray:
    """``n_iter`` ISC matrices against a phase-scrambled second member.

    Only the second member is scrambled, as in :func:`compute_wtc_phase_null`, so a real
    recording is tested against a surrogate. A blank row stays blank, having no spectrum to
    preserve.
    """
    rng = np.random.default_rng(seed)
    finite = np.isfinite(data2).all(axis=1)
    draws = np.empty((n_iter, data1.shape[0], data2.shape[0]))
    for i in range(n_iter):
        surrogate = data2.copy()
        for row in np.flatnonzero(finite):
            surrogate[row] = phase_scramble(data2[row], rng)
        draws[i] = _isc_matrix(data1, surrogate, whiten, max_lag)[0]
    return draws


def _add_isc_null_columns(
    frame: pd.DataFrame, isc_mat: np.ndarray, ch_names: list[str], draws: np.ndarray,
) -> np.ndarray:
    """Summarise the surrogate draws onto ``frame`` in place, and return the 95th percentile.

    That percentile is per cell and is what a chord on the connectogram is drawn against, so
    it leaves as a matrix rather than only as a column.

    **Every null column describes the magnitude**, hence ``null_abs_``: a correlation is
    two-sided, so a surrogate that lands at -0.4 is as far from no coupling as one at +0.4,
    and the rank ``percentile`` reports is |r| among |draws|. The ``r`` column beside them is
    signed.
    """
    absolute = np.abs(draws)
    # a blanked channel makes a cell all-NaN, which every nan-aware reduction warns about and
    # then handles correctly; the blank mask below is what actually decides those cells
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        null_abs_mean = np.nanmean(absolute, axis=0)
        # an estimate from a sample of draws, as the WTC and group nulls take it
        null_abs_sd   = np.nanstd(absolute, axis=0, ddof=1)
        null_abs_p95  = np.nanpercentile(absolute, 95, axis=0)
        beaten        = (absolute < np.abs(isc_mat)[None, :, :]).mean(axis=0) * 100
    # a cell whose draws are all NaN was never ranked against anything, and a count of zero
    # there would read as a real value that lost to every surrogate
    blank = ~np.isfinite(absolute).any(axis=0) | ~np.isfinite(isc_mat)
    beaten = np.where(blank, np.nan, beaten)

    index = {name: i for i, name in enumerate(ch_names)}
    ij = (frame["label"].map(index).to_numpy(), frame["label2"].map(index).to_numpy())
    frame["null_abs_mean"] = null_abs_mean[ij]
    frame["null_abs_sd"]   = null_abs_sd[ij]
    frame["null_abs_p95"]  = null_abs_p95[ij]
    frame["percentile"]    = beaten[ij]
    return null_abs_p95


def _isc_matrix(
    data1: np.ndarray, data2: np.ndarray, whiten: int, max_lag: int = 0,
) -> "tuple[np.ndarray, list[int] | None, list[int] | None, np.ndarray]":
    """The r matrix, the AR order each row was whitened at, and the lag each cell won at.

    Whitening happens after the window has been cut, so each stretch is fitted on its own,
    and before the lag search, so the search runs on the residuals rather than on the
    autocorrelation that would make every shift look alike.
    """
    orders1 = orders2 = None
    if whiten:
        # one transient cut for both members, or sample t of one meets sample t + k of the other
        n1 = data1.shape[0]
        both, orders = _whiten_rows(np.vstack([data1, data2]), whiten)
        data1, data2 = both[:n1], both[n1:]
        orders1, orders2 = orders[:n1], orders[n1:]
    # a rejected channel contributed a row of NaN above, which the products carry
    isc_mat, lags = _isc_from_rows(data1, data2, max_lag)
    return isc_mat, orders1, orders2, lags


def roi_mean_of_isc(
    isc_mat,
    ch_names: list[str],
    roi_map: dict[str, list[str]],
    min_channels: int = ROI_MIN_CHANNELS,
) -> "tuple[np.ndarray, list[str]] | tuple[None, None]":
    """Average the channel-level ISC inside each ROI pair: the ROI number beside the ROI WTC.

    ::

      4x4 r matrix over S1_D1..S4_D4 + {"L": ["S1_D1", "S2_D2"], "R": [...]}
        -> 2x2 r matrix over ["L", "R"]

    Built from the channel values rather than from an ROI-averaged signal, which is how
    :func:`roi_mean_of_channels` builds the ROI coherence, so the two ROI numbers a page
    prints rest on the same channels.

    Cell (i, j) is the first member's ROI i against the other's ROI j, so the matrix is
    crossed and asymmetric exactly as the channel one is.

    Averaged in Fisher z and returned as r, where the coherence path averages its values
    directly: a correlation is signed and bounded and these are, unlike coherences, spread
    across zero.

    Parameters
    ----------
    isc_mat : array, shape (n_ch, n_ch)
        What :func:`compute_isc` returned, NaN where a member lost a channel.
    ch_names : list of str
        The axis of ``isc_mat``, both sides.
    roi_map : dict
        ``{region: [channel label, ...]}``. Its key order is the returned axis.
    min_channels : int
        Least channels with a value each member must contribute to a cell; thinner cells
        come back NaN. Each side is counted on its own, as the ROI coherence counts them.

    Returns
    -------
    (matrix, labels) or (None, None)
        ``(None, None)`` when there is nothing to average.
    """
    if isc_mat is None or not ch_names or not roi_map:
        return None, None
    mat = np.asarray(isc_mat, dtype=float)
    labels = list(roi_map.keys())
    index = {name: i for i, name in enumerate(ch_names)}
    picks = {roi: [index[ch] for ch in chs if ch in index]
             for roi, chs in bare_roi_map(roi_map).items()}

    out = np.full((len(labels), len(labels)), np.nan)
    thin = 0
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            if not picks[a] or not picks[b]:
                continue
            block = mat[np.ix_(picks[a], picks[b])]
            valid = np.isfinite(block)
            # channels with a value on each member's side of the block
            n_side = min(int(valid.any(axis=1).sum()), int(valid.any(axis=0).sum()))
            z = fisher_r_to_z(block[valid])
            z = z[np.isfinite(z)]
            if z.size == 0 or n_side < max(1, min_channels):
                thin += z.size > 0
                continue
            out[i, j] = float(np.tanh(z.mean()))
    if thin:
        logger.info("ROI ISC: %d cell(s) with a member under %d channels, left blank",
                    thin, min_channels)
    return out, labels


def roi_mean_of_homologous_isc(
    isc_mat,
    ch_names: list[str],
    roi_map: dict[str, list[str]],
    min_channels: int = ROI_MIN_CHANNELS,
) -> "tuple[np.ndarray, list[str]] | tuple[None, None]":
    """:func:`roi_mean_of_isc` over the same-channel pairs only, the ISC beside the homologous ROI coherence.

    ::

      4x4 r matrix + {"L": ["S1_D1", "S2_D2"], ...}
        -> L x L is the Fisher z mean of (S1_D1, S1_D1) and (S2_D2, S2_D2); off the
           diagonal NaN

    The counterpart of :func:`~fnirs_pipe.pipeline.hyper.roi.roi_mean_of_homologous`: the
    crossed ROI diagonal averages every pairing inside a region, this averages only a channel
    against the other member's copy of it.
    """
    if isc_mat is None or not ch_names or not roi_map:
        return None, None
    mat = np.asarray(isc_mat, dtype=float)
    same = np.full_like(mat, np.nan)
    np.fill_diagonal(same, np.diag(mat))
    out, labels = roi_mean_of_isc(same, ch_names, roi_map, min_channels=min_channels)
    if out is not None:
        # a channel listed in two regions would otherwise put its own pair off the diagonal
        out[~np.eye(len(labels), dtype=bool)] = np.nan
    return out, labels
