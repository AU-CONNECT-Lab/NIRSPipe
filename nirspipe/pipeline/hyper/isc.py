"""Inter-subject correlation: the same question as wavelet coherence, in the time domain.

::

  compute_isc                     One correlation per channel pair between two members,
                                  optionally after a lag search. The matrix a report
                                  draws.
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

from nirspipe.qc.metrics._helpers import long_channel_picks
from nirspipe.pipeline.hyper._helpers import _shared_sfreq, long_axis_over
from nirspipe.pipeline.hyper.surrogate import phase_rotate, phase_scramble
from nirspipe.utils import ROI_MIN_CHANNELS, bare_roi_map, fisher_r_to_z, pair_of, zscore_rows
from nirspipe.utils.logging import get_logger
from nirspipe.utils.spans import bad_spans, frames_under

logger = get_logger("pipeline.isc")


def _masked_r(left: np.ndarray, right: np.ndarray, keep: "np.ndarray | None") -> np.ndarray:
    """Pearson r of every row against every row, over the samples ``keep`` marks; NaN below two."""
    if keep is not None:
        left, right = left[:, keep], right[:, keep]
        if left.shape[1] < 2:
            return np.full((left.shape[0], right.shape[0]), np.nan)
    return np.clip((zscore_rows(left) @ zscore_rows(right).T) / left.shape[1], -1.0, 1.0)


def _isc_from_rows(
    data1: np.ndarray, data2: np.ndarray, max_lag: int = 0,
    keep: "np.ndarray | None" = None,
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

    ``keep`` marks the samples outside the pairing's excluded time; a pair of samples counts
    only when both of its positions are kept, and fewer than two such pairs give NaN.
    """
    n = data1.shape[1]
    if max_lag <= 0:
        if keep is not None:
            isc_mat = _masked_r(data1, data2, keep)
            return isc_mat, np.zeros_like(isc_mat)
        isc_mat = (zscore_rows(data1) @ zscore_rows(data2).T) / n
        np.clip(isc_mat, -1.0, 1.0, out=isc_mat)
        return isc_mat, np.zeros_like(isc_mat)

    best = best_lag = None
    for shift in range(-int(max_lag), int(max_lag) + 1):
        left  = data1[:, max(0, -shift): n - max(0, shift)]
        right = data2[:, max(0, shift): n - max(0, -shift)]
        if keep is None:
            at_lag = np.clip((zscore_rows(left) @ zscore_rows(right).T) / left.shape[1],
                             -1.0, 1.0)
        else:
            at_lag = _masked_r(left, right, keep[max(0, -shift): n - max(0, shift)]
                               & keep[max(0, shift): n - max(0, -shift)])
        if best is None:
            best, best_lag = at_lag, np.full(at_lag.shape, float(shift))
            continue
        # NaN compares False, so a blank cell keeps the NaN it started with
        stronger = np.abs(at_lag) > np.abs(best)
        if keep is not None:
            # a shift whose kept overlap was too short for an r gives way to one that had enough
            stronger |= np.isnan(best) & np.isfinite(at_lag)
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
    skip_s: float = 0.0,
    excluded: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None] | tuple[None, None, None, None]":
    """The two members' signals on one montage axis and one clock, ready to correlate.

    One row per axis label per member, NaN where that member has no usable channel
    there, cut to ``window`` if one was asked for. Nothing before ``skip_s`` is kept, which
    is where a whitened record carries its filter transient. Everything
    :func:`compute_isc` and :func:`compute_isc_pairs` disagree about happens after this.

    The fourth value marks the samples outside the pairing's excluded time, the union of
    both members' ``BAD_`` spans unless ``excluded`` gives the samples instead; None when
    nothing is excluded.
    """
    if len(subject_ids) < 2:
        return None, None, None, None
    raw1 = aligned_raws.get(subject_ids[0])
    raw2 = aligned_raws.get(subject_ids[1])
    if raw1 is None or raw2 is None:
        return None, None, None, None
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
        return None, None, None, None

    # alignment trims the pair to a common length, but nothing here depends on that having run
    n_times = min(raw1.n_times, raw2.n_times)
    keep = _kept_samples(raw1, raw2, n_times, excluded)

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

    if window is not None or skip_s:
        sfreq = float(raw1.info["sfreq"])
        window = window if window is not None else (0.0, n_times / sfreq)
        first, last = _window_samples(window, skip_s, sfreq, n_times)
        # two samples is the least a correlation can be computed from at all
        if last - first < 2:
            logger.warning("ISC (%s): window %.1f-%.1f s holds %d sample(s) of %d, "
                           "no correlation computed",
                           ch_type, window[0], window[1], max(0, last - first), n_times)
            return None, None, None, None
        data1, data2, keep = data1[:, first:last], data2[:, first:last], keep[first:last]

    if keep.all():
        return data1, data2, ch_names, None
    if keep.sum() < 2:
        logger.warning("ISC (%s): every sample of %s lies in an excluded BAD_ span of %s or "
                       "%s, so its correlations are NaN", ch_type,
                       "the window" if window is not None else "the recording",
                       *subject_ids[:2])
    return data1, data2, ch_names, keep


def _kept_samples(raw1: mne.io.Raw, raw2: mne.io.Raw, n_times: int,
                  excluded: "np.ndarray | None") -> np.ndarray:
    """Samples outside the pairing's excluded time: ``excluded``, else both members' BAD_ spans."""
    if excluded is None:
        excluded = (frames_under(raw1, bad_spans(raw1))[:n_times]
                    | frames_under(raw2, bad_spans(raw2))[:n_times])
    return ~np.asarray(excluded, dtype=bool)[:n_times]


def _window_samples(window: "tuple[float, float]", skip_s: float, sfreq: float,
                    n_times: int) -> tuple[int, int]:
    """The first and one-past-last sample of ``window``, none before ``skip_s``."""
    first = max(0, int(round(float(window[0]) * sfreq)), int(np.ceil(skip_s * sfreq)))
    return first, min(n_times, int(round(float(window[1]) * sfreq)))


def _kept_share(aligned_raws: dict[str, mne.io.Raw], subject_ids: list[str],
                window: "tuple[float, float]", excluded: "np.ndarray | None" = None) -> float:
    """Share of ``window``'s samples a correlation reads, those outside the excluded time."""
    raw1, raw2 = (aligned_raws[sid] for sid in subject_ids[:2])
    n_times = min(raw1.n_times, raw2.n_times)
    first, last = _window_samples(window, 0.0, float(raw1.info["sfreq"]), n_times)
    return float(_kept_samples(raw1, raw2, n_times, excluded)[first:last].mean())


def compute_isc(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    ch_type: str = "hbo",
    sep_bands=None,
    window: "tuple[float, float] | None" = None,
    max_lag_s: float = 0.0,
    band: "tuple[float | None, float | None] | None" = None,
    skip_s: float = 0.0,
    excluded: "np.ndarray | None" = None,
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
    windowed instead; see :func:`~nirspipe.pipeline.hyper.wtc.window_result`.

    Both axes are the *montage's* long channels, rejected ones included, so every dyad's
    matrix has one shape and a group analysis can stack them however their rejections
    differ. That is the convention
    :func:`nirspipe.pipeline.restingstate.compute_fc` follows for the same reason. A
    rejected channel of sub1 leaves a blank row and one of sub2 a blank column -- never
    both, since sub1's copy of a channel is not needed to correlate sub2's against
    everything else.

    Position is not a safe key: a participant with one more rejected channel than the other
    shifts every channel after it, so column j would hold a different pair than its label
    claims. Everything here is looked up by S-D label.

    Rejections arrive on ``raw.info["bads"]``, which is where
    :func:`nirspipe.pipeline.hyper.group_io.load_group_haemo` puts them and the only place
    the WTC path reads them from.

    Raises ValueError if the members were recorded at different sampling rates, which is
    the refusal WTC makes: alignment equalises duration, not rate.

    Whitening is the caller's: hand in the copies
    :func:`~nirspipe.pipeline.hyper.whiten.whiten_raws` made of the whole record, with
    ``skip_s`` past their zeroed transient, so every channel of both members carries one
    order. Whitening shrinks r, so a whitened matrix and an unwhitened one are not
    comparable and the sidecar records which was written.

    Samples under either member's ``BAD_`` spans are left out of every r; ``excluded``, the
    excluded samples on the pair's clock, stands in for those spans when given.

    Args:
        ch_type: "hbo", "hbr" or "hbt".
    """
    data1, data2, ch_names, keep = _isc_rows(aligned_raws, subject_ids, ch_type, sep_bands,
                                             window, band, skip_s, excluded)
    if ch_names is None:
        return None, None
    max_lag = _lag_samples(aligned_raws, subject_ids, max_lag_s)
    return _isc_matrix(data1, data2, max_lag, keep)[0], ch_names


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
    from nirspipe.pipeline.denoise import filter_array

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
    max_lag_s: float = 0.0,
    n_null: int = 0,
    seed: "int | list[int] | None" = None,
    band: "tuple[float | None, float | None] | None" = None,
    cross: bool = True,
    on_draws: "Callable[[pd.DataFrame], None] | None" = None,
    skip_s: float = 0.0,
    excluded: "np.ndarray | None" = None,
    shared_phases: bool = False,
    draw_order: "list[str] | None" = None,
) -> "tuple[np.ndarray, list[str], pd.DataFrame, np.ndarray | None] | tuple[None, None, None, None]":
    """The ISC matrix and the same numbers as one row per channel pair, ranked against a null.

    ::

      a 3 x 3 matrix  ->  9 rows of sub1, sub2, label, label2, r, r_z

    The matrix is what the report draws; the frame is what a group analysis reads, and it is
    the form the extra columns fit in. ``r_z`` is the Fisher r-to-z of ``r``. Whitening,
    ``skip_s`` and the excluded samples are as in :func:`compute_isc`; each surrogate is
    correlated over the same kept samples.

    ``n_null`` phase-scrambles the second member and recomputes; scrambling preserves each
    signal's own power spectrum and so its autocorrelation. It adds ``null_mean``,
    ``null_sd``, ``null_p95`` and ``percentile``, the share of a cell's surrogate draws its
    real value beat. All four are on ``|r|``, since a surrogate is as likely to land either
    side of zero. ``shared_phases`` rotates all of the second member's channels by one set of
    random phases per draw (:func:`~nirspipe.pipeline.hyper.surrogate.phase_rotate`) instead
    of scrambling each channel on its own.

    ``draw_order`` is the order, by label, in which the surrogates draw the second member's
    channels from the stream ``seed`` starts; ``ch_names`` when left at None. A group passes
    its own axis, so one member on one stream is one surrogate in every pairing it closes,
    whatever the montage of the member it is paired with.

    On whitened recordings the null scrambles the residuals, as the coherence's does.

    ``on_draws`` receives every surrogate as rows, one per pair and draw, signed ``r`` and
    ``r_z``, so a test that averages before ranking can be rebuilt from them.

    ``cross`` False keeps each channel against the other member's copy of it only, as an
    uncrossed coherence does: the matrix keeps its shape with every off-diagonal cell NaN,
    and the frame holds the diagonal's rows.
    """
    data1, data2, ch_names, keep = _isc_rows(aligned_raws, subject_ids, ch_type, sep_bands,
                                             window, band, skip_s, excluded)
    if ch_names is None:
        return None, None, None, None

    max_lag = _lag_samples(aligned_raws, subject_ids, max_lag_s)
    isc_mat, lags = _isc_matrix(data1, data2, max_lag, keep)
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
    if max_lag:
        ij = (frame["label"].map(index).to_numpy(), frame["label2"].map(index).to_numpy())
        frame["lag_s"] = lags[ij] / sfreq

    null_level = None
    if n_null > 0:
        order = None
        if draw_order is not None:
            rank = {name: k for k, name in enumerate(draw_order)}
            order = sorted(range(len(ch_names)), key=lambda r: rank.get(ch_names[r], len(rank)))
        draws = _isc_null_draws(data1, data2, max_lag, n_null, seed, keep, shared_phases,
                                order)
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
    data1: np.ndarray, data2: np.ndarray, max_lag: int, n_iter: int,
    seed: "int | list[int] | None", keep: "np.ndarray | None" = None, shared: bool = False,
    order: "list[int] | None" = None,
) -> np.ndarray:
    """``n_iter`` ISC matrices against a phase-scrambled second member.

    Only the second member is scrambled, as in :func:`compute_wtc_phase_null`, so a real
    recording is tested against a surrogate. A blank row stays blank, having no spectrum to
    preserve. ``order`` is the rows in the order they draw from the stream, all rows in
    turn by default.
    """
    rng = np.random.default_rng(seed)
    finite = np.isfinite(data2).all(axis=1)
    rows = [r for r in (range(len(finite)) if order is None else order) if finite[r]]
    draws = np.empty((n_iter, data1.shape[0], data2.shape[0]))
    for i in range(n_iter):
        surrogate = data2.copy()
        if shared:
            surrogate[finite] = phase_rotate(data2[finite], rng)
        else:
            for row in rows:
                surrogate[row] = phase_scramble(data2[row], rng)
        draws[i] = _isc_matrix(data1, surrogate, max_lag, keep)[0]
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
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        null_abs_mean = np.nanmean(absolute, axis=0)
        # an estimate from a sample of draws, as the WTC and group nulls take it
        null_abs_sd   = np.nanstd(absolute, axis=0, ddof=1)
        null_abs_p95  = np.nanpercentile(absolute, 95, axis=0)
        # over the finite draws only, as the re-paired null ranks
        beaten        = ((absolute < np.abs(isc_mat)[None, :, :]).sum(axis=0)
                         / np.isfinite(absolute).sum(axis=0) * 100)
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
    data1: np.ndarray, data2: np.ndarray, max_lag: int = 0,
    keep: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """The r matrix and the lag each cell won at; a rejected channel's NaN row carries through."""
    return _isc_from_rows(data1, data2, max_lag, keep)


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

    The counterpart of :func:`~nirspipe.pipeline.hyper.roi.roi_mean_of_homologous`: the
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
