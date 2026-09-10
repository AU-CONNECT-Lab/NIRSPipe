"""GVTD: the global motion index, its threshold, and censoring on it.

The whole of GVTD is here, from the per-sample trace to the spans a run marks as
``BAD_gvtd``, because the parts are only meaningful together: the threshold is defined
against the trace, the censoring is defined against the threshold, and the channel set the
figures print is the one the censoring used.

Which channels and why the flagged fraction runs high are settled in
``qc/GVTD/gvtd_channel_set_and_threshold.md``, measured on 17 recordings. Two decisions from
it are carried here: ``GVTD_MOTION_BAND`` does not follow ``--mode``, and the channel set is
the long channels, not an option, since the analysis never uses the rest.

``_motion_metrics`` keeps its name, which predates the split and says "motion" where it
means GVTD. The other motion measures, spikes and the correction footprint, are in
``motion.py``.
"""

from typing import Any

import mne
import numpy as np

from fnirs_pipe.qc.metrics._helpers import long_short_channels, _mask_to_segments, _safe_metrics
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.gvtd")


# Hz, Sherafati 2020. One band for every mode: a QC number that moved with the analysis band
# could not be compared across a cohort.
GVTD_MOTION_BAND = (0.01, 0.5)
GVTD_N_STD = 3.0  # one constant: the figures draw this threshold, the record stores it


def gvtd_timetrace(
    data: np.ndarray,
    sfreq: float,
    l_freq: float | None = None,
    h_freq: float | None = None,
    standardize_channels: bool = False,
) -> np.ndarray:
    r"""GVTD time trace: RMS across channels of the temporal derivative.

    A global motion index from the per-sample temporal derivative of all channels
    :footcite:`Sherafati2020`:

    .. math::

        g_i = \sqrt{\frac{1}{N} \sum_{j=1}^{N} \left(y_{j,i} - y_{j,i-1}\right)^2},

    where :math:`N` is the number of channels and :math:`y_{j,i}` the value of
    channel :math:`j` at sample :math:`i`. Two optional steps, both off by default:
    a motion-band bandpass before differencing (``l_freq``/``h_freq``) and
    per-channel standardization (``standardize_channels``).

    Parameters
    ----------
    data : np.ndarray
        Channel-by-time array, typically optical density.
    sfreq : float
        Sampling frequency in Hz.
    l_freq, h_freq : float or None, optional
        Edges of an order-4 Butterworth bandpass applied before differencing, to
        isolate the motion band. An ``h_freq`` at or above Nyquist (``sfreq / 2``)
        is dropped, degrading the bandpass to a high-pass.
    standardize_channels : bool, optional
        If True, divide each channel's derivative by its own SD before the RMS
        (DVARS-vstd analog :footcite:`Nichols2013`), so high-dynamic-range channels do not
        dominate the global value.

    Returns
    -------
    np.ndarray
        GVTD trace, one value per timepoint (length ``n_times - 1``).

    Notes
    -----
    Input NaN/inf are zeroed before filtering and differencing. The trace is
    non-negative by construction, matching the paper's :math:`g_i > 0`.

    References
    ----------
    .. footbibliography::
    """
    # NaN/inf would poison filter + diff
    d = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # drop an h_freq at/above Nyquist (sfreq/2): not a valid IIR cutoff, the butterworth would error
    if h_freq is not None and h_freq >= sfreq / 2:
        h_freq = None

    # Bandpass filter the data before differencing (temporal derivative) to isolate motion-band fluctuations (optional).
    if l_freq is not None or h_freq is not None:
        d = mne.filter.filter_data(
            d, sfreq, l_freq, h_freq, method="iir",
            iir_params=dict(order=4, ftype="butter"), verbose=False,
        )
        
    # Compute the temporal derivative along time (axis=1) and then the RMS across channels (axis=0).
    diff = np.diff(d, axis=1)  # temporal derivative along time (x[:,i] - x[:,i-1])

    # Optional channel-wise standardization (DVARS-vstd analog): divide each channel's derivative by its own SD before the RMS, 
    # so high-dynamic-range channels don't dominate the global value.
    if standardize_channels:
        # z-score each channel's derivative by its own SD (DVARS-vstd)
        sd = diff.std(axis=1, keepdims=True)
        # safe divide: flat channels (sd==0) stay 0 instead of 0/0=nan
        diff = np.divide(diff, sd, out=np.zeros_like(diff), where=sd > 0)
    return np.sqrt(np.mean(diff ** 2, axis=0))  # axis=0: RMS across channels -> one value per timepoint


def gvtd_threshold(gvtd: np.ndarray, n_std: float = 3.0) -> float | None:
    r"""GVTD motion threshold, histogram-mode :footcite:`Sherafati2020`.

    .. math::

        \sigma_L = \sqrt{\frac{1}{|L|} \sum_{g_i \in L} (g_i - m)^2},
        \qquad L = \{\, g_i : g_i < m \,\},

    .. math::

        \tau = m + n_\text{std}\,\sigma_L,

    where :math:`m` is the histogram mode (center of the tallest bin, bins ~ n/5)
    and :math:`\sigma_L` is the left-tail std, the RMS spread of the points below
    the mode, which are free of motion-spike contamination.

    Parameters
    ----------
    gvtd : np.ndarray
        GVTD time trace.
    n_std : float, optional
        Multiplier on the left-tail std.

    Returns
    -------
    float or None
        Motion threshold, or None if the trace has no positive values.

    References
    ----------
    .. footbibliography::
    """
    # keep only finite samples; an all-zero/empty trace has no meaningful threshold
    g = gvtd[np.isfinite(gvtd)]
    gmax = float(g.max()) if g.size else 0.0
    if gmax <= 0:
        return None

    # histogram with ~5 samples per bin: enough resolution to locate the resting peak
    n_bins = max(1, int(round(g.size / 5)))
    bin_w = gmax / n_bins
    counts, edges = np.histogram(g, bins=np.arange(0.0, gmax + bin_w, bin_w))
    if counts.size == 0:
        return None

    # mode = center of the tallest bin = the resting GVTD level (most timepoints are motion-free)
    run_mode = float(edges[int(np.argmax(counts))] + bin_w / 2)

    # left tail = every sample below the mode; these are the motion-free "resting" points
    # (samples above the mode are inflated by motion spikes, so we exclude them from the std)
    below = g[g < run_mode]

    # degenerate case: nothing is below the mode, so there is no spread to measure ->
    # fall back to the bare mode as the threshold (no noise margin added)
    if below.size == 0:
        return run_mode
    
    # left-tail std = RMS distance of those below-mode points from the mode
    left_std = float(np.sqrt(np.sum((below - run_mode) ** 2) / below.size))

    # threshold sits n_std of resting noise above the resting level; anything above = motion
    return run_mode + n_std * left_std


def gvtd_channel_picks(
    raw: mne.io.Raw, sep_bands=None,
) -> "tuple[list[str], str]":
    """Channels for the GVTD trace and carpet, plus the name to label the figure with.

    The long channels, which is the set the analysis uses, so a run is judged on the
    channels it is built from. A montage with no registered optode positions has no long
    channels to pick and an empty pick has no trace at all, so it falls back to every
    channel. The returned label is what actually happened rather than what was intended,
    since it is what the figure prints and a wrong label makes two runs look comparable
    when they are not.

    Example: a 40-channel montage with 22 long and 18 out-of-band channels returns
    ``(22 names, "long")``; the same call on an unregistered montage returns
    ``(40 names, "all")``.
    """
    long_names, _ = long_short_channels(raw, sep_bands)
    if not long_names:
        logger.warning("no long channels by separation; GVTD falls back to every channel")
        return list(raw.ch_names), "all"
    return long_names, "long"


def gvtd_channel_blocks(
    raw: mne.io.Raw, sep_bands=None,
) -> "list[tuple[str, list[str]]]":
    """The GVTD panel's rows, canonical set first, as ``[(set name, channel names), ...]``.

    The canonical block is what :func:`gvtd_channel_picks` selects and is the only one the
    reported scalars and the verdict come from. A second block follows when the montage has
    short channels the canonical set left out, so the panel can show their quality without
    changing the number the run is judged on::

        44-channel montage, 28 long + 16 short  ->  [("long", 28), ("short", 16)]
        hyper montage, 22 long + 0 short        ->  [("long", 22)]
        unregistered montage, no long channels  ->  [("all", 40)]

    The fallback set gets no second block: it already contains the short channels, and a row
    for a subset of the row above it would be read as a comparison between two independent
    sets.
    """
    picks, picked_set = gvtd_channel_picks(raw, sep_bands)
    if picked_set != "long":
        return [(picked_set, picks)]
    _, short_names = long_short_channels(raw, sep_bands)
    blocks = [("long", picks)]
    if short_names:
        blocks.append(("short", short_names))
    return blocks


@_safe_metrics("GVTD metrics", (
    "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95",
    "gvtd_vstd_mean", "gvtd_vstd_p95",
    "gvtd_thresh", "gvtd_thresh_applied",
    "gvtd_num_above_thresh", "gvtd_pct_above_thresh",
))
def _motion_metrics(raw_intensity: mne.io.Raw,
                    thresh: "float | None" = None) -> dict[str, Any]:
    """GVTD motion metrics from OD: mean/p95, motion-band and vstd variants, and threshold.

    Parameters
    ----------
    raw_intensity : mne.io.Raw
        Raw intensity recording, or one already in optical density.
    thresh : float, optional
        Count the above-threshold timepoints against this cutoff instead of this
        recording's own. Given by the caller for the motion-corrected file, so that the
        before and after halves of a pair are counted against one yardstick.

        Without it the pair is not a comparison. The threshold is the mode of the trace's
        own histogram plus 3 SD, so it tracks whatever distribution it is handed: on
        ``sub-p1d01`` it rises 1.54x across the correction (4.81e-04 to 7.39e-04), and the
        share above it falls 52.4% to 4.0% while the share above the *original* cutoff only
        falls to 33.1%. Most of that 13x was the cutoff moving. It is the same rule the
        per-condition views follow, which the project already states: fix the yardstick over
        the run and count the mask, never re-derive it on the part being compared.

        ``gvtd_thresh`` still reports this recording's own cutoff either way, because the
        report has a row for exactly that and its movement is worth seeing;
        ``gvtd_thresh_applied`` names the one the counts used, so a record can never be read
        as counting against a cutoff it did not use.

    Returns
    -------
    dict
        gvtd_mean/p95 (canonical), gvtd_filt_* (motion-band), gvtd_vstd_*
        (channel-standardized), gvtd_thresh, gvtd_thresh_applied, and the num/pct of
        timepoints above the applied one.

    Notes
    -----
    Already-OD input is passed through rather than converted, so the same metrics can be
    measured on the motion-corrected file and compared against the ones taken on the
    original recording. ``optical_density`` raises on anything that is not continuous-wave
    amplitude, so the guard is what makes that second call possible at all.
    """
    raw_od = (raw_intensity if is_optical_density(raw_intensity)
              else mne.preprocessing.nirs.optical_density(raw_intensity.copy()))
    sfreq = float(raw_od.info["sfreq"])
    od_data = np.nan_to_num(raw_od.get_data(), nan=0.0, posinf=0.0, neginf=0.0)
    gvtd_ts = gvtd_timetrace(od_data, sfreq)                           # canonical (unfiltered)
    gvtd_filt = gvtd_timetrace(od_data, sfreq, *GVTD_MOTION_BAND)  # motion-band
    gvtd_vstd = gvtd_timetrace(od_data, sfreq, standardize_channels=True)  # channel-equalized
    own_thresh = gvtd_threshold(gvtd_filt, n_std=GVTD_N_STD)
    applied = own_thresh if thresh is None else float(thresh)
    if applied is not None:
        above = gvtd_filt > applied
        pct_above = float(np.mean(above))
        num_above = int(np.sum(above))
    else:
        pct_above = num_above = None
    return {
        "gvtd_mean": float(gvtd_ts.mean()),
        "gvtd_p95": float(np.percentile(gvtd_ts, 95)),
        "gvtd_filt_mean": float(gvtd_filt.mean()),
        "gvtd_filt_p95": float(np.percentile(gvtd_filt, 95)),
        "gvtd_vstd_mean": float(gvtd_vstd.mean()),
        "gvtd_vstd_p95": float(np.percentile(gvtd_vstd, 95)),
        "gvtd_thresh": own_thresh,
        "gvtd_thresh_applied": applied,
        "gvtd_num_above_thresh": num_above,
        "gvtd_pct_above_thresh": pct_above,
    }


def gvtd_above_segments(raw_intensity: mne.io.Raw, sep_bands=None,
                        picks: "list[str] | None" = None) -> "list[tuple[float, float]]":
    """Time spans above the run's own GVTD threshold, as (onset, duration) pairs.

    The same boolean :func:`_motion_metrics` counts for ``gvtd_pct_above_thresh``, kept as
    spans so a view of part of the recording can count its own share against the threshold
    the whole run set. Recomputing the threshold on a piece would give each piece its own
    yardstick; see :func:`gvtd_threshold`.

    ``picks`` names the channel set; without it the canonical one. GVTD is an RMS across
    channels, so each set has its own trace and its own threshold and the spans of one set
    cannot be derived from another's.
    """
    raw_od = (raw_intensity if is_optical_density(raw_intensity)
              else mne.preprocessing.nirs.optical_density(raw_intensity.copy()))
    if picks is None:
        picks, _ = gvtd_channel_picks(raw_od, sep_bands)
    if picks and len(picks) < len(raw_od.ch_names):
        raw_od = raw_od.copy().pick(picks)
    sfreq = float(raw_od.info["sfreq"])
    od_data = np.nan_to_num(raw_od.get_data(), nan=0.0, posinf=0.0, neginf=0.0)
    gvtd_filt = gvtd_timetrace(od_data, sfreq, *GVTD_MOTION_BAND)
    thresh = gvtd_threshold(gvtd_filt, n_std=GVTD_N_STD)
    if thresh is None:
        return []
    return _mask_to_segments(gvtd_filt > thresh, raw_od.times[1:])


def gvtd_censor_spans(
    raw_od: mne.io.Raw,
    n_std: float = 10.0,
    min_epoch_s: float = 30.0,
    sep_bands=None,
) -> "tuple[list[tuple[float, float]], dict[str, Any]]":
    """Spans of a recording to censor on GVTD, and what censoring them costs.

    :footcite:`Sherafati2020` excludes the timepoints above the GVTD threshold from later
    analysis rather than repairing them, so this returns spans to mark rather than data to
    replace: nothing here interpolates, zero-fills or averages over what it flags.

    Two passes, and the second is why a threshold alone is not enough. First every sample
    above the threshold is flagged. Then any surviving stretch shorter than ``min_epoch_s``
    is flagged as well, since a four-second island between two artifacts is not something a
    spectral or connectivity analysis can use, and leaving it in makes the retained fraction
    look better than the retained data is.

    The rule is about every survivor, not only the islands between artifacts. On a 100 s
    recording with artifacts at 20-22 s and 26-28 s and ``min_epoch_s=30``, the 4 s island
    between them goes, and so does the 20 s head, which is also too short: spans
    ``[(0.0, 28.0)]`` and one surviving epoch of 72 s, where the threshold alone would have
    left three epochs.

    ``n_std`` defaults to the lenient 10 the reference used for censoring, not to the 3.0
    the reports score with. The two answer different questions: scoring asks how far a
    recording departed from its own resting level, censoring decides what to throw away, and
    on a recording that flags half its frames the strict value cascades through the
    ``min_epoch_s`` pass and censors everything.

    Returns
    -------
    spans : list of (onset, duration)
        In seconds, to attach as ``BAD_gvtd`` annotations.
    metrics : dict
        ``gvtd_censor_*``: the threshold used, the censored percentage, how many spans, how
        many surviving epochs and how many seconds they hold.

    References
    ----------
    .. footbibliography::
    """
    picks, picked_set = gvtd_channel_picks(raw_od, sep_bands)
    data = np.nan_to_num(raw_od.get_data(picks=picks), nan=0.0, posinf=0.0, neginf=0.0)
    times = raw_od.times
    sfreq = float(raw_od.info["sfreq"])
    empty = {
        "gvtd_censor_n_std": float(n_std), "gvtd_censor_min_epoch_s": float(min_epoch_s),
        "gvtd_censor_channel_set": picked_set, "gvtd_censor_thresh": None,
        "gvtd_censor_pct": None, "gvtd_censor_n_spans": None,
        "gvtd_censor_n_epochs": None, "gvtd_censor_retained_s": None,
    }
    gvtd = gvtd_timetrace(data, sfreq, *GVTD_MOTION_BAND)
    thresh = gvtd_threshold(gvtd, n_std)
    if thresh is None:
        logger.warning("GVTD has no positive values; nothing censored")
        return [], empty

    # gvtd is one sample shorter than times, the first sample having no derivative. The
    # reference prepends a zero there, which no threshold flags, so the mask starts False.
    flagged = np.concatenate(([False], gvtd > thresh))

    # second pass: a surviving stretch too short to analyse is censored with the artifacts
    if min_epoch_s > 0:
        for onset, dur in _mask_to_segments(~flagged, times):
            if dur < min_epoch_s:
                flagged[(times >= onset) & (times < onset + dur)] = True

    spans = _mask_to_segments(flagged, times)
    kept = _mask_to_segments(~flagged, times)
    metrics = {
        **empty,
        "gvtd_censor_thresh": float(thresh),
        "gvtd_censor_pct": float(np.mean(flagged)),
        "gvtd_censor_n_spans": len(spans),
        "gvtd_censor_n_epochs": len(kept),
        "gvtd_censor_retained_s": float(sum(d for _, d in kept)),
    }
    logger.info("GVTD censoring: %.1f%% of samples in %d span(s), %d epoch(s) of >= %.0fs "
                "left holding %.0fs", 100 * metrics["gvtd_censor_pct"], len(spans),
                len(kept), min_epoch_s, metrics["gvtd_censor_retained_s"])
    if not kept:
        logger.error("GVTD censoring leaves nothing: every sample is either above the "
                     "threshold (n_std=%s) or in a stretch shorter than %.0fs. The data is "
                     "marked, not deleted, so a larger --gvtd-censor-n-std or a smaller "
                     "--gvtd-min-epoch-s recovers it.", n_std, min_epoch_s)
    return spans, metrics


def _windowed_gvtd(
    raw_od: mne.io.Raw,
    window_s: float,
    l_freq: float | None,
    h_freq: float | None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """GVTD trace binned into non-overlapping windows. Returns (mean, p95, center_times).

    The window grid is the one the windowed SCI/PSP use, so all four series land on the same
    time axis: ``ceil`` samples per window, whole windows only, centres read off the real
    sample times. Deriving it independently drifts, because ``window_s * sfreq`` is rarely an
    integer and the rounding difference accumulates over the recording.
    """
    sfreq = float(raw_od.info["sfreq"])
    # per-sample GVTD trace, one value shorter than the recording (it is a difference)
    gvtd_ts = gvtd_timetrace(raw_od.get_data(), sfreq, l_freq=l_freq, h_freq=h_freq)
    n_times = len(raw_od.times)
    win_samples = max(1, int(np.ceil(window_s * sfreq)))
    # whole windows only, leftover tail dropped; capped so no window falls past the trace,
    # which only bites at a window of one sample and would otherwise give an all-NaN row
    n_windows = min(n_times // win_samples, -(-len(gvtd_ts) // win_samples))
    if n_windows == 0 or len(gvtd_ts) == 0:
        return np.array([]), np.array([]), np.array([])

    # NaN-pad rather than truncate: the last window is short by the sample GVTD does not have,
    # and dropping the whole window instead would put us back on a different grid
    padded = np.full(n_windows * win_samples, np.nan)
    usable = gvtd_ts[:n_windows * win_samples]
    padded[:len(usable)] = usable
    grid = padded.reshape(n_windows, win_samples)       # (window x sample-in-window)
    # mean = average motion level; p95 = worst-moment, so transient motion survives averaging
    gvtd_mean = np.nanmean(grid, axis=1)
    gvtd_p95 = np.nanpercentile(grid, 95, axis=1)

    starts = np.arange(n_windows) * win_samples
    ends = np.minimum(starts + win_samples, n_times - 1)
    window_times = (raw_od.times[starts] + raw_od.times[ends]) / 2.0
    return gvtd_mean, gvtd_p95, window_times


def compute_windowed_gvtd(
    raw_od: mne.io.Raw,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean and p95 GVTD per non-overlapping window, unfiltered.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical-density recording.
    window_s : float, optional
        Non-overlapping window length in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean, p95, center_times) per window: average and 95th-percentile GVTD in
        each window, and the window center times.
    """
    return _windowed_gvtd(raw_od, window_s, None, None)


def compute_windowed_filtered_gvtd(
    raw_od: mne.io.Raw,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean and p95 GVTD per window on the motion-band bandpassed OD.

    Same as :func:`compute_windowed_gvtd`, but the OD is first bandpassed to the
    motion band (``GVTD_MOTION_BAND``) to isolate head-motion frequencies.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical-density recording.
    window_s : float, optional
        Non-overlapping window length in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (mean, p95, center_times) per window.
    """
    return _windowed_gvtd(raw_od, window_s, *GVTD_MOTION_BAND)
