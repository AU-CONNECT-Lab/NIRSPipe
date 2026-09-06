"""Compute signal quality metrics (SQM) for fNIRS data.

Established metrics: SCI, PSP, CV, SNR, GVTD.
Experimental (may change or be removed): Cardiac Power (CP), per-chromophore gcor,
spike and motion-correction footprint, low-frequency drift.

Each compute_* function returns a flat dict and does no I/O. Assembling them into a
per-run record on disk is qc/sqm_record.py's job, not this module's.

Bad channels are handled by signal domain, not per function. The Beer-Lambert conversion
is the line:

    intensity / OD (compute_raw_sqm and its helpers, motion_correction_metrics)
        aggregates cover every channel, rejected ones included. A rejected channel is
        still part of what the machine recorded, and this is the archival view.
    haemoglobin (compute_haemo_sqm, compute_prep_haemo_sqm)
        aggregates exclude them, which is what mne.pick_types does by default. By this
        stage the channel is out of the analysis.

Do not "fix" an intensity/OD function to exclude bads. Doing so to _sci_metrics in
particular makes sci_mean an average over channels chosen for having good SCI, which
cannot fall below the threshold no matter how bad the recording is.

Per-channel dicts always list every channel, in both domains.
"""

import functools
from typing import Any

import mne
import numpy as np

from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.lineage import require_stage
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.quantitative_metrics")

# Hz, the band GVTD is measured on. :footcite:`Sherafati2020` computes GVTD on data already
# band-passed by the analysis, and measured the artifact-to-background ratio at four pipeline
# stages: it peaks after filtering, which is why this is applied at all. 0.5 is that paper's
# task low-pass, the fNIRS convention for dropping cardiac; 0.01 sits between its two
# high-passes (0.009 rest, 0.02 task). One band for every mode, not the analysis band of the
# moment: a QC number that moved with the mode could not be compared across a cohort.
GVTD_MOTION_BAND = (0.01, 0.5)

# The window is part of what PSP measures, not a smoothing setting, so it is pinned here
# rather than following the QC window. Lengthening it raises the score on a channel with a
# coherent cardiac component and lowers it on one without: measured over 10 s to 80 s, a
# well-coupled pair grew 8x while an uncoupled pair fell to a fifth. So it changes the
# spread between good and bad channels, and therefore what any fixed threshold selects.
# psp_mean averages over both kinds, so no factor converts one window's value to another's.
PSP_WINDOW_S = 10.0

# ---- CNR windows, relative to stimulus onset (s) ----
# The baseline is the pre-stimulus stretch the response has not reached yet; the response
# window brackets the canonical HRF peak, which fNIRS puts at 5-10 s with a plateau out to
# roughly 15 s for a sustained block. Both are fixed rather than derived from stimulus
# duration: CNR is only comparable across stages of one recording, and a window that moved
# with the design would make two runs' numbers incomparable for a reason unrelated to noise.
CNR_BASELINE_S = (-5.0, 0.0)
CNR_RESPONSE_S = (5.0, 15.0)

# ---- Source-detector separation, mne_nirs' convention ----
# These are the defaults of mne_nirs.channels.get_short_channels(max_dist=) and
# get_long_channels(min_dist=, max_dist=). Note they do not meet: 10-15 mm is neither
# short nor long. That gap is deliberate upstream. A 10-15 mm channel is too far to be
# reading scalp alone and too near to be reading cortex, so it belongs to neither view
# rather than being forced into one. 45 mm is the far edge: beyond it too little light
# returns for the channel to be worth averaging in.
# Consequence worth knowing: on a montage carrying either kind, the long and short
# channel sets do not add up to every channel, by design.
SHORT_MAX_DIST = 0.01   # m, <= this is a short channel
LONG_MIN_DIST  = 0.015  # m, >= this and <= LONG_MAX_DIST is a long channel
LONG_MAX_DIST  = 0.045  # m


# ------------------------------------ Shared helpers ------------------------------------
def long_short_channels(raw: mne.io.Raw) -> "tuple[list[str], list[str]]":
    """Split channel names by source-detector separation, returning ``(long, short)``.

    One definition for the whole package, so the report, the prep-raw figures and the SQM
    record agree on which channels are which. Because the two ranges do not meet (see
    SHORT_MAX_DIST / LONG_MIN_DIST), the two lists need not cover every channel::

        distances 8, 12, 30, 50 mm  ->  long ["30mm"], short ["8mm"]

    A montage with no registered optode positions reports every distance as zero, which
    would make every channel short; that case is logged and yields no split at all.

    Bad channels stay in both lists. Separation is the only thing being asked about, and
    pick_types drops bads by default, which would leave every metric computed from these
    lists averaging over channels that were selected for being good.
    """
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    names = [raw.ch_names[i] for i in picks]
    long_names  = [ch for ch, d in zip(names, dists) if LONG_MIN_DIST <= d <= LONG_MAX_DIST]
    short_names = [ch for ch, d in zip(names, dists) if 0 < d <= SHORT_MAX_DIST]
    if not long_names and not short_names:
        logger.warning("no channel falls in either separation range; optode positions "
                       "are probably missing")
    return long_names, short_names


def _mean_or_none(values) -> "float | None":
    """Mean of a collection of values, or None if it is empty."""
    vals = list(values)
    return float(np.mean(vals)) if vals else None


def _safe_metrics(label: str, keys):
    """Wrap a metric function so it always returns a dict keyed by ``keys``.

    ``keys`` declares the output schema once: the wrapped function starts from that
    schema (all None) and overlays whatever it computes, so a failure logs and leaves
    every key None instead of each function hand-writing an all-None fallback.

    A key ending in ``*`` declares a family whose members are only known at runtime, one
    per wavelength for instance. Those cannot be pre-filled — on failure there is no way
    to know which members would have existed — so the wildcard documents them and keeps
    them from reading as an undeclared key. Anything else the function returns that the
    schema does not mention is logged, because it is present on success and absent on
    failure, which is exactly what this decorator exists to prevent.
    """
    fixed = tuple(k for k in keys if not k.endswith("*"))
    families = tuple(k[:-1] for k in keys if k.endswith("*"))

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            base = dict.fromkeys(fixed)
            try:
                computed = fn(*args, **kwargs) or {}
            except Exception as exc:
                logger.warning("%s failed: %s", label, exc)
                return base
            undeclared = [k for k in computed
                          if k not in base and not k.startswith(families)]
            if undeclared:
                logger.warning("%s returned undeclared keys %s; they disappear when it "
                               "fails, so declare them (a '*' suffix marks a family)",
                               label, undeclared)
            base.update(computed)
            return base
        return wrap
    return deco


# ------------------------------- Motion primitives (GVTD) -------------------------------
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


# ------------------------------ Raw-intensity / OD metrics ------------------------------
def compute_sci_scores(
    raw: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> tuple[dict[str, float], mne.io.Raw]:
    r"""Scalp coupling index (SCI) per channel :footcite:`Pollonini2014`.

    Cardiac-band cross-correlation between the two wavelengths of each channel;
    low SCI flags poor optode-scalp coupling.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw intensity recording, or one already in optical density.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.

    Returns
    -------
    tuple[dict[str, float], mne.io.Raw]
        ({channel: SCI score}, optical-density recording). SCI defaults to 1.0
        for all channels on failure.

    Notes
    -----
    Thin wrapper over ``mne.preprocessing.nirs.scalp_coupling_index`` that adds
    what batch QC needs: it converts to optical density once and returns that OD
    object so later metrics can reuse it, reshapes the array into a per-channel
    dict, and is crash-safe: on failure every channel defaults to 1.0 (so no
    channel is wrongly dropped) instead of raising and aborting the whole run.

    Already-OD input is passed through rather than converted: recomputing scores from a
    tree on disk reads ``desc-sci`` or ``desc-od``, and ``optical_density`` raises on
    anything that is not continuous-wave amplitude.

    References
    ----------
    .. footbibliography::
    """
    raw_od = (raw if is_optical_density(raw)
              else mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False))
    try:
        sci_arr = mne.preprocessing.nirs.scalp_coupling_index(
            raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
        sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
    except Exception as exc:
        logger.warning("SCI failed: %s", exc)
        sci_scores = {ch: 1.0 for ch in raw.ch_names}
    return sci_scores, raw_od


def _sci_metrics(
    sci_scores: dict[str, float],
    bad_channels: list[str],
) -> dict[str, Any]:
    """SCI mean and per-channel scores, plus the fraction of channels retained."""
    n_total = len(sci_scores)
    return {
        "sci_mean": _mean_or_none(sci_scores.values()),
        "sci_per_channel": {k: float(v) for k, v in sci_scores.items()},
        "channel_retention_rate": (
            float((n_total - len(bad_channels)) / n_total) if n_total > 0 else None
        ),
    }


def channel_cv(data: np.ndarray) -> np.ndarray:
    r"""Coefficient of variation per channel: relative noise level (lower = cleaner).

    .. math::

        \mathrm{CV} = \frac{\sigma}{\mu}

    NaN for channels whose mean is 0.
    """
    mu = data.mean(axis=1)
    sigma = data.std(axis=1)
    return np.divide(sigma, mu, out=np.full_like(sigma, np.nan), where=mu != 0)


def channel_snr(data: np.ndarray) -> np.ndarray:
    r"""Signal-to-noise ratio per channel, the reciprocal of CV (higher = better).

    .. math::

        \mathrm{SNR} = \frac{\mu}{\sigma}

    NaN for channels whose std is 0.
    """
    mu = data.mean(axis=1)
    sigma = data.std(axis=1)
    return np.divide(mu, sigma, out=np.full_like(mu, np.nan), where=sigma > 0)


@_safe_metrics("CV/SNR", (
    "cv_mean", "cv_mean_*", "cv_per_channel",     # cv_mean_* is one key per wavelength
    "snr_mean", "snr_per_channel", "snr_pass_rate", "n_flat_channels",
    "mean_amp_mean", "mean_amp_per_channel",
))
def _intensity_metrics(raw_intensity: mne.io.Raw, snr_threshold: float = 2.0) -> dict[str, Any]:
    """Per-channel CV, SNR and mean amplitude from raw intensity (with means; CV also per wavelength).

    snr_pass_rate is the fraction of channels with SNR > snr_threshold (default 2.0,
    a common channel-pruning cutoff; SNR = mean/std, higher is better). Its denominator is
    every channel, not every channel with a finite SNR: a flat or saturated channel has
    std 0 and no finite SNR at all, and letting it drop out of the denominator would mean a
    recording whose channels are dying reads as a recording whose channels are passing.
    n_flat_channels is how many those were. The means are still taken over the finite
    values, since an average cannot carry a NaN.
    """
    int_data = raw_intensity.get_data()
    names = raw_intensity.ch_names
    cv = channel_cv(int_data)
    snr = channel_snr(int_data)
    mean_amp = int_data.mean(axis=1)
    cv_per_ch = {ch: float(cv[i]) for i, ch in enumerate(names) if np.isfinite(cv[i])}
    snr_per_ch = {ch: float(snr[i]) for i, ch in enumerate(names) if np.isfinite(snr[i])}
    mean_amp_per_ch = {ch: float(mean_amp[i]) for i, ch in enumerate(names)}
    # group per-channel CV by wavelength (last token of the channel name)
    wl_groups: dict[str, list[float]] = {}
    for ch, cv_val in cv_per_ch.items():
        wl_groups.setdefault(ch.split()[-1], []).append(cv_val)
    cv_mean_per_wl = {
        f"cv_mean_{wl}": float(np.mean(vals))
        for wl, vals in sorted(wl_groups.items())
    }
    return {
        "cv_mean": _mean_or_none(cv_per_ch.values()),
        **cv_mean_per_wl,
        "cv_per_channel": cv_per_ch,
        "snr_mean": _mean_or_none(snr_per_ch.values()),
        "snr_per_channel": snr_per_ch,
        "snr_pass_rate": (
            float(sum(v > snr_threshold for v in snr_per_ch.values()) / len(names))
            if names else None
        ),
        "n_flat_channels": len(names) - len(snr_per_ch),
        "mean_amp_mean": _mean_or_none(mean_amp_per_ch.values()),
        "mean_amp_per_channel": mean_amp_per_ch,
    }


@_safe_metrics("Channel distance", (
    "ch_dist_mean", "ch_dist_min", "ch_dist_max", "ch_dist_per_channel",
))
def _channel_distance_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    """Source-detector separation per channel, with mean/min/max (metres)."""
    dists = mne.preprocessing.nirs.source_detector_distances(raw_intensity.info)
    dist_per_ch = {
        ch: float(dists[i])
        for i, ch in enumerate(raw_intensity.ch_names)
    }
    return {
        "ch_dist_mean": float(np.mean(dists)),
        "ch_dist_min": float(np.min(dists)),
        "ch_dist_max": float(np.max(dists)),
        "ch_dist_per_channel": dist_per_ch,
    }


@_safe_metrics("PSP", ("psp_mean", "psp_per_channel"))
def _psp_metrics(
    raw_intensity: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, Any]:
    """Peak spectral power per channel, averaged over ``PSP_WINDOW_S`` windows, with mean.

    Measured on optical density, which is what ``peak_power`` is meant for: the metric
    cross-correlates the two wavelengths, so it has no meaning after Beer-Lambert, and
    mne_nirs' own test converts to OD before calling it. Its docstring saying
    "haemoglobin data" is a copy-paste slip shared with ``scalp_coupling_index_windowed``.
    Converting here keeps this agreeing with the windowed PSP series, which is handed OD.

    The window is pinned to ``PSP_WINDOW_S`` rather than following the QC window length; see
    the constant for what changes when it moves. The windowed PSP series is a separate view
    and does follow the QC window, so its colour scale is not on this scalar's scale.
    """
    import mne_nirs.preprocessing as nirs_prep
    raw_od = (raw_intensity if is_optical_density(raw_intensity)
              else mne.preprocessing.nirs.optical_density(raw_intensity.copy()))
    _, psp_scores, _ = nirs_prep.peak_power(
        raw_od.copy(), time_window=PSP_WINDOW_S,
        l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
    psp_per_ch = {
        ch: float(np.mean(psp_scores[i]))
        for i, ch in enumerate(raw_od.ch_names)
    }
    return {
        "psp_mean": _mean_or_none(psp_per_ch.values()),
        "psp_per_channel": psp_per_ch,
    }


@_safe_metrics("Cardiac Power", ("cp_mean", "cp_per_channel", "cp_pass_rate"))
def _cardiac_power_metrics(
    raw: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    cp_threshold: float = 0.5,
) -> dict[str, Any]:
    r"""Cardiac Power (CP): within-band power concentrated at the per-channel cardiac peak :footcite:`Bizzego2022`.

    .. math::

        \text{CP} = \frac{P(f_c \pm 0.2\,\text{Hz})}{P(f_c \pm 0.5\,\text{Hz})},

    where :math:`f_c` is the peak frequency in ``[cardiac_l_freq, cardiac_h_freq]``.
    Experimental: overlaps PSP and its 0.5 gate is calibrated on the 0.83-2.5 Hz
    band; may be removed. SCI + PSP are the primary cardiac quality metrics.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw intensity or optical-density recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    cp_threshold : float, optional
        Good-quality gate (CP >= 0.5 in the reference band).

    Returns
    -------
    dict
        cp_mean, cp_per_channel and cp_pass_rate.

    References
    ----------
    .. footbibliography::
    """
    # CP is defined in the OD domain (aligns with SCI/PSP); convert unless input is already OD
    raw_od = raw if is_optical_density(raw) else mne.preprocessing.nirs.optical_density(raw.copy())
    fmax = min(cardiac_h_freq, raw_od.info["sfreq"] / 2)
    # PSD restricted to the cardiac band, so the ±0.2/±0.5 windows below are auto-clipped to it
    # (equivalent to Bizzego's pre-bandpass; keeps respiration/Mayer power out of the ratio)
    psd = raw_od.compute_psd(fmin=cardiac_l_freq, fmax=fmax, verbose=False)
    freqs = psd.freqs
    # exclude=() keeps the bad channels: this is a raw-domain metric, so it describes the
    # recording as it arrived. It also keeps get_data() aligned with ch_names, which the
    # default does not: get_data() drops bads while ch_names keeps them, and indexing one
    # by the other reads the wrong channel and then runs off the end
    psd_data = psd.get_data(exclude=())
    if freqs.size == 0:
        raise ValueError("no frequencies in cardiac band")
    cp_per_ch: dict[str, float | None] = {}
    for i, ch in enumerate(psd.ch_names):
        ch_psd = psd_data[i]
        fc = freqs[np.argmax(ch_psd)]
        narrow = ch_psd[(freqs >= fc - 0.2) & (freqs <= fc + 0.2)]
        wide = ch_psd[(freqs >= fc - 0.5) & (freqs <= fc + 0.5)]
        # sum = integrated band power (not mean); narrow ⊂ wide gives CP ∈ [0,1], matching the CP≥0.5 gate
        wide_p = float(wide.sum())
        cp_per_ch[ch] = (
            float(narrow.sum() / wide_p)
            if narrow.size > 0 and wide_p > 0
            else None
        )
    valid = [v for v in cp_per_ch.values() if v is not None]
    return {
        "cp_mean": _mean_or_none(valid),
        "cp_per_channel": cp_per_ch,
        "cp_pass_rate": _mean_or_none([v >= cp_threshold for v in valid]),
    }


def _spike_mask(diff_data: np.ndarray) -> np.ndarray:
    """Per-channel robust outlier mask of a temporal-derivative array.

    Threshold uses MAD, not std: std is taken over the whole derivative *including the
    spikes*, so a few large spikes inflate std -> threshold too high -> real spikes fall
    under it (non-robust). MAD (median abs deviation) resists those outliers.
      thresh = median + 3 * 1.4826 * MAD ;  flag where |diff - median| > thresh
    (1.4826*MAD ~= sigma for Gaussian data, so this is a robust 3-sigma). Per-channel
    scale also stops high-dynamic-range channels from dominating.
    """
    med = np.median(diff_data, axis=1, keepdims=True)
    mad = np.median(np.abs(diff_data - med), axis=1, keepdims=True)
    return np.abs(diff_data - med) > (3.0 * 1.4826 * mad)


def _motion_band_diff(od_data: np.ndarray, sfreq: float) -> np.ndarray:
    """Temporal derivative of OD band-limited to the motion band (cardiac removed first).

    EEG detects spikes on band-limited data (line-noise/muscle filtered out first); the
    fNIRS analog filters out the ~1 Hz cardiac band so the derivative reflects motion,
    not pulsation. Uses the same GVTD motion band.
    """
    d = np.nan_to_num(od_data, nan=0.0, posinf=0.0, neginf=0.0)
    h_freq = GVTD_MOTION_BAND[1]
    if h_freq >= sfreq / 2:  # not a valid IIR cutoff at/above Nyquist
        h_freq = None
    d = mne.filter.filter_data(
        d, sfreq, GVTD_MOTION_BAND[0], h_freq, method="iir",
        iir_params=dict(order=4, ftype="butter"), verbose=False)
    return np.diff(d, axis=1)


@_safe_metrics("Spike metrics", (
    "spike_count", "spike_pct", "spike_num_frames", "spike_pct_frames",
    "temporal_derivative_variance",
))
def _spike_metrics(raw_intensity: mne.io.Raw, ch_frac: float = 0.1) -> dict[str, Any]:
    """Spike diagnostics and temporal-derivative variance from the OD derivative.

    Parameters
    ----------
    raw_intensity : mne.io.Raw
        Raw intensity recording, or one already in optical density.
    ch_frac : float, optional
        Fraction of channels that must spike at a timepoint for it to count as a spike
        frame (the frame-level / FD-style aggregation).

    Returns
    -------
    dict
        spike_count (total outliers), spike_pct (fraction of all channel-samples that are
        outliers, an outlier-ratio), spike_num_frames / spike_pct_frames (timepoints with
        >= ch_frac of channels spiking), and temporal_derivative_variance (per channel).

    Notes
    -----
    Experimental. Spikes are detected on the motion-band-filtered OD derivative (cardiac
    removed first, EEG-style filter-then-detect), so they reflect motion rather than
    pulsation; a per-channel MAD 3-sigma outlier count, distinct from the global GVTD
    threshold. temporal_derivative_variance uses the *unfiltered* derivative (per-channel
    derivative energy, the squared DVARS-vstd normaliser) for flagging noisy channels,
    not a standard named metric and not motion detection.
    """
    raw_od = (raw_intensity if is_optical_density(raw_intensity)
              else mne.preprocessing.nirs.optical_density(raw_intensity.copy()))
    od_data = np.nan_to_num(raw_od.get_data(), nan=0.0, posinf=0.0, neginf=0.0)
    diff_raw = np.diff(od_data, axis=1)  # unfiltered: for the per-channel derivative energy
    spikes = _spike_mask(_motion_band_diff(od_data, float(raw_od.info["sfreq"])))
    flagged = spikes.mean(axis=0) >= ch_frac  # timepoints with >= ch_frac channels spiking
    return {
        "spike_count": int(spikes.sum()),
        "spike_pct": float(spikes.mean()) if spikes.size else None,
        "spike_num_frames": int(flagged.sum()),
        "spike_pct_frames": float(flagged.mean()) if flagged.size else None,
        "temporal_derivative_variance": {
            raw_od.ch_names[i]: float(np.var(diff_raw[i]))
            for i in range(len(raw_od.ch_names))
        },
    }


@_safe_metrics("GVTD metrics", (
    "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95",
    "gvtd_vstd_mean", "gvtd_vstd_p95",
    "gvtd_thresh", "gvtd_num_above_thresh", "gvtd_pct_above_thresh",
))
def _motion_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    """GVTD motion metrics from OD: mean/p95, motion-band and vstd variants, and threshold.

    Parameters
    ----------
    raw_intensity : mne.io.Raw
        Raw intensity recording, or one already in optical density.

    Returns
    -------
    dict
        gvtd_mean/p95 (canonical), gvtd_filt_* (motion-band), gvtd_vstd_*
        (channel-standardized), gvtd_thresh, and the num/pct of timepoints above it.

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
    motion_thresh = gvtd_threshold(gvtd_filt, n_std=3.0)
    if motion_thresh is not None:
        above = gvtd_filt > motion_thresh
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
        "gvtd_thresh": motion_thresh,
        "gvtd_num_above_thresh": num_above,
        "gvtd_pct_above_thresh": pct_above,
    }


def _correction_footprint(
    raw_before: mne.io.Raw,
    raw_after: mne.io.Raw,
    rel_thresh: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Find which timepoints a motion correction actually touched, per channel.

    Returns a (channels x times) bool mask (True = this channel was corrected at that
    sample) plus the matching time axis. Flags where the correction changes abruptly,
    i.e. the frame-to-frame rate of correction ``|diff(after - before)|`` exceeds
    ``rel_thresh`` times the channel's sample-to-sample noise (1.4826*MAD of
    ``diff(before)``). This is the FD/censoring analog (mark the moments a repair
    happens), so a continuous corrector like TDDR flags the motion events rather than
    the whole recording. Flat channels contribute nothing.
    """
    before = np.nan_to_num(raw_before.get_data())
    after = np.nan_to_num(raw_after.get_data(picks=raw_before.ch_names))
    d_corr = np.diff(after - before, axis=1)  # rate of correction (frame-to-frame)
    d_before = np.diff(before, axis=1)
    med = np.median(d_corr, axis=1, keepdims=True)
    noise = 1.4826 * np.median(  # channel's sample-to-sample noise
        np.abs(d_before - np.median(d_before, axis=1, keepdims=True)), axis=1, keepdims=True)
    corrected = np.abs(d_corr - med) > (rel_thresh * noise)
    corrected[noise[:, 0] == 0] = False  # flat channel → no reference scale
    return corrected, raw_before.times[1:]


def _mask_to_segments(flagged: np.ndarray, times: np.ndarray) -> "list[tuple[float, float]]":
    """Collapse a per-sample bool mask into (onset, duration) time spans.

    Given which timepoints are flagged (motion / corrected / spike), return each
    contiguous run of True as a time interval, used to draw shaded bands on figures.

    Example: mask [F,T,T,F,T] at 1 Hz -> [(1.0, 2.0), (4.0, 0.0)].
    """
    if not flagged.any():
        return []
    # diff marks the edges: a 0->1 step opens a run, a 1->0 step closes it
    edges = np.diff(flagged.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)   # +1 undoes the one-sample offset diff introduces
    ends = list(np.where(edges == -1)[0] + 1)
    # a run touching the very start / end has no edge to detect, so add that boundary by hand
    if flagged[0]:
        starts.insert(0, 0)
    if flagged[-1]:
        ends.append(len(flagged) - 1)
    return [(float(times[s]), max(float(times[e]) - float(times[s]), 0.0)) for s, e in zip(starts, ends)]


@_safe_metrics("motion correction footprint", (
    "motion_corrected_frac_mean", "motion_corrected_frac_per_channel",
    "motion_corrected_num", "motion_corrected_pct", "motion_corrected_n_segments",
))
def motion_correction_metrics(
    raw_before: mne.io.Raw,
    raw_after: mne.io.Raw,
    rel_thresh: float = 1.0,
    ch_frac: float = 0.1,
) -> dict[str, Any]:
    """Experimental: motion-correction footprint — which timepoints a correction repaired.

    Per channel a sample is a correction event when the frame-to-frame rate of correction
    exceeds ``rel_thresh`` times the channel's sample-to-sample noise. A timepoint is
    flagged globally when at least ``ch_frac`` of channels have an event there, giving
    FD-style scrubbing scalars (num/pct/segments) on top of the per-channel fractions.

    Parameters
    ----------
    raw_before, raw_after : mne.io.Raw
        Optical density immediately before and after the motion-correction step.
    rel_thresh : float, optional
        Correction rate, as a multiple of the channel's sample-to-sample noise, above
        which a sample counts as a correction event.
    ch_frac : float, optional
        Fraction of channels that must be corrected at a timepoint for it to count as a
        globally corrected (scrubbed) timepoint.

    Returns
    -------
    dict
        motion_corrected_frac_mean / _frac_per_channel (per-channel burden) and the
        FD-analog global scalars motion_corrected_num / _pct / _n_segments.

    Notes
    -----
    Experimental and correction-agnostic: it measures the footprint of whatever
    correction ran (e.g. TDDR repairs only the < 0.5 Hz component), not a new
    correction. Overlaps GVTD (both track motion); frame it as correction burden.
    """
    corrected, times = _correction_footprint(raw_before, raw_after, rel_thresh)
    frac_per_ch = {ch: float(corrected[i].mean()) for i, ch in enumerate(raw_before.ch_names)}
    flagged = corrected.mean(axis=0) >= ch_frac  # timepoint corrected across >= ch_frac of channels
    return {
        "motion_corrected_frac_mean": _mean_or_none(frac_per_ch.values()),
        "motion_corrected_frac_per_channel": frac_per_ch,
        "motion_corrected_num": int(flagged.sum()),
        "motion_corrected_pct": float(flagged.mean()) if flagged.size else None,
        "motion_corrected_n_segments": len(_mask_to_segments(flagged, times)),
    }


def motion_corrected_segments(
    raw_before: mne.io.Raw,
    raw_after: mne.io.Raw,
    rel_thresh: float = 1.0,
    ch_frac: float = 0.1,
) -> "list[tuple[float, float]]":
    """Time spans where motion correction touched >= ch_frac of channels (for plotting)."""
    corrected, times = _correction_footprint(raw_before, raw_after, rel_thresh)
    flagged = corrected.mean(axis=0) >= ch_frac
    return _mask_to_segments(flagged, times)


def spike_segments(raw_intensity: mne.io.Raw, ch_frac: float = 0.1) -> "list[tuple[float, float]]":
    """Time spans where >= ch_frac of channels show a motion-band OD spike (for plotting)."""
    raw_od = (raw_intensity if is_optical_density(raw_intensity)
              else mne.preprocessing.nirs.optical_density(raw_intensity.copy()))
    diff_data = _motion_band_diff(raw_od.get_data(), float(raw_od.info["sfreq"]))
    flagged = _spike_mask(diff_data).mean(axis=0) >= ch_frac
    return _mask_to_segments(flagged, raw_od.times[1:])


# --------------------------------- Haemoglobin metrics ----------------------------------
def haemo_quality_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    r"""HbO-HbR correlation per source-detector pair (genuine responses anti-correlate).

    Each HbO channel is paired with the HbR channel of the same source-detector
    pair, and their Pearson correlation is taken:

    .. math::

        \rho = \operatorname{corr}(\text{HbO}, \text{HbR}), \qquad \rho \in [-1, 1].

    A real haemodynamic response drives HbO up and HbR down, so a strongly
    negative correlation (near -1) is expected; values near 0 or positive indicate
    a shared artifact (motion, systemic scalp signal) rather than brain activity.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Haemoglobin recording (HbO/HbR).

    Returns
    -------
    dict
        hbo_hbr_corr_mean and hbo_hbr_corr_per_channel (keyed by source-detector pair).

    Notes
    -----
    The HbO-HbR anti-correlation is well-established physiology; using it as a QC
    metric is a reasonable heuristic rather than a standardized threshold.
    """
    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    hbo_data = raw_haemo.get_data(picks=hbo_picks)
    hbr_data = raw_haemo.get_data(picks=hbr_picks)
    hbo_names = [raw_haemo.ch_names[i] for i in hbo_picks]
    hbr_names = [raw_haemo.ch_names[i] for i in hbr_picks]

    # key on the source-detector pair (drop the chromophore token) to pair HbO with its HbR
    hbo_map = {n.rsplit(" ", 1)[0]: hbo_data[i] for i, n in enumerate(hbo_names)}
    hbr_map = {n.rsplit(" ", 1)[0]: hbr_data[i] for i, n in enumerate(hbr_names)}
    corr_per_ch = {
        key: float(np.corrcoef(hbo_map[key], hbr_map[key])[0, 1])
        for key in hbo_map if key in hbr_map
    }
    return {
        "hbo_hbr_corr_mean": _mean_or_none(corr_per_ch.values()),
        "hbo_hbr_corr_per_channel": corr_per_ch,
    }


@_safe_metrics("CNR", ("cnr_hbo_mean", "cnr_hbr_mean", "cnr_per_channel", "cnr_n_epochs"))
def _cnr_metrics(
    raw_haemo: mne.io.Raw,
    baseline: "tuple[float, float]" = CNR_BASELINE_S,
    response: "tuple[float, float]" = CNR_RESPONSE_S,
) -> dict[str, Any]:
    r"""Contrast-to-noise ratio per channel: how far the evoked response clears its own noise.

    .. math::

        \mathrm{CNR} = \frac{\mu_\text{resp} - \mu_\text{base}}
                            {\sqrt{\sigma^2_\text{resp} + \sigma^2_\text{base}}},

    taken per epoch on the channel's own samples and then averaged over epochs. HbO rises
    and HbR falls with a genuine response, so HbO CNR is positive and HbR CNR negative;
    the two means are reported separately for that reason, as ``gcor`` is.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Haemoglobin recording carrying stimulus annotations.
    baseline, response : tuple[float, float]
        Window edges relative to onset, in seconds.

    Returns
    -------
    dict
        cnr_hbo_mean, cnr_hbr_mean, cnr_per_channel and cnr_n_epochs; all None on a
        recording with no stimulus annotations, which is what a resting run looks like.

    Notes
    -----
    This is the one signal-quality measure that survives a comparison across the bandpass.
    Anything built from band power cannot: the filter removes the out-of-band term by
    construction, so the ratio improves whatever the data did. CNR can move either way,
    because a filter or a regression that eats the response shrinks the numerator at the
    same time as the denominator, which is exactly the failure worth seeing.

    ``BAD_`` annotations are censoring marks rather than stimuli and are excluded from the
    event set; epochs overlapping them are dropped by ``reject_by_annotation``. Bad channels
    are excluded, following ``mne.pick_types``.
    """
    from fnirs_pipe.qc.figures._utils import epochable_events

    events, event_id = epochable_events(raw_haemo, baseline[0], response[1])
    if len(events) == 0:
        return {}
    picks = mne.pick_types(raw_haemo.info, fnirs=True)
    epochs = mne.Epochs(
        raw_haemo, events, event_id, tmin=baseline[0], tmax=response[1],
        picks=picks, baseline=None, preload=True, reject_by_annotation=True, verbose=False,
    )
    if len(epochs) == 0:
        return {}

    times = epochs.times
    base_mask = (times >= baseline[0]) & (times <= baseline[1])
    resp_mask = (times >= response[0]) & (times <= response[1])
    if not base_mask.any() or not resp_mask.any():
        return {}

    data = epochs.get_data(copy=False)          # epoch x channel x time
    base, resp = data[:, :, base_mask], data[:, :, resp_mask]
    contrast = resp.mean(axis=2) - base.mean(axis=2)
    noise = np.sqrt(resp.var(axis=2) + base.var(axis=2))
    cnr = np.divide(contrast, noise, out=np.full_like(contrast, np.nan), where=noise > 0)

    per_ch = {name: float(v)
              for name, v in zip(epochs.ch_names, np.nanmean(cnr, axis=0))
              if np.isfinite(v)}
    return {
        "cnr_hbo_mean": _mean_or_none([v for k, v in per_ch.items() if k.endswith(" hbo")]),
        "cnr_hbr_mean": _mean_or_none([v for k, v in per_ch.items() if k.endswith(" hbr")]),
        "cnr_per_channel": per_ch,
        "cnr_n_epochs": int(len(epochs)),
    }


@_safe_metrics("PSD metrics", (
    "cardiac_band_power_hbo", "cardiac_band_power_hbr",
    "cardiac_band_frac_hbo", "cardiac_band_frac_hbr",
    "resp_band_power_hbo", "resp_band_power_hbr",
    "resp_band_frac_hbo", "resp_band_frac_hbr",
))
def _spectral_metrics(
    raw_haemo: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
) -> dict[str, Any]:
    r"""Cardiac/respiration power in the haemoglobin PSD, per chromophore, absolute and fractional.

    ``*_band_power`` = mean PSD in the band (absolute; scales with overall signal
    amplitude). ``*_band_frac`` = band power / total spectral power, an fALFF-style
    fraction in :math:`[0, 1]` comparable across subjects/channels regardless of
    amplitude. HbO and HbR sit on different amplitude scales, so each is summarised on its
    own (a pooled band_power would be HbO-dominated and a pooled total would skew band_frac).

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Haemoglobin recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    resp_l_freq, resp_h_freq : float
        Respiration band edges in Hz.

    Returns
    -------
    dict
        cardiac/resp band_power and band_frac, each split into _hbo and _hbr.
    """
    psd = raw_haemo.compute_psd(verbose=False)
    freqs = psd.freqs

    def _bands(chroma: str) -> dict[str, float | None]:
        empty = {"cp": None, "cf": None, "rp": None, "rf": None}
        if chroma not in raw_haemo.get_channel_types():
            return empty
        # pick from the spectrum, not via raw_haemo.info: get_data() drops bad channels,
        # so info-derived indices address a longer list and run off the end
        chrom = psd.get_data(picks=chroma)
        if not chrom.size:
            return empty
        total = float(chrom.sum())  # total power over this chromophore's channels and freqs

        def _power(fmin: float, fmax: float) -> float | None:
            # absolute: mean PSD density inside the band
            mask = (freqs >= fmin) & (freqs <= fmax)
            return float(chrom[:, mask].mean()) if mask.any() else None

        def _frac(fmin: float, fmax: float) -> float | None:
            # relative: fraction of this chromophore's total power in the band (sum/sum, in [0,1])
            mask = (freqs >= fmin) & (freqs <= fmax)
            return float(chrom[:, mask].sum() / total) if (mask.any() and total > 0) else None

        return {
            "cp": _power(cardiac_l_freq, cardiac_h_freq),
            "cf": _frac(cardiac_l_freq, cardiac_h_freq),
            "rp": _power(resp_l_freq, resp_h_freq),
            "rf": _frac(resp_l_freq, resp_h_freq),
        }

    hbo = _bands("hbo")
    hbr = _bands("hbr")
    return {
        "cardiac_band_power_hbo": hbo["cp"], "cardiac_band_power_hbr": hbr["cp"],
        "cardiac_band_frac_hbo":  hbo["cf"], "cardiac_band_frac_hbr":  hbr["cf"],
        "resp_band_power_hbo":    hbo["rp"], "resp_band_power_hbr":    hbr["rp"],
        "resp_band_frac_hbo":     hbo["rf"], "resp_band_frac_hbr":     hbr["rf"],
    }


def _gcor(data: np.ndarray) -> "float | None":
    r"""Global correlation (GCOR): mean of all pairwise channel correlations :footcite:`Saad2013`.

    .. math::

        \text{GCOR} = \mathbf{g}^\top \mathbf{g} = \lVert \mathbf{g} \rVert^2,

    where each channel is demeaned and unit-L2-normalised, then averaged into
    :math:`\mathbf{g}`. High = channels move together (global artifact / systemic
    physiology).

    Parameters
    ----------
    data : np.ndarray
        Channel-by-time array.

    Returns
    -------
    float or None
        GCOR, or None if fewer than two channels.

    Notes
    -----
    The GCOR statistic itself is standard, but using it per chromophore as an fNIRS
    QC metric is experimental and may be removed.

    References
    ----------
    .. footbibliography::
    """
    if data.shape[0] < 2:
        return None
    x = data - data.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    g = x.mean(axis=0)
    return float(g @ g)


@_safe_metrics("gcor", ("gcor_hbo", "gcor_hbr"))
def gcor_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    # Per chromophore: HbO and HbR anti-correlate, so a mixed gcor would cancel to ~0.
    hbo = raw_haemo.get_data(picks=mne.pick_types(raw_haemo.info, fnirs="hbo"))
    hbr = raw_haemo.get_data(picks=mne.pick_types(raw_haemo.info, fnirs="hbr"))
    return {"gcor_hbo": _gcor(hbo), "gcor_hbr": _gcor(hbr)}


@_safe_metrics("Drift amplitude", ("lowfreq_drift_amplitude_hbo", "lowfreq_drift_amplitude_hbr"))
def _drift_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    r"""Low-frequency baseline drift amplitude per chromophore (peak-to-peak of a slow trend).

    A low-order (cubic) polynomial trend is fitted per channel; drift is the mean
    peak-to-peak of that trend.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Haemoglobin recording.

    Returns
    -------
    dict
        lowfreq_drift_amplitude_hbo and _hbr.

    Notes
    -----
    Non-standard homegrown metric; may be removed. A polynomial is used instead of
    a 0.01 Hz low-pass, whose FIR length would exceed most recordings.
    """
    # NOTE: non-standard homegrown metric; may remove.
    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    n = len(raw_haemo.times)
    # Low-frequency drift amplitude = peak-to-peak of a slow trend fitted per channel.
    # We fit a low-order (cubic) polynomial rather than low-passing at 0.01 Hz: an
    # 0.01 Hz FIR needs a filter ~hundreds of seconds long (roughly several / 0.01),
    # which exceeds most recordings -> MNE errors, or leaves heavy edge ringing that
    # corrupts the ptp. The polynomial captures the same slow drift with no filter.
    #   trend = V @ lstsq(V, x),  V = [t^3 t^2 t 1] ;  drift = ptp(trend) mean over channels
    t = np.linspace(-1.0, 1.0, n)
    vander = np.vander(t, 4)

    def _drift_ptp(picks) -> "float | None":
        if not len(picks):
            return None
        data = raw_haemo.get_data(picks=picks)
        coef, *_ = np.linalg.lstsq(vander, data.T, rcond=None)
        trend = (vander @ coef).T
        return float(np.ptp(trend, axis=1).mean())

    return {
        "lowfreq_drift_amplitude_hbo": _drift_ptp(hbo_picks),
        "lowfreq_drift_amplitude_hbr": _drift_ptp(hbr_picks),
    }


@_safe_metrics("pct_data_retained", ("pct_data_retained",))
def _retention_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    """Fraction of the recording not covered by BAD annotations (overlaps merged)."""
    total_dur = raw_haemo.times[-1] - raw_haemo.times[0]
    bad_spans = sorted(
        (ann["onset"], ann["onset"] + ann["duration"])
        for ann in raw_haemo.annotations
        if ann["description"].upper().startswith("BAD")
    )
    # merge overlapping BAD spans so overlap is not double-counted
    bad_dur = 0.0
    cur_start = cur_end = None
    for start, end in bad_spans:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                bad_dur += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        bad_dur += cur_end - cur_start
    bad_dur = min(bad_dur, total_dur)
    return {
        "pct_data_retained": float(1.0 - bad_dur / total_dur) if total_dur > 0 else None
    }


# -------------------------------- Sliding-window series ---------------------------------
def compute_windowed_sci(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """SCI per channel in each non-overlapping window.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical-density recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    window_s : float, optional
        Non-overlapping window length in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, times): SCI per channel per window (channel x window), and each
        window's [start, end] times.

    Notes
    -----
    Wrapper over mne_nirs ``scalp_coupling_index_windowed``; used to see how
    coupling drifts over the recording rather than as a single whole-run value.
    """
    from mne_nirs.preprocessing import scalp_coupling_index_windowed
    _, scores, times = scalp_coupling_index_windowed(
        raw_od, time_window=window_s, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times


def compute_windowed_psp(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """Peak spectral power per channel in each non-overlapping window.

    Parameters
    ----------
    raw_od : mne.io.Raw
        Optical-density recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    window_s : float, optional
        Non-overlapping window length in seconds.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (scores, times): PSP per channel per window, and each window's
        [start, end] times.

    Notes
    -----
    Wrapper over mne_nirs ``peak_power``; the windowed companion to the whole-run
    PSP, showing how the cardiac peak strength varies over time.
    """
    from mne_nirs.preprocessing import peak_power
    _, scores, times = peak_power(
        raw_od, time_window=window_s, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times


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


def attach_windowed_series(
    sqm: dict,
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
    *,
    gvtd_od: "mne.io.Raw | None" = None,
) -> dict:
    """Compute sliding-window SCI/PSP/GVTD series, attach summaries to sqm, return raw series.

    Parameters
    ----------
    sqm : dict
        Metric dict; per-window summaries are attached to it in place.
    raw_od : mne.io.Raw
        Optical-density recording the SCI and PSP series are read off.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    window_s : float, optional
        Non-overlapping window length in seconds.
    gvtd_od : mne.io.Raw or None, optional
        A second optical-density recording for the GVTD series alone; ``raw_od`` when None.

    Returns
    -------
    dict
        Raw series (sci_matrix/sci_times/psp_matrix/psp_times) for callers that
        also plot them; all None on failure.

    Notes
    -----
    SCI/PSP and GVTD fail independently; whichever survives is still attached. Center
    times collapse the mne-nirs [start, end] window pairs to their midpoint. ``window_s``
    is stored as ``qc_window_s`` so a record says which grid its series were binned on;
    ``psp_mean`` is deliberately not on that grid, see :data:`PSP_WINDOW_S`.

    ``gvtd_od`` exists because the two families want different stages. SCI and PSP measure
    optode coupling, which the motion correction is not supposed to change, so they belong
    on the uncorrected file where the per-channel scores were taken. GVTD measures movement,
    which the correction is entirely about, so it belongs on the corrected one. Both grids
    are derived from ``window_s`` the same way, so the series stay time-aligned across the
    two files as long as neither was resampled.
    """
    def _center_times(t):
        a = np.asarray(t)
        return (a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a).tolist()

    series = {"sci_matrix": None, "sci_times": None, "psp_matrix": None, "psp_times": None}
    # recorded even when every series below fails: it describes the request, not the result,
    # and without it a stored series cannot be told apart from one binned at another length
    sqm["qc_window_s"] = float(window_s)

    # two try blocks, not one: SCI/PSP filter to the cardiac band and GVTD does not, so a band
    # that the filter rejects must not take the motion series down with it
    sci_matrix = sci_times = psp_matrix = psp_times = None
    try:
        sci_matrix, sci_times = compute_windowed_sci(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
        psp_matrix, psp_times = compute_windowed_psp(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    except Exception as exc:
        logger.warning("windowed SCI/PSP failed: %s", exc)

    gvtd_per_window = gvtd_p95_per_window = gvtd_t = None
    gvtd_filt_per_window = gvtd_filt_p95_per_window = None
    raw_gvtd = raw_od if gvtd_od is None else gvtd_od
    try:
        gvtd_per_window, gvtd_p95_per_window, gvtd_t = compute_windowed_gvtd(raw_gvtd, window_s)
        gvtd_filt_per_window, gvtd_filt_p95_per_window, _ = compute_windowed_filtered_gvtd(raw_gvtd, window_s)
    except Exception as exc:
        logger.warning("windowed GVTD failed: %s", exc)

    # SCI/PSP matrices are channel × window; collapse to a per-window mean over channels
    if sci_matrix is not None and sci_times is not None:
        sqm["sci_per_window"]      = np.asarray(sci_matrix).mean(axis=0).tolist()
        sqm["sci_window_times_s"]  = _center_times(sci_times)
    if psp_matrix is not None and psp_times is not None:
        sqm["psp_per_window"]      = np.asarray(psp_matrix).mean(axis=0).tolist()
        sqm["psp_window_times_s"]  = _center_times(psp_times)
    if gvtd_per_window is not None and len(gvtd_per_window):
        sqm["gvtd_per_window"]     = np.asarray(gvtd_per_window).tolist()
        sqm["gvtd_p95_per_window"] = np.asarray(gvtd_p95_per_window).tolist()
        sqm["gvtd_window_times_s"] = _center_times(gvtd_t)
    if gvtd_filt_per_window is not None and len(gvtd_filt_per_window):
        sqm["gvtd_filt_per_window"]     = np.asarray(gvtd_filt_per_window).tolist()
        sqm["gvtd_filt_p95_per_window"] = np.asarray(gvtd_filt_p95_per_window).tolist()

    series.update(sci_matrix=sci_matrix, sci_times=sci_times,
                  psp_matrix=psp_matrix, psp_times=psp_times)
    return series


# ----------------------------------- SQM aggregators ------------------------------------
def compute_raw_sqm(
    raw_intensity: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, Any]:
    """Metrics computable from raw intensity data (no haemo required).

    Parameters
    ----------
    raw_intensity : mne.io.Raw
        Raw intensity, or already-OD, recording.
    sci_scores : dict[str, float]
        Per-channel SCI, as returned by :func:`compute_sci_scores`.
    bad_channels : list[str]
        Channel names marked bad.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.

    Returns
    -------
    dict
        Flat dict of SCI, channel distance, PSP, CP, and (intensity input only)
        CV/SNR/amplitude plus motion metrics.

    Notes
    -----
    If the input is already optical density, the intensity-value and motion
    metrics are meaningless and set to None; SCI, distance, PSP and CP still run.
    """
    record: dict[str, Any] = {}
    record.update(_sci_metrics(sci_scores, bad_channels))
    record.update(_channel_distance_metrics(raw_intensity))
    record.update(_psp_metrics(raw_intensity, cardiac_l_freq, cardiac_h_freq))
    # CP works in the OD domain, so it runs for both intensity and already-OD input
    record.update(_cardiac_power_metrics(raw_intensity, cardiac_l_freq, cardiac_h_freq))
    if is_optical_density(raw_intensity):
        # intensity-value metrics are meaningless on already-OD data
        logger.warning("input is already optical density; skipping intensity/motion SQM")
        record.update({
            "cv_mean": None, "cv_per_channel": {},
            "snr_mean": None, "snr_per_channel": {}, "snr_pass_rate": None,
            "mean_amp_mean": None, "mean_amp_per_channel": {},
            "spike_count": None, "spike_pct": None,
            "spike_num_frames": None, "spike_pct_frames": None,
            "temporal_derivative_variance": {},
            "gvtd_mean": None, "gvtd_p95": None,
            "gvtd_filt_mean": None, "gvtd_filt_p95": None,
            "gvtd_vstd_mean": None, "gvtd_vstd_p95": None,
            "gvtd_thresh": None, "gvtd_num_above_thresh": None,
            "gvtd_pct_above_thresh": None,
        })
    else:
        record.update(_intensity_metrics(raw_intensity))
        record.update(_spike_metrics(raw_intensity))
        record.update(_motion_metrics(raw_intensity))
    return record




def compute_haemo_sqm(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    """Haemoglobin metrics that stay valid after bandpass and resampling.

    For every stage past ``preproc``, where the signal has usually been filtered,
    downsampled or regressed. Returns HbO-HbR correlation, CNR, gcor, and data retention.
    Band power and drift are deliberately absent: past the bandpass they describe the
    filter rather than the recording, which is what ``compute_prep_haemo_sqm`` is for.
    """
    record: dict[str, Any] = {}
    record.update(haemo_quality_metrics(raw_haemo))
    record.update(_cnr_metrics(raw_haemo))
    record.update(gcor_metrics(raw_haemo))
    record.update(_retention_metrics(raw_haemo))
    return record


def compute_prep_haemo_sqm(
    raw_haemo: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
) -> dict[str, Any]:
    """The above plus spectral band power and drift, which need unfiltered input.

    Band power and drift amplitude describe how much cardiac, respiration, and
    low-frequency content the recording carries. Once a bandpass has removed
    those bands they measure the filter rather than the recording, and above the
    resampled Nyquist they are undefined, so this is restricted to the
    Beer-Lambert output. Band edges are in Hz.
    """
    require_stage(raw_haemo, "preproc")
    record: dict[str, Any] = {}
    record.update(haemo_quality_metrics(raw_haemo))
    record.update(_cnr_metrics(raw_haemo))
    record.update(gcor_metrics(raw_haemo))
    record.update(_spectral_metrics(raw_haemo, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq))
    record.update(_drift_metrics(raw_haemo))
    record.update(_retention_metrics(raw_haemo))
    return record




def comparable_stage_metrics(
    stages: "list[tuple[str, mne.io.Raw]]",
    l_freq: "float | None",
    h_freq: "float | None",
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
) -> dict[str, Any]:
    """Stage-by-stage metrics that may be compared with each other.

    Not read from the record: a record stores each stage measured on the signal as it is at
    that stage, which is the right thing to store and the wrong thing to subtract. A stage
    before the bandpass and one after it are dominated by different frequency content, so
    the difference between their quality metrics is mostly the filter. Every quality metric
    here is therefore recomputed with the analysis passband applied at every stage, which
    is the only way the columns answer the same question::

        stages = [("desc-preproc", raw_a), ("desc-errts", raw_b)], l_freq=0.02, h_freq=0.2
        -> quality["gcor_hbo"] == [<a in 0.02-0.2 Hz>, <b in 0.02-0.2 Hz>]

    ``removed`` is the opposite case and stays on the signal as stored: those rows exist to
    show what left the recording, so band-limiting them would erase the answer. ``banded``
    is False when no passband was given and the quality rows fall back to as-stored, which
    the caller should say out loud.

    Returns
    -------
    dict
        ``labels``, ``banded``, ``quality`` and ``removed`` ({key: value per stage}),
        ``variance_remaining``, the median share of the first stage's per-channel variance
        still present at each stage, and ``hbo_hbr_corr_per_channel``, one dict per stage.
    """
    labels = [label for label, _ in stages]
    raws = [raw for _, raw in stages]
    banded = l_freq is not None or h_freq is not None
    limited = [raw.copy().filter(l_freq, h_freq, verbose=False) if banded else raw
               for raw in raws]

    haemo = [haemo_quality_metrics(r) for r in limited]
    quality: dict[str, list] = {}
    for key, values in (("hbo_hbr_corr_mean", [h.get("hbo_hbr_corr_mean") for h in haemo]),
                        ("gcor_hbo", [gcor_metrics(r).get("gcor_hbo") for r in limited]),
                        ("gcor_hbr", [gcor_metrics(r).get("gcor_hbr") for r in limited]),
                        ("cnr_hbo_mean", [_cnr_metrics(r).get("cnr_hbo_mean") for r in limited]),
                        ("cnr_hbr_mean", [_cnr_metrics(r).get("cnr_hbr_mean") for r in limited])):
        if any(isinstance(v, (int, float)) for v in values):
            quality[key] = values

    # respiration is the one named band that can overlap the passband, and only the overlap
    # survives the filter. Measured over the whole configured band, the drop across the
    # bandpass would be the filter discarding the part above the cutoff; over the overlap it
    # is what the recording lost, so it belongs with the quality rows.
    resp_lo = max(resp_l_freq, l_freq) if l_freq else resp_l_freq
    resp_hi = min(resp_h_freq, h_freq) if h_freq else resp_h_freq
    if resp_lo < resp_hi:
        in_band = [_band_power(r, "hbo", resp_lo, resp_hi) for r in raws]
        if any(isinstance(v, (int, float)) for v in in_band):
            quality["resp_band_power_hbo"] = in_band

    # these two sit entirely outside the passband, so they fall by the filter's stopband
    # attenuation whatever the data did: they say whether the filter ran, not how good the
    # recording is
    removed: dict[str, list] = {}
    spectral = [_spectral_metrics(r, cardiac_l_freq, cardiac_h_freq,
                                  resp_l_freq, resp_h_freq) for r in raws]
    cardiac = [s.get("cardiac_band_power_hbo") for s in spectral]
    if any(isinstance(v, (int, float)) for v in cardiac):
        removed["cardiac_band_power_hbo"] = cardiac
    if l_freq:
        drift = [_band_power(r, "hbo", 0.0, l_freq) for r in raws]
        if any(isinstance(v, (int, float)) for v in drift):
            removed["drift_band_power_hbo"] = drift

    return {"labels": labels, "banded": banded, "quality": quality, "removed": removed,
            "variance_remaining": _variance_remaining(raws),
            "hbo_hbr_corr_per_channel": [h.get("hbo_hbr_corr_per_channel") or {}
                                         for h in haemo]}


def _band_power(raw_haemo: mne.io.Raw, chroma: str, fmin: float, fmax: float) -> "float | None":
    """Mean PSD density in [fmin, fmax], on the same footing as ``_spectral_metrics``."""
    if chroma not in raw_haemo.get_channel_types():
        return None
    psd = raw_haemo.compute_psd(verbose=False)
    data = psd.get_data(picks=chroma)
    mask = (psd.freqs >= fmin) & (psd.freqs <= fmax)
    return float(data[:, mask].mean()) if (data.size and mask.any()) else None


def _variance_remaining(raws: "list[mne.io.Raw]") -> "list[float | None]":
    """Median share of the first stage's per-channel variance still present at each stage."""
    if not raws:
        return []
    shared = [ch for ch in raws[0].ch_names if all(ch in r.ch_names for r in raws)]
    if not shared:
        return [None] * len(raws)
    base = raws[0].get_data(picks=shared).var(axis=1)
    keep = base > 0
    if not keep.any():
        return [None] * len(raws)
    return [float(np.median(r.get_data(picks=shared).var(axis=1)[keep] / base[keep]))
            for r in raws]
