"""How well the optodes are coupled to the scalp, and how strong the signal is.

SCI and PSP read the cardiac oscillation the two wavelengths of a pair should share; CV,
SNR and mean amplitude read the intensity itself; CP is experimental. Every one of these is
a property of the recording as acquired, so they run before any rejection and their
aggregates include the channels that rejection will remove.
"""

from typing import Any

import mne
import numpy as np

from fnirs_pipe.qc.metrics._helpers import SNR_PASS
from fnirs_pipe.qc.metrics._helpers import _mean_or_none, _safe_metrics
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.coupling")


# part of what PSP measures, not smoothing: it moves the spread between good and bad channels
PSP_WINDOW_S = 10.0
# CV is sigma/mu, so a longer window admits slower variation into sigma and the number grows
# with the recording: a whole-run CV reads well above a 10 s one, and the difference is
# the drift, not the noise CV_PASS was set for. Pinned for the same reason PSP is,
# and to the same length so the two scalars describe the same stretch of recording.
CV_WINDOW_S = 10.0
# SCI on that same pinned grid. `sci_mean` is the whole-run correlation of the two
# wavelengths, which a slow drift shared by both inflates; over 10 s the cardiac band is
# most of what is left to correlate. The two disagree enough to swap which channel set
# looks better, so both are reported: the whole-run one is what the published cutoffs were
# set on, the windowed one is what the windowed panels and the per-condition slices show.
SCI_WINDOW_S = 10.0


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


@_safe_metrics("SCI (windowed)", ("sci_win_mean", "sci_win_per_channel"))
def _sci_win_metrics(
    raw: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float,
) -> dict[str, Any]:
    """Per-channel SCI averaged over ``SCI_WINDOW_S`` windows, and its mean over channels.

    The windowed twin of ``sci_mean``, measured on the same optical density and pinned to the
    same 10 s as ``psp_mean`` and ``cv_mean`` rather than following the QC window, so the
    four scalars describe the same stretch of recording whatever ``--qc-window`` is set to.
    """
    from fnirs_pipe.qc.metrics.windowed import compute_windowed_sci

    raw_od = (raw if is_optical_density(raw)
              else mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False))
    scores, _times = compute_windowed_sci(
        raw_od, cardiac_l_freq, cardiac_h_freq, SCI_WINDOW_S)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(np.asarray(scores, dtype=float), axis=1)
    per_channel = {ch: float(v) for ch, v in zip(raw_od.ch_names, means)
                   if np.isfinite(v)}
    return {
        "sci_win_mean": _mean_or_none(per_channel.values()),
        "sci_win_per_channel": per_channel,
    }


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


def _good_frac_metrics(good_frac_scores: dict[str, float] | None) -> dict[str, Any]:
    """The coupled-window share per channel and its mean, or empty when nothing measured it.

    Handed in rather than measured here: the screening already counted these windows, and
    counting them twice is a second pass of windowed SCI and PSP over the whole recording.
    """
    if not good_frac_scores:
        return {"good_frac_mean": None, "good_frac_per_channel": {}}
    return {
        "good_frac_mean": _mean_or_none(good_frac_scores.values()),
        "good_frac_per_channel": {k: float(v) for k, v in good_frac_scores.items()},
    }


def channel_cv_windowed(data: np.ndarray, sfreq: float,
                        window_s: float = CV_WINDOW_S) -> np.ndarray:
    """Per-channel CV averaged over non-overlapping windows; whole-run if one does not fit.

    ::

        3900 s at 10 Hz, 10 s windows  ->  mean of 390 CVs per channel
    """
    n = int(round(window_s * float(sfreq)))
    if n < 2 or data.shape[1] < n:
        return channel_cv(data)
    m = data.shape[1] // n
    w = data[:, :m * n].reshape(data.shape[0], m, n)
    mu, sd = w.mean(axis=2), w.std(axis=2)
    cv = np.divide(sd, mu, out=np.full_like(sd, np.nan), where=mu != 0)
    with np.errstate(invalid="ignore"):
        return np.nanmean(cv, axis=1)


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
def _intensity_metrics(raw_intensity: mne.io.Raw,
                       snr_threshold: float = SNR_PASS) -> dict[str, Any]:
    """Per-channel CV, SNR and mean amplitude from raw intensity (with means; CV also per wavelength).

    snr_pass_rate is the fraction of channels with SNR > snr_threshold (SNR_PASS, the same
    line the per-channel figures draw; SNR = mean/std, higher is better). Its denominator is
    every channel, not every channel with a finite SNR: a flat or saturated channel has
    std 0 and no finite SNR at all, and letting it drop out of the denominator would mean a
    recording whose channels are dying reads as a recording whose channels are passing.
    n_flat_channels is how many those were. The means are still taken over the finite
    values, since an average cannot carry a NaN.
    """
    int_data = raw_intensity.get_data()
    names = raw_intensity.ch_names
    cv = channel_cv_windowed(int_data, raw_intensity.info["sfreq"])
    # 1/CV rather than a second windowed pass, so the two stay exact reciprocals
    snr = np.divide(1.0, cv, out=np.full_like(cv, np.nan), where=np.isfinite(cv) & (cv > 0))
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


def compute_psp_scores(
    raw: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, float]:
    """Peak spectral power per channel, averaged over ``PSP_WINDOW_S`` windows.

    Measured on optical density, which is what ``peak_power`` is meant for: the metric
    cross-correlates the two wavelengths, so it has no meaning after Beer-Lambert, and
    mne_nirs' own test converts to OD before calling it. Its docstring saying
    "haemoglobin data" is a copy-paste slip shared with ``scalp_coupling_index_windowed``.
    Converting here keeps this agreeing with the windowed PSP series, which is handed OD.

    The window is pinned to ``PSP_WINDOW_S`` rather than following the QC window length; see
    the constant for what changes when it moves. The windowed PSP series is a separate view
    and does follow the QC window, so its colour scale is not on this scalar's scale.

    Separate from the metric wrapper below because channel screening needs the scores and
    not the record entry, and computing them twice per run would be paying twice for the
    same measurement.
    """
    import mne_nirs.preprocessing as nirs_prep
    raw_od = (raw if is_optical_density(raw)
              else mne.preprocessing.nirs.optical_density(raw.copy()))
    _, psp_scores, _ = nirs_prep.peak_power(
        raw_od.copy(), time_window=PSP_WINDOW_S,
        l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
    return {ch: float(np.mean(psp_scores[i])) for i, ch in enumerate(raw_od.ch_names)}


@_safe_metrics("PSP", ("psp_mean", "psp_per_channel"))
def _psp_metrics(
    raw_intensity: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, Any]:
    """The PSP record entry: per-channel scores and their mean."""
    psp_per_ch = compute_psp_scores(raw_intensity, cardiac_l_freq, cardiac_h_freq)
    return {
        "psp_mean": _mean_or_none(psp_per_ch.values()),
        "psp_per_channel": psp_per_ch,
    }


@_safe_metrics("Cardiac Power", ("cp_mean", "cp_per_channel"))
def _cardiac_power_metrics(
    raw: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, Any]:
    r"""Cardiac Power (CP): within-band power concentrated at the per-channel cardiac peak :footcite:`Bizzego2022`.

    .. math::

        \text{CP} = \frac{P(f_c \pm 0.2\,\text{Hz})}{P(f_c \pm 0.5\,\text{Hz})},

    where :math:`f_c` is the peak frequency in ``[cardiac_l_freq, cardiac_h_freq]``.

    Reported without a pass/fail line. CP measures the *shape* of one channel's spectrum
    inside the band and never compares the two wavelengths, so it is not a coupling metric
    despite the company it keeps: rotating the cardiac phase of one wavelength until SCI
    inverts leaves CP unmoved. Its source defines a CP >= 0.5 gate on a fixed 0.83-2.5 Hz
    band, which does not survive a user-chosen band, and argues against fixed thresholds on
    these indicators in the same paper. Experimental, and it tracks PSP closely
    (Spearman rho 0.83). SCI + PSP are the primary cardiac quality metrics.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw intensity or optical-density recording.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.

    Returns
    -------
    dict
        cp_mean and cp_per_channel.

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
    }
