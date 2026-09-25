"""Metrics on haemoglobin, i.e. everything past Beer-Lambert.

HbO-HbR anticorrelation, CNR against the stimulus, band powers, gcor and drift. This is the
far side of the line the package docstring draws: these aggregates exclude rejected
channels, because by this stage the channel is out of the analysis rather than part of the
archival record.
"""

from typing import Any

import mne
import numpy as np

from fnirs_pipe.qc.metrics._helpers import (
    _mean_or_none, _safe_metrics, epochable_events,
)


# the span each end is measured over, fixed rather than a fraction of the record so two
# runs of different length are comparable
EDGE_S = 60.0


def edge_to_mid_rms(
    raw: mne.io.Raw, edge_s: float = EDGE_S, picks: "list[int] | None" = None,
) -> "float | None":
    """How much louder the two ends of a filtered recording are than its middle.

    A filter with a low cutoff needs a long impulse response, so its output starts and ends
    with a transient that is the filter settling rather than anything measured. This is the
    RMS of the first and last ``edge_s`` seconds against the RMS of everything between,
    median over channels. A value near 1 means the ends are no louder than the middle;
    above 1 means the transient is a real part of what the file contains.

    A detrend does not reduce it: the bandpass is linear and time-invariant and the trend
    lies in its stopband, so subtracting the trend first changes nothing.

    Parameters
    ----------
    raw : mne.io.Raw
        A filtered stage. On unfiltered data the ratio is still defined but means nothing.
    edge_s : float
        Seconds at each end.
    picks : list of int or None
        Channels to measure; None takes every fNIRS channel, rejected ones included, since
        this is a property of the filter rather than of channel quality.

    Returns
    -------
    float or None
        The median ratio, or None when the record is too short to have a middle.
    """
    n_edge = int(round(edge_s * raw.info["sfreq"]))
    if raw.n_times < 3 * n_edge or n_edge < 1:
        return None
    if picks is None:
        picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    data = raw.get_data(picks=picks)
    if not len(data):
        return None

    def rms(block):
        return np.sqrt(np.mean(block ** 2, axis=1))

    edges = rms(np.concatenate([data[:, :n_edge], data[:, -n_edge:]], axis=1))
    middle = rms(data[:, n_edge:-n_edge])
    good = middle > 0
    if not good.any():
        return None
    return float(np.median(edges[good] / middle[good]))


# fixed rather than derived from stimulus duration, so two runs stay comparable
CNR_BASELINE_S = (-5.0, 0.0)
CNR_RESPONSE_S = (5.0, 15.0)
DRIFT_ORDER_PER_S = 150.0  # one polynomial degree per this many seconds
DRIFT_ORDER_MAX = 30


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
    same time as the denominator.

    ``BAD_`` annotations are censoring marks rather than stimuli and are excluded from the
    event set; epochs overlapping them are dropped by ``reject_by_annotation``. Bad channels
    are excluded, following ``mne.pick_types``.
    """

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
def _drift_metrics(raw_haemo: mne.io.Raw, order: "int | None" = None) -> dict[str, Any]:
    r"""Low-frequency baseline drift amplitude per chromophore (peak-to-peak of a slow trend).

    A polynomial trend is fitted per channel; drift is the mean peak-to-peak of that trend.
    Its order follows the recording length, one degree per ``DRIFT_ORDER_PER_S`` seconds,
    so what counts as drift is a period (about twice that) rather than a shape.

    Parameters
    ----------
    raw_haemo : mne.io.Raw
        Haemoglobin recording.
    order : int, optional
        Polynomial order. None derives it from the duration.

    Returns
    -------
    dict
        lowfreq_drift_amplitude_hbo and _hbr.

    Notes
    -----
    Non-standard homegrown metric. A polynomial is used instead of a 0.01 Hz low-pass,
    whose FIR length would exceed most recordings. The order rule is
    ``1 + floor(run_time / 150)``. Peak-to-peak grows with duration whatever the order, so
    the number is not comparable between recordings of different length.
    """
    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    n = len(raw_haemo.times)
    if order is None:
        order = int(min(DRIFT_ORDER_MAX,
                        1 + np.floor(raw_haemo.times[-1] / DRIFT_ORDER_PER_S)))
    order = max(1, min(order, n - 1))
    #   trend = V @ lstsq(V, x) ;  drift = ptp(trend) mean over channels
    t = np.linspace(-1.0, 1.0, n)
    vander = np.vander(t, order + 1)

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
