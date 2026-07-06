"""Compute and persist image quality metrics (SQM) for fNIRS data.

compute_sqm(): all metrics as a flat dict (no I/O).
write_sqm_record(): append one JSONL line to a sidecar file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import numpy as np

from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.quantitative_metrics")

GVTD_MOTION_BAND = (0.01, 0.5)  # Hz — bandpass for the filtered (motion-specific) GVTD


# Sliding-window QC metrics
def compute_windowed_sci(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray]":
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
    from mne_nirs.preprocessing import peak_power
    _, scores, times = peak_power(
        raw_od, time_window=window_s, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times


def _windowed_gvtd(
    raw_od: mne.io.Raw, window_s: float, l_freq: float | None, h_freq: float | None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    sfreq = float(raw_od.info["sfreq"])
    gvtd_ts = gvtd_timetrace(raw_od.get_data(), sfreq, l_freq=l_freq, h_freq=h_freq)
    win_samples = max(1, int(round(window_s * sfreq)))
    n_windows = len(gvtd_ts) // win_samples
    if n_windows == 0:
        return np.array([]), np.array([]), np.array([])
    truncated = gvtd_ts[:n_windows * win_samples].reshape(n_windows, win_samples)
    # mean = average motion level; p95 = worst-moment, so transient motion survives averaging
    gvtd_mean = truncated.mean(axis=1)
    gvtd_p95 = np.percentile(truncated, 95, axis=1)
    window_times = np.arange(n_windows) * window_s + window_s / 2
    return gvtd_mean, gvtd_p95, window_times


def compute_windowed_gvtd(
    raw_od: mne.io.Raw, window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean & p95 GVTD per non-overlapping window (unfiltered). Returns (mean, p95, center_times)."""
    return _windowed_gvtd(raw_od, window_s, None, None)


def compute_windowed_filtered_gvtd(
    raw_od: mne.io.Raw, window_s: float = 10.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean & p95 GVTD per window on the motion-band bandpassed OD. Returns (mean, p95, center_times)."""
    return _windowed_gvtd(raw_od, window_s, *GVTD_MOTION_BAND)


def compute_sci_scores(
    raw: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float,
) -> tuple[dict[str, float], mne.io.Raw]:
    """Return ({channel: SCI score}, raw_od). SCI defaults to 1.0 for all channels on failure."""
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
    try:
        sci_arr = mne.preprocessing.nirs.scalp_coupling_index(
            raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
        sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
    except Exception as exc:
        logger.warning("SCI failed: %s", exc)
        sci_scores = {ch: 1.0 for ch in raw.ch_names}
    return sci_scores, raw_od


def attach_windowed_series(
    sqm: dict, raw_od: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float,
    window_s: float = 10.0,
) -> dict:
    """Compute sliding-window SCI/PSP/GVTD series, attach summaries to sqm, return raw series.

    Returned dict carries sci_matrix/sci_times/psp_matrix/psp_times for callers that also plot
    them. On failure everything is None and sqm is left unchanged. Center times collapse the
    mne-nirs [start, end] window pairs to their midpoint.
    """
    def _center_times(t):
        a = np.asarray(t)
        return (a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a).tolist()

    series = {"sci_matrix": None, "sci_times": None, "psp_matrix": None, "psp_times": None}
    try:
        sci_matrix, sci_times = compute_windowed_sci(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
        psp_matrix, psp_times = compute_windowed_psp(raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
        gvtd_per_window, gvtd_p95_per_window, gvtd_t = compute_windowed_gvtd(raw_od, window_s)
        gvtd_filt_per_window, gvtd_filt_p95_per_window, _ = compute_windowed_filtered_gvtd(raw_od, window_s)
    except Exception as exc:
        logger.warning("windowed metrics failed: %s", exc)
        return series

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


def _sci_metrics(
    sci_scores: dict[str, float],
    bad_channels: list[str],
) -> dict[str, Any]:
    sci_vals = list(sci_scores.values())
    n_total = len(sci_scores)
    return {
        "sci_mean": float(np.mean(sci_vals)) if sci_vals else None,
        "sci_per_channel": {k: float(v) for k, v in sci_scores.items()},
        "channel_retention_rate": (
            float((n_total - len(bad_channels)) / n_total) if n_total > 0 else None
        ),
    }


def _intensity_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    try:
        int_data = raw_intensity.get_data()
        ch_means = int_data.mean(axis=1)
        ch_stds  = int_data.std(axis=1)
        cv_per_ch = {
            ch: float(ch_stds[i] / ch_means[i])
            for i, ch in enumerate(raw_intensity.ch_names)
            if ch_means[i] != 0
        }
        # SNR = mean/std on raw intensity per channel
        snr_per_ch = {
            ch: float(ch_means[i] / ch_stds[i])
            for i, ch in enumerate(raw_intensity.ch_names)
            if ch_stds[i] > 0
        }
        mean_amp_per_ch = {
            ch: float(ch_means[i])
            for i, ch in enumerate(raw_intensity.ch_names)
        }
        wl_groups: dict[str, list[float]] = {}
        for ch, cv in cv_per_ch.items():
            wl = ch.split()[-1]
            wl_groups.setdefault(wl, []).append(cv)
        cv_mean_per_wl = {
            f"cv_mean_{wl}": float(np.mean(vals))
            for wl, vals in sorted(wl_groups.items())
        }
        return {
            "cv_mean": float(np.mean(list(cv_per_ch.values()))) if cv_per_ch else None,
            **cv_mean_per_wl,
            "cv_per_channel": cv_per_ch,
            "snr_mean": float(np.mean(list(snr_per_ch.values()))) if snr_per_ch else None,
            "snr_per_channel": snr_per_ch,
            "mean_amp_mean": float(np.mean(list(mean_amp_per_ch.values()))) if mean_amp_per_ch else None,
            "mean_amp_per_channel": mean_amp_per_ch,
        }
    except Exception as e:
        logger.warning("CV/SNR failed: %s", e)
        return {
            "cv_mean": None, "cv_per_channel": {},
            "snr_mean": None, "snr_per_channel": {},
            "mean_amp_mean": None, "mean_amp_per_channel": {},
        }


def _channel_distance_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    try:
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
    except Exception as e:
        logger.warning("Channel distance failed: %s", e)
        return {
            "ch_dist_mean": None, "ch_dist_min": None,
            "ch_dist_max": None, "ch_dist_per_channel": {},
        }


def _psp_metrics(raw_intensity: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float) -> dict[str, Any]:
    try:
        import mne_nirs.preprocessing as nirs_prep
        _, psp_scores, _ = nirs_prep.peak_power(
            raw_intensity.copy(), l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
        psp_per_ch = {
            ch: float(np.mean(psp_scores[i]))
            for i, ch in enumerate(raw_intensity.ch_names)
        }
        return {
            "psp_mean": float(np.mean(list(psp_per_ch.values()))),
            "psp_per_channel": psp_per_ch,
        }
    except Exception as e:
        logger.warning("PSP failed: %s", e)
        return {"psp_mean": None, "psp_per_channel": {}}


def _cardiac_power_metrics(
    raw: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    cp_threshold: float = 0.5,
) -> dict[str, Any]:
    """Cardiac Power (CP): within-band power concentrated at the per-channel cardiac peak.

    Experimental: overlaps PSP and its 0.5 gate is calibrated on Bizzego's 0.83-2.5 band;
    may be removed. SCI + PSP are the primary cardiac quality metrics.

    Bizzego et al. 2022 (IEEE TNSRE 30:2292-2300).
    fc = peak frequency in [cardiac_l_freq, cardiac_h_freq]; CP = P(fc±0.2 Hz) / P(fc±0.5 Hz).
    cp_threshold: good-quality gate (Bizzego CP>=0.5; calibrated on their 0.83-2.5 band).
    """
    try:
        # CP is defined in the OD domain (aligns with SCI/PSP); convert unless input is already OD
        raw_od = raw if is_optical_density(raw) else mne.preprocessing.nirs.optical_density(raw.copy())
        fmax = min(cardiac_h_freq, raw_od.info["sfreq"] / 2)
        # PSD restricted to the cardiac band, so the ±0.2/±0.5 windows below are auto-clipped to it
        # (equivalent to Bizzego's pre-bandpass; keeps respiration/Mayer power out of the ratio)
        psd = raw_od.compute_psd(fmin=cardiac_l_freq, fmax=fmax, verbose=False)
        freqs = psd.freqs
        psd_data = psd.get_data()
        if freqs.size == 0:
            raise ValueError("no frequencies in cardiac band")
        cp_per_ch: dict[str, float | None] = {}
        for i, ch in enumerate(raw_od.ch_names):
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
            "cp_mean": float(np.mean(valid)) if valid else None,
            "cp_per_channel": cp_per_ch,
            "cp_pass_rate": (
                float(np.mean([v >= cp_threshold for v in valid])) if valid else None
            ),
        }
    except Exception as e:
        logger.warning("Cardiac Power failed: %s", e)
        return {"cp_mean": None, "cp_per_channel": {}, "cp_pass_rate": None}


def _haemo_quality_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    hbo_data = raw_haemo.get_data(picks=hbo_picks)
    hbr_data = raw_haemo.get_data(picks=hbr_picks)
    hbo_names = [raw_haemo.ch_names[i] for i in hbo_picks]
    hbr_names = [raw_haemo.ch_names[i] for i in hbr_picks]

    # tSNR = mean/std per channel; valid only on unfiltered Beer-Lambert output (mean != 0).
    # TODO: verify raw_haemo passed here is always pre-bandpass when called from pipeline.
    tsnr_hbo = {
        n: (float(hbo_data[i].mean() / hbo_data[i].std()) if hbo_data[i].std() > 0 else None)
        for i, n in enumerate(hbo_names)
    }
    tsnr_hbr = {
        n: (float(hbr_data[i].mean() / hbr_data[i].std()) if hbr_data[i].std() > 0 else None)
        for i, n in enumerate(hbr_names)
    }

    hbo_map = {n.rsplit(" ", 1)[0]: hbo_data[i] for i, n in enumerate(hbo_names)}
    hbr_map = {n.rsplit(" ", 1)[0]: hbr_data[i] for i, n in enumerate(hbr_names)}
    corr_per_ch = {
        key: float(np.corrcoef(hbo_map[key], hbr_map[key])[0, 1])
        for key in hbo_map if key in hbr_map
    }
    return {
        "tsnr_hbo_mean": (
            float(np.nanmean([v for v in tsnr_hbo.values() if v is not None]))
            if tsnr_hbo else None
        ),
        "tsnr_hbr_mean": (
            float(np.nanmean([v for v in tsnr_hbr.values() if v is not None]))
            if tsnr_hbr else None
        ),
        "tsnr_hbo_per_channel": tsnr_hbo,
        "tsnr_hbr_per_channel": tsnr_hbr,
        "hbo_hbr_corr_mean": (
            float(np.mean(list(corr_per_ch.values()))) if corr_per_ch else None
        ),
        "hbo_hbr_corr_per_channel": corr_per_ch,
    }


def _spectral_metrics(
    raw_haemo: mne.io.Raw,
    cardiac_l_freq: float, cardiac_h_freq: float,
    resp_l_freq: float, resp_h_freq: float,
) -> dict[str, Any]:
    """Cardiac/respiration power in the haemoglobin PSD — absolute and as a fraction of total.

    *_band_power = mean PSD in the band (absolute; scales with overall signal amplitude).
    *_band_frac  = band power / total spectral power (fALFF-style fraction in [0,1],
                   so it is comparable across subjects/channels regardless of amplitude).
    """
    try:
        psd = raw_haemo.compute_psd(verbose=False)
        freqs = psd.freqs
        psd_data = psd.get_data()
        total = float(psd_data.sum())  # total spectral power over all channels and freqs

        def _band_power(fmin: float, fmax: float) -> float | None:
            # absolute: mean PSD density inside the band
            mask = (freqs >= fmin) & (freqs <= fmax)
            return float(psd_data[:, mask].mean()) if mask.any() else None

        def _band_frac(fmin: float, fmax: float) -> float | None:
            # relative: fraction of total power falling in the band (sum/sum, in [0,1])
            mask = (freqs >= fmin) & (freqs <= fmax)
            return float(psd_data[:, mask].sum() / total) if (mask.any() and total > 0) else None

        return {
            "cardiac_band_power": _band_power(cardiac_l_freq, cardiac_h_freq),
            "cardiac_band_frac":  _band_frac(cardiac_l_freq, cardiac_h_freq),
            "resp_band_power":    _band_power(resp_l_freq, resp_h_freq),
            "resp_band_frac":     _band_frac(resp_l_freq, resp_h_freq),
        }
    except Exception as e:
        logger.warning("PSD metrics failed: %s", e)
        return {
            "cardiac_band_power": None, "cardiac_band_frac": None,
            "resp_band_power": None, "resp_band_frac": None,
        }


def _gcor(data: np.ndarray) -> "float | None":
    """Global correlation (Saad 2013): mean of all pairwise channel correlations.

    Demean + unit-L2-normalise each channel, average into g, then gcor = g.g = ||g||^2.
    High = channels move together (global artifact / systemic physiology).
    """
    if data.shape[0] < 2:
        return None
    x = data - data.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    g = x.mean(axis=0)
    return float(g @ g)


def _gcor_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    # Per chromophore: HbO and HbR anti-correlate, so a mixed gcor would cancel to ~0.
    try:
        hbo = raw_haemo.get_data(picks=mne.pick_types(raw_haemo.info, fnirs="hbo"))
        hbr = raw_haemo.get_data(picks=mne.pick_types(raw_haemo.info, fnirs="hbr"))
        return {"gcor_hbo": _gcor(hbo), "gcor_hbr": _gcor(hbr)}
    except Exception as e:
        logger.warning("gcor failed: %s", e)
        return {"gcor_hbo": None, "gcor_hbr": None}


def _drift_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    # NOTE: non-standard homegrown metric; may remove.
    try:
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
    except Exception as e:
        logger.warning("Drift amplitude failed: %s", e)
        return {"lowfreq_drift_amplitude_hbo": None, "lowfreq_drift_amplitude_hbr": None}


def gvtd_timetrace(
    data: np.ndarray,
    sfreq: float,
    l_freq: float | None = None,
    h_freq: float | None = None,
    standardize_channels: bool = False,
) -> np.ndarray:
    """GVTD time trace (Sherafati 2020): RMS across channels of the temporal derivative.

    gvtd[i] = sqrt(mean_ch (x[:,i] - x[:,i-1])^2); l_freq/h_freq apply an optional
    Butterworth (order 4) bandpass before differencing (isolates the motion band).
    standardize_channels: DVARS-vstd analog (Nichols 2013) — divide each channel's
    derivative by its own SD before the RMS, so high-dynamic-range channels don't
    dominate the global value.
    """
    d = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if h_freq is not None and h_freq >= sfreq / 2:
        h_freq = None
    if l_freq is not None or h_freq is not None:
        d = mne.filter.filter_data(
            d, sfreq, l_freq, h_freq, method="iir",
            iir_params=dict(order=4, ftype="butter"), verbose=False,
        )
    diff = np.diff(d, axis=1)
    if standardize_channels:
        sd = diff.std(axis=1, keepdims=True)
        diff = np.divide(diff, sd, out=np.zeros_like(diff), where=sd > 0)
    return np.sqrt(np.mean(diff ** 2, axis=0))


def gvtd_threshold(gvtd: np.ndarray, n_std: float = 3.0) -> float | None:
    """GVTD motion threshold (Sherafati 2020, histogram-mode): mode + n_std * left-tail std.

    Mode = center of the tallest histogram bin (bins ~ n/5); left std = RMS of points below it.
    """
    g = gvtd[np.isfinite(gvtd)]
    gmax = float(g.max()) if g.size else 0.0
    if gmax <= 0:
        return None
    n_bins = max(1, int(round(g.size / 5)))
    bin_w = gmax / n_bins
    counts, edges = np.histogram(g, bins=np.arange(0.0, gmax + bin_w, bin_w))
    if counts.size == 0:
        return None
    run_mode = float(edges[int(np.argmax(counts))] + bin_w / 2)
    below = g[g < run_mode]
    if below.size == 0:
        return run_mode
    left_std = float(np.sqrt(np.sum((below - run_mode) ** 2) / below.size))
    return run_mode + n_std * left_std


def _motion_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    try:
        raw_od = mne.preprocessing.nirs.optical_density(raw_intensity.copy())
        sfreq = float(raw_od.info["sfreq"])
        od_data = np.nan_to_num(raw_od.get_data(), nan=0.0, posinf=0.0, neginf=0.0)
        diff_data = np.diff(od_data, axis=1)
        # Spike count = per-channel timepoints whose OD temporal derivative is an outlier.
        # Threshold uses MAD, not std: std is taken over the whole derivative *including the
        # spikes*, so a few large spikes inflate std -> threshold too high -> real spikes fall
        # under it (non-robust). MAD (median abs deviation) resists those outliers.
        #   thresh = median + 3 * 1.4826 * MAD ;  count where |diff - median| > thresh
        # (1.4826*MAD ~= sigma for Gaussian data, so this is a robust 3-sigma). Per-channel
        # scale also stops high-dynamic-range channels from dominating the total count.
        med = np.median(diff_data, axis=1, keepdims=True)
        mad = np.median(np.abs(diff_data - med), axis=1, keepdims=True)
        spike_thresh = 3.0 * 1.4826 * mad
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
            "spike_count": int((np.abs(diff_data - med) > spike_thresh).sum()),
            "temporal_derivative_variance": {
                raw_od.ch_names[i]: float(np.var(diff_data[i]))
                for i in range(len(raw_od.ch_names))
            },
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
    except Exception as e:
        logger.warning("Derivative metrics failed: %s", e)
        return {
            "spike_count": None,
            "temporal_derivative_variance": {},
            "gvtd_mean": None,
            "gvtd_p95": None,
            "gvtd_filt_mean": None,
            "gvtd_filt_p95": None,
            "gvtd_vstd_mean": None,
            "gvtd_vstd_p95": None,
            "gvtd_thresh": None,
            "gvtd_num_above_thresh": None,
            "gvtd_pct_above_thresh": None,
        }


def _retention_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    try:
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
    except Exception as e:
        logger.warning("pct_data_retained failed: %s", e)
        return {"pct_data_retained": None}


def compute_raw_sqm(
    raw_intensity: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> dict[str, Any]:
    """Metrics computable from raw intensity data (no haemo required)."""
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
            "snr_mean": None, "snr_per_channel": {},
            "mean_amp_mean": None, "mean_amp_per_channel": {},
            "spike_count": None, "temporal_derivative_variance": {},
            "gvtd_mean": None, "gvtd_p95": None,
            "gvtd_filt_mean": None, "gvtd_filt_p95": None,
            "gvtd_vstd_mean": None, "gvtd_vstd_p95": None,
            "gvtd_thresh": None, "gvtd_num_above_thresh": None,
            "gvtd_pct_above_thresh": None,
        })
    else:
        record.update(_intensity_metrics(raw_intensity))
        record.update(_motion_metrics(raw_intensity))
    return record


def compute_glm_sqm(residuals: np.ndarray) -> dict[str, Any]:
    """Post-GLM QC from the residual time series (n_channels, n_timepoints).

    Durbin-Watson per channel: DW = sum((e_t - e_{t-1})^2) / sum(e_t^2), in [0, 4];
    ~2 = white residuals (GLM t/p values trustworthy), <2 = positive autocorrelation
    (hemodynamic signals are autocorrelated; DW checks whether prewhitening worked).
    """
    try:
        e = np.asarray(residuals, dtype=float)
        denom = np.sum(e ** 2, axis=1)
        num = np.sum(np.diff(e, axis=1) ** 2, axis=1)
        dw = np.divide(num, denom, out=np.full_like(denom, np.nan), where=denom > 0)
        valid = dw[np.isfinite(dw)]
        return {
            "durbin_watson_mean": float(valid.mean()) if valid.size else None,
            "durbin_watson_per_channel": {
                str(i): float(v) for i, v in enumerate(dw) if np.isfinite(v)
            },
        }
    except Exception as exc:
        logger.warning("Durbin-Watson failed: %s", exc)
        return {"durbin_watson_mean": None, "durbin_watson_per_channel": {}}


def compute_haemo_sqm(
    raw_haemo: mne.io.Raw,
    cardiac_l_freq: float, cardiac_h_freq: float,
    resp_l_freq: float, resp_h_freq: float,
) -> dict[str, Any]:
    """Metrics computable from haemoglobin data (after Beer-Lambert)."""
    record: dict[str, Any] = {}
    record.update(_haemo_quality_metrics(raw_haemo))
    record.update(_gcor_metrics(raw_haemo))
    record.update(_spectral_metrics(raw_haemo, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq))
    record.update(_drift_metrics(raw_haemo))
    record.update(_retention_metrics(raw_haemo))
    return record


def compute_sqm(
    raw_intensity: mne.io.Raw,
    raw_haemo: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
) -> dict[str, Any]:
    record = compute_raw_sqm(raw_intensity, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq)
    record.update(compute_haemo_sqm(raw_haemo, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq))
    return record


def save_sqm_toml(sqm: dict[str, Any], subject: str, out_dir: Path, suffix: str = "") -> None:
    """Write scalar SQM fields to a TOML sidecar. suffix e.g. '_raw'."""
    def _to_toml(data: dict) -> str:
        lines = []
        for k, v in data.items():
            if v is None:
                lines.append(f"# {k} = null")
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            elif isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
        return "\n".join(lines) + "\n"

    scalars = {k: v for k, v in sqm.items() if not isinstance(v, (dict, list))}
    out_path = out_dir / f"sub-{subject}_sqm{suffix}.toml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_toml({"subject": subject, **scalars}), encoding="utf-8")
    logger.info("sub-%s | SQM TOML → %s", subject, out_path)


def write_sqm_record(
    subject: str,
    session: str | None,
    sqm: dict[str, Any],
    out_path: Path,
) -> None:
    record = {
        "subject": subject,
        "session": session,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **sqm,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("sub-%s | SQM record appended: %s", subject, out_path)
