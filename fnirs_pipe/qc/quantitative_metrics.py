"""Compute and persist image quality metrics (IQM) for fNIRS data.

compute_iqm(): all metrics as a flat dict (no I/O).
write_iqm_record(): append one JSONL line to a sidecar file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mne
import numpy as np

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.quantitative_metrics")


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


def _psp_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    try:
        import mne_nirs.preprocessing as nirs_prep
        _, psp_scores, _ = nirs_prep.peak_power(raw_intensity.copy(), verbose=False)
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


def _cardiac_power_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    """Cardiac Power (CP): narrow/wide band power ratio at per-channel cardiac peak.

    Bizzego et al. 2022 (IEEE TNSRE 30:2292-2300).
    fc = peak frequency in 0.83-2.5 Hz; CP = power(fc±0.2 Hz) / power(fc±0.5 Hz).
    """
    try:
        psd = raw_intensity.compute_psd(fmin=0.5, fmax=3.0, verbose=False)
        freqs = psd.freqs
        psd_data = psd.get_data()
        cardiac_mask = (freqs >= 0.83) & (freqs <= 2.5)
        if not cardiac_mask.any():
            raise ValueError("no frequencies in cardiac band")
        cardiac_freqs = freqs[cardiac_mask]
        cp_per_ch: dict[str, float | None] = {}
        for i, ch in enumerate(raw_intensity.ch_names):
            ch_psd = psd_data[i]
            fc = cardiac_freqs[np.argmax(ch_psd[cardiac_mask])]
            narrow = ch_psd[(freqs >= fc - 0.2) & (freqs <= fc + 0.2)]
            wide = ch_psd[(freqs >= fc - 0.5) & (freqs <= fc + 0.5)]
            wide_mean = float(wide.mean()) if wide.size > 0 else 0.0
            cp_per_ch[ch] = (
                float(narrow.mean() / wide_mean)
                if narrow.size > 0 and wide_mean > 0
                else None
            )
        valid = [v for v in cp_per_ch.values() if v is not None]
        return {
            "cp_mean": float(np.mean(valid)) if valid else None,
            "cp_per_channel": cp_per_ch,
        }
    except Exception as e:
        logger.warning("Cardiac Power failed: %s", e)
        return {"cp_mean": None, "cp_per_channel": {}}


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


def _spectral_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    try:
        psd = raw_haemo.compute_psd(verbose=False)
        freqs = psd.freqs
        psd_data = psd.get_data()

        def _band_power(fmin: float, fmax: float) -> float | None:
            mask = (freqs >= fmin) & (freqs <= fmax)
            return float(psd_data[:, mask].mean()) if mask.any() else None

        return {
            "residual_cardiac_power": _band_power(0.7, 1.5),
            "residual_resp_power": _band_power(0.1, 0.5),
        }
    except Exception as e:
        logger.warning("PSD metrics failed: %s", e)
        return {"residual_cardiac_power": None, "residual_resp_power": None}


def _drift_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    try:
        hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
        hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
        raw_filt = raw_haemo.copy().filter(None, 0.01, verbose=False)
        drift_hbo = raw_filt.get_data(picks=hbo_picks)
        drift_hbr = raw_filt.get_data(picks=hbr_picks)
        return {
            "lowfreq_drift_amplitude_hbo": float(np.ptp(drift_hbo, axis=1).mean()) if len(hbo_picks) else None,
            "lowfreq_drift_amplitude_hbr": float(np.ptp(drift_hbr, axis=1).mean()) if len(hbr_picks) else None,
        }
    except Exception as e:
        logger.warning("Drift amplitude failed: %s", e)
        return {"lowfreq_drift_amplitude_hbo": None, "lowfreq_drift_amplitude_hbr": None}


def _motion_metrics(raw_intensity: mne.io.Raw) -> dict[str, Any]:
    try:
        raw_od = mne.preprocessing.nirs.optical_density(raw_intensity.copy())
        od_data = raw_od.get_data()
        diff_data = np.diff(od_data, axis=1)
        # per-channel threshold avoids high-dynamic-range channels dominating spike count
        thresh = 3.0 * diff_data.std(axis=1, keepdims=True)
        gvtd_ts = np.sqrt(np.mean(diff_data ** 2, axis=0))
        # TODO: add gvtd timeseries-derived metrics (e.g. fraction of timepoints above threshold)
        return {
            "spike_count": int((np.abs(diff_data) > thresh).sum()),
            "temporal_derivative_variance": {
                raw_od.ch_names[i]: float(np.var(diff_data[i]))
                for i in range(len(raw_od.ch_names))
            },
            "gvtd_mean": float(gvtd_ts.mean()),
            "gvtd_p95": float(np.percentile(gvtd_ts, 95)),
        }
    except Exception as e:
        logger.warning("Derivative metrics failed: %s", e)
        return {
            "spike_count": None,
            "temporal_derivative_variance": {},
            "gvtd_mean": None,
            "gvtd_p95": None,
        }


def _retention_metrics(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    try:
        total_dur = raw_haemo.times[-1] - raw_haemo.times[0]
        bad_dur = sum(
            ann["duration"]
            for ann in raw_haemo.annotations
            if ann["description"].upper().startswith("BAD")
        )
        return {
            "pct_data_retained": float(1.0 - bad_dur / total_dur) if total_dur > 0 else None
        }
    except Exception as e:
        logger.warning("pct_data_retained failed: %s", e)
        return {"pct_data_retained": None}


def compute_raw_iqm(
    raw_intensity: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
) -> dict[str, Any]:
    """Metrics computable from raw intensity data (no haemo required)."""
    record: dict[str, Any] = {}
    record.update(_sci_metrics(sci_scores, bad_channels))
    record.update(_intensity_metrics(raw_intensity))
    record.update(_channel_distance_metrics(raw_intensity))
    record.update(_psp_metrics(raw_intensity))
    record.update(_cardiac_power_metrics(raw_intensity))
    record.update(_motion_metrics(raw_intensity))
    return record


def compute_glm_iqm(residuals: np.ndarray) -> dict[str, Any]:
    """IQM metrics requiring GLM residuals (post-GLM QC).

    residuals: shape (n_channels, n_timepoints).
    TODO: implement Durbin-Watson per channel.
    """
    return {
        "durbin_watson": None,  # TODO: statsmodels.stats.stattools.durbin_watson per channel
    }


def compute_haemo_iqm(raw_haemo: mne.io.Raw) -> dict[str, Any]:
    """Metrics computable from haemoglobin data (after Beer-Lambert)."""
    record: dict[str, Any] = {}
    record.update(_haemo_quality_metrics(raw_haemo))
    record.update(_spectral_metrics(raw_haemo))
    record.update(_drift_metrics(raw_haemo))
    record.update(_retention_metrics(raw_haemo))
    return record


def compute_iqm(
    raw_intensity: mne.io.Raw,
    raw_haemo: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
) -> dict[str, Any]:
    record = compute_raw_iqm(raw_intensity, sci_scores, bad_channels)
    record.update(compute_haemo_iqm(raw_haemo))
    return record


def save_iqm_toml(iqm: dict[str, Any], subject: str, out_dir: Path, suffix: str = "") -> None:
    """Write scalar IQM fields to a TOML sidecar. suffix e.g. '_raw'."""
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

    scalars = {k: v for k, v in iqm.items() if not isinstance(v, (dict, list))}
    out_path = out_dir / f"sub-{subject}_iqm{suffix}.toml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_to_toml({"subject": subject, **scalars}), encoding="utf-8")
    logger.info("sub-%s | IQM TOML → %s", subject, out_path)


def write_iqm_record(
    subject: str,
    session: str | None,
    iqm: dict[str, Any],
    out_path: Path,
) -> None:
    record = {
        "subject": subject,
        "session": session,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **iqm,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("sub-%s | IQM record appended: %s", subject, out_path)
