"""The dicts a run actually stores, assembled from the families.

One function per signal domain, plus ``comparable_stage_metrics``, which is the one place
that recomputes metrics with a common passband so two stages can be subtracted. Nothing
here measures anything itself; it decides which family applies to which stage and merges
the results. Writing any of it to disk is ``qc/sqm_record.py``'s job.
"""

from typing import Any

import mne
import numpy as np

from nirspipe.qc.metrics.coupling import (
    _cardiac_power_metrics, _channel_distance_metrics, _good_frac_metrics,
    _intensity_metrics, _psp_metrics, _sci_metrics, _sci_win_metrics,
)
from nirspipe.qc.metrics.gvtd import _motion_metrics
from nirspipe.qc.metrics.haemo import (
    gcor_metrics, haemo_quality_metrics, _band_mean, _cnr_metrics, _drift_metrics,
    _retention_metrics, _spectral_metrics,
)
from nirspipe.qc.metrics.motion import _spike_metrics
from nirspipe.pipeline.denoise import band_limited
from nirspipe.utils import is_optical_density
from nirspipe.utils.lineage import require_stage
from nirspipe.utils.logging import get_logger

logger = get_logger("qc.metrics.aggregate")


def compute_raw_sqm(
    raw_intensity: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    good_frac_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Metrics computable from raw intensity data (no haemo required).

    Parameters
    ----------
    raw_intensity : mne.io.Raw
        Raw intensity, or already-OD, recording.
    sci_scores : dict[str, float]
        Per-channel SCI, as returned by :func:`~nirspipe.qc.metrics.coupling.compute_sci_scores`.
    bad_channels : list[str]
        Channel names marked bad.
    cardiac_l_freq, cardiac_h_freq : float
        Cardiac band edges in Hz.
    good_frac_scores : dict[str, float] or None, optional
        Per-channel share of coupled windows, as the screening already counted it. None
        stores no such entry rather than paying for a second windowed pass.

    Returns
    -------
    dict
        Flat dict of SCI (whole-run and windowed), channel distance, PSP, CP, the
        coupled-window share, and
        (intensity input only) CV/SNR/amplitude plus motion metrics.

    Notes
    -----
    If the input is already optical density, the intensity-value and motion
    metrics are meaningless and set to None; SCI, distance, PSP and CP still run.
    """
    record: dict[str, Any] = {}
    record.update(_sci_metrics(sci_scores, bad_channels))
    record.update(_sci_win_metrics(raw_intensity, cardiac_l_freq, cardiac_h_freq))
    record.update(_good_frac_metrics(good_frac_scores))
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
    limited: "list[mne.io.Raw] | None" = None,
    design: "dict[str, Any] | None" = None,
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

    ``limited`` is the stages already band-limited, one per stage, for a caller that had to
    filter before cutting a stretch out; the quality rows read them and ``removed`` still
    reads ``stages`` as stored. ``design`` is the run's ``method`` and ``order``, so the band
    is applied with the filter the data went through; None takes the pipeline's default.

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
    if limited is None:
        limited = [band_limited(raw, l_freq, h_freq, **(design or {})) if banded else raw
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
    # one spectrum per stage, shared by the two bands read off it below
    psds = ([r.compute_psd(verbose=False) for r in raws]
            if resp_lo < resp_hi or l_freq else [])
    if resp_lo < resp_hi:
        in_band = [_band_power(p, "hbo", resp_lo, resp_hi) for p in psds]
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
        drift = [_band_power(p, "hbo", 0.0, l_freq) for p in psds]
        if any(isinstance(v, (int, float)) for v in drift):
            removed["drift_band_power_hbo"] = drift

    return {"labels": labels, "banded": banded, "quality": quality, "removed": removed,
            "variance_remaining": _variance_remaining(raws),
            "hbo_hbr_corr_per_channel": [h.get("hbo_hbr_corr_per_channel") or {}
                                         for h in haemo]}


def _band_power(psd, chroma: str, fmin: float, fmax: float) -> "float | None":
    """Mean PSD density in [fmin, fmax] of one stage's spectrum, as ``_spectral_metrics`` takes it."""
    if chroma not in psd.get_channel_types():
        return None
    return _band_mean(psd.freqs, psd.get_data(picks=chroma), fmin, fmax)


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
