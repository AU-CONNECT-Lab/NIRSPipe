"""Motion measured per channel: spikes, and what a correction repaired.

Two things GVTD does not answer. Spikes are a per-channel MAD outlier count in the motion
band, so a single flailing optode is visible where a global index averages it away. The
correction footprint compares optical density either side of the motion step and flags the
samples the correction actually moved, which is the closest thing to a scrubbing record for
a continuous corrector like TDDR. Both are experimental, and both overlap GVTD by design.
"""

from typing import Any

import mne
import numpy as np

from fnirs_pipe.qc.metrics._helpers import _mask_to_segments, _mean_or_none, _safe_metrics
from fnirs_pipe.qc.metrics.gvtd import GVTD_MOTION_BAND
from fnirs_pipe.utils import is_optical_density


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
    "spike_count", "spike_pct", "spike_pct_per_channel",
    "spike_num_frames", "spike_pct_frames", "temporal_derivative_variance",
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
        outliers, an outlier-ratio), spike_pct_per_channel (the same ratio before it is
        pooled), spike_num_frames / spike_pct_frames (timepoints with >= ch_frac of channels
        spiking), and temporal_derivative_variance (per channel).

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
        # the same mask before pooling, so no weighting by the montage's short fraction
        "spike_pct_per_channel": {
            raw_od.ch_names[i]: float(spikes[i].mean())
            for i in range(len(raw_od.ch_names))
        } if spikes.size else {},
        "spike_num_frames": int(flagged.sum()),
        "spike_pct_frames": float(flagged.mean()) if flagged.size else None,
        "temporal_derivative_variance": {
            raw_od.ch_names[i]: float(np.var(diff_raw[i]))
            for i in range(len(raw_od.ch_names))
        },
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
