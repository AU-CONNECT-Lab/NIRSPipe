"""The bridge between the step names the pipeline records and the prose describing them.

Two vocabularies exist and they sit at different granularities on purpose, so this maps
rather than renames:

- the pipeline records one ``motion_correction`` step with the method as a parameter,
  while ``steps.toml`` needs one paragraph per method because the citations differ
- the pipeline records one ``bandpass`` step with two cutoffs, while the prose splits
  into bandpass / highpass / lowpass depending on which cutoff was given
- ``design_matrix``, ``glm_fit``, ``glm_residuals`` and ``contrasts`` are four files from
  one method described in a single paragraph

Steps with no paragraph are not omissions: reading a file or recording quality metrics is
bookkeeping, not method, and putting it in ``steps.toml`` would leak it into the Methods
section. Those get a plain one-liner here instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---- pipeline step -> steps.toml section ----

_DIRECT = ("od_conversion", "beer_lambert", "resample")


def boilerplate_key(step: str | None, params: dict[str, Any], mode: str | None = None) -> str | None:
    """Section of steps.toml describing this step, or None when it has no method prose."""
    if not step:
        return None
    if step in _DIRECT:
        return step
    if step == "sci_pruning":
        return "sci_marking"
    if step == "motion_correction":
        method = params.get("motion_correction")
        return f"motion_{method}" if method and method != "none" else None
    if step == "bandpass":
        low_edge, high_edge = params.get("high_pass"), params.get("low_pass")
        if low_edge and high_edge:
            return "bandpass"
        if low_edge:
            return "highpass"
        return "lowpass" if high_edge else None
    if step == "glm_fit":
        # rest and denoise run this same code to regress confounds out; only a task run is a
        # first-level GLM, and nothing in the sidecar separates the two
        return "glm" if mode == "glm" else "confound_regression"
    return None


def _regressor_phrase(params: dict[str, Any]) -> str:
    """Name the nuisance columns a confound regression actually carried.

    {"short_channel": "mean", "drift_model": "cosine", "drift_high_pass": 0.01}
      -> "the mean short-channel time course of each chromophore and a discrete cosine
          drift basis (high-pass cutoff: 0.01 Hz)"
    """
    parts = []
    sc = params.get("short_channel")
    if sc == "mean":
        parts.append("the mean short-channel time course of each chromophore")
    elif sc == "pca":
        parts.append("the first principal component of the short channels of each chromophore")

    drift = params.get("drift_model")
    if drift == "cosine":
        parts.append("a discrete cosine drift basis "
                     f"(high-pass cutoff: {params.get('drift_high_pass')} Hz)")
    elif drift == "polynomial":
        parts.append(f"an order-{params.get('drift_order')} polynomial drift basis")

    # the design matrix always holds an intercept, so there is something to say even when
    # neither flag was given, and the sentence stays true rather than naming absent columns
    return " and ".join(parts) if parts else "a constant term only"


def template_slots(key: str, params: dict[str, Any]) -> dict[str, str]:
    """Fill a section's {slots} from a sidecar's parameters.

    Note the naming: a sidecar's ``high_pass`` is the *lower* edge of the band, which the
    filter templates call ``l_freq``.
    """
    if key == "sci_marking":
        return {
            "threshold": str(params.get("sci_threshold", "")),
            "action": "marked as bad and excluded from further analysis",
        }
    if key == "beer_lambert":
        dpf = params.get("dpf")
        return {"dpf": ", ".join(str(d) for d in dpf) if isinstance(dpf, (list, tuple)) else str(dpf)}
    if key == "bandpass":
        return {"l_freq": str(params.get("high_pass")), "h_freq": str(params.get("low_pass"))}
    if key == "highpass":
        return {"l_freq": str(params.get("high_pass"))}
    if key == "lowpass":
        return {"h_freq": str(params.get("low_pass"))}
    if key == "resample":
        return {"sfreq": str(params.get("sfreq") or params.get("resample_sfreq", ""))}
    if key == "confound_regression":
        return {"regressors": _regressor_phrase(params)}
    if key == "glm":
        return {
            "hrf_model": str(params.get("hrf_model", "")),
            "noise_model": str(params.get("noise_model", "")),
            "drift_model": str(params.get("drift_model", "")),
            "drift_high_pass": str(params.get("drift_high_pass", "")),
        }
    return {}


# ---- steps with no method prose ----

STEP_SUMMARY = {
    "load": "Read from disk; the pipeline stage comes from the filename.",
    "od_passthrough": "Input was already optical density, so the conversion was skipped.",
    "motion_correction": "Motion correction, using the method named in the settings.",
    "sqm_raw": "Quality metrics measured on the original intensity recording.",
    "sqm": "Quality metrics for this run, grouped by the stage each was measured on.",
    "design_matrix": "Regressors assembled for the fit: conditions, drift and confounds.",
    "glm_fit": "Per-channel model fit; rejected channels are flagged, not dropped.",
    "contrasts": "Contrast estimates derived from the fitted model.",
    "glm_residuals": "What the model left behind, once the fitted signal was removed.",
    "glm_residuals_broadband": "The same regression without the low-pass, so fALFF keeps a full spectrum.",
    "alff": "Amplitude of low-frequency fluctuation, per channel.",
    "fc": "Channel-by-channel correlation within one chromophore.",
    "fisher_z": "Fisher r-to-z of a correlation matrix, for group-level statistics.",
    "fc_roi": "Connectivity between ROI-averaged signals.",
    "fc_seed": "Correlation of one ROI's mean signal with every channel.",
    "group_sqm_raw": "Quality metrics pooled across the members of a dyad.",
    "group_sqm_raw_channels": "The same pooling, kept per channel.",
    "hyper_wtc": "Wavelet coherence between a pair, averaged over a band and one value per channel.",
    "hyper_wtc_pseudo": "The same average against a phase-scrambled partner: the null.",
    "hyper_wtc_roichan": "Channel-level coherences averaged within each ROI.",
    "hyper_isc": "Correlation of each channel of one brain with each channel of the other.",
    "group_hyper_wtc": "Every dyad's channel-level coherence, merged into one table.",
    "group_hyper_wtc_pseudo": "The same merge, over the phase-scrambled null.",
    "group_hyper_wtc_roichan": "The same merge, over ROI means of channel coherences.",
}


def step_summary(step: str | None) -> str:
    return STEP_SUMMARY.get(step or "", "")


# ---- what each metric means ----

# STEP_SUMMARY above describes steps; this describes the numbers those steps produced. The
# report used to print them bare, so a reader who did not already know the vocabulary got
# "GVTD p95  1.001e-02" and no way to act on it.
#
# Each line says what the number is and which way is good, because a value with no
# direction is not actionable. Where the answer is "it depends", say so rather than
# inventing a threshold: several of these are relative measures with no absolute cutoff,
# and GVTD in particular is judged against gvtd_thresh, which is computed per recording.
#
# Per-channel keys are not listed; they are the same quantity as their scalar sibling.
METRIC_SUMMARY = {
    # coupling
    "sci_mean": "Scalp coupling: how well the two wavelengths share a pulse. Near 1 is good; low means poor optode contact.",
    "channel_retention_rate": "Fraction of channels that survived screening. Higher is better.",
    "psp_mean": "Strength of the shared cardiac peak across the two wavelengths, averaged over 10 s windows and then over channels. Higher is a more clearly detected heartbeat.",
    "cp_mean": "How sharply cardiac power concentrates at the pulse frequency, 0 to 1. Closer to 1 is a cleaner peak. Experimental, overlaps PSP.",
    "cp_pass_rate": "Fraction of channels with cardiac power at or above 0.5. Higher is better. Experimental.",

    # raw intensity
    "cv_mean": "Noise relative to a channel's own brightness (SD / mean). Lower is cleaner.",
    "snr_mean": "Signal size relative to its fluctuation (mean / SD), the reciprocal of CV. Higher is better.",
    "snr_pass_rate": "Fraction of channels with SNR above 2. Higher is better.",
    "n_flat_channels": "How many channels carry no variation at all, flat or saturated. Zero is what you want; these are counted as failures in snr_pass_rate but cannot enter the SNR and CV means.",
    "mean_amp_mean": "Average light level reaching the detectors. No universal good value; use it to spot channels far dimmer than their neighbours.",

    # geometry
    "ch_dist_mean": "Average source-detector separation in metres. Descriptive, not a quality judgement.",
    "ch_dist_min": "Shortest source-detector separation in metres.",
    "ch_dist_max": "Longest source-detector separation in metres.",

    # haemoglobin
    "hbo_hbr_corr_mean": "Correlation between HbO and HbR. Strongly negative is physiologically expected; near zero or positive suggests artifact.",
    "cnr_hbo_mean": "How far the evoked HbO response clears its own noise, averaged over channels. Higher is better. Absent on a run with no stimulus annotations.",
    "cnr_hbr_mean": "The same for HbR. HbR falls with a response, so this one runs negative and more negative is better.",
    "cnr_n_epochs": "How many stimulus epochs the CNR was averaged over. Descriptive; a handful of epochs makes the value noisy.",
    "gcor_hbo": "How much every HbO channel moves together. Higher means a stronger shared systemic or global component rather than localised activity.",
    "gcor_hbr": "The same for HbR.",
    "lowfreq_drift_amplitude_hbo": "Peak-to-peak size of the slow HbO baseline wander. Lower is a more stable baseline. Non-standard, may be removed.",
    "lowfreq_drift_amplitude_hbr": "The same for HbR.",

    # spectral. Power is a mean PSD level inside the band, fraction is a sum over the band
    # against the sum over the whole spectrum; they are not the same quantity rescaled.
    "cardiac_band_power_hbo": "Average HbO spectral density inside the cardiac band. Scales with signal amplitude, so it does not compare across subjects.",
    "cardiac_band_power_hbr": "The same for HbR.",
    "cardiac_band_frac_hbo": "Share of this chromophore's total HbO power that sits in the cardiac band, 0 to 1, comparable across subjects. Visible cardiac content confirms real physiology.",
    "cardiac_band_frac_hbr": "The same for HbR.",
    "resp_band_power_hbo": "Average HbO spectral density inside the respiration band.",
    "resp_band_power_hbr": "The same for HbR.",
    "resp_band_frac_hbo": "Share of total HbO power that sits in the respiration band, 0 to 1.",
    "resp_band_frac_hbr": "The same for HbR.",

    # motion and spikes, all measured on optical density.
    # Note which trace gvtd_thresh belongs to: it is computed from the band-passed trace
    # and compared against it, so it is not a cutoff for the unfiltered gvtd_mean/p95.
    "gvtd_mean": "Average whole-montage movement over the run, unfiltered. Lower is less motion. No absolute cutoff, and gvtd_thresh does not apply to it.",
    "gvtd_p95": "The same at the worst moments, the 95th percentile.",
    "gvtd_filt_mean": "Average movement after band-passing to 0.01-0.5 Hz, where head motion lives. This is the trace gvtd_thresh applies to.",
    "gvtd_filt_p95": "The same at the worst moments. Above gvtd_thresh means motion.",
    "gvtd_vstd_mean": "Average movement with each channel scaled by its own SD first, so a few loud channels cannot dominate.",
    "gvtd_vstd_p95": "The same at the worst moments.",
    "gvtd_thresh": "Motion cutoff for this recording, set from the mode of its own band-passed GVTD histogram. Compare it with gvtd_filt_p95, not with gvtd_mean. Each side of a before → after pair is set from its own data, so a fall here is the cutoff following the recording, not motion being removed.",
    "gvtd_num_above_thresh": "Timepoints whose band-passed GVTD exceeds that cutoff.",
    "gvtd_pct_above_thresh": "Those timepoints as a fraction of the recording, roughly how much is motion-contaminated. Lower is cleaner.",
    "spike_count": "Sudden jumps across all channels, counted on the motion-band-filtered derivative so they reflect movement rather than pulse. Lower is better.",
    "spike_pct": "Those jumps as a fraction of all channel-samples. Experimental.",
    "spike_num_frames": "Timepoints where at least a tenth of channels jumped together. Experimental.",
    "spike_pct_frames": "Those timepoints as a fraction of the recording. Experimental.",
    "motion_corrected_frac_mean": "Average fraction of each channel the motion correction actually altered. Experimental.",
    "motion_corrected_num": "Timepoints the correction altered on at least a tenth of channels at once. Experimental.",
    "motion_corrected_pct": "Those timepoints as a fraction of the recording. Experimental.",
    "motion_corrected_n_segments": "How many separate stretches those timepoints form. Experimental.",

    # time
    "pct_data_retained": "Fraction of the recording not covered by BAD annotations. Higher is more usable data.",
}

# The few that decide whether a subject is usable at all. Everything else is context for
# why. Three questions: are enough channels left, is enough time left, and is the signal
# physiological. The report marks these so a reader knows where to look first.
KEY_METRICS = frozenset({
    "channel_retention_rate",   # enough channels
    "pct_data_retained",        # enough time
    "gvtd_pct_above_thresh",    # ... and how much of it is motion
    "sci_mean",                 # the optodes were coupled
    "hbo_hbr_corr_mean",        # what came out looks like haemodynamics
})


# ---- which stage each number was measured on ----
#
# Kept apart from METRIC_SUMMARY because this is a property of *where* the metric is
# computed, not of what it means. Four of these keys are measured at two stages on data
# separated by a bandpass and a resample; without this line a tooltip cannot say which of
# the two the reader is looking at, and the section name is not visible from inside a
# panel. The section-to-stage mapping this follows lives in the SQM record.
_STAGE_RAW = "Measured on the recording as it arrived, before any processing."
_STAGE_MOTION = (
    "Measured across the motion-correction step, on the optical density either side of it."
)
_STAGE_PREPROC = (
    "Measured after Beer-Lambert and before filtering, since a bandpass would otherwise be "
    "measuring itself."
)
_STAGE_BOTH = (
    "Measured on every haemoglobin file the run wrote, so there is one of these per stage: "
    "Beer-Lambert output, then the bandpass, the resample and the confound regression as "
    "each of those ran. Which one you are reading is the section it sits in."
)
_STAGE_RAW_AND_CORRECTED = (
    "Measured on the recording as it arrived, and again on the motion-corrected file, over "
    "the same channels both times. A pair written before → after is those two "
    "numbers: the left one is the recording, the right one is what motion correction left "
    "behind, so the pair says whether the correction removed what it was there to remove. "
    "A single number means this run has no corrected file to compare against."
)

_RAW_METRICS = (
    "channel_retention_rate", "cp_mean", "cp_pass_rate", "n_flat_channels",
    "cv_mean", "snr_mean", "snr_pass_rate", "mean_amp_mean",
    "ch_dist_mean", "ch_dist_min", "ch_dist_max",
)
# the OD-domain families, measured again on the corrected file so the pair subtracts
_RAW_AND_CORRECTED_METRICS = (
    "sci_mean", "psp_mean",
    "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95",
    "gvtd_vstd_mean", "gvtd_vstd_p95", "gvtd_thresh",
    "gvtd_num_above_thresh", "gvtd_pct_above_thresh",
    "spike_count", "spike_pct", "spike_num_frames", "spike_pct_frames",
)
_MOTION_METRICS = (
    "motion_corrected_frac_mean", "motion_corrected_num",
    "motion_corrected_pct", "motion_corrected_n_segments",
)
_PREPROC_METRICS = (
    "lowfreq_drift_amplitude_hbo", "lowfreq_drift_amplitude_hbr",
    "cardiac_band_power_hbo", "cardiac_band_power_hbr",
    "cardiac_band_frac_hbo", "cardiac_band_frac_hbr",
    "resp_band_power_hbo", "resp_band_power_hbr",
    "resp_band_frac_hbo", "resp_band_frac_hbr",
)
_BOTH_METRICS = ("hbo_hbr_corr_mean", "gcor_hbo", "gcor_hbr", "pct_data_retained",
                 "cnr_hbo_mean", "cnr_hbr_mean", "cnr_n_epochs")

METRIC_STAGE = {
    **{k: _STAGE_RAW for k in _RAW_METRICS},
    **{k: _STAGE_RAW_AND_CORRECTED for k in _RAW_AND_CORRECTED_METRICS},
    **{k: _STAGE_MOTION for k in _MOTION_METRICS},
    **{k: _STAGE_PREPROC for k in _PREPROC_METRICS},
    **{k: _STAGE_BOTH for k in _BOTH_METRICS},
}


def metric_summary(metric: str) -> str:
    """What a metric is, which way is good, and what stage it was measured on.

    Returns '' for an undescribed metric, so the report renders a bare number rather than
    an empty tooltip.
    """
    text = METRIC_SUMMARY.get(metric, "")
    if not text:
        return ""
    stage = METRIC_STAGE.get(metric, "")
    return f"{text} {stage}".rstrip()


def is_key_metric(metric: str) -> bool:
    return metric in KEY_METRICS


# ---- what a run actually did ----

def steps_from_sidecars(nirs_dir: Path, mode: str | None = None) -> list[tuple[str, dict[str, str]]]:
    """(steps.toml key, filled slots) for every method step a run recorded, in run order.

    Ordered by depth in the provenance graph, so the sentences follow the data rather than
    the filenames. A step that ran more than once contributes one entry, with the later
    parameters filling anything the first was missing (the GLM's four outputs each carry
    part of the picture).
    """
    from fnirs_pipe.qc.provenance import scan

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for node in sorted(scan(nirs_dir).values(), key=lambda n: (n.depth, n.label)):
        key = boilerplate_key(node.step, node.params, mode)
        if key is None:
            continue
        if key not in merged:
            merged[key] = {}
            order.append(key)
        merged[key] = {**node.params, **merged[key]}

    return [(key, template_slots(key, merged[key])) for key in order]
