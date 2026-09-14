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

from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ---- pipeline step -> steps.toml section ----

_DIRECT = ("od_conversion", "beer_lambert", "resample", "hyper_isc", "hyper_coherence")

# A dyad's coherence is written once per grouping (channels, ROI means, per condition) and
# once more for the null; they are one method sentence, and the band is the same for all.
_WTC_STEPS = ("hyper_wtc", "hyper_wtc_roichan", "hyper_wtc_bycondition",
              "hyper_wtc_bycondition_roichan", "hyper_wtc_pseudo")


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
    if step in _WTC_STEPS:
        return "hyper_wtc"
    if step in ("hyper_isc_roichan", "hyper_isc_pairs"):
        # the same correlation, grouped into regions or listed pair by pair; one sentence
        # covers all three
        return "hyper_isc"
    if step == "hyper_coherence_windowed":
        # the same measure, taken in windows; one sentence covers both
        return "hyper_coherence"
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
        # All three numbers the screening uses, because none of them describes it alone:
        # SCI and PSP are the per-window lines and `min_good_frac` is what actually rejects
        # a channel. Naming only the first two reads as though either could reject on its
        # own. Each falls back to the criteria table for a record written before the run
        # started stamping it.
        from fnirs_pipe.qc.metrics import criterion_cutoffs
        from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S
        cutoffs = criterion_cutoffs()
        psp = params.get("psp_threshold")
        good_frac = params.get("min_good_frac")
        good_frac = cutoffs["good_frac"] if good_frac is None else good_frac
        return {
            "threshold": str(params.get("sci_threshold", "")),
            "psp_threshold": str(psp if psp is not None else cutoffs["psp"]),
            # a share reads as a percentage in a Methods paragraph, and it is formatted from
            # the same value the screening used rather than written out beside it
            "min_good_frac": f"{float(good_frac) * 100:g}%",
            # the screening window, which is pinned and does not follow --window-length. The
            # record's `qc_window_s` is the QC grid and would be the wrong number to quote
            "window_s": f"{SCREEN_WINDOW_S:g}",
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
    if key == "hyper_wtc":
        # the axis the transform covered and the band it was collapsed over are different
        # numbers and the sentence names both; a run that gave no band averaged the whole axis
        wtc_lo, wtc_hi = params.get("wtc_fmin"), params.get("wtc_fmax")
        return {
            "wtc_fmin":  _num(wtc_lo),
            "wtc_fmax":  _num(wtc_hi),
            "band_fmin": _num(params.get("band_fmin", wtc_lo)),
            "band_fmax": _num(params.get("band_fmax", wtc_hi)),
        }
    if key == "hyper_coherence":
        return {"coh_fmin": _num(params.get("coherence_fmin")),
                "coh_fmax": _num(params.get("coherence_fmax"))}
    if key == "hyper_alignment":
        return {"n_subjects": str(params.get("n_subjects", "member"))}
    return {}


def _num(value: Any) -> str:
    """A frequency as the Methods should print it: 0.004 not 0.004000000000000001."""
    if value is None:
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


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
    "hyper_bads": "The channels excluded for this dyad, over the scope named in the settings.",
    "hyper_coherence": "Band-averaged coherence of each homologous channel pair, over the whole recording.",
    "hyper_coherence_windowed": "The same coherence in sliding windows, one value per window and channel.",
    "hyper_sqm": "Quality metrics for the dyad: alignment, coupling and the members' own.",
    "hyper_screening": "Each window's coherence beside the surrogate null drawn for that window.",
    "hyper_usable": "How much of each channel pair both members could use at once, per condition.",
    "group_sqm_raw": "Quality metrics pooled across the members of a dyad.",
    "group_sqm_raw_channels": "The same pooling, kept per channel.",
    "hyper_wtc": "Wavelet coherence between a pair, averaged over a band and one value per channel.",
    "hyper_wtc_pseudo": "The same average against a phase-scrambled partner: the null.",
    "hyper_wtc_roichan": "Channel-level coherences averaged within each ROI.",
    "hyper_isc": "Correlation of each channel of one brain with each channel of the other.",
    "hyper_isc_roichan": "Channel-level correlations averaged within each ROI.",
    "hyper_isc_pairs": "The same correlations as one row per channel pair.",
    "group_hyper_wtc": "Every dyad's channel-level coherence, merged into one table.",
    "group_hyper_wtc_pseudo": "The same merge, over the phase-scrambled null.",
    "group_hyper_wtc_roichan": "The same merge, over ROI means of channel coherences.",
}


def step_summary(step: str | None) -> str:
    return STEP_SUMMARY.get(step or "", "")


# ---- what each metric means ----

# STEP_SUMMARY above describes steps; this describes the numbers those steps produced, so a
# report never prints one bare.
#
# Each line says what the number is and which way is good, because a value with no
# direction is not actionable. Where the answer is "it depends", say so rather than
# inventing a threshold: several of these are relative measures with no absolute cutoff,
# and GVTD in particular is judged against gvtd_thresh, which is computed per recording.
#
# Per-channel keys are not listed; they are the same quantity as their scalar sibling.
METRIC_SUMMARY = {
    # coupling
    "sci_mean": "Scalp coupling over the whole recording: how well the two wavelengths share a pulse, near 1 being good and low meaning poor optode contact. A slow drift shared by both wavelengths lifts it, which is what sci_win_mean is beside it for.",
    "sci_win_mean": "The same coupling measured inside 10 s windows and then averaged, on the grid psp_mean and cv_mean use. Read this one for coupling; the whole-run sci_mean is what the published cutoffs were set on, and the two can disagree about which channel set coupled better.",
    "channel_retention_rate": "Fraction of channels that survived screening. Higher is better.",
    "psp_mean": "Strength of the shared cardiac peak across the two wavelengths, averaged over 10 s windows and then over channels; higher is a more clearly detected heartbeat.",
    "good_frac_mean": "Share of 10 s windows in which SCI and PSP both pass, averaged over channels; higher is better. This is the line a channel is rejected on.",
    "cp_mean": "How peaked one channel's spectrum is inside the cardiac band, 0 to 1; higher is sharper. Experimental, and it never compares the two wavelengths, so read SCI and PSP for coupling.",

    # raw intensity
    "cv_mean": "Noise relative to a channel's own brightness (SD / mean), per wavelength, measured inside 10 s windows and then averaged, lower being cleaner. Over the whole recording it would read the drift instead of the noise.",
    "snr_mean": "Signal size relative to its fluctuation (mean / SD), the exact reciprocal of CV and on the same 10 s windows. Higher is better.",
    "snr_pass_rate": "Fraction of channels whose SNR clears the per-channel line. Higher is better.",
    "n_flat_channels": "How many channels carry no variation at all, flat or saturated; zero is what you want. They count as failures in snr_pass_rate but cannot enter the SNR and CV means.",
    "mean_amp_mean": "Average light level reaching the detectors. No universal good value; use it to spot channels far dimmer than their neighbours.",

    # geometry
    "ch_dist_mean": "Average source-detector separation in metres. Descriptive, not a quality judgement.",
    "ch_dist_min": "Shortest source-detector separation in metres.",
    "ch_dist_max": "Longest source-detector separation in metres.",

    # haemoglobin
    "hbo_hbr_corr_mean": "Correlation between HbO and HbR. Strongly negative is physiologically expected; near zero or positive suggests artifact.",
    "cnr_hbo_mean": "How far the evoked HbO response clears its own noise, averaged over channels; higher is better. Absent on a run with no stimulus annotations.",
    "cnr_hbr_mean": "The same for HbR. HbR falls with a response, so this one runs negative and more negative is better.",
    "cnr_n_epochs": "How many stimulus epochs the CNR was averaged over. Descriptive; a handful of epochs makes the value noisy.",
    "gcor_hbo": "How much every HbO channel moves together, higher meaning a stronger shared systemic or global component rather than localised activity. A pair's before side is the bandpassed signal, since the bandpass alone raises this.",
    "gcor_hbr": "The same for HbR.",
    "lowfreq_drift_amplitude_hbo": "Peak-to-peak of a fitted trend: how far the HbO baseline travelled, not how slowly, so a step moves it more than a slow sag does and lower is more stable. Non-standard, and not comparable between recordings of different length.",
    "lowfreq_drift_amplitude_hbr": "The same for HbR.",

    # spectral. Power is a mean PSD level inside the band, fraction is a sum over the band
    # against the sum over the whole spectrum; they are not the same quantity rescaled.
    "cardiac_band_power_hbo": "Average HbO spectral density inside the cardiac band. Scales with signal amplitude, so it does not compare across subjects, and it rises with any broadband artifact.",
    "cardiac_band_power_hbr": "The same for HbR.",
    "cardiac_band_frac_hbo": "Share of total HbO power sitting in the cardiac band, 0 to 1, comparable across subjects. Spectral content, not quality: motion lifts every band, so read SCI and PSP for whether the pulse is real.",
    "cardiac_band_frac_hbr": "The same for HbR.",
    "resp_band_power_hbo": "Average HbO spectral density inside the respiration band.",
    "resp_band_power_hbr": "The same for HbR.",
    "resp_band_frac_hbo": "Share of total HbO power that sits in the respiration band, 0 to 1.",
    "resp_band_frac_hbr": "The same for HbR.",

    # motion and spikes, all measured on optical density.
    # Note which trace gvtd_thresh belongs to: it is computed from the band-passed trace
    # and compared against it, so it is not a cutoff for the unfiltered gvtd_mean/p95.
    "gvtd_mean": "Average whole-montage movement over the run, unfiltered; lower is less motion. No absolute cutoff, and gvtd_thresh does not apply to it.",
    "gvtd_p95": "The same at the worst moments, the 95th percentile.",
    "gvtd_filt_mean": "Average movement after band-passing to 0.01-0.5 Hz, where head motion lives. This is the trace gvtd_thresh applies to.",
    "gvtd_filt_p95": "The same at the worst moments. Above gvtd_thresh means motion.",
    "gvtd_vstd_mean": "Average movement with each channel scaled by its own SD first, so a few loud channels cannot dominate.",
    "gvtd_vstd_p95": "The same at the worst moments.",
    "gvtd_thresh": "Motion cutoff derived from this recording's own band-passed GVTD histogram; compare it with gvtd_filt_p95, not gvtd_mean. Each side of a pair derives its own, so a fall here is the cutoff following the recording rather than motion being removed.",
    "gvtd_thresh_applied": "The cutoff the counts below were actually taken against: the uncorrected recording's on both sides of a pair, so the two share one yardstick.",
    "gvtd_num_above_thresh": "Timepoints whose band-passed GVTD exceeds gvtd_thresh_applied.",
    "gvtd_pct_above_thresh": "Those timepoints as a fraction of the recording, roughly how much is motion-contaminated; lower is cleaner. Both sides of a pair are counted against the uncorrected cutoff, so a fall is motion removed rather than the cutoff moving.",
    "gvtd_censor_pct": "Fraction of the recording marked BAD_gvtd, larger than gvtd_pct_above_thresh because it also takes the surviving stretches too short to analyse. Nothing was deleted.",
    "gvtd_censor_retained_s": "Seconds left after censoring, in gvtd_censor_n_epochs continuous stretches; this, not the censored fraction, is what an analysis has to work with.",
    "spike_count": "Sudden jumps across all channels, counted on the motion-band-filtered derivative so they reflect movement rather than pulse; lower is better.",
    "spike_pct": "Those jumps as a fraction of all channel-samples. Experimental.",
    "spike_num_frames": "Timepoints where at least a tenth of channels jumped together. Experimental.",
    "spike_pct_frames": "Those timepoints as a fraction of the recording. Experimental.",
    "motion_corrected_frac_mean": "Average fraction of each channel the motion correction actually altered. Experimental.",
    "motion_corrected_num": "Timepoints the correction altered on at least a tenth of channels at once. Experimental.",
    "motion_corrected_pct": "Those timepoints as a fraction of the recording. Experimental.",
    "motion_corrected_n_segments": "How many separate stretches those timepoints form. Experimental.",

    # time
    "pct_data_retained": "Fraction of the recording not covered by BAD annotations; higher is more usable data. A share of duration rather than of channels, so it is one number for every channel set.",
}

# The few that decide whether a subject is usable at all. Everything else is context for
# why. Three questions: are enough channels left, is enough time left, and is the signal
# physiological. The report marks these so a reader knows where to look first.
KEY_METRICS = frozenset({
    "channel_retention_rate",   # enough channels
    "pct_data_retained",        # enough time
    "gvtd_pct_above_thresh",    # ... and how much of it is motion
    "gvtd_filt_p95",            # ... and how bad it got, in units nothing adaptive sets
    "sci_win_mean",             # the optodes were coupled
    "good_frac_mean",           # ... and stayed coupled, which is what rejects a channel
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
    "Measured on every haemoglobin file the run wrote, so which stage you are reading is the "
    "section it sits in."
)
_STAGE_RAW_AND_CORRECTED = (
    "Measured on the recording as it arrived and again on the motion-corrected file, over the "
    "same channels both times."
)

_RAW_METRICS = (
    # good_frac_mean sits here and not with sci/psp, which it is built from: those two are
    # measured again on the corrected file, and the coupled-window count is taken once, at
    # screening, on the optical density as it arrived
    "good_frac_mean",
    "channel_retention_rate", "cp_mean", "n_flat_channels",
    "cv_mean", "snr_mean", "snr_pass_rate", "mean_amp_mean",
    "ch_dist_mean", "ch_dist_min", "ch_dist_max",
)
# the OD-domain families, measured again on the corrected file so the pair subtracts
_RAW_AND_CORRECTED_METRICS = (
    "sci_mean", "sci_win_mean", "psp_mean",
    "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95",
    "gvtd_vstd_mean", "gvtd_vstd_p95", "gvtd_thresh", "gvtd_thresh_applied",
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


# ---- how a metric is printed ----
#
# One row per metric: the label a panel prints, the number format, and where the colouring
# changes. It sits beside METRIC_SUMMARY because a label and its tooltip drift apart the
# moment they live in different files, and because a threshold spread over the report
# templates, the viewer's JavaScript and the GUI is three chances to disagree.
#
# Format is a Python format spec, plus "pct" for a 0-1 fraction written as a percentage.
#
# Direction and thresholds are separate facts and most metrics have only the first. The
# direction is which end is the better one, which METRIC_SUMMARY has always stated in prose
# ("Lower is cleaner"); a threshold is a defensible cutoff, which far fewer metrics have.
# Keeping them together meant a metric with no published cutoff also had no machine-readable
# direction, so anything that needs to rank without judging -- the per-trial heatmap, which
# colours relative to what one recording actually did -- had to hardcode its own copy.
#
# So: direction None means descriptive, with no better end, and nothing may colour or rank
# it. Thresholds None means no established cutoff, and it prints uncoloured rather than
# against an invented one; a threshold without a direction is meaningless and the tests
# reject it. "higher" tests ``v >= ok``, "lower" tests ``v < ok``, which is what each of the
# original three call sites did.

_HIGHER, _LOWER = "higher", "lower"

METRIC_DISPLAY: dict[str, tuple[str, str, "tuple[float, float] | None", "str | None"]] = {
    # coupling
    "sci_mean":                ("Mean SCI (whole run)", ".3f", (0.75, 0.5), _HIGHER),
    # no cutoffs: the published ones were set on the whole-run estimator above, and a
    # windowed number is not entitled to them just for sharing a name
    "sci_win_mean":            ("Mean SCI (10 s)", ".3f", None, _HIGHER),
    "channel_retention_rate":  ("Channel retention", "pct", (0.9, 0.7), _HIGHER),
    "psp_mean":                ("Mean PSP (10 s)", ".3f", None, _HIGHER),
    "good_frac_mean":          ("Coupled windows", "pct", (0.75, 0.5), _HIGHER),
    "cp_mean":                 ("Mean CP (exp.)", ".3f", None, _HIGHER),

    # raw intensity
    "cv_mean":                 ("Mean CV (10 s)", ".3f", None, _LOWER),
    "snr_mean":                ("Mean SNR (10 s)", ".1f", (100, 20), _HIGHER),
    "snr_pass_rate":           ("SNR pass rate", "pct", None, _HIGHER),
    "n_flat_channels":         ("Flat channels", "d", (1, 2), _LOWER),
    "mean_amp_mean":           ("Mean amplitude", ".3e", None, None),

    # geometry
    "ch_dist_mean":            ("Mean separation (m)", ".3f", None, None),
    "ch_dist_min":             ("Min separation (m)", ".3f", None, None),
    "ch_dist_max":             ("Max separation (m)", ".3f", None, None),

    # haemoglobin
    "hbo_hbr_corr_mean":       ("HbO-HbR corr", ".3f", (-0.3, 0.0), _LOWER),
    "cnr_hbo_mean":            ("CNR HbO", ".3f", None, _HIGHER),
    "cnr_hbr_mean":            ("CNR HbR", ".3f", None, _LOWER),
    "cnr_n_epochs":            ("CNR epochs", "d", None, None),
    "gcor_hbo":                ("Global corr HbO", ".3f", None, None),
    "gcor_hbr":                ("Global corr HbR", ".3f", None, None),
    "lowfreq_drift_amplitude_hbo": ("Low-freq drift (HbO)", ".3e", None, _LOWER),
    "lowfreq_drift_amplitude_hbr": ("Low-freq drift (HbR)", ".3e", None, _LOWER),

    # spectral
    "cardiac_band_power_hbo":  ("Cardiac band power (HbO)", ".3e", None, None),
    "cardiac_band_power_hbr":  ("Cardiac band power (HbR)", ".3e", None, None),
    # descriptive, not a quality reading: injected motion raises this sevenfold, so a
    # "higher is better" arrow would mark a contaminated channel as the good one
    "cardiac_band_frac_hbo":   ("Cardiac band (HbO)", "pct", None, None),
    "cardiac_band_frac_hbr":   ("Cardiac band (HbR)", "pct", None, None),
    "resp_band_power_hbo":     ("Resp band power (HbO)", ".3e", None, None),
    "resp_band_power_hbr":     ("Resp band power (HbR)", ".3e", None, None),
    "resp_band_frac_hbo":      ("Resp band (HbO)", "pct", None, None),
    "resp_band_frac_hbr":      ("Resp band (HbR)", "pct", None, None),

    # motion and spikes
    "gvtd_mean":               ("GVTD mean", ".3e", None, _LOWER),
    "gvtd_p95":                ("GVTD p95", ".3e", None, _LOWER),
    "gvtd_filt_mean":          ("GVTD mean 0.01-0.5 Hz", ".3e", None, _LOWER),
    "gvtd_filt_p95":           ("GVTD p95 0.01-0.5 Hz", ".3e", None, _LOWER),
    "gvtd_vstd_mean":          ("GVTD mean (var-normalised)", ".3e", None, _LOWER),
    "gvtd_vstd_p95":           ("GVTD p95 (var-normalised)", ".3e", None, _LOWER),
    "gvtd_thresh":             ("GVTD threshold", ".3e", None, None),
    "gvtd_thresh_applied":     ("GVTD threshold applied", ".3e", None, None),
    "gvtd_num_above_thresh":   ("GVTD motion frames", "d", None, _LOWER),
    "gvtd_pct_above_thresh":   ("GVTD % motion", "pct", None, _LOWER),
    "gvtd_censor_pct":         ("GVTD censored %", "pct", None, _LOWER),
    "gvtd_censor_retained_s":  ("GVTD retained (s)", ".0f", None, _HIGHER),
    "spike_count":             ("Spike count", "d", (1, 10), _LOWER),
    "spike_pct":               ("Spike % (exp.)", "pct", None, _LOWER),
    "spike_num_frames":        ("Spike frames", "d", None, _LOWER),
    "spike_pct_frames":        ("Spike % frames", "pct", None, _LOWER),
    "motion_corrected_frac_mean":  ("Motion corrected fraction (exp.)", "pct", None, None),
    "motion_corrected_num":        ("Motion corrected frames", "d", None, None),
    "motion_corrected_pct":        ("Motion corrected % (exp.)", "pct", None, None),
    "motion_corrected_n_segments": ("Motion corrected segments", "d", None, None),

    # time
    "pct_data_retained":       ("% data retained", "pct", (0.8, 0.6), _HIGHER),
}

MISSING_VALUE = "\u2014"


def metric_label(metric: str, fallback: str | None = None) -> str:
    spec = METRIC_DISPLAY.get(metric)
    if spec is not None:
        return spec[0]
    return metric if fallback is None else fallback


def metric_format(metric: str) -> str:
    spec = METRIC_DISPLAY.get(metric)
    return spec[1] if spec is not None else ".3f"


def metric_direction(metric: str) -> "str | None":
    """"higher", "lower", or None for a metric with no better end.

    None is a real answer and not a gap: mean amplitude, the separations and the GVTD
    threshold are descriptive, so ranking or colouring them would invent a verdict.
    """
    spec = METRIC_DISPLAY.get(metric)
    return spec[3] if spec is not None else None


def higher_is_better(metric: str) -> "bool | None":
    """The direction as a flag, for callers that scale a value within its own range."""
    direction = metric_direction(metric)
    return None if direction is None else direction == _HIGHER


def format_metric(metric: str, value: Any, fmt: str | None = None) -> str:
    """One metric as a panel prints it, or an em dash when the run did not measure it.

    format_metric("sci_mean", 0.8132)          -> "0.813"
    format_metric("pct_data_retained", 0.9241) -> "92.4%"
    format_metric("spike_count", None)         -> "\u2014"

    ``fmt`` overrides the registry for the rare caller that needs a different width; every
    other caller gets the format the metric is defined with, so one number does not print
    to three decimals in one view and two in the next.
    """
    if value is None:
        return MISSING_VALUE
    spec_fmt = fmt or metric_format(metric)
    try:
        if spec_fmt == "pct":
            return f"{float(value) * 100:.1f}%"
        if spec_fmt == "d":
            return f"{int(round(float(value)))}"
        return format(float(value), spec_fmt)
    except (TypeError, ValueError):
        return str(value)


def metric_class(metric: str, value: Any) -> str:
    """CSS class for a metric's value, or '' when the metric has no established cutoff.

    metric_class("sci_mean", 0.81)  -> "qm-ok"
    metric_class("psp_mean", 0.81)  -> ""        (no published threshold)

    The empty string is deliberate and is not a passing verdict: a metric nobody has a
    cutoff for prints in the default colour rather than being called good.
    """
    spec = METRIC_DISPLAY.get(metric)
    if value is None or spec is None or spec[2] is None:
        return ""
    (ok, warn), direction = spec[2], spec[3]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if direction == _LOWER:
        return "qm-ok" if v < ok else "qm-warn" if v < warn else "qm-bad"
    return "qm-ok" if v >= ok else "qm-warn" if v >= warn else "qm-bad"


def metric_rows(
    scalars: dict[str, Any],
    keys: "Sequence[str] | None" = None,
    *,
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    """A scalar panel's rows, ready for whatever renders them.

    metric_rows({"sci_mean": 0.81}, ["sci_mean"])
    -> [{"key": "sci_mean", "label": "Mean SCI", "value": "0.810", "cls": "qm-ok",
         "tip": "Scalp coupling: ...", "key_metric": True}]

    Every view that prints these numbers goes through here, so a threshold or a label moves
    in one place instead of once per renderer. ``keys`` is the print order and defaults to
    every described metric the dict carries; ``skip_missing`` drops what the run did not
    measure rather than printing a row of dashes for it.
    """
    if keys is None:
        keys = [k for k in METRIC_DISPLAY if k in scalars]
    rows: list[dict[str, Any]] = []
    for key in keys:
        value = scalars.get(key)
        if value is None and skip_missing:
            continue
        rows.append({
            "key":        key,
            "label":      metric_label(key),
            "value":      format_metric(key, value),
            "cls":        metric_class(key, value),
            "tip":        metric_summary(key),
            "key_metric": is_key_metric(key),
        })
    return rows


# ---- what a run actually did ----

def steps_from_sidecars(
    nirs_dir: "Path | Sequence[Path]",
    mode: str | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """(steps.toml key, filled slots) for every method step a run recorded, in run order.

    Ordered by depth in the provenance graph, so the sentences follow the data rather than
    the filenames. A step that ran more than once contributes one entry, with the later
    parameters filling anything the first was missing (the GLM's four outputs each carry
    part of the picture).

    Several directories are scanned in the order given and merged the same way, which is
    how a dyad's Methods paragraph continues from a member subject's preprocessing into
    the group's own steps. Directories the caller passes in the wrong order produce
    sentences in the wrong order; nothing here re-sorts across them.
    """
    from fnirs_pipe.qc.common.provenance import scan

    dirs = [nirs_dir] if isinstance(nirs_dir, (str, Path)) else list(nirs_dir)

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for directory in dirs:
        for node in sorted(scan(directory).values(), key=lambda n: (n.depth, n.label)):
            key = boilerplate_key(node.step, node.params, mode)
            if key is None:
                continue
            if key not in merged:
                merged[key] = {}
                order.append(key)
            merged[key] = {**node.params, **merged[key]}

    return [(key, template_slots(key, merged[key])) for key in order]
