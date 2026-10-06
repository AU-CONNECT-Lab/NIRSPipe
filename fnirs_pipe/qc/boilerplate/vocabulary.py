"""The bridge between the step names the pipeline records and the prose describing them.

Two vocabularies exist and they sit at different granularities on purpose, so this maps
rather than renames:

- the pipeline records one ``motion_correction`` step with the method as a parameter,
  while ``steps.toml`` needs one paragraph per method because the citations differ
- the pipeline records one ``bandpass`` step with two cutoffs, while the prose splits
  into bandpass / highpass / lowpass depending on which cutoff was given
- ``glm_fit`` and ``glm_residuals`` are two files from one regression described in a single
  paragraph, a task GLM or a confound regression depending on the conditions they record

Steps with no paragraph are not omissions: reading a file or recording quality metrics is
bookkeeping, not method, and putting it in ``steps.toml`` would leak it into the Methods
section. Those get a plain one-liner here instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fnirs_pipe.pipeline.motion import WAVELET, WAVELET_IQR_FACTOR
from fnirs_pipe.qc.metrics.coupling import CV_WINDOW_S, PSP_WINDOW_S, SCI_WINDOW_S
from fnirs_pipe.qc.metrics.gvtd import GVTD_MOTION_BAND, GVTD_N_STD
from fnirs_pipe.qc.metrics.imu import AGREEMENT_HALF_WINDOW_S
from fnirs_pipe.qc.metrics.motion import SPIKE_CH_FRAC
from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S
from fnirs_pipe.utils import pair_of

# ---- pipeline step -> steps.toml section ----

_DIRECT = ("od_conversion", "beer_lambert", "resample", "hyper_coherence")

# A dyad's coherence is written once for the whole run and once per condition; they are one
# method sentence. The nulls stay out: they record their own crossing, not the table's.
_WTC_STEPS = ("hyper_wtc", "hyper_wtc_bycondition")

# Every table that groups channel pairs into regions, for either measure: one sentence says
# how a region value is formed.
_ROI_STEPS = ("hyper_wtc_roichan", "hyper_wtc_roihom", "hyper_wtc_bycondition_roichan",
              "hyper_wtc_bycondition_roihom", "hyper_isc_roichan")


def boilerplate_key(step: str | None, params: dict[str, Any]) -> str | None:
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
    if step in ("glm_fit", "glm_residuals"):
        # every mode runs this regression; the conditions it modelled are what make it a task
        # GLM, and a tree that never recorded them cannot say which it was
        conditions = params.get("conditions")
        if conditions is None:
            return "regression_unrecorded"
        return "glm" if conditions else "confound_regression"
    if step in _WTC_STEPS:
        return "hyper_wtc_crossed" if params.get("channel_cross") else "hyper_wtc"
    if step in _ROI_STEPS:
        return "hyper_roi"
    if step in ("hyper_isc", "hyper_isc_pairs"):
        # the same correlation as a matrix and as one row per pair
        return "hyper_isc"
    if step in ("alff", "alff_roi"):
        return "alff"
    if step in ("fc", "fc_roi", "fc_seed", "fisher_z"):
        # one correlation over channels, regions or a seed, and its z transform
        return "fc"
    if step == "hyper_coherence_windowed":
        # the same measure, taken in windows; one sentence covers both
        return "hyper_coherence"
    return None


def _series(items: Sequence[str]) -> str:
    """['a'] -> 'a';  ['a', 'b', 'c'] -> 'a, b and c'"""
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _drift_phrase(params: dict[str, Any]) -> str | None:
    """The drift basis as prose, or None when the design carried none.

    Each model names only the parameter that model uses: a polynomial has no high-pass
    cutoff.
    """
    drift = params.get("drift_model")
    if drift == "cosine":
        return ("a discrete cosine drift basis "
                f"(high-pass cutoff: {params.get('drift_high_pass')} Hz)")
    if drift == "polynomial":
        return f"an order-{params.get('drift_order')} polynomial drift basis"
    return None


def _filter_phrase(params: dict[str, Any]) -> str:
    """Name the filter that ran, from the recorded method rather than a fixed family."""
    method, order = params.get("filter_method"), params.get("filter_order")
    if method == "iir":
        return ("a zero-phase Butterworth filter "
                + (f"(order {order}, applied forward and backward)" if order is not None
                   else "applied forward and backward"))
    if method == "fir":
        return "a zero-phase Hamming-windowed FIR filter"
    # a stage written before the method was recorded. Both branches are zero-phase, so this
    # stays true without naming a family the run may not have used
    return "a zero-phase filter"


def _noise_phrase(value: Any) -> str:
    """The noise model as prose, expanding the one spelling that names no order.

    ``auto`` is mne-nirs' own rule and reaches the sidecar unexpanded, so the sentence has
    to say what it stands for.
    """
    value = str(value or "").strip().lower()
    if value == "auto":
        return ("an autoregressive noise model whose order is four times the sampling rate "
                "in Hz (the MNE-NIRS 'auto' setting)")
    if value.startswith("ar_irls"):
        cap = value[len("ar_irls"):]
        cap = cap if cap.isdigit() else "four times the sampling rate in Hz"
        return ("a robust autoregressive model fitted by iteratively reweighted least squares "
                f"(AR-IRLS), its order chosen by BIC up to {cap}")
    if value.startswith("ar") and value[2:].isdigit():
        return f"an autoregressive noise model of order {value[2:]}"
    if value == "ols":
        return "ordinary least squares without prewhitening"
    return "an unspecified noise model"


_HRF_PHRASE = {
    "spm": "the SPM canonical haemodynamic response function",
    "spm + derivative":
        "the SPM canonical haemodynamic response function and its time derivative",
    "spm + derivative + dispersion":
        "the SPM canonical haemodynamic response function and its time and dispersion "
        "derivatives",
    "glover": "the Glover haemodynamic response function",
    "glover + derivative": "the Glover haemodynamic response function and its time derivative",
    "glover + derivative + dispersion":
        "the Glover haemodynamic response function and its time and dispersion derivatives",
    "fir": "a finite impulse response (FIR) basis",
}


def _hrf_phrase(value: Any) -> str:
    value = str(value or "").strip().lower()
    return _HRF_PHRASE.get(value, f"the '{value}' haemodynamic response model")


def _conditions_phrase(params: dict[str, Any]) -> str:
    """How the task regressors were built: boxcars and the HRF, or an FIR basis.

    {"hrf_model": "glover", "stim_dur": 20.0}
      -> "as 20 s boxcars convolved with the Glover haemodynamic response function"
    """
    hrf = str(params.get("hrf_model") or "").strip().lower()
    if hrf == "fir":
        delays = [int(d) for d in params.get("fir_delays") or []]
        if not delays:
            return f"with {_hrf_phrase(hrf)}"
        whole = delays == list(range(delays[0], delays[-1] + 1))
        span = (f"{delays[0]} to {delays[-1]}" if whole and len(delays) > 1
                else _series([str(d) for d in delays]))
        # each delay's column is the event's boxcar shifted, so the duration is part of it
        if params.get("stim_dur") is not None:
            return (f"with {_hrf_phrase(hrf)} ({_num(params['stim_dur'])} s boxcars at "
                    f"delays of {span} scans)")
        if params.get("event_table"):
            return (f"with {_hrf_phrase(hrf)} (boxcars of each event's own duration at "
                    f"delays of {span} scans)")
        return f"with {_hrf_phrase(hrf)} at delays of {span} scans"
    if params.get("stim_dur") is not None:
        return f"as {_num(params['stim_dur'])} s boxcars convolved with {_hrf_phrase(hrf)}"
    if params.get("event_table"):
        return f"as boxcars of each event's own duration convolved with {_hrf_phrase(hrf)}"
    return f"with {_hrf_phrase(hrf)}"


def _regressor_phrase(params: dict[str, Any]) -> str:
    """Name the nuisance columns a regression actually carried.

    {"short_channel": "mean", "drift_model": "cosine", "drift_high_pass": 0.01}
      -> "the mean of the retained short channels for each chromophore and a discrete
          cosine drift basis (high-pass cutoff: 0.01 Hz)"
    """
    parts = []
    sc = params.get("short_channel")
    # a --config TOML can record `true`, which the regression reads as "mean"
    if sc == "mean" or sc is True:
        parts.append("the mean of the retained short channels for each chromophore")
    # not "principal components": every component is kept, so the basis is no reduction
    elif sc == "pca":
        parts.append("an orthogonal basis spanning the retained short channels "
                     "(HbO and HbR decomposed together)")

    aux = [str(name).removeprefix("aux_") for name in params.get("aux_regressors") or []]
    if aux:
        parts.append(f"the auxiliary signals {_series(aux)}")

    if (drift := _drift_phrase(params)) is not None:
        parts.append(drift)

    # the design matrix always holds an intercept, so there is something to say even when
    # neither flag was given, and the sentence stays true rather than naming absent columns
    return _series(parts) if parts else "only a constant term"


_CHROMA_NAME = {"hbo": "HbO", "hbr": "HbR"}


def _screening_scope(params: dict[str, Any]) -> str:
    """Which windows the coupled share was counted over, as the run recorded it."""
    counted = params.get("screen_scope_counted")
    if counted == "task":
        return "the windows inside annotated task blocks"
    if counted is None and params.get("screen_scope") == "task":
        # asked for task blocks, and a tree this old does not say whether it found any
        return "the windows the screening counted"
    return "their windows"


def _screening_slots(params: dict[str, Any]) -> dict[str, str]:
    """The screening sentence's numbers, its counting scope and any channels marked by hand."""
    # Each line falls back to the criteria table for a record that does not carry it.
    from fnirs_pipe.qc.metrics import criterion_cutoffs
    cutoffs = criterion_cutoffs()
    psp = params.get("psp_threshold")
    good_frac = params.get("min_good_frac")
    good_frac = cutoffs["good_frac"] if good_frac is None else good_frac
    low, high = params.get("cardiac_l_freq"), params.get("cardiac_h_freq")
    manual = list(dict.fromkeys(pair_of(c) for c in params.get("bad_channels") or []))
    return {
        "threshold": _num(params.get("sci_threshold")),
        "psp_threshold": _num(psp if psp is not None else cutoffs["psp"]),
        # a share reads as a percentage in a Methods paragraph, and it is formatted from
        # the same value the screening used rather than written out beside it
        "min_good_frac": f"{float(good_frac) * 100:g}%",
        # the screening window, which is pinned and does not follow --window-length. The
        # record's `qc_window_s` is the QC grid and would be the wrong number to quote
        "window_s": f"{SCREEN_WINDOW_S:g}",
        "cardiac_band": (f"in the {_num(low)}–{_num(high)} Hz cardiac band"
                         if low is not None and high is not None else "in the cardiac band"),
        "scope": _screening_scope(params),
        "manual": (f" {'Channels' if len(manual) > 1 else 'Channel'} {_series(manual)} "
                   f"{'were' if len(manual) > 1 else 'was'} also marked as bad by hand."
                   if manual else ""),
    }


def _dpf_phrase(dpf: Any) -> str:
    """'a differential pathlength factor (DPF) of 6', or one per wavelength when they differ."""
    values = list(dict.fromkeys(_num(d) for d in (dpf if isinstance(dpf, (list, tuple))
                                                  else [dpf])))
    if len(values) == 1:
        return f"a differential pathlength factor (DPF) of {values[0]}"
    # MNE pairs the factors with the wavelengths sorted ascending, whatever order they came in
    return (f"differential pathlength factors (DPF) of {_series(values)}, in ascending order "
            "of wavelength")


def _condition_route(params: dict[str, Any]) -> str:
    """How the per-condition values were taken, or nothing for a run without conditions."""
    if not params.get("condition_windows_s"):
        return ""
    pad = params.get("wtc_cond_pad_s")
    if pad is None:
        return " Per-condition values averaged the same maps over each condition's span only."
    if pad == 0:
        return (" Per-condition values came from a separate transform of each condition, cut "
                "at its own boundaries.")
    return (" Per-condition values came from a separate transform of each condition, cut "
            f"with up to {_num(round(float(pad), 1))} s of recording on either side and averaged "
            "over the condition's span only.")


def _wtc_slots(params: dict[str, Any]) -> dict[str, str]:
    """The coherence sentence's band and the clauses only some runs need."""
    wtc_lo, wtc_hi = params.get("wtc_fmin"), params.get("wtc_fmax")
    whiten_s, window = params.get("wtc_whiten_s"), params.get("analysis_window_s")
    return {
        "chroma": _series([_CHROMA_NAME.get(c, c) for c in params.get("chroma") or ["hbo"]]),
        # a run that gave no band averaged the whole computed axis
        "band_fmin": _num(params.get("band_fmin", wtc_lo)),
        "band_fmax": _num(params.get("band_fmax", wtc_hi)),
        "coi": ", excluding the cone of influence" if params.get("mask_coi") else "",
        "whiten": (" Before the transform, each long channel was prewhitened with an "
                   f"autoregressive model of order {params.get('wtc_whiten_order')} "
                   f"({_num(whiten_s)} s)." if whiten_s else ""),
        "conditions": _condition_route(params),
        "window": (f" Only {_num(window[0])}–{_num(window[1])} s of the aligned recordings "
                   "were analysed." if window else ""),
        "bads": (" A channel rejected in any of a member's runs was left out of all of them."
                 if params.get("bads_scope") == "subject" else ""),
    }


def _isc_slots(params: dict[str, Any]) -> dict[str, str]:
    """The correlation's optional steps, in the order they ran: band limit, whitening, lag."""
    before = []
    low, high = params.get("isc_band_hz") or (None, None)
    if low is not None and high is not None:
        before.append(f"band-pass filtered to {_num(low)}–{_num(high)} Hz")
    elif low is not None:
        before.append(f"high-pass filtered at {_num(low)} Hz")
    elif high is not None:
        before.append(f"low-pass filtered at {_num(high)} Hz")
    if params.get("isc_whiten_max_order"):
        before.append("prewhitened with an autoregressive model whose order was chosen by BIC "
                      f"(at most {params['isc_whiten_max_order']})")
    options = (f" Before correlating, each signal was {' and '.join(before)}."
               if before else "")
    if params.get("isc_max_lag_s"):
        options += (" The correlation was taken at the lag of largest magnitude within "
                    f"±{_num(params['isc_max_lag_s'])} s, keeping its sign.")
    return {"options": options}


def template_slots(key: str, params: dict[str, Any]) -> dict[str, str]:
    """Fill a section's {slots} from a sidecar's parameters.

    Note the naming: a sidecar's ``high_pass`` is the *lower* edge of the band, which the
    filter templates call ``l_freq``.
    """
    if key == "sci_marking":
        return _screening_slots(params)
    if key == "motion_wavelet":
        # constants of the package rather than of the run, so they are read here
        return {"wavelet": WAVELET, "iqr_factor": f"{WAVELET_IQR_FACTOR:g}"}
    if key == "beer_lambert":
        return {"dpf": _dpf_phrase(params.get("dpf"))}
    if key == "bandpass":
        return {"l_freq": _num(params.get("high_pass")), "h_freq": _num(params.get("low_pass")),
                "filter": _filter_phrase(params)}
    if key == "highpass":
        return {"l_freq": _num(params.get("high_pass")), "filter": _filter_phrase(params)}
    if key == "lowpass":
        return {"h_freq": _num(params.get("low_pass")), "filter": _filter_phrase(params)}
    if key == "resample":
        return {"sfreq": _num(params.get("sfreq") or params.get("resample_sfreq"))}
    if key == "confound_regression":
        return {"regressors": _regressor_phrase(params),
                "noise_model": _noise_phrase(params.get("noise_model"))}
    if key == "glm":
        return {"conditions": _conditions_phrase(params),
                "regressors": _regressor_phrase(params),
                "noise_model": _noise_phrase(params.get("noise_model"))}
    if key in ("hyper_wtc", "hyper_wtc_crossed"):
        return _wtc_slots(params)
    if key == "hyper_isc":
        return _isc_slots(params)
    if key == "alff":
        return {"l_freq": _num(params.get("high_pass")), "h_freq": _num(params.get("low_pass"))}
    if key == "hyper_roi":
        least = int(params.get("roi_min_channels") or 1)
        return {"min_channels": (" A region pair was left empty where either member "
                                 f"contributed fewer than {least} channels."
                                 if least > 1 else "")}
    if key == "hyper_input":
        desc = params.get("desc")
        return {"stage": f"desc-{desc}" if desc else "input"}
    if key == "hyper_coherence":
        return {"coh_fmin": _num(params.get("coherence_fmin")),
                "coh_fmax": _num(params.get("coherence_fmax"))}
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
    "crop": "A stretch cut out of the recording; its window and margin are in the sidecar.",
    "aux_extract": "The recording's auxiliary channels, copied to a table beside the haemoglobin file.",
    "wtc_reband": "Band means re-averaged over a new band from the saved coherence maps.",
    "od_passthrough": "Input was already optical density, so the conversion was skipped.",
    "motion_correction": "Motion correction, using the method named in the settings.",
    "sqm_raw": "Quality metrics measured on the original intensity recording.",
    "sqm": "Quality metrics for this run, grouped by the stage each was measured on.",
    "design_matrix": "Regressors assembled for the fit: the conditions on a task run, plus drift and confounds.",
    "glm_fit": "Per-channel model fit; rejected channels are flagged, not dropped.",
    "contrasts": "Contrast estimates derived from the fitted model.",
    "glm_residuals": "What the model left behind, once the fitted signal was removed.",
    "glm_residuals_broadband": "The same confound regression on data that skipped the bandpass entirely, the drift model its only detrend, so fALFF keeps a full spectrum.",
    "alff": "Amplitude of low-frequency fluctuation, per channel.",
    "alff_roi": "Amplitude of low-frequency fluctuation averaged over each ROI's channels.",
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
    "group_sqm_raw": "Each member's quality metrics, one row per member.",
    "group_sqm_raw_channels": "Per member and channel: whole-run SCI and whether the channel was rejected.",
    "hyper_wtc": "Wavelet coherence between a pair, averaged over a band and one value per channel.",
    "hyper_wtc_phasenull": "The same average against a phase-scrambled partner: the null.",
    "hyper_wtc_roichan": "Channel-level coherences averaged within each ROI, or per pair of ROIs on a crossed run, every pairing between them: the ROI number to report.",
    "hyper_wtc_roihom": "Channel-level coherences averaged over each ROI's homologous channel pairs alone, the subset a homologous-only design reports.",
    "hyper_wtc_roihom_phasenull": "The null for that ROI mean, its iterations grouped into regions before they were summarised.",
    "hyper_wtc_bycondition_roihom_phasenull": "The phase-scrambled null for each condition's homologous ROI means, each iteration grouped into regions before the iterations were summarised.",
    "hyper_wtc_roichan_phasenull": "The phase-scrambled null for the ROI x ROI coherence matrix, each iteration grouped into region pairs before the iterations were summarised.",
    "hyper_wtc_bycondition_roichan_phasenull": "The phase-scrambled null for each condition's ROI x ROI coherence matrix, each iteration grouped into region pairs before the iterations were summarised.",
    "hyper_wtc_bycondition_roichan_pairnull": "The re-paired null for each condition's ROI x ROI coherence matrix, each stand-in grouped into region pairs before the stand-ins were summarised.",
    "hyper_isc": "Correlation of each channel of one brain with each channel of the other.",
    "hyper_isc_roichan": "Channel-level correlations averaged within each ROI.",
    "hyper_isc_pairs": "The same correlations as one row per channel pair.",
    "hyper_merge": "Every dyad's table of one kind, merged into one, the dyad and the task kept as columns.",
}


def step_summary(step: str | None) -> str:
    return STEP_SUMMARY.get(step or "", "")


# ---- what each metric means ----

# STEP_SUMMARY above describes steps; this describes the numbers those steps produced, so a
# report never prints one bare.
#
# Each line says what the number is and which way is good. Where the answer is "it
# depends", say so rather than inventing a threshold: several of these are relative measures
# with no absolute cutoff, and GVTD in particular is judged against gvtd_thresh, which is
# computed per recording.
#
# Per-channel keys are not listed; they are the same quantity as their scalar sibling.
METRIC_SUMMARY = {
    # coupling
    "sci_mean": "Scalp coupling over the whole recording: how well the two wavelengths share a pulse, near 1 being good and low meaning poor optode contact. A few loud stretches shared by both wavelengths, such as movement, outweigh the rest of a whole-run correlation, which is what sci_win_mean is beside it for.",
    "sci_win_mean": f"The same coupling measured inside {SCI_WINDOW_S:g} s windows and then averaged, on the window grid psp_mean and cv_mean use. Read this one for coupling; it and sci_mean can disagree about which channel set coupled better.",
    "channel_retention_rate": "Fraction of channels not marked bad, by screening, by --bad-channels or for non-finite samples. Higher is better.",
    "psp_mean": f"Strength of the shared cardiac peak across the two wavelengths, averaged over {PSP_WINDOW_S:g} s windows and then over channels; higher is a more clearly detected heartbeat.",
    "good_frac_mean": f"Share of {SCREEN_WINDOW_S:g} s windows in which SCI and PSP both pass, averaged over channels; higher is better. This is the line a channel is rejected on.",
    "cp_mean": "How peaked one channel's spectrum is inside the cardiac band, 0 to 1. Experimental: it never compares the two wavelengths, so read SCI and PSP for coupling, and it rises when the pulse is removed, so it has no better end.",

    # raw intensity
    "cv_mean": f"Noise relative to a channel's own brightness (SD / mean), per wavelength, measured inside {CV_WINDOW_S:g} s windows and then averaged, lower being cleaner.",
    "snr_mean": f"Mean / SD per channel, the reciprocal of that channel's {CV_WINDOW_S:g} s CV, then averaged over channels, so it is not 1 / cv_mean and one near-constant channel can inflate it. Higher is better.",
    "snr_pass_rate": "Fraction of channels whose SNR clears the per-channel line. Higher is better.",
    "n_flat_channels": "How many channels carry no variation at all; zero is what you want. They fail snr_pass_rate and stay out of both means; a saturated channel with any residual noise is not counted here and inflates snr_mean.",
    "mean_amp_mean": "Average light level reaching the detectors. No universal good value; use it to spot channels far dimmer than their neighbours.",

    # geometry
    "ch_dist_mean": "Average source-detector separation in metres. Descriptive, not a quality judgement.",
    "ch_dist_min": "Shortest source-detector separation in metres.",
    "ch_dist_max": "Longest source-detector separation in metres.",

    # haemoglobin
    "hbo_hbr_corr_mean": "Correlation between HbO and HbR. A cortical response pushes it negative and shared systemic or motion signals push it positive, and it moves with the passband, so compare runs at the same stage rather than against a fixed value.",
    "cnr_hbo_mean": "How far the evoked HbO response clears its own noise, averaged over channels; higher is better. Absent on a run with no stimulus annotations.",
    "cnr_hbr_mean": "The same for HbR. HbR falls with a response, so this one runs negative and more negative is better.",
    "cnr_n_epochs": "How many stimulus epochs the CNR was averaged over. Descriptive; a handful of epochs makes the value noisy.",
    "gcor_hbo": "How much every HbO channel moves together, higher meaning a stronger shared systemic or global component rather than localised activity. A pair's before side is the bandpassed signal.",
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
    "gvtd_mean": "Unfiltered GVTD of this channel set, averaged over the run: the value other implementations report, kept for comparison with them and shown in no report. Not validated as a motion index and with no cutoff of its own (gvtd_thresh does not apply), so read gvtd_filt_mean for movement.",
    "gvtd_p95": "The same at the worst moments, the 95th percentile.",
    "gvtd_filt_mean": f"Average movement after band-passing to {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz, where head motion lives. This is the trace gvtd_thresh applies to.",
    "gvtd_filt_p95": "The same at the worst moments, the 95th percentile.",
    "gvtd_vstd_mean": "Unfiltered GVTD with each channel's derivative scaled to unit SD, so loud channels cannot dominate; recorded, shown in no report. Its mean square is 1 by construction, so the run mean shows how bursty the trace is rather than how much it moved, and it falls as motion concentrates.",
    "gvtd_vstd_p95": "The same at the 95th percentile, which also falls when motion fills under 5% of the run.",
    "gvtd_thresh": f"Motion cutoff from this recording's own band-passed GVTD histogram: its mode plus {GVTD_N_STD:g} times the spread below the mode; read it against gvtd_filt_p95. Each side of a pair sets its own, so a change here is not motion removed.",
    "gvtd_thresh_applied": "The cutoff the counts below were actually taken against: the uncorrected recording's on both sides of a pair, so the two share one yardstick.",
    "gvtd_num_above_thresh": "Timepoints whose band-passed GVTD exceeds gvtd_thresh_applied.",
    "gvtd_pct_above_thresh": "Timepoints well clear of the run's own quiet level (above gvtd_thresh_applied) as a fraction of the recording, not every frame with some movement in it; lower is cleaner. Both sides of a pair are counted against the uncorrected cutoff, so a fall is motion removed rather than the cutoff moving.",
    "gvtd_censor_pct": "Fraction of the recording marked BAD_gvtd: samples above the censoring cutoff, which --gvtd-censor-n-std sets and which by default is the gvtd_thresh rule, plus surviving stretches too short to analyse. Nothing was deleted.",
    "gvtd_censor_retained_s": "Seconds left after censoring, in gvtd_censor_n_epochs continuous stretches; this, not the censored fraction, is what an analysis has to work with.",
    "spike_count": "Sudden jumps across all channels, counted on the motion-band-filtered derivative so they reflect movement rather than pulse; lower is better.",
    "spike_pct": "Those jumps as a fraction of all channel-samples. Experimental.",
    "spike_num_frames": f"Timepoints where at least {100 * SPIKE_CH_FRAC:g}% of channels jumped together. Experimental.",
    "spike_pct_frames": "Those timepoints as a fraction of the recording. Experimental.",
    "motion_corrected_frac_mean": "Average fraction of each channel's samples where the correction changed abruptly, its frame-to-frame change exceeding the channel's own noise. Experimental.",
    "motion_corrected_num": f"Timepoints where at least {100 * SPIKE_CH_FRAC:g}% of channels were corrected abruptly at once. Experimental.",
    "motion_corrected_pct": "Those timepoints as a fraction of the recording. Experimental.",
    "motion_corrected_n_segments": "How many separate stretches those timepoints form. Experimental.",

    # motion, measured by the sensor rather than inferred from the optical data
    "gyro_speed_mean": "Descriptive: how fast the head turned on average, from the gyroscope (each axis minus its median, then the magnitude), in the unit the recording declares (gyro_speed_unit). Not optical, so it belongs to no channel set.",
    "gyro_speed_p95": "Descriptive: the same at the worst moments, the 95th percentile.",
    "accel_jerk_mean": "Descriptive: how abruptly the head's acceleration changed on average, from the accelerometer's time derivative, which removes gravity, in the unit the recording declares (accel_jerk_unit). Not optical, so it belongs to no channel set.",
    "accel_jerk_p95": "Descriptive: the same at the worst moments, the 95th percentile.",
    "gyro_speed_gvtd_rho": f"Spearman correlation, frame by frame, between the band-passed GVTD of the channel set a run is judged on and the gyroscope speed averaged over {AGREEMENT_HALF_WINDOW_S:g} s either side: how closely the optical motion index follows the sensor on this run. Recorded, not shown in the reports.",
    "accel_jerk_gvtd_rho": "The same against the accelerometer jerk.",

    # time
    "pct_data_retained": "Fraction of the recording not covered by BAD annotations; higher is more usable data. A share of duration rather than of channels, so it is one number for every channel set.",
}

# The few that decide whether a subject is usable at all. Everything else is context for
# why. Three questions: are enough channels left, is enough time left, and is the signal
# physiological. The report marks these so a reader knows where to look first.
KEY_METRICS = frozenset({
    "channel_retention_rate",   # enough channels
    "pct_data_retained",        # enough time
    "gvtd_pct_above_thresh",    # ... and how much of it clearly moved
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
    "Measured after Beer-Lambert and before filtering."
)
_STAGE_BOTH = (
    "Measured on each haemoglobin stage the record covers (preproc, filtered, resampled, "
    "errts), so which stage you are reading is the section it sits in."
)
_STAGE_RAW_AND_CORRECTED = (
    "Measured on the recording as it arrived and again on the motion-corrected file, over the "
    "same channels both times."
)

# A condition page reads GVTD off the corrected file's windows alone, so the whole-run line
# above would name a second file and a percentile the page does not hold.
_STAGE_CONDITION_MOTION = (
    "On a condition page: the motion-corrected file's windows sliced to this condition, so a "
    "95th percentile is the mean of the windows' own and the threshold is that file's own."
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


def metric_summary(metric: str, condition: bool = False) -> str:
    """What a metric is, which way is good, and what stage it was measured on.

    Returns '' for an undescribed metric, so the report renders a bare number rather than
    an empty tooltip. ``condition`` is a condition page, whose GVTD numbers come from one file.
    """
    text = METRIC_SUMMARY.get(metric, "")
    if not text:
        return ""
    stage = (_STAGE_CONDITION_MOTION if condition and metric.startswith("gvtd_")
             and metric in _RAW_AND_CORRECTED_METRICS else METRIC_STAGE.get(metric, ""))
    return f"{text} {stage}".rstrip()


def is_key_metric(metric: str) -> bool:
    return metric in KEY_METRICS


# ---- how a metric is printed ----
#
# One row per metric: the label a panel prints, the number format, and where the colouring
# changes. It sits beside METRIC_SUMMARY so a label, its tooltip and its threshold are
# defined once for the report templates, the viewer's JavaScript and the GUI.
#
# Format is a Python format spec, plus "pct" for a 0-1 fraction written as a percentage.
#
# Direction and thresholds are separate facts and most metrics have only the first. The
# direction is which end is the better one, which METRIC_SUMMARY states in prose
# ("Lower is cleaner"); a threshold is a defensible cutoff, which far fewer metrics have.
# Kept apart, a metric with no published cutoff still has a machine-readable direction, so
# anything that ranks without judging (the per-trial heatmap, which colours relative to
# what one recording actually did) needs no copy of its own.
#
# So: direction None means descriptive, with no better end, and nothing may colour or rank
# it. Thresholds None means no established cutoff, and it prints uncoloured rather than
# against an invented one; a threshold without a direction is meaningless and the tests
# reject it. "higher" tests ``v >= ok``, "lower" tests ``v < ok``.

_HIGHER, _LOWER = "higher", "lower"

METRIC_DISPLAY: dict[str, tuple[str, str, "tuple[float, float] | None", "str | None"]] = {
    # coupling
    "sci_mean":                ("Mean SCI (whole run)", ".3f", (0.75, 0.5), _HIGHER),
    # no cutoffs: the published ones are for the whole-run estimator above
    "sci_win_mean":            (f"Mean SCI ({SCI_WINDOW_S:g} s)", ".3f", None, _HIGHER),
    "channel_retention_rate":  ("Channel retention", "pct", (0.9, 0.7), _HIGHER),
    "psp_mean":                (f"Mean PSP ({PSP_WINDOW_S:g} s)", ".3f", None, _HIGHER),
    "good_frac_mean":          ("Coupled windows", "pct", (0.75, 0.5), _HIGHER),
    "cp_mean":                 ("Mean CP (exp.)", ".3f", None, None),

    # raw intensity
    "cv_mean":                 (f"Mean CV ({CV_WINDOW_S:g} s)", ".3f", None, _LOWER),
    "snr_mean":                (f"Mean SNR ({CV_WINDOW_S:g} s)", ".1f", None, _HIGHER),
    "snr_pass_rate":           ("SNR pass rate", "pct", None, _HIGHER),
    "n_flat_channels":         ("Flat channels", "d", (1, 2), _LOWER),
    "mean_amp_mean":           ("Mean amplitude", ".3e", None, None),

    # geometry
    "ch_dist_mean":            ("Mean separation (m)", ".3f", None, None),
    "ch_dist_min":             ("Min separation (m)", ".3f", None, None),
    "ch_dist_max":             ("Max separation (m)", ".3f", None, None),

    # haemoglobin
    "hbo_hbr_corr_mean":       ("HbO-HbR corr", ".3f", None, _LOWER),
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
    # descriptive, not a quality reading: motion raises it, so it has no better end
    "cardiac_band_frac_hbo":   ("Cardiac band (HbO)", "pct", None, None),
    "cardiac_band_frac_hbr":   ("Cardiac band (HbR)", "pct", None, None),
    "resp_band_power_hbo":     ("Resp band power (HbO)", ".3e", None, None),
    "resp_band_power_hbr":     ("Resp band power (HbR)", ".3e", None, None),
    "resp_band_frac_hbo":      ("Resp band (HbO)", "pct", None, None),
    "resp_band_frac_hbr":      ("Resp band (HbR)", "pct", None, None),

    # motion and spikes
    "gvtd_mean":               ("GVTD mean", ".3e", None, _LOWER),
    "gvtd_p95":                ("GVTD p95", ".3e", None, _LOWER),
    "gvtd_filt_mean":          (f"GVTD mean {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz", ".3e", None, _LOWER),
    "gvtd_filt_p95":           (f"GVTD p95 {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz", ".3e", None, _LOWER),
    "gvtd_vstd_mean":          ("GVTD mean (var-normalised)", ".3e", None, None),
    "gvtd_vstd_p95":           ("GVTD p95 (var-normalised)", ".3e", None, None),
    "gvtd_thresh":             ("GVTD threshold", ".3e", None, None),
    "gvtd_thresh_applied":     ("GVTD threshold applied", ".3e", None, None),
    "gvtd_num_above_thresh":   ("GVTD frames above threshold", "d", None, _LOWER),
    "gvtd_pct_above_thresh":   ("GVTD % above threshold", "pct", None, _LOWER),
    "gvtd_censor_pct":         ("GVTD censored %", "pct", None, _LOWER),
    "gvtd_censor_retained_s":  ("GVTD retained (s)", ".0f", None, _HIGHER),
    "spike_count":             ("Spike count", "d", None, _LOWER),
    "spike_pct":               ("Spike % (exp.)", "pct", None, _LOWER),
    "spike_num_frames":        ("Spike frames", "d", None, _LOWER),
    "spike_pct_frames":        ("Spike % frames", "pct", None, _LOWER),
    "motion_corrected_frac_mean":  ("Motion corrected fraction (exp.)", "pct", None, None),
    "motion_corrected_num":        ("Motion corrected frames", "d", None, None),
    "motion_corrected_pct":        ("Motion corrected % (exp.)", "pct", None, None),
    "motion_corrected_n_segments": ("Motion corrected segments", "d", None, None),

    # motion from the sensor; descriptive, since how much a head moved is not a verdict on the data
    "gyro_speed_mean":         ("Gyroscope speed mean", ".3g", None, None),
    "gyro_speed_p95":          ("Gyroscope speed p95", ".3g", None, None),
    "accel_jerk_mean":         ("Accelerometer jerk mean", ".3g", None, None),
    "accel_jerk_p95":          ("Accelerometer jerk p95", ".3g", None, None),

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
    condition: bool = False,
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
            "tip":        metric_summary(key, condition),
            "key_metric": is_key_metric(key),
        })
    return rows


# ---- what a run actually did ----

# Within one depth of the graph, the order sentences read in: a dyad's coherence, its
# correlation, then how both were grouped into regions; a run's regression before ALFF and FC.
_RANK = {"hyper_isc": 1, "hyper_roi": 2, "alff": 3, "fc": 4}


def _describe(nodes) -> list[tuple[str, dict[str, str]]]:
    """(steps.toml key, filled slots) for the method steps among ``nodes``, in data order.

    A step that ran more than once contributes one entry, the later files filling only what
    the earlier ones left out (a regression's fit table and its residual each carry part of
    the picture).
    """
    keyed = [(node, boilerplate_key(node.step, node.params)) for node in nodes]
    keyed = sorted(((node, key) for node, key in keyed if key is not None),
                   key=lambda nk: (nk[0].depth, _RANK.get(nk[1], 0), nk[0].label))
    merged: dict[str, dict[str, Any]] = {}
    for node, key in keyed:
        merged[key] = {**node.params, **merged.get(key, {})}
    return [(key, template_slots(key, params)) for key, params in merged.items()]


def steps_from_sidecars(
    nirs_dir: Path, label: str | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """Every method step the sidecars under ``nirs_dir`` record, ordered by graph depth.

    ``label`` is a run stem (``sub-01_task-rest``) and keeps another run's files out of the
    paragraph, the way :func:`~fnirs_pipe.qc.common.provenance.scan` scopes its graph.
    """
    from fnirs_pipe.qc.common.provenance import scan

    return _describe(scan(nirs_dir, label=label).values())


def steps_from_lineage(path: "str | Path") -> "list[tuple[str, dict[str, str]]] | None":
    """The method steps behind one file, read up its chain of ``Sources``.

    ::

      .../sub-01_task-rest_desc-preproc_nirs.snirf
        -> od_conversion, sci_marking, motion_tddr, beer_lambert   (no filter, no regression)

    What a later stage did to the same recording is not in the chain, so a reader of an
    earlier stage is never described as having read the later one. None when the file's own
    record cannot be found, which is a moved tree or an input this package did not write.
    """
    from fnirs_pipe.qc.common.provenance import _key, scan

    nodes = scan(Path(path).parent)
    start = _key(path)
    if start not in nodes or nodes[start].step is None:
        return None
    chain: dict = {}
    todo = [start]
    while todo:
        key = todo.pop()
        if key in chain or key not in nodes:
            continue
        chain[key] = nodes[key]
        todo += nodes[key].sources
    return _describe(chain.values())
