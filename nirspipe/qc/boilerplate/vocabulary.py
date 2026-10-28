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

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nirspipe.pipeline import motion
from nirspipe.pipeline.population import age_span
from nirspipe.qc.metrics.coupling import CV_WINDOW_S, PSP_WINDOW_S, SCI_WINDOW_S
from nirspipe.qc.metrics.gvtd import GVTD_MOTION_BAND, GVTD_N_STD
from nirspipe.qc.metrics.motion import SPIKE_CH_FRAC
from nirspipe.qc.metrics.windowed import SCREEN_WINDOW_S
from nirspipe.utils import pair_of
from nirspipe.utils.spans import EDGE

# ---- pipeline step -> steps.toml section ----

_DIRECT = ("od_conversion", "beer_lambert", "resample")

# how a placement read the file's coordinates, for each method that is not a registration
_OPTODE_FRAMES = {"head": "the template's head frame",
                  "fsaverage_head": "fsaverage's head frame",
                  "mni": "fsaverage surface coordinates (MNI305)"}

# A dyad's coherence table, the whole run and the conditions in one. The nulls stay out:
# they record their own crossing, not the table's.
_WTC_STEPS = ("hyper_wtc",)

# Every table that groups channel pairs into regions, for either measure: one sentence says
# how a region value is formed.
_ROI_STEPS = ("hyper_wtc_roichan", "hyper_wtc_roihom", "hyper_isc_roichan")

# The cohort tables nirspipe-hyper-groupnull writes from either null's draws: one test.
_GROUPNULL_STEPS = tuple(f"hyper_{null}_null_{table}" for null in ("repaired", "phase")
                         for table in ("by_cell", "by_occasion", "cohort"))

# Every table the phase-scrambled null writes; its level file, like the re-paired one's, is
# left out.
_PHASENULL_STEPS = ("hyper_wtc_phasenull", "hyper_wtc_bycondition_phasenull_draws",
                    "hyper_wtc_roihom_phasenull", "hyper_wtc_roichan_phasenull")
# Every table the re-paired null writes, coherence and correlation, summary and draws: one
# sentence says how its stand-ins were drawn. The level file carries too little to fill it.
_PAIRNULL_STEPS = ("hyper_wtc_bycondition_pairnull", "hyper_wtc_bycondition_pairnull_draws",
                   "hyper_wtc_bycondition_roihom_pairnull",
                   "hyper_wtc_bycondition_roichan_pairnull", "hyper_isc_pairnull",
                   "hyper_isc_bycondition_pairnull", "hyper_isc_bycondition_pairnull_draws")

def boilerplate_key(step: str | None, params: dict[str, Any]) -> str | None:
    """Section of steps.toml describing this step, or None when it has no method prose."""
    if not step:
        return None
    if step in _DIRECT:
        return step
    if step == "sci_pruning":
        return "sci_marking"
    if step == "optode_placement":
        method = params.get("method")
        if method == "landmarks":
            return "optode_landmarks"
        return "optode_frame" if method in _OPTODE_FRAMES else None
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
        # every mode runs this regression; the conditions it modelled are what make it a task GLM
        return "glm" if params["conditions"] else "confound_regression"
    if step in _WTC_STEPS:
        return "hyper_wtc_crossed" if params.get("channel_cross") else "hyper_wtc"
    if step in _ROI_STEPS:
        return "hyper_roi"
    if step in ("hyper_isc", "hyper_isc_pairs"):
        # the same correlation as a matrix and as one row per pair, paired as the coherence
        return "hyper_isc_crossed" if params.get("channel_cross") else "hyper_isc"
    if step in ("alff", "alff_roi"):
        return "alff"
    if step in ("fc", "fc_roi", "fc_seed", "fisher_z"):
        # one correlation over channels, regions or a seed, and its z transform
        return "fc"
    if step == "hyper_wtc_phasescale":
        # the phase per frequency, whole run and per condition
        return "hyper_phasescale"
    if step in _GROUPNULL_STEPS:
        return "hyper_groupnull"
    if step in _PHASENULL_STEPS:
        return "hyper_phasenull"
    if step in _PAIRNULL_STEPS:
        return "hyper_pairnull"
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
        cutoff = _num(params.get("drift_high_pass"))
        auto = params.get("drift_high_pass_auto")
        how = ""
        if auto:
            how = (", set from the run's events to 1/(2T), T being the longest interval "
                   f"between two trials of one condition ({auto['condition']}, "
                   f"{_num(auto['interval_s'])} s)")
            how += {"cap": f", and capped at {cutoff} Hz",
                    "high_pass": ", and raised to the data's high-pass cutoff, which the "
                                 "basis has to cover"}.get(auto.get("bound"), "")
        return f"a discrete cosine drift basis (high-pass cutoff: {cutoff} Hz{how})"
    if drift == "polynomial":
        return f"an order-{params.get('drift_order')} polynomial drift basis"
    return None

def _filter_phrase(params: dict[str, Any]) -> str:
    """Name the filter that ran, from the recorded method rather than a fixed family."""
    if params["filter_method"] == "fir":
        return "a zero-phase Hamming-windowed FIR filter"
    return f"a zero-phase Butterworth filter (order {params['filter_order']}, applied forward and backward)"

_P_CORRECTION_NAMES = {
    "fdr_bh": "the Benjamini–Hochberg false discovery rate",
    "fdr_by": "the Benjamini–Yekutieli false discovery rate",
    "holm": "the Holm–Bonferroni method",
    "bonferroni": "the Bonferroni correction",
}

def _groupnull_measure(params: dict[str, Any]) -> dict[str, str]:
    """What the cohort test averaged and which tail it read, from the recorded test."""
    if params.get("measure") != "isc":
        return {"measure": "coherence", "tail": "one-tailed", "two_sided": ""}
    if params.get("isc_test") == "magnitude":
        return {"measure": "the absolute inter-subject correlation, Fisher z-transformed,",
                "tail": "one-tailed", "two_sided": ""}
    return {"measure": "the inter-subject correlation, Fisher z-transformed,",
            "tail": "two-tailed",
            "two_sided": ", its p doubling the smaller of the two tails"}

def _correction_phrase(value: Any, within: str) -> str:
    # a lineage that reaches the residual but not the fit table has no setting to report
    if value is None:
        return ""
    if value == "none":
        return " P values were not corrected for multiple comparisons."
    return f" P values were corrected by {_P_CORRECTION_NAMES[value]} within {within}."

def _noise_phrase(value: Any, fitted_as_auto: "list | None" = None,
                  fitted_as_ols: "list | None" = None) -> str:
    """The noise model as prose, expanding the one spelling that names no order.

    ``auto`` is mne-nirs' own rule and reaches the sidecar unexpanded, so the sentence has
    to say what it stands for.
    """
    phrase = _model_phrase(value, fitted_as_auto)
    if fitted_as_ols:
        phrase += (f"; {len(fitted_as_ols)} channel(s) holding one value throughout, from "
                   "which no autoregressive model can be estimated, were fitted by ordinary "
                   "least squares")
    return phrase


def _model_phrase(value: Any, fitted_as_auto: "list | None") -> str:
    value = str(value or "").strip().lower()
    if value.startswith("worsley"):
        order = value[len("worsley"):]
        order = order if order.isdigit() else "four times the sampling rate in Hz"
        phrase = (f"an autoregressive noise model of order {order} whose coefficients were "
                  "estimated from the residual autocovariances after the bias correction of "
                  "Worsley et al. (2002)")
        if fitted_as_auto:
            phrase += (f" ({len(fitted_as_auto)} channel(s), whose corrected autocovariances "
                       "were not positive definite at that order, kept the uncorrected "
                       "estimate)")
        return phrase
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
        # --short-channel auto only ever chooses pca
        why = ("; used by default, since the montage has short channels"
               if params.get("short_channel_auto") else "")
        parts.append("an orthogonal basis spanning the retained short channels "
                     f"(HbO and HbR decomposed together{why})")
    aux = [str(name).removeprefix("aux_") for name in params.get("aux_regressors") or []]
    if aux:
        parts.append(f"the auxiliary signals {_series(aux)}")
    if (drift := _drift_phrase(params)) is not None:
        parts.append(drift)

    # the design matrix always holds an intercept, so there is something to say even when
    # neither flag was given, and the sentence stays true rather than naming absent columns
    return _series(parts) if parts else "only a constant term"

_CHROMA_NAME = {"hbo": "HbO", "hbr": "HbR", "hbt": "HbT"}

def _chromophores(params: dict[str, Any]) -> str:
    """The analysis's ``--chromophores`` as prose; a tree written before they were recorded is refused."""
    if params.get("chromophores") is None:
        raise ValueError("The sidecar records no chromophores; it was written by an older "
                         "nirspipe. Rerun nirspipe on this subject.")
    return _series([_CHROMA_NAME.get(c, c) for c in params["chromophores"]])

def _holds_hbt(params: dict[str, Any]) -> bool:
    """Whether a subject analysis (``chromophores``) or a dyad one (``chroma``) ran on HbT."""
    chosen = params.get("chromophores") or params.get("chroma") or ()
    return "hbt" in ([chosen] if isinstance(chosen, str) else chosen)

def _screening_scope(params: dict[str, Any]) -> str:
    """Which windows the passing share was counted over, as the run recorded it."""
    counted = params.get("screen_scope_counted")
    if counted == "task":
        return "the windows inside annotated task blocks"
    if counted is None and params.get("screen_scope") == "task":
        # asked for task blocks, and a tree this old does not say whether it found any
        return "the windows the screening counted"
    return "their windows"

def _screening_slots(params: dict[str, Any]) -> dict[str, str]:
    """The screening sentence's numbers, its counting scope and any channels marked by hand."""
    # The share falls back to the criteria table for a record that does not carry it; PSP
    # has no line to fall back to.
    from nirspipe.qc.metrics import criterion_cutoffs
    cutoffs = criterion_cutoffs()
    psp = params.get("psp_threshold")
    good_frac = params.get("min_good_frac")
    good_frac = cutoffs["good_frac"] if good_frac is None else good_frac
    low, high = params.get("cardiac_l_freq"), params.get("cardiac_h_freq")
    manual = list(dict.fromkeys(pair_of(c) for c in params.get("bad_channels") or []))
    listed = list(dict.fromkeys(pair_of(c) for c in params.get("channels_tsv_bads") or []))
    kept = list(dict.fromkeys(pair_of(c) for c in params.get("kept_channels") or []))
    return {
        "threshold": _num(params.get("sci_threshold")),
        "psp_line": (f" and its peak spectral power (PSP) in the same band at least {_num(psp)}"
                     if psp is not None else ""),
        "psp_shown": (" Peak spectral power (PSP) was computed in the same windows and "
                      "reported without a threshold." if psp is None else ""),
        # a share reads as a percentage in a Methods paragraph, and it is formatted from
        # the same value the screening used rather than written out beside it
        "min_good_frac": f"{float(good_frac) * 100:g}%",
        # the screening window, which is pinned and does not follow --window-length. The
        # record's `qc_window_s` is the QC grid and would be the wrong number to quote
        "window_s": f"{SCREEN_WINDOW_S:g}",
        "cardiac_band": (f"in the {_num(low)}–{_num(high)} Hz cardiac band"
                         f"{_band_population(params)}"
                         if low is not None and high is not None else "in the cardiac band"),
        "scope": _screening_scope(params),
        "manual": (f" {'Channels' if len(manual) > 1 else 'Channel'} {_series(manual)} "
                   f"{'were' if len(manual) > 1 else 'was'} also marked as bad by hand."
                   if manual else ""),
        "kept": (f" {'Channels' if len(kept) > 1 else 'Channel'} {_series(kept)} "
                 f"{'were' if len(kept) > 1 else 'was'} kept after visual review although the "
                 f"screening rejected {'them' if len(kept) > 1 else 'it'}."
                 if kept else ""),
        "channels_tsv": (f" {'Channels' if len(listed) > 1 else 'Channel'} {_series(listed)} "
                         f"{'were' if len(listed) > 1 else 'was'} also marked as bad, as the "
                         f"recording's BIDS channels.tsv lists {'them' if len(listed) > 1 else 'it'}."
                         if listed else ""),
        # GVTD's spans come after the screening and never reach its count
        "spans": (" Windows touching a BAD_ span were not counted."
                  if any(span.get("source") != "--gvtd-censor"
                         for span in params.get("excluded_spans") or []) else ""),
    }

def _band_population(params: dict[str, Any]) -> str:
    """Which edges of the cardiac band ``--population`` set, as a clause, or nothing."""
    filled = set(params.get("from_population") or ())
    edges = [edge for edge, key in (("lower", "cardiac_l_freq"), ("upper", "cardiac_h_freq"))
             if key in filled]
    if not edges:
        return ""
    whose = f"the package's value for {age_span(params['population'])}"
    return f", {whose}" if len(edges) == 2 else f", its {edges[0]} edge {whose}"

def _dpf_phrase(dpf: Any, age: Any = None, population: "str | None" = None) -> str:
    """'a differential pathlength factor (DPF) of 6', or one per wavelength when they differ."""
    if age is not None:
        values = [f"{float(d):.2f}" for d in dpf]
        whose = (f"at {_num(age)} years, the package's age for {age_span(population)},"
                 if population else f"from the participant's age of {_num(age)} years")
        return (f"differential pathlength factors (DPF) of {_series(values)}, in ascending "
                f"order of wavelength, {whose} by the general equation of Scholkmann and "
                f"Wolf (2013)")
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

def _monte_carlo(params: dict[str, Any]) -> str:
    """pycwt's red-noise level, as --wtc-significance drew it, or nothing."""
    if not params.get("wtc_mc_count"):
        return ""
    seed = params.get("wtc_seed")
    return (" A per-frequency significance level was estimated by Monte Carlo simulation: "
            f"coherence was computed for {int(params['wtc_mc_count'])} pairs of first-order "
            "autoregressive (red-noise) surrogates matching each channel's lag-1 "
            f"autocorrelation{f' (random seed {seed})' if seed is not None else ''}, and its "
            "95th percentile taken at each frequency.")

def _device_phrase(params: dict[str, Any]) -> str:
    """The card the transforms ran on, or nothing for the CPU."""
    if params.get("wtc_device") != "cuda":
        return ""
    return (" The wavelet transforms were computed in double precision on a graphics card "
            f"({params.get('wtc_device_name')}; CuPy {params.get('wtc_cupy_version')}, "
            f"CUDA {params.get('wtc_cuda_version')}).")

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
                   f"({_num(whiten_s)} s)"
                   + (", its coefficients fitted on that member's samples outside BAD_ spans"
                      if _has_member_spans(params) else "") + "."
                   if whiten_s else ""),
        "conditions": _condition_route(params),
        "significance": _monte_carlo(params),
        "window": (f" Only {_num(window[0])}–{_num(window[1])} s of the aligned recordings "
                   "were analysed." if window else ""),
        "bads": (" A channel rejected in any of a member's runs was left out of all of them."
                 if params.get("bads_scope") == "subject" else ""),
        "excluded": (" Time inside a BAD_ span of either member was left out of every average, "
                     "with no guard band around it." if _has_member_spans(params) else ""),
        "device": _device_phrase(params),
    }

def _phasenull_slots(params: dict[str, Any]) -> dict[str, str]:
    """How many surrogates, over which pairs, and the seed that makes them repeatable."""
    seed = params.get("seed")
    return {
        "pairs": "every channel pair" if params.get("cross") else "homologous channel pairs",
        "scramble": ("one random phase shift per frequency shared by all of that member's "
                     "channels, so they kept their correlations with each other"
                     if params.get("phase_null_shared") else "drawn independently per channel"),
        "n_iter": str(int(params.get("n_iter") or 0)),
        "seed": f" (random seed {seed})" if seed is not None else "",
    }

def _pairnull_slots(params: dict[str, Any]) -> dict[str, str]:
    """Who stood in, how many, and over what, as the re-paired null's sidecars record it."""
    n, pool_size = int(params.get("n_iter") or 0), params.get("pair_candidates")
    any_pool = params.get("pair_pool") == "any"
    # a group's draw ids join its stand-ins with "+"; a dyad's is its one stand-in
    group = any("+" in str(p) for p in params.get("pair_partners") or [])
    if group:
        pool = ("a member of one other group" if any_pool
                else "the member in the same position of one other group")
        held = ("each pairing kept its earlier-listed member, and in each draw every "
                f"later-listed member was replaced by {pool} recorded on the same task, the "
                "same stand-in in every pairing that member closes")
    else:
        pool = ("each member of every other group" if any_pool
                else "the member in the same position of each other group")
        held = ("the first-listed member was kept and their partner replaced, one at a "
                f"time, by {pool} recorded on the same task")
    unit = "draws" if group else "stand-ins"
    return {
        "held": held,
        "unit": unit,
        # the correlation's tables carry its own settings; without them only coherence ran
        "measures": ("coherence and the inter-subject correlation were"
                     if "isc_whiten_s" in params else "coherence was"),
        "pairs": "every channel pair" if params.get("cross") else "homologous channel pairs",
        "count": (f"{n} of the {pool_size} eligible {unit} were used."
                  if pool_size is not None and n < int(pool_size)
                  else f"All {n} eligible {unit} were used."),
        "pad": _num(round(float(params.get("pair_cond_pad_s") or 0.0), 1)),
        # written only when a region map was given
        "regions": (" A stand-in's region value was compared with the real dyad's over the "
                    "channel pairs both kept, and that difference subtracted from the real "
                    "region value." if "roi_min_channels" in params else ""),
        "device": _device_phrase(params),
    }

def _has_member_spans(params: dict[str, Any]) -> bool:
    """Whether a dyad table's members carried any BAD_ span."""
    return any((params.get("excluded_spans_s") or {}).values())


def _censoring_phrase(params: dict[str, Any]) -> str:
    """What a regression left out of its fit, or nothing for a run without BAD_ spans."""
    frac = params.get("censored_frac")
    if not frac:
        return ""
    dropped = params.get("dropped_conditions") or []
    skipped = params.get("skipped_contrasts") or {}
    # ordinary least squares estimates no noise model, so it has no lags to keep in kept time
    noise = ("" if params.get("noise_model") == "ols" else
             " the noise model was estimated only from stretches whose lags all lay in kept "
             "time,")
    return (f" Frames inside BAD_ spans ({100 * float(frac):.0f}% of the recording) were left "
            f"out of the fit: the design was built on the full time axis,{noise} and the "
            "residual kept its full length."
            + (f" {'Conditions' if len(dropped) > 1 else 'Condition'} {_series(dropped)} "
               f"{'were' if len(dropped) > 1 else 'was'} dropped from the model, having no "
               "signal in the kept frames." if dropped else "")
            + "".join(f" Contrast {name}, which weights {_series(lost)}, was therefore not "
                      "computed." for name, lost in skipped.items()))


def _min_time_phrase(params: dict[str, Any], what: str) -> str:
    min_time = params.get("min_time")
    return (f" {what} computed only when at least {_num(min_time)} s lay outside BAD_ spans."
            if min_time else "")


# how each fill method is described; the Lomb-Scargle numbers are the package's constants
def _fill_phrase(method: Any) -> str:
    from nirspipe.pipeline.censor_fill import LOMB_FMAX_HZ, LOMB_OVERSAMPLE
    return {
        "linear": "by linear interpolation between the kept samples either side",
        "spline": "with one cubic spline through every kept sample",
        "lomb": (f"from a Lomb-Scargle fit to the kept samples (sinusoids up to "
                 f"{LOMB_FMAX_HZ:g} Hz at {LOMB_OVERSAMPLE}x oversampling)"),
    }.get(str(method), f"({method})")


def _isc_slots(params: dict[str, Any]) -> dict[str, str]:
    """Its chromophores, then the optional steps in the order they ran: whitening, band, lag."""
    # the matrices record no chroma; the pair table beside them does, and the two merge
    chroma = _series([_CHROMA_NAME.get(c, c) for c in params.get("chroma") or ["hbo", "hbr"]])
    before = []
    if params.get("isc_whiten_s"):
        before.append("prewhitened with an autoregressive model of order "
                      f"{params.get('isc_whiten_order')} ({_num(params['isc_whiten_s'])} s), "
                      "fitted on the whole aligned recording"
                      + (" with its coefficients taken from that member's samples outside "
                         "BAD_ spans" if _has_member_spans(params) else ""))
    low, high = params.get("isc_band_hz") or (None, None)
    if low is not None and high is not None:
        before.append(f"band-pass filtered to {_num(low)}–{_num(high)} Hz")
    elif low is not None:
        before.append(f"high-pass filtered at {_num(low)} Hz")
    elif high is not None:
        before.append(f"low-pass filtered at {_num(high)} Hz")
    options = (f" Before correlating, each signal was {' and '.join(before)}."
               if before else "")
    if _has_member_spans(params):
        options += (" Samples inside a BAD_ span of either member were left out of every "
                    "correlation.")
    if params.get("isc_max_lag_s"):
        options += (" The correlation was taken at the lag of largest magnitude within "
                    f"±{_num(params['isc_max_lag_s'])} s, keeping its sign.")
    if params.get("isc_phase_null_iter"):
        scramble = ("all shifted by one shared set of random phases in each repetition"
                    if params.get("phase_null_shared") else
                    "each phase-randomised independently")
        options += (f" Its null repeated the correlation {int(params['isc_phase_null_iter'])} "
                    f"times against the second member's channels, {scramble}, and was "
                    "summarised by the mean and the 95th percentile of the absolute "
                    "correlation.")
    return {"chroma": chroma, "options": options}

def template_slots(key: str, params: dict[str, Any]) -> dict[str, str]:
    """Fill a section's {slots} from a sidecar's parameters.

    Note the naming: a sidecar's ``high_pass`` is the *lower* edge of the band, which the
    filter templates call ``l_freq``.
    """
    if key == "sci_marking":
        return _screening_slots(params)
    if key in ("optode_landmarks", "optode_frame"):
        space = str(params.get("space") or "fsaverage")
        return {"space": space,
                # MNE ships no template between its oldest infant one and fsaverage
                "template_note": (" MNE-Python provides no template for 2 to 18 years, so the "
                                  "children's optodes were placed on fsaverage, an adult head."
                                  if params.get("population") == "child"
                                  and space == "fsaverage" else ""),
                "landmarks": str(params.get("landmarks_used") or 0),
                "fit": f"{float(params.get('scalp_fit_mm') or 0):.1f}",
                "frame": _OPTODE_FRAMES.get(params.get("method"), "")}
    # motion: constants of the package, read here, bar the two settings a run may change,
    # which its sidecar records
    if key in ("motion_wavelet", "motion_wavelet_tddr"):
        return {"wavelet": motion.WAVELET, "iqr_factor": _recorded(params, "wavelet_iqr"),
                "lowest_scale": f"{motion.WAVELET_LOWEST_SCALE}"}
    if key == "motion_tpca":
        return {"t_motion": f"{motion.TPCA_T_MOTION:g}", "std_thresh": f"{motion.TPCA_STD_THRESH:g}",
                "amp_thresh": f"{motion.TPCA_AMP_THRESH:g}", "t_mask": f"{motion.TPCA_T_MASK:g}",
                "n_sv": f"{motion.TPCA_N_SV:g}", "max_iter": f"{motion.TPCA_MAX_ITER}"}
    if key == "motion_spline":
        return {"t_motion": f"{motion.SPLINE_T_MOTION:g}",
                "std_thresh": _recorded(params, "motion_std_thresh"),
                "amp_thresh": f"{motion.SPLINE_AMP_THRESH:g}", "t_mask": f"{motion.SPLINE_T_MASK:g}",
                "p": f"{motion.SPLINE_P:g}", "dt_short": f"{motion.SPLINE_DT_SHORT:g}",
                "dt_long": f"{motion.SPLINE_DT_LONG:g}"}
    if key == "motion_spline_wavelet":
        return {"t_motion": f"{motion.SPLINE_WAVELET_T_MOTION:g}",
                "std_thresh": _recorded(params, "motion_std_thresh"),
                "amp_thresh": f"{motion.SPLINE_WAVELET_AMP_THRESH:g}",
                "t_mask": f"{motion.SPLINE_WAVELET_T_MASK:g}", "p": f"{motion.SPLINE_P:g}",
                "dt_short": f"{motion.SPLINE_DT_SHORT:g}", "dt_long": f"{motion.SPLINE_DT_LONG:g}",
                "wavelet": motion.WAVELET, "iqr_factor": _recorded(params, "wavelet_iqr"),
                "lowest_scale": f"{motion.WAVELET_LOWEST_SCALE}"}
    if key == "motion_spline_sg":
        return {"lowpass": f"{motion.SPLINE_SG_LOWPASS:g}",
                "detect_lowpass": f"{motion.SPLINE_SG_DETECT_LOWPASS:g}",
                "grad_iqr": f"{motion.SPLINE_SG_GRAD_IQR:g}",
                "std_window": f"{motion.SPLINE_SG_STD_WINDOW:g}",
                "std_iqr": f"{motion.SPLINE_SG_STD_IQR:g}",
                "shift_lag": f"{motion.SPLINE_SG_SHIFT_LAG:g}",
                "snr_min": f"{motion.SPLINE_SG_SNR_MIN:g}", "pad": f"{motion.SPLINE_SG_PAD:g}",
                "p": f"{motion.SPLINE_SG_P:g}", "sg_order": f"{motion.SPLINE_SG_ORDER}",
                "frame": f"{motion.SPLINE_SG_FRAME:g}",
                "sg_cutoff": f"{motion.spline_sg_cutoff():.2f}"}
    if key == "beer_lambert":
        return {"dpf": _dpf_phrase(params.get("dpf"), params.get("dpf_age"),
                                   params.get("dpf_population"))}
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
                "noise_model": _noise_phrase(params.get("noise_model"),
                                             params.get("worsley_fitted_as_auto"),
                                             params.get("constant_fitted_as_ols")),
                "censoring": _censoring_phrase(params)}
    if key == "glm":
        return {"conditions": _conditions_phrase(params),
                "regressors": _regressor_phrase(params),
                "noise_model": _noise_phrase(params.get("noise_model"),
                                             params.get("worsley_fitted_as_auto"),
                                             params.get("constant_fitted_as_ols")),
                "censoring": _censoring_phrase(params),
                "hbt": (" HbT was fitted as a series of its own, with the same design and noise "
                        "model." if "hbt" in (params.get("chromophores") or ()) else ""),
                "correction": _correction_phrase(
                    params.get("p_correction"),
                    "each regressor or contrast and each chromophore, over the long channels "
                    "not marked as bad")}
    if key == "gvtd_censoring":
        return {"channel_set": str(params.get("gvtd_censor_channel_set")
                                   or params.get("gvtd_censor") or "long"),
                "band": f"{GVTD_MOTION_BAND[0]:g}–{GVTD_MOTION_BAND[1]:g} Hz",
                "n_std": _num(params.get("gvtd_censor_n_std")),
                "min_epoch_s": _num(params.get("gvtd_censor_min_epoch_s"))}
    if key == "censor_fill":
        return {"method": _fill_phrase(params.get("censor_fill"))}
    if key == "fc":
        return {"kept": " over the frames outside BAD_ spans"
                if params.get("excluded_spans") else "",
                "chroma": _chromophores(params),
                "min_time": _min_time_phrase(params, "Connectivity was")}
    if key in ("hyper_wtc", "hyper_wtc_crossed"):
        return _wtc_slots(params)
    if key in ("hyper_isc", "hyper_isc_crossed"):
        return _isc_slots(params)
    if key == "hyper_phasenull":
        return _phasenull_slots(params)
    if key == "hyper_phasescale":
        return {"coi": " outside the cone of influence" if params.get("mask_coi") else ""}
    if key == "hyper_groupnull":
        return {"null": "re-paired" if params.get("null_kind") == "repaired"
                else "phase-randomised",
                **_groupnull_measure(params),
                "n_resample": str(int(params.get("n_resample") or 0)),
                "correction": _correction_phrase(params.get("p_correction"),
                                                 "each condition, level and test")}
    if key == "hyper_pairnull":
        return _pairnull_slots(params)
    if key == "alff":
        return {"chroma": _chromophores(params),
                "l_freq": _num(params.get("high_pass")), "h_freq": _num(params.get("low_pass")),
                "kept": (", from a Lomb-Scargle periodogram of the samples outside BAD_ spans, "
                         "whose fALFF denominator ran up to the Nyquist frequency on the same "
                         "samples" if params.get("excluded_spans") else ""),
                "min_time": _min_time_phrase(params, "ALFF and fALFF were")}
    if key == "hyper_roi":
        least = int(params.get("roi_min_channels") or 1)
        return {"min_channels": (" A region pair was left empty where either member "
                                 f"contributed fewer than {least} channels."
                                 if least > 1 else "")}
    if key == "hyper_input":
        desc = params.get("desc")
        return {"stage": f"desc-{desc}" if desc else "input"}
    return {}

def _recorded(params: dict[str, Any], key: str) -> str:
    """A motion setting from the motion-corrected sidecar; a tree written before it was is refused."""
    if params.get(key) is None:
        raise ValueError(f"The motion-corrected sidecar records no {key}; it was written by an "
                         f"older nirspipe. Rerun nirspipe on this subject.")
    return _num(params[key])

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
    "optode_placement": "Where the optodes sit on the head template; left unplaced when no reading of the file put them on its scalp.",
    "motion_correction": "Motion correction, using the method named in the settings.",
    "sqm_raw": "Quality metrics measured on the original intensity recording.",
    "sqm": "Quality metrics for this run, grouped by the stage each was measured on.",
    "design_matrix": "Regressors assembled for the fit: the conditions on a task run, plus drift and confounds.",
    "glm_fit": "Per-channel model fit; rejected channels are flagged, not dropped.",
    "contrasts": "Contrast estimates derived from the fitted model.",
    "glm_residuals": "What the model left behind, once the fitted signal was removed.",
    "glm_whitening": "How white the fit's residual is after its noise model: spectrum and lag-1 autocorrelation.",
    "block_average": "Each condition's trials averaged per channel over the trial window, on the series the model was fitted to.",
    "export_mean_filled": "The final residual with every BAD_ frame set to its channel's kept mean, for tools that ignore annotations; not for ALFF or a GLM.",
    "glm_residuals_broadband": "The same confound regression on data that skipped the bandpass, so fALFF keeps a full spectrum.",
    "alff": "Amplitude of low-frequency fluctuation, per channel.",
    "alff_roi": "Amplitude of low-frequency fluctuation averaged over each ROI's channels.",
    "fc": "Channel-by-channel correlation within one chromophore.",
    "fisher_z": "Fisher r-to-z of a correlation matrix, for group-level statistics.",
    "fc_roi": "Connectivity between ROI-averaged signals.",
    "fc_seed": "Correlation of one ROI's mean signal with every channel.",
    "hyper_bads": "The channels excluded for this dyad, over the scope named in the settings.",
    "hyper_sqm": "Quality metrics for the dyad: alignment, coupling and the members' own.",
    "hyper_usable": "How much of each channel pair both members could use at once, per condition.",
    "group_sqm_raw": "Each member's quality metrics, one row per member.",
    "group_sqm_raw_channels": "Per member and channel: the windowed and the whole-run SCI, and whether the channel was rejected.",
    "hyper_wtc": "Wavelet coherence between a pair, averaged over a band and one value per channel.",
    "hyper_wtc_phasenull": "The same average against a phase-scrambled partner: the null.",
    "hyper_wtc_roichan": "Channel coherences averaged within each ROI, or per ROI pair on a crossed run: the ROI number to report.",
    "hyper_wtc_roihom": "Channel-level coherences averaged over each ROI's homologous channel pairs alone, the subset a homologous-only design reports.",
    "hyper_wtc_roihom_phasenull": "The null for that ROI mean, its iterations grouped into regions before they were summarised.",
    "hyper_wtc_roichan_phasenull": "The phase-scrambled null for the ROI x ROI coherence matrix, grouped into region pairs per iteration.",
    "hyper_wtc_bycondition_roichan_pairnull": "The re-paired null for each condition's ROI x ROI coherence matrix, grouped into region pairs per stand-in.",
    "hyper_isc": "Correlation of channels across the two brains, paired as the coherence: every pairing, or homologous ones.",
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
    "sci_mean": "Scalp coupling over the whole run: how well the two wavelengths share a heartbeat. Near 1 is good; low means poor optode contact.",
    "sci_win_mean": f"Scalp coupling in {SCI_WINDOW_S:g} s windows, averaged, so brief movement sways it less than the whole-run SCI. Near 1 is good.",
    "channel_retention_rate": "Share of channels not marked bad. Higher is better.",
    "psp_mean": f"Strength of the heartbeat peak the two wavelengths share, in {PSP_WINDOW_S:g} s windows. Higher is better.",
    "good_frac_mean": f"Share of {SCREEN_WINDOW_S:g} s windows passing on SCI, and on PSP when it has a line; channels are rejected on this. Higher is better.",
    "cp_mean": "How peaked the spectrum is inside the cardiac band, 0 to 1. Experimental, with no better end; read SCI and PSP for coupling.",
    "heart_rate": "Heart rate in Hz, the median of the channels' spectral peaks above the respiration band. Descriptive; it should lie inside the cardiac band.",

    # raw intensity
    "cv_mean": f"Noise relative to brightness (SD / mean) of the raw intensity, in {CV_WINDOW_S:g} s windows. Lower is better.",
    "snr_mean": f"Mean over SD (μ/σ) of the raw intensity in {CV_WINDOW_S:g} s windows, a linear ratio, not dB. Higher is better.",
    "snr_pass_rate": "Share of channels whose SNR clears the per-channel line. Higher is better.",
    "n_flat_channels": "Channels with no variation at all. Zero is best.",
    "mean_amp_mean": "Average light level reaching the detectors. Descriptive: use it to spot unusually dim channels.",
    "n_saturated_channels": "Channels held at the detector's ceiling for two samples or more. Zero is best.",
    "saturated_pct": "Share of samples held at the detector's ceiling. Zero is best.",
    # geometry
    "ch_dist_mean": "Average source-detector separation, in mm. Descriptive.",
    "ch_dist_min": "Shortest source-detector separation, in mm. Descriptive.",
    "ch_dist_max": "Longest source-detector separation, in mm. Descriptive.",

    # haemoglobin
    "hbo_hbr_corr_mean": "Correlation of HbO and HbR: cortical signal pushes it negative, systemic or motion signal positive. Lower is better, with no fixed cutoff.",
    "cnr_hbo_mean": "How far the evoked HbO response clears its noise, averaged over channels. Higher is better; needs stimulus annotations.",
    "cnr_hbr_mean": "The same for HbR, which falls with a response, so more negative is better.",
    "cnr_n_epochs": "Stimulus epochs the CNR averaged over; with only a few, the CNR is noisy. Descriptive.",
    "gcor_hbo": "How much all HbO channels move together; high means a strong global or systemic component. Descriptive.",
    "gcor_hbr": "The same for HbR.",
    "lowfreq_drift_amplitude_hbo": "How far the HbO baseline travelled, peak to peak of a fitted trend. Lower is better; not comparable across recording lengths.",
    "lowfreq_drift_amplitude_hbr": "The same for HbR.",

    # spectral. Power is a mean PSD level inside the band, fraction is a sum over the band
    # against the sum over the whole spectrum; they are not the same quantity rescaled.
    "cardiac_band_power_hbo": "Average HbO spectral density in the cardiac band. Descriptive; it scales with signal amplitude, so it does not compare across subjects.",
    "cardiac_band_power_hbr": "The same for HbR.",
    "cardiac_band_frac_hbo": "Share of HbO power in the cardiac band, 0 to 1. Descriptive; read SCI and PSP for pulse quality.",
    "cardiac_band_frac_hbr": "The same for HbR.",
    "resp_band_power_hbo": "Average HbO spectral density in the respiration band. Descriptive.",
    "resp_band_power_hbr": "The same for HbR.",
    "resp_band_frac_hbo": "Share of HbO power in the respiration band, 0 to 1. Descriptive.",
    "resp_band_frac_hbr": "The same for HbR.",
    # motion and spikes, all measured on optical density.
    # Note which trace gvtd_thresh belongs to: it is computed from the band-passed trace
    # and compared against it, so it is not a cutoff for the unfiltered gvtd_mean/p95.
    "gvtd_mean": "Unfiltered GVTD averaged over the run, recorded for comparison with other tools; lower is better. The GVTD threshold does not apply; read the band-passed GVTD for movement.",
    "gvtd_p95": "The same at the worst moments, the 95th percentile.",
    "gvtd_filt_mean": f"Head movement: GVTD band-passed to {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz, averaged over the run. Lower is better.",
    "gvtd_filt_p95": "The same at the worst moments, the 95th percentile. Lower is better.",
    "gvtd_vstd_mean": "Unfiltered GVTD with every channel scaled to unit variance, so loud channels cannot dominate. Recorded only, with no better end.",
    "gvtd_vstd_p95": "The same at the 95th percentile.",
    "gvtd_thresh": f"Motion cutoff from this recording's own band-passed GVTD: its mode plus {GVTD_N_STD:g} times the spread below it. Descriptive.",
    "gvtd_thresh_applied": "The cutoff the frame counts are taken against: the uncorrected recording's, on both sides of a before and after pair. Descriptive.",
    "gvtd_num_above_thresh": "Frames whose band-passed GVTD exceeds the applied threshold. Lower is better.",
    "gvtd_pct_above_thresh": "Share of the recording with clear movement, above the applied GVTD threshold. Lower is better.",
    "gvtd_censor_pct": "Share of the recording marked BAD_gvtd by censoring; nothing was deleted. Lower is better.",
    "gvtd_censor_retained_s": "Seconds left after censoring, the time an analysis can actually use. Higher is better.",
    "whitening_lowband_hbo": "Whitened GLM residual power at 0.01-0.1 Hz over white noise: above 1 inflates t, below 1 deflates it; aim for 1, so no better end.",
    "whitening_lowband_hbr": "The same for HbR.",
    "spike_count": "Sudden jumps across all channels, on the motion-band derivative. Lower is better.",
    "spike_pct": "Spike samples as a share of all channel-samples. Experimental; lower is better.",
    "spike_num_frames": f"Frames where at least {100 * SPIKE_CH_FRAC:g}% of channels jumped together. Experimental; lower is better.",
    "spike_pct_frames": "Spike frames as a share of the recording. Experimental; lower is better.",
    "motion_corrected_frac_mean": "Average share of each channel's samples the correction changed abruptly. Experimental and descriptive.",
    "motion_corrected_num": f"Frames where at least {100 * SPIKE_CH_FRAC:g}% of channels were corrected abruptly at once. Experimental and descriptive.",
    "motion_corrected_pct": "Corrected frames as a share of the recording. Experimental and descriptive.",
    "motion_corrected_n_segments": "Separate stretches the corrected frames form. Experimental and descriptive.",
    # motion, measured by the sensor rather than inferred from the optical data
    "gyro_speed_mean": "Average head turning speed from the gyroscope, in the recording's unit. Descriptive.",
    "gyro_speed_p95": "The same at the worst moments, the 95th percentile. Descriptive.",
    "accel_jerk_mean": "Average jerk (change in acceleration) from the accelerometer, in the recording's unit. Descriptive.",
    "accel_jerk_p95": "The same at the worst moments, the 95th percentile. Descriptive.",
    "gyro_speed_gvtd_rho": "How closely the band-passed GVTD follows the gyroscope speed, frame by frame (Spearman). Recorded only, with no better end.",
    "accel_jerk_gvtd_rho": "The same against the accelerometer jerk.",
    # time
    "pct_data_retained": "Share of the recording outside BAD_ spans. Higher is better.",
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
    "good_frac_mean",           # ... and kept passing its windows, which is what rejects a channel
    "hbo_hbr_corr_mean",        # what came out looks like haemodynamics
})

# ---- which stage each number was measured on ----
#
# Kept apart from METRIC_SUMMARY because this is a property of *where* the metric is
# computed, not of what it means. Four of these keys are measured at two stages on data
# separated by a bandpass and a resample; without this line a tooltip cannot say which of
# the two the reader is looking at, and the section name is not visible from inside a
# panel. The section-to-stage mapping this follows lives in the SQM record.
_STAGE_RAW = "Measured on the raw recording."
_STAGE_MOTION = "Measured across motion correction."
_STAGE_PREPROC = "Measured after Beer-Lambert, before filtering."
_STAGE_BOTH = "Measured at each haemoglobin stage; the section says which."
_STAGE_WHITENING = "Median over good long channels of the task GLM."
# the one haemoglobin metric the raw viewer also measures, on its own Beer-Lambert
_STAGE_BOTH_AND_RAW = (
    "Measured at each haemoglobin stage, unfiltered in the raw viewer; the section says which."
)
_STAGE_RAW_AND_CORRECTED = "Measured on the raw recording and again after motion correction."

# A condition page reads GVTD off the corrected file's windows alone, so the whole-run line
# above would name a second file and a percentile the page does not hold.
_STAGE_CONDITION_MOTION = "Here: the motion-corrected file's windows inside this condition."
_RAW_METRICS = (
    # good_frac_mean sits here and not with sci/psp, which it is built from: those two are
    # measured again on the corrected file, and the passing-window count is taken once, at
    # screening, on the optical density as it arrived
    "good_frac_mean",
    "channel_retention_rate", "cp_mean", "heart_rate", "n_flat_channels",
    "cv_mean", "snr_mean", "snr_pass_rate", "mean_amp_mean",
    "n_saturated_channels", "saturated_pct",
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
    "hbo_hbr_corr_mean": _STAGE_BOTH_AND_RAW,
    "whitening_lowband_hbo": _STAGE_WHITENING,
    "whitening_lowband_hbr": _STAGE_WHITENING,
}

def metric_summary(metric: str, condition: bool = False) -> str:
    """What a metric is, which way is good, and what stage it was measured on.
    Returns '' for an undescribed metric, so the report renders a bare number rather than
    an empty tooltip. ``condition`` is a condition page, whose GVTD numbers come from one file.
    """
    text = (condition and _CONDITION_SUMMARIES.get(metric)) or METRIC_SUMMARY.get(metric, "")
    if not text:
        return ""
    stage = (_STAGE_CONDITION_MOTION if condition and metric.startswith("gvtd_")
             and metric in _RAW_AND_CORRECTED_METRICS else METRIC_STAGE.get(metric, ""))
    return f"{text} {stage}".rstrip()

def is_key_metric(metric: str) -> bool:
    return metric in KEY_METRICS

# ---- the tables a quality record keeps beside its JSON ----
# Column descriptions for their sidecars, one whole sentence each.
_RECORD_STAGES = {
    "raw": "The recording as it arrived, intensity and optical density, bad channels included.",
    "motion": "The motion-correction step, measured on the optical density either side of it, bad channels included.",
    "motion_post": "The motion-corrected optical density, bad channels included.",
    "rawhaemo": "Haemoglobin converted from the recording as it arrived, before motion correction and filtering, bad channels included.",
    "rawhaemo_post": "Haemoglobin converted from the motion-corrected optical density, before filtering, bad channels included.",
    "preproc": "Haemoglobin after Beer-Lambert and before filtering, bad channels excluded.",
    "filtered": "Haemoglobin after the bandpass, bad channels excluded.",
    "resampled": "Haemoglobin after the resample, bad channels excluded.",
    "errts": "The confound-regression residual, bad channels excluded.",
}
_RECORD_SETS = {"": "", "_long": " Long channels only.", "_short": " Short channels only."}
RECORD_SECTIONS = {f"{stage}{suffix}": text + phrase
                   for stage, text in _RECORD_STAGES.items()
                   for suffix, phrase in _RECORD_SETS.items()}
RECORD_CHANNEL_METRICS = {
    "sci_per_channel": "Scalp coupling index over the whole run: the correlation of the two wavelengths in the cardiac band.",
    "sci_win_per_channel": f"The same correlation inside {SCI_WINDOW_S:g} s windows, averaged over the windows.",
    "good_frac_per_channel": f"Share of {SCREEN_WINDOW_S:g} s windows passing on SCI, and on PSP when it has a line; a channel is rejected on this.",
    "ch_dist_per_channel": "Source-detector separation in mm.",
    "psp_per_channel": f"Strength of the cardiac peak the two wavelengths share, averaged over {PSP_WINDOW_S:g} s windows.",
    "cp_per_channel": "How peaked the spectrum is inside the cardiac band, 0 to 1. Experimental.",
    "heart_rate_per_channel": "Frequency in Hz where the channel's spectrum peaks above the respiration band; their median over the set is the heart rate.",
    "cv_per_channel": f"Standard deviation over mean of the raw intensity inside {CV_WINDOW_S:g} s windows, averaged over the windows.",
    "snr_per_channel": "Mean over standard deviation, the reciprocal of the channel's CV.",
    "mean_amp_per_channel": "Mean raw intensity reaching the detector.",
    "saturated_per_channel": "Share of samples held at the recording's highest intensity for two samples or more, the detector's ceiling.",
    "spike_pct_per_channel": "Fraction of samples flagged as sudden jumps on the motion-band-filtered derivative. Experimental.",
    "temporal_derivative_variance": "Variance of the unfiltered sample-to-sample change in optical density.",
    "motion_corrected_frac_per_channel": "Fraction of samples where the correction changed the signal abruptly, its frame-to-frame change exceeding the channel's own noise. Experimental.",
    "hbo_hbr_corr_per_channel": "Correlation between HbO and HbR, one value per source-detector pair.",
    "cnr_per_channel": "How far the evoked response clears its own noise, averaged over the stimulus epochs.",
}

RECORD_CHANNEL_COLUMNS = {
    "name": "Channel name as in the recording; a value taken over both chromophores names the source-detector pair alone.",
    "section": "The file the value was measured on, and over which channel set.",
    "metric": "What was measured.",
    "value": "The measured value.",
}
_GVTD_FAMILIES = {
    "gvtd_per_window": "Unfiltered GVTD averaged within each window",
    "gvtd_p95_per_window": "The 95th percentile of unfiltered GVTD within each window",
    "gvtd_filt_per_window": f"GVTD band-passed to {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz, averaged within each window",
    "gvtd_filt_p95_per_window": "The 95th percentile of the band-passed GVTD within each window",
}
_GVTD_SETS = {"": ", over the long channels.", "_short": ", over the short channels.",
              "_all": ", over every channel."}
_GVTD_FILE = " Measured on the motion-corrected optical density when the run corrected motion."
RECORD_WINDOW_COLUMNS = {
    "sci_per_window": "Scalp coupling index in each window, averaged over every channel.",
    "psp_per_window": "Strength of the shared cardiac peak in each window, averaged over every channel.",
    "cv_per_window": "Standard deviation over mean of the raw intensity in each window, averaged over every channel.",
    "snr_per_window": "The reciprocal of cv_per_window.",
    **{f"{family}{suffix}": text + phrase + _GVTD_FILE
       for family, text in _GVTD_FAMILIES.items() for suffix, phrase in _GVTD_SETS.items()},
    "gyro_speed_per_window": "Gyroscope speed averaged within each window.",
    "gyro_speed_p95_per_window": "The 95th percentile of the gyroscope speed within each window.",
    "accel_jerk_per_window": "Accelerometer jerk averaged within each window.",
    "accel_jerk_p95_per_window": "The 95th percentile of the accelerometer jerk within each window.",
}
# ---- the cohort table's sidecar ----
# Record keys the cohort table carries that are counts or settings, not report metrics.
RECORD_EXTRA_KEYS = {
    "n_channels": "How many channels this channel set holds.",
    "n_bad": "How many channels of this set were marked bad.",
    "n_long_channels": "How many channels fall in the long separation band.",
    "n_short_channels": "How many channels fall in the short separation band.",
    "sep_short_max_mm": "The longest source-detector separation, in mm, counted as a short channel on this run.",
    "sep_long_min_mm": "The shortest source-detector separation, in mm, counted as a long channel on this run.",
    "sep_long_max_mm": "The longest source-detector separation, in mm, counted as a long channel on this run; n/a when no upper bound was set.",
    "qc_window_s": "The window length the per-window series asked for, in seconds; each window used is a whole number of samples long.",
    "gvtd_censor_n_std": "The multiplier the censoring cutoff was set with (--gvtd-censor-n-std).",
    "gvtd_censor_min_epoch_s": "Stretches left between censored spans that are shorter than this, in seconds, are censored too (--gvtd-min-epoch-s).",
    "gvtd_censor_thresh": "The cutoff on the band-passed GVTD that censoring used.",
    "gvtd_censor_n_spans": "How many separate spans were censored.",
    "gvtd_censor_n_epochs": "How many continuous stretches survived censoring.",
    "total_s": "The recording's length in seconds.",
    "kept_s": "Seconds of the recording outside every BAD_ span.",
    "unselected_frac": "Share of the recording marked BAD_unselected, outside the stretches --keep-spans named; the data there is real and was not altered.",
    "corrupted_frac": "Share of the recording under any other BAD_ span, BAD_gvtd, BAD_manual from --bad-spans, BAD_edge where no light was measured or one the input carried, counted once where spans overlap.",
    "n_spans": "How many BAD_ spans the run carries, of both kinds.",
}
# per-wavelength CV keys carry the device's own wavelengths, so they take a template
RECORD_CV_BY_WAVELENGTH = "cv_mean over the {nm} nm channels only."
COHORT_SECTIONS = {
    **RECORD_SECTIONS,
    "censor": "From the GVTD censoring step.",
    "imu": "From the head-motion sensor, not the optical data.",
    "whitening": "From the task GLM's residual after its noise model.",
    "excluded": "From the run's BAD_ spans: --keep-spans, --bad-spans, --gvtd-censor and any the input carried.",
}
COHORT_ID_COLUMN = "The run this row describes, by its BIDS entities."

def record_key_summary(key: str) -> str:
    """What one record key holds, or '' when nothing describes it."""
    if key in METRIC_SUMMARY:
        return METRIC_SUMMARY[key]
    if key in RECORD_EXTRA_KEYS:
        return RECORD_EXTRA_KEYS[key]
    wavelength = key.removeprefix("cv_mean_")
    return RECORD_CV_BY_WAVELENGTH.format(nm=wavelength) if wavelength.isdigit() else ""

# ---- how a metric is printed ----
#
# One row per metric: the label a panel prints, the number format, and where the colouring
# changes. It sits beside METRIC_SUMMARY so a label, its tooltip and its threshold are
# defined once for the report templates, the viewer's JavaScript and the GUI.
#
# Format is a Python format spec, plus "pct" for a 0-1 fraction written as a percentage,
# "sig3" for three significant figures with no exponent, and a key of DISPLAY_UNITS for a
# quantity stored in SI and printed in a unit that keeps it between about 0.1 and 1000.
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

# format key -> (factor from the stored SI value, unit printed beside the label)
DISPLAY_UNITS = {
    "uM":  (1e6, "µM"),
    "uM2": (1e12, "µM²/Hz"),
    "mOD": (1e3, "mOD/sample"),   # GVTD differences consecutive samples, so it is per sample
}
METRIC_DISPLAY: dict[str, tuple[str, str, "tuple[float, float] | None", "str | None"]] = {
    # coupling
    "sci_mean":                ("Mean SCI (whole run)", ".3f", (0.75, 0.5), _HIGHER),
    # no cutoffs: the published ones are for the whole-run estimator above
    "sci_win_mean":            (f"Mean SCI ({SCI_WINDOW_S:g} s)", ".3f", None, _HIGHER),
    "channel_retention_rate":  ("Channel retention", "pct", (0.9, 0.7), _HIGHER),
    "psp_mean":                (f"Mean PSP ({PSP_WINDOW_S:g} s)", ".3f", None, _HIGHER),
    "good_frac_mean":          ("Passing windows", "pct", (0.75, 0.5), _HIGHER),
    "cp_mean":                 ("Mean CP (exp.)", ".3f", None, None),
    "heart_rate":              ("Heart rate (Hz)", ".2f", None, None),
    # raw intensity
    "cv_mean":                 (f"Mean CV ({CV_WINDOW_S:g} s)", ".3f", None, _LOWER),
    "snr_mean":                (f"Mean SNR μ/σ ({CV_WINDOW_S:g} s)", ".1f", None, _HIGHER),
    "snr_pass_rate":           ("SNR pass rate", "pct", None, _HIGHER),
    "n_flat_channels":         ("Flat channels", "d", (1, 2), _LOWER),
    "mean_amp_mean":           ("Mean amplitude", "sig3", None, None),
    "n_saturated_channels":    ("Saturated channels", "d", None, _LOWER),
    "saturated_pct":           ("Saturated share", "pct", None, _LOWER),

    # geometry
    "ch_dist_mean":            ("Mean separation (mm)", ".1f", None, None),
    "ch_dist_min":             ("Min separation (mm)", ".1f", None, None),
    "ch_dist_max":             ("Max separation (mm)", ".1f", None, None),
    # haemoglobin
    "hbo_hbr_corr_mean":       ("HbO-HbR corr", ".3f", None, _LOWER),
    "cnr_hbo_mean":            ("CNR HbO", ".3f", None, _HIGHER),
    "cnr_hbr_mean":            ("CNR HbR", ".3f", None, _LOWER),
    "cnr_n_epochs":            ("CNR epochs", "d", None, None),
    "gcor_hbo":                ("Global corr HbO", ".3f", None, None),
    "gcor_hbr":                ("Global corr HbR", ".3f", None, None),
    "lowfreq_drift_amplitude_hbo": ("Low-freq drift (HbO)", "uM", None, _LOWER),
    "lowfreq_drift_amplitude_hbr": ("Low-freq drift (HbR)", "uM", None, _LOWER),
    # spectral
    "cardiac_band_power_hbo":  ("Cardiac band power (HbO)", "uM2", None, None),
    "cardiac_band_power_hbr":  ("Cardiac band power (HbR)", "uM2", None, None),
    # descriptive, not a quality reading: motion raises it, so it has no better end
    "cardiac_band_frac_hbo":   ("Cardiac band (HbO)", "pct", None, None),
    "cardiac_band_frac_hbr":   ("Cardiac band (HbR)", "pct", None, None),
    "resp_band_power_hbo":     ("Resp band power (HbO)", "uM2", None, None),
    "resp_band_power_hbr":     ("Resp band power (HbR)", "uM2", None, None),
    "resp_band_frac_hbo":      ("Resp band (HbO)", "pct", None, None),
    "resp_band_frac_hbr":      ("Resp band (HbR)", "pct", None, None),

    # motion and spikes
    "gvtd_mean":               ("GVTD mean", "mOD", None, _LOWER),
    "gvtd_p95":                ("GVTD p95", "mOD", None, _LOWER),
    "gvtd_filt_mean":          (f"GVTD mean {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz", "mOD", None, _LOWER),
    "gvtd_filt_p95":           (f"GVTD p95 {GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz", "mOD", None, _LOWER),
    "gvtd_vstd_mean":          ("GVTD mean (var-normalised)", ".3f", None, None),
    "gvtd_vstd_p95":           ("GVTD p95 (var-normalised)", ".3f", None, None),
    "gvtd_thresh":             ("GVTD threshold", "mOD", None, None),
    "gvtd_thresh_applied":     ("GVTD threshold applied", "mOD", None, None),
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
    "gyro_speed_mean":         ("Gyroscope speed mean", "sig3", None, None),
    "gyro_speed_p95":          ("Gyroscope speed p95", "sig3", None, None),
    "accel_jerk_mean":         ("Accelerometer jerk mean", "sig3", None, None),
    "accel_jerk_p95":          ("Accelerometer jerk p95", "sig3", None, None),

    # GLM residual: 1 is the target from both sides, so neither end is the better one here
    "whitening_lowband_hbo":   ("Residual low-band power HbO", ".2f", None, None),
    "whitening_lowband_hbr":   ("Residual low-band power HbR", ".2f", None, None),

    # time
    "pct_data_retained":       ("% data retained", "pct", (0.8, 0.6), _HIGHER),
}

MISSING_VALUE = "\u2014"

# A condition has no verdict of its own: its share is what passes on its stretch, so it is
# named for that wherever a condition is shown, never as a retention.
_CONDITION_LABELS = {
    "channel_retention_rate": "Passing in condition",
}
_CONDITION_SUMMARIES = {
    "channel_retention_rate": "Share of channels that pass on this condition alone, against the run's line; a guide that rejects nothing. Higher is better.",
}

def condition_label(metric: str) -> "str | None":
    return _CONDITION_LABELS.get(metric)

# A condition's windowed values are columns of the run's --window-length grid, not the fixed
# screening window their whole-run labels name.
_GRID_SLICED = {"sci_win_mean": SCI_WINDOW_S, "psp_mean": PSP_WINDOW_S,
                "cv_mean": CV_WINDOW_S, "snr_mean": CV_WINDOW_S}

def grid_window_label(label: str, metric: str, window_s: "float | None") -> str:
    """``label`` naming the grid a condition value was sliced from: ``window_s``, or
    "windowed" when the runs read together were measured on different grids.

    ::

      ("SCI (10 s)", "sci_win_mean", 12.0) -> "SCI (12 s)"
    """
    if metric not in _GRID_SLICED:
        return label
    return label.replace(f"({_GRID_SLICED[metric]:g} s)",
                         f"({window_s:g} s)" if window_s else "(windowed)")

def shared_window(windows) -> "float | None":
    """The one grid every run was measured on, or None when they differ or one is unknown."""
    found = {float(w) if w else None for w in windows}
    return found.pop() if len(found) == 1 else None

def metric_label(metric: str, fallback: str | None = None, condition: bool = False) -> str:
    if condition and metric in _CONDITION_LABELS:
        return _CONDITION_LABELS[metric]
    spec = METRIC_DISPLAY.get(metric)
    if spec is not None:
        return with_unit(metric, spec[0])
    return metric if fallback is None else fallback

def metric_unit(metric: str) -> str:
    return DISPLAY_UNITS.get(metric_format(metric), (1.0, ""))[1]

def display_value(metric: str, value: float) -> float:
    """A stored value in the unit its label names, for a figure that plots it."""
    return value * DISPLAY_UNITS.get(metric_format(metric), (1.0, ""))[0]

def with_unit(metric: str, label: str) -> str:
    """``label`` with the metric's display unit, folded into a trailing parenthesis.

    with_unit("lowfreq_drift_amplitude_hbo", "Low-freq drift (HbO)") -> "Low-freq drift (HbO, µM)"
    with_unit("gvtd_p95", "GVTD p95")                                -> "GVTD p95 (mOD/sample)"
    with_unit("sci_mean", "Mean SCI")                                -> "Mean SCI"
    """
    unit = metric_unit(metric)
    if not unit or unit in label:
        return label
    if label.endswith(")"):
        return f"{label[:-1]}, {unit})"
    return f"{label} ({unit})"

def sig3(value: float) -> str:
    """Three significant figures, never an exponent; thousands grouped above 1000.
    sig3(0.000151) -> "0.000151"   sig3(13.59) -> "13.6"   sig3(574812) -> "575,000"
    """
    v = float(f"{float(value):.3g}")
    if v == 0 or not math.isfinite(v):
        return f"{v:g}"
    decimals = 2 - math.floor(math.log10(abs(v)))
    return f"{v:,.{decimals}f}" if decimals > 0 else f"{v:,.0f}"

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
        if spec_fmt == "sig3":
            return sig3(value)
        if spec_fmt in DISPLAY_UNITS:
            return sig3(float(value) * DISPLAY_UNITS[spec_fmt][0])
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

    ::

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
            "label":      metric_label(key, condition=condition),
            "value":      format_metric(key, value),
            "cls":        metric_class(key, value),
            "tip":        metric_summary(key, condition),
            "key_metric": is_key_metric(key),
        })
    return rows

# ---- what a run actually did ----
# Within one depth of the graph, the order sentences read in: a dyad's coherence and its
# phase, its correlation, how both were grouped into regions, then its nulls; a run's
# regression before ALFF and FC.
_RANK = {"hyper_phasescale": 0.5, "hyper_isc": 1, "hyper_isc_crossed": 1, "hyper_roi": 2, "hyper_phasenull": 3,
         "hyper_pairnull": 4, "alff": 3, "fc": 4, "hbt": -0.25,
         "optode_landmarks": -2, "optode_frame": -2,
         "excluded_unselected": -1, "excluded_edge": -0.9, "excluded_manual": -0.75,
         "gvtd_censoring": 0.5,
         "censor_fill": -0.5}

# the post steps that run on the filled copy, any of which can carry the fill sentence
_FILLED_STEPS = ("bandpass", "resample", "glm_fit", "glm_residuals", "glm_residuals_broadband")


def span_keys(step: str | None, params: dict[str, Any], spans: list) -> list[str]:
    """Sections a step's ``BAD_`` spans add beside its own sentence.

    ::

      sci_pruning with BAD_unselected and --gvtd-censor  ->  ["excluded_unselected", "gvtd_censoring"]
    """
    kinds = {span.get("kind") for span in spans or []}
    if step == "sci_pruning":
        return ((["excluded_unselected"] if "unselected" in kinds else [])
                + (["excluded_edge"] if any(span.get("description") == EDGE
                                            for span in spans or []) else [])
                + (["excluded_manual"] if params.get("bad_spans") else [])
                + (["gvtd_censoring"] if params.get("gvtd_censor") else []))
    if step in _FILLED_STEPS and "corrupted" in kinds and params.get("censor_fill"):
        return ["censor_fill"]
    return []

def _describe(nodes) -> list[tuple[str, dict[str, str]]]:
    """(steps.toml key, filled slots) for the method steps among ``nodes``, in data order.
    A step that ran more than once contributes one entry, the later files filling only what
    the earlier ones left out (a regression's fit table and its residual each carry part of
    the picture).
    """
    keyed = [(node, key) for node in nodes
             for key in [boilerplate_key(node.step, node.params),
                         *span_keys(node.step, node.params, node.excluded_spans),
                         # defined once, before the first sentence that reads HbT
                         *(["hbt"] if _holds_hbt(node.params) else [])]]
    keyed = sorted(((node, key) for node, key in keyed if key is not None),
                   key=lambda nk: (nk[0].depth, _RANK.get(nk[1], 0), nk[0].label))
    merged: dict[str, dict[str, Any]] = {}
    for node, key in keyed:
        merged[key] = {**node.params, "excluded_spans": node.excluded_spans,
                       **merged.get(key, {})}
    return [(key, template_slots(key, params)) for key, params in merged.items()]

def steps_from_sidecars(
    nirs_dir: Path, label: str | None = None,
) -> list[tuple[str, dict[str, str]]]:
    """Every method step the sidecars under ``nirs_dir`` record, ordered by graph depth.
    ``label`` is a run stem (``sub-01_task-rest``) and keeps another run's files out of the
    paragraph, the way :func:`~nirspipe.qc.common.provenance.scan` scopes its graph.
    """
    from nirspipe.qc.common.provenance import scan

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
    from nirspipe.qc.common.provenance import _key, scan
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
