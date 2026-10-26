"""Postprocessing pipeline: mode-driven, parameter-configured.

Parameters come from PostConfig, which can be populated from CLI flags or a TOML file.
All steps within a mode are still individually controllable via PostConfig fields.

"""

from __future__ import annotations
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import mne
import mne.io
import numpy as np
import pandas as pd

from nirspipe.io.auxiliary import find_aux_table
from nirspipe.pipeline.block_average import write_block_average
from nirspipe.pipeline.censor_fill import DEFAULT_CENSOR_FILL, fill_corrupted
from nirspipe.pipeline.chromophores import (DEFAULT_CHROMOPHORES, chroma_of, hbt_names, hbt_of,
                                            in_order, with_hbt)
from nirspipe.pipeline.glm_censored import kept_frames
from nirspipe.io.tables import read_table, write_tsv
from nirspipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    bandpass_filter,
    filter_description,
    resample,
)
from nirspipe.pipeline.glm import glm_inputs, run_glm_pipeline, sole_regressor_channels
from nirspipe.exceptions import StageError
from nirspipe.utils import is_marker
from nirspipe.utils.lineage import Recorder, carried_params, lineage_of, stage_of, stamp
from nirspipe.qc.metrics._helpers import long_short_channels, separation_bands
from nirspipe.utils.logging import get_logger
from nirspipe import __version__
from nirspipe.io.derivatives import (
    read_json, write_sidecar_json, write_step_sidecar, build_output_path, carry_entities, data_state,
    nirs_required_fields, require_nirs_fields, split_sources,
)
from nirspipe.io.snirf import read_snirf, write_snirf
from nirspipe.utils.spans import excluded_spans

logger = get_logger("post.pipeline")

Mode = Literal["denoise", "glm", "rest"]

# --short-channel's default, decided per run: pca on a montage with short channels, else none
SHORT_CHANNEL_AUTO = "auto"
# --drift-high-pass auto, decided per run from its events; glm mode only
DRIFT_HIGH_PASS_AUTO = "auto"
# the highest cutoff auto chooses, however fast the design repeats: nilearn's default
DRIFT_AUTO_CAP = 0.01


@dataclass
class PostConfig:
    subject: str
    cardiac_l_freq: float
    cardiac_h_freq: float
    resp_l_freq: float
    resp_h_freq: float
    session: str | None = None

    # bandpass filter
    high_pass: float | None = None
    low_pass:  float | None = None
    filter_method: str = DEFAULT_FILTER_METHOD
    filter_order:  int = DEFAULT_FILTER_ORDER
    # how corrupted BAD_ spans are filled before the bandpass; see pipeline/censor_fill.py
    censor_fill:   str = DEFAULT_CENSOR_FILL

    roi_map: dict | None = None
    # names the seg- entity on every ROI output, so one tree can hold two ROI
    # definitions. Defaults to the stem of the file the map was read from
    roi_map_name: str = "custom"

    # resample
    resample_sfreq: float | None = None

    # GLM (glm mode)
    stim_dur:        float | None          = None
    hrf_model:       str | None            = None
    # every mode fits it, so the library carries the default too rather than leaving a
    # direct caller to pass None into nilearn. "auto" is an AR order of 4x the sampling rate
    noise_model:     str                   = "auto"
    drift_model:     str | None            = None
    # Hz, or DRIFT_HIGH_PASS_AUTO until resolve_drift_high_pass reads the run's events
    drift_high_pass: float | str | None    = None
    # set where drift_high_pass was decided from "auto": what decided it, for the sidecars
    drift_high_pass_auto: dict | None      = None
    drift_order:     int | None            = None
    fir_delays:      tuple[int, ...] | None = None
    short_channel:   bool | str | None     = None
    # set where short_channel was decided from "auto", so the sidecars can say so
    short_channel_auto: bool               = False
    aux:             bool                  = False
    aux_channels:    list[str] | None      = None
    events_path:     str | None            = None
    contrast_def:    dict[str, Any] | None = None
    fc:              bool                  = False
    # seconds outside every BAD_ span that FC and ALFF need; 0 computes them on any amount
    min_time:        float                 = 0.0
    # a copy of the final residual with BAD_ frames set to each channel's kept mean
    export_mean_filled: bool               = False

    # separation bands, in metres; see PrepConfig
    short_max_dist: float | None = None
    long_min_dist: float | None = None
    long_max_dist: float | None = None

    combine_runs: bool | None = None

    # what the GLM tables, block average, FC and ALFF carry; the stage files keep HbO and HbR
    chromophores: tuple[str, ...] = DEFAULT_CHROMOPHORES
    # a P_CORRECTIONS method for the GLM and contrast tables' p values
    p_correction: str = "none"

    def __post_init__(self) -> None:
        self.chromophores = in_order(self.chromophores)
        # nilearn multiplies the cutoff by the frame times, so a missing one dies deep
        # inside it as "unsupported operand type(s) for *: 'NoneType' and 'float'"
        if self.drift_model == "cosine" and self.drift_high_pass is None:
            raise ValueError(
                "--drift-model cosine needs --drift-high-pass: the cosine drift basis is "
                "defined by its cutoff frequency, and there is no sensible default"
            )
        if isinstance(self.drift_high_pass, str) and self.drift_high_pass != DRIFT_HIGH_PASS_AUTO:
            raise ValueError(f"--drift-high-pass {self.drift_high_pass!r}: expected a cutoff "
                             f"in Hz or {DRIFT_HIGH_PASS_AUTO!r}")


def resolve_short_channel(config: PostConfig, raw: mne.io.Raw) -> PostConfig:
    """``config`` with ``short_channel = "auto"`` decided on ``raw``'s montage; any other value as it is.

    e.g. a montage with an 8 mm channel gets ``"pca"``, one with only 30 mm channels None.
    """
    if config.short_channel != SHORT_CHANNEL_AUTO:
        return config
    n_short = len(long_short_channels(raw, separation_bands(config))[1])
    logger.info("sub-%s | --short-channel auto: %s", config.subject,
                f"pca, {n_short} short channel(s)" if n_short
                else "none, the montage has no short channel")
    return replace(config, short_channel="pca" if n_short else None, short_channel_auto=True)


def _warn_fc_without_bandpass(config: PostConfig) -> None:
    if config.high_pass is None and config.low_pass is None:
        logger.warning("sub-%s | --fc without a bandpass: the correlations will be "
                       "dominated by drift", config.subject)


def _has_confounds(config: PostConfig) -> bool:
    """Whether denoise mode has anything to regress out.

    Unlike rest mode, denoise does not force a drift model: it has no ALFF branch whose
    input skips the bandpass, so the bandpass is already the detrend. Any one of the three
    on its own is enough to make the regression worth running.
    """
    return (bool(config.short_channel) or bool(config.aux)
            or config.drift_model not in (None, "none"))


def _warn_unmatched_design_band(config: PostConfig) -> None:
    """A task model fitted to bandpassed data needs a drift basis that covers the cutoff.

    The data reaches the GLM high-passed; the HRF-convolved task columns do not, so they
    carry variance below the cutoff that the data no longer has. That inflates the
    denominator of the task beta and biases it toward zero. A cosine drift basis at the same
    cutoff fixes it exactly, because projecting a regressor onto the complement of a basis
    spanning everything below the cutoff *is* a high-pass. A low-order polynomial does not
    span it and neither does no drift model at all.

    A warning rather than an error: how much it costs depends on block length against the
    cutoff, and only the caller knows their design.
    """
    if config.high_pass is None:
        return
    covered = (config.drift_model == "cosine"
               and config.drift_high_pass is not None
               and config.drift_high_pass >= config.high_pass)
    if covered:
        return
    logger.warning(
        "sub-%s | the data is high-passed at %g Hz but the task regressors are not, and "
        "--drift-model %s does not span that band. Task betas will be underestimated, "
        "mildly for short blocks and severely for long ones. Use --drift-model cosine with "
        "--drift-high-pass %g to match.",
        config.subject, config.high_pass, config.drift_model or "none", config.high_pass,
    )


def _warn_lowpass_breaks_whitening(config: PostConfig, mode: Mode) -> None:
    """An AR noise model fitted to low-passed data fits the filter, not the noise.

    Prewhitening estimates the residual's autocorrelation and divides it out, which is what
    makes a t value mean anything on a series sampled far faster than the response. A
    low-pass leaves the series almost perfectly predictable from its own past, so the AR
    coefficient lands at the stationarity boundary, the whitening filter becomes a first
    difference of a nearly integrated series, and the residual comes out as autocorrelated
    as it went in.

    What that costs depends on what the run is for, so the warning says a different thing
    per mode. A task GLM keeps its betas and loses its standard errors, which go the
    anti-conservative way. A run whose product is the residual loses more than that: the
    coefficients are estimated under a covariance the whitening got wrong, so the residual
    itself moves, and every inter-brain measure reading that file moves with it.

    Resampling counts as a low-pass: it anti-aliases at the new Nyquist, so it removes the
    same band under another flag's name.

    A warning rather than an error because a caller may want the filtered series for another
    reason.
    """
    if config.noise_model in (None, "ols"):
        return
    causes = []
    if config.low_pass is not None:
        causes.append(f"--low-pass {config.low_pass:g} Hz")
    if config.resample_sfreq is not None:
        causes.append(f"--resample-sfreq {config.resample_sfreq:g} Hz, which anti-aliases "
                      f"at {config.resample_sfreq / 2:g} Hz")
    if not causes:
        return
    cost = ("leaving the residual serially correlated and the t values anti-conservative; "
            "betas are unaffected. Drop the low-pass, or use --noise-model ols and treat the "
            "t values as uncalibrated"
            if mode == "glm" else
            "so the confound coefficients are fitted under the wrong covariance and "
            "desc-errts is not the residual an ordinary least-squares fit would leave. Use "
            "--noise-model ols for that residual")
    logger.warning(
        "sub-%s | --noise-model %s is fitted to data the run has low-passed (%s). The AR "
        "model will fit the filter rather than the noise, %s.",
        config.subject, config.noise_model, " and ".join(causes), cost,
    )


def _design_onsets(raw: mne.io.Raw, events: "pd.DataFrame | None") -> dict[str, list]:
    """Per condition, the onsets the design matrix will be built from.

    ``events`` is the table ``--events-path`` supplied; without one the recording's own
    annotations are what the design is built from. Reading the wrong one is how a warning
    ends up naming a condition the model does not contain. Markers of rejected time are
    dropped either way.
    """
    if events is not None:
        pairs = zip(events["trial_type"].astype(str), events["onset"])
    else:
        pairs = zip(raw.annotations.description, raw.annotations.onset)
    onsets: dict[str, list] = {}
    for desc, onset in pairs:
        if not is_marker(desc):
            continue
        onsets.setdefault(str(desc), []).append(float(onset))
    return onsets


def _repeat_intervals(raw: mne.io.Raw, events: "pd.DataFrame | None" = None) -> dict[str, float]:
    """Per condition, the longest gap between two of its trials.

    That gap is the slowest rhythm the design asks the model to fit, so it is the frequency
    a drift basis has to stay below. Per condition rather than one number for the run: a
    file mixing block markers with stimulus markers has both a slow and a fast rhythm, and
    which one matters depends on which condition the reader cares about. A condition seen
    once has no interval.
    """
    intervals: dict[str, float] = {}
    for desc, onsets in _design_onsets(raw, events).items():
        ordered = np.sort(np.asarray(onsets, dtype=float))
        if ordered.size >= 2:
            intervals[desc] = float(np.max(np.diff(ordered)))
    return intervals


def _warn_drift_absorbs_task(
    config: PostConfig, raw: mne.io.Raw, events: "pd.DataFrame | None" = None,
) -> None:
    """A drift basis reaching a condition's own rhythm fits that condition away as drift.

    The cosine basis spans everything below its cutoff, so the cutoff has to sit below the
    frequency at which the condition repeats. The warning suggests half that frequency, an
    octave of room before the basis reaches the task.
    """
    if config.drift_model != "cosine" or config.drift_high_pass is None:
        return

    intervals = _repeat_intervals(raw, events)
    spanned = {name: gap for name, gap in intervals.items()
               if gap > 0 and config.drift_high_pass >= 1.0 / gap}
    if not spanned:
        return

    slowest = max(spanned.values())
    named = ", ".join(f"{name} (every {gap:.0f} s)"
                      for name, gap in sorted(spanned.items(), key=lambda kv: -kv[1]))
    logger.warning(
        "sub-%s | --drift-high-pass %g Hz spans the rhythm of %d of %d conditions, so the "
        "drift basis will absorb them: %s. Use %.5g Hz or lower to clear the slowest, or "
        "drop the conditions you are not modelling.",
        config.subject, config.drift_high_pass, len(spanned), len(intervals),
        named, 1.0 / (2.0 * slowest),
    )


def resolve_drift_high_pass(
    config: PostConfig, raw: mne.io.Raw, events: "pd.DataFrame | None" = None,
) -> PostConfig:
    """``config`` with ``drift_high_pass = "auto"`` set from the run's events; any other value as it is.

    1/(2 x the longest gap between two trials of one condition), the slowest condition
    deciding, capped at DRIFT_AUTO_CAP and raised to the data's own high-pass, which the
    basis has to cover. e.g. trials of ``a`` at most 100 s apart give 0.005 Hz.
    """
    if config.drift_high_pass != DRIFT_HIGH_PASS_AUTO:
        return config
    onsets = {name: set(times) for name, times in _design_onsets(raw, events).items()}
    # a doubled annotation repeats nothing, so a gap of zero is no interval
    intervals = {name: gap for name, gap in _repeat_intervals(raw, events).items() if gap > 0}
    if not intervals:
        raise ValueError("--drift-high-pass auto: no condition occurs twice in this run, so "
                         "there is no interval to set the cutoff from. Type a cutoff in Hz.")
    condition = max(intervals, key=intervals.get)
    gap = intervals[condition]
    cutoff, bound = 1.0 / (2.0 * gap), None
    if cutoff > DRIFT_AUTO_CAP:
        cutoff, bound = DRIFT_AUTO_CAP, "cap"
    if config.high_pass is not None and cutoff < config.high_pass:
        cutoff, bound = config.high_pass, "high_pass"
    once = sorted(name for name in onsets if name not in intervals)
    auto = {"condition": condition, "interval_s": gap, "trials": len(onsets[condition]),
            "bound": bound, "seen_once": once}

    logger.info("sub-%s | --drift-high-pass auto: %g Hz, 1/(2 x %.1f s), the longest gap "
                "between two trials of %s%s", config.subject, cutoff, gap, condition,
                {"cap": f", capped at {DRIFT_AUTO_CAP:g} Hz",
                 "high_pass": f", raised to the data's {cutoff:g} Hz high-pass"}.get(bound, ""))
    if auto["trials"] == 2:
        logger.warning(
            "sub-%s | --drift-high-pass auto took its cutoff from %s, which occurs only twice, "
            "%.0f s apart. If it marks something other than a condition, leave it out with "
            "--events-path or nirspipe-prep edit-markers, or type a cutoff.",
            config.subject, condition, gap)
    if once:
        logger.warning("sub-%s | --drift-high-pass auto: %s occur(s) once, so set no interval",
                       config.subject, ", ".join(once))
    return replace(config, drift_high_pass=cutoff, drift_high_pass_auto=auto)


def run_post(
    raw_haemo: mne.io.Raw,
    config: PostConfig,
    output_dir: Path,
    mode: Mode = "denoise",
    source_entities: dict[str, str] | None = None,
    source_path: Path | None = None,
    epoch_tmin: float | None = None,
    epoch_tmax: float | None = None,
    epoch_chunk_duration: float | None = None,
) -> tuple:
    """Run post-processing pipeline.

    Returns (result, glm_est, design_matrix, alff_df, fc, fc_seed, fc_roi).
    glm_est / design_matrix are None only when no regression ran; denoise and rest return
    them too, fitted on empty events, so the design holds confound columns and no condition.
    alff_df is None outside rest mode. The FC products are written by rest mode, and by glm
    and denoise mode under ``config.fc``. fc, fc_seed and fc_roi are {chromophore: frame},
    the last two empty without --roi-mapping.
    """

    config = resolve_short_channel(config, raw_haemo)
    if config.drift_high_pass == DRIFT_HIGH_PASS_AUTO and mode != "glm":
        raise ValueError(f"--drift-high-pass auto reads the task's events, which only glm mode "
                         f"models; {mode} needs a cutoff in Hz")
    rec = Recorder()
    if source_path is not None:
        rec.register_input(source_path, raw_haemo)

    # the folder the stage snirfs go to, session level included, so the GLM tables sit beside them
    nirs_dir = output_dir / f"sub-{config.subject}" / (
        f"ses-{config.session}" if config.session else "") / "nirs"

    known_spans = _known_spans(source_path)
    if source_path is not None:
        require_nirs_fields(source_path)
    if config.censor_fill == "spline":
        logger.warning("sub-%s | --censor-fill spline: a cubic spline across a shared gap may "
                       "inflate coherence when both members of a pair miss the same stretch",
                       config.subject)
    # before every continuous step, the broadband branch's included
    filled = fill_corrupted(raw_haemo.copy(), config.censor_fill)
    result = filled.copy()

    aux_path = None
    if config.aux:
        aux_path = find_aux_table(Path(source_path)) if source_path else None
        if aux_path is None:
            logger.warning("sub-%s | --aux-regressors asked for but no aux table beside %s; "
                           "the recording may carry no aux channels, or prep predates them",
                           config.subject, Path(source_path).name if source_path else "the input")
    # every regression below takes the same three, so they travel together
    aux_kwargs = dict(aux_path=aux_path, aux_channels=config.aux_channels,
                      data_band=(config.high_pass, config.low_pass),
                      data_filter_method=config.filter_method,
                      data_filter_order=config.filter_order)

    if config.high_pass is not None or config.low_pass is not None:
        logger.info("sub-%s | bandpass: %s", config.subject,
                    filter_description(config.high_pass, config.low_pass,
                                       config.filter_method, config.filter_order))
        result = bandpass_filter(result, l_freq=config.high_pass, h_freq=config.low_pass,
                                 method=config.filter_method, order=config.filter_order)
        _write_step_snirf(result, config, output_dir, desc="filtered", rec=rec, source_entities=source_entities, known_spans=known_spans)

    if config.resample_sfreq is not None:
        logger.info("sub-%s | resample -> %.1f Hz", config.subject, config.resample_sfreq)
        result = resample(result, config.resample_sfreq)
        _write_step_snirf(result, config, output_dir, desc="resampled", rec=rec, source_entities=source_entities, known_spans=known_spans)

    # SQM is not computed here either; the record is assembled from disk after this
    # pipeline returns, which is what lets one writer own the whole file.

    glm_est = dm = alff_df = None
    fc: dict = {}
    fc_seed: dict = {}
    fc_roi: dict = {}
    raw_resid = None  # set by the glm/rest/denoise branches
    if mode == "glm":
        # noise_model is not here: it has a default, unlike these two, which are design
        # choices with no safe one
        missing = [f for f in ("hrf_model", "drift_model") if getattr(config, f) is None]
        if missing:
            raise ValueError(f"GLM mode requires: {', '.join('--' + f.replace('_', '-') for f in missing)}")
        if config.events_path is not None and config.stim_dur is not None:
            raise ValueError("--events-path and --stim-dur are mutually exclusive")
        logger.info("sub-%s | GLM (%s / %s)", config.subject, config.hrf_model, config.noise_model)
        # the same table the design will be built from, so the warning cannot name a
        # condition the model does not carry
        design_events = read_table(config.events_path) if config.events_path else None
        if not _design_onsets(result, design_events):
            where = f"--events-path {config.events_path}" if config.events_path else "the recording's annotations"
            raise ValueError(
                f"GLM mode found no events in {where}, so the model would hold drift and "
                f"confounds only. Use --mode denoise for confound regression without a task model."
            )
        config = resolve_drift_high_pass(config, result, design_events)
        _warn_unmatched_design_band(config)
        _warn_lowpass_breaks_whitening(config, mode)
        _warn_drift_absorbs_task(config, result, design_events)
        _, glm_est, dm, raw_resid = run_glm_pipeline(
            result,
            stim_dur=config.stim_dur,
            hrf_model=config.hrf_model,
            noise_model=config.noise_model,
            drift_model=config.drift_model,
            high_pass=config.drift_high_pass,
            drift_high_pass_auto=config.drift_high_pass_auto,
            drift_order=config.drift_order,
            fir_delays=config.fir_delays,
            short_channel=config.short_channel,
            short_channel_auto=config.short_channel_auto,
            sep_bands=separation_bands(config),
            **aux_kwargs,
            events_path=config.events_path,
            contrast_def=config.contrast_def,
            output_dir=str(nirs_dir),
            source_path=rec.path_of(result),
            chromophores=config.chromophores,
            p_correction=config.p_correction,
        )
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities, known_spans=known_spans,
                          inputs=glm_inputs(config.events_path, aux_path))
        # on the series the model was fitted to, over the trial window the report's epoch figures use
        write_block_average(result, nirs_dir, rec.path_of(result), epoch_tmin, epoch_tmax,
                            epoch_chunk_duration, events=design_events,
                            inputs=glm_inputs(config.events_path, None),
                            chromophores=config.chromophores)

        # FC on the task residual: the task is in the design matrix, so what correlates here
        # is what the model did not explain.
        if config.fc:
            _warn_fc_without_bandpass(config)
            fc, fc_roi, fc_seed = _write_fc_derivatives(
                raw_resid, config, output_dir, rec, source_entities=source_entities, known_spans=known_spans)

    elif mode == "rest":
        if config.drift_model is None:
            raise ValueError("rest mode requires --drift-model")
        _warn_lowpass_breaks_whitening(config, mode)
        logger.info("sub-%s | rest confound regression", config.subject)
        rest_glm_kwargs = dict(
            stim_dur=None,
            hrf_model="spm",
            noise_model=config.noise_model,
            drift_model=config.drift_model,
            high_pass=config.drift_high_pass,
            drift_order=config.drift_order,
            fir_delays=None,
            short_channel=config.short_channel,
            short_channel_auto=config.short_channel_auto,
            sep_bands=separation_bands(config),
            **aux_kwargs,
            events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),
            chromophores=config.chromophores,
            p_correction=config.p_correction,
        )
        _, glm_est, dm, raw_resid = run_glm_pipeline(
            result,
            output_dir=str(nirs_dir),
            source_path=rec.path_of(result),
            **rest_glm_kwargs,
        )
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities, known_spans=known_spans,
                          inputs=glm_inputs(None, aux_path))

        # ALFF/fALFF need a broadband residual: fALFF's denominator spans the full spectrum,
        # so its input must not be low-passed. Re-run the same confound regression on the
        # un-bandpassed data (the drift model supplies the detrend). FC keeps raw_resid above.
        # This is the only path without a bandpass, so the drift model is its sole detrend.
        # "none" and an order-0 polynomial leave the drift in, and a linear ramp's leakage
        # lands inside the ALFF band, so ALFF is skipped rather than written wrong.
        drift_detrends = (
            config.drift_model == "cosine"
            or (config.drift_model == "polynomial"
                and (config.drift_order is None or config.drift_order >= 1))
        )

        raw_resid_bb = None
        if config.low_pass is not None and config.high_pass is not None:
            if not drift_detrends:
                logger.warning(
                    "sub-%s | skipping ALFF/fALFF: --drift-model %s (order %s) does not remove "
                    "linear drift, and the broadband residual has no bandpass to remove it",
                    config.subject, config.drift_model, config.drift_order,
                )
            else:
                result_bb = filled.copy()
                if config.resample_sfreq is not None:
                    result_bb = resample(result_bb, config.resample_sfreq)
                # its input skipped the bandpass, so band-matching the aux to one would
                # put the regressors in a narrower band than the data they explain
                _, _, _, raw_resid_bb = run_glm_pipeline(
                    result_bb, **{**rest_glm_kwargs, "data_band": None})
                # Same regression, un-bandpassed input. Re-stamp so it stops sharing the "errts"
                # stage with the bandpassed residual, whose file it would otherwise be credited to.
                # carry the regression's own parameters over; a bare re-stamp overwrites them
                # and say it skipped the bandpass, which the stage writer would otherwise
                # fill in from the run's config
                stamp(raw_resid_bb, stage="errtsbroad", step="glm_residuals_broadband",
                      source=raw_haemo,
                      **{**carried_params(raw_resid_bb),
                         "resample_sfreq": config.resample_sfreq,
                         "high_pass": None, "low_pass": None,
                         "filter_method": None, "filter_order": None})
                _write_step_snirf(raw_resid_bb, config, output_dir, desc="errtsbroad",
                                  rec=rec, source_entities=source_entities, known_spans=known_spans,
                                  inputs=glm_inputs(None, aux_path))

        alff_df, fc, fc_roi, fc_seed = _write_rest_derivatives(
            raw_resid, raw_resid_bb, config, output_dir, rec, source_entities=source_entities, known_spans=known_spans)

    elif mode == "denoise":
        # The regression and the FC are independent here. Either can run without the other,
        # which is what separates this mode from rest: rest forces a drift model and always
        # writes ALFF, so bandpass-then-correlate has no route through it.
        if _has_confounds(config):
            # The same empty-events confound regression rest runs, and no task model. Task
            # data whose systemic physiology has to go without the task being modelled
            # (hyperscanning, seed connectivity) ends here.
            _warn_lowpass_breaks_whitening(config, mode)
            logger.info("sub-%s | denoise confound regression", config.subject)
            _, glm_est, dm, raw_resid = run_glm_pipeline(
                result,
                stim_dur=None,
                hrf_model="spm",
                noise_model=config.noise_model,
                drift_model=config.drift_model or "none",
                high_pass=config.drift_high_pass,
                drift_order=config.drift_order,
                fir_delays=None,
                short_channel=config.short_channel,
                short_channel_auto=config.short_channel_auto,
                sep_bands=separation_bands(config),
                **aux_kwargs,
                events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),
                output_dir=str(nirs_dir),
                source_path=rec.path_of(result),
                chromophores=config.chromophores,
                p_correction=config.p_correction,
            )
            _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities, known_spans=known_spans,
                              inputs=glm_inputs(None, aux_path))

        # No ALFF, for glm's reason: the source is bandpassed and fALFF's denominator spans
        # the full spectrum, so the number would be ~1 by construction.
        if config.fc:
            _warn_fc_without_bandpass(config)
            fc, fc_roi, fc_seed = _write_fc_derivatives(
                raw_resid if raw_resid is not None else result,
                config, output_dir, rec, source_entities=source_entities, known_spans=known_spans)

    if config.export_mean_filled:
        # the residual FC reads: the regression's, or the last stage when none ran
        _write_mean_filled(raw_resid if raw_resid is not None else result, config, output_dir,
                           rec, source_entities, known_spans)

    # GCOR around the confound regression is not measured here. Both of its stages are
    # written to disk and the SQM record measures each one per channel set, so the report
    # reads `filtered` -> `errts`.
    return result, glm_est, dm, alff_df, fc, fc_seed, fc_roi


def final_stage(config: PostConfig, mode: Mode) -> "str | None":
    """The desc of the series a run ends on, the one FC reads; None when that is the run's input."""
    if mode in ("glm", "rest") or _has_confounds(config):
        return "errts"
    if config.resample_sfreq is not None:
        return "resampled"
    if config.high_pass is not None or config.low_pass is not None:
        return "filtered"
    return None


def write_combined_runs(config: PostConfig, output_dir: Path, mode: Mode,
                        runs: "list[dict[str, str]]") -> "Path | None":
    """Join one task's runs after postprocessing and compute FC on the joined series.

    ``runs`` holds the source entities of each run of one subject, session and task, in run
    order. Each run's final series (:func:`final_stage`) is z-scored per channel over its kept
    frames and the runs are joined (:func:`~nirspipe.pipeline.combine_runs.join_runs`), a
    channel taking part only in the runs where it is good and NaN in the others; ROIs are
    averaged per run over that run's good members before the z-score. The joined series is
    written without the run entity, its sidecar's ``combined_runs`` naming each run and its
    onset and ``channel_runs`` the runs each channel took part in; it is bad only where every
    run marked it. The FC products the mode writes are computed on it, each pair over the
    kept frames both channels have. ALFF and the GLM's tables stay per run. A run under
    ``--min-time`` is left out, and fewer than two runs left join nothing. Returns the joined
    file, or None.
    """
    from nirspipe.io.optode_space import copy_placement
    from nirspipe.pipeline.combine_runs import join_roi_signals, join_runs

    stage = final_stage(config, mode)
    if stage is None:
        logger.warning("sub-%s | --combine-runs: the runs end on their own input (no bandpass, "
                       "resampling or regression), so nothing is joined", config.subject)
        return None
    paths, raws, labels = [], [], []
    for entities in runs:
        path = build_output_path(output_dir=output_dir, subject=config.subject,
                                 session=config.session,
                                 entities={**carry_entities(entities), "desc": stage},
                                 suffix="nirs", extension=".snirf")
        raw = read_snirf(path)
        keep = kept_frames(raw)
        kept_s = float(keep.sum() / raw.info["sfreq"])
        if kept_s < config.min_time or keep.sum() < 3:
            logger.warning("sub-%s | --combine-runs: %s left out, %.1f s kept under "
                           "--min-time %g s", config.subject, path.name, kept_s,
                           config.min_time)
            continue
        paths.append(path)
        raws.append(raw)
        labels.append(f"run-{entities['run']}")
    if len(raws) < 2:
        logger.warning("sub-%s | --combine-runs: fewer than two runs of task %s to join",
                       config.subject, runs[0].get("task"))
        return None
    # a short channel regressed out of itself has no residual left to correlate in that run
    absent = [set(raw.info["bads"]) | set(sole_regressor_channels(
        raw, config.short_channel, separation_bands(config))) for raw in raws]
    joined, onsets = join_runs(raws, absent)
    rec = Recorder()
    # the first run under a stage of its own, since the joined series takes that run's stage
    rec.register_input(paths[0], stamp(raws[0], stage="combine_input", step="load"))
    stamp(joined, stage=stage, step="combine_runs", source=raws[0],
          combined_runs=[{"run": p.name, "onset_s": round(o, 6)} for p, o in zip(paths, onsets)],
          channel_runs={ch: [label for label, gone in zip(labels, absent) if ch not in gone]
                        for ch in joined.ch_names},
          zscored="per run and channel, over its kept frames")
    known = [{**span, "onset": span["onset"] + onset}
             for path, onset in zip(paths, onsets)
             for span in read_json(path.with_suffix(".json")).get("excluded_spans", [])]
    entities = {k: v for k, v in carry_entities(runs[0]).items() if k != "run"}
    written = _write_step_snirf(joined, config, output_dir, desc=stage, rec=rec,
                                source_entities=entities, known_spans=known, inputs=paths[1:])
    try:
        copy_placement(paths[0], written)
    except StageError as exc:
        logger.warning("sub-%s | --combine-runs: no optode placement carried over: %s",
                       config.subject, exc)
    if mode == "rest" or config.fc:
        # each run is z-scored before the join, so HbT is summed in each run, not after
        hbt_runs = [hbt_of(raw) for raw in raws] if "hbt" in config.chromophores else []
        joined_hbt = (join_runs(hbt_runs, [set(h.info["bads"]) | set(hbt_names(gone))
                                           for h, gone in zip(hbt_runs, absent)])[0]
                      if hbt_runs else None)
        rois = ({c: join_roi_signals(hbt_runs if c == "hbt" else raws, labels, config.roi_map, c)
                 for c in config.chromophores} if config.roi_map else None)
        _write_fc_derivatives(joined, config, output_dir, rec, source_entities=entities,
                              known_spans=known, joined_rois=rois, joined_hbt=joined_hbt)
    return written


# what desc-meanfilled's sidecar says about the frames it altered
MEAN_FILLED_DESCRIPTION = (
    "The final residual with every frame inside a BAD_ span set to its channel's mean over "
    "the frames outside them, annotations kept; for tools that read a continuous array and "
    "ignore annotations. The filled frames are placeholders, not signal: do not compute ALFF, "
    "fALFF or a GLM from this file.")


def mean_filled(raw: mne.io.Raw) -> mne.io.Raw:
    """A copy whose ``BAD_`` frames hold each channel's mean over its kept frames, stamped ``meanfilled``.

    ::

      kept 1, 3 around a BAD_ frame  ->  1, 2, 3
    """
    keep = kept_frames(raw)
    out = raw.copy().load_data()
    if not keep.all():
        out._data[:, ~keep] = out._data[:, keep].mean(axis=1, keepdims=True)
    return stamp(out, stage="meanfilled", step="export_mean_filled", source=raw,
                 mean_filled_s=round(float((~keep).sum() / raw.info["sfreq"]), 3))


def _write_mean_filled(raw: mne.io.Raw, config: PostConfig, output_dir: Path, rec: Recorder,
                       source_entities: "dict[str, str] | None",
                       known_spans: "list[dict] | None") -> Path:
    return _write_step_snirf(mean_filled(raw), config, output_dir, desc="meanfilled",
                             rec=rec, source_entities=source_entities,
                             known_spans=known_spans, description=MEAN_FILLED_DESCRIPTION)

def _write_fc_derivatives(
    raw_resid: mne.io.Raw,
    config: PostConfig,
    output_dir: Path,
    rec: Recorder,
    source_entities: dict[str, str] | None = None,
    known_spans: "list[dict] | None" = None,
    joined_rois: "dict[str, tuple[dict, dict]] | None" = None,
    joined_hbt: "mne.io.BaseRaw | None" = None,
) -> tuple:
    """Write the FC, ROI FC and seed-map TSVs for one residual.

    Returns (fc, fc_roi, fc_seed), each {chromophore: frame}; the last two stay empty
    without a roi_map. ``joined_rois`` is
    {chromophore: (ROI signals, members per run)} from
    :func:`~nirspipe.pipeline.combine_runs.join_roi_signals`, for a joined series, and
    ``joined_hbt`` that series' HbT, joined from each run's own.

    All three modes land here, and the only thing that differs is what arrives: rest
    regresses confounds alone, glm regresses the task as well, and denoise passes either its
    own confound residual or, when no regression ran, the bandpassed data itself. The
    distinction lives in the caller, not here.
    """
    from nirspipe.pipeline.restingstate import (
        _roi_members, compute_fc, compute_fc_roi, compute_fc_seed, fisher_z,
    )

    entities = carry_entities(source_entities)

    series = (raw_resid.copy().add_channels([joined_hbt]) if joined_hbt is not None
              else with_hbt(raw_resid, config.chromophores))
    # the FC matrices keep their bad rows and columns so the shape stays predictable;
    # the sidecar names them so a consumer can drop or ignore them
    bads = list(series.info["bads"])
    # a channel that was the whole short-channel regressor was fitted against a copy of
    # itself, so its residual is zero and every correlation it takes part in is noise
    empty = sole_regressor_channels(raw_resid, config.short_channel,
                                    separation_bands(config))
    empty += hbt_names(empty) if "hbt" in config.chromophores else []
    src_bp = rec.path_of(raw_resid)

    def _path(extra: dict, statistic: str) -> Path:
        """One relationship matrix. What was computed goes in stat-, not in the suffix."""
        return build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities={**entities, **extra, "statistic": statistic},
            suffix="relmat", extension=".tsv",
        )

    def _roi(extra: dict) -> dict:
        return {**extra, "segmentation": config.roi_map_name}

    fc: dict[str, pd.DataFrame] = {}
    fc_roi: dict[str, pd.DataFrame] = {}
    fc_seed: dict[str, pd.DataFrame] = {}

    kept = _kept_time(raw_resid, config, "FC", rec)
    if kept is None:
        return fc, fc_roi, fc_seed

    def _sidecar(path: Path, step: str, **params) -> None:
        write_step_sidecar(path, step, src_bp, bads,
                           excluded_spans=excluded_spans(raw_resid, known=known_spans or ()),
                           chromophores=list(config.chromophores), **params, **kept)

    # FC per chromophore: HbO and HbR anti-correlate, so they never share a matrix
    for chromo in config.chromophores:
        fc_df = compute_fc(series, chromo)
        if fc_df.empty:
            continue
        for ch in (c for c in empty if c in fc_df.index):
            fc_df.loc[ch, :] = np.nan
            fc_df.loc[:, ch] = np.nan
        chromo_entities = {"chromophore": chromo}

        fc_path = _path(chromo_entities, "pearson")
        write_tsv(fc_df, fc_path, index=True, index_label="channel")
        _sidecar(fc_path, "fc", chromophore=chromo)
        logger.info("sub-%s | fc (%s) -> %s", config.subject, chromo, fc_path)

        fcz_path = _path(chromo_entities, "fisherz")
        write_tsv(fisher_z(fc_df), fcz_path, index=True, index_label="channel")
        _sidecar(fcz_path, "fisher_z", chromophore=chromo)
        logger.info("sub-%s | fcz (%s) -> %s", config.subject, chromo, fcz_path)

        if config.roi_map:
            signals = seeds = None
            if joined_rois:
                signals, members = joined_rois[chromo]
                # a seed's own channels are those averaged into it in any run
                seeds = {roi: [ch for ch in series.ch_names
                               if any(ch in run.get(roi, ()) for run in members.values())]
                         for roi in signals}
            else:
                members = _roi_members(series, config.roi_map, chromo)
            fc_roi_df = compute_fc_roi(series, config.roi_map, chromo, signals=signals)
            if not fc_roi_df.empty:
                fc_roi_path = _path(_roi({**chromo_entities, "aggregation": "roi"}), "pearson")
                write_tsv(fc_roi_df, fc_roi_path, index=True, index_label="roi")
                # the members, not the map: an ROI correlation averages them into a
                # signal that exists nowhere else, and which ones survived is a property
                # of this run, so without them the number cannot be reproduced
                _sidecar(fc_roi_path, "fc_roi", chromophore=chromo, roi_channels=members)
                logger.info("sub-%s | fc_roi (%s) -> %s", config.subject, chromo, fc_roi_path)

                fcroiz_path = _path(_roi({**chromo_entities, "aggregation": "roi"}), "fisherz")
                write_tsv(fisher_z(fc_roi_df), fcroiz_path, index=True, index_label="roi")
                _sidecar(fcroiz_path, "fisher_z", chromophore=chromo, roi_channels=members)
                logger.info("sub-%s | fc_roiz (%s) -> %s", config.subject, chromo, fcroiz_path)

                fc_roi[chromo] = fc_roi_df

            # seed map: one side averaged, so it is a third product rather than a view of the
            # two above. Cells for a seed's own channels are NaN, not zero.
            fc_seed_df = compute_fc_seed(series, config.roi_map, chromo, signals=signals,
                                         members=seeds)
            for ch in (c for c in empty if c in fc_seed_df.columns):
                fc_seed_df[ch] = np.nan
            if not fc_seed_df.empty:
                fcseed_path = _path(_roi({**chromo_entities, "aggregation": "seed"}), "pearson")
                write_tsv(fc_seed_df, fcseed_path, index=True, index_label="roi")
                # the channels each seed was actually built from, which is the requested map
                # minus whatever was rejected; without it a reader cannot tell why a cell
                # inside a listed ROI holds a value instead of being blank
                _sidecar(fcseed_path, "fc_seed", chromophore=chromo, seed_channels=members)
                logger.info("sub-%s | fc_seed (%s) -> %s", config.subject, chromo, fcseed_path)

                fcseedz_path = _path(_roi({**chromo_entities, "aggregation": "seed"}), "fisherz")
                write_tsv(fisher_z(fc_seed_df), fcseedz_path, index=True, index_label="roi")
                _sidecar(fcseedz_path, "fisher_z", chromophore=chromo)
                logger.info("sub-%s | fc_seedz (%s) -> %s", config.subject, chromo, fcseedz_path)

                fc_seed[chromo] = fc_seed_df

        fc[chromo] = fc_df

    return fc, fc_roi, fc_seed


def _write_rest_derivatives(
    raw_resid: mne.io.Raw,
    raw_resid_bb: mne.io.Raw | None,
    config: PostConfig,
    output_dir: Path,
    rec: Recorder,
    source_entities: dict[str, str] | None = None,
    known_spans: "list[dict] | None" = None,
) -> tuple:
    """Write ALFF/fALFF and everything :func:`_write_fc_derivatives` writes.

    Returns (alff_df | None, fc, fc_roi, fc_seed).
    ALFF/fALFF use the broadband residual (raw_resid_bb); every FC product uses the
    bandpassed one.
    """
    from nirspipe.pipeline.restingstate import _roi_members, compute_alff, compute_alff_roi

    entities = carry_entities(source_entities)

    alff_df = None
    kept = _kept_time(raw_resid_bb, config, "ALFF", rec) if raw_resid_bb is not None else None
    if kept is not None:
        series = with_hbt(raw_resid_bb, config.chromophores)
        # handed in rather than blanked afterwards: a channel fitted against a copy of
        # itself must also stay out of the mALFF/zALFF reference mean
        empty = sole_regressor_channels(raw_resid_bb, config.short_channel,
                                        separation_bands(config))
        empty += hbt_names(empty) if "hbt" in config.chromophores else []
        alff_df = compute_alff(series, low_pass=config.low_pass,
                               high_pass=config.high_pass, exclude=empty)
        alff_df = alff_df[alff_df["channel"].map(chroma_of).isin(config.chromophores)]
        alff_df = alff_df.reset_index(drop=True)
        alff_path = build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities={**entities, "statistic": "alff"}, suffix="nirsmap", extension=".tsv",
        )
        write_tsv(alff_df, alff_path)
        spans = excluded_spans(raw_resid_bb, known=known_spans or ())
        write_step_sidecar(alff_path, "alff", rec.path_of(raw_resid_bb),
                       list(series.info["bads"]), excluded_spans=spans,
                       low_pass=config.low_pass, high_pass=config.high_pass,
                       chromophores=list(config.chromophores), **kept)
        logger.info("sub-%s | alff -> %s", config.subject, alff_path)

        if config.roi_map:
            alff_roi_df = compute_alff_roi(alff_df, series, config.roi_map, config.chromophores)
            if not alff_roi_df.empty:
                alff_roi_path = build_output_path(
                    output_dir=output_dir, subject=config.subject, session=config.session,
                    entities={**entities, "segmentation": config.roi_map_name,
                              "aggregation": "roi", "statistic": "alff"},
                    suffix="nirsmap", extension=".tsv",
                )
                write_tsv(alff_roi_df, alff_roi_path)
                write_step_sidecar(alff_roi_path, "alff_roi", rec.path_of(raw_resid_bb),
                               list(series.info["bads"]), excluded_spans=spans,
                               low_pass=config.low_pass, high_pass=config.high_pass,
                               roi_channels={c: _roi_members(series, config.roi_map, c)
                                             for c in config.chromophores},
                               chromophores=list(config.chromophores), **kept)
                logger.info("sub-%s | alff_roi -> %s", config.subject, alff_roi_path)
    elif raw_resid_bb is None:
        logger.warning("sub-%s | skipping ALFF: --high-pass and --low-pass required", config.subject)

    fc, fc_roi, fc_seed = _write_fc_derivatives(
        raw_resid, config, output_dir, rec, source_entities=source_entities, known_spans=known_spans)
    return alff_df, fc, fc_roi, fc_seed


# the parameters that make two runs off one recording two different analyses rather than
# one repeated. Named here so the check below cannot drift from what the sidecar records.
_ANALYSIS_KEYS = ("high_pass", "low_pass", "filter_method", "filter_order", "resample_sfreq",
                  "drift_model", "drift_high_pass", "drift_order")


def _warn_if_replacing_another_analysis(out_path: Path, parameters: dict, subject: str) -> None:
    """Say so when this write replaces an output made with a different passband.

    One output directory holds one analysis. A recording serving both a rest band and a task
    band is two runs, and pointing both at the same output_dir replaces the first silently:
    not this file alone but the residual, the design matrix, the quality record and the
    report, none of which carry the band in their name. Two output directories are the
    answer, and this is what makes the collision visible when they are not used.

    A warning rather than a refusal: rerunning after changing a parameter is the normal way
    to work, and only the caller knows whether the earlier output was still wanted.
    """
    sidecar = out_path.with_suffix(".json")
    if not sidecar.exists():
        return
    old = read_json(sidecar).get("parameters") or {}
    changed = {k: (old.get(k), parameters.get(k))
               for k in _ANALYSIS_KEYS if k in old and k in parameters and old[k] != parameters[k]}
    if not changed:
        return
    logger.warning(
        "sub-%s | %s already holds an analysis with %s; replacing it. Everything that run "
        "wrote is being overwritten, including its residual, quality record and report, so "
        "use a second output directory to keep both",
        subject, out_path.name,
        ", ".join(f"{k} {was} -> {now}" for k, (was, now) in changed.items()),
    )


def _kept_time(raw: mne.io.Raw, config: PostConfig, what: str, rec: Recorder) -> dict | None:
    """``{kept_s, min_time}`` for the sidecars when ``what`` may be computed on ``raw``.

    None when the time outside every ``BAD_`` span is under ``config.min_time`` or under the
    three frames a correlation needs; the skip is logged and noted in the sidecar of the
    file ``what`` would have been computed from, so its absence is never silent.
    """
    keep = kept_frames(raw)
    kept_s = float(keep.sum() / raw.info["sfreq"])
    total_s = float(len(keep) / raw.info["sfreq"])
    if kept_s >= config.min_time and keep.sum() >= 3:
        logger.info("sub-%s | %s on %.1f s kept of %.1f s", config.subject, what, kept_s, total_s)
        return {"kept_s": kept_s, "min_time": config.min_time}
    reason = (f"{kept_s:.1f} s lie outside its BAD_ spans, under --min-time {config.min_time:g} s"
              if kept_s < config.min_time else
              f"{int(keep.sum())} frames lie outside its BAD_ spans, under the 3 a correlation needs")
    logger.warning("sub-%s | %s not computed: %s", config.subject, what, reason)
    source = rec.path_of(raw)
    # only a file this run wrote; the input belongs to prep
    if source is not None and source in {e["path"] for e in rec.entries}:
        sidecar_path = Path(source).with_suffix(".json")
        sidecar = read_json(sidecar_path)
        sidecar.setdefault("skipped_outputs", []).append(
            {"output": what, "kept_s": kept_s, "min_time": config.min_time, "reason": reason})
        sidecar_path.write_text(json.dumps(sidecar, indent=2))
    return None


def _known_spans(source_path: Path | None) -> list[dict]:
    """The input stage's ``excluded_spans``, whose recorded sources the post stages keep."""
    if source_path is None:
        return []
    sidecar = read_json(Path(source_path).with_suffix(".json"))
    if "excluded_spans" not in sidecar:
        raise StageError(f"{Path(source_path).name} records no excluded_spans; this "
                         f"derivative tree predates them, rerun nirspipe")
    return sidecar["excluded_spans"]


# the bandpass and resample stamps' names for high_pass, low_pass and resample_sfreq
_RESTATED = ("l_freq", "h_freq", "sfreq")


def _write_step_snirf(haemo: mne.io.Raw, config: PostConfig, output_dir: Path, desc: str, rec: Recorder, source_entities: dict[str, str] | None = None, known_spans: list[dict] | None = None, description: str | None = None, inputs: "list[Path]" = ()) -> Path:
    entities = carry_entities(source_entities)
    entities["desc"] = desc
    lin = lineage_of(haemo)
    if lin is None or lin.stage != desc:
        raise StageError(f"_write_step_snirf({desc!r}) got an object stamped {stage_of(haemo)!r}")
    out_path = build_output_path(
        output_dir=output_dir,
        subject=config.subject,
        session=config.session,
        entities=entities,
        suffix="nirs",
        extension=".snirf",
    )
    # on every stage, not just desc-filtered: a later stage is what downstream tools read,
    # and it is the one that has to name its own passband
    from nirspipe.qc.metrics._helpers import bands_to_record

    parameters = {
        "high_pass": config.high_pass,
        "low_pass": config.low_pass,
        "filter_method": config.filter_method,
        "filter_order": config.filter_order if config.filter_method == "iir" else None,
        "censor_fill": config.censor_fill,
        # state from desc-resampled on; desc-filtered comes before the resampling
        **({} if desc == "filtered" else {"resample_sfreq": config.resample_sfreq}),
        # which channels the run treated as short, so a reader of a derivatives tree can
        # reproduce the split. Without it neither the short-channel regressors nor the
        # blanking of a channel fitted against itself can be recovered from the file.
        **({} if desc in ("filtered", "resampled") else bands_to_record(separation_bands(config))),
        **{k: v for k, v in (lin.params if lin else {}).items() if k not in _RESTATED},
    }
    # before the write, while the previous run's sidecar is still the one on disk
    _warn_if_replacing_another_analysis(out_path, parameters, config.subject)
    write_snirf(haemo, out_path)
    sources = rec.sources_of(haemo)
    linked, external = split_sources(inputs, out_path)
    write_sidecar_json(out_path, {
        **({"Description": description} if description else {}),
        **nirs_required_fields(haemo, read_json(Path(sources[0]).with_suffix(".json")) if sources else {}),
        "pipeline_version": __version__,
        "step": lin.step if lin else None,
        "Sources": sources + linked,
        **({"external_sources": external} if external else {}),
        "parameters": parameters,
        "data": data_state(haemo),
        # read back by read_snirf: SNIRF itself cannot carry the marks
        "bad_channels": list(haemo.info["bads"]),
        "excluded_spans": excluded_spans(haemo, known=known_spans or ()),
    })
    logger.info("sub-%s | %s snirf -> %s", config.subject, desc, out_path)
    return rec.written(out_path, haemo)
