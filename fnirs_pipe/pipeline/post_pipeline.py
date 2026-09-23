"""Postprocessing pipeline — mode-driven, parameter-configured.

Parameters come from PostConfig, which can be populated from CLI flags or a TOML file.
All steps within a mode are still individually controllable via PostConfig fields.

"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mne
import mne.io
import numpy as np
import pandas as pd

from fnirs_pipe.io.auxiliary import find_aux_table
from fnirs_pipe.io.tables import read_table
from fnirs_pipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    bandpass_filter,
    filter_description,
    resample,
)
from fnirs_pipe.pipeline.glm import run_glm_pipeline, sole_regressor_channels
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.utils.lineage import Recorder, carried_params, lineage_of, stage_of, stamp
from fnirs_pipe.qc.metrics._helpers import separation_bands
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe import __version__

logger = get_logger("post.pipeline")

Mode = Literal["denoise", "glm", "rest"]


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

    roi_map: dict | None = None
    # names the seg- entity on every ROI output, so one tree can hold two ROI
    # definitions. Defaults to the stem of the file the map was read from
    roi_map_name: str = "custom"

    # resample
    resample_sfreq: float | None = None

    # GLM (glm mode)
    stim_dur:        float | None          = None
    hrf_model:       str | None            = None
    # every mode fits it now, so the library carries the default too rather than leaving a
    # direct caller to pass None into nilearn. "auto" is an AR order of 4x the sampling rate
    noise_model:     str                   = "auto"
    drift_model:     str | None            = None
    drift_high_pass: float | None          = None
    drift_order:     int | None            = None
    fir_delays:      tuple[int, ...] | None = None
    short_channel:   bool | str | None     = None
    aux:             bool                  = False
    aux_channels:    list[str] | None      = None
    events_path:     str | None            = None
    contrast_def:    dict[str, Any] | None = None
    fc:              bool                  = False

    # separation bands, in metres; see PrepConfig
    short_max_dist: float | None = None
    long_min_dist: float | None = None
    long_max_dist: float | None = None

    combine_runs: bool | None = None

    def __post_init__(self) -> None:
        # nilearn multiplies the cutoff by the frame times, so a missing one dies deep
        # inside it as "unsupported operand type(s) for *: 'NoneType' and 'float'"
        if self.drift_model == "cosine" and self.drift_high_pass is None:
            raise ValueError(
                "--drift-model cosine needs --drift-high-pass: the cosine drift basis is "
                "defined by its cutoff frequency, and there is no sensible default"
            )


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

    How much it costs depends on block length against the cutoff: a design whose blocks sit
    well inside the passband loses a couple of percent, while one whose fundamental falls
    below the cutoff can lose most of the effect. A warning rather than an error for that
    reason, since only the caller knows their design.
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
            "--noise-model ols to reproduce a tree written before every mode honoured this "
            "flag")
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
        if str(desc).lower().startswith(("bad", "edge")):
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
    frequency at which the condition repeats. The usual choice is half that frequency,
    which leaves the basis an octave of room before it reaches the task.
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


def run_post(
    raw_haemo: mne.io.Raw,
    config: PostConfig,
    output_dir: Path,
    mode: Mode = "denoise",
    source_entities: dict[str, str] | None = None,
    source_path: Path | None = None,
) -> tuple:
    """Run post-processing pipeline.

    Returns (result, glm_est, design_matrix, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi).
    glm_est / design_matrix are None only when no regression ran; denoise and rest return
    them too, fitted on empty events, so the design holds confound columns and no condition.
    alff_df is None outside rest mode. The FC products are written by rest mode, and by glm
    and denoise mode under ``config.fc``; fc_df holds the HbO matrix.
    fc_seed and fc_roi are {chromophore: frame}, both empty without --roi-mapping.
    """

    rec = Recorder()
    if source_path is not None:
        rec.register_input(source_path, raw_haemo)

    result = raw_haemo.copy()

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
        _write_step_snirf(result, config, output_dir, desc="filtered", rec=rec, source_entities=source_entities)

    if config.resample_sfreq is not None:
        logger.info("sub-%s | resample -> %.1f Hz", config.subject, config.resample_sfreq)
        result = resample(result, config.resample_sfreq)
        _write_step_snirf(result, config, output_dir, desc="resampled", rec=rec, source_entities=source_entities)

    # SQM is not computed here either; the record is assembled from disk after this
    # pipeline returns, which is what lets one writer own the whole file.

    glm_est = dm = alff_df = fc_df = fc_hbr_df = None
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
        _warn_unmatched_design_band(config)
        _warn_lowpass_breaks_whitening(config, mode)
        # the same table the design will be built from, so the warning cannot name a
        # condition the model does not carry
        design_events = read_table(config.events_path) if config.events_path else None
        _warn_drift_absorbs_task(config, result, design_events)
        _, glm_est, dm, raw_resid = run_glm_pipeline(
            result,
            stim_dur=config.stim_dur,
            hrf_model=config.hrf_model,
            noise_model=config.noise_model,
            drift_model=config.drift_model,
            high_pass=config.drift_high_pass,
            drift_order=config.drift_order,
            fir_delays=config.fir_delays,
            short_channel=config.short_channel,
            sep_bands=separation_bands(config),
            **aux_kwargs,
            events_path=config.events_path,
            contrast_def=config.contrast_def,
            output_dir=str(output_dir / f"sub-{config.subject}" / "nirs"),
            source_path=rec.path_of(result),
        )
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities)

        # FC on the task residual, which is what makes it a connectivity measure rather
        # than a map of who responded to the same stimulus: the task is in the design
        # matrix, so what correlates here is what the model did not explain.
        if config.fc:
            _warn_fc_without_bandpass(config)
            fc_df, fc_hbr_df, fc_roi, fc_seed = _write_fc_derivatives(
                raw_resid, config, output_dir, rec, source_entities=source_entities)

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
            sep_bands=separation_bands(config),
            **aux_kwargs,
            events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),
        )
        _, glm_est, dm, raw_resid = run_glm_pipeline(
            result,
            output_dir=str(output_dir / f"sub-{config.subject}" / "nirs"),
            source_path=rec.path_of(result),
            **rest_glm_kwargs,
        )
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities)

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
                result_bb = raw_haemo.copy()
                if config.resample_sfreq is not None:
                    result_bb = resample(result_bb, config.resample_sfreq)
                # its input skipped the bandpass, so band-matching the aux to one would
                # put the regressors in a narrower band than the data they explain
                _, _, _, raw_resid_bb = run_glm_pipeline(
                    result_bb, **{**rest_glm_kwargs, "data_band": None})
                # Same regression, un-bandpassed input. Re-stamp so it stops sharing the "errts"
                # stage with the bandpassed residual, whose file it would otherwise be credited to.
                # carry the regression's own parameters over; a bare re-stamp overwrites them
                stamp(raw_resid_bb, stage="errtsbroad", step="glm_residuals_broadband",
                      source=raw_haemo,
                      **{**carried_params(raw_resid_bb),
                         "resample_sfreq": config.resample_sfreq})
                _write_step_snirf(raw_resid_bb, config, output_dir, desc="errtsbroad",
                                  rec=rec, source_entities=source_entities)

        alff_df, fc_df, fc_hbr_df, fc_roi, fc_seed = _write_rest_derivatives(
            raw_resid, raw_resid_bb, config, output_dir, rec, source_entities=source_entities)

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
                sep_bands=separation_bands(config),
                **aux_kwargs,
                events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),
                output_dir=str(output_dir / f"sub-{config.subject}" / "nirs"),
                source_path=rec.path_of(result),
            )
            _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities)

        # No ALFF, for glm's reason: the source is bandpassed and fALFF's denominator spans
        # the full spectrum, so the number would be ~1 by construction.
        if config.fc:
            _warn_fc_without_bandpass(config)
            fc_df, fc_hbr_df, fc_roi, fc_seed = _write_fc_derivatives(
                raw_resid if raw_resid is not None else result,
                config, output_dir, rec, source_entities=source_entities)

    # GCOR around the confound regression is not measured here. Both of its stages are
    # written to disk, and the SQM record measures each one per channel set; computing it a
    # second time in memory gave the report a number over every channel to print beside
    # rows that were long-channel, which reversed the direction the regression appeared to
    # move HbR global correlation in. The report reads `filtered` -> `errts` instead.
    return result, glm_est, dm, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi

def _deriv_sidecar(path: Path, step: str, source: str | None, bads: list[str], **params) -> None:
    from fnirs_pipe.io.derivatives import write_sidecar_json

    write_sidecar_json(path, {
        "pipeline_version": __version__,
        "step": step,
        "Sources": [source] if source else [],
        "parameters": params,
        "bad_channels": bads,
    })


def _write_fc_derivatives(
    raw_resid: mne.io.Raw,
    config: PostConfig,
    output_dir: Path,
    rec: Recorder,
    source_entities: dict[str, str] | None = None,
) -> tuple:
    """Write the FC, ROI FC and seed-map TSVs for one residual.

    Returns (fc_hbo_df, fc_hbr_df, fc_roi, fc_seed). The last two are
    {chromophore: frame} and stay empty without a roi_map.

    All three modes land here, and the only thing that differs is what arrives: rest
    regresses confounds alone, glm regresses the task as well, and denoise passes either its
    own confound residual or, when no regression ran, the bandpassed data itself.
    Correlating a task residual is what makes the result connectivity rather than a map of
    who responded to the same stimulus, so the distinction lives in the caller, not here.
    """
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities
    from fnirs_pipe.pipeline.restingstate import (
        _roi_members, compute_fc, compute_fc_roi, compute_fc_seed, fisher_z,
    )

    entities = carry_entities(source_entities)

    # the FC matrices keep their bad rows and columns so the shape stays predictable;
    # the sidecar names them so a consumer can drop or ignore them
    bads = list(raw_resid.info["bads"])
    # a channel that was the whole short-channel regressor was fitted against a copy of
    # itself, so its residual is zero and every correlation it takes part in is noise
    empty = sole_regressor_channels(raw_resid, config.short_channel,
                                    separation_bands(config))
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

    def _sidecar(path: Path, step: str, **params) -> None:
        _deriv_sidecar(path, step, src_bp, bads, **params)

    fc_hbo_df = fc_hbr_df = None
    fc_roi: dict[str, pd.DataFrame] = {}
    fc_seed: dict[str, pd.DataFrame] = {}

    # FC per chromophore: HbO and HbR anti-correlate, so they never share a matrix. Both are
    # written and both reach the report.
    for chromo in ("hbo", "hbr"):
        fc_df = compute_fc(raw_resid, chromo)
        if fc_df.empty:
            continue
        for ch in (c for c in empty if c in fc_df.index):
            fc_df.loc[ch, :] = np.nan
            fc_df.loc[:, ch] = np.nan
        chromo_entities = {"chromophore": chromo}

        fc_path = _path(chromo_entities, "pearson")
        fc_df.to_csv(fc_path, sep="	", index_label="channel")
        _sidecar(fc_path, "fc", chromophore=chromo)
        logger.info("sub-%s | fc (%s) -> %s", config.subject, chromo, fc_path)

        fcz_path = _path(chromo_entities, "fisherz")
        fisher_z(fc_df).to_csv(fcz_path, sep="	", index_label="channel")
        _sidecar(fcz_path, "fisher_z", chromophore=chromo)
        logger.info("sub-%s | fcz (%s) -> %s", config.subject, chromo, fcz_path)

        if config.roi_map:
            fc_roi_df = compute_fc_roi(raw_resid, config.roi_map, chromo)
            if not fc_roi_df.empty:
                fc_roi_path = _path(_roi({**chromo_entities, "aggregation": "roi"}), "pearson")
                fc_roi_df.to_csv(fc_roi_path, sep="	", index_label="roi")
                # the members, not the map: an ROI correlation averages them into a
                # signal that exists nowhere else, and which ones survived is a property
                # of this run, so without them the number cannot be reproduced
                _sidecar(fc_roi_path, "fc_roi", chromophore=chromo,
                         roi_channels=_roi_members(raw_resid, config.roi_map, chromo))
                logger.info("sub-%s | fc_roi (%s) -> %s", config.subject, chromo, fc_roi_path)

                fcroiz_path = _path(_roi({**chromo_entities, "aggregation": "roi"}), "fisherz")
                fisher_z(fc_roi_df).to_csv(fcroiz_path, sep="	", index_label="roi")
                _sidecar(fcroiz_path, "fisher_z", chromophore=chromo,
                         roi_channels=_roi_members(raw_resid, config.roi_map, chromo))
                logger.info("sub-%s | fc_roiz (%s) -> %s", config.subject, chromo, fcroiz_path)

                fc_roi[chromo] = fc_roi_df

            # seed map: one side averaged, so it is a third product rather than a view of the
            # two above. Cells for a seed's own channels are NaN, not zero.
            fc_seed_df = compute_fc_seed(raw_resid, config.roi_map, chromo)
            if not fc_seed_df.empty:
                fcseed_path = _path(_roi({**chromo_entities, "aggregation": "seed"}), "pearson")
                fc_seed_df.to_csv(fcseed_path, sep="	", index_label="roi")
                # the channels each seed was actually built from, which is the requested map
                # minus whatever was rejected; without it a reader cannot tell why a cell
                # inside a listed ROI holds a value instead of being blank
                _sidecar(fcseed_path, "fc_seed", chromophore=chromo,
                         seed_channels=_roi_members(raw_resid, config.roi_map, chromo))
                logger.info("sub-%s | fc_seed (%s) -> %s", config.subject, chromo, fcseed_path)

                fcseedz_path = _path(_roi({**chromo_entities, "aggregation": "seed"}), "fisherz")
                fisher_z(fc_seed_df).to_csv(fcseedz_path, sep="	", index_label="roi")
                _sidecar(fcseedz_path, "fisher_z", chromophore=chromo)
                logger.info("sub-%s | fc_seedz (%s) -> %s", config.subject, chromo, fcseedz_path)

                fc_seed[chromo] = fc_seed_df

        if chromo == "hbo":
            fc_hbo_df = fc_df
        else:
            fc_hbr_df = fc_df

    return fc_hbo_df, fc_hbr_df, fc_roi, fc_seed


def _write_rest_derivatives(
    raw_resid: mne.io.Raw,
    raw_resid_bb: mne.io.Raw | None,
    config: PostConfig,
    output_dir: Path,
    rec: Recorder,
    source_entities: dict[str, str] | None = None,
) -> tuple:
    """Write ALFF/fALFF and everything :func:`_write_fc_derivatives` writes.

    Returns (alff_df | None, fc_hbo_df, fc_hbr_df, fc_roi, fc_seed).
    ALFF/fALFF use the broadband residual (raw_resid_bb); every FC product uses the
    bandpassed one.
    """
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities
    from fnirs_pipe.pipeline.restingstate import _roi_members, compute_alff, compute_alff_roi

    entities = carry_entities(source_entities)

    alff_df = None
    if raw_resid_bb is not None:
        # handed in rather than blanked afterwards: a channel fitted against a copy of
        # itself must also stay out of the mALFF/zALFF reference, and doing that here and
        # the blanking there is what let a numerical zero into the reference mean
        empty = sole_regressor_channels(raw_resid_bb, config.short_channel,
                                        separation_bands(config))
        alff_df = compute_alff(raw_resid_bb, low_pass=config.low_pass,
                               high_pass=config.high_pass, exclude=empty)
        alff_path = build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities={**entities, "statistic": "alff"}, suffix="nirsmap", extension=".tsv",
        )
        alff_df.to_csv(alff_path, sep="	", index=False)
        _deriv_sidecar(alff_path, "alff", rec.path_of(raw_resid_bb),
                       list(raw_resid.info["bads"]),
                       low_pass=config.low_pass, high_pass=config.high_pass)
        logger.info("sub-%s | alff -> %s", config.subject, alff_path)

        if config.roi_map:
            alff_roi_df = compute_alff_roi(alff_df, raw_resid_bb, config.roi_map)
            if not alff_roi_df.empty:
                alff_roi_path = build_output_path(
                    output_dir=output_dir, subject=config.subject, session=config.session,
                    entities={**entities, "segmentation": config.roi_map_name,
                              "aggregation": "roi", "statistic": "alff"},
                    suffix="nirsmap", extension=".tsv",
                )
                alff_roi_df.to_csv(alff_roi_path, sep="	", index=False)
                _deriv_sidecar(alff_roi_path, "alff_roi", rec.path_of(raw_resid_bb),
                               list(raw_resid.info["bads"]),
                               low_pass=config.low_pass, high_pass=config.high_pass,
                               roi_channels={c: _roi_members(raw_resid_bb, config.roi_map, c)
                                             for c in ("hbo", "hbr")})
                logger.info("sub-%s | alff_roi -> %s", config.subject, alff_roi_path)
    else:
        logger.warning("sub-%s | skipping ALFF: --high-pass and --low-pass required", config.subject)

    fc_hbo_df, fc_hbr_df, fc_roi, fc_seed = _write_fc_derivatives(
        raw_resid, config, output_dir, rec, source_entities=source_entities)
    return alff_df, fc_hbo_df, fc_hbr_df, fc_roi, fc_seed


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
    try:
        old = (json.loads(sidecar.read_text(encoding="utf-8")).get("parameters") or {})
    except (OSError, json.JSONDecodeError):
        return
    changed = {k: (old.get(k), parameters.get(k))
               for k in _ANALYSIS_KEYS if k in old and old.get(k) != parameters.get(k)}
    if not changed:
        return
    logger.warning(
        "sub-%s | %s already holds an analysis with %s; replacing it. Everything that run "
        "wrote is being overwritten, including its residual, quality record and report, so "
        "use a second output directory to keep both",
        subject, out_path.name,
        ", ".join(f"{k} {was} -> {now}" for k, (was, now) in changed.items()),
    )


def _write_step_snirf(haemo: mne.io.Raw, config: PostConfig, output_dir: Path, desc: str, rec: Recorder, source_entities: dict[str, str] | None = None) -> Path:
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities, data_state, write_sidecar_json
    from fnirs_pipe.io.snirf import write_snirf

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
    from fnirs_pipe.qc.metrics._helpers import bands_to_record

    parameters = {
        "high_pass": config.high_pass,
        "low_pass": config.low_pass,
        "filter_method": config.filter_method,
        "filter_order": config.filter_order if config.filter_method == "iir" else None,
        "resample_sfreq": config.resample_sfreq,
        # which channels the run treated as short, so a reader of a derivatives tree can
        # reproduce the split. Without it neither the short-channel regressors nor the
        # blanking of a channel fitted against itself can be recovered from the file.
        **bands_to_record(separation_bands(config)),
        **(lin.params if lin else {}),
    }
    # before the write, while the previous run's sidecar is still the one on disk
    _warn_if_replacing_another_analysis(out_path, parameters, config.subject)
    write_snirf(haemo, out_path)
    write_sidecar_json(out_path, {
        "pipeline_version": __version__,
        "step": lin.step if lin else None,
        "Sources": rec.sources_of(haemo),
        "parameters": parameters,
        "data": data_state(haemo),
        # read back by read_snirf: SNIRF itself cannot carry the marks
        "bad_channels": list(haemo.info["bads"]),
    })
    logger.info("sub-%s | %s snirf -> %s", config.subject, desc, out_path)
    return rec.written(out_path, haemo)
