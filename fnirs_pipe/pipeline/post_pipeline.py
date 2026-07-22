"""Postprocessing pipeline — mode-driven, parameter-configured.

Parameters come from PostConfig, which can be populated from CLI flags or a TOML file.
All steps within a mode are still individually controllable via PostConfig fields.

With dry_run=True the pipeline writes an inspectable Python script instead of executing.

"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mne
import mne.io
import pandas as pd

from fnirs_pipe.pipeline.denoise import bandpass_filter, resample
from fnirs_pipe.pipeline.glm import run_glm_pipeline
from fnirs_pipe.utils.logging import get_logger

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
    dry_run: bool = False

    # bandpass filter
    high_pass: float | None = None
    low_pass:  float | None = None

    roi_map: dict | None = None

    # resample
    resample_sfreq: float | None = None

    # GLM (glm mode)
    stim_dur:        float | None          = None
    hrf_model:       str | None            = None
    noise_model:     str | None            = None
    drift_model:     str | None            = None
    drift_high_pass: float | None          = None
    drift_order:     int | None            = None
    fir_delays:      tuple[int, ...] | None = None
    short_channel:   bool | str | None     = None
    events_path:     str | None            = None
    contrast_def:    dict[str, Any] | None = None

    combine_runs: bool | None = None



def run_post(
    raw_haemo: mne.io.Raw,
    config: PostConfig,
    output_dir: Path,
    mode: Mode = "denoise",
    source_entities: dict[str, str] | None = None,
) -> tuple:
    """Run post-processing pipeline.

    Returns (result, glm_est, design_matrix, alff_df, fc_df, gcor_reg).
    glm_est / design_matrix are None for non-GLM modes.
    alff_df / fc_df are None for non-rest modes.
    gcor_reg (pre/post short-channel regression GCOR) is None unless short_channel ran.
    """

    result = raw_haemo.copy()
    last_snirf_path: Path | None = None

    if config.high_pass is not None or config.low_pass is not None:
        logger.info("sub-%s | bandpass: l_freq=%s h_freq=%s", config.subject, config.high_pass, config.low_pass)
        result = bandpass_filter(result, l_freq=config.high_pass, h_freq=config.low_pass)
        if mode in ("denoise", "glm"):
            last_snirf_path = _write_step_snirf(result, config, output_dir, desc="filtered", source_entities=source_entities)

    if config.resample_sfreq is not None:
        logger.info("sub-%s | resample → %.1f Hz", config.subject, config.resample_sfreq)
        result = resample(result, config.resample_sfreq)
        if mode in ("denoise", "glm"):
            last_snirf_path = _write_step_snirf(result, config, output_dir, desc="resampled", source_entities=source_entities)

    haemo_sqm = None
    if last_snirf_path is not None:
        try:
            from fnirs_pipe.qc.quantitative_metrics import compute_haemo_sqm, save_sqm_toml
            haemo_sqm = compute_haemo_sqm(
                result, config.cardiac_l_freq, config.cardiac_h_freq,
                config.resp_l_freq, config.resp_h_freq)
            save_sqm_toml(haemo_sqm, config.subject, last_snirf_path.parent)
        except Exception:
            logger.warning("sub-%s | haemo SQM failed", config.subject, exc_info=True)

    glm_est = dm = alff_df = fc_df = None
    raw_resid = None  # set by the glm/rest branches
    if mode == "glm":
        missing = [f for f in ("hrf_model", "noise_model", "drift_model") if getattr(config, f) is None]
        if missing:
            raise ValueError(f"GLM mode requires: {', '.join('--' + f.replace('_', '-') for f in missing)}")
        if config.events_path is not None and config.stim_dur is not None:
            raise ValueError("--events-path and --stim-dur are mutually exclusive")
        logger.info("sub-%s | GLM (%s / %s)", config.subject, config.hrf_model, config.noise_model)
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
            events_path=config.events_path,
            contrast_def=config.contrast_def,
            output_dir=str(output_dir / f"sub-{config.subject}" / "nirs"),
        )
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", source_entities=source_entities)
        # Durbin-Watson (GLM residual autocorrelation) merged into the final SQM toml
        if haemo_sqm is not None and last_snirf_path is not None:
            try:
                from fnirs_pipe.qc.quantitative_metrics import compute_glm_sqm, save_sqm_toml
                haemo_sqm.update(compute_glm_sqm(raw_resid.get_data()))
                save_sqm_toml(haemo_sqm, config.subject, last_snirf_path.parent)
            except Exception:
                logger.warning("sub-%s | GLM SQM failed", config.subject, exc_info=True)

    elif mode == "rest":
        if config.drift_model is None:
            raise ValueError("rest mode requires --drift-model")
        logger.info("sub-%s | rest confound regression", config.subject)
        _, glm_est, dm, raw_resid = run_glm_pipeline(
            result,
            stim_dur=None,
            hrf_model="spm",
            noise_model="ols",
            drift_model=config.drift_model,
            high_pass=config.drift_high_pass,
            drift_order=config.drift_order,
            fir_delays=None,
            short_channel=config.short_channel,
            events=pd.DataFrame({"trial_type": [], "onset": [], "duration": []}),
            output_dir=str(output_dir / f"sub-{config.subject}" / "nirs"),
        )
        errts_path = _write_step_snirf(raw_resid, config, output_dir, desc="errts", source_entities=source_entities)
        try:
            from fnirs_pipe.qc.quantitative_metrics import compute_glm_sqm, save_sqm_toml
            save_sqm_toml(compute_glm_sqm(raw_resid.get_data()), config.subject, errts_path.parent)
        except Exception:
            logger.warning("sub-%s | rest GLM SQM failed", config.subject, exc_info=True)
        alff_df, fc_df = _write_rest_derivatives(raw_resid, config, output_dir, source_entities=source_entities)

    # GCOR around the short-channel regression (the fNIRS GSR analog): pre-regression vs
    # residuals. Expected to drop if the regression removed global/systemic signal.
    gcor_reg = None
    if raw_resid is not None and config.short_channel:
        try:
            from fnirs_pipe.qc.quantitative_metrics import gcor_metrics
            pre, post = gcor_metrics(result), gcor_metrics(raw_resid)
            gcor_reg = {
                "gcor_hbo_prereg": pre["gcor_hbo"], "gcor_hbr_prereg": pre["gcor_hbr"],
                "gcor_hbo_postreg": post["gcor_hbo"], "gcor_hbr_postreg": post["gcor_hbr"],
            }
        except Exception:
            logger.warning("sub-%s | regression GCOR failed", config.subject, exc_info=True)

    return result, glm_est, dm, alff_df, fc_df, gcor_reg

def _write_rest_derivatives(
    raw_resid: mne.io.Raw,
    config: PostConfig,
    output_dir: Path,
    source_entities: dict[str, str] | None = None,
) -> tuple:
    """Write ALFF/fALFF and FC TSVs. Returns (alff_df | None, fc_df)."""
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities
    from fnirs_pipe.pipeline.restingstate import compute_alff, compute_fc, compute_fc_roi, fisher_z

    entities = carry_entities(source_entities)

    alff_df = None
    if config.low_pass is not None and config.high_pass is not None:
        alff_df = compute_alff(raw_resid, low_pass=config.low_pass, high_pass=config.high_pass)
        alff_path = build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities=entities, suffix="alff", extension=".tsv",
        )
        alff_df.to_csv(alff_path, sep="\t", index=False)
        logger.info("sub-%s | alff → %s", config.subject, alff_path)
    else:
        logger.warning("sub-%s | skipping ALFF: --high-pass and --low-pass required", config.subject)

    fc_df = compute_fc(raw_resid)
    fc_path = build_output_path(
        output_dir=output_dir, subject=config.subject, session=config.session,
        entities=entities, suffix="fc", extension=".tsv",
    )
    fc_df.to_csv(fc_path, sep="\t", index_label="channel")
    logger.info("sub-%s | fc → %s", config.subject, fc_path)

    fcz_path = build_output_path(
        output_dir=output_dir, subject=config.subject, session=config.session,
        entities=entities, suffix="fcz", extension=".tsv",
    )
    fisher_z(fc_df).to_csv(fcz_path, sep="\t", index_label="channel")
    logger.info("sub-%s | fcz → %s", config.subject, fcz_path)

    if config.roi_map:
        fc_roi_df = compute_fc_roi(raw_resid, config.roi_map)
        if not fc_roi_df.empty:
            fc_roi_path = build_output_path(
                output_dir=output_dir, subject=config.subject, session=config.session,
                entities=entities, suffix="fcroi", extension=".tsv",
            )
            fc_roi_df.to_csv(fc_roi_path, sep="\t", index_label="roi")
            logger.info("sub-%s | fc_roi → %s", config.subject, fc_roi_path)

            fcroiz_path = build_output_path(
                output_dir=output_dir, subject=config.subject, session=config.session,
                entities=entities, suffix="fcroiz", extension=".tsv",
            )
            fisher_z(fc_roi_df).to_csv(fcroiz_path, sep="\t", index_label="roi")
            logger.info("sub-%s | fc_roiz → %s", config.subject, fcroiz_path)

    return alff_df, fc_df


def _write_step_snirf(haemo: mne.io.Raw, config: PostConfig, output_dir: Path, desc: str, source_entities: dict[str, str] | None = None) -> Path:
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities, write_sidecar_json
    from fnirs_pipe.io.snirf import write_snirf

    entities = carry_entities(source_entities)
    entities["desc"] = desc
    out_path = build_output_path(
        output_dir=output_dir,
        subject=config.subject,
        session=config.session,
        entities=entities,
        suffix="nirs",
        extension=".snirf",
    )
    write_snirf(haemo, out_path)
    write_sidecar_json(out_path, {
        "pipeline_version": __version__,
        "high_pass": config.high_pass,
        "low_pass": config.low_pass,
        "resample_sfreq": config.resample_sfreq,
    })
    logger.info("sub-%s | %s snirf → %s", config.subject, desc, out_path)
    return out_path
