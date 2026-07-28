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
from fnirs_pipe.exceptions import StageError
from fnirs_pipe.utils.lineage import Recorder, lineage_of, stage_of, stamp
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
    source_path: Path | None = None,
) -> tuple:
    """Run post-processing pipeline.

    Returns (result, glm_est, design_matrix, alff_df, fc_df, gcor_reg).
    glm_est / design_matrix are None for non-GLM modes.
    alff_df / fc_df are None for non-rest modes.
    gcor_reg (pre/post short-channel regression GCOR) is None unless short_channel ran.
    """

    rec = Recorder()
    if source_path is not None:
        rec.register_input(source_path, raw_haemo)

    result = raw_haemo.copy()
    last_snirf_path: Path | None = None

    if config.high_pass is not None or config.low_pass is not None:
        logger.info("sub-%s | bandpass: l_freq=%s h_freq=%s", config.subject, config.high_pass, config.low_pass)
        result = bandpass_filter(result, l_freq=config.high_pass, h_freq=config.low_pass)
        last_snirf_path = _write_step_snirf(result, config, output_dir, desc="filtered", rec=rec, source_entities=source_entities)

    if config.resample_sfreq is not None:
        logger.info("sub-%s | resample → %.1f Hz", config.subject, config.resample_sfreq)
        result = resample(result, config.resample_sfreq)
        last_snirf_path = _write_step_snirf(result, config, output_dir, desc="resampled", rec=rec, source_entities=source_entities)

    haemo_sqm = None
    if last_snirf_path is not None:
        try:
            from fnirs_pipe.qc.quantitative_metrics import compute_haemo_sqm, save_sqm_toml
            haemo_sqm = compute_haemo_sqm(result)
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
            source_path=rec.path_of(result),
        )
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", rec=rec, source_entities=source_entities)

    elif mode == "rest":
        if config.drift_model is None:
            raise ValueError("rest mode requires --drift-model")
        logger.info("sub-%s | rest confound regression", config.subject)
        rest_glm_kwargs = dict(
            stim_dur=None,
            hrf_model="spm",
            noise_model="ols",
            drift_model=config.drift_model,
            high_pass=config.drift_high_pass,
            drift_order=config.drift_order,
            fir_delays=None,
            short_channel=config.short_channel,
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
        raw_resid_bb = None
        if config.low_pass is not None and config.high_pass is not None:
            result_bb = raw_haemo.copy()
            if config.resample_sfreq is not None:
                result_bb = resample(result_bb, config.resample_sfreq)
            _, _, _, raw_resid_bb = run_glm_pipeline(result_bb, **rest_glm_kwargs)
            # Same regression, un-bandpassed input. Re-stamp so it stops sharing the "errts"
            # stage with the bandpassed residual, whose file it would otherwise be credited to.
            stamp(raw_resid_bb, stage="errtsbroad", step="glm_residuals_broadband",
                  source=raw_haemo, resample_sfreq=config.resample_sfreq)
            _write_step_snirf(raw_resid_bb, config, output_dir, desc="errtsbroad",
                              rec=rec, source_entities=source_entities)

        alff_df, fc_df = _write_rest_derivatives(
            raw_resid, raw_resid_bb, config, output_dir, rec, source_entities=source_entities)

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
    raw_resid_bb: mne.io.Raw | None,
    config: PostConfig,
    output_dir: Path,
    rec: Recorder,
    source_entities: dict[str, str] | None = None,
) -> tuple:
    """Write ALFF/fALFF and FC TSVs. Returns (alff_df | None, fc_df).

    ALFF/fALFF use the broadband residual (raw_resid_bb); FC/FC-ROI use the bandpassed one.
    """
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities, write_sidecar_json
    from fnirs_pipe.pipeline.restingstate import compute_alff, compute_fc, compute_fc_roi, fisher_z

    entities = carry_entities(source_entities)

    def _sidecar(path: Path, step: str, source: str | None, **params) -> None:
        write_sidecar_json(path, {
            "pipeline_version": __version__,
            "step": step,
            "Sources": [source] if source else [],
            "parameters": params,
        })

    src_bp = rec.path_of(raw_resid)                                        # bandpassed residual
    src_bb = rec.path_of(raw_resid_bb) if raw_resid_bb is not None else None

    alff_df = None
    if raw_resid_bb is not None:
        alff_df = compute_alff(raw_resid_bb, low_pass=config.low_pass, high_pass=config.high_pass)
        alff_path = build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities=entities, suffix="alff", extension=".tsv",
        )
        alff_df.to_csv(alff_path, sep="\t", index=False)
        _sidecar(alff_path, "alff", src_bb, low_pass=config.low_pass, high_pass=config.high_pass)
        logger.info("sub-%s | alff → %s", config.subject, alff_path)
    else:
        logger.warning("sub-%s | skipping ALFF: --high-pass and --low-pass required", config.subject)

    # FC per chromophore: HbO and HbR anti-correlate, so they never share a matrix. Both are
    # written; the report consumes the HbO matrix.
    fc_hbo_df = None
    for chromo in ("hbo", "hbr"):
        fc_df = compute_fc(raw_resid, chromo)
        if fc_df.empty:
            continue
        chromo_entities = {**entities, "desc": chromo}
        fc_path = build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities=chromo_entities, suffix="fc", extension=".tsv",
        )
        fc_df.to_csv(fc_path, sep="\t", index_label="channel")
        _sidecar(fc_path, "fc", src_bp, chromophore=chromo)
        logger.info("sub-%s | fc (%s) → %s", config.subject, chromo, fc_path)

        fcz_path = build_output_path(
            output_dir=output_dir, subject=config.subject, session=config.session,
            entities=chromo_entities, suffix="fcz", extension=".tsv",
        )
        fisher_z(fc_df).to_csv(fcz_path, sep="\t", index_label="channel")
        _sidecar(fcz_path, "fisher_z", src_bp, chromophore=chromo)
        logger.info("sub-%s | fcz (%s) → %s", config.subject, chromo, fcz_path)

        if config.roi_map:
            fc_roi_df = compute_fc_roi(raw_resid, config.roi_map, chromo)
            if not fc_roi_df.empty:
                fc_roi_path = build_output_path(
                    output_dir=output_dir, subject=config.subject, session=config.session,
                    entities=chromo_entities, suffix="fcroi", extension=".tsv",
                )
                fc_roi_df.to_csv(fc_roi_path, sep="\t", index_label="roi")
                _sidecar(fc_roi_path, "fc_roi", src_bp, chromophore=chromo)
                logger.info("sub-%s | fc_roi (%s) → %s", config.subject, chromo, fc_roi_path)

                fcroiz_path = build_output_path(
                    output_dir=output_dir, subject=config.subject, session=config.session,
                    entities=chromo_entities, suffix="fcroiz", extension=".tsv",
                )
                fisher_z(fc_roi_df).to_csv(fcroiz_path, sep="\t", index_label="roi")
                _sidecar(fcroiz_path, "fisher_z", src_bp, chromophore=chromo)
                logger.info("sub-%s | fc_roiz (%s) → %s", config.subject, chromo, fcroiz_path)

        if chromo == "hbo":
            fc_hbo_df = fc_df

    return alff_df, fc_hbo_df


def _write_step_snirf(haemo: mne.io.Raw, config: PostConfig, output_dir: Path, desc: str, rec: Recorder, source_entities: dict[str, str] | None = None) -> Path:
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import build_output_path, carry_entities, write_sidecar_json
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
    write_snirf(haemo, out_path)
    write_sidecar_json(out_path, {
        "pipeline_version": __version__,
        "step": lin.step if lin else None,
        "Sources": rec.sources_of(haemo),
        "high_pass": config.high_pass,
        "low_pass": config.low_pass,
        "resample_sfreq": config.resample_sfreq,
    })
    logger.info("sub-%s | %s snirf → %s", config.subject, desc, out_path)
    return rec.written(out_path, haemo)
