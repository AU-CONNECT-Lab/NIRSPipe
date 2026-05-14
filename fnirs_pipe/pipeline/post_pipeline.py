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
    session: str | None = None
    dry_run: bool = False

    # bandpass filter
    high_pass: float | None = None
    low_pass:  float | None = None

    # resample
    resample_sfreq: float | None = None

    # crop (applied after filtering, before GLM)
    # segments_path takes priority; crop_tmin/crop_tmax used only when segments_path is None
    segments_path: str | None = None
    crop_tmin: float | None = None
    crop_tmax: float | None = None

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
    """Run post-processing pipeline. Returns (result, glm_est, design_matrix).

    glm_est and design_matrix are None for non-GLM modes.
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

    if config.segments_path is not None:
        logger.info("sub-%s | crop to segments: %s", config.subject, config.segments_path)
        result = _crop_to_segments(result, config.segments_path)
    elif config.crop_tmin is not None or config.crop_tmax is not None:
        logger.info("sub-%s | crop: tmin=%s tmax=%s", config.subject, config.crop_tmin, config.crop_tmax)
        result = result.crop(tmin=config.crop_tmin, tmax=config.crop_tmax)

    if last_snirf_path is not None:
        try:
            from fnirs_pipe.qc.quantitative_metrics import compute_haemo_iqm, save_iqm_toml
            save_iqm_toml(compute_haemo_iqm(result), config.subject, last_snirf_path.parent)
        except Exception:
            logger.warning("sub-%s | haemo IQM failed", config.subject, exc_info=True)

    glm_est = dm = None
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
        _write_step_snirf(raw_resid, config, output_dir, desc="errts", source_entities=source_entities)

    return result, glm_est, dm

def _crop_to_segments(raw: mne.io.Raw, segments_path: str) -> mne.io.Raw:
    df = pd.read_csv(segments_path, sep="\t")
    segments = [
        raw.copy().crop(tmin=row.onset, tmax=row.onset + row.duration)
        for _, row in df.iterrows()
    ]
    return mne.concatenate_raws(segments)


def _write_step_snirf(haemo: mne.io.Raw, config: PostConfig, output_dir: Path, desc: str, source_entities: dict[str, str] | None = None) -> None:
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import build_output_path, write_sidecar_json
    from fnirs_pipe.io.snirf import write_snirf

    entities = {k: v for k, v in (source_entities or {}).items() if k in ("task", "run")}
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
