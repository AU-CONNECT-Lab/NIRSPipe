from __future__ import annotations
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import mne
import mne.io

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("post.glm")

HRFModel   = Literal[
    "spm",
    "spm + derivative",
    "spm + derivative + dispersion",
    "glover",
    "glover + derivative",
    "glover + derivative + dispersion",
    "fir",
]
# arN (e.g. "ar2", "ar3") is also valid but cannot be expressed as a Literal
NoiseModel = Literal["ols", "ar1", "ar2", "ar3", "ar4", "ar5", "auto"]
DriftModel = Literal["cosine", "polynomial", "none"]
SCRStrategy = Literal["mean", "pca"]

# May add motion parameters (if available) and/or other confounds in the future

def _short_channel_regressors(haemo: mne.io.Raw, strategy: SCRStrategy) -> dict[str, np.ndarray]:
    from mne_nirs.channels import get_short_channels

    short = get_short_channels(haemo)
    hbo_data = short.copy().pick(picks="hbo").get_data()  # (n_channels, n_times)
    hbr_data = short.copy().pick(picks="hbr").get_data()
    if strategy == "pca":
        # TODO: check PCA
        from sklearn.decomposition import PCA
        return {
            "short_ch_hbo_pc1": PCA(n_components=1).fit_transform(hbo_data.T)[:, 0],
            "short_ch_hbr_pc1": PCA(n_components=1).fit_transform(hbr_data.T)[:, 0],
        }
    return {
        "short_ch_hbo_mean": hbo_data.mean(axis=0),
        "short_ch_hbr_mean": hbr_data.mean(axis=0),
    }

def build_design_matrix(
    raw: mne.io.Raw,
    stim_dur: float | None,
    hrf_model: HRFModel,
    drift_model: DriftModel,
    high_pass: float | None,
    drift_order: int | None,
    fir_delays: tuple[int, ...] = (0,),
    add_regs: pd.DataFrame | None = None,
    add_reg_names: list[str] | None = None,
    min_onset: float = -24,
    oversampling: int = 50,
    events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a design matrix for GLM analysis.
    Modified from mne_nirs.experimental_design.make_first_level_design_matrix to accept external events and confounds,
    and to handle the case where no events are present (read from snirf annotations instead).

    Parameters
    ----------
    raw : mne.io.Raw
        The raw fNIRS data.
    stim_dur : float | None
        Stimulus duration in seconds. Required when events is None (annotation fallback).
    hrf_model : HRFModel
        The HRF model to use.
    drift_model : DriftModel
        The drift model to use.
    high_pass : float | None
        High-pass cutoff for cosine drift (Hz). Only used when drift_model='cosine'.
    drift_order : int | None
        Polynomial drift order. Only used when drift_model='polynomial'.
    fir_delays : tuple[int, ...], optional
        FIR delay bins in scans, by default (0,).
    add_regs : pd.DataFrame | None, optional
        Additional regressors to include in the design matrix, by default None.
    add_reg_names : list[str] | None, optional
        Names for the additional regressors, by default None.
    min_onset : float, optional
        The minimum onset time, by default -24.
    oversampling : int, optional
        The oversampling factor, by default 50.
    events : pd.DataFrame | None, optional
        Events DataFrame with columns 'onset', 'duration', 'trial_type'. If None, reads from snirf annotations.

    Returns
    -------
    pd.DataFrame
        The design matrix.

    # https://mne.tools/mne-nirs/dev/_modules/mne_nirs/experimental_design/_experimental_design.html#make_first_level_design_matrix
    # https://nilearn.github.io/dev/modules/generated/nilearn.glm.first_level.make_first_level_design_matrix.html
    """
    
    from nilearn.glm.first_level import make_first_level_design_matrix

    frame_times = raw.times

    if events is None:
        if stim_dur is None:
            raise ValueError("stim_dur required when no events DataFrame provided")
        conditions = raw.annotations.description
        onsets = raw.annotations.onset - raw.first_time
        duration = stim_dur * np.ones(len(conditions))
        events = pd.DataFrame({"trial_type": conditions, "onset": onsets, "duration": duration})

    return make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model=hrf_model,
        drift_model=drift_model if drift_model != "none" else None,
        high_pass=high_pass if drift_model == "cosine" else None,
        drift_order=drift_order if drift_model == "polynomial" else None,
        fir_delays=fir_delays,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
        min_onset=min_onset,
        oversampling=oversampling,
    )


def fit_glm(
    haemo: mne.io.Raw,
    design_matrix: pd.DataFrame,
    noise_model: NoiseModel = "ar1",
) -> Any:
    from mne_nirs.statistics import run_glm
    return run_glm(haemo, design_matrix, noise_model=noise_model)


def compute_contrasts(
    glm_est: Any,
    contrast_def: dict[str, Any],
) -> dict[str, Any]:
    from mne_nirs.statistics import compute_contrast
    return {name: compute_contrast(glm_est, weights) for name, weights in contrast_def.items()}

def run_glm_pipeline(
    haemo: mne.io.Raw,
    stim_dur: float | None,
    hrf_model: HRFModel,
    noise_model: NoiseModel,
    drift_model: DriftModel,
    high_pass: float | None,
    drift_order: int | None,
    fir_delays: tuple[int, ...] | None,
    events_path: str | None = None,
    events: pd.DataFrame | None = None,
    short_channel: bool | SCRStrategy | None = None,
    contrast_def: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> mne.io.Raw:
    # explicit events take precedence; then external TSV; then snirf annotations
    if events is None:
        events = pd.read_csv(events_path, sep="\t") if events_path else None

    confound_cols = _short_channel_regressors(haemo, short_channel) if short_channel else {}
    confounds = pd.DataFrame(confound_cols) if confound_cols else None

    dm = build_design_matrix(
        raw=haemo,
        stim_dur=stim_dur,
        hrf_model=hrf_model,
        drift_model=drift_model,
        high_pass=high_pass,
        drift_order=drift_order,
        fir_delays=fir_delays,
        add_regs=confounds,
        events=events,
    )
    logger.debug("design matrix: %d scans × %d regressors — %s", dm.shape[0], dm.shape[1], list(dm.columns))

    glm_est = fit_glm(haemo, dm, noise_model=noise_model)

    # nilearn stores residuals as (n_times, 1) per channel; squeeze removes the trailing dim
    resid_data = np.array([glm_est.data[ch].residuals for ch in glm_est.ch_names]).squeeze(-1)
    raw_resid = haemo.copy()
    raw_resid._data[:] = resid_data

    contrasts = compute_contrasts(glm_est, contrast_def) if contrast_def else None

    if output_dir:
        _save_glm_outputs(glm_est, dm, Path(output_dir), contrasts=contrasts)

    return haemo, glm_est, dm, raw_resid


def _save_glm_outputs(
    glm_est: Any,
    design_matrix: pd.DataFrame,
    output_dir: Path,
    contrasts: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(output_dir / "design_matrix.csv", index=False)

    glm_est.to_dataframe().to_csv(output_dir / "glm_results.csv", index=False)
    # glm_est.save(str(output_dir / "glm.h5"), overwrite=True)

    if contrasts:
        frames = []
        for name, result in contrasts.items():
            df = result.to_dataframe()
            df.insert(0, "contrast", name)
            frames.append(df)
        pd.concat(frames, ignore_index=True).to_csv(output_dir / "contrasts.csv", index=False)

    logger.info("GLM outputs written to %s", output_dir)
