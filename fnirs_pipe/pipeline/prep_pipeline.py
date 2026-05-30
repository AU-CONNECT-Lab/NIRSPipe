"""Preprocessing pipeline — all prep steps in a single module.

Step outputs written to output_dir/sub-XX/[ses-YY/]nirs/:
  desc-od            raw intensity → ΔOD
  desc-sci           SCI-pruned OD
  desc-motcorrected  motion-corrected OD
  desc-preproc       final HbO/HbR  (Beer-Lambert output)

Each snirf is accompanied by a JSON provenance sidecar.

motion correction methods other than tddr are not implemented yet (no MNE backend).
considering adding detrending/dispike from NIRS-KIT toolbox

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mne
import mne.io
import numpy as np
from bids import BIDSLayout

from fnirs_pipe import __version__
from fnirs_pipe.io.derivatives import build_output_path, write_sidecar_json
from fnirs_pipe.io.snirf import write_snirf
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.prep")

MotionMethod = Literal["tddr", "wavelet", "spline", "none"]

# Step 1: BIDS validation
def validate(bids_dir: Path) -> None:
    """Validate BIDS dataset structure; raise BIDSValidationError on critical errors."""
    BIDSLayout(str(bids_dir), validate=True)

# Step 2: OD conversion
def intensity_to_od(raw: mne.io.Raw) -> mne.io.Raw:
    """Convert raw intensity signal to optical density."""
    return mne.preprocessing.nirs.optical_density(raw)

# Step 3: SCI / bad channel pruning
def compute_sci(raw_od: mne.io.Raw) -> dict[str, float]:
    """Return SCI score per channel name."""
    from mne.preprocessing.nirs import scalp_coupling_index
    scores = scalp_coupling_index(raw_od)
    return dict(zip(raw_od.ch_names, scores))

# For QC reporting, sliding window
def compute_windowed_sci(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float = 0.7,
    cardiac_h_freq: float = 1.5,
) -> "tuple[np.ndarray, np.ndarray]":
    from mne_nirs.preprocessing import scalp_coupling_index_windowed
    _, scores, times = scalp_coupling_index_windowed(
        raw_od, time_window=30, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times

def compute_windowed_psp(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float = 0.7,
    cardiac_h_freq: float = 1.5,
) -> "tuple[np.ndarray, np.ndarray]":
    from mne_nirs.preprocessing import peak_power
    _, scores, times = peak_power(
        raw_od, time_window=10, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times

def compute_windowed_gvtd(
    raw_od: mne.io.Raw, window_s: float = 30.0,
) -> "tuple[np.ndarray, np.ndarray]":
    """Mean GVTD per non-overlapping window. Returns (gvtd_per_window, window_center_times)."""
    import numpy as np
    sfreq = float(raw_od.info["sfreq"])
    diff_data = np.diff(raw_od.get_data(), axis=1)
    gvtd_ts = np.sqrt(np.mean(diff_data ** 2, axis=0))
    win_samples = max(1, int(round(window_s * sfreq)))
    n_windows = len(gvtd_ts) // win_samples
    if n_windows == 0:
        return np.array([]), np.array([])
    truncated = gvtd_ts[:n_windows * win_samples].reshape(n_windows, win_samples)
    gvtd_per_window = truncated.mean(axis=1)
    window_times = np.arange(n_windows) * window_s + window_s / 2
    return gvtd_per_window, window_times


def mark_bad_channels(
    raw_od: mne.io.Raw,
    threshold: float,
) -> tuple[mne.io.Raw, list[str], dict[str, float]]:
    """Mark channels below SCI threshold into raw.info['bads'].

    Returns raw (modified in-place), list of bad channel names, and SCI scores dict.
    """
    sci_scores = compute_sci(raw_od)
    bad_chs = [ch for ch, score in sci_scores.items() if score < threshold]
    raw_od.info["bads"] = bad_chs
    return raw_od, bad_chs, sci_scores


# Step 4: Motion correction
def correct_motion(raw_od: mne.io.Raw, method: MotionMethod | None = None) -> mne.io.Raw:
    """Apply motion artifact correction to the OD signal.

    Returns corrected raw (modified in-place for tddr).
    Raises NotImplementedError for wavelet and spline (no MNE backend yet).
    """
    if method is None:
        raise ValueError("--motion-correction is required.")
    if method == "tddr":
        return mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    elif method == "wavelet":
        raise NotImplementedError(
            "Wavelet correction has no MNE backend yet. "
            "Use --motion-correction tddr or none."
        )
    elif method == "spline":
        raise NotImplementedError(
            "Spline correction has no MNE backend yet. "
            "Use --motion-correction tddr or none."
        )
    elif method == "none":
        return raw_od
    raise ValueError(f"Unknown motion correction method: {method}")

# Step 5: Beer-Lambert
def od_to_haemo(raw_od: mne.io.Raw, dpf: list[float]) -> mne.io.Raw:
    """Convert OD to haemoglobin concentration via Beer-Lambert law."""
    from mne.preprocessing.nirs import beer_lambert_law
    ppf = dpf[0] if len(dpf) == 1 else dpf
    return beer_lambert_law(raw_od, ppf=ppf)

# Pipeline orchestration
@dataclass
class PrepResult:
    """Outputs from run_prep() needed for reporting and downstream use."""
    raw_haemo: mne.io.Raw
    sci_scores: dict[str, float]
    bad_channels: list[str]
    sci_scores_matrix: "np.ndarray | None" = None
    sci_win_times: "np.ndarray | None" = None
    psp_scores_matrix: "np.ndarray | None" = None
    psp_win_times: "np.ndarray | None" = None
    raw_od_before_motion: "mne.io.Raw | None" = None
    raw_od_after_motion: "mne.io.Raw | None" = None
    iqm_raw: "dict | None" = None
    iqm_final: "dict | None" = None

@dataclass
class PrepConfig:
    subject: str
    dpf: list[float]                        # required; one value or one per wavelength
    sci_threshold: float                    # required; e.g. 0.8
    cardiac_l_freq: float
    cardiac_h_freq: float
    session: str | None = None
    motion_correction: str | None = None
    exclude_channels: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)

def run_prep(
    raw: mne.io.Raw,
    config: PrepConfig,
    output_dir: Path,
    source_entities: dict[str, str] | None = None,
    work_dir: Path | None = None,
) -> PrepResult:
    """Run the full preprocessing pipeline in locked step order.

    Steps (order is fixed):
      1. BIDS validation   -> handled upstream before this call
      2. OD conversion     -> desc-od_nirs.snirf
      3. SCI channel marking -> desc-sci_nirs.snirf
      4. Motion correction -> desc-motcorrected_nirs.snirf
      5. Beer-Lambert      -> desc-preproc_nirs.snirf
    """
    # Propagate task/run from source file so output filenames mirror input entities.
    entities_base = {k: v for k, v in (source_entities or {}).items() if k in ("task", "run")}
    ses = config.session

    def _save(raw_step: mne.io.Raw, desc: str, step: str, extra_provenance: dict = {}) -> Path:
        path = build_output_path(
            output_dir=output_dir,
            subject=config.subject,
            entities={**entities_base, "desc": desc},
            suffix="nirs",
            extension=".snirf",
            session=ses,
        )
        write_snirf(raw_step, path)
        write_sidecar_json(path, {
            "pipeline_version": __version__,
            "step": step,
            "parameters": _config_dict(config),
            **extra_provenance,
        })
        return path

    # step 2: OD conversion
    logger.info("sub-%s | step 2: OD conversion (%d ch)", config.subject, len(raw.ch_names))
    raw_od = intensity_to_od(raw)
    if config.exclude_channels:
        logger.info("sub-%s | excluding channels: %s", config.subject, config.exclude_channels)
        raw_od.drop_channels(config.exclude_channels)
    _save(raw_od, "od", "od_conversion")

    # step 3: SCI channel marking
    logger.info("sub-%s | step 3: SCI marking (threshold=%.2f, %d ch)", config.subject, config.sci_threshold, len(raw_od.ch_names))
    raw_od, bad_chs, sci_scores = mark_bad_channels(raw_od, threshold=config.sci_threshold)
    n_bad, n_total = len(bad_chs), len(sci_scores)
    logger.info(
        "sub-%s | bad channels: %d/%d%s",
        config.subject, n_bad, n_total,
        f" — {bad_chs}" if bad_chs else "",
    )
    sci_path = _save(raw_od, "sci", "sci_pruning", extra_provenance={"bad_channels": bad_chs})

    # raw IQM checkpoint — intensity metrics on original signal before any correction
    iqm_raw: dict | None = None
    try:
        from fnirs_pipe.qc.quantitative_metrics import compute_raw_iqm, save_iqm_toml
        iqm_raw = compute_raw_iqm(raw, sci_scores, bad_chs)
        save_iqm_toml(iqm_raw, config.subject, sci_path.parent, suffix="_raw")
    except Exception:
        logger.warning("sub-%s | raw IQM failed", config.subject, exc_info=True)

    # step 4: motion correction (spike/step artifact repair)
    logger.info("sub-%s | step 4: motion correction (%s)", config.subject, config.motion_correction)
    raw_od_before_motion = raw_od.copy()
    raw_od = correct_motion(raw_od, method=config.motion_correction)
    _save(raw_od, "motcorrected", "motion_correction")

    # step 5: Beer-Lambert
    logger.info("sub-%s | step 5: Beer-Lambert (dpf=%s)", config.subject, config.dpf)
    raw_haemo = od_to_haemo(raw_od, dpf=config.dpf)
    preproc_path = _save(raw_haemo, "preproc", "beer_lambert")

    # haemo IQM checkpoint — baseline, overwritten by post-pipeline if filter/resample runs
    iqm_final: dict | None = None
    try:
        from fnirs_pipe.qc.quantitative_metrics import compute_haemo_iqm, save_iqm_toml
        iqm_final = compute_haemo_iqm(raw_haemo)
        save_iqm_toml(iqm_final, config.subject, preproc_path.parent)
    except Exception:
        logger.warning("sub-%s | haemo IQM failed", config.subject, exc_info=True)

    sci_matrix = sci_times = psp_matrix = psp_times = None
    try:
        sci_matrix, sci_times = compute_windowed_sci(raw_od, config.cardiac_l_freq, config.cardiac_h_freq)
        psp_matrix, psp_times = compute_windowed_psp(raw_od, config.cardiac_l_freq, config.cardiac_h_freq)
    except Exception:
        logger.warning("sub-%s | windowed SCI/PSP computation failed", config.subject, exc_info=True)

    return PrepResult(
        raw_haemo=raw_haemo,
        sci_scores=sci_scores,
        bad_channels=bad_chs,
        sci_scores_matrix=sci_matrix,
        sci_win_times=sci_times,
        psp_scores_matrix=psp_matrix,
        psp_win_times=psp_times,
        raw_od_before_motion=raw_od_before_motion,
        raw_od_after_motion=raw_od,
        iqm_raw=iqm_raw,
        iqm_final=iqm_final,
    )


def _config_dict(config: PrepConfig) -> dict:
    return {
        "subject": config.subject,
        "session": config.session,
        "dpf": config.dpf,
        "motion_correction": config.motion_correction,
        "sci_threshold": config.sci_threshold,
        "cardiac_l_freq": config.cardiac_l_freq,
        "cardiac_h_freq": config.cardiac_h_freq,
        "exclude_channels": config.exclude_channels,
        "ignore": config.ignore,
    }
