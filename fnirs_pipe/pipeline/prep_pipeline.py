"""Preprocessing pipeline — all prep steps in a single module.

Step outputs written to output_dir/sub-XX/[ses-YY/]nirs/:
  desc-od            raw intensity → ΔOD
  desc-sci           SCI-pruned OD
  desc-motcorrected  motion-corrected OD
  desc-preproc       final HbO/HbR  (Beer-Lambert output)

Each snirf is accompanied by a JSON provenance sidecar.

motion correction: tddr and wavelet implemented; spline not yet.

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mne
import mne.io
import numpy as np

from fnirs_pipe import __version__
from fnirs_pipe.io.derivatives import build_output_path, carry_entities, write_sidecar_json
from fnirs_pipe.io.snirf import write_snirf
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.prep")

MotionMethod = Literal["tddr", "wavelet", "spline", "none"]

# Step 1: OD conversion
def intensity_to_od(raw: mne.io.Raw) -> mne.io.Raw:
    """Convert raw intensity signal to optical density."""
    return mne.preprocessing.nirs.optical_density(raw)

# Step 2: SCI / bad channel pruning
def compute_sci(raw_od: mne.io.Raw, cardiac_l_freq: float, cardiac_h_freq: float) -> dict[str, float]:
    """Return SCI score per channel name."""
    from mne.preprocessing.nirs import scalp_coupling_index
    scores = scalp_coupling_index(raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq)
    return dict(zip(raw_od.ch_names, scores))

# For QC reporting, sliding window
def compute_windowed_sci(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> "tuple[np.ndarray, np.ndarray]":
    from mne_nirs.preprocessing import scalp_coupling_index_windowed
    _, scores, times = scalp_coupling_index_windowed(
        raw_od, time_window=30, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times

def compute_windowed_psp(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> "tuple[np.ndarray, np.ndarray]":
    from mne_nirs.preprocessing import peak_power
    _, scores, times = peak_power(
        raw_od, time_window=10, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq
    )
    return scores, times

def _windowed_gvtd(
    raw_od: mne.io.Raw, window_s: float, l_freq: float | None, h_freq: float | None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    import numpy as np

    from fnirs_pipe.qc.quantitative_metrics import gvtd_timetrace
    sfreq = float(raw_od.info["sfreq"])
    gvtd_ts = gvtd_timetrace(raw_od.get_data(), sfreq, l_freq=l_freq, h_freq=h_freq)
    win_samples = max(1, int(round(window_s * sfreq)))
    n_windows = len(gvtd_ts) // win_samples
    if n_windows == 0:
        return np.array([]), np.array([]), np.array([])
    truncated = gvtd_ts[:n_windows * win_samples].reshape(n_windows, win_samples)
    # mean = average motion level; p95 = worst-moment, so transient motion survives averaging
    gvtd_mean = truncated.mean(axis=1)
    gvtd_p95 = np.percentile(truncated, 95, axis=1)
    window_times = np.arange(n_windows) * window_s + window_s / 2
    return gvtd_mean, gvtd_p95, window_times


def compute_windowed_gvtd(
    raw_od: mne.io.Raw, window_s: float = 30.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean & p95 GVTD per non-overlapping window (unfiltered). Returns (mean, p95, center_times)."""
    return _windowed_gvtd(raw_od, window_s, None, None)


def compute_windowed_filtered_gvtd(
    raw_od: mne.io.Raw, window_s: float = 30.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Mean & p95 GVTD per window on the motion-band bandpassed OD. Returns (mean, p95, center_times)."""
    from fnirs_pipe.qc.quantitative_metrics import GVTD_MOTION_BAND
    return _windowed_gvtd(raw_od, window_s, *GVTD_MOTION_BAND)


def _expand_bad_pairs(raw: mne.io.Raw, labels: list[str]) -> list[str]:
    """Match S-D pair labels (or full channel names) to channels present in raw."""
    wanted = set(labels)
    return [ch for ch in raw.ch_names if ch in wanted or ch.rsplit(" ", 1)[0] in wanted]


def mark_bad_channels(
    raw_od: mne.io.Raw,
    threshold: float,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
) -> tuple[mne.io.Raw, list[str], dict[str, float]]:
    """Mark channels below SCI threshold into raw.info['bads'].

    Returns raw (modified in-place), list of bad channel names, and SCI scores dict.
    """
    sci_scores = compute_sci(raw_od, cardiac_l_freq, cardiac_h_freq)
    bad_chs = [ch for ch, score in sci_scores.items() if score < threshold]
    raw_od.info["bads"] = bad_chs
    return raw_od, bad_chs, sci_scores


def _wl_clip_iqr(block: np.ndarray, iqr_factor: float) -> None:
    q25, q75 = np.percentile(block, [25, 75])
    fence = iqr_factor * (q75 - q25)
    block[:] = np.where((block > q75 + fence) | (block < q25 - fence), 0.0, block)


def _wl_filter_coeffs(coeffs, iqr_factor: float, signal_length: int):
    """Zero SWT detail-coefficient outliers per block per level (Homer3 hmrR_MotionCorrectWavelet)."""
    n = len(coeffs[0][0])
    n_levels = len(coeffs)
    cAf = coeffs[0][0].copy()          # highest-level approximation
    out = []
    for i, (_, cD) in enumerate(coeffs):
        n_blocks = 2 ** (n_levels - i - 1)
        block_length = n // n_blocks
        cDf = cD.copy()
        for b in range(n_blocks):
            start, end = b * block_length, min(signal_length, (b + 1) * block_length)
            if end > start:
                _wl_clip_iqr(cDf[start:end], iqr_factor)
        out.append((cAf, cDf))
    return out


def _wavelet_motion_correct(raw_od: mne.io.Raw, wavelet: str = "db2", iqr_factor: float = 1.5, level: int = 4) -> mne.io.Raw:
    """Wavelet motion correction (Molavi 2012), per channel in OD space.

    Pad to 2^k → remove DC → MAD noise-normalize → SWT → zero detail-coefficient outliers beyond
    Q1/Q3 ± iqr_factor·IQR (per block) → iSWT → denormalize → restore DC and length.
    """
    import pywt
    raw = raw_od.copy()

    def _corr(signal):
        n0 = len(signal)
        padded = np.zeros(2 ** int(np.ceil(np.log2(n0))))
        padded[:n0] = signal
        dc = padded.mean()
        padded = padded - dc
        mad_ds = np.median(np.abs(padded[::2] - np.median(padded[::2])))
        norm_coef = 1.0 / (1.4826 * mad_ds) if mad_ds != 0 else 1.0
        normed = padded * norm_coef
        lvl = min(level, int(np.log2(len(normed))) - 1)
        coeffs = _wl_filter_coeffs(pywt.swt(normed, wavelet, level=lvl), iqr_factor, n0)
        return (pywt.iswt(coeffs, wavelet) / norm_coef)[:n0] + dc

    raw.apply_function(_corr, channel_wise=True)
    return raw


# Step 3: Motion correction
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
        return _wavelet_motion_correct(raw_od)
    elif method == "spline":
        raise NotImplementedError(
            "Spline correction has no MNE backend yet. "
            "Use --motion-correction tddr or none."
        )
    elif method == "none":
        return raw_od
    raise ValueError(f"Unknown motion correction method: {method}")

# Step 4: Beer-Lambert
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
    sqm_raw: "dict | None" = None
    sqm_final: "dict | None" = None

@dataclass
class PrepConfig:
    subject: str
    dpf: list[float]                        # required; one value or one per wavelength
    sci_threshold: float                    # required; e.g. 0.8
    cardiac_l_freq: float
    cardiac_h_freq: float
    resp_l_freq: float
    resp_h_freq: float
    session: str | None = None
    motion_correction: str | None = None
    bad_channels: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)

def run_prep(
    raw: mne.io.Raw,
    config: PrepConfig,
    output_dir: Path,
    source_entities: dict[str, str] | None = None,
    work_dir: Path | None = None,
) -> PrepResult:
    """Run the full preprocessing pipeline in locked step order.

    Steps (order is fixed; BIDS validation handled upstream before this call):
      1. OD conversion     -> desc-od_nirs.snirf
      2. SCI channel marking -> desc-sci_nirs.snirf
      3. Motion correction -> desc-motcorrected_nirs.snirf
      4. Beer-Lambert      -> desc-preproc_nirs.snirf
    """
    entities_base = carry_entities(source_entities)
    ses = config.session

    def _save(raw_step: mne.io.Raw, desc: str, step: str, extra_provenance: dict | None = None) -> Path:
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
            **(extra_provenance or {}),
        })
        return path

    # step 1: OD conversion (skip if input is already optical density)
    if is_optical_density(raw):
        logger.warning(
            "sub-%s | input is already optical density; skipping OD conversion "
            "and raw-intensity QC", config.subject,
        )
        raw_od = raw.copy()
    else:
        logger.info("sub-%s | step 1: OD conversion (%d ch)", config.subject, len(raw.ch_names))
        raw_od = intensity_to_od(raw)
    _save(raw_od, "od", "od_conversion")

    # step 2: SCI channel marking
    logger.info("sub-%s | step 2: SCI marking (threshold=%.2f, %d ch)", config.subject, config.sci_threshold, len(raw_od.ch_names))
    raw_od, bad_chs, sci_scores = mark_bad_channels(
        raw_od, threshold=config.sci_threshold,
        cardiac_l_freq=config.cardiac_l_freq, cardiac_h_freq=config.cardiac_h_freq)
    if config.bad_channels:
        manual = _expand_bad_pairs(raw_od, config.bad_channels)
        if not manual:
            logger.warning("sub-%s | --bad-channels matched no channels: %s", config.subject, config.bad_channels)
        else:
            logger.info("sub-%s | manual bad channels: %s", config.subject, manual)
        bad_chs = sorted(set(bad_chs) | set(manual))
        raw_od.info["bads"] = bad_chs
    n_bad, n_total = len(bad_chs), len(sci_scores)
    logger.info(
        "sub-%s | bad channels: %d/%d%s",
        config.subject, n_bad, n_total,
        f" — {bad_chs}" if bad_chs else "",
    )
    sci_path = _save(raw_od, "sci", "sci_pruning", extra_provenance={"bad_channels": bad_chs})

    # raw SQM checkpoint — intensity metrics on original signal before any correction
    sqm_raw: dict | None = None
    try:
        from fnirs_pipe.qc.quantitative_metrics import compute_raw_sqm, save_sqm_toml
        sqm_raw = compute_raw_sqm(raw, sci_scores, bad_chs, config.cardiac_l_freq, config.cardiac_h_freq)
        save_sqm_toml(sqm_raw, config.subject, sci_path.parent, suffix="_raw")
    except Exception:
        logger.warning("sub-%s | raw SQM failed", config.subject, exc_info=True)

    # step 3: motion correction (spike/step artifact repair)
    logger.info("sub-%s | step 3: motion correction (%s)", config.subject, config.motion_correction)
    raw_od_before_motion = raw_od.copy()
    raw_od = correct_motion(raw_od, method=config.motion_correction)
    _save(raw_od, "motcorrected", "motion_correction")

    # step 4: Beer-Lambert
    logger.info("sub-%s | step 4: Beer-Lambert (dpf=%s)", config.subject, config.dpf)
    raw_haemo = od_to_haemo(raw_od, dpf=config.dpf)
    preproc_path = _save(raw_haemo, "preproc", "beer_lambert")

    # haemo SQM checkpoint — baseline, overwritten by post-pipeline if filter/resample runs
    sqm_final: dict | None = None
    try:
        from fnirs_pipe.qc.quantitative_metrics import compute_haemo_sqm, save_sqm_toml
        sqm_final = compute_haemo_sqm(
            raw_haemo, config.cardiac_l_freq, config.cardiac_h_freq,
            config.resp_l_freq, config.resp_h_freq)
        save_sqm_toml(sqm_final, config.subject, preproc_path.parent)
    except Exception:
        logger.warning("sub-%s | haemo SQM failed", config.subject, exc_info=True)

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
        sqm_raw=sqm_raw,
        sqm_final=sqm_final,
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
        "bad_channels": config.bad_channels,
        "ignore": config.ignore,
    }
