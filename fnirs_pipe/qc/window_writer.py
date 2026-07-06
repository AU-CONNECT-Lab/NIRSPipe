"""Time-windowed group QC: crop each subject's raw SNIRF to [tstart, tend],
recompute SQM scalars, render the same heatmap/boxplot/outlier panels as
`fnirs-qc group-raw`. Output goes into `<output_dir>/group_nirs_{name}.{tsv,html}`."""

from __future__ import annotations

from pathlib import Path

import mne

from fnirs_pipe.io.bids import get_layout, iter_run_files
from fnirs_pipe.qc.group_writer import _render_group, rows_to_dataframe
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.window_writer")


def _crop_to_window(
    raw: mne.io.Raw,
    tstart: float,
    tend: float,
    align: str = "none",
    trigger_name: str | None = None,
) -> mne.io.Raw | None:
    """Crop raw to [tstart, tend] (seconds). Returns None if window invalid."""
    t0 = 0.0
    if align == "trigger":
        if not trigger_name:
            raise ValueError("--align trigger requires --trigger-name")
        onsets = [
            float(a["onset"]) for a in raw.annotations
            if str(a["description"]) == trigger_name
        ]
        if not onsets:
            logger.warning("trigger '%s' not found; falling back to t=0", trigger_name)
        else:
            t0 = onsets[0]

    abs_start = t0 + tstart
    abs_end   = t0 + tend
    duration  = float(raw.times[-1])
    if abs_start >= duration or abs_end <= 0:
        return None
    cropped = raw.copy().crop(
        tmin=max(0.0, abs_start),
        tmax=min(duration, abs_end),
    )
    return cropped if cropped.times.size > 1 else None


def _sqm_for_cropped(cropped: mne.io.Raw, sci_threshold: float,
                     cardiac_l_freq: float, cardiac_h_freq: float,
                     windowed: bool = True) -> dict:
    """Raw SQM (SCI + scalars, + optional windowed series) for an already-cropped raw.

    Shared by window-raw (one fixed window) and epoch QC (one window per trial).
    windowed=False skips the sliding-window series (meaningless on short trial windows).
    """
    from fnirs_pipe.qc.quantitative_metrics import (
        attach_windowed_series, compute_raw_sqm, compute_sci_scores,
    )

    sci_scores, raw_od = compute_sci_scores(cropped, cardiac_l_freq, cardiac_h_freq)
    bad_channels = [ch for ch, s in sci_scores.items() if s < sci_threshold]
    try:
        sqm = compute_raw_sqm(cropped, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq)
    except Exception as exc:
        logger.warning("compute_raw_sqm failed: %s", exc)
        sqm = {}

    if windowed:
        attach_windowed_series(sqm, raw_od, cardiac_l_freq, cardiac_h_freq)
    return sqm


def _compute_sqm_for_window(snirf_path: Path, tstart: float, tend: float,
                            align: str, trigger_name: str | None,
                            sci_threshold: float,
                            cardiac_l_freq: float, cardiac_h_freq: float) -> dict | None:
    """Crop SNIRF to one fixed window + recompute raw SQM (scalars + windowed metrics)."""
    raw = mne.io.read_raw_snirf(str(snirf_path), preload=True, verbose=False)
    cropped = _crop_to_window(raw, tstart, tend, align, trigger_name)
    if cropped is None:
        logger.warning("skip %s: window [%g, %g] out of range", snirf_path.name, tstart, tend)
        return None
    return _sqm_for_cropped(cropped, sci_threshold, cardiac_l_freq, cardiac_h_freq, windowed=True)


def build_window_raw_report(
    bids_dir: Path,
    output_dir: Path,
    task: str,
    tstart: float,
    tend: float,
    participant_label: list[str] | None = None,
    session_label: list[str] | None = None,
    align: str = "none",
    trigger_name: str | None = None,
    name: str | None = None,
    sci_threshold: float = 0.8,
    *,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    skip_bids_validation: bool = False,
) -> Path:
    """Aggregate windowed SQM across subjects → group_nirs_{name}.{tsv,html}."""
    layout = get_layout(bids_dir, validate=not skip_bids_validation)
    subjects = participant_label or sorted(layout.get_subjects())
    sessions = session_label or [None]

    rows: list[dict] = []
    for snirf_path, bids_name in iter_run_files(layout, subjects, sessions, task):
        sqm = _compute_sqm_for_window(
            snirf_path, tstart, tend, align, trigger_name, sci_threshold,
            cardiac_l_freq, cardiac_h_freq,
        )
        if sqm is None:
            continue
        rows.append({"bids_name": bids_name, **sqm})
        logger.info("computed window SQM for %s", bids_name)

    if not rows:
        logger.warning("no SNIRF processed; report will be empty")

    out_stem = f"group_nirs_{name}" if name else f"group_nirs_window-{tstart:g}-{tend:g}"
    title    = f"Window QC (task={task}, t=[{tstart:g}, {tend:g}]s)"
    df       = rows_to_dataframe(rows)
    return _render_group(output_dir, out_stem, title, df, rows)
