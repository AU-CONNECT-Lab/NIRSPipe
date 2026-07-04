"""Time-windowed group QC: crop each subject's raw SNIRF to [tstart, tend],
recompute SQM scalars, render the same heatmap/boxplot/outlier panels as
`fnirs-qc group-raw`. Output goes into `<output_dir>/group_nirs_{name}.{tsv,html}`."""

from __future__ import annotations

from pathlib import Path

import mne

from fnirs_pipe.io.bids import get_layout, get_nirs_files
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


def _compute_sqm_for_window(snirf_path: Path, tstart: float, tend: float,
                            align: str, trigger_name: str | None,
                            sci_threshold: float) -> dict | None:
    """Crop SNIRF + recompute raw SQM (channel scalars + windowed metrics)."""
    from fnirs_pipe.pipeline.prep_pipeline import (
        compute_windowed_gvtd, compute_windowed_psp, compute_windowed_sci,
    )
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_sqm

    raw = mne.io.read_raw_snirf(str(snirf_path), preload=True, verbose=False)
    cropped = _crop_to_window(raw, tstart, tend, align, trigger_name)
    if cropped is None:
        logger.warning("skip %s: window [%g, %g] out of range", snirf_path.name, tstart, tend)
        return None

    try:
        raw_od     = mne.preprocessing.nirs.optical_density(cropped.copy(), verbose=False)
        sci_arr    = mne.preprocessing.nirs.scalp_coupling_index(raw_od, verbose=False)
        sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(cropped.ch_names)}
    except Exception as exc:
        logger.warning("SCI failed for %s: %s", snirf_path.name, exc)
        sci_scores = {ch: 1.0 for ch in cropped.ch_names}
        raw_od     = mne.preprocessing.nirs.optical_density(cropped.copy(), verbose=False)

    bad_channels = [ch for ch, s in sci_scores.items() if s < sci_threshold]
    try:
        sqm = compute_raw_sqm(cropped, sci_scores, bad_channels)
    except Exception as exc:
        logger.warning("compute_raw_sqm failed for %s: %s", snirf_path.name, exc)
        sqm = {}

    # Windowed series (same convention as prep_raw_report; center-time scalar list)
    import numpy as _np
    def _center_times(t):
        a = _np.asarray(t)
        return (a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a).tolist()
    try:
        sci_matrix, sci_times   = compute_windowed_sci(raw_od)
        psp_matrix, psp_times   = compute_windowed_psp(raw_od)
        gvtd_per_window, gvtd_t = compute_windowed_gvtd(raw_od)
        if sci_matrix is not None:
            sqm["sci_per_window"]      = _np.asarray(sci_matrix).mean(axis=0).tolist()
            sqm["sci_window_times_s"]  = _center_times(sci_times)
        if psp_matrix is not None:
            sqm["psp_per_window"]      = _np.asarray(psp_matrix).mean(axis=0).tolist()
            sqm["psp_window_times_s"]  = _center_times(psp_times)
        if gvtd_per_window is not None and len(gvtd_per_window):
            sqm["gvtd_per_window"]     = _np.asarray(gvtd_per_window).tolist()
            sqm["gvtd_window_times_s"] = _center_times(gvtd_t)
    except Exception as exc:
        logger.warning("windowed metrics failed for %s: %s", snirf_path.name, exc)

    return sqm


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
    skip_bids_validation: bool = False,
) -> Path:
    """Aggregate windowed SQM across subjects → group_nirs_{name}.{tsv,html}."""
    layout = get_layout(bids_dir, validate=not skip_bids_validation)
    subjects = participant_label or sorted(layout.get_subjects())
    sessions = session_label or [None]

    rows: list[dict] = []
    for sub in subjects:
        for ses in sessions:
            files = get_nirs_files(layout, subject=sub, session=ses, task=task)
            for f in files:
                entities = layout.parse_file_entities(str(f))
                parts = [f"sub-{sub}"]
                if entities.get("session"): parts.append(f"ses-{entities['session']}")
                parts.append(f"task-{entities['task']}")
                if entities.get("run"):     parts.append(f"run-{entities['run']}")
                bids_name = "_".join(parts)

                sqm = _compute_sqm_for_window(
                    Path(f), tstart, tend, align, trigger_name, sci_threshold,
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
