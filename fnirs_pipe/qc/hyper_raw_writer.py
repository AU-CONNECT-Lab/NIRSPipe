"""Hyperscanning raw QC: build figure files + SQM JSON for the iframe-based viewer."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.pipeline.hyperscanning import GroupEntry
from fnirs_pipe.qc.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.figures.hyper_figures import (
    _cond_colors,
    build_channel_summary,
    build_coherence_bar,
    build_coherence_timeseries,
    build_epoch,
    build_layout_2d,
    build_layout_3d,
    build_psd,
    build_signal_overlay_pair,
    build_trigger_timeline,
    compute_hyper_sqm,
    compute_windowed_coherence,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_raw_writer")


def _process_hyper_raw_group(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    sqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    coherence_df: pd.DataFrame,
    output_dir: Path,
    raw_raws: dict[str, mne.io.Raw] | None = None,
    session: str | None = None,
    sci_threshold: float = 0.8,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    coherence_window_s: float = 30.0,
    coherence_step_s: float = 5.0,
) -> dict:
    """Compute hyper raw figures, save each as a standalone HTML, write SQM JSON.
    Returns the metadata dict the main viewer HTML needs."""

    label_parts = [f"group-{group_id}"]
    if session:
        label_parts.append(f"ses-{session}")
    label_parts.append(f"task-{task}")
    label    = "_".join(label_parts)

    group_dir = output_dir / f"group-{group_id}"
    fig_dir   = group_dir / "figures"
    sqm_dir   = group_dir / (f"ses-{session}" if session else "") / "nirs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sqm_dir.mkdir(parents=True, exist_ok=True)

    subject_ids  = [e.subject_id for e in group]
    first_raw    = aligned_raws.get(subject_ids[0]) if subject_ids else None
    markers_list = extract_markers(first_raw) if first_raw else []
    all_descs    = list(dict.fromkeys(m["description"] for m in markers_list))
    cond_colors_ = _cond_colors(all_descs)

    alignment_rows = [
        {
            "subject_id": sid,
            "offset_s":   round(offsets.get(sid, 0.0), 3),
            "duration_s": round(aligned_raws[sid].times[-1], 1)
                          if sid in aligned_raws else None,
        }
        for sid in subject_ids
    ]

    try:
        windowed_coh_df = compute_windowed_coherence(
            aligned_raws, coherence_fmin, coherence_fmax,
            window_s=coherence_window_s, step_s=coherence_step_s,
        )
    except Exception as exc:
        logger.warning("windowed coherence failed: %s", exc)
        windowed_coh_df = pd.DataFrame()

    figure_paths: dict = {}

    def _safe_save(name: str, desc: str, fn, *args) -> None:
        try:
            fig = fn(*args)
            if fig is None:
                return
            fname = f"{label}_desc-{desc}_nirs.html"
            h = _save_figure_html(fig, fig_dir / fname)
            figure_paths[name] = {"src": f"{group_dir.name}/figures/{fname}", "h": h}
        except Exception as exc:
            logger.warning("%s figure failed: %s", name, exc)

    if raw_raws:
        _safe_save("trigger_timeline_raw", "triggerraw",
                   build_trigger_timeline, raw_raws, subject_ids, "Time (s) [raw]")
    _safe_save("trigger_timeline", "triggeraligned",
               build_trigger_timeline, aligned_raws, subject_ids)
    _safe_save("coherence_bar",  "cohbar", build_coherence_bar, coherence_df)
    _safe_save("coh_timeseries", "cohts",  build_coherence_timeseries, windowed_coh_df)
    _safe_save("layout_2d",      "layout2d",
               build_layout_2d, aligned_raws, sqm_data, subject_ids, sci_threshold)
    _safe_save("layout_3d",      "layout3d",
               build_layout_3d, aligned_raws, sqm_data, subject_ids, sci_threshold)
    _safe_save("ch_summary",     "chsummary",
               build_channel_summary, sqm_data, subject_ids, sci_threshold)

    ch_pairs: list[str] = get_channel_pairs(first_raw) if first_raw else []
    for pair in ch_pairs:
        try:
            trace_fig = build_signal_overlay_pair(
                aligned_raws, subject_ids, pair, markers_list, cond_colors_,
            )
            psd_fig   = build_psd(
                aligned_raws, pair, subject_ids,
                cardiac=(cardiac_l_freq, cardiac_h_freq) if cardiac_l_freq is not None else None,
            )
            epoch_fig = build_epoch(
                aligned_raws, pair, subject_ids, epoch_tmin, epoch_tmax,
            )
            fname = f"{label}_desc-ch{_pair_fname(pair)}_nirs.html"
            _save_multi_fig_html(
                [trace_fig, psd_fig, epoch_fig], fig_dir / fname,
            )
        except Exception as exc:
            logger.warning("channel %s figure failed: %s", pair, exc)

    if ch_pairs:
        figure_paths["ch_detail_template"] = (
            f"{group_dir.name}/figures/{label}_desc-ch{{pair}}_nirs.html"
        )

    sqm = compute_hyper_sqm(
        sqm_data, coherence_df, aligned_raws, offsets, subject_ids, sci_threshold,
    )
    sqm_path = sqm_dir / f"{label}_desc-sqm_nirs.json"
    sqm_path.write_text(json.dumps(sqm, indent=2, default=str), encoding="utf-8")
    logger.info("Hyper SQM JSON → %s", sqm_path)

    return {
        "subject_ids":   subject_ids,
        "alignment":     alignment_rows,
        "sqm":           sqm,
        "ch_pairs":      ch_pairs,
        "figure_paths":  figure_paths,
        "data_subdir":   group_dir.name,
    }
