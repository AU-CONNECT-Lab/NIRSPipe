"""Hyperscanning raw QC: build figure files + IQM JSON for the iframe-based viewer."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.pipeline.hyperscanning import GroupEntry
from fnirs_pipe.qc.figure_io import _pair_fname, _save_figure_html, _save_multi_fig_html
from fnirs_pipe.qc.figures.hyper_figures import (
    _cond_colors,
    _extract_markers,
    build_channel_summary,
    build_coherence_bar,
    build_coherence_timeseries,
    build_epoch,
    build_layout_2d,
    build_layout_3d,
    build_psd,
    build_signal_overlay_pair,
    build_trigger_timeline,
    compute_hyper_iqm,
    compute_windowed_coherence,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_raw_writer")


def _process_hyper_raw_group(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    iqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    coherence_df: pd.DataFrame,
    output_dir: Path,
    raw_raws: dict[str, mne.io.Raw] | None = None,
    sci_threshold: float = 0.8,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    coherence_window_s: float = 30.0,
    coherence_step_s: float = 5.0,
) -> dict:
    """Compute hyper raw figures, save each as a standalone HTML, write IQM JSON.
    Returns the metadata dict the main viewer HTML needs."""

    label    = f"group-{group_id}_task-{task}"
    data_dir = output_dir / f"{label}_desc-hyperraw_nirs"
    fig_dir  = data_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    subject_ids  = [e.subject_id for e in group]
    first_raw    = aligned_raws.get(subject_ids[0]) if subject_ids else None
    markers_list = _extract_markers(first_raw) if first_raw else []
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
            figure_paths[name] = {"src": f"{data_dir.name}/figures/{fname}", "h": h}
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
               build_layout_2d, aligned_raws, iqm_data, subject_ids, sci_threshold)
    _safe_save("layout_3d",      "layout3d",
               build_layout_3d, aligned_raws, iqm_data, subject_ids, sci_threshold)
    _safe_save("ch_summary",     "chsummary",
               build_channel_summary, iqm_data, subject_ids, sci_threshold)

    ch_pairs: list[str] = []
    if first_raw:
        hbo_picks = mne.pick_types(first_raw.info, fnirs="hbo")
        seen: set[str] = set()
        for pick in hbo_picks:
            ch_name = first_raw.ch_names[pick]
            pair    = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name
            if pair in seen:
                continue
            seen.add(pair)
            ch_pairs.append(pair)

            try:
                trace_fig = build_signal_overlay_pair(
                    aligned_raws, subject_ids, pair, markers_list, cond_colors_,
                )
                psd_fig   = build_psd(aligned_raws, pair, subject_ids)
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
            f"{data_dir.name}/figures/{label}_desc-ch{{pair}}_nirs.html"
        )

    iqm = compute_hyper_iqm(
        iqm_data, coherence_df, aligned_raws, offsets, subject_ids, sci_threshold,
    )
    iqm_path = data_dir / f"{label}_desc-iqm_nirs.json"
    iqm_path.write_text(json.dumps(iqm, indent=2, default=str), encoding="utf-8")
    logger.info("Hyper IQM JSON → %s", iqm_path)

    return {
        "subject_ids":   subject_ids,
        "alignment":     alignment_rows,
        "iqm":           iqm,
        "ch_pairs":      ch_pairs,
        "figure_paths":  figure_paths,
        "data_subdir":   data_dir.name,
    }
