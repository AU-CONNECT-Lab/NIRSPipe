"""Generate hyperscanning group-level raw QC HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.pipeline.hyperscanning import GroupEntry
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
    build_signal_overlay,
    build_trigger_timeline,
    compute_hyper_iqm,
    compute_windowed_coherence,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def build_hyper_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    iqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    coherence_df: pd.DataFrame,
    output_dir: Path,
    sci_threshold: float = 0.8,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    coherence_window_s: float = 30.0,
    coherence_step_s: float = 5.0,
) -> Path:
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

    def _safe(name: str, fn, *args):
        try:
            fig = fn(*args)
            return fig.to_dict() if fig is not None else None
        except Exception as exc:
            logger.warning("%s figure failed: %s", name, exc)
            return None

    try:
        windowed_coh_df = compute_windowed_coherence(
            aligned_raws, coherence_fmin, coherence_fmax,
            window_s=coherence_window_s, step_s=coherence_step_s,
        )
    except Exception as exc:
        logger.warning("windowed coherence failed: %s", exc)
        windowed_coh_df = pd.DataFrame()

    figs = {
        "trigger_timeline": _safe("trigger_timeline", build_trigger_timeline,
                                  aligned_raws, subject_ids),
        "trace":            _safe("trace", build_signal_overlay,
                                  aligned_raws, subject_ids, markers_list, cond_colors_),
        "coherence_bar":    _safe("coherence_bar", build_coherence_bar, coherence_df),
        "coh_timeseries":   _safe("coh_timeseries", build_coherence_timeseries, windowed_coh_df),
        "layout_2d":        _safe("layout_2d", build_layout_2d,
                                  aligned_raws, iqm_data, subject_ids, sci_threshold),
        "layout_3d":        _safe("layout_3d", build_layout_3d,
                                  aligned_raws, iqm_data, subject_ids, sci_threshold),
        "ch_summary":       _safe("ch_summary", build_channel_summary,
                                  iqm_data, subject_ids, sci_threshold),
    }

    ch_pairs: list[str] = []
    per_channel: dict[str, dict] = {}
    if first_raw:
        hbo_picks = mne.pick_types(first_raw.info, fnirs="hbo")
        for pick in hbo_picks:
            ch_name = first_raw.ch_names[pick]
            pair    = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name
            if pair in per_channel:
                continue
            ch_pairs.append(pair)
            per_channel[pair] = {
                "psd":   _safe("psd", build_psd, aligned_raws, pair, subject_ids),
                "epoch": _safe("epoch", build_epoch,
                               aligned_raws, pair, subject_ids, epoch_tmin, epoch_tmax),
            }

    iqm = compute_hyper_iqm(
        iqm_data, coherence_df, aligned_raws, offsets, subject_ids, sci_threshold
    )

    output_path = output_dir / f"group-{group_id}_task-{task}_hyper-raw.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template("hyper_report.html.j2").render(
        group_id=group_id,
        task=task,
        subject_ids=subject_ids,
        sci_threshold=sci_threshold,
        coherence_fmin=coherence_fmin,
        coherence_fmax=coherence_fmax,
        alignment_json=json.dumps(alignment_rows),
        figures_json=json.dumps(figs),
        per_channel_json=json.dumps(per_channel),
        ch_pairs_json=json.dumps(ch_pairs),
        iqm_json=json.dumps(iqm),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper raw report saved: %s", output_path)
    return output_path
