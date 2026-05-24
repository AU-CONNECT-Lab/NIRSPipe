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
)
from fnirs_pipe.qc.hyper_raw_writer import _process_hyper_raw_group
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_BASE_CSS     = (_TEMPLATE_DIR / "_base.css").read_text(encoding="utf-8")


def build_hyper_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    iqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    coherence_df: pd.DataFrame,
    output_dir: Path,
    raw_raws: dict[str, mne.io.Raw] | None = None,
    session: str | None = None,
    sci_threshold: float = 0.8,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    coherence_window_s: float = 30.0,
    coherence_step_s: float = 5.0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _process_hyper_raw_group(
        group_id=group_id, task=task, group=group,
        iqm_data=iqm_data, aligned_raws=aligned_raws, offsets=offsets,
        coherence_df=coherence_df, output_dir=output_dir,
        raw_raws=raw_raws, session=session, sci_threshold=sci_threshold,
        coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax,
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        coherence_window_s=coherence_window_s, coherence_step_s=coherence_step_s,
    )

    sci_per_subject = {
        sid: iqm_data.get(sid, {}).get("sci_per_channel", {})
        for sid in meta["subject_ids"]
    }

    name_parts = [f"group-{group_id}"]
    if session:
        name_parts.append(f"ses-{session}")
    name_parts.append(f"task-{task}")
    output_path = output_dir / ("_".join(name_parts) + "_desc-hyperraw_nirs.html")

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template("hyper_report.html.j2").render(
        base_css=_BASE_CSS,
        group_id=group_id,
        task=task,
        subject_ids=meta["subject_ids"],
        sci_threshold=sci_threshold,
        coherence_fmin=coherence_fmin,
        coherence_fmax=coherence_fmax,
        alignment_json=json.dumps(meta["alignment"]),
        ch_pairs_json=json.dumps(meta["ch_pairs"]),
        iqm_json=json.dumps(meta["iqm"], default=str),
        figure_paths=meta["figure_paths"],
        ch_detail_template_json=json.dumps(meta["figure_paths"].get("ch_detail_template")),
        sci_per_subject_json=json.dumps(sci_per_subject),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper raw report saved: %s", output_path)
    return output_path


def build_hyper_post_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    output_dir: Path,
    roi_map: dict[str, list[str]] | None = None,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    isc_threshold: float = 0.3,
) -> Path:
    """Build hyperscanning post-QC report.

    Sections:
      1. Per-channel WTC  — Morlet wavelet coherence, one heatmap per channel
      2. ISC matrix       — inter-brain Pearson r heatmap (channel × channel)
      3. ISC connectogram — inter-brain arcs filtered by isc_threshold
      TODO: ROI-level WTC — WTC averaged within anatomical ROIs (requires roi_map)
    """
    from fnirs_pipe.pipeline.hyperscanning import WTCResult, compute_wtc
    from fnirs_pipe.qc.figures.hyper_post_figures import (
        build_isc_panel,
        build_wtc_channel,
        compute_isc,
    )

    subject_ids  = [e.subject_id for e in group]
    ref_raw      = aligned_raws.get(subject_ids[0]) if subject_ids else None
    markers_list = _extract_markers(ref_raw) if ref_raw else []
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

    def _safe_post(name: str, fn, *args):
        try:
            fig = fn(*args)
            return fig.to_dict() if fig is not None else None
        except Exception as exc:
            logger.warning("%s figure failed: %s", name, exc)
            return None

    # Compute WTC
    wtc_result: WTCResult | None = None
    try:
        logger.info("Computing WTC for %d channels...", len(subject_ids))
        wtc_result = compute_wtc(aligned_raws, fmin=wtc_fmin, fmax=wtc_fmax)
    except Exception as exc:
        logger.warning("WTC computation failed: %s", exc)

    pair_key   = next(iter(wtc_result.pairs)) if wtc_result and wtc_result.pairs else None
    pair_label = f"{pair_key[0]} × {pair_key[1]}" if pair_key else ""

    # Build per-channel WTC figures
    ch_pairs_post: list[str] = []
    per_channel_post: dict[str, dict] = {}
    if ref_raw:
        hbo_picks = mne.pick_types(ref_raw.info, fnirs="hbo")
        for pick in hbo_picks:
            ch_name = ref_raw.ch_names[pick]
            pair    = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name
            if pair in per_channel_post:
                continue
            ch_pairs_post.append(pair)

            wtc_fig = None
            if wtc_result and pair_key:
                ch_data = wtc_result.pairs.get(pair_key, {}).get(pair)
                wtc_fig = _safe_post(
                    "wtc", build_wtc_channel,
                    ch_data, wtc_result.freqs, wtc_result.times,
                    pair_label, markers_list, cond_colors_,
                )
            per_channel_post[pair] = {"wtc": wtc_fig}

    # Compute ISC panels (HbO and HbR)
    def _isc_panel(ch_type: str) -> str:
        try:
            isc_mat, isc_ch_names = compute_isc(aligned_raws, subject_ids, ch_type)
            if isc_mat is None:
                return ""
            return build_isc_panel(
                isc_mat, isc_ch_names, subject_ids,
                ch_type=ch_type, isc_threshold=isc_threshold,
            )
        except Exception as exc:
            logger.warning("ISC panel (%s) failed: %s", ch_type, exc)
            return ""

    isc_panel_hbo_b64 = _isc_panel("hbo")
    isc_panel_hbr_b64 = _isc_panel("hbr")

    output_path = output_dir / f"group-{group_id}_task-{task}_hyper-post.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env  = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    html = env.get_template("hyper_post_report.html.j2").render(
        base_css=_BASE_CSS,
        group_id=group_id,
        task=task,
        subject_ids=subject_ids,
        wtc_fmin=wtc_fmin,
        wtc_fmax=wtc_fmax,
        isc_threshold=isc_threshold,
        alignment_json=json.dumps(alignment_rows),
        per_channel_post_json=json.dumps(per_channel_post),
        ch_pairs_post_json=json.dumps(ch_pairs_post),
        isc_panel_hbo_b64=isc_panel_hbo_b64,
        isc_panel_hbr_b64=isc_panel_hbr_b64,
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper post report saved: %s", output_path)
    return output_path
