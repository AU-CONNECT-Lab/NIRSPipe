"""Hyperscanning raw QC: build figure files + SQM JSON for the iframe-based viewer."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.io.derivatives import group_data_dir, group_report_dir
from fnirs_pipe.pipeline.hyperscanning import (
    GroupEntry, _hyper_sidecar, alignment_params,
)
from fnirs_pipe.qc.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.figures.hyper_figures import (
    _cond_colors,
    build_channel_summary,
    build_psd,
    build_signal_overlay_pair,
    build_trigger_timeline,
    compute_hyper_sqm,
)
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.qc.report_shell import guard, note
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_raw_writer")


def _write_coherence_tsv(
    df: pd.DataFrame,
    path: Path,
    step: str,
    aligned_raws: dict[str, mne.io.Raw],
    **params,
) -> None:
    """Write a coherence table beside the report, so its numbers can leave the report.

    The figure and the file are the same DataFrame, which is the point: a reader who wants
    the coherence of one channel should not have to hover a heatmap for it. An empty frame
    writes nothing, because a dyad the measure could not be taken on has no table.

    The alignment goes on the sidecar here rather than at the call sites, so a table added
    later cannot be written without it.
    """
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    _hyper_sidecar(path, step,
                   [p for p in (path_from(raw) for raw in aligned_raws.values()) if p],
                   **alignment_params(aligned_raws), **params)
    logger.info("coherence table -> %s", path)


def _hyper_sqm_record(sqm: dict, aligned_raws: dict[str, mne.io.Raw]) -> dict:
    """The dyad's quality record, with the provenance keys the graph reads.

    The keys sit in the file rather than in a sidecar beside it, the way
    :func:`~fnirs_pipe.qc.sqm_record.sqm_record_dict` puts them there: a sidecar for
    ``x.json`` would resolve to ``x.json`` itself. ``n_metrics`` is what marks the node as
    a QC record measured off the chain rather than a signal file on it, so the graph draws
    it without edges. The step name is its own: two objects sharing one is how a stage gets
    credited to the wrong file.

    ``alignment`` is nested rather than flattened in beside the scalars, which are what
    ``metrics`` is counted over: a clock is not a measurement of the dyad.
    """
    from fnirs_pipe import __version__

    metrics = [k for k, v in sqm.items() if isinstance(v, (int, float))]
    return {
        "pipeline_version": __version__,
        "step": "hyper_sqm",
        "Sources": [p for p in (path_from(raw) for raw in aligned_raws.values()) if p],
        "data": {"metrics": metrics, "n_metrics": len(metrics)},
        "alignment": alignment_params(aligned_raws),
        **sqm,
    }


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
    sci_threshold: float = SCI_PASS,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    errors: list | None = None,
    notes: list | None = None,
) -> dict:
    """Compute hyper raw figures, save each as a standalone HTML, write SQM JSON.

    Returns the metadata dict the main viewer HTML needs. A panel that fails goes into
    ``errors`` and a panel the dyad does not carry into ``notes``, both of which the
    report's footer prints, so a missing figure is visible in the report and not only in
    the run log.
    """
    errors = errors if errors is not None else []
    notes  = notes  if notes  is not None else []

    label_parts = [f"group-{group_id}"]
    if session:
        label_parts.append(f"ses-{session}")
    label_parts.append(f"task-{task}")
    label    = "_".join(label_parts)

    group_dir = group_report_dir(output_dir, group_id)
    fig_dir   = group_dir / "figures"
    sqm_dir   = group_data_dir(output_dir, group_id, session)
    fig_dir.mkdir(parents=True, exist_ok=True)

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

    with guard("Coherence table", errors, label):
        _write_coherence_tsv(
            coherence_df, sqm_dir / f"{label}_hyper-coherence.tsv",
            "hyper_coherence", aligned_raws,
            coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax)

    figure_paths: dict = {}

    def _safe_save(name: str, desc: str, fn, *args) -> None:
        with guard(f"{name} figure", errors, label):
            fig = fn(*args)
            if fig is None:
                return
            fname = f"{label}_desc-{desc}_nirs.html"
            h = _save_figure_html(fig, fig_dir / fname)
            figure_paths[name] = {"src": f"figures/{fname}", "h": h}

    if raw_raws:
        _safe_save("trigger_timeline_raw", "triggerraw",
                   build_trigger_timeline, raw_raws, subject_ids, "Time (s) [raw]")
    _safe_save("trigger_timeline", "triggeraligned",
               build_trigger_timeline, aligned_raws, subject_ids)
    _safe_save("ch_summary",     "chsummary",
               build_channel_summary, sqm_data, subject_ids, sci_threshold)

    ch_pairs: list[str] = get_channel_pairs(first_raw) if first_raw else []
    if not ch_pairs:
        note(notes, label, "no channel pairs on the aligned recordings: "
                           "the per-channel panel is empty")
    for pair in ch_pairs:
        with guard(f"Channel {pair}", errors, label):
            trace_fig = build_signal_overlay_pair(
                aligned_raws, subject_ids, pair, markers_list, cond_colors_,
            )
            psd_fig   = build_psd(
                aligned_raws, pair, subject_ids,
                cardiac=(cardiac_l_freq, cardiac_h_freq) if cardiac_l_freq is not None else None,
            )
            fname = f"{label}_desc-ch{_pair_fname(pair)}_nirs.html"
            # No epoch panel: this design gives one block per condition, so a "mean epoch"
            # would average a single trial and draw its first seconds as an evoked response.
            _save_multi_fig_html([trace_fig, psd_fig], fig_dir / fname)

    if ch_pairs:
        figure_paths["ch_detail_template"] = (
            f"figures/{label}_desc-ch{{pair}}_nirs.html"
        )

    sqm = compute_hyper_sqm(
        sqm_data, coherence_df, aligned_raws, offsets, subject_ids, sci_threshold,
    )
    sqm_path = sqm_dir / f"{label}_desc-sqm_nirs.json"
    sqm_path.write_text(json.dumps(_hyper_sqm_record(sqm, aligned_raws), indent=2,
                                  default=str), encoding="utf-8")
    logger.info("Hyper SQM JSON → %s", sqm_path)

    return {
        "subject_ids":   subject_ids,
        "label":         label,
        "sqm_dir":       sqm_dir,
        "alignment":     alignment_rows,
        "sqm":           sqm,
        "ch_pairs":      ch_pairs,
        "figure_paths":  figure_paths,
    }
