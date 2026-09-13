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
from fnirs_pipe.qc.common.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html, get_channel_pairs,
)
from fnirs_pipe.qc.figures.hyper_figures import (
    _cond_colors,
    build_alignment_timeline,
    build_channel_summary,
    build_head_by_condition,
    build_head_slider,
    build_motion_panel,
    build_psd,
    build_screening_strip,
    build_signal_overlay_pair,
    build_usable_time,
    head_geometry,
    motion_series,
)
from fnirs_pipe.qc.metrics.hyper import (
    compute_hyper_sqm, coupled_grid, member_series, motion_summary, screening_summary,
)
from fnirs_pipe.pipeline.synchrony import SCREEN_NULL_ITER, screening_coherence
from fnirs_pipe.qc.hyper.hyper_usable import usable_scalars, write_usable_table
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.qc.common.report_shell import guard, note
from fnirs_pipe.qc.common.windows import condition_windows, markers_on_data_axis
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
    :func:`~fnirs_pipe.qc.subject.sqm_record.sqm_record_dict` puts them there: a sidecar for
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


def _condition_spans(raw: "mne.io.Raw | None") -> dict:
    """``{block: (start, stop)}`` on the shared clock, by the rule the post report uses.

    The dict is what every panel that splits by condition reads, so none of them can be
    drawn against a different set of blocks. The rule behind it is
    :func:`~fnirs_pipe.qc.common.windows.condition_windows` rather than a second copy of it here.
    The copy took each annotation's own duration, which on a system that writes
    zero-duration triggers gave every block a zero-length span, and keyed the dict on the
    bare description, which kept only the last occurrence of a repeated one.

    ``min_duration`` is 0: these windows only split panels, so no frequency has to fit
    inside one and nothing is dropped for being short.
    """
    if raw is None:
        return {}
    return {label: (t0, t1) for label, t0, t1 in condition_windows(raw, min_duration=0.0)}


def _process_hyper_raw_group(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    sqm_data: dict[str, dict],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    output_dir: Path,
    raw_raws: dict[str, mne.io.Raw] | None = None,
    intensity_raws: dict[str, mne.io.Raw] | None = None,
    after_raws: dict[str, mne.io.Raw] | None = None,
    session: str | None = None,
    sci_threshold: float = SCI_PASS,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    coherence_fmin: float = 0.01,
    coherence_fmax: float = 0.10,
    sep_bands=None,
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
    markers_list = markers_on_data_axis(first_raw) if first_raw else []
    all_descs    = list(dict.fromkeys(m["description"] for m in markers_list))
    cond_colors_ = _cond_colors(all_descs)

    conditions = _condition_spans(first_raw)
    if not conditions:
        note(notes, label, "no annotated blocks on the aligned recordings: the panels that "
                           "split by condition are drawn over the whole run instead")

    alignment_rows = [
        {
            "subject_id": sid,
            "offset_s":   round(offsets.get(sid, 0.0), 3),
            "duration_s": round(aligned_raws[sid].times[-1], 1)
                          if sid in aligned_raws else None,
        }
        for sid in subject_ids
    ]

    # the lines this run screened by, read off the members rather than re-resolved, so the
    # rules drawn on the series are the ones the carpet under them was masked at
    cutoffs = next((dict((sqm_data.get(sid) or {}).get("screen_cutoffs") or {})
                    for sid in subject_ids
                    if (sqm_data.get(sid) or {}).get("screen_cutoffs")), {})

    screening_df = pd.DataFrame()
    with guard("Screening coherence", errors, label):
        screening_df = screening_coherence(
            aligned_raws, fmin=coherence_fmin, fmax=coherence_fmax,
            windows=[(name, a, b) for name, (a, b) in conditions.items()],
            sep_bands=sep_bands)
    # The whole-run rows are the plain pairwise coherence, so the table and the record read
    # them out of the pass that already measured them rather than measuring a second time.
    # Two passes meant two copies of the segment-length rule, and a value on this page is
    # compared with its own null: they have to be the same estimate.
    coherence_df = (screening_df[screening_df["window"] == "whole run"]
                    [["ch_name", "sub1", "sub2", "coherence"]].reset_index(drop=True)
                    if not screening_df.empty else pd.DataFrame(
                        columns=["ch_name", "sub1", "sub2", "coherence"]))

    with guard("Coherence tables", errors, label):
        _write_coherence_tsv(
            coherence_df, sqm_dir / f"{label}_hyper-coherence.tsv",
            "hyper_coherence", aligned_raws,
            coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax)
        _write_coherence_tsv(
            screening_df, sqm_dir / f"{label}_hyper-screening.tsv",
            "hyper_screening", aligned_raws,
            coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax,
            n_iter=SCREEN_NULL_ITER, null="phase_scramble")

    figure_paths: dict = {}

    def _safe_save(name: str, desc: str, fn, *args) -> None:
        with guard(f"{name} figure", errors, label):
            fig = fn(*args)
            if fig is None:
                return
            fname = f"{label}_desc-{desc}_nirs.html"
            h = _save_figure_html(fig, fig_dir / fname)
            figure_paths[name] = {"src": f"figures/{fname}", "h": h}

    _safe_save("alignment_timeline", "alignment",
               build_alignment_timeline, raw_raws, aligned_raws, subject_ids)
    _safe_save("ch_summary", "chsummary",
               build_channel_summary, sqm_data, subject_ids, sci_threshold)

    # motion is measured off the optical density, not off the aligned haemoglobin the rest
    # of the page is drawn on, so it is its own pass rather than a row on the screening grid
    motion, motion_scalars = {}, {}
    with guard("Motion panel", errors, label):
        motion = motion_series(intensity_raws or {}, after_raws, subject_ids, sep_bands)
    if not motion:
        note(notes, label, "no optical density for the members: the motion panels "
                           "are empty")
    else:
        motion_scalars = motion_summary(motion)
        for stage in motion["stages"]:
            with guard(f"Motion panel ({stage})", errors, label):
                fig = build_motion_panel(motion, stage, conditions)
                if fig is None:
                    continue
                fname = f"{label}_desc-motion{stage}_nirs.html"
                h = _save_figure_html(fig, fig_dir / fname)
                figure_paths[f"motion_{stage}"] = {"src": f"figures/{fname}", "h": h}
        if "after" not in motion["stages"]:
            note(notes, label, "no motion-corrected derivative lines up with both members: "
                               "the panel shows the recording as it arrived only")

    duration_s = first_raw.times[-1] if first_raw is not None else None
    grid, series, geo = None, {}, {}
    with guard("Screening grid", errors, label):
        grid = coupled_grid(sqm_data, subject_ids, offsets, duration_s=duration_s)
    if grid is None:
        note(notes, label, "no shared screening grid: the members were screened on "
                           "different window grids, or their records carry none. The "
                           "usable-time and head panels are empty")
    else:
        series = {sid: member_series(sqm_data, sid, grid, offsets.get(sid, 0.0))
                  for sid in subject_ids}
        geo = {sid: head_geometry(aligned_raws[sid],
                                  grid.get("long_pairs") or grid["pairs"])
               for sid in subject_ids if sid in aligned_raws}
        geo = {sid: g for sid, g in geo.items() if g is not None}
        if not geo:
            note(notes, label, "no optode positions on either member: the head panels "
                               "are empty")
        _safe_save("usable_time", "usable", build_usable_time,
                   grid, subject_ids, series, conditions, cutoffs)
        if geo:
            _safe_save("head_by_condition", "headcond", build_head_by_condition,
                       geo, subject_ids, grid, conditions)
            _safe_save("head_slider", "headslider", build_head_slider,
                       geo, subject_ids, grid, series, conditions)

    _safe_save("screening_strip", "screening", build_screening_strip, screening_df)

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
    # the screening verdict beside the measured coherence, since the value alone is not
    # readable: see `screening_summary`
    sqm["screening"] = screening_summary(screening_df)
    sqm["motion"] = motion_scalars
    if grid is not None:
        sqm.update(usable_scalars(grid, subject_ids))
        with guard("Usable-time table", errors, label):
            write_usable_table(
                sqm_dir / f"{label}_hyper-usable.tsv", grid, subject_ids, conditions,
                sources=[p for p in (path_from(raw) for raw in aligned_raws.values()) if p],
                sci_threshold=sci_threshold)
    sqm_path = sqm_dir / f"{label}_desc-sqm_nirs.json"
    sqm_path.write_text(json.dumps(_hyper_sqm_record(sqm, aligned_raws), indent=2,
                                  default=str), encoding="utf-8")
    logger.info("Hyper SQM JSON → %s", sqm_path)

    member_info = [
        {
            "subject_id": sid,
            "sfreq": round(float(aligned_raws[sid].info["sfreq"]), 4)
                     if sid in aligned_raws else None,
            "duration_s": round(float(aligned_raws[sid].times[-1]), 1)
                          if sid in aligned_raws else None,
            "n_pairs": len(get_channel_pairs(aligned_raws[sid]))
                       if sid in aligned_raws else 0,
            "n_long": len((grid or {}).get("pairs") or []),
        }
        for sid in subject_ids
    ]

    return {
        "subject_ids":   subject_ids,
        "label":         label,
        "sqm_dir":       sqm_dir,
        "alignment":     alignment_rows,
        "conditions":    {k: [round(a, 2), round(b, 2)]
                          for k, (a, b) in conditions.items()},
        "member_info":   member_info,
        "sqm":           sqm,
        "ch_pairs":      ch_pairs,
        "figure_paths":  figure_paths,
    }
