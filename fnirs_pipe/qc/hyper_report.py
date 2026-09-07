"""Generate hyperscanning group-level raw QC HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.io.derivatives import (
    group_data_dir, group_report_dir, subject_report_dir,
)
from fnirs_pipe.pipeline.hyperscanning import GroupEntry
from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.boilerplate.vocabulary import metric_summary
from fnirs_pipe.qc.figure_io import extract_markers, get_channel_pairs
from fnirs_pipe.qc.figures.hyper_figures import _cond_colors
from fnirs_pipe.qc.hyper_raw_writer import _process_hyper_raw_group
from fnirs_pipe.qc.report_shell import footer_vars, guard, note, page_vars, render
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_report")


# ---- Per-subject quality metrics ----
#
# (key, label, better direction, format). Rendered in both hyper reports, which until now
# showed SCI and nothing else. Absent keys render as a dash, so one spec serves the raw
# path (intensity metrics only) and the post path (which adds motion and haemoglobin).
_SUBJECT_METRICS = [
    ("sci_mean",               "SCI",               "↑", "{:.3f}"),
    ("psp_mean",               "PSP",               "↑", "{:.2f}"),
    ("cv_mean",                "CV",                "↓", "{:.3f}"),
    ("snr_mean",               "SNR",               "↑", "{:.1f}"),
    ("gvtd_filt_p95",          "GVTD p95",          "↓", "{:.4f}"),
    ("motion_corrected_pct",   "Motion corrected",  "↓", "{:.3f}"),
    ("spike_pct_frames",       "Spike frames",      "↓", "{:.3f}"),
    ("hbo_hbr_corr_mean",      "HbO-HbR r",         "↓", "{:+.3f}"),
    ("channel_retention_rate", "Channels kept",     "↑", "{:.3f}"),
]


def _metric_class(key: str, value: float, sci_threshold: float) -> str:
    """Colour only where a threshold is actually justified.

    SCI has one the caller chose. HbO-HbR is judged by sign, because the two chromophores
    moving together says a shared artifact dominates the channel. The rest are shown plain:
    the published cut-offs (PSP 0.1, say) sit far below anything a real montage produces,
    so colouring by them would mark every run as passing.
    """
    if key == "sci_mean":
        return "qm-ok" if value >= sci_threshold else "qm-bad"
    if key == "hbo_hbr_corr_mean":
        return "qm-bad" if value >= 0 else "qm-ok"
    return ""


def subject_metric_rows(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> list[dict]:
    """One row per metric, one cell per subject, for the per-subject quality table."""
    rows = []
    for key, label, direction, fmt in _SUBJECT_METRICS:
        cells = []
        for sid in subject_ids:
            value = (sqm_data.get(sid) or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append({"text": fmt.format(value),
                              "cls": _metric_class(key, value, sci_threshold)})
            else:
                cells.append({"text": "—", "cls": ""})
        if any(c["text"] != "—" for c in cells):
            rows.append({"label": label, "direction": direction, "cells": cells,
                         "summary": metric_summary(key)})
    return rows


def write_isc_matrix(
    tsv_path: Path,
    isc_mat,
    ch_names: list[str],
    ch_type: str,
    sources: list[str],
    subject_ids: list[str],
) -> None:
    """Write the matrix the ISC panel is drawn from, so the numbers can leave the report.

    Both axes carry the first subject's channel labels, which is how :func:`compute_isc`
    pairs the two brains: cell (i, j) is that subject's channel i against the other's
    channel j. Rejected channels are blank rather than absent, so the file's shape is the
    montage's however many channels a given dyad lost.

    A failure here costs the file and not the panel: the report is still readable without it.
    """
    from fnirs_pipe.pipeline.hyperscanning import _hyper_sidecar

    try:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(isc_mat, index=ch_names, columns=ch_names).to_csv(
            tsv_path, sep="\t", index_label="channel")
        _hyper_sidecar(tsv_path, "hyper_isc", sources,
                       chromophore=ch_type, subjects=subject_ids)
        logger.info("ISC matrix saved: %s", tsv_path)
    except Exception as exc:
        logger.warning("ISC matrix (%s) not written: %s", ch_type, exc)


def _member_nirs_dir(output_dir: Path, entry: GroupEntry) -> Path:
    """A member's ``nirs/``, laid out the way :func:`group_data_dir` lays out a group's."""
    folder = subject_report_dir(output_dir, entry.subject_id)
    if entry.session:
        folder = folder / f"ses-{entry.session}"
    return folder / "nirs"


def group_methods(
    output_dir: Path,
    group: list[GroupEntry],
    group_nirs: Path,
    own_steps: list[tuple[str, dict]],
    versions: dict[str, str],
    notes: list,
    scope: str,
) -> dict[str, str]:
    """Methods prose for a dyad: a member's preprocessing, then the group's own steps.

    Three sources in the order a reader needs them: what was done to each recording (read
    from a member's sidecars), what was done to bring them onto one clock (``own_steps``,
    which leaves no file to scan), then what was measured across the two (read from the
    group's sidecars).

    The paragraph describes **one** preprocessing pipeline, because that is what a
    manuscript can use. When the members were not processed the same way, that becomes a
    note rather than a sentence naming both, since a Methods section hedging every
    parameter is worse than one that says which subject it describes.
    """
    from fnirs_pipe.qc.boilerplate import generate_methods_text
    from fnirs_pipe.qc.boilerplate.vocabulary import steps_from_sidecars

    per_member = {e.subject_id: steps_from_sidecars(_member_nirs_dir(output_dir, e))
                  for e in group}
    chains = list(per_member.values())
    if chains and any(chain != chains[0] for chain in chains[1:]):
        note(notes, scope,
             "the members of this group were not preprocessed identically, so the Methods "
             f"paragraph describes sub-{next(iter(per_member))} only")
    if not any(chains):
        note(notes, scope,
             "no preprocessing sidecars found for the members, so the Methods paragraph "
             "covers the group-level steps only")

    steps = list(chains[0]) if chains else []
    steps += own_steps
    steps += steps_from_sidecars(group_nirs)
    return generate_methods_text(versions=versions, steps=steps)


def build_hyper_report(
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    notes: list[str] = []

    meta = _process_hyper_raw_group(
        group_id=group_id, task=task, group=group,
        sqm_data=sqm_data, aligned_raws=aligned_raws, offsets=offsets,
        coherence_df=coherence_df, output_dir=output_dir,
        raw_raws=raw_raws, session=session, sci_threshold=sci_threshold,
        cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
        coherence_fmin=coherence_fmin, coherence_fmax=coherence_fmax,
        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
        coherence_window_s=coherence_window_s, coherence_step_s=coherence_step_s,
        errors=errors, notes=notes,
    )

    sci_per_subject = {
        sid: sqm_data.get(sid, {}).get("sci_per_channel", {})
        for sid in meta["subject_ids"]
    }

    from fnirs_pipe.qc.boilerplate.vocabulary import template_slots
    versions = collect_software_versions()
    # only the alignment is passed in: it leaves no file, so no sidecar describes it.
    # The coherence sentence comes off the table the writer just wrote.
    own_steps = [
        ("hyper_alignment", template_slots(
            "hyper_alignment", {"n_subjects": len(meta["subject_ids"])})),
    ]
    methods = group_methods(output_dir, group, meta["sqm_dir"], own_steps,
                            versions, notes, meta["label"])

    name_parts = [f"group-{group_id}"]
    if session:
        name_parts.append(f"ses-{session}")
    name_parts.append(f"task-{task}")
    output_path = (group_report_dir(output_dir, group_id)
                   / ("_".join(name_parts) + "_desc-hyperraw_nirs.html"))

    html = render(
        "hyper_report.html.j2",
        **page_vars(
            title=f"fnirs-pipe Hyper Raw Report — {group_id} / {task}",
            heading="fnirs‑pipe   Hyper Raw Report",
            nav_meta=[("group", group_id), ("task", task),
                      ("subjects", ", ".join(meta["subject_ids"]))],
            nav_note=(f"SCI thr: {sci_threshold:.2f} • "
                      f"Coh: {coherence_fmin:.3f}–{coherence_fmax:.3f} Hz"),
        ),
        **footer_vars(
            scope=meta["label"], errors=errors, notes=notes,
            nirs_dir=meta["sqm_dir"],
            methods=methods, versions=versions,
        ),
        group_id=group_id,
        task=task,
        subject_ids=meta["subject_ids"],
        sci_threshold=sci_threshold,
        coherence_fmin=coherence_fmin,
        coherence_fmax=coherence_fmax,
        alignment_json=json.dumps(meta["alignment"]),
        ch_pairs_json=json.dumps(meta["ch_pairs"]),
        sqm_json=json.dumps(meta["sqm"], default=str),
        figure_paths=meta["figure_paths"],
        ch_detail_template_json=json.dumps(meta["figure_paths"].get("ch_detail_template")),
        sci_per_subject_json=json.dumps(sci_per_subject),
        subject_metrics_rows=subject_metric_rows(
            sqm_data, meta["subject_ids"], sci_threshold),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper raw report saved: %s", output_path)
    return output_path


def condition_windows(
    raw: "mne.io.Raw",
    min_duration: float,
) -> "list[tuple[str, float, float]]":
    """[(label, tstart, tend)] on the aligned clock, one window per task annotation.

    e.g. annotations "Video" at 60 s for 240 s and "Talk" at 300 s for 240 s give
    ``[("Video", 60.0, 300.0), ("Talk", 300.0, 540.0)]``.

    An annotation's own ``duration`` is the window when it has one. Many acquisition systems
    write triggers with a duration of zero, so a window with none runs from its onset to the
    next annotation's, and the last to the end of the recording. Which rule each window came
    from is logged, and the tail of a zero-duration run is as long as the recording happens
    to be rather than as long as the block was.

    A description that occurs more than once gets one window per occurrence, numbered
    ``desc#1``, ``desc#2``. They are not spliced into a single record: a wavelet transform
    reads a join between two non-adjacent segments as a step, which lands in the result as
    broadband coherence at the join.

    ``min_duration`` drops windows shorter than that, in seconds. One cycle of the lowest
    frequency asked for is the sensible floor, and it is why an event-related design with
    two-second trials yields nothing here: no crop can carry a frequency whose period is
    longer than the crop.
    """
    from fnirs_pipe.qc.figure_io import extract_markers

    markers = sorted(extract_markers(raw), key=lambda m: m["onset"])
    if not markers:
        return []

    end = float(raw.times[-1])
    counts: dict[str, int] = {}
    for m in markers:
        counts[m["description"]] = counts.get(m["description"], 0) + 1
    seen: dict[str, int] = {}

    windows: list[tuple[str, float, float]] = []
    n_from_duration = 0
    for i, m in enumerate(markers):
        onset = float(m["onset"])
        if float(m["duration"]) > 0:
            stop = onset + float(m["duration"])
            n_from_duration += 1
        else:
            stop = float(markers[i + 1]["onset"]) if i + 1 < len(markers) else end
        stop = min(stop, end)

        desc = m["description"]
        if counts[desc] > 1:
            seen[desc] = seen.get(desc, 0) + 1
            label = f"{desc}#{seen[desc]}"
        else:
            label = desc

        if stop - onset < min_duration:
            logger.info("condition %s spans %.1fs, under the %.1fs floor: skipped",
                        label, stop - onset, min_duration)
            continue
        windows.append((label, onset, stop))

    logger.info("condition windows: %d kept, %d took their own duration and %d ran to the "
                "next trigger", len(windows), n_from_duration, len(markers) - n_from_duration)
    return windows


def build_hyper_post_report(
    group_id: str,
    task: str,
    group: list[GroupEntry],
    aligned_raws: dict[str, mne.io.Raw],
    offsets: dict[str, float],
    output_dir: Path,
    roi_map: dict[str, list[str]] | None = None,
    bad_channels: dict[str, list[str]] | None = None,
    subject_sqm: dict[str, dict] | None = None,
    wtc_fmin: float = 0.004,
    wtc_fmax: float = 0.20,
    wtc_band_fmin: float | None = None,
    wtc_band_fmax: float | None = None,
    wtc_significance: bool = False,
    wtc_seed: int | None = None,
    wtc_mc_count: int = 300,
    wtc_channel_cross: bool = False,
    wtc_by_condition: bool = False,
    wtc_limit_scales: bool = True,
    wtc_save_maps: bool = False,
    wtc_mask_coi: bool = False,
    wtc_roi_min_channels: int = 2,
    isc_threshold: float = 0.3,
    sci_threshold: float = 0.8,
) -> Path:
    """Build hyperscanning post-QC report.

    Sections:
      1. Per-channel WTC  — Morlet wavelet coherence, one heatmap per channel
      2. Per-ROI WTC      — the member channels' maps averaged cell by cell (when roi_map given)
      3. ISC matrix       — inter-brain Pearson r heatmap (channel × channel)
      4. ISC connectogram — inter-brain arcs filtered by isc_threshold

    Each WTC map is also collapsed to one number per channel over
    [wtc_band_fmin, wtc_band_fmax] and written as a TSV under the group's nirs/, so a
    group analysis reads the same values the figures were drawn from.

    ``wtc_channel_cross`` crosses every long channel with every other, 14 channels giving 196
    rows in ``hyper-wtc.tsv`` instead of 14. The extra pairs reach the TSV, the heatmap
    selector keeps the homologous ones: every map carried in the page is a full
    frequency × time array and a report embedding all of them would be too large to open.
    Crossing is also what produces the ROI × ROI matrix, since the ROI numbers are grouped
    from the channel ones.

    ``wtc_by_condition`` repeats the whole coherence analysis inside each task annotation's
    own window, on top of the whole-run pass, which stays as it was. The band means of every
    window land in one ``hyper-wtcbycond.tsv`` with a ``condition`` column, and each window
    gets its own figures. See :func:`condition_windows` for how a window is decided, and note
    that the runtime is roughly doubled: the windows together are about one more pass over
    the recording.

    ``wtc_mask_coi`` restricts each band mean to the cone of influence. Off by default; the
    share inside the cone is reported either way as ``n_valid_frac``.

    ``wtc_save_maps`` writes the full time-frequency maps beside each TSV as ``.npz``, so a
    different band can be averaged later without a second wavelet transform. See
    :mod:`fnirs_pipe.qc.wtc_store`. ``wtc_limit_scales`` computes only the scales inside
    ``[wtc_fmin, wtc_fmax]`` plus margin, which is most of the runtime and, given that the
    scales land on pycwt's own grid and the margin exceeds its scale-smoothing window,
    reproduces the unrestricted coherences bit for bit.
    """
    from fnirs_pipe.pipeline.hyperscanning import (
        WTCResult,
        _hyper_sidecar,
        compute_wtc,
        crop_aligned_window,
        roi_maps_from_channels,
        roi_mean_of_channels,
        wtc_band_mean,
    )
    from fnirs_pipe.qc.figures.hyper_post_figures import (
        build_isc_panel,
        build_wtc_channel,
        build_wtc_cross_matrix,
        build_wtc_roi_grid,
        build_wtc_roi_matrix,
        compute_isc,
    )

    errors: list[str] = []
    notes: list[str] = []
    scope = f"group-{group_id}_task-{task}"

    subject_ids  = [e.subject_id for e in group]
    ref_raw      = aligned_raws.get(subject_ids[0]) if subject_ids else None
    markers_list = extract_markers(ref_raw) if ref_raw else []
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

    band_fmin = wtc_band_fmin if wtc_band_fmin is not None else wtc_fmin
    band_fmax = wtc_band_fmax if wtc_band_fmax is not None else wtc_fmax
    if wtc_band_fmin is None or wtc_band_fmax is None:
        logger.info(
            "WTC band mean taken over the whole %.4f-%.4f Hz axis; name a narrower band "
            "to average over the frequencies the task lives in", band_fmin, band_fmax)

    def _write_band_tsv(result: "WTCResult | None", kind: str, step: str):
        if result is None or not result.pairs:
            return None
        df = None
        with guard(f"WTC band mean ({kind})", errors, scope):
            df = wtc_band_mean(result, band_fmin, band_fmax, mask_coi=wtc_mask_coi)
        if df is None:
            return None
        tsv_path = _write_df_tsv(df, kind, step)
        if wtc_save_maps:
            from fnirs_pipe.qc.wtc_store import save_wtc
            with guard(f"Saving WTC maps ({kind})", errors, scope):
                save_wtc(result, tsv_path.with_suffix(".npz"))
        logger.info("WTC band means saved: %s", tsv_path)
        return df

    def _write_df_tsv(df, kind: str, step: str, **extra) -> Path:
        tsv_path = (group_data_dir(output_dir, group_id)
                    / f"group-{group_id}_task-{task}_hyper-{kind}.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        _hyper_sidecar(
            tsv_path, step,
            [p for p in (path_from(r) for r in aligned_raws.values()) if p],
            band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=wtc_mask_coi,
            wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, **extra,
        )
        return tsv_path

    def _safe_post(name: str, fn, *args):
        out = None
        with guard(f"{name} figure", errors, scope):
            fig = fn(*args)
            out = fig.to_dict() if fig is not None else None
        return out

    # Compute WTC
    wtc_result: WTCResult | None = None
    with guard("WTC computation", errors, scope):
        if wtc_significance:
            logger.warning("WTC significance on: %d Monte Carlo surrogates per channel pair, "
                           "this is slow.", wtc_mc_count)
        if wtc_channel_cross:
            logger.warning("WTC channel crossing on: every long channel against every "
                           "other, so the pair count is squared and so is the runtime.")
        logger.info("Computing WTC for %d subjects...", len(subject_ids))
        wtc_result = compute_wtc(
            aligned_raws, fmin=wtc_fmin, fmax=wtc_fmax, significance=wtc_significance,
            seed=wtc_seed, mc_count=wtc_mc_count, cross=wtc_channel_cross,
            limit_scales=wtc_limit_scales)

    chan_band_df = _write_band_tsv(wtc_result, "wtc", "hyper_wtc")

    pair_key   = next(iter(wtc_result.pairs)) if wtc_result and wtc_result.pairs else None
    pair_label = f"{pair_key[0]} × {pair_key[1]}" if pair_key else ""

    chan_matrix_b64 = ""
    if wtc_channel_cross and chan_band_df is not None:
        with guard("WTC channel cross matrix", errors, scope):
            chan_labels = sorted({*chan_band_df["label"], *chan_band_df["label2"]})
            chan_matrix_b64 = build_wtc_cross_matrix(
                chan_band_df, chan_labels, subject_ids, band_fmin, band_fmax,
                kind="channel") or ""

    # Build per-channel WTC figures
    ch_pairs_post: list[str] = get_channel_pairs(ref_raw) if ref_raw else []
    per_channel_post: dict[str, dict] = {}
    for pair in ch_pairs_post:
        wtc_fig = None
        if wtc_result and pair_key:
            # crossing keys every pair, so the homologous one is the (ch, ch) cell
            ch_key  = (pair, pair) if wtc_channel_cross else pair
            ch_data = wtc_result.pairs.get(pair_key, {}).get(ch_key)
            wtc_fig = _safe_post(
                "wtc", build_wtc_channel,
                ch_data, wtc_result.freqs, wtc_result.times,
                pair_label, markers_list, cond_colors_,
            )
        per_channel_post[pair] = {"wtc": wtc_fig}

    # Compute ISC panels (HbO and HbR)
    def _isc_panel(ch_type: str) -> str:
        panel = ""
        with guard(f"ISC panel ({ch_type})", errors, scope):
            isc_mat, isc_ch_names = compute_isc(
                aligned_raws, subject_ids, ch_type, bad_channels=bad_channels
            )
            if isc_mat is None:
                return ""
            write_isc_matrix(
                group_data_dir(output_dir, group_id)
                / f"group-{group_id}_task-{task}_hyper-isc-{ch_type}.tsv",
                isc_mat, isc_ch_names, ch_type,
                [p for p in (path_from(r) for r in aligned_raws.values()) if p],
                subject_ids,
            )
            panel = build_isc_panel(
                isc_mat, isc_ch_names, subject_ids,
                ch_type=ch_type, isc_threshold=isc_threshold,
            )
        return panel

    isc_panel_hbo_b64 = _isc_panel("hbo")
    isc_panel_hbr_b64 = _isc_panel("hbr")

    roi_rows: list[dict] = []
    roi_labels: list[str] = []
    per_roi_post: dict[str, dict] = {}
    roi_matrix_fig = None
    roi_grid_b64 = ""
    if roi_map:
        assigned = {ch for chs in roi_map.values() for ch in chs}
        for roi_name, chs in roi_map.items():
            roi_rows.append({"roi": roi_name, "channels": chs})
        unassigned = [c for c in ch_pairs_post if c not in assigned]
        if unassigned:
            roi_rows.append({"roi": "Unassigned", "channels": unassigned})

        # the ROI number the WTC literature reports: coherence per channel pair, then averaged
        roi_band_df = None
        if chan_band_df is not None:
            with guard("ROI mean of channel WTC", errors, scope):
                roi_band_df = roi_mean_of_channels(
                    chan_band_df, roi_map, min_channels=wtc_roi_min_channels)
                path = _write_df_tsv(roi_band_df, "wtc-roichan", "hyper_wtc_roichan")
                logger.info("WTC ROI means from channels saved: %s", path)

        # the maps grouped the same way, so the picture and the table are the same average
        roi_wtc: WTCResult | None = None
        if wtc_result is not None:
            with guard("ROI WTC maps from channels", errors, scope):
                roi_wtc = roi_maps_from_channels(wtc_result, roi_map)

        roi_pair_key = next(iter(roi_wtc.pairs)) if roi_wtc and roi_wtc.pairs else None
        roi_labels   = list(roi_map.keys())
        if roi_band_df is not None and wtc_channel_cross:
            roi_matrix_fig = _safe_post(
                "wtc-roi-matrix", build_wtc_roi_matrix,
                roi_band_df, roi_labels, subject_ids, band_fmin, band_fmax,
            )
        # drawn crossed or not: it is the only figure carrying the phase arrows
        with guard("WTC ROI map grid", errors, scope):
            roi_grid_b64 = build_wtc_roi_grid(
                roi_wtc, roi_labels, roi_pair_key, subject_ids) or ""
        for roi_name in roi_labels:
            roi_fig = None
            if roi_wtc and roi_pair_key:
                # crossing keys every pair, so the homologous one is the (roi, roi) cell
                roi_key  = (roi_name, roi_name) if wtc_channel_cross else roi_name
                roi_data = roi_wtc.pairs.get(roi_pair_key, {}).get(roi_key)
                roi_fig = _safe_post(
                    "wtc-roichan", build_wtc_channel,
                    roi_data, roi_wtc.freqs, roi_wtc.times,
                    pair_label, markers_list, cond_colors_,
                )
            per_roi_post[roi_name] = {"wtc": roi_fig}

    # ---- per condition ----
    # The whole-run pass above stands; this adds the same analysis inside each task window,
    # which is the comparison a block design is run for.
    condition_figs: list[dict] = []
    if wtc_by_condition:
        windows = condition_windows(ref_raw, min_duration=1.0 / wtc_fmin) if ref_raw else []
        if not windows:
            note(notes, scope,
                 "--wtc-by-condition asked for, but no annotation window is long enough "
                 f"for one cycle of {wtc_fmin:.4f} Hz: no per-condition figures")
        else:
            logger.info("--wtc-by-condition: %d window(s), each a further WTC pass",
                        len(windows))
        chan_frames, roi_frames = [], []
        for label, tstart, tstop in windows:
            cond_chan = None
            with guard(f"Condition {label}: WTC", errors, scope):
                cropped = crop_aligned_window(aligned_raws, tstart, tstop)
                cond_wtc = compute_wtc(
                    cropped, fmin=wtc_fmin, fmax=wtc_fmax, significance=wtc_significance,
                    seed=wtc_seed, mc_count=wtc_mc_count, cross=wtc_channel_cross,
                    limit_scales=wtc_limit_scales)
                cond_chan = wtc_band_mean(cond_wtc, band_fmin, band_fmax,
                                          mask_coi=wtc_mask_coi)
            if cond_chan is None:
                continue

            cond_chan.insert(0, "condition", label)
            chan_frames.append(cond_chan)

            entry: dict = {"label": label, "tstart": round(tstart, 1),
                           "tstop": round(tstop, 1), "matrix": "", "grid": ""}
            if wtc_channel_cross:
                with guard(f"Condition {label}: channel matrix", errors, scope):
                    ch_labels = sorted({*cond_chan["label"], *cond_chan["label2"]})
                    entry["matrix"] = build_wtc_cross_matrix(
                        cond_chan, ch_labels, subject_ids, band_fmin, band_fmax,
                        kind="channel") or ""

            if roi_map:
                with guard(f"Condition {label}: ROI means", errors, scope):
                    cond_roi = roi_mean_of_channels(
                        cond_chan.drop(columns="condition"), roi_map,
                        min_channels=wtc_roi_min_channels)
                    cond_roi.insert(0, "condition", label)
                    roi_frames.append(cond_roi)
                with guard(f"Condition {label}: ROI grid", errors, scope):
                    cond_roi_wtc = roi_maps_from_channels(cond_wtc, roi_map)
                    key = next(iter(cond_roi_wtc.pairs), None)
                    entry["grid"] = build_wtc_roi_grid(
                        cond_roi_wtc, list(roi_map), key, subject_ids) or ""

            condition_figs.append(entry)

        # the windows are the one thing a reader cannot reconstruct from the table
        spans = {label: [round(t0, 3), round(t1, 3)] for label, t0, t1 in windows}
        if chan_frames:
            path = _write_df_tsv(pd.concat(chan_frames, ignore_index=True),
                                 "wtcbycond", "hyper_wtc_bycondition",
                                 condition_windows_s=spans)
            logger.info("WTC band means per condition saved: %s", path)
        if roi_frames:
            path = _write_df_tsv(pd.concat(roi_frames, ignore_index=True),
                                 "wtcbycond-roichan", "hyper_wtc_bycondition_roichan",
                                 condition_windows_s=spans)
            logger.info("WTC ROI means per condition saved: %s", path)

    bad_pairs_all: set[str] = set()
    if bad_channels:
        for chs in bad_channels.values():
            bad_pairs_all |= {c.rsplit(" ", 1)[0] for c in chs}

    output_path = (group_report_dir(output_dir, group_id)
                   / f"group-{group_id}_task-{task}_hyper-post.html")

    # Rendered here rather than by the caller: every sidecar the scan reads was written by
    # the passes above, so this is the first moment the graph is complete. Same stem
    # `fnirs-qc provenance` uses, so re-running that refreshes the image this report links.
    provenance_path = None
    with guard("Provenance diagram", errors, scope):
        from fnirs_pipe.qc.provenance import write_provenance

        for written in write_provenance(
            group_data_dir(output_dir, group_id),
            group_report_dir(output_dir, group_id) / "figures",
            stem="provenance", title=f"group-{group_id}_task-{task}",
        ):
            if written.suffix == ".png":
                provenance_path = f"figures/{written.name}"

    from fnirs_pipe.qc.boilerplate.vocabulary import template_slots
    versions = collect_software_versions()
    own_steps = [("hyper_alignment", template_slots(
        "hyper_alignment", {"n_subjects": len(subject_ids)}))]
    methods = group_methods(output_dir, group, group_data_dir(output_dir, group_id),
                            own_steps, versions, notes, scope)

    html = render(
        "hyper_post_report.html.j2",
        **page_vars(
            title=f"fnirs-pipe Hyper Post Report — {group_id} / {task}",
            heading="fnirs‑pipe   Hyper Post Report",
            nav_meta=[("group", group_id), ("task", task),
                      ("subjects", ", ".join(subject_ids))],
            nav_note=f"WTC: {wtc_fmin:.3f}–{wtc_fmax:.3f} Hz",
        ),
        **footer_vars(
            scope=f"group-{group_id}_task-{task}", errors=errors, notes=notes,
            nirs_dir=group_data_dir(output_dir, group_id),
            provenance_path=provenance_path,
            methods=methods, versions=versions,
        ),
        group_id=group_id,
        task=task,
        subject_ids=subject_ids,
        wtc_fmin=wtc_fmin,
        wtc_fmax=wtc_fmax,
        isc_threshold=isc_threshold,
        alignment_json=json.dumps(alignment_rows),
        per_channel_post_json=json.dumps(per_channel_post),
        ch_pairs_post_json=json.dumps(ch_pairs_post),
        bad_pairs_json=json.dumps(sorted(bad_pairs_all)),
        isc_panel_hbo_b64=isc_panel_hbo_b64,
        isc_panel_hbr_b64=isc_panel_hbr_b64,
        roi_rows=roi_rows,
        roi_labels_json=json.dumps(roi_labels),
        per_roi_post_json=json.dumps(per_roi_post),
        wtc_roi_matrix_json=json.dumps(roi_matrix_fig),
        wtc_chan_matrix_b64=chan_matrix_b64,
        wtc_roi_grid_b64=roi_grid_b64,
        condition_figs=condition_figs,
        subject_metrics_rows=subject_metric_rows(
            subject_sqm or {}, subject_ids, sci_threshold),
    )
    output_path.write_text(html, encoding="utf-8")
    logger.info("Hyper post report saved: %s", output_path)
    return output_path
