"""Generate hyperscanning group-level raw QC HTML report."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.io.derivatives import (
    group_data_dir, group_report_dir, subject_report_dir,
)
from fnirs_pipe.pipeline.hyperscanning import GroupEntry, unfiltered_stage_note
from fnirs_pipe.qc.boilerplate import collect_software_versions
from fnirs_pipe.qc.boilerplate.vocabulary import (
    MISSING_VALUE, format_metric, metric_class, metric_direction, metric_label,
    metric_summary,
)
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.qc.figure_io import (
    _pair_fname, extract_markers, get_channel_pairs, save_png,
)
from fnirs_pipe.qc.figures.hyper_figures import _cond_colors
from fnirs_pipe.qc.hyper_raw_writer import _process_hyper_raw_group
from fnirs_pipe.qc.report_shell import footer_vars, guard, note, page_vars, render
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.hyper_report")


# ---- Per-subject quality metrics ----
#
# Which scalars the table lists, and in what order. Keys only: the label, the format, the
# better direction and the colour all come from the metric registry, the same one the
# individual reports read, so a metric cannot print to three decimals in a subject report
# and four here. Absent keys render as a dash, so one list serves the raw path (intensity
# metrics only) and the post path (which adds motion and haemoglobin).
_SUBJECT_METRICS = [
    "sci_mean",
    "psp_mean",
    "cv_mean",
    "snr_mean",
    "gvtd_filt_p95",
    "motion_corrected_pct",
    "spike_pct_frames",
    "hbo_hbr_corr_mean",
    "channel_retention_rate",
]

# What one WTC result yields, keyed to the type of each. Declared once because two places
# read it: `_figure_set` fills a fresh one, and a page renderer falls back to an empty value
# for a window whose figures failed. A key added to only one of the two would leave a panel
# on a condition page showing the whole run's picture with nothing saying so.
#
# Types rather than values, and `_empty_figures` calls them. Holding the empty containers
# themselves would share one dict between every call, so each figure set would write into
# its predecessor's and every page would end up pointing at the last window's figures.
_FIGURE_SET: dict = {"per_channel": dict, "per_roi": dict,
                     "chan_matrix": str, "roi_matrix": str}


def _empty_figures() -> dict:
    """A figure set with every key present and its own empty value."""
    return {key: factory() for key, factory in _FIGURE_SET.items()}

_ARROW = {"higher": "↑", "lower": "↓"}
_CHROMA_LABEL = {"hbo": "HbO", "hbr": "HbR"}


def _metric_class(key: str, value: float, sci_threshold: float) -> str:
    """The registry's verdict, except for SCI, which is judged against the run's own line.

    The exception is the same one the per-channel tables make: this run screened at
    ``--sci-threshold``, so colouring its SCI against the registry's cutoff would show a
    verdict the run did not reach. Everything else is the registry's, and a metric with no
    published cutoff there prints uncoloured on purpose.
    """
    if key == "sci_mean":
        return "qm-ok" if value >= sci_threshold else "qm-bad"
    return metric_class(key, value)


def subject_metric_rows(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> list[dict]:
    """One row per metric, one cell per subject, for the per-subject quality table."""
    rows = []
    for key in _SUBJECT_METRICS:
        cells = []
        for sid in subject_ids:
            value = (sqm_data.get(sid) or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append({"text": format_metric(key, value),
                              "cls": _metric_class(key, value, sci_threshold)})
            else:
                cells.append({"text": MISSING_VALUE, "cls": ""})
        if any(c["text"] != MISSING_VALUE for c in cells):
            rows.append({"label": metric_label(key),
                         "direction": _ARROW.get(metric_direction(key), ""),
                         "cells": cells,
                         "summary": metric_summary(key)})
    return rows


def condition_subject_metrics(
    subject_sqm: dict,
    subject_ids: "list[str]",
    windows: "list[tuple[str, float, float]]",
    sci_threshold: float,
) -> dict:
    """``{condition: rows}`` for the per-subject quality table, one entry per window.

    Each member's quality record already holds the channel-by-window SCI, PSP and CV
    matrices, so a condition is a column selection out of the pass prep made; nothing is
    measured again and no recording is read. This is the same slice the subject report's own
    per-condition pages are built from, through the same function, so a channel's SCI under
    one condition cannot differ between a subject page and a dyad page.

    **It reports, it does not re-decide.** The coherence on a condition's page was computed
    on the channel set the whole recording was screened into, because the window is read out
    of a transform of the whole recording and a per-condition channel set would need a
    transform of its own. Screening each condition separately would also make a contrast
    between two conditions a contrast between two montages. So these rows say how the
    channels held up over this stretch, and the set they were drawn from is the run's.

    A member whose record predates the stored matrices contributes nothing and its column
    reads as absent, which is the honest answer: the values cannot be recovered from the
    whole-run scalars.
    """
    from fnirs_pipe.qc.condition_views import condition_scalars, condition_slices_from_record
    from fnirs_pipe.qc.metrics import resolve_cutoffs

    if not windows:
        return {}

    per_subject: dict = {}
    for sid in subject_ids:
        sqm = subject_sqm.get(sid) or {}
        order = sqm.get("channel_order") or []
        if not sqm.get("windowed") or not order:
            logger.info("%s: no windowed matrices in the quality record, so the "
                        "per-condition quality table has no column for it", sid)
            continue
        # that subject's own lines, falling back to the dyad page's SCI threshold only for
        # a sidecar too old to record them
        cutoffs = resolve_cutoffs(None, **(sqm.get("screen_cutoffs")
                                           or {"sci": sci_threshold}))
        sliced = condition_slices_from_record(
            {"windowed": sqm["windowed"]}, order, windows,
            cutoffs["sci"], cutoffs["psp"])
        for label, values in sliced.items():
            frac = values.get("good_frac_per_channel") or {}
            retention = (sum(v >= cutoffs["good_frac"] for v in frac.values()) / len(frac)
                         if frac else None)
            per_subject.setdefault(label, {})[sid] = condition_scalars(
                values, retention=retention)

    return {label: subject_metric_rows(by_sid, subject_ids, sci_threshold)
            for label, by_sid in per_subject.items()}


def write_isc_matrix(
    tsv_path: Path,
    isc_mat,
    ch_names: list[str],
    ch_type: str,
    sources: list[str],
    subject_ids: list[str],
) -> None:
    """Write the matrix the ISC panel is drawn from, so the numbers can leave the report.

    Both axes carry the montage's channel labels, which is how :func:`compute_isc` pairs the
    two brains: cell (i, j) is the first subject's channel i against the other's channel j.
    Rejected channels are blank rather than absent, so the file's shape is the montage's
    however many channels a given dyad lost.

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
    sci_threshold: float = SCI_PASS,
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


def crop_provenance(raw: "mne.io.Raw") -> "dict | None":
    """What a cropped input's sidecar says about the cut, or None if it is a whole recording.

    ::

      a file cut to [3555, 3927] with a 47 s margin
        -> {"window": [3555.0, 3927.0], "analysis": [3602.4, 3902.5], "margin_s": 47.1}

    ``fnirs-prep crop`` records the span it wrote and, with ``--margin``, the narrower span
    the cut was made for. Nothing downstream had read either, so a segment and a whole
    recording were treated identically and the tool said nothing about it. That is the one
    way into edge-inflated coherence that a user cannot see, since the numbers look ordinary.

    The sidecar is read off disk rather than the lineage stamp, which carries only the filter
    keys across a SNIRF round trip.
    """
    path = path_from(raw)
    if not path:
        return None
    try:
        side = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    params = (side.get("parameters") or {})
    windows = params.get("crop_windows_s")
    if not windows:
        return None
    analysis = params.get("crop_analysis_windows_s")
    return {"window": windows[0] if len(windows) == 1 else windows,
            "analysis": (analysis[0] if analysis and len(analysis) == 1 else analysis),
            "margin_s": float(params.get("crop_margin_s") or 0.0),
            "n_windows": len(windows)}


def markers_on_data_axis(raw: "mne.io.Raw") -> list[dict]:
    """Non-BAD annotations with their onsets moved onto the data axis, which starts at zero.

    ::

      a raw cropped from 22.4 s, annotation "talk" at onset 3602.4  ->  onset 3580.0

    Annotations of a cropped recording still sit on the original recording's axis, with the
    offset held in ``first_time``, while the data axis and everything computed from it start
    at zero. :func:`~fnirs_pipe.pipeline.hyperscanning.align_recordings` crops every member
    from its first shared trigger, so raw annotation onsets are late by that trigger's onset
    against any figure or window drawn on the aligned clock.

    Every consumer of the aligned markers goes through this. The condition windows did the
    subtraction and the figures' block boundaries did not, which drew every boundary line on
    every coherence map late by that same offset. It is invisible on a tree whose input
    files each held one condition cropped to its own start, where the offset is zero.
    """
    origin = float(raw.first_time)
    markers = extract_markers(raw)
    if not origin:
        return markers
    return [{**m, "onset": float(m["onset"]) - origin} for m in markers]


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
    two-second trials yields nothing here: a window holding less than one cycle has no
    average of that frequency to report, however the coherence was computed.
    """
    markers = sorted(markers_on_data_axis(raw), key=lambda m: m["onset"])
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
    wtc_cond_pad_s: "float | None" = None,
    wtc_limit_scales: bool = True,
    wtc_save_maps: bool = False,
    wtc_mask_coi: bool = True,
    wtc_roi_min_channels: int = 2,
    wtc_chroma: "tuple[str, ...] | list[str]" = ("hbo", "hbr"),
    isc_threshold: float = 0.3,
    sci_threshold: float = SCI_PASS,
    sep_bands=None,
    cond_windows: "list[tuple[str, float, float]] | None" = None,
    analysis_window: "tuple[float, float] | None" = None,
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
    rows in ``hyper-wtc.tsv`` instead of 14. The extra pairs reach the TSV and the crossed
    matrix, while the map selector keeps the homologous ones: 196 options is not a list
    anybody reads through, and drawing a full frequency × time map for each of them per
    chromophore is most of what the figures cost. Crossing is also what produces the
    ROI × ROI matrix, since the ROI numbers are grouped from the channel ones.

    ``wtc_by_condition`` repeats the whole coherence analysis inside each task annotation's
    own window, on top of the whole-run pass, which stays as it was. The band means of every
    window land in one ``hyper-wtcbycond.tsv`` with a ``condition`` column, and each window
    gets its own figures. See :func:`condition_windows` for how a window is decided, and note
    that the runtime is roughly doubled: the windows together are about one more pass over
    the recording.

    ``cond_windows`` supplies those windows instead of resolving them here. The caller passes
    the same list to the pseudo-dyad null, and the two tables can only be subtracted row by
    row if they describe the same windows. Left at None the windows are resolved here, which
    is what a caller that writes no null wants.

    ``analysis_window`` is ``--tstart``/``--tend``: the stretch the whole-run pass reports
    on. It is taken out of the transform of the whole recording, the way a condition window
    is, so a run restricted to a window carries the recording's cone of influence rather than
    two edges of its own. The recordings themselves are never cut.

    ``wtc_cond_pad_s`` transforms each condition on its own instead of reading it out of the
    whole-run transform, over a cut padded by that many seconds on each side. Left at None,
    which is the default, conditions are windowed out of the whole-run transform. See
    :func:`_transform_condition`: with adequate padding the two give the same number, so this
    exists for a caller who wants per-condition transforms, and 0 reproduces the unpadded cut
    other pipelines take. Costs one transform per condition per chromophore.

    ``wtc_mask_coi`` restricts each band mean to the cone of influence. On by default; the
    share inside the cone is reported either way as ``n_valid_frac``.

    ``wtc_chroma`` is the chromophores to run, ``("hbo",)``, ``("hbr",)`` or both. Both is
    the default and costs exactly twice as much, since the two are the same computation run
    twice: a member's HbO pairs only with the other member's HbO, and the two are never
    mixed and never averaged. The reason to have both is a consistency check rather than two
    results, HbO carrying the larger amplitude and HbR the less scalp contamination, so a
    coupling in HbO with nothing in HbR is a caution flag. Every band-mean table gains a
    ``chromophore`` column rather than splitting per chromophore, the tables being
    long-format. The figures are keyed by chromophore instead and the page carries all of
    them, switched together by one control, which is the comparison the second chromophore
    exists for; ``wtc_chroma[0]`` is only what the page opens on.

    ``wtc_save_maps`` writes the full time-frequency maps beside the tables as ``.npz``, one
    per chromophore, so a different band can be averaged later without a second wavelet
    transform. See :mod:`fnirs_pipe.qc.wtc_store`. ``wtc_limit_scales`` computes only the scales inside
    ``[wtc_fmin, wtc_fmax]`` plus margin, which is most of the runtime and, given that the
    scales land on pycwt's own grid and the margin exceeds its scale-smoothing window,
    reproduces the unrestricted coherences bit for bit.
    """
    from fnirs_pipe.exceptions import StageError
    from fnirs_pipe.pipeline.hyperscanning import (
        WTCResult,
        _hyper_sidecar,
        compute_wtc,
        roi_maps_from_channels,
        roi_mean_of_channels,
        window_result,
        wtc_band_mean,
    )
    from fnirs_pipe.pipeline.synchrony import long_axis_over, wtc_grid_params
    from fnirs_pipe.qc.figures.hyper_post_figures import (
        build_isc_panel,
        build_wtc_channel,
        build_wtc_cross_matrix,
        compute_isc,
    )

    errors: list[str] = []
    notes: list[str] = []
    scope = f"group-{group_id}_task-{task}"

    chroma = tuple(dict.fromkeys(wtc_chroma))
    if not chroma or set(chroma) - {"hbo", "hbr"}:
        raise ValueError(f"wtc_chroma must be some of ('hbo', 'hbr'), got {wtc_chroma!r}")
    # the chromophore the page opens on. Every requested one is in the page and the switch
    # moves between them; this is only what shows before anyone touches it, and it is also
    # what a reader with JavaScript off sees named in the titles
    fig_chroma = chroma[0]

    cond_pad_s   = None if wtc_cond_pad_s is None else max(0.0, float(wtc_cond_pad_s))
    subject_ids  = [e.subject_id for e in group]
    ref_raw      = aligned_raws.get(subject_ids[0]) if subject_ids else None

    # ---- is this a segment rather than a recording? ----
    # A cut carries two edges of its own, and everything this report computes from a wavelet
    # transform loses a share of its band at them that grows as the cut shortens. The tool
    # used to treat a segment and a whole recording identically and say nothing, which is the
    # one way into inflated numbers a reader cannot see. It still computes; it now says so.
    crop_info = crop_provenance(ref_raw) if ref_raw else None
    if crop_info:
        span = crop_info["window"]
        where = (f"{span[0]:.1f}-{span[1]:.1f} s of the source recording"
                 if isinstance(span, list) and len(span) == 2 and not isinstance(span[0], list)
                 else f"{crop_info['n_windows']} separate windows of the source recording")
        if crop_info["margin_s"] > 0:
            note(notes, scope,
                 f"Input is a cut ({where}), made with a {crop_info['margin_s']:.1f} s "
                 f"margin on each side. --wtc-by-condition windows each block out of the "
                 f"transform, so the margin is what absorbs the cone and is not averaged.")
        else:
            note(notes, scope,
                 f"Input is a cut ({where}), made with no margin. A wavelet transform of a "
                 f"segment has two edges of its own, so this run's band means are inflated "
                 f"by an amount that grows as the segment shortens; n_valid_frac reports the "
                 f"share of band cells that survived. Re-cut with "
                 f"`fnirs-prep crop --margin auto --band-fmin <f>` to avoid it.")
        logger.warning("cropped input: %s, margin %.1f s", where, crop_info["margin_s"])
    markers_list = markers_on_data_axis(ref_raw) if ref_raw else []
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

    # One subdirectory per group and task, the way a per-run subject report gets one: a
    # group with five cropped tasks writes five sets of these and they would otherwise
    # overwrite each other under one figures/.
    figures_dir = group_report_dir(output_dir, group_id) / "figures" / scope

    def _fig(b64: "str | None", name: str) -> "str | None":
        """One figure onto disk, returning the URL the page links it by, or None.

        Every figure in this report goes out as a file. Embedding them instead is what took
        this page to 174 MB on a 14-channel dyad: a coherence map is 51 x 3962 cells, and
        the page carried one of them per channel per chromophore.
        """
        return save_png(b64, figures_dir, name)

    def _write_df_tsv(df, kind: str, step: str, **extra) -> Path:
        tsv_path = (group_data_dir(output_dir, group_id)
                    / f"group-{group_id}_task-{task}_hyper-{kind}.tsv")
        df.to_csv(tsv_path, sep="\t", index=False)
        _hyper_sidecar(
            tsv_path, step,
            [p for p in (path_from(r) for r in aligned_raws.values()) if p],
            band_fmin=band_fmin, band_fmax=band_fmax, mask_coi=wtc_mask_coi,
            wtc_fmin=wtc_fmin, wtc_fmax=wtc_fmax, chroma=list(chroma),
            **wtc_grid_params(aligned_raws), **extra,
        )
        return tsv_path

    def _safe_post(name: str, fname: str, fn, *args) -> "str | None":
        """Build one figure, write it to ``figures/``, and hand back its URL."""
        out = None
        with guard(f"{name} figure", errors, scope):
            out = _fig(fn(*args), fname)
        return out

    def _maps(dest: dict, result, pair_key, pair_label: str, axis: list[str],
              stem: str, ch_type: str, suffix: str, what: str) -> None:
        """Fill ``dest`` with one coherence map per pairing of ``axis`` against itself.

        ::

          axis ["S1_D1", "S1_D2"], crossed
            -> dest["S1_D1"]["S1_D2"] = {"wtc": "figures/.../wtc_hbo_S1D1_x_S1D2.png"}

        Both map panels are built here, the channels with the channel axis and the ROIs with
        the ROI labels, so the two cannot drift in how they key or name their files. The
        nesting is ``[label of the first member][label of the second]`` and is written even
        for an uncrossed run, which fills only the diagonal: the page then reads one shape
        and shows one selector instead of two.

        A pairing the result has no entry for, or whose builder failed, lands as ``None``
        and the page hides its image. That is the rejected channel's blank row arriving on
        the figure side of the same rule the tables follow.
        """
        for label1 in axis:
            row: dict = {}
            for label2 in (axis if wtc_channel_cross else [label1]):
                fig = None
                if result and pair_key:
                    key = (label1, label2) if wtc_channel_cross else label1
                    data = result.pairs.get(pair_key, {}).get(key)
                    site = label1 if label1 == label2 else f"{label1} × {label2}"
                    fname = (f"{stem}_{_pair_fname(label1)}{suffix}.png"
                             if label1 == label2 else
                             f"{stem}_{_pair_fname(label1)}_x_{_pair_fname(label2)}"
                             f"{suffix}.png")
                    fig = _safe_post(
                        f"wtc map {site} ({what}, {ch_type})", fname,
                        build_wtc_channel,
                        data, result.freqs, result.times,
                        pair_label, markers_list, cond_colors_, site,
                    )
                row[label2] = {"wtc": fig}
            dest[label1] = row

    def _transform_condition(tstart: float, tstop: float, ch_type: str):
        """One condition's own transform, taken over a padded cut and windowed back.

        ::

          condition [3580, 3880] with cond_pad_s 47
            -> transform [3533, 3927], then window the result to [3580, 3880]

        The alternative to reading the window out of the whole-run transform, for a caller
        who wants each condition transformed on its own. **It is the same number**, to four
        decimal places, as long as the padding is wide enough: measured on d01, a padded cut
        and the whole-record transform agree per map cell to 0.00014 at a correlation of
        1.00000, and a condition's band mean stops moving once the padding passes
        :func:`~fnirs_pipe.pipeline.synchrony.cone_margin_s`. What the padding buys is the
        cone: it lands in the margin instead of eating the condition's own edges, which is
        the whole difference between this and cutting a condition to its own boundaries.

        ``cond_pad_s`` of 0 does cut to the boundaries, which reproduces the route most
        published per-condition pipelines take and is biased upward by an amount that grows
        as the condition shortens. It is here to reproduce such a result, not to produce one.

        Costs one transform per condition per chromophore on top of the whole-run pass, which
        is why it is not the default.
        """
        lo = max(0.0, tstart - cond_pad_s)
        hi = min(min(float(r.times[-1]) for r in aligned_raws.values()), tstop + cond_pad_s)
        cut = {sid: raw.copy().crop(tmin=lo, tmax=hi)
               for sid, raw in aligned_raws.items()}
        res = compute_wtc(cut, fmin=wtc_fmin, fmax=wtc_fmax,
                          significance=wtc_significance, seed=wtc_seed,
                          mc_count=wtc_mc_count, cross=wtc_channel_cross,
                          limit_scales=wtc_limit_scales, ch_type=ch_type,
                          sep_bands=sep_bands)
        # the cut's own clock starts at zero, so the condition sits `tstart - lo` into it
        return window_result(res, tstart - lo, tstart - lo + (tstop - tstart))

    def _tag(df, ch_type: str):
        """The column saying which chromophore a row is, added after every aggregation.

        ``roi_mean_of_channels`` and ``wtc_band_mean`` group on the columns they know and
        drop the rest, so tagging before them loses the tag. Same reason ``condition`` is
        inserted after the band mean rather than carried into it.
        """
        df.insert(0, "chromophore", ch_type)
        return df

    def _save_maps(result, kind: str, ch_type: str) -> None:
        """The full time-frequency maps beside the table, one archive per chromophore.

        Named ``hyper-<kind>-<ch_type>.npz`` rather than after the TSV, because the TSV now
        holds both chromophores and two archives cannot share one name. ``fnirs-hyper band``
        globs ``*_hyper-wtc*.npz``, which this still matches.
        """
        from fnirs_pipe.qc.wtc_store import save_wtc
        npz_path = (group_data_dir(output_dir, group_id)
                    / f"group-{group_id}_task-{task}_hyper-{kind}-{ch_type}.npz")
        with guard(f"Saving WTC maps ({kind} {ch_type})", errors, scope):
            save_wtc(result, npz_path)

    def _band_means(result, kind: str, ch_type: str):
        if result is None or not result.pairs:
            return None
        df = None
        with guard(f"WTC band mean ({kind} {ch_type})", errors, scope):
            df = wtc_band_mean(result, band_fmin, band_fmax, mask_coi=wtc_mask_coi)
        return df

    # the channel selector and the ROI grouping table describe the montage, so they are the
    # same whichever chromophore is drawn and are built once
    ch_pairs_post: list[str] = get_channel_pairs(ref_raw) if ref_raw else []

    # ---- the axis both channel panels are indexed by ----
    # The union of the members' long channels, rejections kept, which is the axis the
    # cross matrix is drawn on: selector and matrix have to name the same set or a reader
    # cannot find a matrix cell in the selector. `ch_pairs_post` above is the whole montage
    # including the short channels, which have no coherence and left the old selector with
    # eight dead entries. Labels do not carry the chromophore, so one pass serves both.
    _members = [aligned_raws[s] for s in subject_ids if s in aligned_raws]
    chan_axis: list[str] = (long_axis_over(_members, fig_chroma, sep_bands) if _members
                            else ch_pairs_post)
    roi_rows: list[dict] = []
    roi_labels: list[str] = list(roi_map.keys()) if roi_map else []
    if roi_map:
        assigned = {ch for chs in roi_map.values() for ch in chs}
        for roi_name, chs in roi_map.items():
            roi_rows.append({"roi": roi_name, "channels": chs})
        unassigned = [c for c in ch_pairs_post if c not in assigned]
        if unassigned:
            roi_rows.append({"roi": "Unassigned", "channels": unassigned})

    # the task windows are a property of the annotations, not of the chromophore
    if not wtc_by_condition:
        cond_windows = []
    else:
        if cond_windows is None:
            cond_windows = (condition_windows(ref_raw, min_duration=1.0 / wtc_fmin)
                            if ref_raw else [])
        if not cond_windows:
            note(notes, scope,
                 "--wtc-by-condition asked for, but no annotation window is long enough "
                 f"for one cycle of {wtc_fmin:.4f} Hz: no per-condition figures")
        else:
            logger.info("--wtc-by-condition: %d window(s), each read off the whole-run "
                        "transform", len(cond_windows))

    def _figure_set(result, chan_band_df, ch_type: str, suffix: str = "") -> dict:
        """Every figure one WTC result yields: the maps and the three matrices.

        Called once with the whole-run result and again with each condition window's, so a
        condition page carries the panels the run's own page carries instead of the two
        pictures a window used to get. Nothing here decides per panel whether a condition
        has it; the only difference between the two calls is what result comes in.

        ``suffix`` names the files, ``""`` for the run and ``"_<slug>"`` for a window, the
        same rule a per-condition subject page names its panels by. That is what lets both
        sets sit in one ``figures/`` directory without the window overwriting the run.

        Returns ``{"per_channel", "per_roi", "chan_matrix", "roi_matrix", "roichan"}``.
        The two map sets are nested ``{label_sub1: {label_sub2: {"wtc": url}}}`` whether or
        not the run crossed, an uncrossed one holding only the diagonal, so the page reads
        one shape and the pair of selectors above each panel is the only difference.
        ``roichan`` is the ROI band-mean frame, untagged: the caller adds the chromophore
        and, for a window, the condition, because the aggregations inside drop columns they
        do not know.
        """
        out: dict = {**_empty_figures(), "roichan": None}
        what = f"condition {suffix.lstrip('_')}" if suffix else "whole run"

        pair_key   = next(iter(result.pairs)) if result and result.pairs else None
        pair_label = f"{pair_key[0]} × {pair_key[1]}" if pair_key else ""

        if wtc_channel_cross and chan_band_df is not None:
            with guard(f"WTC channel cross matrix ({what}, {ch_type})", errors, scope):
                # the montage, not the labels the table happens to carry: a dyad that lost a
                # channel still gets a matrix of the same shape as one that did not
                chan_labels = chan_axis or sorted({*chan_band_df["label"],
                                                   *chan_band_df["label2"]})
                out["chan_matrix"] = _fig(build_wtc_cross_matrix(
                    chan_band_df, chan_labels, subject_ids, band_fmin, band_fmax,
                    kind="channel"), f"wtc_chanmatrix_{ch_type}{suffix}.png") or ""

        _maps(out["per_channel"], result, pair_key, pair_label, chan_axis,
              f"wtc_{ch_type}", ch_type, suffix, what)

        if not roi_map:
            return out

        # the ROI number the WTC literature reports: coherence per channel pair, then
        # averaged
        roi_band_df = None
        if chan_band_df is not None:
            with guard(f"ROI mean of channel WTC ({what}, {ch_type})", errors, scope):
                roi_band_df = roi_mean_of_channels(
                    chan_band_df, roi_map, min_channels=wtc_roi_min_channels)
        out["roichan"] = roi_band_df

        # the maps grouped the same way, so the picture and the table are one average
        roi_wtc: WTCResult | None = None
        if result is not None:
            with guard(f"ROI WTC maps from channels ({what}, {ch_type})", errors, scope):
                roi_wtc = roi_maps_from_channels(result, roi_map)
        roi_pair_key = next(iter(roi_wtc.pairs)) if roi_wtc and roi_wtc.pairs else None

        if roi_band_df is not None and wtc_channel_cross:
            # the same builder the channel matrix uses, with the ROI labels: the two were
            # one lookup and one heatmap, written twice in two libraries
            out["roi_matrix"] = _safe_post(
                f"wtc-roi-matrix ({what}, {ch_type})",
                f"wtc_roimatrix_{ch_type}{suffix}.png",
                build_wtc_cross_matrix,
                roi_band_df, roi_labels, subject_ids, band_fmin, band_fmax, "ROI",
            ) or ""
        _maps(out["per_roi"], roi_wtc, roi_pair_key, pair_label, roi_labels,
              f"wtcroi_{ch_type}", ch_type, suffix, what)

        return out

    def _wtc_pass(ch_type: str) -> dict:
        """The whole WTC analysis for one chromophore: rows for the tables, and figures.

        HbO and HbR are two parallel runs of the same code. A member's HbO pairs only with
        the other member's HbO, the two are never mixed and never averaged, and nothing
        about the statistic changes, so this is a loop over ``--wtc-chroma`` rather than a
        branch anywhere inside it. Cost is one full multiple per chromophore.

        Returns the rows each table wants, untagged, plus every figure. The caller
        concatenates the rows across chromophores and writes one table per kind: the
        band-mean tables are long-format, so a ``chromophore`` column keeps their filenames
        stable. The figures stay keyed by chromophore instead, because each page carries all
        of them and switches between them, which is the comparison the second chromophore
        exists for.

        ``cond_figs`` is one figure set per window, positional, so the chromophores' lists
        line up for the switch even where a guard failed on one of them.
        """
        out: dict = {"chan": None, "roichan": None, "cond_chan": [], "cond_roi": [],
                     "run_figs": {}, "cond_figs": []}

        wtc_result: WTCResult | None = None
        with guard(f"WTC computation ({ch_type})", errors, scope):
            logger.info("Computing WTC on %s for %d subjects...", ch_type, len(subject_ids))
            wtc_result = compute_wtc(
                aligned_raws, fmin=wtc_fmin, fmax=wtc_fmax, significance=wtc_significance,
                seed=wtc_seed, mc_count=wtc_mc_count, cross=wtc_channel_cross,
                limit_scales=wtc_limit_scales, ch_type=ch_type, sep_bands=sep_bands)

        # --tstart/--tend: the window is read out of the transform, never cut from the
        # recording, so the whole-run pass below is the window's and the conditions inside
        # it are windows of the same transform. One route for both.
        if wtc_result is not None and analysis_window is not None:
            with guard(f"Analysis window ({ch_type})", errors, scope):
                wtc_result = window_result(wtc_result, *analysis_window)

        chan_band_df = _band_means(wtc_result, "wtc", ch_type)
        out["chan"] = chan_band_df
        if chan_band_df is not None and wtc_save_maps:
            _save_maps(wtc_result, "wtc", ch_type)

        out["run_figs"] = _figure_set(wtc_result, chan_band_df, ch_type)
        out["roichan"] = out["run_figs"]["roichan"]

        # ---- per condition ----
        # Each task window read out of the whole-run pass above rather than transformed on
        # its own, which is the comparison a block design is run for.
        #
        # Windowed, not recomputed: a window transformed alone has two edges of its own and
        # the cone of influence reaches further at longer periods, so a short condition
        # keeps a smaller share of its band cells and the ones it keeps anyway are padded
        # against those edges. Recomputing inflated the band mean by an amount that tracked
        # window length, which in a design whose conditions differ in length is confounded
        # with the contrast. See `window_result`. It is also cheaper: the whole-run
        # transform is computed either way, and this adds no second one.
        for label, tstart, tstop in cond_windows:
            out["cond_figs"].append({})
            cond_chan = cond_wtc = None
            with guard(f"Condition {label}: WTC ({ch_type})", errors, scope):
                if cond_pad_s is None:
                    if wtc_result is None:
                        raise StageError("the whole-run WTC failed, so no window can be "
                                         "read out of it")
                    cond_wtc = window_result(wtc_result, tstart, tstop)
                else:
                    cond_wtc = _transform_condition(tstart, tstop, ch_type)
                cond_chan = wtc_band_mean(cond_wtc, band_fmin, band_fmax,
                                          mask_coi=wtc_mask_coi)
            if cond_chan is None:
                continue

            # the same panels the run's own page gets, over this window
            figs = _figure_set(cond_wtc, cond_chan, ch_type, f"_{_pair_fname(label)}")
            out["cond_figs"][-1] = figs

            cond_roi = figs.pop("roichan", None)
            if cond_roi is not None and not cond_roi.empty:
                cond_roi = cond_roi.copy()
                cond_roi.insert(0, "condition", label)
                out["cond_roi"].append(_tag(cond_roi, ch_type))

            cond_chan = cond_chan.copy()
            cond_chan.insert(0, "condition", label)
            out["cond_chan"].append(_tag(cond_chan, ch_type))

        return out

    if wtc_significance:
        logger.warning("WTC significance on: %d Monte Carlo surrogates per channel pair, "
                       "this is slow.", wtc_mc_count)
    if wtc_channel_cross:
        logger.warning("WTC channel crossing on: every long channel against every other, "
                       "so the pair count is squared and so is the runtime.")
    if len(chroma) > 1:
        logger.info("--wtc-chroma %s: %d full WTC passes, one per chromophore",
                    "+".join(chroma), len(chroma))

    passes = {ch_type: _wtc_pass(ch_type) for ch_type in chroma}

    def _stack(key: str):
        """One kind's rows from every chromophore, tagged, or None when nothing ran."""
        frames = []
        for ch_type, result in passes.items():
            df = result[key]
            if df is not None and not df.empty:
                frames.append(_tag(df.copy(), ch_type))
        return pd.concat(frames, ignore_index=True) if frames else None

    chan_band_df = _stack("chan")
    if chan_band_df is not None:
        logger.info("WTC band means saved: %s",
                    _write_df_tsv(chan_band_df, "wtc", "hyper_wtc"))
    roi_band_df = _stack("roichan")
    if roi_band_df is not None:
        logger.info("WTC ROI means from channels saved: %s",
                    _write_df_tsv(roi_band_df, "wtc-roichan", "hyper_wtc_roichan"))

    # the windows are the one thing a reader cannot reconstruct from the table
    spans = {label: [round(t0, 3), round(t1, 3)] for label, t0, t1 in cond_windows}
    cond_chan_frames = [f for r in passes.values() for f in r["cond_chan"]]
    cond_roi_frames  = [f for r in passes.values() for f in r["cond_roi"]]
    if cond_chan_frames:
        logger.info("WTC band means per condition saved: %s",
                    _write_df_tsv(pd.concat(cond_chan_frames, ignore_index=True),
                                  "wtcbycond", "hyper_wtc_bycondition",
                                  condition_windows_s=spans))
    if cond_roi_frames:
        logger.info("WTC ROI means per condition saved: %s",
                    _write_df_tsv(pd.concat(cond_roi_frames, ignore_index=True),
                                  "wtcbycond-roichan", "hyper_wtc_bycondition_roichan",
                                  condition_windows_s=spans))

    # ISC panels, for the whole run and for each window. Not driven by --wtc-chroma: ISC is
    # cheap, so it has always run on both, and an ISC table is a channel-by-channel matrix
    # that cannot share a file with a second one the way the long-format WTC tables can.
    #
    # A window here is a real cut of the recording, unlike the coherence, which is sliced out
    # of the whole-run transform. Both are right: a correlation has no frequency axis and
    # nothing in it filters, so a cut window carries no edge the record would not have had,
    # while a wavelet transform of a cut window has two edges and a cone of its own. See
    # `compute_isc` and `window_result`, which each say why they do it their way.
    def _isc_panel(ch_type: str, label: "str | None" = None,
                   window: "tuple[float, float] | None" = None) -> str:
        what = f"condition {label}" if label else "whole run"
        suffix = f"_{_pair_fname(label)}" if label else ""
        panel = ""
        with guard(f"ISC panel ({what}, {ch_type})", errors, scope):
            isc_mat, isc_ch_names = compute_isc(aligned_raws, subject_ids, ch_type,
                                                sep_bands, window=window)
            if isc_mat is None:
                return ""
            # the desc- entity a condition's page takes, so its table is named the way its
            # page is and a reader can pair the two without a rule of their own
            desc = f"_desc-{_pair_fname(label)}" if label else ""
            write_isc_matrix(
                group_data_dir(output_dir, group_id)
                / f"group-{group_id}_task-{task}{desc}_hyper-isc-{ch_type}.tsv",
                isc_mat, isc_ch_names, ch_type,
                [p for p in (path_from(r) for r in aligned_raws.values()) if p],
                subject_ids,
            )
            panel = _fig(build_isc_panel(
                isc_mat, isc_ch_names, subject_ids,
                ch_type=ch_type, isc_threshold=isc_threshold,
            ), f"isc_{ch_type}{suffix}.png") or ""
        return panel

    # {label or None: {chromophore: href}}, one entry per page below
    isc_panels: dict = {None: {c: _isc_panel(c, window=analysis_window)
                               for c in ("hbo", "hbr")}}
    for label, tstart, tstop in cond_windows:
        isc_panels[label] = {c: _isc_panel(c, label, (tstart, tstop))
                             for c in ("hbo", "hbr")}

    # ISC has no frequency axis, so an unfiltered stage reaches the number directly; WTC
    # does not care. Said on the page as well as in the log, since the two are read by
    # different people
    isc_unfiltered_note = unfiltered_stage_note(aligned_raws)
    if isc_unfiltered_note:
        logger.warning("ISC: %s", isc_unfiltered_note)

    # the quality table: the run's over the whole recording, and each window's sliced out
    # of the same stored matrices
    run_metric_rows = subject_metric_rows(subject_sqm or {}, subject_ids, sci_threshold)
    cond_metric_rows: dict = {}
    with guard("Per-condition quality table", errors, scope):
        cond_metric_rows = condition_subject_metrics(
            subject_sqm or {}, subject_ids, cond_windows, sci_threshold)

    bad_pairs_all: set[str] = set()
    if bad_channels:
        for chs in bad_channels.values():
            bad_pairs_all |= {c.rsplit(" ", 1)[0] for c in chs}

    # desc-hyperpost, matching the raw report's desc-hyperraw: the two are one pair of
    # pages and were named by two conventions, one BIDS-shaped and one not.
    #
    # A condition keeps the run's task- entity and takes a desc- of its own,
    # `..._task-full_desc-baseline_hyperpost_nirs.html`, which is the rule the subject
    # report follows. Putting the label in `task-` instead would name a condition page the
    # same as the run page of a tree where that condition was cropped to its own task, and
    # the two are not the same number: one carries the whole recording's cone of influence
    # and the other two edges of its own.
    def _page_path(label: "str | None") -> Path:
        desc = "hyperpost" if label is None else f"{_pair_fname(label)}_hyperpost"
        return (group_report_dir(output_dir, group_id)
                / f"group-{group_id}_task-{task}_desc-{desc}_nirs.html")

    # every page carries the whole strip, so any one of them reaches the others in a click
    nav_pages = [(None, "Whole run")] + [(label, label) for label, _, _ in cond_windows]

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

    def _render_page(figs: dict, label: "str | None",
                     window: "tuple[float, float] | None") -> Path:
        """One page: the whole run's when ``label`` is None, else that condition's.

        Both go through this, so a panel cannot exist on the run's page and be missing from
        a condition's by anyone forgetting to add it. ``figs`` is
        ``{chromophore: figure set}`` and is the only thing that differs between them, the
        figure sets having been built by one function.

        One panel is the run's alone and is left out rather than repeated: the per-subject
        quality table is measured over the whole recording, and printing it under a
        condition's heading would read as that condition's numbers. It is a click away on
        the run's own page, and the page says so.
        """
        # Every figure's URL keyed by chromophore, which is what the page's chromophore
        # switch reads. The panels are one set of DOM nodes filled from these, not one set
        # per chromophore, so a reader is always looking at one chromophore across the
        # whole page rather than at an HbO channel panel above an HbR ROI panel. What
        # travels is a path under figures/, so the switch changes an img src and the
        # browser fetches one file.
        def _by_chroma(key: str) -> dict:
            return {c: (figs.get(c) or {}).get(key) or _FIGURE_SET[key]() for c in chroma}

        per_channel = _by_chroma("per_channel")
        per_roi     = _by_chroma("per_roi")
        roi_matrix  = _by_chroma("roi_matrix")
        chan_matrix = _by_chroma("chan_matrix")

        out_path = _page_path(label)
        heading = "fnirs‑pipe   Hyper Post Report"
        nav_meta = [("group", group_id), ("task", task),
                    ("subjects", ", ".join(subject_ids))]
        if label is not None:
            nav_meta.insert(2, ("condition", label))

        html = render(
            "hyper_post_report.html.j2",
            **page_vars(
                title=(f"fnirs-pipe Hyper Post Report — {group_id} / {task}"
                       + (f" / {label}" if label else "")),
                heading=heading + (f" — {label}" if label else ""),
                nav_meta=nav_meta,
                nav_note=(f"WTC: {wtc_fmin:.3f}–{wtc_fmax:.3f} Hz · "
                          f"{'+'.join(_CHROMA_LABEL[c] for c in chroma)}"),
            ),
            **footer_vars(
                scope=scope, errors=errors, notes=notes,
                nirs_dir=group_data_dir(output_dir, group_id),
                provenance_path=provenance_path,
                methods=methods, versions=versions,
            ),
            group_id=group_id,
            task=task,
            subject_ids=subject_ids,
            wtc_fmin=wtc_fmin,
            wtc_fmax=wtc_fmax,
            wtc_fig_chroma=_CHROMA_LABEL[fig_chroma],
            wtc_chroma_labels=[_CHROMA_LABEL[c] for c in chroma],
            wtc_chroma_json=json.dumps(list(chroma)),
            wtc_chroma_names_json=json.dumps(
                {c: _CHROMA_LABEL[c] for c in chroma}),
            isc_threshold=isc_threshold,
            alignment_json=json.dumps(alignment_rows),
            per_channel_post_json=json.dumps(per_channel),
            # the long axis, not `ch_pairs_post`: the selector has to name the set the
            # matrix beside it is drawn on, and the short channels have no coherence
            ch_pairs_post_json=json.dumps(chan_axis),
            bad_pairs_json=json.dumps(sorted(bad_pairs_all)),
            roi_rows=roi_rows,
            roi_labels_json=json.dumps(roi_labels),
            per_roi_post_json=json.dumps(per_roi),
            wtc_roi_matrix_json=json.dumps(roi_matrix),
            wtc_chan_matrix_json=json.dumps(chan_matrix),
            # the cards these guard exist when any chromophore produced the picture
            has_chan_matrix=any(chan_matrix.values()),
            has_roi_matrix=any(roi_matrix.values()),
            # a second selector on each map panel, which an uncrossed run has no pairings
            # for: it holds the diagonal alone
            chan_crossed=wtc_channel_cross,
            roi_crossed=wtc_channel_cross,
            # ---- what tells the two kinds of page apart ----
            condition_label=label,
            condition_window=(f"{window[0]:.1f}–{window[1]:.1f} s on the aligned clock"
                              if window else ""),
            analysis_window=(f"{analysis_window[0]:.1f}–{analysis_window[1]:.1f} s"
                             if analysis_window else ""),
            run_href=_page_path(None).name,
            nav_links=[{"label": text, "href": _page_path(lab).name,
                        "current": lab == label} for lab, text in nav_pages],
            isc_unfiltered_note=isc_unfiltered_note,
            isc_panel_hbo_path=isc_panels.get(label, {}).get("hbo", ""),
            isc_panel_hbr_path=isc_panels.get(label, {}).get("hbr", ""),
            subject_metrics_rows=(run_metric_rows if label is None
                                  else cond_metric_rows.get(label, [])),
        )
        out_path.write_text(html, encoding="utf-8")
        return out_path

    # the conditions first and the run last, so the run's page carries the complete error
    # and note lists: a guard that failed while a window was being drawn belongs on both
    for i, (label, tstart, tstop) in enumerate(cond_windows):
        figs = {c: (passes[c]["cond_figs"][i] if i < len(passes[c]["cond_figs"]) else {})
                for c in chroma}
        written = _render_page(figs, label, (tstart, tstop))
        logger.info("group-%s | condition %s → %s", group_id, label, written.name)

    output_path = _render_page({c: passes[c]["run_figs"] for c in chroma}, None, None)
    logger.info("Hyper post report saved: %s", output_path)
    return output_path
