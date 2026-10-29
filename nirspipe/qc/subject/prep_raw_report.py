"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from nirspipe.utils import pair_of
from nirspipe.exceptions import StageError
from nirspipe.io.auxiliary import imu_traces, read_aux_snirf
from nirspipe.io.derivatives import bids_uris
from nirspipe.io.head_template import FSAVERAGE, space_of
from nirspipe.io.naming import report_name
from nirspipe.io.optode_space import hold_placement, place_optodes, placement_of
from nirspipe.io.snirf import read_raw_snirf
from nirspipe.pipeline.dpf import (
    as_ppf, dpf_caveat, dpf_from_age, dpf_text, resolve_dpf, wavelengths,
)
from nirspipe.pipeline.motion import motion_settings
from nirspipe.pipeline.prep_pipeline import _expand_bad_pairs, join_marks
from nirspipe.qc.subject.condition_views import (
    carpet_view_table as _carpet_views, condition_view_table, PSD_NFFT_CAP,
)
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.common.figure_io import (
    FILL_FRAME_JS, TOPO_PICK_JS, _pair_fname, _save_figure_html, _save_multi_fig_html,
    save_fragment_views_html,
    extract_markers, figure_namer, get_channel_pairs, plotly_script_tags,
)
from nirspipe.qc.common.windows import markers_on_data_axis, refuse_colliding_labels
from nirspipe.qc.common.channel_table import (
    HAEMO_SPLIT_COLUMNS, MOTION_SPLIT_COLUMNS, OD_LOOSE_KEYS, OD_SPLIT_COLUMNS,
    RAW_CHANNEL_METRICS_SUFFIX, RAW_CSV_FIELDS, WAVELENGTH_COLUMNS, channel_columns,
    metric_set_table,
    channel_rows, format_rows, heart_rate_note, heatmap_args, pair_rows, pair_wavelengths,
    placement_note, region_table, template_note,
    save_channel_csv,
    separation_blocks,
    separation_notes,
    split_table,
)
from nirspipe.qc.metrics import (
    IMU_STAT_KEYS, attach_windowed_series, compute_raw_sqm, compute_sci_scores,
    haemo_quality_metrics, imu_section, imu_windowed, long_short_channels, resolve_cutoffs,
    screen_channels, screening_gap, screening_scores,
)
from nirspipe.qc.metrics._helpers import EPOCH_TMAX as _EPOCH_TMAX, EPOCH_TMIN as _EPOCH_TMIN, separation_orphans
from nirspipe.qc.metrics.windowed import _in_scope, window_centers, windowed_key
from nirspipe.qc.common.report_shell import (
    collapse_messages, footer_vars, guard, note, page_vars, render,
)
from nirspipe.qc.subject.trial_qc import (
    MIN_SCI_TRIAL_S, MIN_TRIAL_S, score_trials, short_trial_counts, trial_windows,
)
from nirspipe.utils.lineage import lineage_of
from nirspipe.utils.logging import get_logger
from nirspipe.utils.spans import bad_spans, mark_manual, mark_unselected, measured_from
from nirspipe.qc.boilerplate import collect_software_versions
from nirspipe.qc.boilerplate.vocabulary import metric_rows
from nirspipe.qc.common.screen_scope import resolve_screen_scope
from nirspipe.qc.subject.record_io import read_record, write_record
from nirspipe.qc.subject.sqm_record import (
    RECORD_SUFFIXES, haemo_sections, motion_sections, raw_condition_sections, raw_sections,
    sqm_record_dict,
)

logger = get_logger("qc.prep_raw_report")

_MAX_TS_PTS   = 4000
# the scalars the viewer's metrics panel lists, in order. Names, formats, thresholds and
# tooltips all come from the metric registry, so this is only the choice of which ones and
# in what order: the same ones the subject report prints, minus what a raw recording has no
# later stage to measure.
#
# Two lists, as the subject report has two cases. A montage that splits gets the two tables
# instead, and the flat list then keeps what the tables have no column for. `_short_section`
# computes neither cardiac power nor the flat-channel count, so those have nothing to put in
# a Short row; the spike frame counts do split but are printed for one set, each set's count
# being against its own tenth-of-the-channels bar and so not a reading of another's.
# A montage with no short channels has no table to put anything in and gets one list.
#
# Derived by subtracting what the tables carry rather than written out again, so a column
# added to either one leaves the list on its own.
_VIEW_MONTAGE_KEYS = tuple(
    k for k in (*OD_LOOSE_KEYS, "mean_amp_mean",
                "spike_count", "spike_pct_frames", "spike_num_frames", *IMU_STAT_KEYS)
    if k not in {key for key, _ in (*OD_SPLIT_COLUMNS, *MOTION_SPLIT_COLUMNS)}
)
_VIEW_SCALAR_KEYS = (
    "channel_retention_rate", "sci_win_mean", "sci_mean", "good_frac_mean", "psp_mean",
    "heart_rate", "snr_mean", "cv_mean",
    *OD_LOOSE_KEYS, "mean_amp_mean", "n_saturated_channels",
    "gvtd_filt_mean", "gvtd_filt_p95", "gvtd_thresh",
    "gvtd_pct_above_thresh", "gvtd_num_above_thresh",
    "spike_count", "spike_pct", "spike_pct_frames", "spike_num_frames",
    *IMU_STAT_KEYS,
)

# the loose metrics that belong with motion rather than with the optical density
_MOTION_LOOSE = frozenset({key for key, _ in MOTION_SPLIT_COLUMNS} | set(IMU_STAT_KEYS))

def _sqm_table(sqm: dict, n_all: int) -> dict:
    """The page's metrics as one table, from the set tables and loose rows its payload carries.

    One builder for the run page and the condition pages, whose payloads have the same
    ``sqm`` shape; a page with no split gets one column.
    """
    split = sqm.get("split") or {}
    loose = sqm.get("rows") or []
    return metric_set_table(
        [(r["name"], r["n"]) for r in split.get("rows") or []] or [("All", n_all)],
        [("Optical density", split, [r for r in loose if r["key"] not in _MOTION_LOOSE]),
         ("Motion", sqm.get("motion_split") or {}, [r for r in loose if r["key"] in _MOTION_LOOSE]),
         ("Haemoglobin", sqm.get("haemo_split") or {}, [])])


# ---- Channel decisions table ----
# Its columns, like every other view's, come from channel_table. It draws the decision chip
# itself, that being the one column that is not a measurement. The keys go over as JSON
# because the table body is built in the browser.
def _column_vars(columns: list) -> dict:
    return {"ch_columns": columns, "ch_column_keys_json": json.dumps([key for key, _ in columns])}


# raw QC writes no channel table, so it has no regions to print, and regresses nothing
_CH_COLUMN_VARS = _column_vars(channel_columns(("separation", "region", "short_share")))
# a condition has no whole-run SCI: that estimate has no windows to select from
_COND_CH_COLUMN_VARS = _column_vars(channel_columns(("separation", "sci_whole", "region",
                                                     "short_share")))


def _store_matrices(windowed: dict, series: dict) -> None:
    """The channel-by-window matrices into the record, under the pipeline's own key names."""
    for key in ("sci_matrix", "psp_matrix", "cv_matrix",
                "sci_times", "psp_times", "cv_times",
                "sci_channels", "psp_channels", "cv_channels"):
        if series.get(key) is not None:
            windowed[key] = np.asarray(series[key]).tolist()


def _store_spans(windowed: dict, raw, raw_after, sep_bands, errors: list, label: str) -> None:
    """The flagged spans, kept as spans so a condition counts the run's own boolean.

    ::

      windowed["spike_spans_long_s"] -> [[412.0, 1.3], [880.5, 0.8]]

    Two families, each on its own channel sets, because both tests are over a set rather
    than over a channel: a spike span found on the short channels is not a claim about the
    long ones, and GVTD is an RMS across whatever it is given, so each set has its own trace
    and its own threshold. Every channel under the plain key, as the pipeline's record has it.

    Spikes are found on the recording as delivered and GVTD on ``raw_after``, the corrected
    copy where a correction ran, as the pipeline's record finds them, so a condition's GVTD
    share reads the file its GVTD means were taken from.
    """
    from nirspipe.qc.metrics import (
        gvtd_above_segments, gvtd_channel_picks, long_short_channels, spike_segments,
    )

    long_names, short_names = long_short_channels(raw, sep_bands)
    with guard("Spike spans", errors, label):
        for channel_set, names in (("all", list(raw.ch_names)), ("long", long_names),
                                   ("short", short_names)):
            if names:
                windowed[windowed_key("spike_spans", channel_set, spans=True)] = [
                    list(span) for span in spike_segments(raw.copy().pick(names))]
    gvtd_raw = raw if raw_after is None else raw_after
    with guard("GVTD above-threshold spans", errors, label):
        # the long set is the canonical trace's, absent on a montage that never split
        gvtd_long, canonical = gvtd_channel_picks(gvtd_raw, sep_bands)
        for channel_set, picks in (("all", list(gvtd_raw.ch_names)),
                                   ("long", gvtd_long if canonical == "long" else []),
                                   ("short", short_names)):
            if picks:
                windowed[windowed_key("gvtd_above_spans", channel_set, spans=True)] = [
                    list(span) for span in gvtd_above_segments(gvtd_raw, sep_bands, picks=picks)]


def _span_sets(windowed: dict, stem: str, first: str) -> dict:
    """{channel set: spans} for the GVTD panel's first set and the short one, which it shades."""
    return {name: [tuple(sp) for sp in windowed.get(windowed_key(stem, name, spans=True)) or []]
            or None for name in (first, "short")}


def _process_run(
    run: dict,
    sci_threshold: float,
    sub_dir: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
    dpf: list[float],
    window_s: float = 10.0,
    epoch_qc: bool = False,
    epoch_tmin: float | None = None,
    epoch_tmax: float | None = None,
    psp_threshold: float | None = None,
    sep_bands=None,
    min_good_frac: float | None = None,
    screen_scope: str = "run",
    motion_correction: str | None = None,
    roi_map: "dict | None" = None,
    motion_std_thresh: float | None = None,
    wavelet_iqr: float | None = None,
) -> "tuple[dict, dict]":
    """Compute all data, save figure HTMLs + SQM JSON. Returns (inline dict, context).

    A panel that fails costs that panel and lands in the returned ``errors``, which the
    viewer prints for the selected run, so a viewer missing half its figures does not look
    the same as one whose recording had nothing to plot.

    The second return value is what the per-condition views need and the viewer must never
    see: the windowed matrices, the record and the condition windows. It is kept out of the
    payload, which is serialised with ``json.dumps`` and cannot hold a numpy array.
    """
    from nirspipe.qc.figures import (
        build_channel_figure,
        build_epoch_preview_figure,
        build_evoked_topo_figure,
        build_trial_image_by_condition,
        build_trial_image_views,
        build_psd_mean_figure,
        build_sci_psp_figure,
        build_trigger_timeline_single,
        build_ts_figure,
        carpet_gvtd_figure,
        channel_quality_heatmap,
        condition_colors,
        trial_quality_heatmap,
    )

    label   = run["label"]
    session = run.get("session")
    fig_dir = sub_dir / "figures"
    fig_name = figure_namer(label, prefix="raw")
    sqm_dir = sub_dir / (f"ses-{session}" if session else "") / "nirs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sqm_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    notes: list[str] = []

    raw = read_raw_snirf(run["snirf_path"], preload=True, verbose=False)
    # on the run's head template, held so every figure below draws on the same one
    hold_placement(raw, place_optodes(run["snirf_path"],
                                      space_of(run.get("template") or FSAVERAGE)))
    # the reviewer's stretches, marked before anything is measured, as the pipeline marks them
    keep_spans = run.get("keep_spans") or []
    if keep_spans and run.get("keep_spans_from"):
        keep_spans = measured_from(raw, keep_spans, run["keep_spans_from"])
    mark_manual(mark_unselected(raw, keep_spans), run.get("bad_spans") or [])

    sci_scores, raw_od = compute_sci_scores(raw, cardiac_l_freq, cardiac_h_freq)
    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold,
                              good_frac=min_good_frac)
    scope = resolve_screen_scope(raw, screen_scope)
    screen_scores = screening_scores(raw_od, cardiac_l_freq, cardiac_h_freq,
                                     have={"sci": sci_scores}, cutoffs=cutoffs,
                                     scope=scope)
    # a recording screening cannot judge is still shown, said so above its channel table
    gap = screening_gap(raw_od, screen_scores, scope)
    if gap:
        logger.warning("%s | channel screening could not judge this recording: %s", label, gap)
    bad_list = [] if gap else screen_channels(screen_scores, cutoffs)[0]
    tsv_names = run.get("channels_tsv_bads") or []
    joined, kept = join_marks(raw, bad_list, keep=run.get("keep_channels") or [],
                              manual=run.get("bad_channels") or [], listed=tsv_names,
                              label=label)
    bad_channels: set[str] = set(joined)
    tsv_bads = _expand_bad_pairs(raw, tsv_names)

    # The one preprocessing step this report runs, for the motion panels' before and after.
    # Nothing is written back; the corrected copy lives for the length of this function.
    raw_motcorr = None
    if motion_correction and motion_correction != "none":
        with guard("Motion correction", errors, label):
            from nirspipe.pipeline.motion import correct_motion, overmarked
            raw_motcorr = correct_motion(raw_od.copy(), method=motion_correction,
                                         std_thresh=motion_std_thresh, iqr_factor=wavelet_iqr)
            summary = (lineage_of(raw_motcorr).params or {}).get(motion_correction)
            if overmarked(motion_correction, summary):
                note(notes, label, section_note(
                    "caveat.motion_overmarked", method=motion_correction,
                    share=summary["motion_share"], sfreq=raw_od.info["sfreq"],
                    thresh=motion_settings(motion_correction,
                                           motion_std_thresh)["motion_std_thresh"]))

    factors = resolve_dpf(dpf, raw_od, run.get("age"))
    ppf = as_ppf(factors)
    if dpf_from_age(dpf) and (caveat := dpf_caveat(run["age"], wavelengths(raw_od))):
        note(notes, label, caveat)
    # both conversions carry the run's rejections, as the pipeline's haemoglobin stages do
    raw_haemo = None
    with guard("Beer-Lambert", errors, label):
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(
            _marked_copy(raw_od, bad_channels), ppf=ppf)
    if raw_haemo is None:
        note(notes, label, "no haemoglobin conversion, so the per-channel detail panel "
                           "is empty")
    # the corrected copy converted too, only for the correlation's after side
    raw_haemo_post = None
    if raw_motcorr is not None:
        with guard("Beer-Lambert after motion correction", errors, label):
            raw_haemo_post = mne.preprocessing.nirs.beer_lambert_law(
                _marked_copy(raw_motcorr, bad_channels), ppf=ppf)

    sqm: dict = {}
    with guard("Quality metrics", errors, label):
        sqm = compute_raw_sqm(raw, sci_scores, list(bad_channels),
                              cardiac_l_freq, cardiac_h_freq,
                              screen_scores.get("good_frac"))

    # `sqm` stays the flat all-channel view the per-window figures below read. The record
    # written to disk is the sectioned one, built through the same function the pipeline
    # uses, and the panels read it too, so short channels stay out of the long verdict.
    raw_secs, raw_pc = raw_sections(
        raw, sci_scores, list(bad_channels), cardiac_l_freq, cardiac_h_freq, sep_bands,
        screen_scores.get("good_frac"), resp_h_freq=resp_h_freq)
    # HbO-HbR correlation on both conversions above, split by separation as the pipeline's
    # haemoglobin stages are
    for name, haemo in (("rawhaemo", raw_haemo), ("rawhaemo_post", raw_haemo_post)):
        if haemo is not None:
            with guard("HbO-HbR correlation", errors, label):
                secs, pc = haemo_sections(name, haemo, haemo_quality_metrics, sep_bands)
                raw_secs.update(secs)
                raw_pc.update(pc)
    # A window shorter than two screening windows has too few to count, so it is not offered
    # a condition at all. Each condition's passing-window share is in the record's
    # `by_condition`, sliced from the windowed pass. Bound before the guard: it swallows
    # the exception, and the per-condition views read it afterwards.
    cond_windows: list = []
    with guard("Condition windows", errors, label):
        from nirspipe.qc.common.windows import condition_windows
        from nirspipe.qc.metrics.windowed import SCREEN_WINDOW_S
        cond_windows = condition_windows(raw, min_duration=2 * SCREEN_WINDOW_S)
    # measured under every label, but the views and pages below are named by a slug of it;
    # two labels sharing one would overwrite each other, so neither is drawn
    view_windows = cond_windows
    try:
        refuse_colliding_labels([w[0] for w in cond_windows])
    except StageError as exc:
        errors.append(f"Per-condition views: {exc}")
        logger.error("%s | %s", label, exc)
        view_windows = []

    record_view = {**raw_secs, "per_channel": raw_pc}
    ch_rows = channel_rows(record_view, sci_scores, bad_channels)
    # the verdict is read off the long channels wherever the montage was split, exactly as
    # the subject report reads it, so two views of one recording cannot disagree
    view_scalars = raw_secs.get("raw_long") or raw_secs.get("raw") or {}
    heart_rate = view_scalars.get("heart_rate")
    hr_note = heart_rate_note(heart_rate, (cardiac_l_freq, cardiac_h_freq))
    if hr_note:
        note(notes, label, hr_note)
    raw_all = raw_secs.get("raw") or {}
    sqm_split = bool(raw_secs.get("raw_long") and raw_secs.get("raw_short"))

    # Persist windowed series so group_raw can build time × subject heatmaps. Their own
    # section, since `_split_scalars` would file every one of these lists under per_channel.
    windowed: dict = {}
    # GVTD off the corrected file where there is one, as the pipeline's own record does;
    # SCI and PSP stay on the uncorrected file, being coupling.
    series = attach_windowed_series(windowed, raw_od, cardiac_l_freq, cardiac_h_freq,
                                    window_s, gvtd_od=raw_motcorr,
                                    raw_intensity=raw, sep_bands=sep_bands)
    sci_matrix, sci_win_times = series["sci_matrix"], series["sci_times"]
    psp_matrix, psp_win_times = series["psp_matrix"], series["psp_times"]
    # the channel by window matrices too, not just the channel-averaged series: a
    # per-condition number is a column selection out of these, and without them on disk it
    # could only be recomputed. Same keys the pipeline's own record writes.
    _store_matrices(windowed, series)
    _store_spans(windowed, raw, raw_motcorr, sep_bands, errors, label)

    if raw_motcorr is not None:
        with guard("Motion sections", errors, label):
            from nirspipe.qc.subject.sqm_record import _section_writer
            motion_sections(
                raw_od, raw_motcorr, _section_writer(raw_secs, raw_pc), windowed,
                lambda name: (raw_secs.get(name) or {}).get("gvtd_thresh"),
                cardiac_l_freq, cardiac_h_freq, sep_bands)

    set_rows = [
        ("All",   len(sci_scores),                raw_all,                         False),
        ("Long",  raw_all.get("n_long_channels"), raw_secs.get("raw_long") or {},  True),
        ("Short", raw_all.get("n_short_channels"), raw_secs.get("raw_short") or {}, False),
    ]
    split = split_table(set_rows, fold_counts=True) if sqm_split else {}
    # GVTD per channel set, as its own table for the reason MOTION_SPLIT_COLUMNS records.
    # Every set is in `raw`/`raw_long`/`raw_short` already, so this is a second reading of
    # what is on disk rather than a second measurement.
    # the corrected side merged in under `_post`, so each cell prints before -> after where
    # a correction ran. Each set against its own `raw*` cutoff, never another set's.
    # the correction footprint joins unsuffixed: measured across the step rather than either
    # side of it, so it has no before and after to pair
    def _motion_row(raw_key: str, mc_key: str, post_key: str) -> dict:
        return {**(raw_secs.get(raw_key) or {}), **(raw_secs.get(mc_key) or {}),
                **{f"{k}_post": v for k, v in (raw_secs.get(post_key) or {}).items()}}

    motion_rows = [
        ("All",   len(sci_scores),
         _motion_row("raw", "motion", "motion_post"), False),
        ("Long",  raw_all.get("n_long_channels"),
         _motion_row("raw_long", "motion_long", "motion_post_long"), True),
        ("Short", raw_all.get("n_short_channels"),
         _motion_row("raw_short", "motion_short", "motion_post_short"), False),
    ]
    motion_split = (split_table(motion_rows, MOTION_SPLIT_COLUMNS, fold_counts=True)
                    if sqm_split else {})

    # one row per set the record carries, so an unsplit montage still gets its All row
    def _haemo_row(suffix: str) -> dict:
        return {**(raw_secs.get(f"rawhaemo{suffix}") or {}),
                **{f"{k}_post": v
                   for k, v in (raw_secs.get(f"rawhaemo_post{suffix}") or {}).items()}}

    haemo_rows = [
        (name, n, _haemo_row(suffix), colour)
        for name, n, suffix, colour in (
            ("All",   len(sci_scores),                 "",       False),
            ("Long",  raw_all.get("n_long_channels"),  "_long",  True),
            ("Short", raw_all.get("n_short_channels"), "_short", False))
        if raw_secs.get(f"rawhaemo{suffix}")
    ]
    haemo_split = split_table(haemo_rows, HAEMO_SPLIT_COLUMNS) if haemo_rows else {}

    # the data axis, which is what every panel on this page is drawn on and what
    # `score_trials` crops against; `extract_markers` leaves the onsets on the original
    # recording's axis, offset by `first_time` and zero only on an uncropped input
    markers = markers_on_data_axis(raw)
    cond_colors_ = condition_colors(markers)
    for m in markers:
        m["color"] = cond_colors_.get(m["description"], "#f39c12")
    # `build_channel_figure` is the one figure drawn on the original axis, because it is
    # also the one handed a recording cropped in memory (the subject report's condition
    # pages do that); it offsets by `first_time` itself, so it takes the unshifted list.
    # Derived from the recording it will be drawn against, never from the panel above.
    detail_markers = extract_markers(raw)

    psp_per_ch    = sqm.get("psp_per_channel", {})
    figure_paths: dict = {}

    # ── inline: ts figure (kept in-memory for click interactivity) ─────────────
    ts_inline: dict = {}
    with guard("Raw signal", errors, label):
        fig, _mkdata, cond_colors_out, band_shapes, t_start, t_end = build_ts_figure(
            raw, markers, bad_channels, _MAX_TS_PTS, sep_bands,
        )
        ts_inline = {
            "figure":      fig.to_dict(),
            "markers":     markers,
            "cond_colors": cond_colors_out,
            "band_shapes": band_shapes,
            "t_start":     t_start,
            "t_end":       t_end,
        }

    # ── file: carpet GVTD ──────────────────────────────────────────────────
    #   an iframe rather than inlined like the panels above: the carpet is a channels x 2000
    #   heatmap, and every run of the viewer would carry one in the page itself
    carpet_inline: dict = {}
    # named out here because the per-channel motion figures below read them too: a channel's
    # GVTD row has to be its own separation class's, the same blocks the carpet drew
    gvtd_blocks: "list[tuple[str, list[str]]]" = []
    # the source snirf still carries its aux group; empty where it has no IMU
    imu: dict = {}
    with guard("IMU", errors, label):
        imu = imu_traces(*read_aux_snirf(run["snirf_path"]))
    imu_sec: dict = {}
    if imu:
        with guard("IMU summary", errors, label):
            imu_sec = imu_section(imu, raw, sep_bands)
            # on the grid the GVTD series above were binned on, from the file they came off
            gvtd_od = raw_motcorr if raw_motcorr is not None else raw_od
            windowed.update(imu_windowed(imu, gvtd_od.times, float(gvtd_od.info["sfreq"]),
                                         window_s))
    with guard("GVTD carpet", errors, label):
        from nirspipe.qc.metrics import gvtd_channel_blocks
        gvtd_blocks = gvtd_channel_blocks(raw, sep_bands)
        gvtd_set = gvtd_blocks[0][0]
        gvtd_picks = [c for _, names in gvtd_blocks for c in names]
        raw_carpet = raw.copy().pick(gvtd_picks)
        # before and after in one panel where a correction ran, which is the pairing the
        # subject report's carpet draws and the reason the flag exists
        carpet_after = (None if raw_motcorr is None
                        else raw_motcorr.copy().pick(
                            [c for c in gvtd_picks if c in raw_motcorr.ch_names]))
        corrected = _span_sets(windowed, "motion_corrected_spans", gvtd_set)[gvtd_set]
        fig   = carpet_gvtd_figure(raw_carpet, raw_carpet.ch_names,
                                   corrected_segments=corrected,
                                   spike_segments=_span_sets(windowed, "spike_spans", gvtd_set),
                                   raw_after=carpet_after,
                                   channel_set=gvtd_set, blocks=gvtd_blocks, imu=imu,
                                   roi_map=roi_map)
        fname = fig_name("carpet")
        # not written per condition: its GVTD filter, z-scale and colour scale are run-wide,
        # so one file is narrowed by URL fragment
        h     = _save_figure_html(fig, fig_dir / fname,
                                  views=_carpet_views(fig, view_windows))
        figure_paths["carpet"] = {"src": f"figures/{fname}", "h": h}
        if roi_map:
            # the set whose rows the ROI strip groups, named by the note under the heading
            figure_paths["carpet"]["roi_set"] = gvtd_set
        carpet_inline = {"figure": fig.to_dict()}

    # ── file: SCI / PSP ────────────────────────────────────────────────────────
    sci_psp_inline: dict = {}
    with guard("SCI / PSP", errors, label):
        # the CV row too, off the same windowed pass and the same grid, as the subject
        # report's copy of this panel draws it
        fig   = build_sci_psp_figure(
            sci_scores, psp_per_ch, bad_channels, sci_threshold,
            psp_threshold=cutoffs["psp"], window_s=window_s,
            sci_matrix=sci_matrix, sci_win_times=sci_win_times,
            psp_matrix=psp_matrix, psp_win_times=psp_win_times,
            cv_per_channel=(raw_pc.get("raw") or {}).get("cv_per_channel") or {},
            cv_matrix=series.get("cv_matrix"), cv_win_times=series.get("cv_times"),
            warn=errors,
        )
        fname = fig_name("scipsp")
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["sci_psp"] = {"src": f"figures/{fname}", "h": h}
        sci_psp_inline = {"figure": fig.to_dict()}

    # ── file: head slider ──────────────────────────────────────────────────────
    # the subject report's own, off the same matrices as the panel above
    remake_head_slider = _head_slider_maker(raw, series, list(sci_scores), cutoffs, sep_bands,
                                            run["label"], fig_dir)
    slider = remake_head_slider(fig_name, None, errors)
    if slider:
        figure_paths["head_slider"] = slider

    # ── file: PSD mean ─────────────────────────────────────────────────────────
    psd_inline: dict = {}
    with guard("PSD", errors, label):
        fig = build_psd_mean_figure(raw, cardiac=(cardiac_l_freq, cardiac_h_freq),
                                    resp=(resp_l_freq, resp_h_freq), sep_bands=sep_bands,
                                    heart_rate=heart_rate)
        if fig:
            fname = fig_name("psd")
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["psd"] = {"src": f"figures/{fname}", "h": h}
            psd_inline = {"figure": fig.to_dict()}

    # ── file: trigger timeline ─────────────────────────────────────────────────
    trigger_timeline_inline: dict = {}
    with guard("Trigger timeline", errors, label):
        fig = build_trigger_timeline_single(markers, cond_colors_)
        if fig:
            fname = fig_name("trigger")
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["trigger"] = {"src": f"figures/{fname}", "h": h}
            trigger_timeline_inline = {"figure": fig.to_dict()}

    # ── file: channel quality summary ──────────────────────────────────────────
    ch_summary_inline: dict = {}
    with guard("Channel quality summary", errors, label):
        # long block first with the divider between the two, from the same helper the
        # subject report uses, so the grid and the per-channel table below it read in one
        # order and a short channel never lands in a long channel's verdict
        fig = channel_quality_heatmap(sci_thresh=cutoffs["sci"], psp_thresh=cutoffs["psp"],
                                      good_frac_thresh=cutoffs["good_frac"],
                                      **heatmap_args(ch_rows))
        fname = fig_name("chsummary", suffix="qc")
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["ch_summary"] = {"src": f"figures/{fname}", "h": h}
        ch_summary_inline = {"figure": fig.to_dict()}

    # the epoch figures need concrete bounds; per-trial QC reads None as "use event duration"
    fig_tmin = _EPOCH_TMIN if epoch_tmin is None else epoch_tmin
    fig_tmax = _EPOCH_TMAX if epoch_tmax is None else epoch_tmax
    epoch_view = None
    if raw_haemo is not None:
        with guard("Epoch display low-pass", errors, label):
            epoch_view = _below_cardiac(raw_haemo, cardiac_l_freq)

    # ── file: grand mean ───────────────────────────────────────────────────────
    # One row per condition, every kept long channel averaged with the short ones dotted.
    if epoch_view is not None:
        with guard("Grand mean", errors, label):
            fig = build_epoch_preview_figure(epoch_view, epoch_tmin=fig_tmin,
                                             epoch_tmax=fig_tmax, sep_bands=sep_bands)
            if fig:
                fname = fig_name("epochmean")
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["epoch_mean"] = {"src": f"figures/{fname}", "h": h}

    # ── file: per-channel motion detail ────────────────────────────────────────
    # The subject report's own figures, builder and saver, so the two reports cannot
    # disagree about what a channel did. One file per channel, carrying every
    # condition's window, so a condition page narrows the view instead of remeasuring.
    motion_channels: list[str] = []
    if raw_motcorr is not None:
        with guard("Motion detail", errors, label):
            from nirspipe.qc.subject.report import (
                _motion_detail_figures, _section_motion_detail,
            )
            built = _motion_detail_figures(
                raw_od, raw_motcorr, label, errors,
                corrected_by_set=_span_sets(windowed, "motion_corrected_spans",
                                            gvtd_blocks[0][0] if gvtd_blocks else "all"),
                spike_by_set=_span_sets(windowed, "spike_spans",
                                        gvtd_blocks[0][0] if gvtd_blocks else "all"),
                gvtd_blocks=gvtd_blocks or None,
                imu=imu,
            )
            if built:
                saved = _section_motion_detail(
                    built, label, errors, fig_dir, fig_name,
                    condition_spans=view_windows,
                )["motion_detail_pairs"]
                motion_channels = [entry["pair"] for entry in saved]
                # every one of these is the same four-row layout, so one height serves the
                # panel and the viewer does not carry one per channel
                figure_paths["motion_detail_template"] = {
                    "src": f"figures/{fig_name('motion', channel='{ch}')}",
                    "h": saved[0]["h"] if saved else 700,
                }

    # ── file: HbO-HbR correlation ──────────────────────────────────────────────
    if raw_haemo is not None:
        with guard("HbO-HbR correlation panel", errors, label):
            from nirspipe.qc.figures import hbo_hbr_fit_js
            fig = _hbo_hbr_figure(raw_haemo, raw_haemo_post, sep_bands, motion_correction,
                                  {pair_of(c) for c in bad_channels})
            if fig is not None:
                fname = fig_name("hbohbrcorr")
                h     = _save_figure_html(fig, fig_dir / fname, extra_js=hbo_hbr_fit_js(fig))
                figure_paths["hbo_hbr_corr"] = {"src": f"figures/{fname}", "h": h}

    # ── file: per-channel trial images and Signal by position ──────────────────
    # Trials down the rows, so a channel that was fine for the first half and lost for the
    # second reads as a band rather than being averaged away. Split by event name, pooled
    # first, as the subject report's is; Signal by position beside it picks the channel.
    trial_img_pairs: list = []
    trial_img_by_cond: dict = {}
    if epoch_view is not None:
        for ch in [c for c in epoch_view.ch_names if c.endswith(" hbo")]:
            with guard("Trial image", errors, f"{label} | {ch}"):
                views = build_trial_image_views(epoch_view, ch, fig_tmin, fig_tmax)
                if not views:
                    continue
                pair = pair_of(ch)
                fname = fig_name("trialimage", channel=_pair_fname(pair))
                h = save_fragment_views_html(views, fig_dir / fname,
                                             section_note("epoch.trial_image_missing"))
                trial_img_pairs.append(
                    {"pair": pair, "src": f"figures/{fname}", "h": h})
        if trial_img_pairs:
            with guard("Signal by position", errors, label):
                fig = build_evoked_topo_figure(epoch_view, markers_on_data_axis(epoch_view),
                                               _MAX_TS_PTS, fig_tmin, fig_tmax)
                if fig is not None:
                    fname = fig_name("signaltopo", suffix="nirsmap")
                    h = _save_figure_html(fig, fig_dir / fname,
                                          extra_js=TOPO_PICK_JS + FILL_FRAME_JS)
                    figure_paths["signal_topo"] = {"src": f"figures/{fname}", "h": h}
    # a condition page's own panel, split by condition window rather than event name: a
    # page holds one window, and a block design's trials all share one name
    if epoch_view is not None and view_windows:
        for ch in [c for c in epoch_view.ch_names if c.endswith(" hbo")]:
            with guard("Condition trial image", errors, f"{label} | {ch}"):
                by_label = build_trial_image_by_condition(
                    epoch_view, ch, view_windows, fig_tmin, fig_tmax)
                for cond_label, cond_figs in (by_label or {}).items():
                    trial_img_by_cond.setdefault(cond_label, []).append((pair_of(ch), cond_figs))

    # ── file: per-channel detail HTML ──────────────────────────────────────────
    # one file per pair, holding every condition's view: `condition_views` names the windows
    # and the file picks one off its URL fragment, so a condition page links to the run's
    # figure rather than to a copy of it
    channel_pairs: list[str] = []
    channel_labels: dict[str, str] = {}
    if raw_haemo is not None:
        # this one figure is drawn on the original recording's axis, so the windows are
        # shifted onto it before its views are measured
        detail_origin = float(raw.first_time)
        detail_spans = [(lab, t0 + detail_origin, t1 + detail_origin)
                        for lab, t0, t1 in view_windows]
        # rejected pairs too, named as such, as the subject report's picker has them
        from nirspipe.qc.subject.report import _pair_label
        channel_pairs = get_channel_pairs(raw_haemo, exclude=())
        short_pairs = {pair_of(n) for n in long_short_channels(raw_haemo, sep_bands)[1]}
        rejected = {pair_of(c) for c in bad_channels}
        channel_labels = {p: _pair_label(p, p in short_pairs, p in rejected)
                          for p in channel_pairs}
        for pair in channel_pairs:
            with guard("Channel detail", errors, f"{label} | {pair}"):
                # no epoch panel: the trial images carry each condition's HbO and HbR
                # average, low-passed, where this stage would draw cardiac ripple
                detail_fig, psd_fig, _ = build_channel_figure(
                    raw_haemo, detail_markers, pair, _MAX_TS_PTS, fig_tmin, fig_tmax,
                    cardiac=(cardiac_l_freq, cardiac_h_freq),
                    resp=(resp_l_freq, resp_h_freq), epoch=False,
                )
                fname = fig_name("detail", channel=_pair_fname(pair))
                _save_multi_fig_html(
                    [detail_fig, psd_fig], fig_dir / fname,
                    views=condition_view_table(detail_fig, detail_spans))
        if channel_pairs:
            figure_paths["ch_detail_template"] = (
                f"figures/{fig_name('detail', channel='{pair}')}"
            )

    # `channel_pairs or None` so a run whose Beer-Lambert failed still gets a table, built
    # from the pairs the intensity recording carries rather than from an empty list
    pair_cells = format_rows(pair_rows(ch_rows, channel_pairs or None, by_wavelength=True),
                             sci_threshold, name_key="pair", psp_threshold=cutoffs["psp"])

    # ── file: per-trial QC ─────────────────────────────────────────────────────
    # scored here rather than persisted: a trial is not a BIDS entity, so per-trial records
    # have nowhere to live in the derivatives tree without colliding on filename
    trial_qc_inline: dict = {}
    trial_rows: list = []
    if epoch_qc:
        with guard("Per-trial quality", errors, label):
            labels, sqms = score_trials(raw, markers, sci_threshold,
                                        cardiac_l_freq, cardiac_h_freq,
                                        epoch_tmin, epoch_tmax,
                                        psp_threshold=cutoffs["psp"],
                                        min_good_frac=cutoffs["good_frac"])
            # the onset beside each scored trial, so a condition page takes its own rows out
            # of this table rather than scoring the same windows a second time
            windows = trial_windows(markers, epoch_tmin, epoch_tmax, float(raw.times[-1]))
            trial_rows = [(onset, lab, sqm) for (_, _, _, onset), lab, sqm in zip(
                windows, labels, sqms)]
            partial, unscored = short_trial_counts(windows, float(raw.info["sfreq"]))
            if partial:
                note(notes, label, section_note("caveat.partial_trials", n=partial,
                                                total=len(windows), floor=MIN_SCI_TRIAL_S,
                                                window=MIN_TRIAL_S))
            if unscored:
                note(notes, label, section_note("caveat.short_trials", n=unscored,
                                                total=len(windows), window=MIN_SCI_TRIAL_S))
            fig = trial_quality_heatmap(labels, sqms)
            if fig:
                fname = fig_name("trialqc", suffix="qc")
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["trial_qc"] = {"src": f"figures/{fname}", "h": h}
                trial_qc_inline = {"figure": fig.to_dict(), "n_trials": len(labels)}
            elif not labels:
                note(notes, label, "no event window fits inside the recording, so "
                                   "per-trial quality was skipped")

    # ── file: SQM JSON ─────────────────────────────────────────────────────────
    # desc-sqmraw, not desc-sqm: the pipeline writes a record at the latter path for the
    # same run, and one silently overwriting the other loses whichever ran first. Same
    # shape as that one, so the group table reads both through one path.
    sqm_path = sqm_dir / (label + RECORD_SUFFIXES["sqmraw"])
    sections = {**raw_secs, **({"imu": imu_sec} if imu_sec else {}),
                "windowed": windowed, "per_channel": raw_pc}
    # written whenever the recording carries conditions, independently of `by_condition`:
    # the flag decides what a report shows, the record says what was measured. Deliberately
    # not in `SECTIONS`, so the group table does not descend into it.
    by_cond: dict = {}
    if cond_windows:
        with guard("Per-condition metrics", errors, label):
            by_cond = raw_condition_sections(sections, raw, cond_windows, cutoffs, sep_bands,
                                             imu=imu, haemo=raw_haemo,
                                             haemo_post=raw_haemo_post)
    record = sqm_record_dict(sections, bids_uris([run["snirf_path"]], sqm_path))
    if by_cond:
        record["by_condition"] = by_cond
    # what the dyad level reads this member by, so it never screens or converts it again
    record["bad_channels"] = sorted(bad_channels)
    record["screening_gap"] = gap
    record["parameters"] = {
        "sci_threshold": cutoffs["sci"],
        "psp_threshold": cutoffs["psp"],
        "min_good_frac": cutoffs["good_frac"],
        "screen_scope": screen_scope,
        "screen_scope_counted": "task" if scope else "run",
        "cardiac_l_freq": cardiac_l_freq,
        "cardiac_h_freq": cardiac_h_freq,
        "resp_l_freq": resp_l_freq,
        "resp_h_freq": resp_h_freq,
        "qc_window_s": window_s,
        "dpf": [float(f) for f in factors],
        **({"dpf_age": run["age"]} if dpf_from_age(dpf) else {}),
        **({"dpf_population": run["population"]}
           if "dpf" in (run.get("from_population") or ()) else {}),
        "template": run.get("template") or FSAVERAGE,
        "motion_correction": motion_correction or "none",
        **motion_settings(motion_correction, motion_std_thresh, wavelet_iqr),
        "bad_channels": list(run.get("bad_channels") or []),
        "keep_channels": list(run.get("keep_channels") or []),
        "kept_channels": kept,
        "channels_tsv_bads": list(tsv_names),
        "ignore": list(run.get("ignore") or []),
        "keep_spans": [list(span) for span in keep_spans],
        "keep_spans_from": run.get("keep_spans_from"),
        "bad_spans": [list(span) for span in run.get("bad_spans") or []],
        **({"population": run["population"],
            "from_population": list(run["from_population"])}
           if run.get("from_population") else {}),
    }
    write_record(sqm_path, record)
    logger.info("SQM JSON -> %s", sqm_path)
    save_channel_csv(ch_rows, label, sqm_dir, sci_threshold, psp_threshold=cutoffs["psp"],
                     suffix=RAW_CHANNEL_METRICS_SUFFIX, fields=RAW_CSV_FIELDS)

    n_total = len(sci_scores)
    bad_rate = 100 * len(bad_channels) / n_total if n_total else 0.0
    loose_rows = metric_rows({**view_scalars, **imu_sec},
                             _VIEW_MONTAGE_KEYS if sqm_split else _VIEW_SCALAR_KEYS,
                             skip_missing=True)
    return {
        # what the page opens with. Per run, because a viewer holding several switches
        # between them and the rejected count is the first thing that differs.
        "summary": {
            "run": label,
            "n_bad": len(bad_channels),
            "n_total": n_total,
            "bad_rate": bad_rate,
            "badge_class": ("badge-green" if bad_rate < 10
                            else "badge-yellow" if bad_rate < 30 else "badge-red"),
            "sci_threshold": cutoffs["sci"],
            "good_frac": cutoffs["good_frac"],
            "dpf": dpf_text(dpf, wavelengths(raw_od), run.get("age")),
            "cardiac": [cardiac_l_freq, cardiac_h_freq],
            "heart_rate": heart_rate,
            "heart_rate_outside": hr_note is not None,
            "resp": [resp_l_freq, resp_h_freq],
            "motion_correction": motion_correction or "none",
            "scope": (section_note("raw.screened_long") if sqm_split
                      else section_note("raw.screened_every")),
        },
        "ts":           ts_inline,
        "carpet_gvtd":  carpet_inline,
        "sci_psp":          sci_psp_inline,
        "psd":              psd_inline,
        "ch_summary":       ch_summary_inline,
        "trigger_timeline": trigger_timeline_inline,
        "trial_qc":         trial_qc_inline,
        # already labelled, formatted and coloured by the metric registry, so the views
        # print these and carry no copy of the cutoffs. The panels read the record's own
        # channel-set sections, and the record on disk is where the raw numbers live.
        "sqm": {
            "rows": loose_rows,
            # one table, as the subject report prints it: a metric per row, a set per column
            "table": _sqm_table({"rows": loose_rows, "split": split,
                                 "motion_split": motion_split, "haemo_split": haemo_split},
                                n_total),
            "split":        split,
            "motion_split": motion_split,
            "haemo_split":  haemo_split,
            "channel_set":  "long channels" if sqm_split else "every channel",
        },
        # each --roi-mapping region's long channels kept, as the subject report lists them
        "regions": _regions_view(ch_rows, roi_map),
        # One table, at pair granularity: a decision is taken per source-detector pair. The
        # per-wavelength numbers are in the CSV written next to the record.
        "channels": {
            "pairs":  pair_cells,
            "wavelengths": pair_wavelengths(ch_rows),
            "blocks": separation_blocks(pair_cells, sep_bands),
            # sep_bands, or a run with non-default bands gets the default gap quoted at it
            "notes":  [n for n in (section_note("caveat.unscreened", reason=gap) if gap else "",
                                   placement_note(raw),
                                   template_note(run.get("population"), run.get("template")),
                                   *separation_notes(raw_all, ch_rows, sep_bands=sep_bands,
                                                     orphan_mm=separation_orphans(raw, sep_bands)),
                                   section_note(
                                       "raw.channels_tsv",
                                       pairs=", ".join(sorted({pair_of(c) for c in tsv_bads})))
                                   if tsv_bads else "")
                       if n],
        },
        "channel_pairs": channel_pairs,
        "channel_labels": channel_labels,
        # one entry per channel that got a motion figure, for that panel's own picker
        "motion_channels": motion_channels,
        # one entry per channel that had trials to draw, for this panel's own picker
        "trial_images":  trial_img_pairs,
        "figure_paths":  figure_paths,
        # the GUI builds its per-channel figures on demand and needs the run's own band to
        # shade the PSD the way the ones built here are shaded
        "cardiac":       [cardiac_l_freq, cardiac_h_freq],
        # what this run failed at and what it left out, already collapsed and ready to
        # print; the viewer renders these itself, in JavaScript
        "errors": collapse_messages(errors),
        "notes":  collapse_messages(notes),
    }, {
        # where the per-condition numbers are, rather than the numbers: the pages read the
        # record off disk, the way the subject report's do
        "remake_psd":    _psd_maker(raw, (cardiac_l_freq, cardiac_h_freq),
                                    (resp_l_freq, resp_h_freq), sep_bands,
                                    build_psd_mean_figure, heart_rate),
        "remake_epoch":  _epoch_maker(epoch_view, fig_tmin, fig_tmax, sep_bands,
                                      build_epoch_preview_figure),
        "remake_hbo_hbr_corr": _hbo_hbr_maker(raw_haemo, raw_haemo_post, sep_bands,
                                              motion_correction,
                                              {pair_of(c) for c in bad_channels}),
        # drawn by the page writer, not here: the GUI shares this function and shows no copy
        "remake_brain":  _brain_maker(raw, raw_haemo, bad_channels, label, fig_dir,
                                      sci_threshold=cutoffs["sci"], sep_bands=sep_bands),
        "brain_flat":    not placement_of(raw).placed,
        "remake_head_slider": remake_head_slider,
        "sci_win_scores": (raw_pc.get("raw") or {}).get("sci_win_per_channel") or {},
        "trial_images_by_condition": trial_img_by_cond,
        "sep_bands":     sep_bands,
        "has_spans":     bool(bad_spans(raw)),
        "sqm_path":      sqm_path,
        "fig_dir":       fig_dir,
        "sci_scores":    sci_scores,
        "bad_channels":  bad_channels,
        "channel_pairs": channel_pairs,
        "series":        series,
        "cutoffs":       cutoffs,
        "trial_rows":    trial_rows,
    }


def _regions_view(rows: list[dict], roi_map: "dict | None") -> dict:
    """The region table's rows and its not-counted note, ready to print."""
    table = region_table(rows, roi_map)
    return {"rows": table["rows"],
            "note": (section_note("metrics.regions_not_counted", pairs=table["not_counted"])
                     if table["not_counted"] else "")}


def _marked_copy(raw_od, bad_channels: "set[str]"):
    marked = raw_od.copy()
    marked.info["bads"] = [c for c in marked.ch_names if c in bad_channels]
    return marked


def _hbo_hbr_figure(haemo, haemo_post, sep_bands, method, bad_pairs):
    """The subject report's correlation panel, with motion correction as the step between."""
    from nirspipe.qc.figures import hbo_hbr_correlation_figure
    return hbo_hbr_correlation_figure(
        haemo, title="HbO–HbR correlation", sep_bands=sep_bands, raw_after=haemo_post,
        stage_labels=("before motion correction", f"after {method}"), bad_pairs=bad_pairs)


def _hbo_hbr_maker(haemo, haemo_post, sep_bands, method, bad_pairs):
    """``(t0, t1) -> figure`` on both conversions cut to one condition, or None without one.

    Cut rather than sliced, as the record's own numbers are: a correlation reads only the
    samples it is handed.
    """
    if haemo is None:
        return None

    def remake(t0: float, t1: float):
        lo, hi = max(0.0, float(t0)), min(float(haemo.times[-1]), float(t1))
        if hi <= lo:
            return None
        after = None if haemo_post is None else haemo_post.copy().crop(tmin=lo, tmax=hi)
        return _hbo_hbr_figure(haemo.copy().crop(tmin=lo, tmax=hi), after, sep_bands, method,
                               bad_pairs)

    return remake


def _brain_maker(raw, raw_haemo, bad_channels, label: str, fig_dir: Path, *,
                 sci_threshold: float, sep_bands=None):
    """``(namer, SCI per channel, errors) -> href`` of the subject report's brain views.

    The haemoglobin file's HbO channels place the channels, rejected ones included, as the
    pipeline places them for its own copy. A condition page hands in its own SCI and a namer
    carrying its slug; the rejections stay the run's.
    """
    if raw_haemo is None:
        return None
    # a pyvista render, kept out of module import like the report it comes from
    from nirspipe.qc.subject.report import _section_brain

    picks = mne.pick_types(raw_haemo.info, fnirs="hbo", exclude=[])
    names = [raw_haemo.ch_names[i] for i in picks]
    coords = np.array([raw_haemo.info["chs"][i]["loc"][:3] for i in picks])
    bad_pairs = {pair_of(c) for c in bad_channels}
    good = np.array([pair_of(n) not in bad_pairs for n in names])

    def remake(namer, sci: dict, errors: list) -> "str | None":
        return _section_brain(sci, sorted(bad_channels), coords, good, raw, label, errors,
                              fig_dir, namer, ch_names_brain=names,
                              sci_threshold=sci_threshold, sep_bands=sep_bands)["brain_views_path"]

    return remake


def _head_slider_maker(raw, series: dict, channels: list, cutoffs: dict, sep_bands,
                       label: str, fig_dir: Path):
    """``(namer, window, errors) -> {"src", "h"}`` of the head slider, or None.

    ``window`` cuts the stored matrices to one condition's windows, as the SCI/PSP panel's
    condition copy is cut; None keeps the whole run.
    """
    # imported where used, as every reach into the subject report module here is
    from nirspipe.qc.subject.report import _section_head_slider

    def remake(namer, window, errors: list) -> "dict | None":
        sci, psp, times = (series.get(k) for k in ("sci_matrix", "psp_matrix", "sci_times"))
        if sci is None or psp is None or times is None:
            return None
        sci, psp, times = np.asarray(sci), np.asarray(psp), np.asarray(times)
        if window is not None:
            keep = _in_scope(window_centers(times), [window])
            if not keep.any():
                return None
            sci, psp, times = sci[:, keep], psp[:, keep], times[keep]
        out = _section_head_slider(raw, sci, psp, channels, times, cutoffs, sep_bands,
                                   label.split("_")[0].removeprefix("sub-"), errors, fig_dir,
                                   namer)
        return ({"src": out["head_slider_path"], "h": out["head_slider_h"]}
                if out["head_slider_path"] else None)

    return remake


def _below_cardiac(raw_haemo, cardiac_l_freq: float):
    """A copy for the epoch figures only, low-passed to stop at the cardiac band's lower edge.

    This stage is otherwise unfiltered, and an average of a few trials keeps the heartbeat as
    a ripple on a curve read for its shape. e.g. ``--cardiac-l-freq 0.7`` passes up to 0.56 Hz
    and stops from 0.7 Hz.
    """
    return raw_haemo.copy().filter(None, 0.8 * cardiac_l_freq,
                                   h_trans_bandwidth=0.2 * cardiac_l_freq, verbose="error")


def _epoch_maker(raw_haemo, tmin: float, tmax: float, sep_bands, build):
    """``(t0, t1) -> figure`` for one condition's grand mean, or None without one.

    Rebuilt on a crop rather than sliced: the figure epochs the recording itself, so there
    is no windowed series to take a column out of. Safe for the reason the spectrum is,
    nothing in an epoch average reading outside the samples it is handed. The crop keeps the
    epoch window's own room after the last onset, or the final block loses its epoch.
    """
    if raw_haemo is None:
        return None

    def remake(t0: float, t1: float):
        lo = max(0.0, float(t0) + min(0.0, tmin))
        hi = min(float(raw_haemo.times[-1]), float(t1) + max(0.0, tmax))
        if hi <= lo:
            return None
        return build(raw_haemo.copy().crop(tmin=lo, tmax=hi),
                     epoch_tmin=tmin, epoch_tmax=tmax, sep_bands=sep_bands)

    return remake


def _psd_maker(raw, cardiac: "tuple[float, float]", resp: "tuple[float, float]", sep_bands,
               build, heart_rate: "float | None" = None):
    """``(t0, t1) -> figure`` for one condition's own spectrum, or None when the cut is short.

    Recomputed on a crop rather than sliced, because there is nothing to slice: it is one
    spectrum, not a time-by-frequency matrix. Safe to recompute for the reason the subject
    report's condition pages recompute theirs: ``compute_psd`` is Welch, which segments and
    tapers but does not band-pass, so a cut carries no filter edge the whole run would not
    have had. What a cut does change is resolution, and only once it is shorter than the
    transform: below ``PSD_NFFT_CAP`` samples it would land on a coarser frequency grid than
    the run's and is left out instead, at the same floor the record stops writing its band
    scalars at.
    """
    def remake(t0: float, t1: float):
        lo, hi = max(0.0, float(t0)), min(float(raw.times[-1]), float(t1))
        if hi <= lo:
            return None
        cut = raw.copy().crop(tmin=lo, tmax=hi)
        if len(cut.times) < PSD_NFFT_CAP:
            logger.info("condition %.1f-%.1f s is %d samples, under the %d the transform "
                        "needs; no spectrum", lo, hi, len(cut.times), PSD_NFFT_CAP)
            return None
        return build(cut, cardiac=cardiac, resp=resp, sep_bands=sep_bands,
                     heart_rate=heart_rate)

    return remake


def _condition_summary(ctx: dict, payload: dict, run_label: str) -> None:
    """Every condition's channel grid on the run page, regrouped by metric.

    The subject report's figure, from the conditions the record carries; a condition with no
    counted window has no assessment and is left out. Adds ``cond_summary`` to the run's
    ``figure_paths`` when there are two conditions to compare.
    """
    from nirspipe.qc.figures import condition_quality_heatmap
    from nirspipe.qc.subject.condition_views import slice_record, with_condition_corr

    record = read_record(Path(ctx["sqm_path"]))
    by_condition = record.get("by_condition") or {}
    conditions = [
        (label, heatmap_args(channel_rows(
            with_condition_corr(slice_record(record, entry.get("per_channel") or {}),
                                (entry.get("per_channel") or {}).get("hbo_hbr_corr_per_channel"),
                                section="rawhaemo"),
            ctx["sci_scores"], set(entry.get("bad_channels") or ()))))
        for label, entry in by_condition.items() if entry.get("counted_windows") != 0]
    cutoffs = ctx["cutoffs"]
    fig = condition_quality_heatmap(conditions, sci_thresh=cutoffs["sci"],
                                    psp_thresh=cutoffs["psp"],
                                    good_frac_thresh=cutoffs["good_frac"],
                                    status_label="In condition", status_words=("pass", "fail"))
    if fig is None:
        return
    fname = figure_namer(run_label, prefix="raw")("condsummary", suffix="qc")
    h = _save_figure_html(fig, Path(ctx["fig_dir"]) / fname)
    payload.setdefault("figure_paths", {})["cond_summary"] = {"src": f"figures/{fname}", "h": h}


def _write_condition_views(ctx: dict, payload: dict, output_path: Path, run_label: str,
                           sci_threshold: float) -> None:
    """One report file per condition, beside the run's own, read out of the quality record.

    Every number on these pages comes from the record's ``by_condition`` section, which
    :func:`~nirspipe.qc.subject.sqm_record.raw_condition_sections` wrote a moment earlier; nothing
    is measured here. A record carrying no such section gets no pages rather than a second
    copy of the numbers free to disagree with the first.

    Each page is the same viewer with a single entry, so nothing about how these are read has
    to be learned twice. The file name comes from :func:`condition_page_name`, keyed by the
    run's own label and not the report's, since one report holds every run of a task.
    """
    from nirspipe.qc.figures import hbo_hbr_fit_js
    from nirspipe.qc.subject.condition_views import condition_payloads
    from nirspipe.qc.subject.report import condition_page_name

    sqm_path = ctx.get("sqm_path")
    if sqm_path is None or not Path(sqm_path).exists():
        logger.warning("%s | no quality record; no per-condition pages", run_label)
        return
    record = read_record(Path(sqm_path))
    by_condition = record.get("by_condition") or {}
    if not by_condition:
        logger.info("%s | the record carries no by_condition section; no per-condition "
                    "pages", run_label)
        return
    try:
        refuse_colliding_labels(list(by_condition))
    except StageError as exc:
        # the run's own page already lists it; its views were dropped for the same reason
        logger.error("%s | no per-condition pages: %s", run_label, exc)
        return

    fig_dir = Path(ctx["fig_dir"])

    def save_figure(panel: str, slug: str, fig) -> "dict | None":
        """One condition's own figure file, named so `figure_leaks` can recognise it."""
        desc = panel.replace("_", "")
        # the two quality tables carry qc, as the run page names them
        fname = figure_namer(run_label, slug, prefix="raw")(
            desc, suffix="qc" if desc in ("chsummary", "trialqc") else "nirs")
        # the correlation panel fits its square matrix to the page, as the run's copy does
        extra_js = hbo_hbr_fit_js(fig) if panel == "hbo_hbr_corr" else ""
        h = _save_figure_html(fig, fig_dir / fname, extra_js=extra_js)
        return {"src": f"figures/{fname}", "h": h}

    remake_brain = ctx.get("remake_brain")
    remake_slider = ctx.get("remake_head_slider")
    # what the remade panels failed at, per condition page
    panel_errors: "dict[str, list]" = {}

    def save_brain(slug: str, sci: dict) -> "dict | None":
        """One condition's brain figure, graded by its own SCI over the run's rejections."""
        href = remake_brain(figure_namer(run_label, slug, prefix="raw"), sci,
                            panel_errors.setdefault(slug, []))
        return {"src": href, "flat": ctx.get("brain_flat", False)} if href else None

    def save_head_slider(slug: str, window) -> "dict | None":
        """One condition's head slider, on the run's matrices cut to its windows."""
        return remake_slider(figure_namer(run_label, slug, prefix="raw"), window,
                             panel_errors.setdefault(slug, []))

    def save_stack(panel: str, slug: str, name: str, figs) -> "dict | None":
        """The same, for a panel whose file holds several figures of one channel."""
        fname = figure_namer(run_label, slug, prefix="raw")(
            panel.replace("_", ""), channel=_pair_fname(name))
        h = _save_multi_fig_html(list(figs), fig_dir / fname)
        return {"pair": name, "src": f"figures/{fname}", "h": h}

    views = condition_payloads(
        payload, record=record, by_condition=by_condition,
        sci_scores=ctx["sci_scores"], bad_channels=ctx["bad_channels"],
        channel_pairs=ctx["channel_pairs"], series=ctx["series"],
        sci_threshold=sci_threshold, cutoffs=ctx["cutoffs"],
        trial_rows=ctx.get("trial_rows"), save_figure=save_figure,
        remake_psd=ctx.get("remake_psd"), remake_epoch=ctx.get("remake_epoch"),
        remake_hbo_hbr_corr=ctx.get("remake_hbo_hbr_corr"),
        remake_brain=save_brain if remake_brain else None,
        remake_head_slider=save_head_slider if remake_slider else None,
        trial_images=ctx.get("trial_images_by_condition"),
        save_stack=save_stack,
        sep_bands=ctx.get("sep_bands"),
        has_spans=bool(ctx.get("has_spans")),
    )
    if not views:
        return
    written: set[str] = set()
    for label, view in views:
        slug = _pair_fname(label)
        # the slug also names this condition's figures and URL fragments, so a page of its
        # own could only show another condition's files under this one's numbers
        if not slug or slug in written:
            logger.warning("%s | condition %r reduces to the slug %r, empty or taken by an "
                           "earlier one; no page for it", run_label, label, slug)
            continue
        written.add(slug)
        if panel_errors.get(slug):
            view["errors"] = list(view.get("errors") or []) + collapse_messages(panel_errors[slug])
        out = output_path.with_name(condition_page_name(run_label, label, desc="raw"))
        view_label = f"{run_label}_cond-{slug}"
        # the run page's one table, built here from the condition's own set tables
        view.setdefault("sqm", {})["table"] = _sqm_table(view.get("sqm") or {},
                                                         len(ctx["sci_scores"]))
        html = render(
            "raw_viewer.html",
            run_labels_json=json.dumps([view_label]),
            run_labels=[view_label],
            stem=out.stem,
            data_json=json.dumps([view]),
            ch_wavelengths=list((payload.get("channels") or {}).get("wavelengths") or []),
            wavelength_columns=WAVELENGTH_COLUMNS,
            **ctx["shell"],
            **_COND_CH_COLUMN_VARS,
            decisions=False,
            bad_heading=section_note("summary.condition_failing"),
            plotly_tags=plotly_script_tags(out),
        )
        out.write_text(html, encoding="utf-8")
        logger.info("condition %s \u2192 %s", label, out.name)


def _shell_vars(runs: list[dict], output_path: Path, sci_threshold: float) -> dict:
    """The head, nav bar and four closing sections every QC report shares.

    The raw viewer is the one report that cannot extend ``_report_base.html.j2``, being a
    single JavaScript-driven document with its own body, so it takes the same variables and
    includes the same footer partial instead. The errors block is deliberately left out:
    those are per run here and the viewer renders them itself.

    No Methods and no Provenance: this is a quality view of the recording, its lineage is the
    one input and the record, and the record's sidecar lists that source.
    """
    session = runs[0].get("session") if runs else None
    versions = collect_software_versions()
    meta = [("runs", str(len(runs)))]
    if session:
        meta.append(("session", session))
    return {
        **page_vars(
            title=f"nirspipe raw QC  \u00b7  {output_path.stem}",
            heading="nirspipe Raw Viewer",
            nav_meta=meta,
            nav_note=f"SCI thr: {sci_threshold:.2f}",
        ),
        **footer_vars(scope=output_path.stem, versions=versions),
        # the bar's identity chip, named as the subject report's
        "subject": runs[0]["subject_id"] if runs else "",
        # the subject index is rebuilt by the same command, so the bar can always point at it
        "index_href": (report_name(f"sub-{runs[0]['subject_id']}", desc="index")
                       if runs else ""),
    }


def build_prep_raw_report(
    runs: list[dict],
    output_path: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
    dpf: list[float],
    sci_threshold: float,
    psp_threshold: float | None = None,
    min_good_frac: float | None = None,
    screen_scope: str = "run",
    window_s: float = 10.0,
    epoch_qc: bool = False,
    epoch_tmin: float | None = None,
    epoch_tmax: float | None = None,
    sep_bands=None,
    by_condition: bool = False,
    motion_correction: str | None = None,
    roi_map: "dict | None" = None,
    motion_std_thresh: float | None = None,
    wavelet_iqr: float | None = None,
) -> None:
    """Generate raw QC report: lightweight HTML + per-run folders with figure HTMLs + SQM JSON.

    ``by_condition`` writes one extra report per annotated condition beside the run's own,
    carrying the condition in its ``cond-`` entity. Their numbers are sliced out of the run's
    windowed pass rather than measured on a cut of it. See
    :mod:`nirspipe.qc.subject.condition_views`.
    """
    # the report sits in the subject's own folder, so its figures are one level in from it
    # rather than a sibling tree, and sub-<id>/ can be moved or copied whole
    sub_dir     = output_path.parent
    static_data = []
    shell       = _shell_vars(runs, output_path, sci_threshold)

    for i, run in enumerate(runs):
        label = run["label"]
        logger.info("[%d/%d] processing %s ...", i + 1, len(runs), label)
        # a run that fails outright still gets an entry, carrying the reason: the viewer's
        # run selector lists it either way, and an empty panel with no explanation reads as
        # a broken viewer rather than as a run that could not be read
        run_errors: list[str] = []
        d: dict = {}
        ctx: dict = {}
        with guard("Processing this run", run_errors, label):
            d, ctx = _process_run(run, sci_threshold, sub_dir, cardiac_l_freq, cardiac_h_freq,
                                  resp_l_freq, resp_h_freq,
                                  dpf, window_s, epoch_qc, epoch_tmin, epoch_tmax,
                                  psp_threshold, sep_bands, min_good_frac, screen_scope,
                                  motion_correction, roi_map=roi_map,
                                  motion_std_thresh=motion_std_thresh, wavelet_iqr=wavelet_iqr)
        if run_errors:
            d = {"errors": run_errors, "notes": []}
        if ctx.get("remake_brain"):
            brain_errors: list[str] = []
            href = ctx["remake_brain"](figure_namer(label, prefix="raw"),
                                       ctx["sci_win_scores"], brain_errors)
            if href:
                d["figure_paths"]["brain_views"] = {"src": href, "flat": ctx.get("brain_flat", False)}
            d["errors"] = list(d.get("errors") or []) + collapse_messages(brain_errors)
        static_data.append(d)
        if by_condition and ctx:
            with guard("Per-condition views", run_errors, label):
                _write_condition_views({**ctx, "shell": shell}, d, output_path,
                                       label, sci_threshold)
            with guard("Per-condition channel quality", run_errors, label):
                _condition_summary(ctx, d, label)

    run_labels = [r["label"] for r in runs]
    # one header for every run the page switches between, so it names the wavelengths
    # only when the runs agree on them
    wavelengths = {tuple((d.get("channels") or {}).get("wavelengths") or ()) for d in static_data}
    wavelengths.discard(())

    html = render(
        "raw_viewer.html",
        run_labels_json=json.dumps(run_labels),
        run_labels=run_labels,
        stem=output_path.stem,
        data_json=json.dumps(static_data),
        ch_wavelengths=list(wavelengths.pop()) if len(wavelengths) == 1 else [],
        wavelength_columns=WAVELENGTH_COLUMNS,
        **shell,
        **_CH_COLUMN_VARS,
        decisions=True,
        plotly_tags=plotly_script_tags(output_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Raw QC report saved: %s", output_path)
