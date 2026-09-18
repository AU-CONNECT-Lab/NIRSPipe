"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import json
from pathlib import Path

import mne

from fnirs_pipe.qc.subject.condition_views import (
    carpet_view_table as _carpet_views, condition_view_table,
)
from fnirs_pipe.qc.common.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.common.windows import markers_on_data_axis
from fnirs_pipe.qc.common.channel_table import (
    MOTION_SPLIT_COLUMNS, OD_SPLIT_COLUMNS, channel_columns, channel_rows, format_rows,
    heatmap_args, pair_rows, save_channel_csv, separation_blocks, separation_notes,
    split_table,
)
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.qc.metrics._helpers import (_mean_or_none, separation_bands,
                                           separation_orphans)
from fnirs_pipe.qc.common.report_shell import (
    collapse_messages, footer_vars, guard, note, page_vars, render,
)
from fnirs_pipe.qc.subject.trial_qc import score_trials, trial_windows
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.prep_raw_report")

_MAX_TS_PTS   = 4000
_EPOCH_TMIN   = -5.0
_EPOCH_TMAX   = 25.0
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
# added to either one leaves the list on its own: mean amplitude was in both for a while.
_VIEW_MONTAGE_KEYS = tuple(
    k for k in ("cp_mean", "n_flat_channels", "mean_amp_mean",
                "spike_count", "spike_pct_frames", "spike_num_frames")
    if k not in {key for key, _ in (*OD_SPLIT_COLUMNS, *MOTION_SPLIT_COLUMNS)}
)
_VIEW_SCALAR_KEYS = (
    "channel_retention_rate", "sci_win_mean", "sci_mean", "good_frac_mean", "psp_mean",
    "snr_mean", "cv_mean",
    "cp_mean", "n_flat_channels", "mean_amp_mean",
    "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95", "gvtd_thresh",
    "gvtd_pct_above_thresh", "gvtd_num_above_thresh",
    "spike_count", "spike_pct", "spike_pct_frames", "spike_num_frames",
)

# ---- Channel decisions table ----
# Its columns, like every other view's, come from channel_table. It draws the decision chip
# itself, that being the one column that is not a measurement, and the HbO-HbR correlation
# is a haemoglobin measurement this intensity view has no stage for. The keys go over as
# JSON because the table body is built in the browser.
_CH_COLUMNS     = channel_columns(("corr", "separation"))
_CH_COLUMN_VARS = {"ch_columns": _CH_COLUMNS,
                   "ch_column_keys_json": json.dumps([key for key, _ in _CH_COLUMNS])}


def _store_matrices(windowed: dict, series: dict) -> None:
    """The channel-by-window matrices into the record, under the pipeline's own key names."""
    import numpy as np

    for key in ("sci_matrix", "psp_matrix", "cv_matrix",
                "sci_times", "psp_times", "cv_times"):
        if series.get(key) is not None:
            windowed[key] = np.asarray(series[key]).tolist()


def _store_spans(windowed: dict, raw, sep_bands, errors: list, label: str) -> None:
    """The flagged spans, kept as spans so a condition counts the run's own boolean.

    ::

      windowed["spike_spans_s"] -> [[412.0, 1.3], [880.5, 0.8]]

    Two families, each on its own channel sets, because both tests are over a set rather
    than over a channel: a spike span found on the short channels is not a claim about the
    long ones, and GVTD is an RMS across whatever it is given, so each set has its own trace
    and its own threshold. The long set keeps the plain key, as the pipeline's record does.

    No ``motion_corrected_spans_s``: that needs the optical density either side of the
    correction step and this pass runs before it. The key is absent rather than empty.
    """
    from fnirs_pipe.qc.metrics import (
        gvtd_above_segments, long_short_channels, spike_segments,
    )

    long_names, short_names = long_short_channels(raw, sep_bands)
    with guard("Spike spans", errors, label):
        for key, names in (("spike_spans_s", long_names),
                           ("spike_spans_short_s", short_names)):
            if names:
                picked = raw.copy().pick(names)
            elif key == "spike_spans_s":
                picked = raw               # unsplit montage: every channel, as the record does
            else:
                continue
            windowed[key] = [list(span) for span in spike_segments(picked)]
    with guard("GVTD above-threshold spans", errors, label):
        windowed["gvtd_above_spans_s"] = [
            list(span) for span in gvtd_above_segments(raw, sep_bands)]
        for key, picks in (("gvtd_above_spans_short_s", short_names),
                           ("gvtd_above_spans_all_s", list(raw.ch_names))):
            if picks:
                windowed[key] = [list(span) for span in
                                 gvtd_above_segments(raw, sep_bands, picks=picks)]


def _process_run(
    run: dict,
    sci_threshold: float,
    sub_dir: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
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
) -> "tuple[dict, dict]":
    """Compute all data, save figure HTMLs + SQM JSON. Returns (inline dict, context).

    A panel that fails costs that panel and lands in the returned ``errors``, which the
    viewer prints for the selected run. Failures used to reach the log only, so a viewer
    missing half its figures looked the same as one whose recording had nothing to plot.

    The second return value is what the per-condition views need and the viewer must never
    see: the windowed matrices, the record and the condition windows. It is kept out of the
    payload because that one is serialised with ``json.dumps`` and a numpy array is not
    serialisable, so folding these in would turn a working report into a crash.
    """
    from fnirs_pipe.qc.figures import (
        build_channel_figure,
        build_epoch_preview_figure,
        build_evoked_topo_figure,
        build_layout_figure,
        build_trial_image_by_condition,
        build_psd_mean_figure,
        build_sci_psp_figure,
        build_trigger_timeline_single,
        build_ts_figure,
        carpet_gvtd_figure,
        channel_quality_heatmap,
        condition_colors,
        trial_quality_heatmap,
    )
    from fnirs_pipe.qc.boilerplate.vocabulary import metric_rows
    from fnirs_pipe.qc.metrics import (
        attach_windowed_series, compute_raw_sqm, compute_sci_scores,
        resolve_cutoffs, screen_channels, screening_scores,
    )
    from fnirs_pipe.qc.common.screen_scope import resolve_screen_scope
    from fnirs_pipe.qc.subject.sqm_record import (
        motion_sections, raw_condition_sections, raw_sections, sqm_record_dict,
    )

    label   = run["label"]
    session = run.get("session")
    fig_dir = sub_dir / "figures"
    sqm_dir = sub_dir / (f"ses-{session}" if session else "") / "nirs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sqm_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    notes: list[str] = []

    raw = mne.io.read_raw_snirf(run["snirf_path"], preload=True, verbose=False)

    sci_scores, raw_od = compute_sci_scores(raw, cardiac_l_freq, cardiac_h_freq)
    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold,
                              good_frac=min_good_frac)
    scope = resolve_screen_scope(raw, screen_scope)
    screen_scores = screening_scores(raw_od, cardiac_l_freq, cardiac_h_freq,
                                     have={"sci": sci_scores}, cutoffs=cutoffs,
                                     scope=scope)
    bad_list, _why = screen_channels(screen_scores, cutoffs)
    bad_channels: set[str] = set(bad_list)

    # The one preprocessing step this report runs, and only because the panel it feeds is
    # unreadable without it: motion is what a raw recording is judged on, and a figure of
    # the uncorrected trace cannot say whether the correction would have dealt with it.
    # Nothing is written back; the corrected copy lives for the length of this function.
    raw_motcorr = None
    if motion_correction and motion_correction != "none":
        with guard("Motion correction", errors, label):
            from fnirs_pipe.pipeline.motion import correct_motion
            raw_motcorr = correct_motion(raw_od.copy(), method=motion_correction)

    sqm: dict = {}
    with guard("Quality metrics", errors, label):
        sqm = compute_raw_sqm(raw, sci_scores, list(bad_channels),
                              cardiac_l_freq, cardiac_h_freq,
                              screen_scores.get("good_frac"))

    # `sqm` stays the flat all-channel view the per-window figures below read. The record
    # written to disk is the sectioned one, built through the same function the pipeline
    # uses, and it is now what the panels read too: the flat view averages a short
    # channel's coupling in with the long ones, which lifts SCI, PSP and SNR and can make a
    # poorly coupled recording read as a good one.
    # the figures colour a channel short or not short, so only the short edge applies
    short_thresh = (sep_bands if sep_bands is not None else separation_bands())[0]
    raw_secs, raw_pc = raw_sections(
        raw, sci_scores, list(bad_channels), cardiac_l_freq, cardiac_h_freq, sep_bands,
        screen_scores.get("good_frac"))
    # Reported per condition, never screened on: one channel set has to serve every
    # condition, or a contrast between two conditions is also a contrast between two
    # montages. This is what says "the channel was fine in rest and dead in the second
    # game" without changing which channels the analysis gets. A window shorter than two
    # screening windows has too few to count, so it is not offered a share at all.
    # both bound before the guard: it swallows the exception, and the per-condition views
    # read these afterwards
    cond_frac: dict = {}
    cond_windows: list = []
    with guard("Coupled windows per condition", errors, label):
        from fnirs_pipe.qc.common.windows import condition_windows
        from fnirs_pipe.qc.metrics.windowed import (
            SCREEN_WINDOW_S, condition_window_fractions,
        )
        cond_windows = condition_windows(raw, min_duration=2 * SCREEN_WINDOW_S)
        if cond_windows:
            cond_frac = condition_window_fractions(
                raw_od, cond_windows, cardiac_l_freq, cardiac_h_freq,
                sci_cutoff=cutoffs["sci"], psp_cutoff=cutoffs["psp"])
    if cond_frac and "raw" in raw_secs:
        raw_secs["raw"]["good_frac_by_condition"] = {
            label_: _mean_or_none(shares.values()) for label_, shares in cond_frac.items()}
        raw_pc.setdefault("raw", {})["good_frac_by_condition_per_channel"] = cond_frac

    record_view = {**raw_secs, "per_channel": raw_pc}
    ch_rows = channel_rows(record_view, sci_scores, bad_channels)
    # the verdict is read off the long channels wherever the montage was split, exactly as
    # the subject report reads it, so two views of one recording cannot disagree
    view_scalars = raw_secs.get("raw_long") or raw_secs.get("raw") or {}
    raw_all = raw_secs.get("raw") or {}
    sqm_split = bool(raw_secs.get("raw_long") and raw_secs.get("raw_short"))

    # Persist windowed series so group_raw can build time × subject heatmaps. Their own
    # section, since `_split_scalars` would file every one of these lists under per_channel.
    windowed: dict = {}
    # GVTD off the corrected file where there is one, as the pipeline's own record does:
    # GVTD measures the movement the correction exists to remove, so the corrected stage is
    # the informative one for it. SCI and PSP stay on the uncorrected file, being coupling.
    series = attach_windowed_series(windowed, raw_od, cardiac_l_freq, cardiac_h_freq,
                                    window_s, gvtd_od=raw_motcorr,
                                    raw_intensity=raw, sep_bands=sep_bands)
    sci_matrix, sci_win_times = series["sci_matrix"], series["sci_times"]
    psp_matrix, psp_win_times = series["psp_matrix"], series["psp_times"]
    # the channel by window matrices too, not just the channel-averaged series: a
    # per-condition number is a column selection out of these, and without them on disk it
    # could only be recomputed. Same keys the pipeline's own record writes.
    _store_matrices(windowed, series)
    _store_spans(windowed, raw, sep_bands, errors, label)

    if raw_motcorr is not None:
        with guard("Motion sections", errors, label):
            from fnirs_pipe.qc.subject.sqm_record import _section_writer
            motion_sections(
                raw_od, raw_motcorr, _section_writer(raw_secs, raw_pc), windowed,
                lambda name: (raw_secs.get(name) or {}).get("gvtd_thresh"),
                cardiac_l_freq, cardiac_h_freq, sep_bands)

    set_rows = [
        ("All",   len(sci_scores),                raw_all,                         False),
        ("Long",  raw_all.get("n_long_channels"), raw_secs.get("raw_long") or {},  True),
        ("Short", raw_all.get("n_short_channels"), raw_secs.get("raw_short") or {}, False),
    ]
    split = split_table(set_rows) if sqm_split else {}
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
    motion_split = split_table(motion_rows, MOTION_SPLIT_COLUMNS) if sqm_split else {}

    raw_haemo = None
    with guard("Beer-Lambert", errors, label):
        ppf = dpf[0] if len(dpf) == 1 else dpf
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=ppf)
    if raw_haemo is None:
        note(notes, label, "no haemoglobin conversion, so the per-channel detail panel "
                           "is empty")

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
            raw, markers, bad_channels, _MAX_TS_PTS, short_thresh,
        )
        ts_inline = {
            "figure":      fig.to_dict(),
            "markers":     markers,
            "cond_colors": cond_colors_out,
            "band_shapes": band_shapes,
            "t_start":     t_start,
            "t_end":       t_end,
        }

    # ── inline: layout figures (kept for click interactivity) ──────────────────
    layout_inline: dict = {}
    with guard("Optode layout", errors, label):
        fig_2d, fig_3d = build_layout_figure(raw, bad_channels, sci_scores, short_thresh)
        layout_inline = {
            "layout_2d_figure": fig_2d.to_dict() if fig_2d else None,
            "layout_3d_figure": fig_3d.to_dict() if fig_3d else None,
        }

    # ── file: carpet GVTD ──────────────────────────────────────────────────
    #   an iframe rather than inlined like the panels above: the carpet is a channels x 2000
    #   heatmap, and every run of the viewer would carry one in the page itself
    carpet_inline: dict = {}
    # named out here because the per-channel motion figures below read them too: a channel's
    # GVTD row has to be its own separation class's, the same blocks the carpet drew
    gvtd_blocks: "list[tuple[str, list[str]]]" = []
    with guard("GVTD carpet", errors, label):
        from fnirs_pipe.qc.metrics import gvtd_channel_blocks
        gvtd_blocks = gvtd_channel_blocks(raw, sep_bands)
        gvtd_set = gvtd_blocks[0][0]
        gvtd_picks = [c for _, names in gvtd_blocks for c in names]
        raw_carpet = raw.copy().pick(gvtd_picks)
        # before and after in one panel where a correction ran, which is the pairing the
        # subject report's carpet draws and the reason the flag exists
        carpet_after = (None if raw_motcorr is None
                        else raw_motcorr.copy().pick(
                            [c for c in gvtd_picks if c in raw_motcorr.ch_names]))
        fig   = carpet_gvtd_figure(raw_carpet, raw_carpet.ch_names,
                                   corrected_segments=[tuple(sp) for sp in
                                                       windowed.get("motion_corrected_spans_s") or []] or None,
                                   spike_segments={gvtd_set: [tuple(sp) for sp in
                                                              windowed.get("spike_spans_s") or []] or None,
                                                   "short": [tuple(sp) for sp in
                                                             windowed.get("spike_spans_short_s") or []] or None},
                                   raw_after=carpet_after,
                                   channel_set=gvtd_set, blocks=gvtd_blocks)
        fname = f"{label}_desc-carpet_nirs.html"
        # not written per condition: its GVTD is filtered, its carpet z-scored per channel
        # and its colour scale taken over the run, so a cut would give each condition a
        # scale no other one can be read against. One file, narrowed by URL fragment.
        h     = _save_figure_html(fig, fig_dir / fname,
                                  views=_carpet_views(fig, cond_windows))
        figure_paths["carpet"] = {"src": f"figures/{fname}", "h": h}
        carpet_inline = {"figure": fig.to_dict()}

    # ── file: SCI / PSP ────────────────────────────────────────────────────────
    sci_psp_inline: dict = {}
    with guard("SCI / PSP", errors, label):
        # the CV row too, off the same windowed pass and the same grid, as the subject
        # report's copy of this panel draws it
        fig   = build_sci_psp_figure(
            sci_scores, psp_per_ch, bad_channels, sci_threshold,
            sci_matrix=sci_matrix, sci_win_times=sci_win_times,
            psp_matrix=psp_matrix, psp_win_times=psp_win_times,
            cv_per_channel=(raw_pc.get("raw") or {}).get("cv_per_channel") or {},
            cv_matrix=series.get("cv_matrix"), cv_win_times=series.get("cv_times"),
        )
        fname = f"{label}_desc-scipsp_nirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["sci_psp"] = {"src": f"figures/{fname}", "h": h}
        sci_psp_inline = {"figure": fig.to_dict()}

    # ── file: PSD mean ─────────────────────────────────────────────────────────
    psd_inline: dict = {}
    with guard("PSD", errors, label):
        fig = build_psd_mean_figure(raw, cardiac=(cardiac_l_freq, cardiac_h_freq))
        if fig:
            fname = f"{label}_desc-psd_nirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["psd"] = {"src": f"figures/{fname}", "h": h}
            psd_inline = {"figure": fig.to_dict()}

    # ── file: trigger timeline ─────────────────────────────────────────────────
    trigger_timeline_inline: dict = {}
    with guard("Trigger timeline", errors, label):
        fig = build_trigger_timeline_single(markers, cond_colors_)
        if fig:
            fname = f"{label}_desc-trigger_nirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["trigger"] = {"src": f"figures/{fname}", "h": h}
            trigger_timeline_inline = {"figure": fig.to_dict()}

    # ── file: channel quality summary ──────────────────────────────────────────
    ch_summary_inline: dict = {}
    with guard("Channel quality summary", errors, label):
        # long block first with the divider between the two, from the same helper the
        # subject report uses, so the grid and the per-channel table below it read in one
        # order and a short channel never lands in a long channel's verdict
        fig = channel_quality_heatmap(sci_thresh=sci_threshold, **heatmap_args(ch_rows))
        fname = f"{label}_desc-chsummary_nirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["ch_summary"] = {"src": f"figures/{fname}", "h": h}
        ch_summary_inline = {"figure": fig.to_dict()}

    # the epoch figures need concrete bounds; per-trial QC reads None as "use event duration"
    fig_tmin = _EPOCH_TMIN if epoch_tmin is None else epoch_tmin
    fig_tmax = _EPOCH_TMAX if epoch_tmax is None else epoch_tmax

    # ── inline only: evoked topo ───────────────────────────────────────────────
    # for the GUI's Data Prep page, which reads this result. The report dropped the panel:
    # at this stage the average is taken on unfiltered, uncorrected concentration and is
    # mostly drift, and the GUI already shows it interactively.
    evoked_topo_inline: dict = {}
    if raw_haemo is not None:
        with guard("Evoked topography", errors, label):
            fig = build_evoked_topo_figure(
                raw_haemo, markers, _MAX_TS_PTS, fig_tmin, fig_tmax,
            )
            if fig:
                evoked_topo_inline = {"figure": fig.to_dict()}

    # ── file: grand mean ───────────────────────────────────────────────────────
    # One row per condition, every long channel averaged with the short ones dotted. The
    # dotted trace is what the panel is read for at this stage: when it rises with the solid
    # one the response is scalp haemodynamics, and nothing downstream will separate them.
    if raw_haemo is not None:
        with guard("Grand mean", errors, label):
            fig = build_epoch_preview_figure(raw_haemo, epoch_tmin=fig_tmin,
                                             epoch_tmax=fig_tmax, sep_bands=sep_bands)
            if fig:
                fname = f"{label}_desc-epochmean_nirs.html"
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["epoch_mean"] = {"src": f"figures/{fname}", "h": h}

    # ── file: per-channel motion detail ────────────────────────────────────────
    # The subject report's own figures, builder and saver: this panel answers the same
    # question on the same two recordings, and a second copy of it here is how two reports
    # start disagreeing about what a channel did. One file per channel, carrying every
    # condition's window, so a condition page narrows the view instead of remeasuring.
    motion_channels: list[str] = []
    if raw_motcorr is not None:
        with guard("Motion detail", errors, label):
            from fnirs_pipe.qc.subject.report import (
                _motion_detail_figures, _section_motion_detail,
            )
            built = _motion_detail_figures(
                raw_od, raw_motcorr, label, errors,
                corrected_segments=[tuple(sp) for sp in
                                    windowed.get("motion_corrected_spans_s") or []] or None,
                spike_by_set={
                    gvtd_blocks[0][0] if gvtd_blocks else "all":
                        [tuple(sp) for sp in windowed.get("spike_spans_s") or []] or None,
                    "short": [tuple(sp) for sp in
                              windowed.get("spike_spans_short_s") or []] or None,
                },
                gvtd_blocks=gvtd_blocks or None,
            )
            if built:
                saved = _section_motion_detail(
                    built, label, errors, fig_dir, condition_spans=cond_windows,
                    filename=f"{label}_desc-motion{{ch}}_nirs.html",
                )["motion_detail_pairs"]
                motion_channels = [entry["pair"] for entry in saved]
                # every one of these is the same four-row layout, so one height serves the
                # panel and the viewer does not carry eighty-eight of them
                figure_paths["motion_detail_template"] = {
                    "src": f"figures/{label}_desc-motion{{ch}}_nirs.html",
                    "h": saved[0]["h"] if saved else 700,
                }

    # ── file: per-channel trial images ─────────────────────────────────────────
    # Trials down the rows, so a channel that was fine for the first half and lost for the
    # second reads as a band rather than being averaged away. HbO only, as the subject
    # report's is: single-trial HbR is too low-amplitude to read as an image. A condition
    # holding fewer than two trials draws nothing, which is what a block design is.
    trial_img_pairs: list = []
    trial_img_by_cond: dict = {}
    if raw_haemo is not None and cond_windows:
        hbo_names = [c for c in raw_haemo.ch_names if c.endswith(" hbo")]
        for ch in hbo_names:
            with guard("Trial image", errors, f"{label} | {ch}"):
                by_label = build_trial_image_by_condition(
                    raw_haemo, ch, cond_windows, fig_tmin, fig_tmax)
                if not by_label:
                    continue
                pair = ch.rsplit(" ", 1)[0]
                fname = f"{label}_desc-trialimg{_pair_fname(pair)}_nirs.html"
                figs = [f for figs in by_label.values() for f in figs]
                h = _save_multi_fig_html(figs, fig_dir / fname)
                trial_img_pairs.append(
                    {"pair": pair, "src": f"figures/{fname}", "h": h})
                # kept in memory so a condition page writes its own single-panel file
                # rather than this stack of every condition
                for cond_label, cond_figs in by_label.items():
                    trial_img_by_cond.setdefault(cond_label, []).append((pair, cond_figs))

    # ── file: per-channel detail HTML ──────────────────────────────────────────
    # one file per pair, holding every condition's view: `condition_views` names the windows
    # and the file picks one off its URL fragment, so a condition page links to the run's
    # figure rather than to a copy of it
    channel_pairs: list[str] = []
    if raw_haemo is not None:
        # this one figure is drawn on the original recording's axis, so the windows are
        # shifted onto it before its views are measured
        detail_origin = float(raw.first_time)
        detail_spans = [(lab, t0 + detail_origin, t1 + detail_origin)
                        for lab, t0, t1 in cond_windows]
        channel_pairs = get_channel_pairs(raw_haemo)
        for pair in channel_pairs:
            with guard("Channel detail", errors, f"{label} | {pair}"):
                detail_fig, psd_fig, epoch_fig = build_channel_figure(
                    raw_haemo, detail_markers, pair, _MAX_TS_PTS, fig_tmin, fig_tmax,
                    cardiac=(cardiac_l_freq, cardiac_h_freq),
                )
                fname = f"{label}_desc-ch{_pair_fname(pair)}_nirs.html"
                _save_multi_fig_html(
                    [detail_fig, psd_fig, epoch_fig], fig_dir / fname,
                    views=condition_view_table(detail_fig, detail_spans))
        if channel_pairs:
            figure_paths["ch_detail_template"] = (
                f"figures/{label}_desc-ch{{pair}}_nirs.html"
            )

    # `channel_pairs or None` so a run whose Beer-Lambert failed still gets a table, built
    # from the pairs the intensity recording carries rather than from an empty list
    pair_cells = format_rows(pair_rows(ch_rows, channel_pairs or None), sci_threshold,
                             name_key="pair", psp_threshold=cutoffs["psp"])

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
            trial_rows = [(onset, lab, sqm) for (_, _, _, onset), lab, sqm in zip(
                trial_windows(markers, epoch_tmin, epoch_tmax, float(raw.times[-1])),
                labels, sqms)]
            fig = trial_quality_heatmap(labels, sqms)
            if fig:
                fname = f"{label}_desc-trialqc_nirs.html"
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
    sqm_path = sqm_dir / f"{label}_desc-sqmraw_nirs.json"
    sections = {**raw_secs, "windowed": windowed, "per_channel": raw_pc}
    # written whenever the recording carries conditions, independently of `by_condition`:
    # the flag decides what a report shows, the record says what was measured. Deliberately
    # not in `SECTIONS`, so the group table does not descend into it.
    by_cond: dict = {}
    if cond_windows:
        with guard("Per-condition metrics", errors, label):
            by_cond = raw_condition_sections(sections, raw, cond_windows, cutoffs, sep_bands)
    record = sqm_record_dict(sections, [str(run["snirf_path"])])
    if by_cond:
        record["by_condition"] = by_cond
    sqm_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("SQM JSON -> %s", sqm_path)
    save_channel_csv(ch_rows, label, sqm_dir, sci_threshold, psp_threshold=cutoffs["psp"])

    n_total = len(sci_scores)
    bad_rate = 100 * len(bad_channels) / n_total if n_total else 0.0
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
            "dpf": list(dpf),
            "cardiac": [cardiac_l_freq, cardiac_h_freq],
            "motion_correction": motion_correction or "none",
            "scope": ("Screened on the long channels."
                      if sqm_split else "Screened on every channel: this montage carries "
                                        "no short channels to judge separately."),
        },
        "ts":           ts_inline,
        "layout":       layout_inline,
        "evoked_topo":  evoked_topo_inline,
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
            "rows": metric_rows(
                view_scalars,
                _VIEW_MONTAGE_KEYS if sqm_split else _VIEW_SCALAR_KEYS,
                skip_missing=True),
            "split":        split,
            "motion_split": motion_split,
            "channel_set":  "long channels" if sqm_split else "every channel",
        },
        # One table, at pair granularity: a decision is taken per source-detector pair and
        # SCI is a property of the pair rather than of either wavelength, so a per-wavelength
        # table beside this one would list every channel twice for no extra information. The
        # per-wavelength numbers are in the CSV written next to the record.
        "channels": {
            "pairs":  pair_cells,
            "blocks": separation_blocks(pair_cells),
            # sep_bands, or a run with non-default bands gets the default gap quoted at it
            "notes":  separation_notes(raw_all, ch_rows, sep_bands=sep_bands,
                                       orphan_mm=separation_orphans(raw, sep_bands)),
        },
        "channel_pairs": channel_pairs,
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
        "remake_psd":    _psd_maker(raw, cardiac_l_freq, cardiac_h_freq,
                                    build_psd_mean_figure),
        "remake_epoch":  _epoch_maker(raw_haemo, fig_tmin, fig_tmax, sep_bands,
                                      build_epoch_preview_figure),
        "trial_images_by_condition": trial_img_by_cond,
        "sqm_path":      sqm_path,
        "fig_dir":       fig_dir,
        "sci_scores":    sci_scores,
        "bad_channels":  bad_channels,
        "channel_pairs": channel_pairs,
        "series":        series,
        "cutoffs":       cutoffs,
        "trial_rows":    trial_rows,
    }


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


def _psd_maker(raw, cardiac_l_freq: float, cardiac_h_freq: float, build):
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
    from fnirs_pipe.qc.subject.condition_views import PSD_NFFT_CAP

    def remake(t0: float, t1: float):
        lo, hi = max(0.0, float(t0)), min(float(raw.times[-1]), float(t1))
        if hi <= lo:
            return None
        cut = raw.copy().crop(tmin=lo, tmax=hi)
        if len(cut.times) < PSD_NFFT_CAP:
            logger.info("condition %.1f-%.1f s is %d samples, under the %d the transform "
                        "needs; no spectrum", lo, hi, len(cut.times), PSD_NFFT_CAP)
            return None
        return build(cut, cardiac=(cardiac_l_freq, cardiac_h_freq))

    return remake


def _write_condition_views(ctx: dict, payload: dict, output_path: Path, run_label: str,
                           sci_threshold: float) -> None:
    """One report file per condition, beside the run's own, read out of the quality record.

    Every number on these pages comes from the record's ``by_condition`` section, which
    :func:`~fnirs_pipe.qc.subject.sqm_record.raw_condition_sections` wrote a moment earlier; nothing
    is measured here. A record carrying no such section gets no pages rather than a second
    copy of the numbers free to disagree with the first.

    Each page is the same viewer with a single entry, so nothing about how these are read has
    to be learned twice. The file name comes from :func:`condition_stems`, which follows the
    rule ``fnirs-prep crop`` set for a segment: the condition becomes the ``task-`` entity.
    """
    from fnirs_pipe.qc.subject.condition_views import (
        condition_payloads, condition_stem, condition_stems,
    )

    sqm_path = ctx.get("sqm_path")
    if sqm_path is None or not Path(sqm_path).exists():
        logger.warning("%s | no quality record; no per-condition pages", run_label)
        return
    record = json.loads(Path(sqm_path).read_text(encoding="utf-8"))
    by_condition = record.get("by_condition") or {}
    if not by_condition:
        logger.info("%s | the record carries no by_condition section; no per-condition "
                    "pages", run_label)
        return

    fig_dir = Path(ctx["fig_dir"])

    def save_figure(panel: str, slug: str, fig) -> "dict | None":
        """One condition's own figure file, named so `figure_leaks` can recognise it."""
        fname = f"{run_label}_desc-{panel.replace('_', '')}_{slug}_nirs.html"
        h = _save_figure_html(fig, fig_dir / fname)
        return {"src": f"figures/{fname}", "h": h}

    def save_stack(panel: str, slug: str, name: str, figs) -> "dict | None":
        """The same, for a panel whose file holds several figures of one channel."""
        fname = (f"{run_label}_desc-{panel.replace('_', '')}"
                 f"{_pair_fname(name)}_{slug}_nirs.html")
        h = _save_multi_fig_html(list(figs), fig_dir / fname)
        return {"pair": name, "src": f"figures/{fname}", "h": h}

    views = condition_payloads(
        payload, record=record, by_condition=by_condition,
        sci_scores=ctx["sci_scores"], bad_channels=ctx["bad_channels"],
        channel_pairs=ctx["channel_pairs"], series=ctx["series"],
        sci_threshold=sci_threshold, cutoffs=ctx["cutoffs"],
        trial_rows=ctx.get("trial_rows"), save_figure=save_figure,
        remake_psd=ctx.get("remake_psd"), remake_epoch=ctx.get("remake_epoch"),
        trial_images=ctx.get("trial_images_by_condition"),
        save_stack=save_stack,
    )
    if not views:
        return
    stems = condition_stems(output_path.stem, [label for label, _ in views])
    for i, ((label, view), stem) in enumerate(zip(views, stems), start=1):
        view_label = condition_stem(run_label, label, i)
        html = render(
            "raw_viewer.html",
            run_labels_json=json.dumps([view_label]),
            run_labels=[view_label],
            stem=stem,
            data_json=json.dumps([view]),
            **ctx["shell"],
            **_CH_COLUMN_VARS,
        )
        out = output_path.with_name(f"{stem}.html")
        out.write_text(html, encoding="utf-8")
        logger.info("condition %s \u2192 %s", label, out.name)


def _shell_vars(runs: list[dict], output_path: Path, sub_dir: Path,
                sci_threshold: float) -> dict:
    """The head, nav bar and four closing sections every QC report shares.

    The raw viewer is the one report that cannot extend ``_report_base.html.j2``, being a
    single JavaScript-driven document with its own body, so it takes the same variables and
    includes the same footer partial instead. The errors block is deliberately left out:
    those are per run here and the viewer renders them itself.

    The Methods prose and the provenance table are read from the sidecars in ``nirs/``, so
    on a tree where only this command has run they describe that one step rather than a
    pipeline that has not happened yet.
    """
    from fnirs_pipe.qc.boilerplate import collect_software_versions, generate_methods_text

    session = runs[0].get("session") if runs else None
    nirs_dir = sub_dir / (f"ses-{session}" if session else "") / "nirs"
    versions = collect_software_versions()
    meta = [("runs", str(len(runs)))]
    if session:
        meta.append(("session", session))
    return {
        **page_vars(
            title=f"fnirs-pipe raw QC \u2014 {output_path.stem}",
            heading="fnirs\u2011pipe Raw Viewer",
            nav_meta=meta,
            nav_note=f"SCI thr: {sci_threshold:.2f}",
        ),
        **footer_vars(
            scope=output_path.stem, nirs_dir=nirs_dir,
            methods=generate_methods_text(versions=versions, nirs_dir=nirs_dir),
            versions=versions,
        ),
        # the subject index is rebuilt by the same command, so the bar can always point at it
        "index_href": f"sub-{runs[0]['subject_id']}_qc.html" if runs else "",
    }


def build_prep_raw_report(
    runs: list[dict],
    output_path: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    dpf: list[float],
    sci_threshold: float = SCI_PASS,
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
) -> None:
    """Generate raw QC report: lightweight HTML + per-run folders with figure HTMLs + SQM JSON.

    ``by_condition`` writes one extra report per annotated condition beside the run's own,
    named the way ``fnirs-prep crop`` names a segment. They are separate files rather than a
    switch inside this one because this report is already long, and their numbers are sliced
    out of the run's windowed pass rather than measured on a cut of it. See
    :mod:`fnirs_pipe.qc.subject.condition_views`.
    """
    # the report sits in the subject's own folder, so its figures are one level in from it
    # rather than a sibling tree, and sub-<id>/ can be moved or copied whole
    sub_dir     = output_path.parent
    static_data = []
    shell       = _shell_vars(runs, output_path, sub_dir, sci_threshold)

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
                                  dpf, window_s, epoch_qc, epoch_tmin, epoch_tmax,
                                  psp_threshold, sep_bands, min_good_frac, screen_scope,
                                  motion_correction)
        if run_errors:
            d = {"errors": run_errors, "notes": []}
        static_data.append(d)
        if by_condition and ctx:
            with guard("Per-condition views", run_errors, label):
                _write_condition_views({**ctx, "shell": shell}, d, output_path,
                                       label, sci_threshold)

    run_labels = [r["label"] for r in runs]

    html = render(
        "raw_viewer.html",
        run_labels_json=json.dumps(run_labels),
        run_labels=run_labels,
        stem=output_path.stem,
        data_json=json.dumps(static_data),
        **shell,
        **_CH_COLUMN_VARS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Raw QC report saved: %s", output_path)
