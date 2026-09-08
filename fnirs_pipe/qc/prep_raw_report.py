"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import json
from pathlib import Path

import mne

from fnirs_pipe.qc.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.channel_table import (
    channel_rows, format_rows, heatmap_args, pair_rows, save_channel_csv,
    separation_blocks, separation_notes, split_table,
)
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.qc.metrics._helpers import separation_bands
from fnirs_pipe.qc.report_shell import (
    collapse_messages, dashboard_css, guard, note, render,
)
from fnirs_pipe.qc.trial_qc import score_trials
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.prep_raw_report")

_MAX_TS_PTS   = 4000
_EPOCH_TMIN   = -5.0
_EPOCH_TMAX   = 25.0
# the scalars the viewer's metrics panel lists, in order. Names, formats, thresholds and
# tooltips all come from the metric registry, so this is only the choice of which ones and
# in what order: the same ones the subject report prints, minus what a raw recording has no
# later stage to measure.
_VIEW_SCALAR_KEYS = (
    "channel_retention_rate", "sci_mean", "psp_mean", "snr_mean", "cv_mean",
    "cp_mean", "n_flat_channels", "mean_amp_mean",
    "gvtd_mean", "gvtd_filt_p95", "gvtd_thresh",
    "gvtd_pct_above_thresh", "gvtd_num_above_thresh", "spike_count",
)


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
    gvtd_channels: str = "long",
    psp_threshold: float | None = None,
    sep_bands=None,
) -> dict:
    """Compute all data, save figure HTMLs + SQM JSON. Returns inline dict for HTML.

    A panel that fails costs that panel and lands in the returned ``errors``, which the
    viewer prints for the selected run. Failures used to reach the log only, so a viewer
    missing half its figures looked the same as one whose recording had nothing to plot.
    """
    from fnirs_pipe.qc.figures import (
        build_channel_figure,
        build_evoked_topo_figure,
        build_layout_figure,
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
    from fnirs_pipe.qc.sqm_record import raw_sections, sqm_record_dict

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
    screen_scores = screening_scores(raw_od, cardiac_l_freq, cardiac_h_freq,
                                     have={"sci": sci_scores})
    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold)
    bad_list, _why = screen_channels(screen_scores, cutoffs)
    bad_channels: set[str] = set(bad_list)

    sqm: dict = {}
    with guard("Quality metrics", errors, label):
        sqm = compute_raw_sqm(raw, sci_scores, list(bad_channels),
                              cardiac_l_freq, cardiac_h_freq)

    # `sqm` stays the flat all-channel view the per-window figures below read. The record
    # written to disk is the sectioned one, built through the same function the pipeline
    # uses, and it is now what the panels read too: the flat view averages a short
    # channel's coupling in with the long ones, which lifts SCI, PSP and SNR and can make a
    # poorly coupled recording read as a good one.
    # the figures colour a channel short or not short, so only the short edge applies
    short_thresh = (sep_bands if sep_bands is not None else separation_bands())[0]
    raw_secs, raw_pc = raw_sections(
        raw, sci_scores, list(bad_channels), cardiac_l_freq, cardiac_h_freq, sep_bands)
    record_view = {**raw_secs, "per_channel": raw_pc}
    ch_rows = channel_rows(record_view, sci_scores, bad_channels)
    # the verdict is read off the long channels wherever the montage was split, exactly as
    # the subject report reads it, so two views of one recording cannot disagree
    view_scalars = raw_secs.get("raw_long") or raw_secs.get("raw") or {}
    raw_all = raw_secs.get("raw") or {}
    sqm_split = bool(raw_secs.get("raw_long") and raw_secs.get("raw_short"))
    split = split_table([
        ("All",   len(sci_scores),                raw_all,                         False),
        ("Long",  raw_all.get("n_long_channels"), raw_secs.get("raw_long") or {},  True),
        ("Short", raw_all.get("n_short_channels"), raw_secs.get("raw_short") or {}, False),
    ]) if sqm_split else {}

    # Persist windowed series so group_raw can build time × subject heatmaps. Their own
    # section, since `_split_scalars` would file every one of these lists under per_channel.
    windowed: dict = {}
    series = attach_windowed_series(windowed, raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    sci_matrix, sci_win_times = series["sci_matrix"], series["sci_times"]
    psp_matrix, psp_win_times = series["psp_matrix"], series["psp_times"]

    raw_haemo = None
    with guard("Beer-Lambert", errors, label):
        ppf = dpf[0] if len(dpf) == 1 else dpf
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=ppf)
    if raw_haemo is None:
        note(notes, label, "no haemoglobin conversion, so the epoch, evoked-topography "
                           "and per-channel detail panels are empty")

    markers = extract_markers(raw)
    cond_colors_ = condition_colors(markers)
    for m in markers:
        m["color"] = cond_colors_.get(m["description"], "#f39c12")

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
    with guard("GVTD carpet", errors, label):
        from fnirs_pipe.qc.metrics import gvtd_channel_blocks
        gvtd_blocks = gvtd_channel_blocks(raw, gvtd_channels, sep_bands)
        gvtd_set = gvtd_blocks[0][0]
        gvtd_picks = [c for _, names in gvtd_blocks for c in names]
        raw_carpet = raw.copy().pick(gvtd_picks)
        fig   = carpet_gvtd_figure(raw_carpet, raw_carpet.ch_names,
                                   channel_set=gvtd_set, blocks=gvtd_blocks)
        fname = f"{label}_desc-carpet_nirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["carpet"] = {"src": f"figures/{fname}", "h": h}
        carpet_inline = {"figure": fig.to_dict()}

    # ── file: SCI / PSP ────────────────────────────────────────────────────────
    sci_psp_inline: dict = {}
    with guard("SCI / PSP", errors, label):
        fig   = build_sci_psp_figure(
            sci_scores, psp_per_ch, bad_channels, sci_threshold,
            sci_matrix=sci_matrix, sci_win_times=sci_win_times,
            psp_matrix=psp_matrix, psp_win_times=psp_win_times,
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

    # ── file: evoked topo ──────────────────────────────────────────────────────
    evoked_topo_inline: dict = {}
    if raw_haemo is not None:
        with guard("Evoked topography", errors, label):
            fig = build_evoked_topo_figure(
                raw_haemo, markers, _MAX_TS_PTS, fig_tmin, fig_tmax,
            )
            if fig:
                fname = f"{label}_desc-evokedtopo_nirs.html"
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["evoked_topo"] = {"src": f"figures/{fname}", "h": h}
                evoked_topo_inline = {"figure": fig.to_dict()}

    # ── file: per-channel detail HTML ──────────────────────────────────────────
    channel_pairs: list[str] = []
    if raw_haemo is not None:
        channel_pairs = get_channel_pairs(raw_haemo)
        for pair in channel_pairs:
            with guard("Channel detail", errors, f"{label} | {pair}"):
                detail_fig, psd_fig, epoch_fig = build_channel_figure(
                    raw_haemo, markers, pair, _MAX_TS_PTS, fig_tmin, fig_tmax,
                    cardiac=(cardiac_l_freq, cardiac_h_freq),
                )
                fname = f"{label}_desc-ch{_pair_fname(pair)}_nirs.html"
                _save_multi_fig_html([detail_fig, psd_fig, epoch_fig], fig_dir / fname)
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
    if epoch_qc:
        with guard("Per-trial quality", errors, label):
            labels, sqms = score_trials(raw, markers, sci_threshold,
                                        cardiac_l_freq, cardiac_h_freq,
                                        epoch_tmin, epoch_tmax,
                                        psp_threshold=cutoffs["psp"])
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
    record = sqm_record_dict(
        {**raw_secs, "windowed": windowed, "per_channel": raw_pc},
        [str(run["snirf_path"])],
    )
    sqm_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("SQM JSON → %s", sqm_path)
    save_channel_csv(ch_rows, label, sqm_dir, sci_threshold, psp_threshold=cutoffs["psp"])

    return {
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
        # print these and carry no copy of the cutoffs. The flat all-channel scalars used to
        # travel here too; nothing reads them now that the panels read the record's own
        # channel-set sections, and the quality record on disk is where the raw numbers live.
        "sqm": {
            "rows":        metric_rows(view_scalars, _VIEW_SCALAR_KEYS, skip_missing=True),
            "split":       split,
            "channel_set": "long channels" if sqm_split else "every channel",
        },
        # One table, at pair granularity: a decision is taken per source-detector pair and
        # SCI is a property of the pair rather than of either wavelength, so a per-wavelength
        # table beside this one would list every channel twice for no extra information. The
        # per-wavelength numbers are in the CSV written next to the record.
        "channels": {
            "pairs":  pair_cells,
            "blocks": separation_blocks(pair_cells),
            "notes":  separation_notes(raw_all, ch_rows),
        },
        "channel_pairs": channel_pairs,
        "figure_paths":  figure_paths,
        # the GUI builds its per-channel figures on demand and needs the run's own band to
        # shade the PSD the way the ones built here are shaded
        "cardiac":       [cardiac_l_freq, cardiac_h_freq],
        # what this run failed at and what it left out, already collapsed and ready to
        # print; the viewer renders these itself, in JavaScript
        "errors": collapse_messages(errors),
        "notes":  collapse_messages(notes),
    }


def build_prep_raw_report(
    runs: list[dict],
    output_path: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    dpf: list[float],
    sci_threshold: float = SCI_PASS,
    psp_threshold: float | None = None,
    window_s: float = 10.0,
    epoch_qc: bool = False,
    epoch_tmin: float | None = None,
    epoch_tmax: float | None = None,
    gvtd_channels: str = "long",
    sep_bands=None,
) -> None:
    """Generate raw QC report: lightweight HTML + per-run folders with figure HTMLs + SQM JSON."""
    # the report sits in the subject's own folder, so its figures are one level in from it
    # rather than a sibling tree, and sub-<id>/ can be moved or copied whole
    sub_dir     = output_path.parent
    static_data = []

    for i, run in enumerate(runs):
        label = run["label"]
        logger.info("[%d/%d] processing %s ...", i + 1, len(runs), label)
        # a run that fails outright still gets an entry, carrying the reason: the viewer's
        # run selector lists it either way, and an empty panel with no explanation reads as
        # a broken viewer rather than as a run that could not be read
        run_errors: list[str] = []
        d: dict = {}
        with guard("Processing this run", run_errors, label):
            d = _process_run(run, sci_threshold, sub_dir, cardiac_l_freq, cardiac_h_freq,
                             dpf, window_s, epoch_qc, epoch_tmin, epoch_tmax,
                             gvtd_channels, psp_threshold, sep_bands)
        if run_errors:
            d = {"errors": run_errors, "notes": []}
        static_data.append(d)

    run_labels = [r["label"] for r in runs]

    html = render(
        "raw_viewer.html",
        base_css=dashboard_css(),
        run_labels_json=json.dumps(run_labels),
        run_labels=run_labels,
        stem=output_path.stem,
        data_json=json.dumps(static_data),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Raw QC report saved: %s", output_path)
