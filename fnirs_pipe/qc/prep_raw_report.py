"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import json
from pathlib import Path

import mne
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.qc.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.quantitative_metrics import SHORT_MAX_DIST
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.prep_raw_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_MAX_TS_PTS   = 4000
# the figures colour a channel short or not short, so only the short edge applies here
_SHORT_THRESH = SHORT_MAX_DIST
_EPOCH_TMIN   = -5.0
_EPOCH_TMAX   = 25.0


def _trial_windows(
    markers: list[dict],
    tmin: float | None,
    tmax: float | None,
    duration: float,
) -> list[tuple[str, float, float]]:
    """Turn the event list into (label, t0, t1) windows to score one at a time.

    Two ways to size a window, chosen by whether tmin/tmax were given:
    fixed, `[onset+tmin, onset+tmax]`, which lets a negative tmin pull in a baseline; or the
    event's own duration, `[onset, onset+duration]`, for block designs that record one.

    Example: an event at 30.0 s of 8 s with tmin/tmax unset yields
    ``("trial-001_30s_speak", 30.0, 38.0)``.

    Events that describe no window are dropped rather than guessed at: a zero duration with no
    tmin/tmax has no extent, and a window starting past the end of the recording has no data.
    """
    fixed = tmin is not None and tmax is not None
    windows: list[tuple[str, float, float]] = []
    for i, m in enumerate(markers, start=1):
        onset = float(m["onset"])
        if fixed:
            t0, t1 = onset + tmin, onset + tmax
        elif float(m["duration"]) > 0:
            t0, t1 = onset, onset + float(m["duration"])
        else:
            logger.warning("trial %d at %.1fs has no duration and no --epoch-tmin/--epoch-tmax; "
                           "skipping", i, onset)
            continue
        t0, t1 = max(0.0, t0), min(duration, t1)
        if t1 - t0 <= 0:
            logger.warning("trial %d at %.1fs falls outside the recording; skipping", i, onset)
            continue
        cond = str(m.get("description", "")).strip()
        windows.append((f"trial-{i:03d}_{onset:.0f}s" + (f"_{cond}" if cond else ""), t0, t1))
    return windows


def _trial_sqm(raw, t0: float, t1: float,
               sci_threshold: float, cardiac_l_freq: float, cardiac_h_freq: float) -> dict:
    """SQM scalars for one trial window, scored the way the whole recording was.

    The intensity recording is what gets cropped, not the optical density derived from it,
    so that a trial's CV, SNR and spike count sit on the same scale as the recording-level
    numbers in the same report. Reusing the whole-recording OD object would be cheaper by one
    conversion per trial but would put the two sets of figures on different footings.

    No sliding-window series is attached: a window of a few seconds has no room for the 10 s
    grid the recording-level series uses.
    """
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_sqm, compute_sci_scores

    seg = raw.copy().crop(tmin=t0, tmax=t1)
    sci_scores, _ = compute_sci_scores(seg, cardiac_l_freq, cardiac_h_freq)
    bad = [ch for ch, v in sci_scores.items() if v < sci_threshold]
    try:
        return compute_raw_sqm(seg, sci_scores, bad, cardiac_l_freq, cardiac_h_freq)
    except Exception as exc:
        logger.warning("trial SQM failed: %s", exc)
        return {}


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
) -> dict:
    """Compute all data, save figure HTMLs + SQM JSON. Returns inline dict for HTML."""
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
    from fnirs_pipe.qc.quantitative_metrics import (
        attach_windowed_series, compute_raw_sqm, compute_sci_scores,
    )
    from fnirs_pipe.qc.sqm_record import raw_sections, sqm_record_dict

    label   = run["label"]
    session = run.get("session")
    fig_dir = sub_dir / "figures"
    sqm_dir = sub_dir / (f"ses-{session}" if session else "") / "nirs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sqm_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_snirf(run["snirf_path"], preload=True, verbose=False)

    sci_scores, raw_od = compute_sci_scores(raw, cardiac_l_freq, cardiac_h_freq)
    bad_channels: set[str] = {ch for ch, s in sci_scores.items() if s < sci_threshold}

    try:
        sqm = compute_raw_sqm(raw, sci_scores, list(bad_channels), cardiac_l_freq, cardiac_h_freq)
    except Exception as exc:
        logger.warning("SQM failed: %s", exc)
        sqm = {}

    # `sqm` stays the flat all-channel view the figures below read. The record written to
    # disk is the sectioned one, built through the same function the pipeline uses.
    raw_secs, raw_pc = raw_sections(
        raw, sci_scores, list(bad_channels), cardiac_l_freq, cardiac_h_freq)

    # Persist windowed series so group_raw can build time × subject heatmaps. Their own
    # section, since `_split_scalars` would file every one of these lists under per_channel.
    windowed: dict = {}
    series = attach_windowed_series(windowed, raw_od, cardiac_l_freq, cardiac_h_freq, window_s)
    sci_matrix, sci_win_times = series["sci_matrix"], series["sci_times"]
    psp_matrix, psp_win_times = series["psp_matrix"], series["psp_times"]

    raw_haemo = None
    try:
        ppf = dpf[0] if len(dpf) == 1 else dpf
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=ppf)
    except Exception as exc:
        logger.warning("Beer-Lambert failed: %s", exc)

    markers = extract_markers(raw)
    cond_colors_ = condition_colors(markers)
    for m in markers:
        m["color"] = cond_colors_.get(m["description"], "#f39c12")

    psp_per_ch    = sqm.get("psp_per_channel", {})
    figure_paths: dict = {}

    # ── inline: ts figure (kept in-memory for click interactivity) ─────────────
    ts_inline: dict = {}
    try:
        fig, _mkdata, cond_colors_out, band_shapes, t_start, t_end = build_ts_figure(
            raw, markers, bad_channels, _MAX_TS_PTS, _SHORT_THRESH,
        )
        ts_inline = {
            "figure":      fig.to_dict(),
            "markers":     markers,
            "cond_colors": cond_colors_out,
            "band_shapes": band_shapes,
            "t_start":     t_start,
            "t_end":       t_end,
        }
    except Exception as exc:
        logger.warning("ts_figure failed: %s", exc)

    # ── inline: layout figures (kept for click interactivity) ──────────────────
    layout_inline: dict = {}
    try:
        fig_2d, fig_3d = build_layout_figure(raw, bad_channels, sci_scores, _SHORT_THRESH)
        layout_inline = {
            "layout_2d_figure": fig_2d.to_dict() if fig_2d else None,
            "layout_3d_figure": fig_3d.to_dict() if fig_3d else None,
        }
    except Exception as exc:
        logger.warning("layout_figure failed: %s", exc)

    # ── file: carpet GVTD ──────────────────────────────────────────────────
    #   an iframe rather than inlined like the panels above: the carpet is a channels x 2000
    #   heatmap, and every run of the viewer would carry one in the page itself
    carpet_inline: dict = {}
    try:
        from fnirs_pipe.qc.quantitative_metrics import gvtd_channel_picks
        gvtd_picks, gvtd_set = gvtd_channel_picks(raw, gvtd_channels)
        raw_carpet = raw.copy().pick(gvtd_picks)
        fig   = carpet_gvtd_figure(raw_carpet, raw_carpet.ch_names, channel_set=gvtd_set)
        fname = f"{label}_desc-carpet_nirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["carpet"] = {"src": f"figures/{fname}", "h": h}
        carpet_inline = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("carpet_gvtd_figure failed: %s", exc)

    # ── file: SCI / PSP ────────────────────────────────────────────────────────
    sci_psp_inline: dict = {}
    try:
        fig   = build_sci_psp_figure(
            sci_scores, psp_per_ch, bad_channels, sci_threshold,
            sci_matrix=sci_matrix, sci_win_times=sci_win_times,
            psp_matrix=psp_matrix, psp_win_times=psp_win_times,
        )
        fname = f"{label}_desc-scipsp_nirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["sci_psp"] = {"src": f"figures/{fname}", "h": h}
        sci_psp_inline = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("sci_psp_figure failed: %s", exc)

    # ── file: PSD mean ─────────────────────────────────────────────────────────
    try:
        fig = build_psd_mean_figure(raw)
        if fig:
            fname = f"{label}_desc-psd_nirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["psd"] = {"src": f"figures/{fname}", "h": h}
    except Exception as exc:
        logger.warning("psd_mean_figure failed: %s", exc)

    # ── file: trigger timeline ─────────────────────────────────────────────────
    trigger_timeline_inline: dict = {}
    try:
        fig = build_trigger_timeline_single(markers, cond_colors_)
        if fig:
            fname = f"{label}_desc-trigger_nirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["trigger"] = {"src": f"figures/{fname}", "h": h}
            trigger_timeline_inline = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("trigger_timeline_single failed: %s", exc)

    # ── file: channel quality summary ──────────────────────────────────────────
    ch_summary_inline: dict = {}
    try:
        ch_names = list(sci_scores.keys())
        is_bad   = [ch in bad_channels for ch in ch_names]
        fig = channel_quality_heatmap(
            ch_names, is_bad,
            sci_per_ch=sqm.get("sci_per_channel", sci_scores),
            cv_per_ch=sqm.get("cv_per_channel", {}),
            snr_per_ch=sqm.get("snr_per_channel", {}),
            psp_per_ch=psp_per_ch,
            sci_thresh=sci_threshold,
        )
        fname = f"{label}_desc-chsummary_nirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["ch_summary"] = {"src": f"figures/{fname}", "h": h}
        ch_summary_inline = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("channel_quality_heatmap failed: %s", exc)

    # the epoch figures need concrete bounds; per-trial QC reads None as "use event duration"
    fig_tmin = _EPOCH_TMIN if epoch_tmin is None else epoch_tmin
    fig_tmax = _EPOCH_TMAX if epoch_tmax is None else epoch_tmax

    # ── file: evoked topo ──────────────────────────────────────────────────────
    evoked_topo_inline: dict = {}
    if raw_haemo is not None:
        try:
            fig = build_evoked_topo_figure(
                raw_haemo, markers, _MAX_TS_PTS, fig_tmin, fig_tmax,
            )
            if fig:
                fname = f"{label}_desc-evokedtopo_nirs.html"
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["evoked_topo"] = {"src": f"figures/{fname}", "h": h}
                evoked_topo_inline = {"figure": fig.to_dict()}
        except Exception as exc:
            logger.warning("evoked_topo_figure failed: %s", exc)

    # ── file: per-channel detail HTML ──────────────────────────────────────────
    channel_pairs: list[str] = []
    if raw_haemo is not None:
        channel_pairs = get_channel_pairs(raw_haemo)
        for pair in channel_pairs:
            try:
                detail_fig, psd_fig, epoch_fig = build_channel_figure(
                    raw_haemo, markers, pair, _MAX_TS_PTS, fig_tmin, fig_tmax,
                )
                fname = f"{label}_desc-ch{_pair_fname(pair)}_nirs.html"
                _save_multi_fig_html([detail_fig, psd_fig, epoch_fig], fig_dir / fname)
            except Exception as exc:
                logger.warning("channel_figure %s failed: %s", pair, exc)
        if channel_pairs:
            figure_paths["ch_detail_template"] = (
                f"figures/{label}_desc-ch{{pair}}_nirs.html"
            )

    # ── file: per-trial QC ─────────────────────────────────────────────────────
    # scored here rather than persisted: a trial is not a BIDS entity, so per-trial records
    # have nowhere to live in the derivatives tree without colliding on filename
    trial_qc_inline: dict = {}
    if epoch_qc:
        try:
            windows = _trial_windows(markers, epoch_tmin, epoch_tmax, float(raw.times[-1]))
            labels  = [w[0] for w in windows]
            sqms    = [_trial_sqm(raw, t0, t1, sci_threshold,
                                  cardiac_l_freq, cardiac_h_freq) for _, t0, t1 in windows]
            fig = trial_quality_heatmap(labels, sqms)
            if fig:
                fname = f"{label}_desc-trialqc_nirs.html"
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["trial_qc"] = {"src": f"figures/{fname}", "h": h}
                trial_qc_inline = {"figure": fig.to_dict(), "n_trials": len(labels)}
            elif not windows:
                logger.warning("%s: no usable trial windows; per-trial QC skipped", label)
        except Exception as exc:
            logger.warning("trial_quality_heatmap failed: %s", exc)

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

    return {
        "ts":           ts_inline,
        "layout":       layout_inline,
        "evoked_topo":  evoked_topo_inline,
        "carpet_gvtd":  carpet_inline,
        "sci_psp":          sci_psp_inline,
        "ch_summary":       ch_summary_inline,
        "trigger_timeline": trigger_timeline_inline,
        "trial_qc":         trial_qc_inline,
        "sqm": {
            "scalars":     {k: v for k, v in sqm.items() if not isinstance(v, (dict, list))},
            "per_channel": {"sci_per_channel": sqm.get("sci_per_channel", {})},
        },
        "channel_pairs": channel_pairs,
        "figure_paths":  figure_paths,
    }


def build_prep_raw_report(
    runs: list[dict],
    output_path: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    dpf: list[float],
    sci_threshold: float = 0.8,
    window_s: float = 10.0,
    epoch_qc: bool = False,
    epoch_tmin: float | None = None,
    epoch_tmax: float | None = None,
    gvtd_channels: str = "long",
) -> None:
    """Generate raw QC report: lightweight HTML + per-run folders with figure HTMLs + SQM JSON."""
    # the report sits in the subject's own folder, so its figures are one level in from it
    # rather than a sibling tree, and sub-<id>/ can be moved or copied whole
    sub_dir     = output_path.parent
    static_data = []

    for i, run in enumerate(runs):
        label = run["label"]
        logger.info("[%d/%d] processing %s ...", i + 1, len(runs), label)
        try:
            d = _process_run(run, sci_threshold, sub_dir, cardiac_l_freq, cardiac_h_freq, dpf,
                             window_s, epoch_qc, epoch_tmin, epoch_tmax, gvtd_channels)
            static_data.append(d)
        except Exception as exc:
            logger.error("Failed to process run %s: %s", label, exc)
            static_data.append({})

    run_labels = [r["label"] for r in runs]

    env      = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    template = env.get_template("raw_viewer.html")
    html     = template.render(
        run_labels_json=json.dumps(run_labels),
        run_labels=run_labels,
        stem=output_path.stem,
        data_json=json.dumps(static_data),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Raw QC report saved: %s", output_path)
