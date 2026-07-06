"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import mne
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.qc.figure_io import (
    _pair_fname, _save_figure_html, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.prep_raw_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_MAX_TS_PTS   = 4000
_SHORT_THRESH = 0.015
_EPOCH_TMIN   = -5.0
_EPOCH_TMAX   = 25.0


def _process_run(
    run: dict,
    sci_threshold: float,
    sub_dir: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
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
    )
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_sqm

    label   = run["label"]
    session = run.get("session")
    fig_dir = sub_dir / "figures"
    sqm_dir = sub_dir / (f"ses-{session}" if session else "") / "nirs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sqm_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_snirf(run["snirf_path"], preload=True, verbose=False)

    try:
        raw_od  = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        sci_arr = mne.preprocessing.nirs.scalp_coupling_index(
            raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
        sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
    except Exception as exc:
        logger.warning("SCI failed: %s", exc)
        sci_scores = {ch: 1.0 for ch in raw.ch_names}
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)

    bad_channels: set[str] = {ch for ch, s in sci_scores.items() if s < sci_threshold}

    sci_matrix = sci_win_times = psp_matrix = psp_win_times = None
    gvtd_per_window = gvtd_win_times = gvtd_filt_per_window = None
    try:
        from fnirs_pipe.pipeline.prep_pipeline import (
            compute_windowed_filtered_gvtd, compute_windowed_gvtd,
            compute_windowed_psp, compute_windowed_sci,
        )
        sci_matrix, sci_win_times = compute_windowed_sci(raw_od, cardiac_l_freq, cardiac_h_freq)
        psp_matrix, psp_win_times = compute_windowed_psp(raw_od, cardiac_l_freq, cardiac_h_freq)
        gvtd_per_window, gvtd_win_times = compute_windowed_gvtd(raw_od)
        gvtd_filt_per_window, _ = compute_windowed_filtered_gvtd(raw_od)
    except Exception as exc:
        logger.warning("Windowed SCI/PSP/GVTD failed: %s", exc)

    try:
        sqm = compute_raw_sqm(raw, sci_scores, list(bad_channels), cardiac_l_freq, cardiac_h_freq)
    except Exception as exc:
        logger.warning("SQM failed: %s", exc)
        sqm = {}

    # Persist windowed series so group_raw can build time × subject heatmaps.
    # mne-nirs returns ndarray scores but list-of-[start,end] times → collapse to center.
    import numpy as _np
    def _center_times(t):
        a = _np.asarray(t)
        return (a.mean(axis=1) if a.ndim == 2 and a.shape[1] == 2 else a).tolist()

    if sci_matrix is not None and sci_win_times is not None:
        sqm["sci_per_window"]      = _np.asarray(sci_matrix).mean(axis=0).tolist()
        sqm["sci_window_times_s"]  = _center_times(sci_win_times)
    if psp_matrix is not None and psp_win_times is not None:
        sqm["psp_per_window"]      = _np.asarray(psp_matrix).mean(axis=0).tolist()
        sqm["psp_window_times_s"]  = _center_times(psp_win_times)
    if gvtd_per_window is not None and len(gvtd_per_window):
        sqm["gvtd_per_window"]     = _np.asarray(gvtd_per_window).tolist()
        sqm["gvtd_window_times_s"] = _center_times(gvtd_win_times)
    if gvtd_filt_per_window is not None and len(gvtd_filt_per_window):
        sqm["gvtd_filt_per_window"] = _np.asarray(gvtd_filt_per_window).tolist()

    raw_haemo = None
    try:
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=6.0)
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

    # ── file: carpet GVTD (PNG) ────────────────────────────────────────────────
    carpet_b64 = None
    try:
        carpet_b64 = carpet_gvtd_figure(raw, raw.ch_names)
        png_name   = f"{label}_desc-carpet_nirs.png"
        (fig_dir / png_name).write_bytes(base64.b64decode(carpet_b64))
        figure_paths["carpet"] = {"src": f"{sub_dir.name}/figures/{png_name}"}
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
        figure_paths["sci_psp"] = {"src": f"{sub_dir.name}/figures/{fname}", "h": h}
        sci_psp_inline = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("sci_psp_figure failed: %s", exc)

    # ── file: PSD mean ─────────────────────────────────────────────────────────
    try:
        fig = build_psd_mean_figure(raw)
        if fig:
            fname = f"{label}_desc-psd_nirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["psd"] = {"src": f"{sub_dir.name}/figures/{fname}", "h": h}
    except Exception as exc:
        logger.warning("psd_mean_figure failed: %s", exc)

    # ── file: trigger timeline ─────────────────────────────────────────────────
    trigger_timeline_inline: dict = {}
    try:
        fig = build_trigger_timeline_single(markers, cond_colors_)
        if fig:
            fname = f"{label}_desc-trigger_nirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["trigger"] = {"src": f"{sub_dir.name}/figures/{fname}", "h": h}
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
        figure_paths["ch_summary"] = {"src": f"{sub_dir.name}/figures/{fname}", "h": h}
        ch_summary_inline = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("channel_quality_heatmap failed: %s", exc)

    # ── file: evoked topo ──────────────────────────────────────────────────────
    evoked_topo_inline: dict = {}
    if raw_haemo is not None:
        try:
            fig = build_evoked_topo_figure(raw_haemo, markers)
            if fig:
                fname = f"{label}_desc-evokedtopo_nirs.html"
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["evoked_topo"] = {"src": f"{sub_dir.name}/figures/{fname}", "h": h}
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
                    raw_haemo, markers, pair, _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX,
                )
                fname = f"{label}_desc-ch{_pair_fname(pair)}_nirs.html"
                _save_multi_fig_html([detail_fig, psd_fig, epoch_fig], fig_dir / fname)
            except Exception as exc:
                logger.warning("channel_figure %s failed: %s", pair, exc)
        if channel_pairs:
            figure_paths["ch_detail_template"] = (
                f"{sub_dir.name}/figures/{label}_desc-ch{{pair}}_nirs.html"
            )

    # ── file: SQM JSON ─────────────────────────────────────────────────────────
    sqm_path = sqm_dir / f"{label}_desc-sqm_nirs.json"
    sqm_path.write_text(json.dumps(sqm, indent=2, default=str), encoding="utf-8")
    logger.info("SQM JSON → %s", sqm_path)

    return {
        "ts":           ts_inline,
        "layout":       layout_inline,
        "evoked_topo":  evoked_topo_inline,
        "carpet_gvtd":  {"b64": carpet_b64} if carpet_b64 else {},
        "sci_psp":          sci_psp_inline,
        "ch_summary":       ch_summary_inline,
        "trigger_timeline": trigger_timeline_inline,
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
    sci_threshold: float = 0.8,
) -> None:
    """Generate raw QC report: lightweight HTML + per-run folders with figure HTMLs + SQM JSON."""
    output_dir  = output_path.parent
    static_data = []

    for i, run in enumerate(runs):
        label   = run["label"]
        sub_dir = output_dir / f"sub-{run['subject_id']}"
        logger.info("[%d/%d] processing %s ...", i + 1, len(runs), label)
        try:
            d = _process_run(run, sci_threshold, sub_dir, cardiac_l_freq, cardiac_h_freq)
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
