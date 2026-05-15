"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import json
from pathlib import Path

import mne
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.prep_raw_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_MAX_TS_PTS = 4000
_SHORT_THRESH = 0.015
_EPOCH_TMIN = -5.0
_EPOCH_TMAX = 25.0


def _process_run(run: dict, sci_threshold: float) -> dict:
    from fnirs_pipe.qc.figures import (
        build_channel_figure,
        build_evoked_topo_figure,
        build_layout_figure,
        build_psd_mean_figure,
        build_sci_psp_figure,
        build_ts_figure,
        channel_quality_heatmap,
        condition_colors,
    )
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_iqm

    raw = mne.io.read_raw_snirf(run["snirf_path"], preload=True, verbose=False)

    # SCI
    try:
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        sci_arr = mne.preprocessing.nirs.scalp_coupling_index(raw_od, verbose=False)
        sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
    except Exception as exc:
        logger.warning("SCI failed: %s", exc)
        sci_scores = {ch: 1.0 for ch in raw.ch_names}
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)

    bad_channels: set[str] = {ch for ch, s in sci_scores.items() if s < sci_threshold}

    sci_matrix = sci_win_times = psp_matrix = psp_win_times = None
    try:
        from fnirs_pipe.pipeline.prep_pipeline import compute_windowed_psp, compute_windowed_sci
        sci_matrix, sci_win_times = compute_windowed_sci(raw_od)
        psp_matrix, psp_win_times = compute_windowed_psp(raw_od)
    except Exception as exc:
        logger.warning("Windowed SCI/PSP failed: %s", exc)

    try:
        iqm = compute_raw_iqm(raw, sci_scores, list(bad_channels))
    except Exception as exc:
        logger.warning("IQM failed: %s", exc)
        iqm = {}

    raw_haemo = None
    try:
        raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od.copy(), ppf=6.0)
    except Exception as exc:
        logger.warning("Beer-Lambert failed: %s", exc)

    markers = [
        {
            "onset":       float(a["onset"]),
            "duration":    float(a["duration"]),
            "description": str(a["description"]),
        }
        for a in raw.annotations
        if not str(a["description"]).upper().startswith("BAD")
    ]
    cond_colors = condition_colors(markers)
    for m in markers:
        m["color"] = cond_colors.get(m["description"], "#f39c12")

    psp_per_ch = iqm.get("psp_per_channel", {})

    # ts_figure
    ts_data = {}
    try:
        fig, marker_data, cond_colors_, band_shapes, t_start, t_end = build_ts_figure(
            raw, markers, bad_channels, _MAX_TS_PTS, _SHORT_THRESH,
        )
        ts_data = {
            "figure":      fig.to_dict(),
            "markers":     markers,
            "cond_colors": cond_colors_,
            "band_shapes": band_shapes,
            "t_start":     t_start,
            "t_end":       t_end,
        }
    except Exception as exc:
        logger.warning("ts_figure failed: %s", exc)

    # layout
    layout_data = {}
    try:
        fig_2d, fig_3d = build_layout_figure(raw, bad_channels, sci_scores, _SHORT_THRESH)
        layout_data = {
            "layout_2d_figure": fig_2d.to_dict() if fig_2d else None,
            "layout_3d_figure": fig_3d.to_dict() if fig_3d else None,
        }
    except Exception as exc:
        logger.warning("layout_figure failed: %s", exc)

    # sci_psp
    sci_psp_data = {}
    try:
        fig = build_sci_psp_figure(
            sci_scores, psp_per_ch, bad_channels, sci_threshold,
            sci_matrix=sci_matrix, sci_win_times=sci_win_times,
            psp_matrix=psp_matrix, psp_win_times=psp_win_times,
        )
        sci_psp_data = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("sci_psp_figure failed: %s", exc)

    # psd_mean
    psd_data = {}
    try:
        fig = build_psd_mean_figure(raw)
        if fig:
            psd_data = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("psd_mean_figure failed: %s", exc)

    # channel_summary
    ch_summary_data = {}
    try:
        ch_names = list(sci_scores.keys())
        is_bad = [ch in bad_channels for ch in ch_names]
        fig = channel_quality_heatmap(
            ch_names, is_bad,
            sci_per_ch=iqm.get("sci_per_channel", sci_scores),
            cv_per_ch=iqm.get("cv_per_channel", {}),
            snr_per_ch=iqm.get("snr_per_channel", {}),
            psp_per_ch=psp_per_ch,
            sci_thresh=sci_threshold,
        )
        ch_summary_data = {"figure": fig.to_dict()}
    except Exception as exc:
        logger.warning("channel_quality_heatmap failed: %s", exc)

    # iqm
    iqm_data = {
        "scalars":     {k: v for k, v in iqm.items() if not isinstance(v, (dict, list))},
        "per_channel": {k: v for k, v in iqm.items() if isinstance(v, dict) and k.endswith("_per_channel")},
    }

    # per-channel detail
    channels: dict[str, dict] = {}
    if raw_haemo is not None:
        pairs = sorted({ch.rsplit(" ", 1)[0] for ch in raw_haemo.ch_names if ch.endswith(" hbo")})
        for pair in pairs:
            try:
                detail_fig, psd_fig, epoch_fig = build_channel_figure(
                    raw_haemo, markers, pair, _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX,
                )
                channels[pair] = {
                    "detail_figure": detail_fig.to_dict() if detail_fig else None,
                    "psd_figure":    psd_fig.to_dict()    if psd_fig    else None,
                    "epoch_figure":  epoch_fig.to_dict()  if epoch_fig  else None,
                }
            except Exception as exc:
                logger.warning("channel_figure %s failed: %s", pair, exc)

    evoked_topo_data = {}
    if raw_haemo is not None:
        try:
            fig = build_evoked_topo_figure(raw_haemo, markers)
            if fig:
                evoked_topo_data = {"figure": fig.to_dict()}
        except Exception as exc:
            logger.warning("evoked_topo_figure failed: %s", exc)

    return {
        "ts":          ts_data,
        "layout":      layout_data,
        "sci_psp":     sci_psp_data,
        "psd":         psd_data,
        "ch_summary":  ch_summary_data,
        "iqm":         iqm_data,
        "channels":    channels,
        "evoked_topo": evoked_topo_data,
    }


def build_prep_raw_report(
    runs: list[dict],
    output_path: Path,
    sci_threshold: float = 0.8,
) -> None:
    """Generate static HTML raw QC report (same layout as the Flask viewer)."""
    static_data = []
    for i, run in enumerate(runs):
        logger.info("[%d/%d] processing %s ...", i + 1, len(runs), run["label"])
        try:
            static_data.append(_process_run(run, sci_threshold))
        except Exception as exc:
            logger.error("Failed to process run %s: %s", run["label"], exc)
            static_data.append({})

    run_labels = [r["label"] for r in runs]

    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    template = env.get_template("raw_viewer.html")
    html = template.render(
        run_labels_json=json.dumps(run_labels),
        data_json=json.dumps(static_data),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Raw QC report saved: %s", output_path)
