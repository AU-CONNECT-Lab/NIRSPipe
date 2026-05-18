"""Save the raw QC viewer as a static HTML file (no Flask needed)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import mne
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.prep_raw_report")

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_MAX_TS_PTS   = 4000
_SHORT_THRESH = 0.015
_EPOCH_TMIN   = -5.0
_EPOCH_TMAX   = 25.0
_IFRAME_CSS   = "html,body{margin:0;padding:0;overflow:hidden;width:100%;height:100%;}"


def _ensure_plotly_js(target_dir: Path) -> None:
    """Copy plotly.min.js from the installed package to target_dir (once)."""
    dst = target_dir / "plotly.min.js"
    if not dst.exists():
        import shutil
        import plotly as _plotly
        src = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
        shutil.copy2(src, dst)


def _figure_height(fig, default: int = 500) -> int:
    h = getattr(getattr(fig, "layout", None), "height", None)
    if h:
        return int(h)
    for trace in getattr(fig, "data", ()):
        if getattr(trace, "type", "") == "heatmap":
            y = getattr(trace, "y", None)
            if y is not None:
                return max(300, min(len(y) * 22 + 140, 1400))
    return default


def _save_figure_html(fig, path: Path) -> int:
    """Save a single Plotly figure as standalone iframe-ready HTML. Returns height px."""
    h = _figure_height(fig)
    fig.update_layout(height=h)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_plotly_js(path.parent)
    html = fig.to_html(
        full_html=True, include_plotlyjs=False, config={"responsive": True}
    )
    html = html.replace(
        "<head>",
        '<head>\n<style>' + _IFRAME_CSS + '</style>\n<script src="plotly.min.js"></script>',
        1,
    )
    path.write_text(html, encoding="utf-8")
    return h


def _save_multi_fig_html(figs: list, path: Path) -> int:
    """Stack multiple Plotly figures in one HTML file. Returns total height px."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_plotly_js(path.parent)
    head = (
        '<meta charset="UTF-8">'
        '<script src="plotly.min.js"></script>'
        f'<style>*{{box-sizing:border-box;}}{_IFRAME_CSS}.pfig{{margin-bottom:2px;}}</style>'
    )
    parts = [f"<!DOCTYPE html><html><head>{head}</head><body>"]
    total_h = 0
    for i, fig in enumerate(figs):
        if fig is None:
            continue
        h = _figure_height(fig)
        total_h += h + 2
        fig.update_layout(height=h)
        fig_html = fig.to_html(
            full_html=False, include_plotlyjs=False,
            div_id=f"pfig{i}", config={"responsive": True},
        )
        parts.append(f'<div class="pfig">{fig_html}</div>')
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return max(total_h, 100)


def _pair_fname(pair: str) -> str:
    return pair.replace(" ", "_").replace("/", "-").replace("\\", "-")


def _process_run(
    run: dict,
    sci_threshold: float,
    run_dir: Path,
) -> dict:
    """Compute all data, save figure HTMLs + IQM JSON. Returns inline dict for HTML."""
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
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_iqm

    label   = run["label"]
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_snirf(run["snirf_path"], preload=True, verbose=False)

    try:
        raw_od  = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
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
    cond_colors_ = condition_colors(markers)
    for m in markers:
        m["color"] = cond_colors_.get(m["description"], "#f39c12")

    psp_per_ch    = iqm.get("psp_per_channel", {})
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
    try:
        b64      = carpet_gvtd_figure(raw, raw.ch_names)
        png_name = f"{label}_desc-carpet_fnirs.png"
        (fig_dir / png_name).write_bytes(base64.b64decode(b64))
        figure_paths["carpet"] = {"src": f"{label}/figures/{png_name}"}
    except Exception as exc:
        logger.warning("carpet_gvtd_figure failed: %s", exc)

    # ── file: SCI / PSP ────────────────────────────────────────────────────────
    try:
        fig   = build_sci_psp_figure(
            sci_scores, psp_per_ch, bad_channels, sci_threshold,
            sci_matrix=sci_matrix, sci_win_times=sci_win_times,
            psp_matrix=psp_matrix, psp_win_times=psp_win_times,
        )
        fname = f"{label}_desc-scipsp_fnirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["sci_psp"] = {"src": f"{label}/figures/{fname}", "h": h}
    except Exception as exc:
        logger.warning("sci_psp_figure failed: %s", exc)

    # ── file: PSD mean ─────────────────────────────────────────────────────────
    try:
        fig = build_psd_mean_figure(raw)
        if fig:
            fname = f"{label}_desc-psd_fnirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["psd"] = {"src": f"{label}/figures/{fname}", "h": h}
    except Exception as exc:
        logger.warning("psd_mean_figure failed: %s", exc)

    # ── file: trigger timeline ─────────────────────────────────────────────────
    try:
        fig = build_trigger_timeline_single(markers, cond_colors_)
        if fig:
            fname = f"{label}_desc-trigger_fnirs.html"
            h     = _save_figure_html(fig, fig_dir / fname)
            figure_paths["trigger"] = {"src": f"{label}/figures/{fname}", "h": h}
    except Exception as exc:
        logger.warning("trigger_timeline_single failed: %s", exc)

    # ── file: channel quality summary ──────────────────────────────────────────
    try:
        ch_names = list(sci_scores.keys())
        is_bad   = [ch in bad_channels for ch in ch_names]
        fig = channel_quality_heatmap(
            ch_names, is_bad,
            sci_per_ch=iqm.get("sci_per_channel", sci_scores),
            cv_per_ch=iqm.get("cv_per_channel", {}),
            snr_per_ch=iqm.get("snr_per_channel", {}),
            psp_per_ch=psp_per_ch,
            sci_thresh=sci_threshold,
        )
        fname = f"{label}_desc-chsummary_fnirs.html"
        h     = _save_figure_html(fig, fig_dir / fname)
        figure_paths["ch_summary"] = {"src": f"{label}/figures/{fname}", "h": h}
    except Exception as exc:
        logger.warning("channel_quality_heatmap failed: %s", exc)

    # ── file: evoked topo ──────────────────────────────────────────────────────
    if raw_haemo is not None:
        try:
            fig = build_evoked_topo_figure(raw_haemo, markers)
            if fig:
                fname = f"{label}_desc-evokedtopo_fnirs.html"
                h     = _save_figure_html(fig, fig_dir / fname)
                figure_paths["evoked_topo"] = {"src": f"{label}/figures/{fname}", "h": h}
        except Exception as exc:
            logger.warning("evoked_topo_figure failed: %s", exc)

    # ── file: per-channel detail HTML ──────────────────────────────────────────
    channel_pairs: list[str] = []
    if raw_haemo is not None:
        channel_pairs = sorted({
            ch.rsplit(" ", 1)[0] for ch in raw_haemo.ch_names if ch.endswith(" hbo")
        })
        for pair in channel_pairs:
            try:
                detail_fig, psd_fig, epoch_fig = build_channel_figure(
                    raw_haemo, markers, pair, _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX,
                )
                fname = f"{label}_desc-ch-{_pair_fname(pair)}_fnirs.html"
                _save_multi_fig_html([detail_fig, psd_fig, epoch_fig], fig_dir / fname)
            except Exception as exc:
                logger.warning("channel_figure %s failed: %s", pair, exc)
        if channel_pairs:
            figure_paths["ch_detail_template"] = (
                f"{label}/figures/{label}_desc-ch-{{pair}}_fnirs.html"
            )

    # ── file: IQM JSON ─────────────────────────────────────────────────────────
    iqm_path = run_dir / f"{label}_desc-iqm_fnirs.json"
    iqm_path.write_text(json.dumps(iqm, indent=2, default=str), encoding="utf-8")
    logger.info("IQM JSON → %s", iqm_path)

    return {
        "ts":     ts_inline,
        "layout": layout_inline,
        "iqm": {
            "scalars":     {k: v for k, v in iqm.items() if not isinstance(v, (dict, list))},
            "per_channel": {"sci_per_channel": iqm.get("sci_per_channel", {})},
        },
        "channel_pairs": channel_pairs,
        "figure_paths":  figure_paths,
    }


def build_prep_raw_report(
    runs: list[dict],
    output_path: Path,
    sci_threshold: float = 0.8,
) -> None:
    """Generate raw QC report: lightweight HTML + per-run folders with figure HTMLs + IQM JSON."""
    output_dir  = output_path.parent
    static_data = []

    for i, run in enumerate(runs):
        label   = run["label"]
        run_dir = output_dir / label
        logger.info("[%d/%d] processing %s ...", i + 1, len(runs), label)
        try:
            d = _process_run(run, sci_threshold, run_dir)
            static_data.append(d)
        except Exception as exc:
            logger.error("Failed to process run %s: %s", label, exc)
            static_data.append({})

    _ensure_plotly_js(output_dir)
    run_labels = [r["label"] for r in runs]

    env      = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    template = env.get_template("raw_viewer.html")
    html     = template.render(
        run_labels_json=json.dumps(run_labels),
        data_json=json.dumps(static_data),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Raw QC report saved: %s", output_path)
