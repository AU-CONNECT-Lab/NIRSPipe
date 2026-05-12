"""Generate per-subject HTML QC report using Jinja2 + Plotly.

Report architecture: reportlets
---------------------------------
Each preprocessing/postprocessing step contributes an independent HTML fragment
(a "reportlet"). build_subject_report() collects all reportlets and renders them
into a single per-subject HTML via Jinja2.

If a step fails or is skipped, its reportlet slot is left empty with a warning
banner — the rest of the report still renders.

References:
  MNE built-in report:  https://mne.tools/stable/auto_tutorials/preprocessing/14_quality_control_report.html

IQM output
----------------------------------
Inspired by MRIQC/AFNI. See quantitative_metrics.py for metric definitions.
Candidate sidecar format: JSONL (one record per subject/session).

Raw signal QC (computed on intensity or OD data, before Beer-Lambert):
  - sci_mean / sci_per_channel    : Scalp Coupling Index
  - psp_mean / psp_per_channel    : Peak Spectral Power at cardiac frequency
  - cv_mean / cv_per_channel      : Coefficient of Variation of raw intensity
  - channel_retention_rate        : fraction of channels surviving SCI threshold

Processed data QC (computed on HbO/HbR time series, model-agnostic):
  - tsnr_hbo / tsnr_hbr           : temporal SNR per channel (mean / std)
  - hbo_hbr_corr                  : HbO-HbR Pearson r per channel (expect < 0)
  - residual_cardiac_power        : HbO power in 0.7-1.5 Hz after bandpass
  - residual_resp_power           : HbO power in 0.1-0.5 Hz after bandpass
  - lowfreq_drift_amplitude_hbo   : HbO peak-to-peak amplitude below 0.01 Hz
  - lowfreq_drift_amplitude_hbr   : HbR peak-to-peak amplitude below 0.01 Hz
  - spike_count                   : number of 1st-derivative threshold crossings
  - temporal_derivative_variance  : var(diff(hbo)) — sensitive to abrupt changes
  - pct_data_retained             : fraction of timepoints not annotated as bad

Design rationale (following MRIQC, Esteban et al. 2017 PLOS ONE):
  metrics are selected from literature + expert judgement; they must be
  (1) model-agnostic — no task design or HRF assumption required,
  (2) computable from standard MNE/numpy without extra dependencies,
  (3) interpretable as a scalar per channel or a single summary scalar.
  R2, CNR of HRF, and other GLM-derived quantities are explicitly excluded
  because they depend on the task design and are not data quality indicators.

Report sections and content
-----------------------------
1. Executive Summary (top of report)
   - Subject metadata: ID, session, age, acquisition hardware
   - BIDS validation result
   - Key QC indicators with traffic lights:
       bad channel rate  (<10% green / 10-30% yellow / >30% red)
       mean SCI
       mean cardiac peak power
       % frames censored by FD threshold (if used)
       HbO/HbR mean correlation

2. Per-step preprocessing reportlets (in pipeline order)
   a. Raw data
      - mne.Report: raw signal traces (HbO/HbR or intensity)
      - mne.Report: PSD of raw signal
   b. OD conversion
      - mne.Report: PSD after OD conversion (verify no DC offset issues)
   c. SCI / bad channels
      - figures.sci_topography(): 2-D probe layout coloured by SCI score
      - figures.peak_power_plot(): cardiac peak power bar chart per channel
      - figures.channel_snr_plot(): per-channel SNR in physiological band
      - Table: all channels flagged as bad + their SCI scores
      NOTE: this section is the primary decision aid for --bad-channel-action.
      When --bad-channel-action keep is used, bad channels are shown with a
      distinct marker so the user can inspect raw traces before re-running with drop.
   d. Motion detection & correction
      - figures.carpet_plot()
      - GVTD timeseries (top panel, computed on OD)
      - bad segment zoom: top-N worst segments before/after correction
   e. Beer-Lambert / final HbO/HbR
      - figures.hbo_hbr_correlation(): per-channel HbO vs HbR scatter coloured by
          channel (short/long); fit line + r value; expected r < -0.5 for good signal.
      - mne.Report: HbO/HbR time series traces
      - figures.psd_plot(): PSD before/after bandpass (if applied),
          annotate cardiac (~1 Hz) and Mayer wave (~0.1 Hz) peaks
          + filter response curve (TODO)

3. Postprocessing reportlet (mode-dependent, appended after prep)
   glm mode:
      - Design matrix heatmap
      - Per-channel HRF with uncertainty + control condition
      - Contrast map (channel-level t-values, coloured probe layout)

4. Methods & provenance
   - boilerplate.generate_methods_text(): paste-ready Methods paragraph
   - Software versions table
   - Pipeline parameters used for this run (from sidecar JSON)

Planned additions
-----------------
Split raw checkpoint
"""

import base64
import csv
from contextlib import contextmanager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
import mne.io
from jinja2 import Environment, FileSystemLoader

from fnirs_pipe.qc.boilerplate import collect_software_versions, generate_methods_text
from fnirs_pipe.qc.figures import (
    motion_correction_panel,
    carpet_static_figure,
    bad_segment_zoom_figure,
    hbo_hbr_correlation_panel,
    psd_figure,
    quality_brain_views,
    optode_layout_static,
    short_channel_figure,
    design_matrix_static_figure,
    design_matrix_heatmap,
    build_epoch_preview_figure,
    build_ts_figure,
    build_layout_figure,
    build_sci_psp_figure,
    build_channel_figure,
    build_motion_detail_figure,
)
from fnirs_pipe.qc.quantitative_metrics import compute_iqm
from fnirs_pipe.utils.logging import get_logger

if TYPE_CHECKING:
    from fnirs_pipe.pipeline.prep_pipeline import PrepConfig

logger = get_logger("qc.report")
_TEMPLATE_DIR = Path(__file__).parent / "templates"
_PLOTLY_CDN   = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _pair_fname(pair: str) -> str:
    return pair.replace(" ", "_").replace("/", "-").replace("\\", "-")


def _save_multi_fig_html(figs: list, path: Path) -> int:
    """Stack multiple Plotly figures in one HTML file. Returns total height px."""
    head = (
        f'<meta charset="UTF-8">'
        f'<script src="{_PLOTLY_CDN}"></script>'
        f'<style>*{{box-sizing:border-box;}}body{{margin:0;padding:0;background:#fff;}}'
        f'.pfig{{margin-bottom:2px;}}</style>'
    )
    parts = [f"<!DOCTYPE html><html><head>{head}</head><body>"]
    total_h = 0
    for i, fig in enumerate(figs):
        if fig is None:
            continue
        h = _figure_height(fig)
        total_h += h + 2
        fig_html = fig.to_html(
            full_html=False, include_plotlyjs=False,
            div_id=f"pfig{i}", config={"responsive": True},
        )
        parts.append(f'<div class="pfig">{fig_html}</div>')
    parts.append("</body></html>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return max(total_h, 100)


# ---------------------------------------------------------------------------
# Error-handling helper
# ---------------------------------------------------------------------------

@contextmanager
def _guard(label: str, errors: list, subject: str):
    try:
        yield
    except Exception as e:
        errors.append(f"{label}: {e}")
        logger.exception("sub-%s | %s failed", subject, label)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _save_mpl_fig(fig, path: Path) -> None:
    import matplotlib.figure
    if not isinstance(fig, matplotlib.figure.Figure):
        if hasattr(fig, "canvas") and hasattr(fig.canvas, "figure"):
            fig = fig.canvas.figure
        elif hasattr(fig, "fig"):
            fig = fig.fig
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_b64_png(b64: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(b64))


_IFRAME_CSS = "<style>html,body{margin:0;padding:0;overflow:hidden;width:100%;height:100%;}</style>"


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


def _save_plotly_html(fig, path: Path, div_id: str | None = None) -> tuple[str, int]:
    """Save Plotly figure as standalone HTML. Returns (relative_path, height_px)."""
    h = _figure_height(fig)
    fig.update_layout(height=h)  # width stays auto (responsive)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"div_id": div_id} if div_id else {}
    html = fig.to_html(full_html=True, include_plotlyjs="cdn",
                       config={"responsive": True}, **kwargs)
    html = html.replace("<head>", f"<head>\n{_IFRAME_CSS}", 1)
    path.write_text(html, encoding="utf-8")
    return f"figures/{path.name}", h


def _plotly_to_html(fig, div_id: str | None = None) -> str:
    kwargs = {"div_id": div_id} if div_id else {}
    return fig.to_html(full_html=False, include_plotlyjs="cdn",
                       config={"responsive": True}, **kwargs)


# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------

def _prepare_long_raw(raw_intensity: mne.io.Raw, subject: str) -> mne.io.Raw:
    raw = raw_intensity.copy()
    picks = mne.pick_types(raw.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    long_picks = picks[dists > 0.01]
    if len(long_picks) == 0:
        logger.warning("sub-%s | no channels with dist > 1 cm; using all", subject)
        long_picks = picks
    raw.pick(long_picks)
    return raw


# ---------------------------------------------------------------------------
# Section builders — each returns a dict of template variables
# ---------------------------------------------------------------------------

def _section_od(raw_long: mne.io.Raw, subject: str, errors: list, figures_dir: Path) -> dict:
    return {}


def _section_sci(
    raw_intensity: mne.io.Raw,
    sci_scores: dict,
    bad_channels: list,
    config: Any,
    sci_scores_matrix: np.ndarray | None,
    sci_win_times: np.ndarray | None,
    psp_scores_matrix: np.ndarray | None,
    psp_win_times: np.ndarray | None,
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    ch_names = list(sci_scores.keys())
    sci_psp_panel_path = None
    sci_psp_panel_h = 0

    psp_per_ch: dict = {}
    if psp_scores_matrix is not None and psp_scores_matrix.shape[0] == len(ch_names):
        psp_per_ch = dict(zip(ch_names, psp_scores_matrix.mean(axis=1)))

    with _guard("SCI/PSP panel", errors, subject):
        fig = build_sci_psp_figure(
            sci_scores, psp_per_ch, set(bad_channels),
            sci_threshold=getattr(config, "sci_threshold", 0.75),
            sci_matrix=sci_scores_matrix,
            sci_win_times=sci_win_times,
            psp_matrix=psp_scores_matrix,
            psp_win_times=psp_win_times,
        )
        sci_psp_panel_path, sci_psp_panel_h = _save_plotly_html(
            fig, figures_dir / "sci_psp_panel.html"
        )

    return {"sci_psp_panel_path": sci_psp_panel_path, "sci_psp_panel_h": sci_psp_panel_h}


def _section_raw_viewer(
    raw_intensity: mne.io.Raw,
    bad_channels: list[str],
    sci_scores: dict[str, float],
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    ts_path = ts_h = None
    layout_2d_path = layout_3d_path = None
    layout_h = 500

    anns = raw_intensity.annotations
    markers = [
        {"onset": float(a["onset"]), "duration": float(a["duration"]),
         "description": str(a["description"])}
        for a in anns
    ]

    with _guard("Raw TS figure", errors, subject):
        fig, _, _, _, _, _ = build_ts_figure(
            raw_intensity, markers, set(bad_channels), 4000, 0.015,
        )
        ts_path, ts_h = _save_plotly_html(fig, figures_dir / "raw_ts.html")

    with _guard("Layout figures", errors, subject):
        fig_2d, fig_3d = build_layout_figure(
            raw_intensity, set(bad_channels), sci_scores, 0.015,
        )
        if fig_2d:
            fig_2d.update_layout(height=500)
            layout_2d_path, layout_h = _save_plotly_html(fig_2d, figures_dir / "layout_2d.html")
        if fig_3d:
            fig_3d.update_layout(height=500)
            layout_3d_path, _ = _save_plotly_html(fig_3d, figures_dir / "layout_3d.html")

    return {
        "raw_ts_path":     ts_path,        "raw_ts_h":     ts_h,
        "layout_2d_path":  layout_2d_path,
        "layout_3d_path":  layout_3d_path, "layout_h":     layout_h,
    }


def _section_channel_detail(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    max_pts: int = 4000,
) -> dict:
    anns = raw_haemo.annotations
    markers = [
        {"onset": float(a["onset"]), "duration": float(a["duration"]),
         "description": str(a["description"])}
        for a in anns
        if not str(a["description"]).upper().startswith("BAD")
    ]
    pairs = sorted(set(ch[:-4] for ch in raw_haemo.ch_names if ch.endswith(" hbo")))
    saved = []
    for pair in pairs:
        with _guard(f"Channel detail {pair}", errors, subject):
            detail_fig, psd_fig, epoch_fig = build_channel_figure(
                raw_haemo, markers, pair, max_pts, epoch_tmin, epoch_tmax,
            )
            fname = f"ch_detail_{_pair_fname(pair)}.html"
            h = _save_multi_fig_html([detail_fig, psd_fig, epoch_fig], figures_dir / fname)
            saved.append({"pair": pair, "path": f"figures/{fname}", "h": h})
    return {"channel_pairs": saved}


def _section_motion_detail(
    raw_long: mne.io.Raw,
    raw_before_motion: mne.io.Raw | None,
    subject: str,
    errors: list,
    figures_dir: Path,
    max_pts: int = 4000,
) -> dict:
    if raw_before_motion is None:
        return {"motion_detail_pairs": []}
    shared_chs = [c for c in raw_long.ch_names if c in raw_before_motion.ch_names]
    saved = []
    for ch in shared_chs:
        with _guard(f"Motion detail {ch}", errors, subject):
            fig = build_motion_detail_figure(raw_before_motion, raw_long, ch, max_pts)
            fname = f"motion_detail_{_pair_fname(ch)}.html"
            h = _save_multi_fig_html([fig], figures_dir / fname)
            saved.append({"pair": ch, "path": f"figures/{fname}", "h": h})
    return {"motion_detail_pairs": saved}


def _section_psd_detail(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    l_freq: float | None = None,
    h_freq: float | None = 0.4,
) -> dict:
    pairs = sorted(set(ch[:-4] for ch in raw_haemo.ch_names if ch.endswith(" hbo")))
    saved = []
    for pair in pairs:
        with _guard(f"PSD detail {pair}", errors, subject):
            picks = [c for c in (f"{pair} hbo", f"{pair} hbr") if c in raw_haemo.ch_names]
            if not picks:
                continue
            raw_sub = raw_haemo.copy().pick(picks)
            fig = psd_figure(raw_sub, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
                             title=f"PSD — {pair}")
            fname = f"psd_detail_{_pair_fname(pair)}.html"
            path, h = _save_plotly_html(fig, figures_dir / fname)
            saved.append({"pair": pair, "path": path, "h": h})
    return {"psd_detail_pairs": saved}


def _section_motion(
    raw_long: mne.io.Raw,
    sci_scores: dict,
    config: Any,
    segments: dict | None,
    subject: str,
    errors: list,
    figures_dir: Path,
    raw_before_motion: mne.io.Raw | None = None,
) -> dict:
    motion_panel_path = None
    motion_panel_h = 0
    bad_segment_zoom_path = None
    carpet_path = None

    with _guard("Carpet plot", errors, subject):
        b64 = carpet_static_figure(raw_long, ch_names=raw_long.ch_names, segments=segments)
        _save_b64_png(b64, figures_dir / "carpet.png")
        carpet_path = "figures/carpet.png"

    with _guard("Motion panel", errors, subject):
        sci_thr = getattr(config, "sci_threshold", 0.75)
        motion_colors = [
            "#3498db" if sci_scores.get(c, 0) >= sci_thr else "#e74c3c"
            for c in raw_long.ch_names
        ]
        fig_motion = motion_correction_panel(
            raw_long, raw_long.ch_names, motion_colors, segments or {},
        )
        motion_panel_path, motion_panel_h = _save_plotly_html(fig_motion, figures_dir / "motion_panel.html")

    with _guard("Bad segment zoom", errors, subject):
        all_spans = [
            (onset, dur)
            for spans in (segments or {}).values()
            for onset, dur in spans
        ]
        if all_spans:
            sorted_chs = sorted(sci_scores.keys(), key=lambda c: sci_scores.get(c, 0), reverse=True)
            rep_chs = [c for c in sorted_chs if c in raw_long.ch_names][:3]
            b64 = bad_segment_zoom_figure(
                raw_after=raw_long,
                bad_segments=all_spans,
                ch_names=rep_chs or raw_long.ch_names[:3],
                raw_before=raw_before_motion,
            )
            _save_b64_png(b64, figures_dir / "bad_segment_zoom.png")
            bad_segment_zoom_path = "figures/bad_segment_zoom.png"

    return {
        "motion_panel_path": motion_panel_path, "motion_panel_h": motion_panel_h,
        "bad_segment_zoom_path": bad_segment_zoom_path,
        "carpet_path": carpet_path,
    }


def _section_haemo(
    raw_haemo: mne.io.Raw,
    config: Any,
    subject: str,
    errors: list,
    figures_dir: Path,
    l_freq: float | None = None,
    h_freq: float | None = None,
) -> dict:
    hbo_hbr_path = psd_panel_path = None
    psd_panel_h = 0
    with _guard("HbO-HbR correlation panel", errors, subject):
        b64 = hbo_hbr_correlation_panel(raw_haemo)
        _save_b64_png(b64, figures_dir / "hbo_hbr_corr.png")
        hbo_hbr_path = "figures/hbo_hbr_corr.png"
    with _guard("PSD figure", errors, subject):
        fig_psd_custom = psd_figure(raw_haemo, l_freq=l_freq, h_freq=h_freq, fmax=2.0)
        psd_panel_path, psd_panel_h = _save_plotly_html(fig_psd_custom, figures_dir / "psd_panel.html")
    return {
        "hbo_hbr_path":   hbo_hbr_path,
        "psd_panel_path": psd_panel_path, "psd_panel_h": psd_panel_h,
    }


def _section_epoch_preview(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
) -> dict:
    epoch_preview_path = None
    epoch_preview_h = 0
    with _guard("Epoch preview", errors, subject):
        fig = build_epoch_preview_figure(raw_haemo, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)
        if fig is not None:
            epoch_preview_path, epoch_preview_h = _save_plotly_html(
                fig, figures_dir / "epoch_preview.html"
            )
    return {"epoch_preview_path": epoch_preview_path, "epoch_preview_h": epoch_preview_h}


def _scalars_to_toml(data: dict) -> str:
    lines = []
    for k, v in data.items():
        if v is None:
            lines.append(f"# {k} = null")
        elif isinstance(v, bool):
            lines.append(f"{k} = {str(v).lower()}")
        elif isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{k} = "{escaped}"')
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
    return "\n".join(lines) + "\n"


# def _save_iqm_toml(iqm: dict, subject: str, out_dir: Path) -> None:
#     scalars = {k: v for k, v in iqm.items() if not isinstance(v, (dict, list))}
#     out_path = out_dir / f"sub-{subject}_iqm.toml"
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(
#         _scalars_to_toml({"subject": subject, **scalars}), encoding="utf-8"
#     )
#     logger.info("sub-%s | IQM sidecar saved: %s", subject, out_path)


def _save_channel_csv(channel_rows: list, subject: str, out_dir: Path) -> None:
    if not channel_rows:
        return
    out_path = out_dir / f"sub-{subject}_channel_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "sci", "snr", "cv", "corr", "is_bad"]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(channel_rows)
    logger.info("sub-%s | channel metrics CSV saved: %s", subject, out_path)


def _section_iqm(
    raw_long: mne.io.Raw,
    raw_haemo: mne.io.Raw,
    sci_scores: dict,
    bad_channels: list,
    subject: str,
    errors: list,
    out_dir: Path | None = None,
) -> dict:
    iqm: dict = {}
    with _guard("IQM computation", errors, subject):
        iqm = compute_iqm(raw_long, raw_haemo, sci_scores, bad_channels)
    channel_rows = []
    import re as _re
    for ch in sci_scores:
        pair_key = _re.sub(r'\s+(\d+|hbo|hbr)$', '', ch, flags=_re.IGNORECASE)
        channel_rows.append({
            "name":   ch,
            "sci":    iqm.get("sci_per_channel", {}).get(ch),
            "snr":    iqm.get("snr_per_channel", {}).get(ch),
            "cv":     iqm.get("cv_per_channel", {}).get(ch),
            "corr":   iqm.get("hbo_hbr_corr_per_channel", {}).get(pair_key),
            "is_bad": ch in bad_channels,
        })
    if out_dir is not None and iqm:
        with _guard("Channel metrics CSV", errors, subject):
            _save_channel_csv(channel_rows, subject, out_dir)
    return {"iqm": iqm, "channel_rows": channel_rows}


def _section_brain(
    sci_scores: dict,
    bad_channels: list,
    coords_head: np.ndarray | None,
    good_mask: np.ndarray | None,
    raw_intensity: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    ch_names_brain: list[str] | None = None,
) -> dict:
    brain_views_path = None
    if coords_head is not None and good_mask is not None and sci_scores:
        with _guard("Brain views", errors, subject):
            import io as _io
            from PIL import Image as _PILImage

            ch_names = ch_names_brain if ch_names_brain is not None else list(sci_scores.keys())
            brain_b64 = quality_brain_views(ch_names, coords_head, good_mask, raw=raw_intensity)

            optode_b64 = None
            with _guard("Optode flat map", errors, subject):
                optode_b64 = optode_layout_static(raw_intensity, sci_scores, bad_channels)

            if brain_b64 and optode_b64:
                brain_pil = _PILImage.open(_io.BytesIO(base64.b64decode(brain_b64))).convert("RGBA")
                opt_pil   = _PILImage.open(_io.BytesIO(base64.b64decode(optode_b64))).convert("RGBA")
                h = brain_pil.height
                opt_w = int(opt_pil.width * h / opt_pil.height)
                opt_pil = opt_pil.resize((opt_w, h), _PILImage.LANCZOS)
                out = _PILImage.new("RGBA", (brain_pil.width + opt_w, h), (255, 255, 255, 255))
                out.paste(brain_pil, (0, 0))
                out.paste(opt_pil, (brain_pil.width, 0))
                buf = _io.BytesIO()
                out.convert("RGB").save(buf, format="PNG", optimize=True)
                buf.seek(0)
                combined_b64 = base64.b64encode(buf.read()).decode()
                _save_b64_png(combined_b64, figures_dir / "brain_views.png")
            elif brain_b64:
                _save_b64_png(brain_b64, figures_dir / "brain_views.png")
            else:
                raise RuntimeError("brain_b64 is None")

            brain_views_path = "figures/brain_views.png"
    return {"brain_views_path": brain_views_path}



def _section_short_channel(
    raw_intensity: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    short_channel_path = None
    with _guard("Short-channel figure", errors, subject):
        b64 = short_channel_figure(raw_intensity)
        if b64 is not None:
            _save_b64_png(b64, figures_dir / "short_channel_psd.png")
            short_channel_path = "figures/short_channel_psd.png"
    return {"short_channel_path": short_channel_path}


def _section_glm(
    design_matrix: "Any | None",
    glm_est: "Any | None",
    raw_haemo: "mne.io.Raw | None",
    subject: str,
    errors: list,
    figures_dir: Path,
    segments: "dict | None" = None,
) -> dict:
    glm_design_path = None
    glm_design_heatmap_path = None
    glm_activation_path = None

    conditions: list[str] = []
    if design_matrix is not None:
        design_matrix = design_matrix.rename(columns=str)
        conditions = [
            c for c in design_matrix.columns
            if not c.startswith(("drift_", "cosine_", "constant", "intercept", "short"))
        ]
        with _guard("GLM design matrix (timeseries)", errors, subject):
            b64 = design_matrix_static_figure(design_matrix, conditions, segments=segments)
            _save_b64_png(b64, figures_dir / "glm_design_timeseries.png")
            glm_design_path = "figures/glm_design_timeseries.png"
        with _guard("GLM design matrix (heatmap)", errors, subject):
            b64 = design_matrix_heatmap(design_matrix, conditions=conditions)
            _save_b64_png(b64, figures_dir / "glm_design_heatmap.png")
            glm_design_heatmap_path = "figures/glm_design_heatmap.png"

    if glm_est is not None and conditions and raw_haemo is not None:
        with _guard("GLM activation panel", errors, subject):
            from fnirs_pipe.qc.figures import activation_panel
            df = glm_est.to_dataframe().reset_index()
            if "Contrast" not in df.columns:
                for alt in ("contrast", "Regressor", "regressor", "condition", "Condition"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "Contrast"})
                        break
            if "Contrast" in df.columns:
                df["Contrast"] = df["Contrast"].astype(str)
                results_dict = {c: df[df["Contrast"] == c] for c in conditions}
                b64 = activation_panel(raw_haemo, results_dict)
                _save_b64_png(b64, figures_dir / "glm_activation.png")
                glm_activation_path = "figures/glm_activation.png"

    return {
        "glm_design_path": glm_design_path,
        "glm_design_heatmap_path": glm_design_heatmap_path,
        "glm_activation_path": glm_activation_path,
    }


def _glm_betas_table(df: "Any", conditions: list[str]) -> str:
    """Return an HTML table of per-channel GLM betas (theta)."""
    import html as _html

    df = df[df["Contrast"].isin(conditions)].copy()
    if df.empty:
        return None

    df["pair"] = df["ch_name"].str.replace(r"\s+(hbo|hbr)$", "", regex=True)
    df["chroma"] = df["ch_name"].str.extract(r"(hbo|hbr)$")[0]

    val_col = "theta" if "theta" in df.columns else (
              "Coef." if "Coef." in df.columns else None)
    if val_col is None:
        return None

    try:
        piv = df.pivot_table(index=["pair", "chroma"], columns="Contrast",
                             values=val_col, aggfunc="mean")
        piv = piv.reindex(columns=conditions)
    except Exception:
        return None

    th_cells = "".join(f"<th>{_html.escape(c)}</th>" for c in conditions)
    header = f"<tr><th>Channel</th><th>Type</th>{th_cells}</tr>"

    rows = []
    for (pair, chroma), row in piv.iterrows():
        td_vals = []
        for v in row:
            if v != v:  # NaN
                td_vals.append("<td>—</td>")
            else:
                cls = "pos" if v > 0 else "neg"
                td_vals.append(f'<td class="{cls}">{v:.3e}</td>')
        type_cls = "hbo" if chroma == "hbo" else "hbr"
        rows.append(
            f'<tr><td>{_html.escape(str(pair))}</td>'
            f'<td class="chroma-{type_cls}">{chroma}</td>'
            + "".join(td_vals) + "</tr>"
        )

    return (
        '<table class="glm-beta-table">'
        f"<thead>{header}</thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_subject_report(
    subject: str,
    raw_intensity: mne.io.Raw,
    raw_haemo: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    config: "PrepConfig",
    run_command: str,
    out_path: Path,
    sci_scores_matrix: np.ndarray | None = None,
    sci_win_times: np.ndarray | None = None,
    psp_scores_matrix: np.ndarray | None = None,
    psp_win_times: np.ndarray | None = None,
    motion_spans: list[tuple[float, float]] | None = None,
    segments: dict | None = None,
    coords_head: np.ndarray | None = None,
    good_mask: np.ndarray | None = None,
    ch_names_brain: list[str] | None = None,
    raw_before_motion: mne.io.Raw | None = None,
    design_matrix: "Any | None" = None,
    glm_est: "Any | None" = None,
    l_freq: float | None = None,
    h_freq: float | None = None,
) -> None:
    """Render a per-subject prep QC report and save as HTML."""
    errors: list[str] = []
    versions = collect_software_versions()
    # raw_long: long-channel-only copy used for OD/motion/IQM figures
    # raw_intensity: full original (all channels) passed to SCI/brain sections
    raw_long = _prepare_long_raw(raw_intensity, subject)

    figures_dir = out_path.parent / "figures"

    od_vars           = _section_od(raw_long, subject, errors, figures_dir)
    raw_viewer_vars   = _section_raw_viewer(raw_intensity, bad_channels, sci_scores,
                                            subject, errors, figures_dir)
    sci_vars          = _section_sci(
                            raw_intensity, sci_scores, bad_channels, config,
                            sci_scores_matrix, sci_win_times,
                            psp_scores_matrix, psp_win_times,
                            subject, errors, figures_dir)
    motion_vars       = _section_motion(
                            raw_long, sci_scores, config, segments, subject, errors,
                            figures_dir, raw_before_motion=raw_before_motion)
    motion_det_vars   = _section_motion_detail(raw_long, raw_before_motion, subject, errors, figures_dir)
    haemo_vars        = _section_haemo(raw_haemo, config, subject, errors, figures_dir,
                                       l_freq=l_freq, h_freq=h_freq)
    channel_det_vars  = _section_channel_detail(raw_haemo, subject, errors, figures_dir)
    psd_det_vars      = _section_psd_detail(raw_haemo, subject, errors, figures_dir,
                                            l_freq=l_freq, h_freq=h_freq)
    brain_vars        = _section_brain(
                            sci_scores, bad_channels, coords_head, good_mask, raw_intensity,
                            subject, errors, figures_dir, ch_names_brain=ch_names_brain)
    short_ch_vars     = _section_short_channel(raw_intensity, subject, errors, figures_dir)
    epoch_vars        = _section_epoch_preview(raw_haemo, subject, errors, figures_dir)
    glm_vars          = _section_glm(design_matrix, glm_est, raw_haemo, subject, errors, figures_dir, segments=segments)
    iqm_vars          = _section_iqm(raw_long, raw_haemo, sci_scores, bad_channels, subject, errors,
                                     out_dir=out_path.parent / "nirs")


    n_bad    = len(bad_channels)
    n_total  = len(sci_scores)
    bad_rate = 100 * n_bad / n_total if n_total > 0 else 0.0
    badge_class = (
        "badge-green"  if bad_rate < 10
        else "badge-yellow" if bad_rate < 30
        else "badge-red"
    )

    env      = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=False)
    template = env.get_template("subject_report.html.j2")
    html = template.render(
        subject=subject,
        run_date=date.today().isoformat(),
        run_command=run_command,
        versions=versions,
        n_bad=n_bad,
        n_total=n_total,
        bad_rate=bad_rate,
        badge_class=badge_class,
        bad_channels=bad_channels,
        sci_scores=sci_scores,
        config=config,
        errors=errors,
        methods=generate_methods_text(config, versions=versions),
        **od_vars,
        **raw_viewer_vars,
        **sci_vars,
        **motion_vars,
        **motion_det_vars,
        **haemo_vars,
        **channel_det_vars,
        **psd_det_vars,
        **iqm_vars,
        **brain_vars,
        **short_ch_vars,
        **epoch_vars,
        **glm_vars,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    logger.info("sub-%s | QC report saved: %s", subject, out_path)
    logger.info("sub-%s | figures saved: %s", subject, figures_dir)

    _build_mne_report(subject, raw_intensity, raw_haemo, out_path, errors)



def _build_mne_report(
    subject: str,
    raw_intensity: mne.io.Raw,
    raw_haemo: mne.io.Raw,
    out_path: Path,
    errors: list[str],
) -> None:
    try:
        report = mne.Report(title=f"sub-{subject} fNIRS QC", verbose=False)
        report.add_raw(raw_intensity, title="Raw intensity", psd=True, butterfly=False)
        report.add_raw(raw_haemo, title="HbO / HbR", psd=True, butterfly=False)
        mne_path = out_path.with_name(out_path.stem + "_mne.html")
        report.save(str(mne_path), overwrite=True, open_browser=False, verbose=False)
        logger.info("sub-%s | MNE report saved: %s", subject, mne_path)
    except Exception as e:
        errors.append(f"MNE report: {e}")
        logger.warning("sub-%s | MNE report failed: %s", subject, e)


