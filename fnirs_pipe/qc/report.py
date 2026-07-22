"""Generate per-subject HTML QC report using Jinja2 + Plotly.

Each section is an independent _section_*() builder that returns a dict of
template variables. Failures are caught by _guard() and appended to the errors
list — the rest of the report still renders.

Report sections
---------------
Summary
  Subject metadata, bad-channel badge, run command.

a. Raw Signal
  Per-channel HbO/HbR timeseries + PSD + epoch preview (dropdown selector).

b. Raw Signal Quality (SCI / PSP)
  Windowed SCI/PSP heatmap + lollipop summary (build_sci_psp_figure).
  Brain-surface quality map + optode flat map (if head coordinates available).

c. Motion Correction
  GVTD + carpet plot; bad-segment zoom; per-channel before/after OD traces.

d. HbO / HbR (Beer-Lambert)
  HbO–HbR correlation panel.

e. PSD (before / after bandpass)
  Full-dataset PSD panel + per-channel PSD detail (dropdown selector).

f. Epoch / HRF Preview
  Grand-mean HbO/HbR averaged across good channels, baseline-corrected.

Postprocessing (GLM mode)
  Design-matrix timeseries + heatmap; activation panel per condition.

Quantitative Metrics
  SQM scalar summary (channel retention, SCI, PSP, SNR, HbO–HbR corr, etc.)
  + per-channel table; CSV sidecar saved to nirs/ output directory.

Errors / Methods / Software Versions
"""

import base64
import csv
import re
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
from fnirs_pipe.qc.figure_io import (
    PLOTLY_CDN_URL, _IFRAME_CSS, _RESIZE_JS,
    _figure_height, _pair_fname, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.figures import (
    carpet_gvtd_figure,
    carpet_compare_figure,
    bad_segment_zoom_figure,
    hbo_hbr_correlation_panel,
    psd_figure,
    quality_brain_views,
    optode_layout_static,
    design_matrix_static_figure,
    design_matrix_heatmap,
    build_epoch_preview_figure,
    build_erpimage_figure,
    build_roi_erpimage_figure,
    build_sci_psp_figure,
    build_channel_figure,
    build_motion_detail_figure,
    channel_quality_heatmap,
    alff_falff_figure,
    fc_matrix_figure,
    fc_connectogram,
)
from fnirs_pipe.qc.quantitative_metrics import compute_sqm
from fnirs_pipe.utils.logging import get_logger

if TYPE_CHECKING:
    from fnirs_pipe.pipeline.prep_pipeline import PrepConfig

logger = get_logger("qc.report")
_TEMPLATE_DIR = Path(__file__).parent / "templates"


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


def _save_plotly_html(fig, path: Path, div_id: str | None = None) -> tuple[str, int]:
    """Save Plotly figure as standalone iframe-ready HTML. Returns (relative_path, height_px)."""
    h = _figure_height(fig)
    fig.update_layout(height=h)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"div_id": div_id} if div_id else {}
    html = fig.to_html(full_html=True, include_plotlyjs=False,
                       config={"responsive": True}, **kwargs)
    html = html.replace(
        "<head>",
        f'<head>\n<style>{_IFRAME_CSS}</style>\n<script src="{PLOTLY_CDN_URL}"></script>\n{_RESIZE_JS}',
        1,
    )
    path.write_text(html, encoding="utf-8")
    return f"figures/{path.name}", h


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


def _section_channel_detail(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    max_pts: int = 4000,
) -> dict:
    markers = extract_markers(raw_haemo)
    pairs = get_channel_pairs(raw_haemo)
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
    raw_od_before: mne.io.Raw | None,
    raw_od_after: mne.io.Raw | None,
    subject: str,
    errors: list,
    figures_dir: Path,
    segments: dict | None = None,
    max_pts: int = 4000,
    corrected_segments: list | None = None,
    spike_segments: list | None = None,
) -> dict:
    if raw_od_before is None or raw_od_after is None:
        return {"motion_detail_pairs": []}
    shared_chs = [c for c in raw_od_after.ch_names if c in raw_od_before.ch_names]
    saved = []
    for ch in shared_chs:
        with _guard(f"Motion detail {ch}", errors, subject):
            fig = build_motion_detail_figure(raw_od_before, raw_od_after, ch, segments, max_pts,
                                             corrected_segments=corrected_segments,
                                             spike_segments=spike_segments)
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
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
) -> dict:
    pairs = get_channel_pairs(raw_haemo)
    saved = []
    for pair in pairs:
        with _guard(f"PSD detail {pair}", errors, subject):
            picks = [c for c in (f"{pair} hbo", f"{pair} hbr") if c in raw_haemo.ch_names]
            if not picks:
                continue
            raw_sub = raw_haemo.copy().pick(picks)
            fig = psd_figure(raw_sub, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
                             title=f"PSD — {pair}", cardiac=cardiac, resp=resp)
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
    raw_after_motion: mne.io.Raw | None = None,
) -> dict:
    carpet_gvtd_path = None
    bad_segment_zoom_path = None

    corrected_segments = None
    motion_corr_sqm: dict = {}
    if raw_before_motion is not None and raw_after_motion is not None:
        with _guard("Motion-correction footprint", errors, subject):
            from fnirs_pipe.qc.quantitative_metrics import (
                motion_corrected_segments, motion_correction_metrics,
            )
            corrected_segments = motion_corrected_segments(raw_before_motion, raw_after_motion)
            motion_corr_sqm = {
                k: v for k, v in motion_correction_metrics(raw_before_motion, raw_after_motion).items()
                if not isinstance(v, dict)  # scalars only for the metrics panel
            }

    spike_spans = None
    with _guard("Spike segments", errors, subject):
        from fnirs_pipe.qc.quantitative_metrics import spike_segments
        spike_spans = spike_segments(raw_long)

    with _guard("Carpet + GVTD", errors, subject):
        b64 = carpet_gvtd_figure(raw_long, raw_long.ch_names, segments,
                                 corrected_segments=corrected_segments,
                                 spike_segments=spike_spans)
        _save_b64_png(b64, figures_dir / "carpet_gvtd.png")
        carpet_gvtd_path = "figures/carpet_gvtd.png"

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
        "carpet_gvtd_path": carpet_gvtd_path,
        "bad_segment_zoom_path": bad_segment_zoom_path,
        "motion_corrected_sqm": motion_corr_sqm,
        "corrected_segments": corrected_segments,
        "spike_spans": spike_spans,
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
        fig_psd_custom = psd_figure(
            raw_haemo, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
            resp=(config.resp_l_freq, config.resp_h_freq))
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


def _section_erpimage(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    roi_map: dict | None = None,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
) -> dict:
    # raw_haemo here is the denoised (bandpassed, pre-regression) signal, not preproc.
    # only for task data with (non-BAD) events; skip early otherwise
    if not any(not str(a["description"]).upper().startswith("BAD") for a in raw_haemo.annotations):
        return {"erpimage_pairs": [], "erpimage_roi_pairs": []}

    roi_saved = []
    for roi_name, chans in (roi_map or {}).items():
        with _guard(f"erpimage ROI {roi_name}", errors, subject):
            fig = build_roi_erpimage_figure(raw_haemo, str(roi_name), chans, epoch_tmin, epoch_tmax)
            if fig is not None:
                fname = f"erpimage_roi_{_pair_fname(str(roi_name))}.html"
                h = _save_multi_fig_html([fig], figures_dir / fname)
                roi_saved.append({"pair": str(roi_name), "path": f"figures/{fname}", "h": h})

    saved = []
    for ch in [c for c in raw_haemo.ch_names if c.endswith(" hbo")]:
        with _guard(f"erpimage {ch}", errors, subject):
            fig = build_erpimage_figure(raw_haemo, ch, epoch_tmin, epoch_tmax)
            if fig is not None:
                fname = f"erpimage_{_pair_fname(ch)}.html"
                h = _save_multi_fig_html([fig], figures_dir / fname)
                saved.append({"pair": ch, "path": f"figures/{fname}", "h": h})
    return {"erpimage_pairs": saved, "erpimage_roi_pairs": roi_saved}


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


# def _save_sqm_toml(sqm: dict, subject: str, out_dir: Path) -> None:
#     scalars = {k: v for k, v in sqm.items() if not isinstance(v, (dict, list))}
#     out_path = out_dir / f"sub-{subject}_sqm.toml"
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(
#         _scalars_to_toml({"subject": subject, **scalars}), encoding="utf-8"
#     )
#     logger.info("sub-%s | SQM sidecar saved: %s", subject, out_path)


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


def _section_sqm(
    raw_long: mne.io.Raw,
    raw_haemo: mne.io.Raw,
    sci_scores: dict,
    bad_channels: list,
    subject: str,
    errors: list,
    out_dir: Path | None = None,
    *,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
) -> dict:
    sqm: dict = {}
    with _guard("SQM computation", errors, subject):
        sqm = compute_sqm(raw_long, raw_haemo, sci_scores, bad_channels,
                          cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq)
    channel_rows = []
    for ch in sci_scores:
        pair_key = re.sub(r'\s+(\d+|hbo|hbr)$', '', ch, flags=re.IGNORECASE)
        channel_rows.append({
            "name":   ch,
            "sci":    sqm.get("sci_per_channel", {}).get(ch),
            "snr":    sqm.get("snr_per_channel", {}).get(ch),
            "cv":     sqm.get("cv_per_channel", {}).get(ch),
            "corr":   sqm.get("hbo_hbr_corr_per_channel", {}).get(pair_key),
            "is_bad": ch in bad_channels,
        })
    if out_dir is not None and sqm:
        with _guard("Channel metrics CSV", errors, subject):
            _save_channel_csv(channel_rows, subject, out_dir)
    return {"sqm": sqm, "channel_rows": channel_rows}


def _section_channel_summary(
    channel_rows: list,
    sqm: dict,
    subject: str,
    errors: list,
    figures_dir: Path,
    sci_thresh: float = 0.75,
) -> dict:
    path, h = None, 0
    with _guard("Channel quality summary", errors, subject):
        ch_names = [r["name"] for r in channel_rows]
        is_bad   = [r["is_bad"] for r in channel_rows]
        sci_pc   = {r["name"]: r["sci"] for r in channel_rows if r["sci"] is not None}
        cv_pc    = {r["name"]: r["cv"]  for r in channel_rows if r["cv"]  is not None}
        snr_pc   = {r["name"]: r["snr"] for r in channel_rows if r["snr"] is not None}
        psp_pc   = sqm.get("psp_per_channel", {})
        fig = channel_quality_heatmap(
            ch_names, is_bad, sci_pc, cv_pc, snr_pc, psp_pc,
            sci_thresh=sci_thresh,
        )
        path, h = _save_plotly_html(fig, figures_dir / "channel_summary.html")
    return {"channel_summary_path": path, "channel_summary_h": h}


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

    durbin_watson_mean = None
    if glm_est is not None:
        with _guard("Durbin-Watson", errors, subject):
            import numpy as np

            from fnirs_pipe.qc.quantitative_metrics import compute_glm_sqm
            resid = np.array(
                [glm_est.data[ch].residuals for ch in glm_est.ch_names]
            ).squeeze(-1)
            durbin_watson_mean = compute_glm_sqm(resid).get("durbin_watson_mean")

    return {
        "glm_design_path": glm_design_path,
        "glm_design_heatmap_path": glm_design_heatmap_path,
        "glm_activation_path": glm_activation_path,
        "durbin_watson_mean": durbin_watson_mean,
    }


def _section_rest(
    alff_df: "Any | None",
    fc_df: "Any | None",
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    alff_path = fc_path = fc_circle_path = None
    with _guard("ALFF/fALFF figure", errors, subject):
        if alff_df is not None:
            b64 = alff_falff_figure(alff_df)
            _save_b64_png(b64, figures_dir / "rest_alff.png")
            alff_path = "figures/rest_alff.png"
    with _guard("FC matrix figure", errors, subject):
        if fc_df is not None:
            b64 = fc_matrix_figure(fc_df)
            _save_b64_png(b64, figures_dir / "rest_fc.png")
            fc_path = "figures/rest_fc.png"
    with _guard("FC connectogram", errors, subject):
        if fc_df is not None:
            b64 = fc_connectogram(fc_df)
            _save_b64_png(b64, figures_dir / "rest_fc_circle.png")
            fc_circle_path = "figures/rest_fc_circle.png"
    return {"rest_alff_path": alff_path, "rest_fc_path": fc_path, "rest_fc_circle_path": fc_circle_path}


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
    raw_after_motion: mne.io.Raw | None = None,
    design_matrix: "Any | None" = None,
    glm_est: "Any | None" = None,
    l_freq: float | None = None,
    h_freq: float | None = None,
    mode: str | None = None,
    alff_df: "Any | None" = None,
    fc_df: "Any | None" = None,
    after_haemo: mne.io.Raw | None = None,
    gcor_reg: dict | None = None,
    roi_map: dict | None = None,
) -> None:
    """Render a per-subject prep QC report and save as HTML."""
    errors: list[str] = []
    versions = collect_software_versions()
    # raw_long: long-channel-only copy used for OD/motion/SQM figures
    # raw_intensity: full original (all channels) passed to SCI/brain sections
    raw_long = _prepare_long_raw(raw_intensity, subject)

    figures_dir = out_path.parent / "figures"

    sci_vars          = _section_sci(
                            raw_intensity, sci_scores, bad_channels, config,
                            sci_scores_matrix, sci_win_times,
                            psp_scores_matrix, psp_win_times,
                            subject, errors, figures_dir)
    motion_vars       = _section_motion(
                            raw_long, sci_scores, config, segments, subject, errors,
                            figures_dir, raw_before_motion=raw_before_motion,
                            raw_after_motion=raw_after_motion)
    motion_det_vars   = _section_motion_detail(
                            raw_before_motion, raw_after_motion, subject, errors, figures_dir,
                            segments=segments,
                            corrected_segments=motion_vars.get("corrected_segments"),
                            spike_segments=motion_vars.get("spike_spans"))
    haemo_vars        = _section_haemo(raw_haemo, config, subject, errors, figures_dir,
                                       l_freq=l_freq, h_freq=h_freq)
    denoise_carpet_path = None
    if after_haemo is not None:
        with _guard("Denoising carpet", errors, subject):
            b64 = carpet_compare_figure(raw_haemo, after_haemo, roi_map=roi_map)
            _save_b64_png(b64, figures_dir / "denoise_carpet.png")
            denoise_carpet_path = "figures/denoise_carpet.png"
    channel_det_vars  = _section_channel_detail(raw_haemo, subject, errors, figures_dir)
    psd_det_vars      = _section_psd_detail(raw_haemo, subject, errors, figures_dir,
                                            l_freq=l_freq, h_freq=h_freq,
                                            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
                                            resp=(config.resp_l_freq, config.resp_h_freq))
    brain_vars        = _section_brain(
                            sci_scores, bad_channels, coords_head, good_mask, raw_intensity,
                            subject, errors, figures_dir, ch_names_brain=ch_names_brain)
    epoch_vars        = _section_epoch_preview(raw_haemo, subject, errors, figures_dir)
    # erpimage on the denoised (bandpassed, pre-regression) haemo so drift/noise is gone and the
    # task response is intact; fall back to preproc only if no post-processing ran.
    erpimage_vars     = _section_erpimage(after_haemo if after_haemo is not None else raw_haemo,
                                          subject, errors, figures_dir, roi_map=roi_map)
    glm_vars          = _section_glm(design_matrix, glm_est, raw_haemo, subject, errors, figures_dir, segments=segments)
    rest_vars         = _section_rest(alff_df, fc_df, subject, errors, figures_dir)
    sqm_vars          = _section_sqm(raw_long, raw_haemo, sci_scores, bad_channels, subject, errors,
                                     out_dir=out_path.parent / "nirs",
                                     cardiac_l_freq=config.cardiac_l_freq,
                                     cardiac_h_freq=config.cardiac_h_freq,
                                     resp_l_freq=config.resp_l_freq,
                                     resp_h_freq=config.resp_h_freq)
    # GCOR before→after the short-channel regression (fNIRS GSR analog): the meaningful
    # comparison (expected to drop). Bandpass alone raises GCOR, so we do not compare that.
    if gcor_reg and sqm_vars.get("sqm") is not None:
        sqm_vars["sqm"].update(gcor_reg)
    # motion-correction footprint scalars (computed in _section_motion) into the metrics panel
    if sqm_vars.get("sqm") is not None:
        sqm_vars["sqm"].update(motion_vars.get("motion_corrected_sqm") or {})
    ch_summary_vars   = _section_channel_summary(
                            sqm_vars["channel_rows"], sqm_vars["sqm"], subject, errors, figures_dir,
                            sci_thresh=getattr(config, "sci_threshold", 0.75))

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

        **sci_vars,
        **motion_vars,
        **motion_det_vars,
        **erpimage_vars,
        **haemo_vars,
        **channel_det_vars,
        **psd_det_vars,
        **sqm_vars,
        **brain_vars,

        **epoch_vars,
        **glm_vars,
        **rest_vars,
        **ch_summary_vars,
        denoise_carpet_path=denoise_carpet_path,
        mode=mode or "",
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


