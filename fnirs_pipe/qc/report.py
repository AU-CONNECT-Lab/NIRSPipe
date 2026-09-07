"""Generate per-subject HTML QC report using Jinja2 + Plotly.

Each section is an independent _section_*() builder that returns a dict of
template variables. Failures are caught by _guard() and appended to the errors
list — the rest of the report still renders. A section skipped because the run does
not carry what it needs is not a failure: it goes to the notes list instead.

Report sections
---------------
::

  Summary
    Subject metadata, bad-channel badge, run command.

  a. Raw Signal
    Per-channel HbO/HbR timeseries + PSD + epoch preview (dropdown selector).

  b. Raw Signal Quality (SCI / PSP)
    Windowed SCI/PSP heatmap + lollipop summary (build_sci_psp_figure), all on the
    uncorrected optical density. Per-channel SCI/PSP across the motion correction.
    Brain-surface quality map + optode flat map (if head coordinates available).

  c. Motion Correction
    GVTD + carpet plot; bad-segment zoom; per-channel before/after OD traces.

  d. HbO / HbR (Beer-Lambert)
    HbO–HbR correlation panel before and after denoising, one small panel per metric
    across the haemoglobin stages, and HbO–HbR r per channel over the same stages.

  e. PSD by stage
    Full-dataset PSD panel + per-channel PSD detail (dropdown selector), one line
    per stage file on disk.

  f. Epoch / HRF Preview
    Event timeline (one row per condition), then the grand-mean HbO/HbR averaged across
    good channels, baseline-corrected.

  Postprocessing (GLM mode)
    Design-matrix timeseries + heatmap; activation panel per condition.

  Quantitative Metrics
    SQM scalar summary (channel retention, SCI, PSP, SNR, HbO–HbR corr, etc.)
    + per-channel table; CSV sidecar saved to nirs/ output directory.

  Per-trial Quality
    Every trial window scored on its own, over the epoch window, on the intensity
    recording.

  Errors / Methods / Software Versions
"""

import base64
import json
from contextlib import contextmanager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
import mne.io

from fnirs_pipe.qc.boilerplate import collect_software_versions, generate_methods_text
from fnirs_pipe.qc.channel_table import (
    OD_SPLIT_COLUMNS, channel_rows, format_rows, heatmap_args, save_channel_csv,
    separation_blocks, separation_notes,
)
from fnirs_pipe.qc.figure_io import (
    PLOTLY_CDN_URL, _IFRAME_CSS, _RESIZE_JS,
    _figure_height, _pair_fname, _save_multi_fig_html,
    extract_markers, get_channel_pairs,
)
from fnirs_pipe.qc.metrics import SCI_PASS, gvtd_channel_picks
from fnirs_pipe.qc.figures import (
    build_trigger_timeline_single,
    condition_colors,
    trial_quality_heatmap,
    carpet_gvtd_figure,
    carpet_compare_figure,
    bad_segment_zoom_figure,
    hbo_hbr_correlation_panel,
    psd_figure,
    quality_brain_views,
    optode_layout_static,
    evoked_topomap_static,
    design_matrix_static_figure,
    design_matrix_heatmap,
    build_epoch_preview_figure,
    build_trial_image_figure,
    build_roi_trial_image_figure,
    build_sci_psp_figure,
    denoise_stage_panels,
    stage_metrics_figure,
    build_channel_figure,
    build_motion_detail_figure,
    channel_quality_heatmap,
    alff_falff_figure,
    alff_topo_figure,
    fc_matrix_figure,
    fc_roi_matrix_figure,
    fc_seed_topo_figure,
    fc_connectogram,
)
from fnirs_pipe.qc.report_shell import (
    footer_vars, guard, note, page_vars, render, stylesheet,
)
from fnirs_pipe.qc.sqm_record import record_path as _sqm_record_path
from fnirs_pipe.qc.trial_qc import score_trials
from fnirs_pipe.utils.logging import get_logger

if TYPE_CHECKING:
    from fnirs_pipe.pipeline.prep_pipeline import PrepConfig

logger = get_logger("qc.report")


# ---------------------------------------------------------------------------
# Error-handling helper
# ---------------------------------------------------------------------------
#
# Thin wrappers over the shared recorders, kept because every section in this file passes
# the bare subject id rather than the scope string the shell logs under.

@contextmanager
def _guard(label: str, errors: list, subject: str):
    with guard(label, errors, f"sub-{subject}"):
        yield


def _note(notes: list, subject: str, message: str) -> None:
    note(notes, f"sub-{subject}", message)


# ---- Epoching gate ----

_EPOCH_TMIN, _EPOCH_TMAX = -5.0, 25.0


def _no_epoch_reason(raw_haemo: mne.io.Raw) -> "str | None":
    """Why nothing can be epoched over the report's window, or None when something can.

    A block design that only marks where each condition starts and ends carries annotations
    but no trials: both windows run off an edge of the run. Asking first is what keeps the
    epoch sections from rendering empty and MNE from warning once per figure.
    """
    from fnirs_pipe.qc.figures._utils import epochable_events

    events, _ = epochable_events(raw_haemo, _EPOCH_TMIN, _EPOCH_TMAX)
    if len(events) > 0:
        return None
    n_marks = sum(1 for a in raw_haemo.annotations
                  if not str(a["description"]).upper().startswith("BAD"))
    if not n_marks:
        return "the run carries no events"
    return (f"the run carries {n_marks} marker(s), none of them leaving a full "
            f"{_EPOCH_TMIN:g} to {_EPOCH_TMAX:g} s window inside the recording")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _fig_href(figures_dir: Path, name: str) -> str:
    """URL of a figure as the report must link to it, the report sitting above ``figures/``.

    figures/            + carpet_gvtd.html -> "figures/carpet_gvtd.html"
    figures/sub-01_task-rest/ + same       -> "figures/sub-01_task-rest/carpet_gvtd.html"

    Per-run reports put their figures in a subdirectory so several runs of one subject stop
    overwriting each other; a caller that passes a bare ``figures/`` still gets the old URL.
    """
    if figures_dir.parent.name == "figures":
        return f"figures/{figures_dir.name}/{name}"
    return f"figures/{name}"


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
    return _fig_href(path.parent, path.name), h


# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------

def _prepare_long_raw(raw_intensity: mne.io.Raw, subject: str) -> mne.io.Raw:
    from fnirs_pipe.qc.metrics import long_short_channels

    raw = raw_intensity.copy()
    long_names, _ = long_short_channels(raw)
    if not long_names:
        logger.warning("sub-%s | no long channels by separation; using all", subject)
        return raw
    raw.pick(long_names)
    return raw


# ---------------------------------------------------------------------------
# Section builders — each returns a dict of template variables
# ---------------------------------------------------------------------------

def _section_sci(
    raw_intensity: mne.io.Raw,
    sci_scores: dict,
    bad_channels: list,
    config: Any,
    windowed: dict | None,
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    """The SCI/PSP panel, per channel and per window.

    ``windowed`` is the record's section of that name; the series are read from it rather
    than recomputed, so the panel and the stored numbers cannot disagree. An absent section
    leaves the per-window half of the panel out and the per-channel half intact.

    Every view in the panel is the uncorrected optical density. SCI and PSP measure optode
    coupling, which is a property of how the cap sat rather than of anything the pipeline
    does, so the stage that answers "was this channel worth keeping" is the one before the
    correction.
    """
    def _series(key: str) -> "np.ndarray | None":
        value = (windowed or {}).get(key)
        return None if value is None else np.asarray(value)

    sci_scores_matrix = _series("sci_matrix")
    sci_win_times     = _series("sci_times")
    psp_scores_matrix = _series("psp_matrix")
    psp_win_times     = _series("psp_times")
    ch_names = list(sci_scores.keys())
    sci_psp_panel_path = None
    sci_psp_panel_h = 0

    psp_per_ch: dict = {}
    if psp_scores_matrix is not None and psp_scores_matrix.shape[0] == len(ch_names):
        psp_per_ch = dict(zip(ch_names, psp_scores_matrix.mean(axis=1)))

    with _guard("SCI/PSP panel", errors, subject):
        fig = build_sci_psp_figure(
            sci_scores, psp_per_ch, set(bad_channels),
            sci_threshold=getattr(config, "sci_threshold", SCI_PASS),
            sci_matrix=sci_scores_matrix,
            sci_win_times=sci_win_times,
            psp_matrix=psp_scores_matrix,
            psp_win_times=psp_win_times,
        )
        sci_psp_panel_path, sci_psp_panel_h = _save_plotly_html(
            fig, figures_dir / "sci_psp_panel.html"
        )

    return {"sci_psp_panel_path": sci_psp_panel_path, "sci_psp_panel_h": sci_psp_panel_h}


def _uncorrected_haemo(
    raw_od_before: "mne.io.Raw | None",
    config: Any,
    subject: str,
    errors: list,
) -> "mne.io.Raw | None":
    """Beer-Lambert on desc-sci, the optical density before the motion step.

    The section this feeds is labelled raw, and desc-preproc is not: it sits one step after
    the motion correction. There is no uncorrected haemoglobin file on disk (the pipeline
    converts once, after correcting), so it is derived here from the OD that is, using the
    run's own DPF. Returning None leaves the caller on desc-preproc.
    """
    if raw_od_before is None:
        return None
    with _guard("Uncorrected haemo", errors, subject):
        from mne.preprocessing.nirs import beer_lambert_law
        dpf = list(getattr(config, "dpf", []) or [])
        if not dpf:
            return None
        return beer_lambert_law(raw_od_before.copy(), ppf=dpf[0] if len(dpf) == 1 else dpf)


def _section_channel_detail(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    max_pts: int = 4000,
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
) -> dict:
    markers = extract_markers(raw_haemo)
    pairs = get_channel_pairs(raw_haemo)
    saved = []
    for pair in pairs:
        with _guard(f"Channel detail {pair}", errors, subject):
            detail_fig, psd_fig, epoch_fig = build_channel_figure(
                raw_haemo, markers, pair, max_pts, epoch_tmin, epoch_tmax,
                cardiac=cardiac, resp=resp,
            )
            fname = f"ch_detail_{_pair_fname(pair)}.html"
            h = _save_multi_fig_html([detail_fig, psd_fig, epoch_fig], figures_dir / fname)
            saved.append({"pair": pair, "path": _fig_href(figures_dir, fname), "h": h})
    return {"channel_pairs": saved}


def _section_motion_detail(
    raw_od_before: mne.io.Raw | None,
    raw_od_after: mne.io.Raw | None,
    subject: str,
    errors: list,
    figures_dir: Path,
    segments: dict | None = None,
    corrected_segments: list | None = None,
    spike_segments: list | None = None,
) -> dict:
    if raw_od_before is None or raw_od_after is None:
        return {"motion_detail_pairs": []}
    shared_chs = [c for c in raw_od_after.ch_names if c in raw_od_before.ch_names]
    saved = []
    for ch in shared_chs:
        with _guard(f"Motion detail {ch}", errors, subject):
            fig = build_motion_detail_figure(raw_od_before, raw_od_after, ch, segments,
                                             corrected_segments=corrected_segments,
                                             spike_segments=spike_segments)
            fname = f"motion_detail_{_pair_fname(ch)}.html"
            h = _save_multi_fig_html([fig], figures_dir / fname)
            saved.append({"pair": ch, "path": _fig_href(figures_dir, fname), "h": h})
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
    psd_stages: "list[tuple[str, mne.io.Raw]] | None" = None,
) -> dict:
    pairs = get_channel_pairs(raw_haemo)
    saved = []
    for pair in pairs:
        with _guard(f"PSD detail {pair}", errors, subject):
            picks = [c for c in (f"{pair} hbo", f"{pair} hbr") if c in raw_haemo.ch_names]
            if not picks:
                continue
            raw_sub = raw_haemo.copy().pick(picks)
            # a later stage may have dropped the pair (bad channel), so each stage is
            # narrowed to whatever it still carries and skipped when that is nothing
            stages_sub = None
            if psd_stages is not None:
                stages_sub = [
                    (label, raw.copy().pick(present))
                    for label, raw in psd_stages
                    if (present := [c for c in picks if c in raw.ch_names])
                ]
            fig = psd_figure(raw_sub, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
                             title=f"PSD — {pair}", cardiac=cardiac, resp=resp,
                             stages=stages_sub)
            fname = f"psd_detail_{_pair_fname(pair)}.html"
            path, h = _save_plotly_html(fig, figures_dir / fname)
            saved.append({"pair": pair, "path": path, "h": h})
    return {"psd_detail_pairs": saved}


def _section_motion(
    raw_long: mne.io.Raw,
    raw_gvtd: mne.io.Raw,
    gvtd_set: str,
    sci_scores: dict,
    config: Any,
    segments: dict | None,
    subject: str,
    errors: list,
    figures_dir: Path,
    windowed: dict | None = None,
    raw_before_motion: mne.io.Raw | None = None,
    raw_after_motion: mne.io.Raw | None = None,
) -> dict:
    """The carpet and GVTD panel, with the flagged spans drawn over it.

    Both span lists come from the record's ``windowed`` section rather than being detected
    here. They were measured on the same channel set this panel draws, so reading them back
    is not a shortcut: it is what keeps the stripes on the carpet and the counts in the
    metrics table describing one event each.

    ``raw_after_motion`` puts the corrected trace and carpet in the same panel as the
    uncorrected one, matching the ``before -> after`` pairs in the metrics table.

    ``raw_gvtd`` is the channel set the GVTD panel covers, which ``--gvtd-channels`` decides
    and which is not always ``raw_long``. Everything else here stays on the long channels:
    the zoom below is per-channel optical density, so a wider set would only add rows.
    """
    carpet_gvtd_path = None
    carpet_gvtd_h = 600
    bad_segment_zoom_path = None

    def _spans(key: str) -> "list | None":
        value = (windowed or {}).get(key)
        return [tuple(span) for span in value] if value else None

    corrected_segments = _spans("motion_corrected_spans_s")
    spike_spans = _spans("spike_spans_s")

    with _guard("Carpet + GVTD", errors, subject):
        fig = carpet_gvtd_figure(raw_gvtd, raw_gvtd.ch_names, segments,
                                 corrected_segments=corrected_segments,
                                 spike_segments=spike_spans,
                                 raw_after=raw_after_motion,
                                 channel_set=gvtd_set)
        carpet_gvtd_path, carpet_gvtd_h = _save_plotly_html(
            fig, figures_dir / "carpet_gvtd.html")

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
            bad_segment_zoom_path = _fig_href(figures_dir, "bad_segment_zoom.png")

    return {
        "carpet_gvtd_path": carpet_gvtd_path,
        "carpet_gvtd_h": carpet_gvtd_h,
        "bad_segment_zoom_path": bad_segment_zoom_path,
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
    raw_errts: mne.io.Raw | None = None,
    psd_stages: "list[tuple[str, mne.io.Raw]] | None" = None,
    record: dict | None = None,
) -> dict:
    """Beer-Lambert output and what the denoising did to it.

    The correlation panels are drawn from the recordings, the strips beside them from the
    record: a figure needs the samples, a number must not be computed twice. ``raw_errts``
    is the confound-regression residual, the stage the denoising before/after is taken
    against; it is absent on a prep-only run and on a denoise run with nothing to regress,
    and then only the before panel is drawn.
    """
    hbo_hbr_path = hbo_hbr_after_path = psd_panel_path = None
    psd_panel_h = 0
    with _guard("HbO-HbR correlation panel", errors, subject):
        b64 = hbo_hbr_correlation_panel(
            raw_haemo, title="HbO–HbR Signal Quality — desc-preproc (before denoising)")
        _save_b64_png(b64, figures_dir / "hbo_hbr_corr.png")
        hbo_hbr_path = _fig_href(figures_dir, "hbo_hbr_corr.png")
    if raw_errts is not None:
        with _guard("HbO-HbR correlation panel (after)", errors, subject):
            b64 = hbo_hbr_correlation_panel(
                raw_errts, title="HbO–HbR Signal Quality — desc-errts (after denoising)")
            _save_b64_png(b64, figures_dir / "hbo_hbr_corr_after.png")
            hbo_hbr_after_path = _fig_href(figures_dir, "hbo_hbr_corr_after.png")

    # Recomputed rather than read from the record: the record measures each stage on the
    # signal as it stands there, which cannot be compared across the bandpass. See
    # comparable_stage_metrics.
    record = record or {}
    stage_metrics_path = None
    stage_metrics_h = 0
    stage_banded = False
    stages = [("desc-preproc", raw_haemo)]
    stages += [(label, raw) for label, raw in (psd_stages or [])
               if label == "desc-filtered" and "hbo" in raw.get_channel_types()]
    if raw_errts is not None:
        stages.append(("desc-errts", raw_errts))

    if len(stages) > 1:
        from fnirs_pipe.qc.metrics import (
            comparable_stage_metrics, long_short_channels,
        )

        def _long_only(raw: mne.io.Raw) -> mne.io.Raw:
            """The stage measured where the verdict lives, or unchanged with nothing to drop.

            Long channels only, so these panels answer the question the metrics table above
            them answers. A short channel is a regressor rather than a measurement, and an
            average over both moved the correlation that reads as the verdict: on a montage
            with eight short channels it sat at -0.18 where the long channels alone gave -0.50.
            """
            names, _ = long_short_channels(raw)
            if not names or len(names) == len(raw.ch_names):
                return raw
            return raw.copy().pick(names)

        with _guard("Denoising stage metrics", errors, subject):
            stage_metrics = comparable_stage_metrics(
                [(label, _long_only(raw)) for label, raw in stages], l_freq, h_freq,
                config.cardiac_l_freq, config.cardiac_h_freq,
                config.resp_l_freq, config.resp_h_freq)
            stage_banded = bool(stage_metrics["banded"])
            fig = stage_metrics_figure(stage_metrics["labels"],
                                       denoise_stage_panels(stage_metrics))
            if fig is not None:
                stage_metrics_path, stage_metrics_h = _save_plotly_html(
                    fig, figures_dir / "denoise_stage_metrics.html")

    with _guard("PSD figure", errors, subject):
        fig_psd_custom = psd_figure(
            raw_haemo, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
            resp=(config.resp_l_freq, config.resp_h_freq),
            stages=psd_stages)
        psd_panel_path, psd_panel_h = _save_plotly_html(fig_psd_custom, figures_dir / "psd_panel.html")
    return {
        "hbo_hbr_path":   hbo_hbr_path,
        "hbo_hbr_after_path": hbo_hbr_after_path,
        "stage_metrics_path": stage_metrics_path, "stage_metrics_h": stage_metrics_h,
        "stage_banded": stage_banded,
        "psd_panel_path": psd_panel_path, "psd_panel_h": psd_panel_h,
        "psd_stage_labels": [label for label, _ in (psd_stages or [])],
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


def _section_trigger_timeline(
    raw: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    """Every event on one time axis, one row per condition.

    The epoch figures below average trials together, which is what hides a condition that
    stopped being delivered halfway through or an experimenter who started a block twice.
    This is the same panel the raw QC viewer opens with, so a run inspected before the
    pipeline and after it is read off one picture of its design.
    """
    path, h = None, 0
    with _guard("Trigger timeline", errors, subject):
        markers = extract_markers(raw)
        fig = build_trigger_timeline_single(markers, condition_colors(markers))
        if fig is not None:
            path, h = _save_plotly_html(fig, figures_dir / "trigger_timeline.html")
    return {"trigger_timeline_path": path, "trigger_timeline_h": h}


def _section_trial_qc(
    raw_intensity: mne.io.Raw,
    config: Any,
    subject: str,
    errors: list,
    figures_dir: Path,
) -> dict:
    """Each trial window scored on its own, so one bad trial is visible before averaging.

    Scored over the same window the epoch figures average, ``_EPOCH_TMIN`` to
    ``_EPOCH_TMAX``, and on the intensity recording, so a trial's SCI and SNR are on the
    scale the metrics table prints rather than on the haemoglobin one. The scoring is shared
    with ``fnirs-qc raw``, which is where this panel came from.
    """
    path, h = None, 0
    with _guard("Per-trial quality", errors, subject):
        labels, sqms = score_trials(
            raw_intensity, extract_markers(raw_intensity),
            getattr(config, "sci_threshold", SCI_PASS),
            config.cardiac_l_freq, config.cardiac_h_freq,
            _EPOCH_TMIN, _EPOCH_TMAX,
        )
        fig = trial_quality_heatmap(labels, sqms)
        if fig is not None:
            path, h = _save_plotly_html(fig, figures_dir / "trial_qc.html")
    return {"trial_qc_path": path, "trial_qc_h": h}


def _section_evoked_topomap(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
) -> dict:
    path = None
    with _guard("Evoked topomap", errors, subject):
        b64 = evoked_topomap_static(raw_haemo, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)
        if b64:
            _save_b64_png(b64, figures_dir / "evoked_topomap.png")
            path = _fig_href(figures_dir, "evoked_topomap.png")
    return {"evoked_topomap_path": path}


def _section_trial_image(
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
        return {"trial_image_pairs": [], "trial_image_roi_pairs": []}

    roi_saved = []
    for roi_name, chans in (roi_map or {}).items():
        with _guard(f"trial image ROI {roi_name}", errors, subject):
            figs = build_roi_trial_image_figure(raw_haemo, str(roi_name), chans, epoch_tmin, epoch_tmax)
            if figs:
                fname = f"trialimage_roi_{_pair_fname(str(roi_name))}.html"
                h = _save_multi_fig_html(figs, figures_dir / fname)
                roi_saved.append({"pair": str(roi_name), "path": _fig_href(figures_dir, fname), "h": h})

    saved = []
    # HbO only: single-trial HbR is too low-amplitude to read as an image, and the HbO/HbR
    # relation is already reported by hbo_hbr_corr and the per-channel detail figure
    for ch in [c for c in raw_haemo.ch_names if c.endswith(" hbo")]:
        with _guard(f"trial image {ch}", errors, subject):
            figs = build_trial_image_figure(raw_haemo, ch, epoch_tmin, epoch_tmax)
            if figs:
                fname = f"trialimage_{_pair_fname(ch)}.html"
                h = _save_multi_fig_html(figs, figures_dir / fname)
                saved.append({"pair": ch, "path": _fig_href(figures_dir, fname), "h": h})
    return {"trial_image_pairs": saved, "trial_image_roi_pairs": roi_saved}


def _load_record(
    out_dir: Path | None,
    sqm_label: str | None,
    subject: str,
    errors: list,
) -> dict:
    """The run's SQM record, empty when there is none to read.

    Read once here rather than inside each section that wants a slice of it. Separate from
    :func:`_section_sqm`, which keeps its own read and its own validation because it is the
    one that must report a missing record as an error; the before/after strips fed from
    here simply do not appear when their numbers are absent.
    """
    if out_dir is None or sqm_label is None:
        return {}
    # its own label, so a missing record is not reported twice over: _section_sqm reads the
    # same file again and is the one that must say the metrics table has nothing to show
    with _guard("Reading the SQM record", errors, subject):
        return json.loads(_sqm_record_path(out_dir, sqm_label).read_text(encoding="utf-8"))
    return {}


def _record_pair(record: dict, before: str, after: str, key: str) -> "tuple[dict, dict]":
    """One per-channel metric at two sections of the record, as ``(before, after)``.

    ``_split_scalars`` files every per-channel dict under ``per_channel``, so that is where
    these live::

        _record_pair(rec, "preproc", "errts", "cnr_per_channel")
        -> (rec["per_channel"]["preproc"]["cnr_per_channel"], ... same for "errts")

    Either half is ``{}`` when that section or that key is absent, which is what a run
    without the second stage looks like; the caller draws nothing rather than half a figure.
    """
    per_channel = record.get("per_channel") or {}
    return ((per_channel.get(before) or {}).get(key) or {},
            (per_channel.get(after) or {}).get(key) or {})


def _load_stage_raw(
    out_dir: Path | None,
    sqm_label: str | None,
    desc: str,
    subject: str,
    errors: list,
) -> "mne.io.Raw | None":
    """One of the run's stage files, read back off disk.

    The optical density either side of the motion step used to travel here in memory on
    ``PrepResult``, two full recordings held for the length of a run to draw two figures.
    They are on disk as ``desc-sci`` and ``desc-motcorrected``, which is where the quality
    record reads them from, so the report reads the same files rather than a copy.
    """
    if out_dir is None or sqm_label is None:
        return None
    with _guard(f"Reading desc-{desc}", errors, subject):
        from fnirs_pipe.io.snirf import read_snirf
        from fnirs_pipe.qc.sqm_record import scan_runs
        path = (scan_runs(out_dir).get(sqm_label) or {}).get(desc)
        return None if path is None else read_snirf(path)
    return None


def _section_sqm(
    sci_scores: dict,
    bad_channels: list,
    subject: str,
    errors: list,
    out_dir: Path | None = None,
    *,
    sqm_label: str | None = None,
    gvtd_channels: str = "long",
    sci_threshold: float = SCI_PASS,
) -> dict:
    """Read this run's SQM record; the report displays, it does not compute.

    The panel judges data quality, so its scalars come from the long-channel sections;
    ``raw`` stands in when the montage has no short channels to exclude. Nothing
    recomputes here: a missing record is reported as an error rather than silently
    measured a second time, and so is a record that carries none of the sections the panel
    reads, which is what a foreign file at this path looks like.

    The per-channel table is the one place short channels appear, read from ``raw_short``.
    They are pruned against the same SCI threshold as everything else, so their status is
    a real verdict with a downstream cost -- a bad short channel is a bad regressor -- and
    printing that verdict without the score behind it leaves it uncheckable. Assembling
    those rows is :mod:`fnirs_pipe.qc.channel_table`, which the raw views share, so the
    three per-channel tables in the package read one record the same way.

    Every family is read at its long-channel split where the record carries one, so the
    panel's verdict is never an average over long and short channels together. ``preproc``
    is merged underneath ``preproc_long`` rather than replaced by it, because the split
    sections deliberately omit the metrics that describe the recording's duration instead
    of its channels.

    The motion-corrected side of the same channel set is added under a ``_post`` suffix, and
    the confound-regression residual under an ``_errts`` one. Both carry the same key names
    as the section they are paired with, so a plain merge would silently overwrite the
    before values; the suffix is what lets the template print ``before -> after``.
    """
    sqm: dict = {}
    sqm_all: dict = {}
    sqm_long: dict = {}
    sqm_short: dict = {}
    hb_all: dict = {}
    hb_long: dict = {}
    hb_short: dict = {}
    record_read: dict = {}
    with _guard("SQM record", errors, subject):
        if out_dir is None or sqm_label is None:
            raise FileNotFoundError("no SQM record location for this run")
        record_file = _sqm_record_path(out_dir, sqm_label)
        record = record_read = json.loads(record_file.read_text(encoding="utf-8"))
        per_channel = record.get("per_channel") or {}
        raw_key = "raw_long" if "raw_long" in record else "raw"
        keys = (raw_key, "motion", "preproc")
        if not any(record.get(k) for k in keys):
            raise ValueError(
                f"{record_file.name} holds none of {keys}; not a sectioned SQM record")
        # `preproc` before `preproc_long`, so the long values win where they exist and the
        # whole-file ones the split does not carry (pct_data_retained) survive underneath
        # `censor` last and unsuffixed: its keys are all gvtd_censor_* so nothing collides
        for key in (*keys, "preproc_long", "censor"):
            sqm.update(record.get(key) or {})
            sqm.update(per_channel.get(key) or {})
        # The corrected side of the *same* channel set, suffixed rather than merged: it
        # carries the same key names as `raw_key` by design, so a plain update would
        # silently replace the pre-correction values with the post ones. No fallback to a
        # different split if this one is missing -- a long-channel GVTD against an
        # all-channel one would read as an effect of the correction.
        post_key = "motion_post_long" if raw_key == "raw_long" else "motion_post"
        for k, v in (record.get(post_key) or {}).items():
            sqm[f"{k}_post"] = v
        # The denoised side of the haemoglobin metrics. `errts` rather than `filtered`: the
        # bandpass alone moves gcor and the band powers for reasons that are the filter's,
        # not the recording's, while the regression is the step whose effect is worth a
        # number. A run that regressed nothing has no `errts` and prints single values.
        errts_key = "errts_long" if record.get("errts_long") else "errts"
        for k, v in (record.get(errts_key) or {}).items():
            sqm[f"{k}_errts"] = v
        # `--gvtd-channels all` moves the GVTD scalars alone, not the panel around them: it
        # names one metric, and SCI and PSP measure coupling per channel rather than across
        # channels, so widening their set would answer a different question than was asked.
        # The corrected side moves with it, since a post value read off a different channel
        # set than its pre value makes the correction look like an effect it is not.
        gvtd_key = "raw" if gvtd_channels == "all" else raw_key
        if gvtd_key != raw_key:
            for src, suffix in ((record.get(gvtd_key), ""),
                                (record.get("motion_post"), "_post")):
                for k, v in (src or {}).items():
                    if k.startswith("gvtd_"):
                        sqm[f"{k}{suffix}"] = v
        # The same raw file measured over three channel sets, kept as three dicts so the
        # panel can print them side by side. `sqm` above already carries one of them and
        # decides the verdict; these are for the comparison, not for it.
        sqm_all = dict(record.get("raw") or {})
        sqm_long = dict(record.get("raw_long") or {})
        sqm_short = dict(record.get("raw_short") or {})
        # the same three views of the haemoglobin file, for the second half of the table
        hb_all = dict(record.get("preproc") or {})
        hb_long = dict(record.get("preproc_long") or {})
        hb_short = dict(record.get("preproc_short") or {})
        # how the montage split, which lives in `raw` whichever section the scalars came from
        for key in ("n_long_channels", "n_short_channels"):
            if sqm_all.get(key) is not None:
                sqm[key] = sqm_all[key]

    rows = channel_rows(record_read, sci_scores, bad_channels)
    if out_dir is not None and sqm:
        with _guard("Channel metrics CSV", errors, subject):
            save_channel_csv(rows, sqm_label or f"sub-{subject}", out_dir)
    # the raw rows stay for the CSV and the quality grid, which want the numbers; the
    # template gets them formatted, so the per-channel table prints the same widths and the
    # same SCI verdict as the raw viewer and the GUI
    cells = format_rows(rows, sci_threshold)
    return {
        "sqm": sqm,
        "channel_rows": rows,
        "channel_cells": cells,
        "channel_blocks": separation_blocks(cells),
        "sqm_all": sqm_all,
        "sqm_long": sqm_long,
        "sqm_short": sqm_short,
        "hb_all": hb_all,
        "hb_long": hb_long,
        "hb_short": hb_short,
        # three columns are worth printing only when all three are real
        "sqm_split": bool(sqm_long and sqm_short),
        "hb_split": bool(hb_long and hb_short),
    }


def _note_separation(
    notes: list,
    subject: str,
    sqm: dict,
    rows: list,
    short_channel_requested: bool = False,
) -> None:
    """File the montage-split warnings as run notes, one note each.

    The wording is shared with the raw views; what differs is where it goes. Here it joins
    the report's notes list and the run log, so a reader who never opens the per-channel
    table still learns the split did not come out the way the metrics assume.
    """
    for message in separation_notes(sqm, rows, short_channel_requested):
        _note(notes, subject, message)


def _section_channel_summary(
    rows: list,
    subject: str,
    errors: list,
    figures_dir: Path,
    sci_thresh: float = SCI_PASS,
) -> dict:
    path, h = None, 0
    with _guard("Channel quality summary", errors, subject):
        fig = channel_quality_heatmap(sci_thresh=sci_thresh, **heatmap_args(rows))
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
            brain_b64 = quality_brain_views(ch_names, coords_head, good_mask,
                                            raw=raw_intensity, sci_scores=sci_scores)

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

            brain_views_path = _fig_href(figures_dir, "brain_views.png")
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
            glm_design_path = _fig_href(figures_dir, "glm_design_timeseries.png")
        with _guard("GLM design matrix (heatmap)", errors, subject):
            b64 = design_matrix_heatmap(design_matrix, conditions=conditions)
            _save_b64_png(b64, figures_dir / "glm_design_heatmap.png")
            glm_design_heatmap_path = _fig_href(figures_dir, "glm_design_heatmap.png")

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
                glm_activation_path = _fig_href(figures_dir, "glm_activation.png")

    return {
        "glm_design_path": glm_design_path,
        "glm_design_heatmap_path": glm_design_heatmap_path,
        "glm_activation_path": glm_activation_path,
    }


def _section_rest(
    alff_df: "Any | None",
    fc_df: "Any | None",
    subject: str,
    errors: list,
    figures_dir: Path,
    fc_hbr_df: "Any | None" = None,
    fc_seed: dict | None = None,
    fc_roi: dict | None = None,
    raw_haemo: "mne.io.Raw | None" = None,
) -> dict:
    alff_path = alff_topo_path = fc_path = fc_roi_path = fc_circle_path = fc_seed_path = None
    with _guard("ALFF/fALFF figure", errors, subject):
        if alff_df is not None:
            b64 = alff_falff_figure(alff_df)
            _save_b64_png(b64, figures_dir / "rest_alff.png")
            alff_path = _fig_href(figures_dir, "rest_alff.png")
    with _guard("ALFF topography", errors, subject):
        if alff_df is not None and raw_haemo is not None:
            b64 = alff_topo_figure(raw_haemo, alff_df)
            if b64 is not None:   # None means the montage carries no optode positions
                _save_b64_png(b64, figures_dir / "rest_alff_topo.png")
                alff_topo_path = _fig_href(figures_dir, "rest_alff_topo.png")
    with _guard("FC matrix figure", errors, subject):
        if fc_df is not None:
            b64 = fc_matrix_figure(fc_df, fc_hbr_df)
            _save_b64_png(b64, figures_dir / "rest_fc.png")
            fc_path = _fig_href(figures_dir, "rest_fc.png")
    with _guard("ROI FC matrix", errors, subject):
        if fc_roi:
            b64 = fc_roi_matrix_figure(fc_roi)
            if b64 is not None:
                _save_b64_png(b64, figures_dir / "rest_fc_roi.png")
                fc_roi_path = _fig_href(figures_dir, "rest_fc_roi.png")
    with _guard("FC connectogram", errors, subject):
        if fc_df is not None:
            b64 = fc_connectogram(fc_df, fc_hbr_df)
            _save_b64_png(b64, figures_dir / "rest_fc_circle.png")
            fc_circle_path = _fig_href(figures_dir, "rest_fc_circle.png")
    with _guard("FC seed topography", errors, subject):
        if fc_seed and raw_haemo is not None:
            b64 = fc_seed_topo_figure(raw_haemo, fc_seed.get("hbo"), fc_seed.get("hbr"))
            if b64 is not None:   # None means the montage carries no optode positions
                _save_b64_png(b64, figures_dir / "rest_fc_seed.png")
                fc_seed_path = _fig_href(figures_dir, "rest_fc_seed.png")
    return {"rest_alff_path": alff_path, "rest_alff_topo_path": alff_topo_path,
            "rest_fc_path": fc_path, "rest_fc_roi_path": fc_roi_path,
            "rest_fc_circle_path": fc_circle_path, "rest_fc_seed_path": fc_seed_path}


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
    motion_spans: list[tuple[float, float]] | None = None,
    segments: dict | None = None,
    coords_head: np.ndarray | None = None,
    good_mask: np.ndarray | None = None,
    ch_names_brain: list[str] | None = None,
    design_matrix: "Any | None" = None,
    glm_est: "Any | None" = None,
    l_freq: float | None = None,
    h_freq: float | None = None,
    mode: str | None = None,
    alff_df: "Any | None" = None,
    fc_df: "Any | None" = None,
    fc_hbr_df: "Any | None" = None,
    fc_seed: dict | None = None,
    fc_roi: dict | None = None,
    after_haemo: mne.io.Raw | None = None,
    gcor_reg: dict | None = None,
    roi_map: dict | None = None,
    provenance_path: str | None = None,
    sqm_label: str | None = None,
) -> list[str]:
    """Render the QC report for one run and save as HTML. Returns its run-level notes.

    sqm_label is the run this report covers, as a BIDS stem (``sub-01_task-rest``). It picks
    the SQM record and the intermediate stage files off disk, and it gives the run its own
    ``figures/<sqm_label>/`` directory so several runs of one subject stop overwriting each
    other's figures. Passing None keeps the flat ``figures/`` layout.

    provenance_path is the already-rendered flow diagram, relative to out_path
    (the caller renders it: the report embeds, it does not draw).
    """
    errors: list[str] = []
    notes: list[str] = []
    versions = collect_software_versions()
    # raw_long: long-channel-only copy used for OD/motion/SQM figures
    # raw_intensity: full original (all channels) passed to SCI/brain sections
    raw_long = _prepare_long_raw(raw_intensity, subject)
    # the GVTD panel's channel set, which config decides and which need not be raw_long
    gvtd_channels = getattr(config, "gvtd_channels", None) or "long"
    gvtd_picks, gvtd_set = gvtd_channel_picks(raw_intensity, gvtd_channels)
    raw_gvtd = raw_intensity.copy().pick(gvtd_picks)

    figures_dir = out_path.parent / "figures"
    if sqm_label:
        figures_dir = figures_dir / sqm_label

    nirs_dir = out_path.parent / "nirs"
    record            = _load_record(nirs_dir, sqm_label, subject, errors)
    windowed_section  = record.get("windowed") or None
    raw_before_motion = _load_stage_raw(nirs_dir, sqm_label, "sci", subject, errors)
    raw_after_motion  = _load_stage_raw(nirs_dir, sqm_label, "motcorrected", subject, errors)
    raw_errts         = _load_stage_raw(nirs_dir, sqm_label, "errts", subject, errors)
    # the haemo chain as it exists on disk, in the order it was written. The PSD figure used
    # to re-filter `raw_haemo` in memory to invent its "after" row, which showed the filter
    # rather than the run; a stage missing here simply does not get a line. It stops at the
    # bandpass: desc-errts is the confound regression's output, and what that step did is
    # not a spectral question.
    psd_stages        = [
        (f"desc-{desc}", raw)
        for desc in ("filtered", "resampled")
        if (raw := _load_stage_raw(nirs_dir, sqm_label, desc, subject, errors)) is not None
    ] or None
    sci_vars          = _section_sci(
                            raw_intensity, sci_scores, bad_channels, config,
                            windowed_section, subject, errors, figures_dir)
    motion_vars       = _section_motion(
                            raw_long, raw_gvtd, gvtd_set,
                            sci_scores, config, segments, subject, errors,
                            figures_dir, windowed=windowed_section,
                            raw_before_motion=raw_before_motion,
                            raw_after_motion=raw_after_motion)
    motion_det_vars   = _section_motion_detail(
                            raw_before_motion, raw_after_motion, subject, errors, figures_dir,
                            segments=segments,
                            corrected_segments=motion_vars.get("corrected_segments"),
                            spike_segments=motion_vars.get("spike_spans"))
    haemo_vars        = _section_haemo(raw_haemo, config, subject, errors, figures_dir,
                                       l_freq=l_freq, h_freq=h_freq,
                                       raw_errts=raw_errts, psd_stages=psd_stages,
                                       record=record)
    denoise_carpet_path = None
    if after_haemo is not None:
        with _guard("Denoising carpet", errors, subject):
            b64 = carpet_compare_figure(raw_haemo, after_haemo, roi_map=roi_map)
            _save_b64_png(b64, figures_dir / "denoise_carpet.png")
            denoise_carpet_path = _fig_href(figures_dir, "denoise_carpet.png")
    # the "Raw Signal" section is the recording before anything was done to it, so its
    # figures come off desc-sci rather than the corrected desc-preproc the rest of the
    # report is built on. Same stage as the SCI/PSP windows and the SNR/CV numbers below.
    raw_haemo_uncorr  = _uncorrected_haemo(raw_before_motion, config, subject, errors)
    channel_det_vars  = _section_channel_detail(
                            raw_haemo_uncorr if raw_haemo_uncorr is not None else raw_haemo,
                            subject, errors, figures_dir,
                            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
                            resp=(config.resp_l_freq, config.resp_h_freq))
    channel_det_vars["channel_detail_stage"] = (
        "desc-sci" if raw_haemo_uncorr is not None else "desc-preproc")
    psd_det_vars      = _section_psd_detail(raw_haemo, subject, errors, figures_dir,
                                            l_freq=l_freq, h_freq=h_freq,
                                            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
                                            resp=(config.resp_l_freq, config.resp_h_freq),
                                            psd_stages=psd_stages)
    brain_vars        = _section_brain(
                            sci_scores, bad_channels, coords_head, good_mask, raw_intensity,
                            subject, errors, figures_dir, ch_names_brain=ch_names_brain)
    # trial image and topomap on the denoised (bandpassed, pre-regression) haemo so drift/noise
    # is gone and the task response is intact; fall back to preproc only if no post-processing ran.
    epoch_haemo       = after_haemo if after_haemo is not None else raw_haemo
    epoch_skip        = _no_epoch_reason(raw_haemo)
    if epoch_skip is not None:
        _note(notes, subject,
              f"Epoch preview, evoked topomap, trial images and per-trial quality were "
              f"skipped because {epoch_skip}. The per-channel and layout figures show the "
              f"continuous signal instead.")
        epoch_vars       = {"epoch_preview_path": None, "epoch_preview_h": 0}
        trial_image_vars = {"trial_image_pairs": [], "trial_image_roi_pairs": []}
        topomap_vars     = {"evoked_topomap_path": None}
        trial_qc_vars    = {"trial_qc_path": None, "trial_qc_h": 0}
    else:
        epoch_vars        = _section_epoch_preview(raw_haemo, subject, errors, figures_dir)
        trial_image_vars  = _section_trial_image(epoch_haemo, subject, errors, figures_dir,
                                                 roi_map=roi_map)
        topomap_vars      = _section_evoked_topomap(epoch_haemo, subject, errors, figures_dir)
        trial_qc_vars     = _section_trial_qc(raw_intensity, config, subject, errors,
                                              figures_dir)
    glm_vars          = _section_glm(design_matrix, glm_est, raw_haemo, subject, errors, figures_dir, segments=segments)
    rest_vars         = _section_rest(alff_df, fc_df, subject, errors, figures_dir, fc_hbr_df=fc_hbr_df,
                                      fc_seed=fc_seed, fc_roi=fc_roi, raw_haemo=raw_haemo)
    sqm_vars          = _section_sqm(sci_scores, bad_channels, subject, errors,
                                     out_dir=out_path.parent / "nirs",
                                     sqm_label=sqm_label,
                                     gvtd_channels=gvtd_channels,
                                     sci_threshold=getattr(config, "sci_threshold", SCI_PASS))
    _note_separation(notes, subject, sqm_vars["sqm"], sqm_vars["channel_rows"],
                     short_channel_requested=bool(getattr(config, "short_channel", None)))
    # GCOR before→after the short-channel regression (fNIRS GSR analog): the meaningful
    # comparison (expected to drop). Bandpass alone raises GCOR, so we do not compare that.
    if gcor_reg and sqm_vars.get("sqm") is not None:
        sqm_vars["sqm"].update(gcor_reg)
    trigger_vars      = _section_trigger_timeline(raw_intensity, subject, errors, figures_dir)
    ch_summary_vars   = _section_channel_summary(
                            sqm_vars["channel_rows"], subject, errors, figures_dir,
                            sci_thresh=getattr(config, "sci_threshold", SCI_PASS))

    n_bad    = len(bad_channels)
    n_total  = len(sci_scores)
    bad_rate = 100 * n_bad / n_total if n_total > 0 else 0.0
    badge_class = (
        "badge-green"  if bad_rate < 10
        else "badge-yellow" if bad_rate < 30
        else "badge-red"
    )

    from fnirs_pipe.qc.boilerplate.vocabulary import (
        format_metric, is_key_metric, metric_class, metric_summary,
    )

    from fnirs_pipe.qc.sqm_record import entities_of

    run_label_text = sqm_label or f"sub-{subject}"
    html = render(
        "subject_report.html.j2",
        **page_vars(
            title=f"fnirs-pipe QC \u2014 sub-{subject}",
            heading=f"fnirs-pipe QC Report \u2014 {run_label_text}",
            css=stylesheet("subject.css"),
        ),
        **footer_vars(
            scope=f"sub-{subject}", errors=errors, notes=notes,
            nirs_dir=out_path.parent / "nirs", mode=mode, label=sqm_label,
            provenance_path=provenance_path,
            methods=generate_methods_text(config, versions=versions, mode=mode,
                                          nirs_dir=out_path.parent / "nirs"),
            versions=versions,
        ),
        metric_summary=metric_summary,
        is_key_metric=is_key_metric,
        format_metric=format_metric,
        metric_class=metric_class,
        od_split_columns=OD_SPLIT_COLUMNS,
        subject=subject,
        run_label=sqm_label,
        run_entities={k: v for k, v in entities_of(sqm_label or "").items() if v},
        index_href=(f"sub-{subject}_qc.html" if sqm_label else None),
        run_command=run_command,
        n_bad=n_bad,
        n_total=n_total,
        bad_rate=bad_rate,
        badge_class=badge_class,
        bad_channels=bad_channels,
        sci_scores=sci_scores,
        config=config,

        **sci_vars,
        **motion_vars,
        **motion_det_vars,
        **trial_image_vars,
        **topomap_vars,
        **haemo_vars,
        **channel_det_vars,
        **psd_det_vars,
        **sqm_vars,
        **brain_vars,

        **epoch_vars,
        **trigger_vars,
        **trial_qc_vars,
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
    return notes



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


