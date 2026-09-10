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
from fnirs_pipe.qc.metrics import SCI_PASS, gvtd_channel_blocks, separation_bands
from fnirs_pipe.qc.metrics._helpers import bands_from_record
from fnirs_pipe.qc.figures._utils import chunk_annotations
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


def _epoch_window_mismatch(raw_haemo: mne.io.Raw, epoch_tmax: float) -> "float | None":
    """The events' own length when it outruns the window the figures average over.

    The figures average trials together, so they need one window for all of them and cannot
    follow each event's own duration the way the per-trial scoring can. That makes the
    fallback a guess, and on a design whose blocks are minutes long it is a wrong one: the
    epoch figures describe the first ``epoch_tmax`` seconds of a block and nothing says so.
    Returns the median duration when it exceeds the window, else None.

    A 240 s conversation block against the 25 s fallback -> 240.0; a 5 s trial -> None.
    """
    durations = [float(a["duration"]) for a in raw_haemo.annotations
                 if not str(a["description"]).upper().startswith("BAD")
                 and float(a["duration"]) > 0]
    if not durations:
        return None
    median = float(np.median(durations))
    return median if median > epoch_tmax else None


def _no_epoch_reason(
    raw_haemo: mne.io.Raw,
    epoch_tmin: float = _EPOCH_TMIN,
    epoch_tmax: float = _EPOCH_TMAX,
) -> "str | None":
    """Why nothing can be epoched over the report's window, or None when something can.

    A block design that only marks where each condition starts and ends carries annotations
    but no trials: both windows run off an edge of the run. Asking first is what keeps the
    epoch sections from rendering empty and MNE from warning once per figure. The window is
    the run's own, so the reason it prints names the window that was actually asked for.
    """
    from fnirs_pipe.qc.figures._utils import epochable_events

    events, _ = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
    if len(events) > 0:
        return None
    n_marks = sum(1 for a in raw_haemo.annotations
                  if not str(a["description"]).upper().startswith("BAD"))
    if not n_marks:
        return "the run carries no events"
    return (f"the run carries {n_marks} marker(s), none of them leaving a full "
            f"{epoch_tmin:g} to {epoch_tmax:g} s window inside the recording")


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

def _prepare_long_raw(
    raw_intensity: mne.io.Raw, subject: str, sep_bands=None,
) -> mne.io.Raw:
    from fnirs_pipe.qc.metrics import long_short_channels

    raw = raw_intensity.copy()
    long_names, _ = long_short_channels(raw, sep_bands)
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
    suffix: str = "",
) -> dict:
    """The SCI/PSP panel, per channel and per window.

    ``suffix`` names the figure, so a per-condition page writes its own instead of
    overwriting the run's. Handed a ``windowed`` whose matrices are already sliced to one
    condition, this panel is that condition's: nothing inside it filters or re-measures,
    which is why a real slice works here where the carpet has to be narrowed instead.

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
            fig, figures_dir / f"sci_psp_panel{suffix}.html"
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
    suffix: str = "",
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
            fname = f"ch_detail_{_pair_fname(pair)}{suffix}.html"
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
    spike_by_set: dict | None = None,
    gvtd_blocks: "list[tuple[str, list[str]]] | None" = None,
) -> dict:
    """One per-channel motion figure per channel, each with its own class's GVTD on top.

    The GVTD row follows the channel the figure is about rather than staying fixed: it is
    read against the derivative row directly below it, and the two separation classes are not
    on one scale. A channel in neither class falls back to the canonical set, which is the
    only one there is a reported number for.
    """
    if raw_od_before is None or raw_od_after is None:
        return {"motion_detail_pairs": []}
    shared_chs = [c for c in raw_od_after.ch_names if c in raw_od_before.ch_names]

    # the same blocks the carpet panel drew, so the two figures never name sets differently:
    # a montage with no long channels has one block and every channel lands in it
    blocks = gvtd_blocks or [("all", list(raw_od_before.ch_names))]
    members = [(name, names, set(names)) for name, names in blocks]

    def set_of(ch: str) -> "tuple[str, list[str]]":
        for name, names, lookup in members:
            if ch in lookup:
                return name, names
        return blocks[0]          # in no block (neither separation range): the canonical set

    saved = []
    for ch in shared_chs:
        with _guard(f"Motion detail {ch}", errors, subject):
            set_name, picks = set_of(ch)
            fig = build_motion_detail_figure(raw_od_before, raw_od_after, ch, segments,
                                             corrected_segments=corrected_segments,
                                             spike_segments=(spike_by_set or {}).get(set_name),
                                             gvtd_picks=picks, gvtd_set=set_name)
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
    suffix: str = "",
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
            fname = f"psd_detail_{_pair_fname(pair)}{suffix}.html"
            path, h = _save_plotly_html(fig, figures_dir / fname)
            saved.append({"pair": pair, "path": path, "h": h})
    return {"psd_detail_pairs": saved}


def _section_motion(
    raw_long: mne.io.Raw,
    raw_gvtd: mne.io.Raw,
    gvtd_set: str,
    gvtd_blocks: "list[tuple[str, list[str]]]",
    sci_scores: dict,
    config: Any,
    segments: dict | None,
    subject: str,
    errors: list,
    figures_dir: Path,
    windowed: dict | None = None,
    raw_before_motion: mne.io.Raw | None = None,
    raw_after_motion: mne.io.Raw | None = None,
    suffix: str = "",
    xrange: "tuple[float, float] | None" = None,
) -> dict:
    """The carpet and GVTD panel, with the flagged spans drawn over it.

    ``suffix`` names both figures, so a per-condition page writes its own rather than
    overwriting the run's. ``xrange`` narrows the view to one condition **after** the panel
    is built, which is the only correct way to make this figure per condition: it derives
    its GVTD (filtered 0.01-0.5 Hz), its per-channel z-scoring and its threshold from
    whatever recording it is handed, so a cropped one would get filter edges on a short
    piece, a colour scale no other condition shares, and a threshold of its own. The same
    argument this docstring already makes for the corrected-versus-uncorrected pair.

    Both span lists come from the record's ``windowed`` section rather than being detected
    here. They were measured on the same channel set this panel draws, so reading them back
    is not a shortcut: it is what keeps the stripes on the carpet and the counts in the
    metrics table describing one event each.

    ``raw_after_motion`` puts the corrected trace and carpet in the same panel as the
    uncorrected one, matching the ``before -> after`` pairs in the metrics table.

    ``raw_gvtd`` is every channel the GVTD panel covers and ``gvtd_blocks`` is how it splits
    them into rows, normally long then short. ``gvtd_set`` names the first block, the
    canonical one and the only one the reported scalars come from; the rest are drawn for
    comparison. Everything else here stays on the long channels: the zoom below is
    per-channel optical density, so a wider set would only add rows.
    """
    carpet_gvtd_path = None
    carpet_gvtd_h = 600
    bad_segment_zoom_path = None

    def _spans(key: str) -> "list | None":
        value = (windowed or {}).get(key)
        return [tuple(span) for span in value] if value else None

    corrected_segments = _spans("motion_corrected_spans_s")
    spike_spans = _spans("spike_spans_s")
    # keyed by channel set, so each GVTD row shades the spans found on its own channels
    spike_by_set = {gvtd_set: spike_spans, "short": _spans("spike_spans_short_s")}

    with _guard("Carpet + GVTD", errors, subject):
        fig = carpet_gvtd_figure(raw_gvtd, raw_gvtd.ch_names, segments,
                                 corrected_segments=corrected_segments,
                                 spike_segments=spike_by_set,
                                 raw_after=raw_after_motion,
                                 channel_set=gvtd_set, blocks=gvtd_blocks)
        if xrange is not None:
            from fnirs_pipe.qc.condition_views import zoom_to_condition
            zoom_to_condition(fig, *xrange)
        carpet_gvtd_path, carpet_gvtd_h = _save_plotly_html(
            fig, figures_dir / f"carpet_gvtd{suffix}.html")

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
            _save_b64_png(b64, figures_dir / f"bad_segment_zoom{suffix}.png")
            bad_segment_zoom_path = _fig_href(figures_dir,
                                              f"bad_segment_zoom{suffix}.png")

    return {
        "carpet_gvtd_path": carpet_gvtd_path,
        "carpet_gvtd_h": carpet_gvtd_h,
        "bad_segment_zoom_path": bad_segment_zoom_path,
        "corrected_segments": corrected_segments,
        "spike_spans": spike_spans,
        "spike_by_set": spike_by_set,
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
    sep_bands=None,
    suffix: str = "",
    crop: "tuple[float, float] | None" = None,
) -> dict:
    """Beer-Lambert output and what the denoising did to it.

    The correlation panels are drawn from the recordings, the strips beside them from the
    record: a figure needs the samples, a number must not be computed twice. ``raw_errts``
    is the confound-regression residual, the stage the denoising before/after is taken
    against; it is absent on a prep-only run and on a denoise run with nothing to regress,
    and then only the before panel is drawn.

    ``crop`` narrows every panel here to one condition, and it is a parameter rather than a
    cropped input because the order matters and only this function knows it. The correlation
    panels and the spectra can be cut and then measured: a correlation is over whatever
    samples it gets, and ``compute_psd`` is Welch, which segments and tapers but does not
    band-pass. The stage comparison cannot, because it band-limits every stage to the
    analysis passband before subtracting them, deliberately (see
    :func:`~fnirs_pipe.qc.metrics.comparable_stage_metrics`), and at a 0.01 Hz high-pass
    that FIR runs about 330 s, longer than a 300 s condition. Cut first and it is filtered
    against its own two edges, which is the mismatch the old 0.02 Hz workaround existed for.

    So the stages are band-limited over the whole run and cut afterwards, the order the
    pipeline itself uses, and ``comparable_stage_metrics`` is then told not to filter again.
    """
    hbo_hbr_path = hbo_hbr_after_path = psd_panel_path = None
    psd_panel_h = 0

    def _cut(raw):
        """The span `crop` names, or the recording unchanged when there is no crop."""
        if raw is None or crop is None:
            return raw
        t0, t1 = max(0.0, float(crop[0])), min(float(raw.times[-1]), float(crop[1]))
        return raw if t1 <= t0 else raw.copy().crop(tmin=t0, tmax=t1)

    raw_haemo_cut, raw_errts_cut = _cut(raw_haemo), _cut(raw_errts)
    psd_stages_cut = [(label, _cut(raw)) for label, raw in (psd_stages or [])] or None

    with _guard("HbO-HbR correlation panel", errors, subject):
        b64 = hbo_hbr_correlation_panel(
            raw_haemo_cut,
            title="HbO–HbR Signal Quality — desc-preproc (before denoising)",
            sep_bands=sep_bands)
        _save_b64_png(b64, figures_dir / f"hbo_hbr_corr{suffix}.png")
        hbo_hbr_path = _fig_href(figures_dir, f"hbo_hbr_corr{suffix}.png")
    if raw_errts is not None:
        with _guard("HbO-HbR correlation panel (after)", errors, subject):
            b64 = hbo_hbr_correlation_panel(
                raw_errts_cut,
                title="HbO–HbR Signal Quality — desc-errts (after denoising)",
                sep_bands=sep_bands)
            _save_b64_png(b64, figures_dir / f"hbo_hbr_corr_after{suffix}.png")
            hbo_hbr_after_path = _fig_href(figures_dir, f"hbo_hbr_corr_after{suffix}.png")

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
            names, _ = long_short_channels(raw, sep_bands)
            if not names or len(names) == len(raw.ch_names):
                return raw
            return raw.copy().pick(names)

        with _guard("Denoising stage metrics", errors, subject):
            banded_here = crop is not None and (l_freq is not None or h_freq is not None)
            if banded_here:
                # filter over the whole run, cut after: the reverse gives a 330 s FIR two
                # edges of its own on a 300 s condition
                staged = [(label, _cut(_long_only(raw).copy()
                                       .filter(l_freq, h_freq, verbose=False)))
                          for label, raw in stages]
                stage_metrics = comparable_stage_metrics(
                    staged, None, None,
                    config.cardiac_l_freq, config.cardiac_h_freq,
                    config.resp_l_freq, config.resp_h_freq)
            else:
                stage_metrics = comparable_stage_metrics(
                    [(label, _long_only(raw)) for label, raw in stages], l_freq, h_freq,
                    config.cardiac_l_freq, config.cardiac_h_freq,
                    config.resp_l_freq, config.resp_h_freq)
            # the rows are band-limited either way; `banded` only reports whether that
            # function did it, and here it was done before the cut instead
            stage_banded = banded_here or bool(stage_metrics["banded"])
            fig = stage_metrics_figure(stage_metrics["labels"],
                                       denoise_stage_panels(stage_metrics))
            if fig is not None:
                stage_metrics_path, stage_metrics_h = _save_plotly_html(
                    fig, figures_dir / f"denoise_stage_metrics{suffix}.html")

    with _guard("PSD figure", errors, subject):
        fig_psd_custom = psd_figure(
            raw_haemo_cut, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
            resp=(config.resp_l_freq, config.resp_h_freq),
            stages=psd_stages_cut)
        psd_panel_path, psd_panel_h = _save_plotly_html(fig_psd_custom, figures_dir / f"psd_panel{suffix}.html")
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
    suffix: str = "",
) -> dict:
    epoch_preview_path = None
    epoch_preview_h = 0
    with _guard("Epoch preview", errors, subject):
        fig = build_epoch_preview_figure(raw_haemo, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)
        if fig is not None:
            epoch_preview_path, epoch_preview_h = _save_plotly_html(
                fig, figures_dir / f"epoch_preview{suffix}.html"
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

    Scored over the same window the epoch figures average, and on the intensity recording,
    so a trial's SCI and SNR are on the scale the metrics table prints rather than on the
    haemoglobin one. The scoring is shared with ``fnirs-qc prep-raw``, which is where this
    panel came from, and so is the meaning of an unset window: the figures fall back to
    ``_EPOCH_TMIN`` / ``_EPOCH_TMAX`` while the scoring uses each event's own duration,
    which is what a block design records and a fixed window would cut off.
    """
    tmin = getattr(config, "epoch_tmin", None)
    tmax = getattr(config, "epoch_tmax", None)
    window = (f"{tmin:g} to {tmax:g} s from each onset" if tmin is not None and tmax is not None
              else "each event's own duration")
    path, h = None, 0
    with _guard("Per-trial quality", errors, subject):
        labels, sqms = score_trials(
            raw_intensity, extract_markers(raw_intensity),
            getattr(config, "sci_threshold", SCI_PASS),
            config.cardiac_l_freq, config.cardiac_h_freq,
            tmin, tmax,
            psp_threshold=getattr(config, "psp_threshold", None),
            min_good_frac=getattr(config, "min_good_frac", None),
        )
        fig = trial_quality_heatmap(labels, sqms)
        if fig is not None:
            path, h = _save_plotly_html(fig, figures_dir / "trial_qc.html")
    return {"trial_qc_path": path, "trial_qc_h": h, "trial_qc_window": window}


def _section_evoked_topomap(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    suffix: str = "",
) -> dict:
    path = None
    with _guard("Evoked topomap", errors, subject):
        b64 = evoked_topomap_static(raw_haemo, epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax)
        if b64:
            _save_b64_png(b64, figures_dir / f"evoked_topomap{suffix}.png")
            path = _fig_href(figures_dir, f"evoked_topomap{suffix}.png")
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
    sci_threshold: float = SCI_PASS,
    psp_threshold: float | None = None,
) -> dict:
    """Read this run's SQM record; the report displays, it does not compute.

    The panel judges data quality, so its scalars come from the long-channel sections;
    ``raw`` stands in when the montage has no short channels to exclude. Nothing
    recomputes here: a missing record is reported as an error rather than silently
    measured a second time, and so is a record that carries none of the sections the panel
    reads, which is what a foreign file at this path looks like.

    The per-channel table is the one place short channels appear, read from ``raw_short``.
    They are screened by the same criteria as everything else, so their status is
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
            save_channel_csv(rows, sqm_label or f"sub-{subject}", out_dir, sci_threshold,
                             psp_threshold=psp_threshold)
    # the raw rows stay for the CSV and the quality grid, which want the numbers; the
    # template gets them formatted, so the per-channel table prints the same widths and the
    # same SCI verdict as the raw viewer and the GUI
    cells = format_rows(rows, sci_threshold, psp_threshold=psp_threshold)
    return {
        "sqm": sqm,
        "channel_rows": rows,
        "channel_cells": cells,
        "channel_blocks": separation_blocks(cells, bands_from_record(sqm)),
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
    for message in separation_notes(sqm, rows, short_channel_requested,
                                    bands_from_record(sqm)):
        _note(notes, subject, message)


def _section_channel_summary(
    rows: list,
    subject: str,
    errors: list,
    figures_dir: Path,
    sci_thresh: float = SCI_PASS,
    name: str = "channel_summary.html",
) -> dict:
    """``name`` so a per-condition page writes its own grid instead of overwriting the run's."""
    path, h = None, 0
    with _guard("Channel quality summary", errors, subject):
        fig = channel_quality_heatmap(sci_thresh=sci_thresh, **heatmap_args(rows))
        path, h = _save_plotly_html(fig, figures_dir / name)
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
    glm_activation_conditions: list[dict] = []

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

    # One file per condition behind a switch, not one tall image: five conditions stacked
    # reach ~3500 px, where a condition cannot be looked at on its own and two cannot be
    # compared. The colour scale is still shared across them, which is what keeps the
    # switch a comparison rather than five separate pictures.
    if glm_est is not None and conditions and raw_haemo is not None:
        with _guard("GLM activation panel", errors, subject):
            from fnirs_pipe.qc.figures import activation_condition_figures
            df = glm_est.to_dataframe().reset_index()
            if "Contrast" not in df.columns:
                for alt in ("contrast", "Regressor", "regressor", "condition", "Condition"):
                    if alt in df.columns:
                        df = df.rename(columns={alt: "Contrast"})
                        break
            if "Contrast" in df.columns:
                df["Contrast"] = df["Contrast"].astype(str)
                results_dict = {c: df[df["Contrast"] == c] for c in conditions}
                # _pair_fname strips everything but alphanumerics, so "game-1" and
                # "game 1" would land on one file and the switcher would offer two labels
                # pointing at the same picture
                used: set[str] = set()
                for label, b64 in activation_condition_figures(raw_haemo, results_dict):
                    slug = _pair_fname(label) or "cond"
                    if slug in used:
                        slug = f"{slug}{len(used) + 1}"
                    used.add(slug)
                    name = f"glm_activation_{slug}.png"
                    _save_b64_png(b64, figures_dir / name)
                    glm_activation_conditions.append(
                        {"label": label, "path": _fig_href(figures_dir, name)})
                # the first condition is what the img element loads before anything is
                # picked, and it is also what still gates the GLM section being open
                if glm_activation_conditions:
                    glm_activation_path = glm_activation_conditions[0]["path"]

    return {
        "glm_design_path": glm_design_path,
        "glm_design_heatmap_path": glm_design_heatmap_path,
        "glm_activation_path": glm_activation_path,
        "glm_activation_conditions": glm_activation_conditions,
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
    sep_bands=None,
) -> dict:
    alff_path = alff_topo_path = fc_path = fc_roi_path = fc_circle_path = fc_seed_path = None
    with _guard("ALFF/fALFF figure", errors, subject):
        if alff_df is not None:
            b64 = alff_falff_figure(alff_df)
            _save_b64_png(b64, figures_dir / "rest_alff.png")
            alff_path = _fig_href(figures_dir, "rest_alff.png")
    with _guard("ALFF topography", errors, subject):
        if alff_df is not None and raw_haemo is not None:
            b64 = alff_topo_figure(raw_haemo, alff_df, sep_bands=sep_bands)
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
            b64 = fc_seed_topo_figure(raw_haemo, fc_seed.get("hbo"), fc_seed.get("hbr"),
                                      sep_bands=sep_bands)
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
    by_condition: bool = False,
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
    # one resolution for the whole report, so every panel and the record agree
    sep_bands = separation_bands(config)
    raw_long = _prepare_long_raw(raw_intensity, subject, sep_bands)
    # the GVTD panel's channel set, which follows the separation bands and need not be raw_long
    gvtd_blocks = gvtd_channel_blocks(raw_intensity, sep_bands)
    gvtd_set = gvtd_blocks[0][0]
    raw_gvtd = raw_intensity.copy().pick([c for _, names in gvtd_blocks for c in names])

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
                            raw_long, raw_gvtd, gvtd_set, gvtd_blocks,
                            sci_scores, config, segments, subject, errors,
                            figures_dir, windowed=windowed_section,
                            raw_before_motion=raw_before_motion,
                            raw_after_motion=raw_after_motion)
    motion_det_vars   = _section_motion_detail(
                            raw_before_motion, raw_after_motion, subject, errors, figures_dir,
                            segments=segments,
                            corrected_segments=motion_vars.get("corrected_segments"),
                            spike_by_set=motion_vars.get("spike_by_set"),
                            gvtd_blocks=gvtd_blocks)
    haemo_vars        = _section_haemo(raw_haemo, config, subject, errors, figures_dir,
                                       l_freq=l_freq, h_freq=h_freq,
                                       raw_errts=raw_errts, psd_stages=psd_stages,
                                       record=record, sep_bands=sep_bands)
    denoise_carpet_path = None
    if after_haemo is not None:
        with _guard("Denoising carpet", errors, subject):
            b64 = carpet_compare_figure(raw_haemo, after_haemo, roi_map=roi_map)
            _save_b64_png(b64, figures_dir / "denoise_carpet.png")
            denoise_carpet_path = _fig_href(figures_dir, "denoise_carpet.png")
    # the trial window every epoch figure averages over. None on the config means the
    # report's own default, so an unset flag draws exactly what it always drew
    epoch_tmin = _EPOCH_TMIN if getattr(config, "epoch_tmin", None) is None else config.epoch_tmin
    epoch_tmax = _EPOCH_TMAX if getattr(config, "epoch_tmax", None) is None else config.epoch_tmax

    # --epoch-chunk-duration cuts the long annotations up once, here, rather than in each
    # figure: the epoch sections, the event timeline and the per-trial scoring then all read
    # one set of trials. A raw with no long annotation, or no flag, comes back unchanged.
    chunk = getattr(config, "epoch_chunk_duration", None)
    if chunk:
        raw_intensity = chunk_annotations(raw_intensity, chunk)
        raw_haemo = chunk_annotations(raw_haemo, chunk)
        if after_haemo is not None:
            after_haemo = chunk_annotations(after_haemo, chunk)
        _note(notes, subject,
              f"Task annotations were cut into {chunk:g} s trials before epoching, so a "
              f"trial in the epoch figures and the per-trial panel is one piece of a block "
              f"rather than the whole block.")

    # the "Raw Signal" section is the recording before anything was done to it, so its
    # figures come off desc-sci rather than the corrected desc-preproc the rest of the
    # report is built on. Same stage as the SCI/PSP windows and the SNR/CV numbers below.
    raw_haemo_uncorr  = _uncorrected_haemo(raw_before_motion, config, subject, errors)
    channel_det_vars  = _section_channel_detail(
                            raw_haemo_uncorr if raw_haemo_uncorr is not None else raw_haemo,
                            subject, errors, figures_dir,
                            epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
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
    if getattr(config, "epoch_tmin", None) is None:
        outruns = _epoch_window_mismatch(raw_haemo, epoch_tmax)
        if outruns is not None:
            _note(notes, subject,
                  f"The epoch figures average a {epoch_tmin:g} to {epoch_tmax:g} s window "
                  f"while this run's events are {outruns:g} s long, so they describe the "
                  f"start of each block rather than the whole of it. Pass --epoch-tmin / "
                  f"--epoch-tmax to widen it. The per-trial panel below is unaffected: it "
                  f"scores each event over its own duration.")

    epoch_skip        = _no_epoch_reason(raw_haemo, epoch_tmin, epoch_tmax)
    if epoch_skip is not None:
        _note(notes, subject,
              f"Epoch preview, evoked topomap, trial images and per-trial quality were "
              f"skipped because {epoch_skip}. The per-channel and layout figures show the "
              f"continuous signal instead.")
        epoch_vars       = {"epoch_preview_path": None, "epoch_preview_h": 0}
        trial_image_vars = {"trial_image_pairs": [], "trial_image_roi_pairs": []}
        topomap_vars     = {"evoked_topomap_path": None}
        trial_qc_vars    = {"trial_qc_path": None, "trial_qc_h": 0, "trial_qc_window": ""}
    else:
        epoch_vars        = _section_epoch_preview(raw_haemo, subject, errors, figures_dir,
                                                   epoch_tmin=epoch_tmin,
                                                   epoch_tmax=epoch_tmax)
        trial_image_vars  = _section_trial_image(epoch_haemo, subject, errors, figures_dir,
                                                 roi_map=roi_map,
                                                 epoch_tmin=epoch_tmin,
                                                 epoch_tmax=epoch_tmax)
        topomap_vars      = _section_evoked_topomap(epoch_haemo, subject, errors, figures_dir,
                                                    epoch_tmin=epoch_tmin,
                                                    epoch_tmax=epoch_tmax)
        trial_qc_vars     = _section_trial_qc(raw_intensity, config, subject, errors,
                                              figures_dir)
    glm_vars          = _section_glm(design_matrix, glm_est, raw_haemo, subject, errors, figures_dir, segments=segments)
    rest_vars         = _section_rest(alff_df, fc_df, subject, errors, figures_dir, fc_hbr_df=fc_hbr_df,
                                      fc_seed=fc_seed, fc_roi=fc_roi, raw_haemo=raw_haemo,
                                      sep_bands=sep_bands)
    sqm_vars          = _section_sqm(sci_scores, bad_channels, subject, errors,
                                     out_dir=out_path.parent / "nirs",
                                     sqm_label=sqm_label,
                                     sci_threshold=getattr(config, "sci_threshold", SCI_PASS),
                                     psp_threshold=getattr(config, "psp_threshold", None))
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
    # figures that reach the template as loose keywords rather than inside a section dict.
    # They live in one here so `_blanked` can empty them: passed loose, a whole-run figure
    # survives onto a per-condition page, which is how denoise_carpet first got there.
    loose_figure_vars = {"denoise_carpet_path": denoise_carpet_path}
    report_vars = dict(
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
        **loose_figure_vars,
        gvtd_set=gvtd_set,
        mode=mode or "",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render("subject_report.html.j2", **report_vars), encoding="utf-8")
    logger.info("sub-%s | QC report saved: %s", subject, out_path)
    logger.info("sub-%s | figures saved: %s", subject, figures_dir)

    if by_condition:
        with _guard("Per-condition reports", errors, subject):
            _write_condition_reports(
                report_vars,
                section_vars=(sci_vars, motion_vars, motion_det_vars, trial_image_vars,
                              topomap_vars, haemo_vars, channel_det_vars, psd_det_vars,
                              brain_vars, epoch_vars, trigger_vars, trial_qc_vars,
                              glm_vars, rest_vars, loose_figure_vars),
                raw_intensity=raw_intensity, config=config, subject=subject,
                out_path=out_path, out_dir=nirs_dir, sqm_label=sqm_label,
                figures_dir=figures_dir, sci_scores=sci_scores,
                bad_channels=bad_channels, errors=errors,
                # closures rather than another ten parameters: both panels take a long
                # arg list that already exists here, and only the suffix, the slice and
                # the view span differ per condition
                remake_sci=lambda suffix, windowed_slice, sci_pc: _section_sci(
                    raw_intensity, sci_pc, bad_channels, config, windowed_slice,
                    subject, errors, figures_dir, suffix=suffix),
                remake_motion=lambda suffix, span: _section_motion(
                    raw_long, raw_gvtd, gvtd_set, gvtd_blocks, sci_scores, config,
                    segments, subject, errors, figures_dir, windowed=windowed_section,
                    raw_before_motion=raw_before_motion,
                    raw_after_motion=raw_after_motion, suffix=suffix, xrange=span),
                remake_cropped=lambda suffix, span: _cropped_sections(
                    span, suffix, raw_haemo=raw_haemo, epoch_haemo=epoch_haemo,
                    raw_haemo_uncorr=raw_haemo_uncorr, raw_errts=raw_errts,
                    psd_stages=psd_stages, record=record, config=config,
                    l_freq=l_freq, h_freq=h_freq, sep_bands=sep_bands,
                    epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                    subject=subject, errors=errors, figures_dir=figures_dir))

    _build_mne_report(subject, raw_intensity, raw_haemo, out_path, errors)
    return notes


def _blanked(section_vars: tuple) -> dict:
    """Every section variable emptied, keeping its type.

    A per-condition page blanks by exclusion rather than listing what it shows. The other
    way round, a section added later would keep its whole-run figure on a page whose
    numbers describe one condition, and nothing would say so; this way a new section goes
    blank there until somebody decides what its per-condition form is.

    A string blanks to "" and not to None, even where the rest of the package spells an
    absent figure None. Both are falsy so every ``{% if path %}`` gate behaves the same,
    but Jinja prints None as the word "None", so an unguarded ``{{ }}`` would put it in the
    page; "" puts nothing there. A number blanks to 0 rather than None because the markup
    adds to the heights.
    """
    out: dict = {}
    for group in section_vars:
        for key, value in group.items():
            out[key] = (type(value)() if isinstance(value, (list, dict, str))
                        else 0 if isinstance(value, (int, float)) and not isinstance(value, bool)
                        else None)
    return out


# Figures a per-condition page may point at whatever the condition. The rule for being
# here is narrow: the figure has to describe the *run* and be unmistakable for the
# condition, so a reader cannot take it as this condition's. Two qualify. The provenance
# graph is one file lineage and identical on every page. The GLM design matrix is one model
# fitted over the whole recording with every condition as a column of it, all of them drawn
# and labelled, so it reads as the model and not as one condition's; a per-condition design
# matrix would be a different model, not a view of this one.
#
# Nothing else belongs. The carpet, the spectra and the correlation panels would all look
# like the condition's, which is why they are rebuilt per condition instead.
#
# Everything not listed has to carry the condition's own name, which `_figure_leaks` checks
# by suffix rather than by listing the panels: a per-panel prefix list would also pass a
# *different* condition's figure, worse than a run-wide one because the page would look
# per-condition and be the wrong condition.
_CONDITION_PAGE_FIGURES = ("provenance.", "glm_design_")


def _figure_leaks(page: dict, label_slug: str) -> "list[str]":
    """Whole-run figures that survived onto a condition page, by value rather than by name.

    `_blanked` empties the section variables it is given, so a figure passed to the template
    some other way slips through it. That is not a hypothetical: `denoise_carpet_path` was
    a loose keyword and appeared on every per-condition page. Checking the assembled values
    catches the next one without anybody remembering to extend a list.
    """
    leaks = []
    for key, value in page.items():
        if not isinstance(value, str) or "figures/" not in value:
            continue
        name = value.rsplit("/", 1)[-1]
        # every per-condition figure is written as <panel>_<slug>.<ext>
        if name.rsplit(".", 1)[0].endswith(f"_{label_slug}"):
            continue
        if any(name.startswith(ok) for ok in _CONDITION_PAGE_FIGURES):
            continue
        leaks.append(f"{key}={name}")
    return leaks


def _cropped_sections(
    span: "tuple[float, float]",
    suffix: str,
    *,
    raw_haemo, epoch_haemo, raw_haemo_uncorr, raw_errts, psd_stages, record, config,
    l_freq, h_freq, sep_bands, epoch_tmin, epoch_tmax, subject, errors, figures_dir,
) -> dict:
    """The panels that are safe to rebuild on a cropped copy, over one condition.

    Cropping is the right move for exactly these and the wrong one for the carpet and the
    SCI/PSP panel, and the line between them is whether the panel filters. It does not here:
    the haemoglobin timeseries are drawn as they are, the HbO-HbR correlation is a
    correlation over whatever samples it is given, and ``compute_psd`` is Welch, which
    segments and tapers but does not band-pass. So a cropped condition carries no filter
    edge that the whole run would not have had. A shorter span costs frequency resolution
    (1/900 Hz against 1/3900 Hz here) and averages fewer Welch segments, which makes the
    spectrum noisier and leaves it unbiased.

    The stage files handed to the PSD panels are cropped too, and they were filtered over
    the whole run before being written, so cutting them adds no filtering either.

    Cropping also drops the annotations outside the span, which is what makes the epoch and
    topography panels this condition's without being told which one it is. They stay honest
    only as far as the epoch window does: on a 900 s block in a -5 to 25 s window they
    describe its first seconds, the same limitation the run's own report carries, and
    ``--epoch-chunk-duration`` is what fixes it for both.
    """
    def crop(raw, pad: "tuple[float, float]" = (0.0, 0.0)):
        if raw is None:
            return None
        t0 = max(0.0, float(span[0]) + pad[0])
        t1 = min(float(raw.times[-1]), float(span[1]) + pad[1])
        return None if t1 <= t0 else raw.copy().crop(tmin=t0, tmax=t1)

    # The epoch panels get the window's own room on either side. A block annotation starts
    # exactly where the condition starts, so a crop on the bare span puts the event at t=0
    # with nothing before it, MNE drops the epoch for want of a baseline, and the panel
    # reports "no stimulus events found" on a recording that has five. Padding is only
    # about having samples to average; nothing here filters, so it carries no edge of its
    # own the way padding a crop before the bandpass would.
    epoch_pad = (min(0.0, float(epoch_tmin or 0.0)), max(0.0, float(epoch_tmax or 0.0)))

    haemo = crop(raw_haemo)
    if haemo is None:
        return {}
    stages = [(name, cropped) for name, raw in (psd_stages or [])
              if (cropped := crop(raw)) is not None] or None
    bands = {"cardiac": (config.cardiac_l_freq, config.cardiac_h_freq),
             "resp": (config.resp_l_freq, config.resp_h_freq)}
    out: dict = {}
    # the one panel handed a span rather than a cropped recording: it holds the stage list,
    # so only it can band-limit the whole run before cutting, which is the order its stage
    # comparison needs. See its docstring.
    out.update(_section_haemo(raw_haemo, config, subject, errors, figures_dir,
                              l_freq=l_freq, h_freq=h_freq, raw_errts=raw_errts,
                              psd_stages=psd_stages, record=record, sep_bands=sep_bands,
                              suffix=suffix, crop=span))
    out.update(_section_psd_detail(haemo, subject, errors, figures_dir,
                                   l_freq=l_freq, h_freq=h_freq, psd_stages=stages,
                                   suffix=suffix, **bands))
    # epoch_pad here too: this section's third panel epochs, like the two below it
    out.update(_section_channel_detail(crop(raw_haemo_uncorr, epoch_pad) or haemo, subject, errors,
                                       figures_dir, epoch_tmin=epoch_tmin,
                                       epoch_tmax=epoch_tmax, suffix=suffix, **bands))
    out.update(_section_epoch_preview(crop(raw_haemo, epoch_pad) or haemo, subject, errors,
                                      figures_dir, epoch_tmin=epoch_tmin,
                                      epoch_tmax=epoch_tmax, suffix=suffix))
    out.update(_section_evoked_topomap(crop(epoch_haemo, epoch_pad) or haemo, subject,
                                       errors, figures_dir, epoch_tmin=epoch_tmin,
                                       epoch_tmax=epoch_tmax, suffix=suffix))
    return out


def _condition_glm(report_vars: dict, label: str) -> dict:
    """The run's GLM panel narrowed to one condition's activation figure.

    Nothing is recomputed. ``_section_glm`` already rendered one figure per condition
    against a colour scale shared across them, so a condition page shows the one that is
    already on disk and drops the switcher, which would have one option.
    """
    conditions = [c for c in (report_vars.get("glm_activation_conditions") or [])
                  if c.get("label") == label]
    return {
        "glm_design_path": report_vars.get("glm_design_path"),
        "glm_design_heatmap_path": report_vars.get("glm_design_heatmap_path"),
        "glm_activation_conditions": conditions,
        "glm_activation_path": conditions[0]["path"] if conditions else None,
    }


def _windowed_slice(record: dict, windows: list, label: str) -> dict:
    """The record's ``windowed`` section with its matrices cut to one condition's columns.

    Only the four keys the SCI/PSP panel reads. The rest of that section is spans and
    channel-averaged series measured over the run, and handing those to a per-condition
    panel would put run-wide stripes over per-condition columns.
    """
    import numpy as np

    from fnirs_pipe.qc.metrics.windowed import _in_scope, window_centers

    windowed = record.get("windowed") or {}
    out: dict = {}
    for matrix_key, times_key in (("sci_matrix", "sci_times"), ("psp_matrix", "psp_times")):
        matrix, times = windowed.get(matrix_key), windowed.get(times_key)
        if not matrix or times is None:
            continue
        keep = _in_scope(window_centers(np.asarray(times)),
                         [w for w in windows if w[0] == label])
        if not keep.any():
            continue
        out[matrix_key] = np.asarray(matrix)[:, keep]
        out[times_key] = np.asarray(times)[keep]
    return out


def _write_condition_reports(
    report_vars: dict,
    *,
    section_vars: tuple,
    raw_intensity: mne.io.Raw,
    config: "PrepConfig",
    subject: str,
    out_path: Path,
    out_dir: Path | None,
    sqm_label: str | None,
    figures_dir: Path,
    sci_scores: dict,
    bad_channels: list,
    errors: list,
    remake_sci=None,
    remake_motion=None,
    remake_cropped=None,
) -> None:
    """One subject-report page per annotated condition, sliced out of the quality record.

    Nothing is measured again and the recording is never cut. The record already stores the
    channel-by-window SCI and PSP matrices, so a condition is a column selection out of the
    pass the run made; see :mod:`fnirs_pipe.qc.condition_views`.

    Four panels are per condition. The scalar panel and the channel table come off the
    sliced record; the SCI/PSP panel is rebuilt from the matrices sliced to the condition's
    columns, which is a real slice because nothing inside that figure filters; and the
    carpet is rebuilt over the whole run and *narrowed* to the condition, because that
    figure derives its GVTD, its z-scoring and its threshold from whatever it is handed.

    What stays blank is the response half: epochs, topographies, trial images and the GLM.
    The epoch window is set for a trial rather than a block, so those panels would describe
    the first seconds of each condition instead of the condition. The PSD stays blank for a
    different reason: it is one spectrum rather than a matrix, so there is nothing to slice.

    The channel set is the run's throughout, since one set has to serve every condition.
    """
    from fnirs_pipe.qc.condition_views import (
        condition_scalars, condition_slices_from_record, slice_record, span_share,
    )
    from fnirs_pipe.qc.hyper_report import condition_windows
    from fnirs_pipe.qc.metrics import resolve_cutoffs
    from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S, condition_window_means

    if out_dir is None or sqm_label is None:
        logger.warning("sub-%s | no quality record location; no per-condition pages", subject)
        return
    windows = condition_windows(raw_intensity, min_duration=2 * SCREEN_WINDOW_S)
    if not windows:
        logger.info("sub-%s | no annotation holds two screening windows; no per-condition "
                    "pages", subject)
        return
    record = json.loads(_sqm_record_path(out_dir, sqm_label).read_text(encoding="utf-8"))
    cutoffs = resolve_cutoffs(config)
    ch_names = list(((record.get("per_channel") or {}).get("raw") or {})
                    .get("sci_per_channel") or {})
    sliced_all = condition_slices_from_record(
        record, ch_names, windows, cutoffs["sci"], cutoffs["psp"])
    if not sliced_all:
        return

    gvtd = (record.get("windowed") or {})
    gvtd_by_cond = condition_window_means(
        gvtd.get("gvtd_per_window") or [], gvtd.get("gvtd_window_times_s") or [], windows
    ) if gvtd.get("gvtd_per_window") else {}

    # the run's booleans, kept as spans, so a condition counts them over its own stretch
    gvtd_above_spans = (record.get("windowed") or {}).get("gvtd_above_spans_s") or []
    spike_spans = (record.get("windowed") or {}).get("spike_spans_s") or []

    blanked = _blanked(section_vars)
    window_of = {w[0]: w for w in windows}
    for label, sliced in sliced_all.items():
        cond_record = slice_record(record, sliced)
        # this condition's own verdict, on the run's line. The page is a view of one
        # condition and says so; the run's verdict, which is what the data was processed
        # under, is one click away on the run's own page
        cond_frac = sliced.get("good_frac_per_channel") or {}
        cond_bad = ({ch for ch, v in cond_frac.items() if v < cutoffs["good_frac"]}
                    if cond_frac else set(bad_channels))
        retention = (1.0 - len(cond_bad) / len(cond_frac)) if cond_frac else None
        rows = channel_rows(cond_record, sci_scores, cond_bad)
        cells = format_rows(rows, cutoffs["sci"], psp_threshold=cutoffs["psp"])
        t0, t1 = window_of[label][1], window_of[label][2]
        scalars = condition_scalars(
            sliced, None if label not in gvtd_by_cond else float(gvtd_by_cond[label]),
            gvtd_above=span_share(gvtd_above_spans, t0, t1),
            spike_share=span_share(spike_spans, t0, t1),
            retention=retention)
        slug = _pair_fname(label)
        summary = _section_channel_summary(
            rows, subject, errors, figures_dir, cutoffs["sci"],
            name=f"channel_summary_{slug}.html")
        # the SCI/PSP panel over this condition's columns: a real slice, since the figure
        # is handed its matrices and derives nothing from a recording
        panels: dict = {}
        if remake_sci is not None:
            panels.update(remake_sci(f"_{slug}", _windowed_slice(record, windows, label),
                                     sliced.get("sci_per_channel") or {}))
        # and the carpet narrowed to it: measured over the run, viewed over the condition
        span = (t0, t1)
        if remake_motion is not None:
            panels.update(remake_motion(f"_{slug}", span))
        # and everything a crop is safe for: the haemoglobin panels, the spectra and the
        # per-channel detail. See _cropped_sections for where that line falls and why
        if remake_cropped is not None:
            panels.update(remake_cropped(f"_{slug}", span))
        # The GLM is already per condition and needs nothing rebuilt: one model is fitted
        # over the whole recording and each condition is a contrast of it, so this page
        # keeps its own activation figure out of the set the run rendered. The design
        # matrix stays whole, because every condition is a column of that one model and a
        # per-condition version of it would be a different model.
        panels.update(_condition_glm(report_vars, label))
        stem = f"{out_path.stem.removesuffix('_qc')}_desc-{_pair_fname(label)}_qc"
        page = {
            **report_vars, **blanked, **summary, **panels,
            "sqm": scalars,
            "channel_rows": rows,
            "channel_cells": cells,
            "channel_blocks": separation_blocks(cells),
            # no All / Long / Short columns: each would need its own slice, and the run's
            # split is one page away
            "sqm_all": {}, "sqm_long": {}, "sqm_short": {},
            "hb_all": {}, "hb_long": {}, "hb_short": {},
            "sqm_split": False, "hb_split": False,
            "heading": f"{report_vars.get('heading', '')} \u2014 {label}",
            "condition_label": label,
            "index_href": out_path.name,
        }
        leaks = _figure_leaks(page, _pair_fname(label))
        if leaks:
            # loud rather than silent: a run-wide figure under per-condition numbers reads
            # as that condition's, and nothing on the page would say otherwise
            logger.warning("sub-%s | condition %s still points at run-wide figures (%s); "
                           "they are being dropped", subject, label, ", ".join(leaks))
            page = {**page, **{k.split("=")[0]: None for k in leaks}}
        out = out_path.with_name(f"{stem}.html")
        out.write_text(render("subject_report.html.j2", **page), encoding="utf-8")
        logger.info("sub-%s | condition %s \u2192 %s", subject, label, out.name)



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


