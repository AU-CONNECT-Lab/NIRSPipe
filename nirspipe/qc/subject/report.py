"""Generate per-subject HTML QC report using Jinja2 + Plotly.

Each section is an independent _section_*() builder that returns a dict of
template variables. Failures are caught by _guard() and appended to the errors
list, and the rest of the report still renders. A section skipped because the run does
not carry what it needs is not a failure: it goes to the notes list instead.

Report sections
---------------
::

  Summary
    Subject metadata, bad-channel badge, run command.

  a. Raw Signal
    Per-channel HbO/HbR timeseries + PSD (dropdown selector).

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

  Per-condition Channel Quality
    Each condition page's channel grid, one block per metric, one row per condition.

  Per-trial Quality
    Every trial window scored on its own, over the epoch window, on the intensity
    recording.

  Errors / Methods / Software Versions
"""

import base64
from contextlib import contextmanager
from nirspipe.utils import is_marker, pair_of
from nirspipe.exceptions import StageError
from nirspipe.io.auxiliary import (
    ImuTrace, aux_table_units, find_aux_table, imu_traces, read_aux_table, table_channels,
)
from nirspipe.io.derivatives import entity_of
from nirspipe.pipeline.denoise import band_limited
from nirspipe.io.naming import parse_path, report_name
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Any
import html as _html

import mne
import mne.io

from nirspipe.qc.boilerplate import collect_software_versions, generate_methods_text
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.metrics.coupling import CV_WINDOW_S, PSP_WINDOW_S, SCI_WINDOW_S
from nirspipe.qc.metrics.gvtd import GVTD_MOTION_BAND
from nirspipe.qc.metrics.motion import SPIKE_CH_FRAC
from nirspipe.qc.common.channel_table import (
    CONDITION_OD_SPLIT_COLUMNS, MOTION_SPLIT_COLUMNS, OD_SPLIT_COLUMNS, WHOLE_RUN_ONLY_COLUMNS, channel_columns,
    channel_rows, format_rows, heatmap_args, measured_columns,
    registration_note, roi_overlap_note, separation_blocks, separation_notes,
)
from nirspipe.qc.common.figure_io import (
    CENTER_FIGURE_CSS, _fig_href, _pair_fname, save_png,
    _save_figure_html, _save_multi_fig_html,
    extract_markers, figure_namer, get_channel_pairs,
)
from nirspipe.qc.metrics import (
    CV_PASS, IMU_STAT_KEYS, gvtd_channel_blocks,
    registration_offset, separation_bands, separation_orphans, epochable_events,
    resolve_cutoffs,
)
from nirspipe.qc.metrics._helpers import bands_from_record
from nirspipe.qc.figures.common._utils import chunk_annotations
from nirspipe.qc.figures import (
    build_trigger_timeline_single,
    condition_colors,
    trial_quality_heatmap,
    carpet_gvtd_figure,
    carpet_compare_figure,
    bad_segment_zoom_figure,
    hbo_hbr_correlation_figure,
    hbo_hbr_fit_js,
    psd_figure,
    quality_brain_views,
    optode_layout_static,
    evoked_channel_map_figure,
    design_matrix_static_figure,
    design_matrix_heatmap,
    build_epoch_preview_figure,
    build_trial_image_figure,
    build_roi_trial_image_figure,
    build_trial_image_by_condition,
    build_roi_trial_image_by_condition,
    build_sci_psp_figure,
    denoise_stage_panels,
    stage_metrics_figure,
    build_channel_figure,
    build_motion_detail_figure,
    channel_quality_heatmap,
    condition_quality_heatmap,
    alff_topo_figure,
    fc_roi_matrix_figure,
    fc_seed_topo_figure,
    rest_channel_panel,
)
from nirspipe.qc.common.report_shell import (
    footer_vars, guard, note, page_vars, render,
)
from nirspipe.qc.subject.record_io import read_record
from nirspipe.qc.subject.sqm_record import record_path as _sqm_record_path, entities_of
from nirspipe.qc.subject.trial_qc import MIN_TRIAL_S, score_trials, trial_fits, trial_windows
from nirspipe.utils.lineage import lineage_of
from nirspipe.utils.logging import get_logger
from nirspipe.qc.metrics.windowed import _in_scope, window_centers
from nirspipe.qc.boilerplate.vocabulary import (
    format_metric, is_key_metric, metric_class, metric_label, metric_summary, with_unit,
)
from nirspipe.qc.common.record_views import condition_verdict_view
from nirspipe.qc.common.windows import refuse_colliding_labels
from nirspipe.qc.subject.condition_views import (
    condition_view_table, carpet_view_table, slice_record, with_condition_corr,
)

if TYPE_CHECKING:
    from nirspipe.pipeline.prep_pipeline import PrepConfig

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

    A 240 s task block against the 25 s fallback -> 240.0; a 5 s trial -> None.
    """
    durations = [float(a["duration"]) for a in raw_haemo.annotations
                 if is_marker(a["description"]) and float(a["duration"]) > 0]
    if not durations:
        return None
    median = float(np.median(durations))
    return median if median > epoch_tmax else None


def _no_epoch_reason(
    raw_haemo: mne.io.Raw,
    epoch_tmin: float = _EPOCH_TMIN,
    epoch_tmax: float = _EPOCH_TMAX,
    single_trial: bool = False,
) -> "str | None":
    """Why nothing can be epoched over the report's window, or None when something can.

    A block design that only marks where each condition starts and ends carries annotations
    but no trials: both windows run off an edge of the run. Asking first is what keeps the
    epoch sections from rendering empty and MNE from warning once per figure. The window is
    the run's own, so the reason it prints names the window that was actually asked for.

    A design of one long block per condition is the other way to have nothing to epoch. The
    events survive the window check, but no condition repeats, so every figure in the
    section averages one trial with itself and draws a 30 s slice of a block that runs for
    minutes. That reads as a response and is not one, so the section is skipped.

    ``single_trial`` (``--epoch-single-trial``) waives that last one, for a run where the
    one trial is the thing to look at: a block onset does evoke a transient, and n=1 makes it
    noisy rather than absent. It waives nothing else. The other two reasons are a run with no
    events and a window that fits inside none of them, and no flag makes either epochable.
    """
    events, event_id = epochable_events(raw_haemo, epoch_tmin, epoch_tmax)
    if len(events) > 0:
        counts = {name: int((events[:, 2] == code).sum()) for name, code in event_id.items()}
        counts = {k: v for k, v in counts.items() if v}
        if counts and max(counts.values()) < 2 and not single_trial:
            return (f"no condition repeats ({len(counts)} condition(s), one event each), so "
                    f"nothing in this section would be averaged")
        return None
    n_marks = sum(1 for a in raw_haemo.annotations if is_marker(a["description"]))
    if not n_marks:
        return "the run carries no events"
    return (f"the run carries {n_marks} marker(s), none of them leaving a full "
            f"{epoch_tmin:g} to {epoch_tmax:g} s window inside the recording")


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


def _save_plotly_html(fig, path: Path, div_id: str | None = None, extra_css: str = "",
                      views: "dict | None" = None, extra_js: str = "") -> tuple[str, int]:
    """:func:`_save_figure_html` plus the URL this report must link to the file by."""
    h = _save_figure_html(fig, path, extra_css=extra_css, div_id=div_id, views=views,
                          extra_js=extra_js)
    return _fig_href(path.name), h


# ---------------------------------------------------------------------------
# Shared preprocessing helper
# ---------------------------------------------------------------------------

def _prepare_long_raw(
    raw_intensity: mne.io.Raw, subject: str, sep_bands=None,
) -> mne.io.Raw:
    from nirspipe.qc.metrics import long_short_channels

    raw = raw_intensity.copy()
    long_names, _ = long_short_channels(raw, sep_bands)
    if not long_names:
        logger.warning("sub-%s | no long channels by separation; using all", subject)
        return raw
    raw.pick(long_names)
    return raw


# ---------------------------------------------------------------------------
# Section builders: each returns a dict of template variables
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
    fig_name,
) -> dict:
    """The SCI/PSP/CV panel, per channel and per window.

    ``fig_name`` names the figure, so a per-condition page writes its own instead of
    overwriting the run's. Handed a ``windowed`` whose matrices are already sliced to one
    condition, this panel is that condition's: nothing inside it filters or re-measures,
    which is why a real slice works here where the carpet has to be narrowed instead.

    ``windowed`` is the record's section of that name; the series are read from it rather
    than recomputed, so the panel and the stored numbers cannot disagree. An absent section
    leaves the per-window half of the panel out and the per-channel half intact.

    Every view in the panel is the uncorrected optical density, the stage coupling is judged
    on.
    """
    def _series(key: str) -> "np.ndarray | None":
        value = (windowed or {}).get(key)
        return None if value is None else np.asarray(value)

    sci_scores_matrix = _series("sci_matrix")
    sci_win_times     = _series("sci_times")
    psp_scores_matrix = _series("psp_matrix")
    psp_win_times     = _series("psp_times")
    cv_scores_matrix  = _series("cv_matrix")
    cv_win_times      = _series("cv_times")
    # the matrix rows follow the record's own channel list, so the labels must too
    stored = list((windowed or {}).get("sci_channels") or [])
    if stored and set(stored) == set(sci_scores):
        sci_scores = {ch: sci_scores[ch] for ch in stored}
    ch_names = list(sci_scores.keys())
    sci_psp_panel_path = None
    sci_psp_panel_h = 0

    def _per_channel(matrix) -> dict:
        if matrix is None or matrix.shape[0] != len(ch_names):
            return {}
        return dict(zip(ch_names, np.nanmean(matrix, axis=1)))

    psp_per_ch = _per_channel(psp_scores_matrix)
    cv_per_ch  = _per_channel(cv_scores_matrix)
    if cv_scores_matrix is not None and not cv_per_ch:
        cv_scores_matrix = None       # a matrix on a different channel set is not this panel's

    cutoffs = resolve_cutoffs(config)
    with _guard("SCI/PSP panel", errors, subject):
        fig = build_sci_psp_figure(
            sci_scores, psp_per_ch, set(bad_channels),
            sci_threshold=cutoffs["sci"],
            psp_threshold=cutoffs["psp"],
            sci_matrix=sci_scores_matrix,
            sci_win_times=sci_win_times,
            psp_matrix=psp_scores_matrix,
            psp_win_times=psp_win_times,
            cv_per_channel=cv_per_ch,
            cv_matrix=cv_scores_matrix,
            cv_win_times=cv_win_times,
            window_s=(windowed or {}).get("qc_window_s"),
        )
        sci_psp_panel_path, sci_psp_panel_h = _save_plotly_html(
            fig, figures_dir / fig_name("scipsp")
        )

    return {"sci_psp_panel_path": sci_psp_panel_path, "sci_psp_panel_h": sci_psp_panel_h,
            "cv_threshold": CV_PASS}


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


def _pair_label(pair: str, short: bool, rejected: bool) -> str:
    tags = [tag for tag, on in (("short", short), ("rejected", rejected)) if on]
    return f"{pair} ({', '.join(tags)})" if tags else pair


def _section_channel_detail(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    max_pts: int = 4000,
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
    sep_bands=None,
) -> dict:
    from nirspipe.qc.metrics import long_short_channels

    markers = extract_markers(raw_haemo)
    # rejected pairs too, named as such: this section is where a reader looks to see why
    pairs = get_channel_pairs(raw_haemo, exclude=())
    rejected = {pair_of(n) for n in raw_haemo.info["bads"]}
    # the selector mixed the two separations under names that do not say which is which, so
    # picking a short pair showed scalp haemodynamics with nothing on the page saying so
    _, short_names = long_short_channels(raw_haemo, sep_bands)
    short_pairs = {pair_of(n) for n in short_names}
    saved = []
    for pair in pairs:
        with _guard(f"Channel detail {pair}", errors, subject):
            # no epoch panel: this stage is unfiltered, so an epoch average here draws
            # cardiac ripple where a slow curve should be. The epoch section carries the
            # per-channel response on the denoised stage, in the trial image
            detail_fig, psd_fig, _ = build_channel_figure(
                raw_haemo, markers, pair, max_pts,
                cardiac=cardiac, resp=resp, epoch=False,
            )
            fname = fig_name("detail", channel=_pair_fname(pair))
            h = _save_multi_fig_html([detail_fig, psd_fig], figures_dir / fname)
            is_short = pair in short_pairs
            saved.append({"pair": pair, "path": _fig_href(fname), "h": h,
                          "label": _pair_label(pair, is_short, pair in rejected),
                          "short": is_short})
    return {"channel_pairs": saved}


def _motion_detail_figures(
    raw_od_before: mne.io.Raw | None,
    raw_od_after: mne.io.Raw | None,
    subject: str,
    errors: list,
    gvtd_blocks: "list[tuple[str, list[str]]]",
    segments: dict | None = None,
    corrected_by_set: dict | None = None,
    spike_by_set: dict | None = None,
    imu: "dict[str, ImuTrace] | None" = None,
) -> "list[tuple[str, Any]]":
    """One per-channel motion figure per channel, each with its own class's GVTD on top.

    The GVTD row follows the channel the figure is about rather than staying fixed: it is
    read against the derivative row directly below it, and the two separation classes are not
    on one scale. A channel in neither class falls back to the canonical set, which is the
    only one there is a reported number for.

    Built here and saved by :func:`_section_motion_detail`, because a per-condition page
    shows the same figures narrowed rather than remeasured: the GVTD row on each of them is
    filtered and thresholded over the whole run.
    """
    if raw_od_before is None or raw_od_after is None:
        return []
    shared_chs = [c for c in raw_od_after.ch_names if c in raw_od_before.ch_names]

    # the same blocks the carpet panel drew, so the two figures never name sets differently.
    # Required, not defaulted: gvtd_channel_blocks already handles a montage with no long channels
    members = [(name, names, set(names)) for name, names in gvtd_blocks]

    def set_of(ch: str) -> "tuple[str, list[str]]":
        for name, names, lookup in members:
            if ch in lookup:
                return name, names
        return gvtd_blocks[0]     # in no block (neither separation range): the canonical set

    built = []
    for ch in shared_chs:
        with _guard(f"Motion detail {ch}", errors, subject):
            set_name, picks = set_of(ch)
            built.append((ch, build_motion_detail_figure(
                raw_od_before, raw_od_after, ch, segments,
                corrected_segments=(corrected_by_set or {}).get(set_name),
                spike_segments=(spike_by_set or {}).get(set_name),
                gvtd_picks=picks, gvtd_set=set_name, imu=imu)))
    return built


def _condition_views(fig, spans: "list[tuple[str, float, float]]") -> "dict | None":
    """Every condition's view of one run-wide figure, on this report's figures.

    The table is assembled by :func:`~nirspipe.qc.subject.condition_views.condition_view_table`,
    which the raw viewer builds its own fragment views with, so the two cannot fork.
    """
    return condition_view_table(fig, spans)


def _section_motion_detail(
    figures: "list[tuple[str, Any]]",
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    condition_spans: "list[tuple[str, float, float]] | None" = None,
) -> dict:
    """The built per-channel motion figures written out, one file per channel.

    Every condition shows the same traces: these figures are measured over the run and only
    the view moves, the way ``_section_motion`` narrows the carpet. So the file is written
    once and carries each condition's window as a table the page picks from by URL fragment,
    rather than once per condition as copies identical but for a few axis numbers.

    The y axes follow the window, which the carpet's colour scale does not;
    ``window_view_spec`` is the same measurement as ``rescale_y_to_window``, handed over rather
    than applied.

    ``fig_name`` is the run's namer; the channel goes in as an entity, so the raw viewer
    and this report spell these the same way.
    """
    saved = []
    for ch, fig in figures:
        with _guard(f"Motion detail {ch}", errors, subject):
            views = _condition_views(fig, condition_spans or [])
            fname = fig_name("motion", channel=_pair_fname(ch))
            h = _save_multi_fig_html([fig], figures_dir / fname, views=views)
            saved.append({"pair": ch, "path": _fig_href(fname), "h": h})
    return {"motion_detail_pairs": saved}


def _condition_motion_detail(run_pairs: list, slug: str) -> dict:
    """The run's per-channel motion files, addressed at one condition's window."""
    return {"motion_detail_pairs": [{**p, "path": f"{p['path']}#{slug}"} for p in run_pairs]}


def _section_psd_detail(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    l_freq: float | None = None,
    h_freq: float | None = 0.4,
    cardiac: "tuple[float, float] | None" = None,
    resp: "tuple[float, float] | None" = None,
    psd_stages: "list[tuple[str, mne.io.Raw]] | None" = None,
    sep_bands=None,
) -> dict:
    from nirspipe.qc.metrics import long_short_channels

    pairs = get_channel_pairs(raw_haemo, exclude=())
    rejected = {pair_of(n) for n in raw_haemo.info["bads"]}
    short_pairs = {pair_of(n) for n in long_short_channels(raw_haemo, sep_bands)[1]}
    saved = []
    for pair in pairs:
        with _guard(f"PSD detail {pair}", errors, subject):
            picks = [c for c in (f"{pair} hbo", f"{pair} hbr") if c in raw_haemo.ch_names]
            if not picks:
                continue
            # a pair's own page draws it whatever its verdict; the spectrum drops marked bads
            raw_sub = _unmarked(raw_haemo.copy().pick(picks))
            # a later stage may have dropped the pair (bad channel), so each stage is
            # narrowed to whatever it still carries and skipped when that is nothing
            stages_sub = None
            if psd_stages is not None:
                stages_sub = [
                    (label, _unmarked(raw.copy().pick(present)))
                    for label, raw in psd_stages
                    if (present := [c for c in picks if c in raw.ch_names])
                ]
            fig = psd_figure(raw_sub, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
                             title=f"PSD: {pair}", cardiac=cardiac, resp=resp,
                             stages=stages_sub)
            fname = fig_name("psddetail", channel=_pair_fname(pair))
            path, h = _save_plotly_html(fig, figures_dir / fname)
            saved.append({"pair": pair, "path": path, "h": h,
                          "label": _pair_label(pair, pair in short_pairs, pair in rejected)})
    return {"psd_detail_pairs": saved}


def _unmarked(raw: mne.io.Raw) -> mne.io.Raw:
    raw.info["bads"] = []
    return raw


def _segments_in_window(segments: dict | None,
                        window: "tuple[float, float] | None") -> "list[tuple[float, float]]":
    """The flagged spans a page should draw, flattened out of the annotation groups.

    ::

      _segments_in_window({"BAD_gvtd": [(10, 5), (150, 40)]}, (100, 300))
      -> [(150, 40)]

    ``window`` None is the run, which draws all of them. A condition draws the ones that
    touch its window, a span across the boundary included: it is a movement the condition
    sat through whichever side it began on.
    """
    return [
        (onset, dur)
        for spans in (segments or {}).values()
        for onset, dur in spans
        if window is None or (onset < window[1] and onset + dur > window[0])
    ]


def _carpet_views(fig, spans: "list[tuple[str, float, float]]") -> "dict | None":
    """Each condition's window on the carpet, shared with the raw viewer's own carpet."""
    return carpet_view_table(fig, spans)


def _condition_carpet(run_vars: dict, slug: str) -> dict:
    """The run's carpet, addressed at one condition's window."""
    path = run_vars.get("carpet_gvtd_path")
    return {"carpet_gvtd_path": f"{path}#{slug}" if path else None,
            "carpet_gvtd_h": run_vars.get("carpet_gvtd_h")}


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
    fig_name,
    windowed: dict | None = None,
    raw_before_motion: mne.io.Raw | None = None,
    raw_after_motion: mne.io.Raw | None = None,
    condition_spans: "list[tuple[str, float, float]] | None" = None,
    skip_carpet: bool = False,
    window: "tuple[float, float] | None" = None,
    imu: "dict[str, ImuTrace] | None" = None,
) -> dict:
    """The carpet and GVTD panel, with the flagged spans drawn over it.

    ``fig_name`` names the bad-segment zoom, so a per-condition page writes its own rather
    than overwriting the run's. The carpet is not written per condition at all: it is
    narrowed to one **after** the panel is built, which is the only correct way to make this
    figure per condition, and narrowing moves nothing but the x range. So the run's file
    carries every condition's window in ``condition_spans`` and a condition page asks for
    one by URL fragment, ``skip_carpet`` saying it already has the file it needs.

    Building it per condition instead would derive its GVTD (filtered to the motion band), its
    per-channel z-scoring and its threshold from a cropped recording, so it would get filter
    edges on a short piece, a colour scale no other condition shares, and a threshold of its
    own. The same argument this docstring already makes for the corrected-versus-uncorrected
    pair.

    Both span lists come from the record's ``windowed`` section rather than being detected
    here. They were measured on the same channel set this panel draws, so reading them back
    is not a shortcut: it is what keeps the stripes on the carpet and the counts in the
    metrics table describing one event each.

    ``raw_after_motion`` puts the corrected trace and carpet in the same panel as the
    uncorrected one, matching the ``before -> after`` pairs in the metrics table.

    ``window`` keeps the bad-segment zoom to the segments inside one condition. Without it a
    condition page showed the ten longest in the recording, which on a censored run are
    mostly the ones GVTD censoring flagged and are as likely as not to have happened during
    another condition entirely, under a heading that read as this one's. A condition nothing
    was flagged in now has no zoom rather than the run's.

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
    corrected_by_set = {gvtd_set: corrected_segments,
                        "short": _spans("motion_corrected_spans_short_s")}

    if not skip_carpet:
        with _guard("Carpet + GVTD", errors, subject):
            fig = carpet_gvtd_figure(raw_gvtd, raw_gvtd.ch_names,
                                     corrected_segments=corrected_segments,
                                     spike_segments=spike_by_set,
                                     raw_after=raw_after_motion,
                                     channel_set=gvtd_set, blocks=gvtd_blocks, imu=imu)
            carpet_gvtd_path, carpet_gvtd_h = _save_plotly_html(
                fig, figures_dir / fig_name("carpet"),
                views=_carpet_views(fig, condition_spans or []))

    with _guard("Bad segment zoom", errors, subject):
        all_spans = _segments_in_window(segments, window)
        # both rows are optical density, either side of the motion correction
        if all_spans and raw_after_motion is not None:
            sorted_chs = sorted(sci_scores.keys(), key=lambda c: sci_scores.get(c, 0), reverse=True)
            rep_chs = [c for c in sorted_chs if c in raw_long.ch_names][:3]
            b64 = bad_segment_zoom_figure(
                raw_after=raw_after_motion,
                bad_segments=all_spans,
                ch_names=rep_chs or raw_long.ch_names[:3],
                raw_before=raw_before_motion,
            )
            bad_segment_zoom_path = save_png(b64, figures_dir,
                                             fig_name("badsegmentzoom", extension=".png"))

    return {
        "carpet_gvtd_path": carpet_gvtd_path,
        "carpet_gvtd_h": carpet_gvtd_h,
        "bad_segment_zoom_path": bad_segment_zoom_path,
        "spike_spans": spike_spans,
        "spike_by_set": spike_by_set,
        "corrected_by_set": corrected_by_set,
    }


def _section_haemo(
    raw_haemo: mne.io.Raw,
    config: Any,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    l_freq: float | None = None,
    h_freq: float | None = None,
    raw_errts: mne.io.Raw | None = None,
    psd_stages: "list[tuple[str, mne.io.Raw]] | None" = None,
    record: dict | None = None,
    sep_bands=None,
    crop: "tuple[float, float] | None" = None,
    psd: bool = True,
    mode: str | None = None,
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
    band-pass. The stage comparison band-limits every stage (see
    :func:`~nirspipe.qc.metrics.comparable_stage_metrics`), so the stages are filtered over
    the whole run and cut afterwards, and ``comparable_stage_metrics`` is then told not to
    filter again.

    ``psd`` False leaves the spectrum out and keeps the rest; only a caller holding both
    spans can tell whether the cut clears mne's ``n_fft``. See :func:`_cropped_sections`.
    """
    hbo_hbr_path = psd_panel_path = None
    hbo_hbr_h = psd_panel_h = 0

    def _cut(raw):
        """The span `crop` names, or the recording unchanged when there is no crop."""
        if raw is None or crop is None:
            return raw
        t0, t1 = max(0.0, float(crop[0])), min(float(raw.times[-1]), float(crop[1]))
        return raw if t1 <= t0 else raw.copy().crop(tmin=t0, tmax=t1)

    raw_haemo_cut, raw_errts_cut = _cut(raw_haemo), _cut(raw_errts)
    psd_stages_cut = [(label, _cut(raw)) for label, raw in (psd_stages or [])] or None

    # One figure for both stages rather than one each: they share a colour scale and a
    # channel order, which is what makes the pair subtract by eye. A run with no denoising
    # passes raw_after=None and the panel draws its one-stage form.
    with _guard("HbO-HbR correlation panel", errors, subject):
        fig = hbo_hbr_correlation_figure(
            raw_haemo_cut,
            title="HbO–HbR Signal Quality",
            sep_bands=sep_bands,
            raw_after=raw_errts_cut,
            task_modelled=(mode == "glm"))
        if fig is None:
            raise RuntimeError("no haemoglobin channels to correlate")
        # the panel sizes itself to the page: its matrix is square-constrained and only the
        # browser knows how wide the column is. See `fit_js`.
        hbo_hbr_path, hbo_hbr_h = _save_plotly_html(
            fig, figures_dir / fig_name("hbohbrcorr"),
            extra_js=hbo_hbr_fit_js(fig))

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
        from nirspipe.qc.metrics import (
            comparable_stage_metrics, long_short_channels,
        )

        def _long_only(raw: mne.io.Raw) -> mne.io.Raw:
            """The stage measured where the verdict lives, or unchanged with nothing to drop.

            Long channels only, so these panels answer the question the metrics table above
            them answers.
            """
            names, _ = long_short_channels(raw, sep_bands)
            if not names or len(names) == len(raw.ch_names):
                return raw
            return raw.copy().pick(names)

        with _guard("Denoising stage metrics", errors, subject):
            design = _filter_design(stages)
            banded_here = crop is not None and (l_freq is not None or h_freq is not None)
            if banded_here:
                # filter over the whole run and cut after, for the quality rows; what left
                # the recording is read off the stages as stored, cut the same way
                limited = [_cut(band_limited(_long_only(raw), l_freq, h_freq, **design))
                           for _, raw in stages]
                stage_metrics = comparable_stage_metrics(
                    [(label, _cut(_long_only(raw))) for label, raw in stages], l_freq, h_freq,
                    config.cardiac_l_freq, config.cardiac_h_freq,
                    config.resp_l_freq, config.resp_h_freq, limited=limited)
            else:
                stage_metrics = comparable_stage_metrics(
                    [(label, _long_only(raw)) for label, raw in stages], l_freq, h_freq,
                    config.cardiac_l_freq, config.cardiac_h_freq,
                    config.resp_l_freq, config.resp_h_freq, design=design)
            # the rows are band-limited either way; `banded` only reports whether that
            # function did it, and here it was done before the cut instead
            stage_banded = banded_here or bool(stage_metrics["banded"])
            fig = stage_metrics_figure(stage_metrics["labels"],
                                       denoise_stage_panels(stage_metrics))
            if fig is not None:
                stage_metrics_path, stage_metrics_h = _save_plotly_html(
                    fig, figures_dir / fig_name("denoisestages"))

    if psd:
        with _guard("PSD figure", errors, subject):
            fig_psd_custom = psd_figure(
                raw_haemo_cut, l_freq=l_freq, h_freq=h_freq, fmax=2.0,
                cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
                resp=(config.resp_l_freq, config.resp_h_freq),
                stages=psd_stages_cut)
            psd_panel_path, psd_panel_h = _save_plotly_html(
                fig_psd_custom, figures_dir / fig_name("psd"))
    return {
        "hbo_hbr_path":   hbo_hbr_path,
        "hbo_hbr_h":      hbo_hbr_h,
        "stage_metrics_path": stage_metrics_path, "stage_metrics_h": stage_metrics_h,
        "stage_banded": stage_banded,
        "psd_panel_path": psd_panel_path, "psd_panel_h": psd_panel_h,
        "psd_stage_labels": [label for label, _ in (psd_stages or [])],
        # the caption names the same two bands `physio_bands` shades and the band scalars
        # integrate over, read off the config so it cannot name a band the run did not use
        "psd_cardiac_band": (config.cardiac_l_freq, config.cardiac_h_freq),
        "psd_resp_band": (config.resp_l_freq, config.resp_h_freq),
    }


def _carpet_stages(raw_haemo: mne.io.Raw, psd_stages: "list | None") -> tuple:
    """``([(label, raw)], reference)``: the denoised haemo stage, and desc-preproc.

    One block rather than a chain. The reference is the Beer-Lambert output, drawn in the
    motion section and quoted here only as the SD each title is measured against.
    """
    # by name, not by position: a reordering of psd_stages must not swap the residual out
    haemo = {label: raw for label, raw in (psd_stages or [])
             if "hbo" in raw.get_channel_types()}
    reference = ("desc-preproc", raw_haemo)
    pick = next(((lab, haemo[lab]) for lab in ("desc-errts", "desc-filtered") if lab in haemo),
                None)
    return ([pick] if pick else [reference]), reference


def _section_stage_carpets(
    stages: tuple,
    roi_map: "dict | None",
    raw_gvtd: "mne.io.Raw | None",
    span: "tuple[float, float] | None",
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    raw_gvtd_after: "mne.io.Raw | None" = None,
    gvtd_blocks: "list | None" = None,
) -> dict:
    """The denoised carpet under one GVTD row per channel set.

    ``stages`` is what :func:`_carpet_stages` returns; ``raw_gvtd`` and ``raw_gvtd_after`` are
    the recordings either side of motion correction.
    """
    panels = []
    stages, reference = stages
    if not stages:
        return {"carpet_panels": panels}
    with _guard("Stage carpet", errors, subject):
        fig = carpet_compare_figure(stages, roi_map=roi_map, raw_gvtd=raw_gvtd,
                                    raw_gvtd_after=raw_gvtd_after, gvtd_blocks=gvtd_blocks,
                                    xlim=span, reference=reference)
        if fig is not None:
            path, h = _save_plotly_html(fig, figures_dir / fig_name("carpetstage"))
            panels.append({"label": "", "path": path, "h": h})
    return {"carpet_panels": panels,
            "carpet_stage_labels": [lab for lab, _ in stages]}


def _section_epoch_preview(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    sep_bands=None,
) -> dict:
    epoch_preview_path = None
    epoch_preview_h = 0
    with _guard("Epoch preview", errors, subject):
        fig = build_epoch_preview_figure(raw_haemo, epoch_tmin=epoch_tmin,
                                         epoch_tmax=epoch_tmax, sep_bands=sep_bands)
        if fig is not None:
            epoch_preview_path, epoch_preview_h = _save_plotly_html(
                fig, figures_dir / fig_name("epochmean")
            )
    return {"epoch_preview_path": epoch_preview_path, "epoch_preview_h": epoch_preview_h}


def _section_trigger_timeline(
    raw: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
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
            path, h = _save_plotly_html(fig, figures_dir / fig_name("trigger"))
    return {"trigger_timeline_path": path, "trigger_timeline_h": h}


def _section_trial_qc(
    raw_intensity: mne.io.Raw,
    config: Any,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    notes: list,
) -> dict:
    """Each trial window scored on its own, so one bad trial is visible before averaging.

    Scored over the same window the epoch figures average, and on the intensity recording,
    so a trial's SCI and SNR are on the scale the metrics table prints rather than on the
    haemoglobin one. The scoring is shared with ``nirspipe-qc prep-raw``, which is where this
    panel came from, and so is the meaning of an unset window: the figures fall back to
    ``_EPOCH_TMIN`` / ``_EPOCH_TMAX`` while the scoring uses each event's own duration,
    which is what a block design records and a fixed window would cut off.
    """
    from nirspipe.qc.common.windows import markers_on_data_axis

    tmin = getattr(config, "epoch_tmin", None)
    tmax = getattr(config, "epoch_tmax", None)
    window = (f"{tmin:g} to {tmax:g} s from each onset" if tmin is not None and tmax is not None
              else "each event's own duration")
    path, h = None, 0
    rows: list = []
    with _guard("Per-trial quality", errors, subject):
        # the data axis, because trial_sqm crops on it; extract_markers leaves the onsets on
        # the original recording's axis, which is the same offset for every trial and zero
        # only when the input was never cropped
        markers = markers_on_data_axis(raw_intensity)
        labels, sqms = score_trials(
            raw_intensity, markers,
            config.sci_threshold,
            config.cardiac_l_freq, config.cardiac_h_freq,
            tmin, tmax,
            psp_threshold=getattr(config, "psp_threshold", None),
            min_good_frac=getattr(config, "min_good_frac", None),
        )
        # the onset beside each scored trial, so a condition page can take its own rows out
        # of this table rather than scoring the same windows a second time
        windows = trial_windows(markers, tmin, tmax, float(raw_intensity.times[-1]))
        sfreq = float(raw_intensity.info["sfreq"])
        short = sum(not trial_fits(sfreq, t0, t1) for _, t0, t1, _ in windows)
        if short:
            _note(notes, subject, section_note("caveat.short_trials", n=short,
                                               total=len(windows), window=MIN_TRIAL_S))
        rows = [(onset, label, sqm)
                for (_, _, _, onset), label, sqm in zip(windows, labels, sqms)]
        fig = trial_quality_heatmap(labels, sqms)
        if fig is not None:
            path, h = _save_plotly_html(fig, figures_dir / fig_name("trialqc",
                                                                       suffix="qc"))
    return {"trial_qc_path": path, "trial_qc_h": h, "trial_qc_window": window,
            "trial_qc_rows": rows}


def _condition_trial_qc(
    rows: list,
    span: "tuple[float, float]",
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    min_trials: int,
    window: str = "",
) -> dict:
    """The run's per-trial table cut to the trials whose onset falls in one condition.

    ``window`` is the run's trial window as its caption prints it, carried over because the
    condition page blanks the run's own section variables.

    Nothing is rescored. A trial's SQM is measured on a crop of its own window and reads
    nothing outside it, so a condition's rows are the run's rows and recomputing them could
    only produce a second copy free to disagree.

    ``min_trials`` is the whole reason a block design sees nothing here: its condition window
    holds the single annotation that defines it, and one row is not a comparison. The test on
    ``t0`` is strict for the same reason it is in ``_trial_image_by_span``, so the two panels
    on one page always describe the same set of trials.

    The reason a page has no panel travels back with the result, because an empty section
    explains nothing and on a block design this one is always empty: the reason is what a
    reader will actually see here.
    """
    t0, t1 = float(span[0]), float(span[1])
    keep = [(label, sqm) for onset, label, sqm in rows if t0 < onset < t1]
    if len(keep) < min_trials:
        if not rows:
            reason = "the run carries no per-trial table to take rows from"
        elif not keep:
            reason = section_note("caveat.block_design_trials")
        else:
            # min_trials is 1 or 2, so a short window here holds exactly one trial
            reason = section_note("caveat.one_trial", n=len(keep))
        return {"trial_qc_path": None, "trial_qc_h": 0, "trial_qc_window": window,
                "condition_trial_reason": reason}
    path, h = None, 0
    with _guard("Per-trial quality", errors, subject):
        fig = trial_quality_heatmap([label for label, _ in keep], [sqm for _, sqm in keep])
        if fig is not None:
            path, h = _save_plotly_html(fig, figures_dir / fig_name("trialqc",
                                                                       suffix="qc"))
    return {"trial_qc_path": path, "trial_qc_h": h, "trial_qc_window": window,
            "condition_trial_reason": ""}


def _section_condition_trial_images(
    epoch_haemo: mne.io.Raw,
    spans: "list[tuple[str, float, float]]",
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    roi_map: dict | None = None,
    roi_map_name: "str | None" = None,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    min_trials: int = 2,
) -> dict:
    """Every condition's trial images, built in one pass so they share a colour scale.

    Returns ``{condition label: section vars}``. The pass is run-wide and the split is by
    condition window, which is why it cannot sit in :func:`_cropped_sections` with the other
    rebuilt panels: a scale taken on one cropped condition at a time would differ from page
    to page, and these pages are read against each other.
    """
    from nirspipe.qc.common.windows import markers_on_data_axis

    # the annotations alone say whether any window could fill a panel, and answering from
    # them costs nothing; the pass below epochs the recording once per channel
    onsets = [float(m["onset"]) for m in markers_on_data_axis(epoch_haemo)]
    if not any(sum(1 for o in onsets if t0 < o < t1) >= min_trials for _, t0, t1 in spans):
        return {}

    out: dict = {}

    def _collect(figs_by_label, entities, key, name):
        for label, figs in (figs_by_label or {}).items():
            fname = fig_name("trialimage", condition=_pair_fname(label), **entities)
            h = _save_multi_fig_html(figs, figures_dir / fname)
            entry = out.setdefault(label, {"trial_image_pairs": [], "trial_image_roi_pairs": []})
            entry[key].append({"pair": name, "path": _fig_href(fname), "h": h})

    for roi_name, chans in (roi_map or {}).items():
        with _guard(f"condition trial image ROI {roi_name}", errors, subject):
            _collect(build_roi_trial_image_by_condition(
                epoch_haemo, str(roi_name), chans, spans, epoch_tmin, epoch_tmax,
                min_trials=min_trials),
                {"segmentation": roi_map_name, "label": _pair_fname(str(roi_name))},
                "trial_image_roi_pairs", str(roi_name))

    # HbO only, as the run's own trial image is, and for the same reason: single-trial HbR
    # is too low-amplitude to read as an image
    for ch in [c for c in epoch_haemo.ch_names if c.endswith(" hbo")]:
        with _guard(f"condition trial image {ch}", errors, subject):
            _collect(build_trial_image_by_condition(
                epoch_haemo, ch, spans, epoch_tmin, epoch_tmax,
                min_trials=min_trials), {"channel": _pair_fname(ch)},
                "trial_image_pairs", ch)
    return out


def _section_evoked_topomap(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
    sep_bands=None,
) -> dict:
    """The evoked response per channel, long and short rows, with a time slider.

    ``sep_bands`` has to be the run's own separations: the short row is only a contamination
    check if it holds the channels the regression treated as short.
    """
    path, h = None, 0
    with _guard("Evoked channel map", errors, subject):
        fig = evoked_channel_map_figure(raw_haemo, epoch_tmin=epoch_tmin,
                                        epoch_tmax=epoch_tmax, sep_bands=sep_bands)
        if fig is not None:
            # it sets its own width, so without this it sits at the left of a wide page
            path, h = _save_plotly_html(
                fig, figures_dir / fig_name("evokedtopo", suffix="nirsmap"),
                extra_css=CENTER_FIGURE_CSS)
    return {"evoked_topomap_path": path, "evoked_topomap_h": h}


def _section_trial_image(
    raw_haemo: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    roi_map: dict | None = None,
    roi_map_name: "str | None" = None,
    epoch_tmin: float = -5.0,
    epoch_tmax: float = 25.0,
) -> dict:
    # raw_haemo here is the denoised (bandpassed, pre-regression) signal, not preproc.
    # only for task data with (non-BAD) events; skip early otherwise
    if not any(is_marker(a["description"]) for a in raw_haemo.annotations):
        return {"trial_image_pairs": [], "trial_image_roi_pairs": []}

    roi_saved = []
    for roi_name, chans in (roi_map or {}).items():
        with _guard(f"trial image ROI {roi_name}", errors, subject):
            figs = build_roi_trial_image_figure(raw_haemo, str(roi_name), chans, epoch_tmin, epoch_tmax)
            if figs:
                fname = fig_name("trialimage", segmentation=roi_map_name,
                                 label=_pair_fname(str(roi_name)))
                h = _save_multi_fig_html(figs, figures_dir / fname)
                roi_saved.append({"pair": str(roi_name), "path": _fig_href(fname), "h": h})

    saved = []
    # HbO only: single-trial HbR is too low-amplitude to read as an image, and the HbO/HbR
    # relation is already reported by hbo_hbr_corr and the per-channel detail figure
    for ch in [c for c in raw_haemo.ch_names if c.endswith(" hbo")]:
        with _guard(f"trial image {ch}", errors, subject):
            figs = build_trial_image_figure(raw_haemo, ch, epoch_tmin, epoch_tmax)
            if figs:
                fname = fig_name("trialimage", channel=_pair_fname(ch))
                h = _save_multi_fig_html(figs, figures_dir / fname)
                saved.append({"pair": ch, "path": _fig_href(fname), "h": h})
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
        return read_record(_sqm_record_path(out_dir, sqm_label))
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

    The optical density either side of the motion step is on disk as ``desc-sci`` and
    ``desc-motcorrected``, which is where the quality record reads it from, so the report
    reads the same files rather than holding two full recordings in memory.
    """
    if out_dir is None or sqm_label is None:
        return None
    with _guard(f"Reading desc-{desc}", errors, subject):
        from nirspipe.io.snirf import read_snirf
        from nirspipe.qc.subject.sqm_record import scan_runs
        path = (scan_runs(out_dir).get(sqm_label) or {}).get(desc)
        return None if path is None else read_snirf(path)
    return None


def _load_imu(
    out_dir: Path | None,
    sqm_label: str | None,
    subject: str,
    errors: list,
) -> "dict[str, ImuTrace] | None":
    """The run's IMU traces, from the aux table preprocessing left beside its stages."""
    if out_dir is None or sqm_label is None:
        return None
    with _guard("Reading IMU", errors, subject):
        from nirspipe.qc.subject.sqm_record import scan_runs
        stages = list((scan_runs(out_dir).get(sqm_label) or {}).values())
        table = find_aux_table(stages[0]) if stages else None
        if table is None:
            return None
        return imu_traces(*table_channels(read_aux_table(table)), aux_table_units(table))
    return None


def _section_sqm(
    sci_scores: dict,
    bad_channels: list,
    subject: str,
    errors: list,
    out_dir: Path | None = None,
    *,
    sqm_label: str | None = None,
    sci_threshold: float,
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
    those rows is :mod:`nirspipe.qc.common.channel_table`, which the raw views share, so the
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
    motion_sets: dict = {}
    record_read: dict = {}
    with _guard("SQM record", errors, subject):
        if out_dir is None or sqm_label is None:
            raise FileNotFoundError("no SQM record location for this run")
        record_file = _sqm_record_path(out_dir, sqm_label)
        record = record_read = read_record(record_file)
        per_channel = record.get("per_channel") or {}
        raw_key = "raw_long" if "raw_long" in record else "raw"
        keys = (raw_key, "motion", "preproc")
        if not any(record.get(k) for k in keys):
            raise ValueError(
                f"{record_file.name} holds none of {keys}; not a sectioned SQM record")
        # `preproc` before `preproc_long`, so the long values win where they exist and the
        # whole-file ones the split does not carry (pct_data_retained) survive underneath
        # `motion_long` the same way: only its frame counts differ from `motion`
        # `censor` and `imu` last and unsuffixed: their keys (gvtd_censor_*, gyro_speed_*,
        # accel_jerk_*) collide with nothing
        for key in (*keys, "motion_long", "preproc_long", "censor", "imu"):
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
        # The denoised side of the haemoglobin metrics, from `errts`; a run that regressed
        # nothing prints single values.
        errts_key = "errts_long" if record.get("errts_long") else "errts"
        for k, v in (record.get(errts_key) or {}).items():
            sqm[f"{k}_errts"] = v
        # ---- GCOR either side of the confound regression ----
        # `filtered` on the before side, so the pair isolates the regression. Both sides come
        # off the record's own sections, the same channel set as every row beside it.
        filtered_key = "filtered_long" if record.get("filtered_long") else "filtered"
        for key in ("gcor_hbo", "gcor_hbr"):
            pre = (record.get(filtered_key) or {}).get(key)
            post = (record.get(errts_key) or {}).get(key)
            if pre is not None and post is not None:
                sqm[f"{key}_prereg"], sqm[f"{key}_postreg"] = pre, post
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
        # the motion panel over the same three sets, each carrying both sides of the
        # correction. GVTD is an RMS across channels, so these are three measurements and
        # not three groupings of one, which is why they cannot come out of the table above.
        # the correction footprint joins them unsuffixed: it is measured across the step
        # rather than either side of it, so it has no before and after to pair
        motion_sets = {
            name: {**(record.get(raw_name) or {}),
                   **(record.get(mc_name) or {}),
                   **{f"{k}_post": v
                      for k, v in (record.get(post_name) or {}).items()}}
            for name, raw_name, mc_name, post_name in (
                ("all", "raw", "motion", "motion_post"),
                ("long", "raw_long", "motion_long", "motion_post_long"),
                ("short", "raw_short", "motion_short", "motion_post_short"))
        }

    rows = channel_rows(record_read, sci_scores, bad_channels)
    # the raw rows stay for the quality grid, which wants the numbers; the
    # template gets them formatted, so the per-channel table prints the same widths and the
    # same SCI verdict as the raw viewer and the GUI
    cells = format_rows(rows, sci_threshold, psp_threshold=psp_threshold)
    return {
        "sqm": sqm,
        "channel_rows": rows,
        "channel_cells": cells,
        # the bands are stamped in `raw` only, which `sqm` was not filled from on a split run
        "channel_blocks": separation_blocks(cells, bands_from_record(sqm_all)),
        # the table groups by separation, so the block header says which side a row is on
        "channel_columns": channel_columns(("separation",)),
        "sqm_all": sqm_all,
        "sqm_long": sqm_long,
        "sqm_short": sqm_short,
        "hb_all": hb_all,
        "hb_long": hb_long,
        "hb_short": hb_short,
        "motion_sets": motion_sets,
        # three columns are worth printing only when all three are real
        "sqm_split": bool(sqm_long and sqm_short),
        "hb_split": bool(hb_long and hb_short),
        "motion_split": bool(motion_sets.get("long") and motion_sets.get("short")),
    }


def _note_separation(
    notes: list,
    subject: str,
    sqm: dict,
    rows: list,
    short_channel_requested: bool = False,
    orphan_mm: "dict[str, float] | None" = None,
    sep_bands=None,
) -> None:
    """File the montage-split warnings as run notes, one note each.

    The wording is shared with the raw views; what differs is where it goes. Here it joins
    the report's notes list and the run log, so a reader who never opens the per-channel
    table still learns the split did not come out the way the metrics assume.

    ``sep_bands`` is the run's own, not read back off ``sqm``: the report assembles that in
    memory and does not stamp it, so the note would quote the default gap at any run.
    """
    for message in separation_notes(sqm, rows, short_channel_requested,
                                    sep_bands if sep_bands is not None else bands_from_record(sqm),
                                    orphan_mm):
        _note(notes, subject, message)


def _filter_design(stages: "list[tuple[str, mne.io.Raw]]") -> dict:
    """The filter a run's stages went through, read off the stamp of the first that records it.

    ``{"method": "iir", "order": 4}``, or ``{}`` for the pipeline's default when none does.
    """
    for _, raw in stages:
        params = (lineage_of(raw).params if lineage_of(raw) else None) or {}
        method = params.get("filter_method")
        if method:
            order = params.get("filter_order")
            return {"method": method, **({"order": int(order)} if order is not None else {})}
    return {}


def _section_channel_summary(
    rows: list,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    sci_thresh: float,
    psp_thresh: "float | None" = None,
    good_frac_thresh: "float | None" = None,
) -> dict:
    """``fig_name`` so a per-condition page writes its own grid instead of overwriting the run's.

    The PSP and coupled-share lines colour their rows; None keeps the package default.
    """
    path, h = None, 0
    with _guard("Channel quality summary", errors, subject):
        lines = {k: v for k, v in (("psp_thresh", psp_thresh),
                                   ("good_frac_thresh", good_frac_thresh)) if v is not None}
        fig = channel_quality_heatmap(sci_thresh=sci_thresh, **lines, **heatmap_args(rows))
        path, h = _save_plotly_html(fig, figures_dir / fig_name("chsummary", suffix="qc"))
    return {"channel_summary_path": path, "channel_summary_h": h}


def _bad_count(bad: "list[str]", n_total: int) -> "tuple[int, int, float, str]":
    """The summary box's count, total, share in percent and badge colour."""
    rate = 100 * len(bad) / n_total if n_total > 0 else 0.0
    badge = "badge-green" if rate < 10 else "badge-yellow" if rate < 30 else "badge-red"
    return len(bad), n_total, rate, badge


def _failing_summary(failing: "list[str]", n_total: int) -> dict:
    """A condition page's summary box: the channels failing on its own stretch, so named."""
    n_bad, n_total, bad_rate, badge_class = _bad_count(failing, n_total)
    return {"n_bad": n_bad, "n_total": n_total, "bad_rate": bad_rate,
            "badge_class": badge_class, "bad_channels": failing,
            "bad_heading": section_note("summary.condition_failing")}


def _condition_channel_rows(record: dict, entry: dict, sci_scores: dict,
                            bad_channels: "set[str]") -> list:
    """One condition's channel rows, ``bad_channels`` deciding Status.

    A condition page passes the run's rejections; the run page's per-condition grid passes
    the condition's own assessment, ``entry["bad_channels"]``.
    """
    sliced = entry.get("per_channel") or {}
    cond_record = with_condition_corr(slice_record(record, sliced),
                                      sliced.get("hbo_hbr_corr_per_channel") or {})
    return channel_rows(cond_record, sci_scores, set(bad_channels))


def _section_condition_summary(
    record: dict,
    sci_scores: dict,
    config: Any,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
) -> dict:
    """Every condition page's channel grid on the run page, regrouped by metric."""
    path, h = None, 0
    by_condition = record.get("by_condition") or {}
    if len(by_condition) >= 2:
        with _guard("Per-condition channel quality", errors, subject):
            conditions = [
                (label, heatmap_args(_condition_channel_rows(
                    record, entry, sci_scores, set(entry.get("bad_channels") or ()))))
                for label, entry in by_condition.items()]
            # the condition pages' own cutoffs, so a cell here matches the cell there
            cutoffs = resolve_cutoffs(config)
            # each condition's own assessment, for choosing conditions: not a rejection
            fig = condition_quality_heatmap(conditions, sci_thresh=cutoffs["sci"],
                                            psp_thresh=cutoffs["psp"],
                                            good_frac_thresh=cutoffs["good_frac"],
                                            status_label="In condition",
                                            status_words=("pass", "fail"))
            if fig is not None:
                path, h = _save_plotly_html(fig, figures_dir / fig_name("condsummary",
                                                                        suffix="qc"))
    return {"condition_summary_path": path, "condition_summary_h": h}


def _section_brain(
    sci_scores: dict,
    bad_channels: list,
    coords_head: np.ndarray | None,
    good_mask: np.ndarray | None,
    raw_intensity: mne.io.Raw,
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
    ch_names_brain: list[str] | None = None,
    *,
    sci_threshold: float,
) -> dict:
    """The 3D quality views and the optode flat map, side by side in one PNG.

    ``fig_name`` names the file, so a per-condition page writes its own. Both figures colour a
    channel by its SCI against the run's line and mark the rejected ones, and a condition has
    both of those of its own, so this is a real per-condition figure rather than a narrowed
    view: nothing in it is a time series.
    """
    brain_views_path = None
    if coords_head is not None and good_mask is not None and sci_scores:
        with _guard("Brain views", errors, subject):
            import io as _io
            from PIL import Image as _PILImage

            views_name = fig_name("brainviews", extension=".png")
            ch_names = ch_names_brain if ch_names_brain is not None else list(sci_scores.keys())
            brain_b64 = quality_brain_views(ch_names, coords_head, good_mask,
                                            raw=raw_intensity, sci_scores=sci_scores,
                                            sci_threshold=sci_threshold)

            optode_b64 = None
            with _guard("Optode flat map", errors, subject):
                optode_b64 = optode_layout_static(raw_intensity, sci_scores, bad_channels,
                                                  sci_threshold=sci_threshold)

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
                brain_views_path = save_png(combined_b64, figures_dir, views_name)
            elif brain_b64:
                brain_views_path = save_png(brain_b64, figures_dir, views_name)
            else:
                raise RuntimeError("brain_b64 is None")
    return {"brain_views_path": brain_views_path}



def _section_glm(
    design_matrix: "Any | None",
    glm_est: "Any | None",
    raw_haemo: "mne.io.Raw | None",
    subject: str,
    errors: list,
    figures_dir: Path,
    fig_name,
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
            if not c.startswith(("drift_", "cosine_", "constant", "intercept", "short", "aux_"))
        ]
        with _guard("GLM design matrix (timeseries)", errors, subject):
            b64 = design_matrix_static_figure(design_matrix, conditions, segments=segments)
            glm_design_path = save_png(b64, figures_dir,
                                       fig_name("timeseries", suffix="design", extension=".png"))
        with _guard("GLM design matrix (heatmap)", errors, subject):
            b64 = design_matrix_heatmap(design_matrix)
            glm_design_heatmap_path = save_png(b64, figures_dir,
                                               fig_name("heatmap", suffix="design", extension=".png"))

    # One file per condition behind a switch, not one tall image: five conditions stacked
    # reach ~3500 px, where a condition cannot be looked at on its own and two cannot be
    # compared. The colour scale is still shared across them, which is what keeps the
    # switch a comparison rather than five separate pictures.
    if glm_est is not None and conditions and raw_haemo is not None:
        with _guard("GLM activation panel", errors, subject):
            from nirspipe.qc.figures import activation_condition_figures
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
                failed: list[tuple[str, str]] = []
                rendered = activation_condition_figures(raw_haemo, results_dict, failed=failed)
                errors.extend(f"GLM activation panel ({cond}): {reason}"
                              for cond, reason in failed)
                for label, b64 in rendered:
                    slug = _pair_fname(label) or "cond"
                    if slug in used:
                        slug = f"{slug}{len(used) + 1}"
                    used.add(slug)
                    path = save_png(b64, figures_dir, fig_name("glmactivation", suffix="nirsmap",
                                                                extension=".png", condition=slug))
                    if path:
                        glm_activation_conditions.append({"label": label, "path": path})
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
    fig_name,
    fc_hbr_df: "Any | None" = None,
    fc_seed: dict | None = None,
    fc_roi: dict | None = None,
    raw_haemo: "mne.io.Raw | None" = None,
    roi_map_name: "str | None" = None,
    sep_bands=None,
) -> dict:
    panel_path = alff_topo_path = fc_roi_path = fc_seed_path = None
    alff_topo_h = fc_seed_h = panel_h = fc_roi_h = 0
    with _guard("Rest channel panel", errors, subject):
        if fc_df is not None:
            fig = rest_channel_panel(fc_df, fc_hbr_df, alff_df, raw=raw_haemo,
                                     sep_bands=sep_bands)
            if fig is not None:
                panel_path, panel_h = _save_plotly_html(
                    fig, figures_dir / fig_name("restpanel"))
    with _guard("ALFF topography", errors, subject):
        if alff_df is not None and raw_haemo is not None:
            fig = alff_topo_figure(raw_haemo, alff_df, sep_bands=sep_bands)
            if fig is not None:   # None means the montage carries no optode positions
                alff_topo_path, alff_topo_h = _save_plotly_html(
                    fig, figures_dir / fig_name("topo", suffix="nirsmap",
                                                statistic="alff"))
    with _guard("ROI FC matrix", errors, subject):
        if fc_roi:
            fig = fc_roi_matrix_figure(fc_roi)
            if fig is not None:
                fc_roi_path, fc_roi_h = _save_plotly_html(
                    fig, figures_dir / fig_name(
                        "matrix", suffix="relmat", segmentation=roi_map_name,
                        aggregation="roi", statistic="pearson"))
    with _guard("FC seed topography", errors, subject):
        if fc_seed and raw_haemo is not None:
            fig = fc_seed_topo_figure(raw_haemo, fc_seed.get("hbo"), fc_seed.get("hbr"),
                                      sep_bands=sep_bands)
            if fig is not None:   # None means the montage carries no optode positions
                fc_seed_path, fc_seed_h = _save_plotly_html(
                    fig, figures_dir / fig_name(
                        "matrix", suffix="relmat", segmentation=roi_map_name,
                        aggregation="seed", statistic="pearson"))
    return {"rest_panel_path": panel_path, "rest_panel_h": panel_h,
            "rest_alff_topo_path": alff_topo_path, "rest_alff_topo_h": alff_topo_h,
            "rest_fc_roi_path": fc_roi_path, "rest_fc_roi_h": fc_roi_h,
            "rest_fc_seed_path": fc_seed_path,
            "rest_fc_seed_h": fc_seed_h}


def _glm_betas_table(df: "Any", conditions: list[str]) -> str:
    """Return an HTML table of per-channel GLM betas (theta)."""
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

def _filled_by(filled_settings: list[tuple[str, Any, str]], mode: str | None) -> list:
    """Group the settings nobody typed by the layer that set them, for the run command note.

    ``[("high_pass", 0.01, "mode"), ("hrf_model", "spm", "config")]`` under rest gives
    ``[("the --mode rest defaults", [("high_pass", 0.01)]), ("--config", [("hrf_model", "spm")])]``.
    """
    labels = {"mode": f"the --mode {mode} defaults", "config": "--config"}
    return [(labels[layer], [(arg, val) for arg, val, src in filled_settings if src == layer])
            for layer in ("mode", "config")
            if any(src == layer for _, _, src in filled_settings)]


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
    epoch_single_trial: bool = False,
    alff_df: "Any | None" = None,
    fc_df: "Any | None" = None,
    fc_hbr_df: "Any | None" = None,
    fc_seed: dict | None = None,
    fc_roi: dict | None = None,
    after_haemo: mne.io.Raw | None = None,
    roi_map: dict | None = None,
    provenance_path: str | None = None,
    sqm_label: str | None = None,
    roi_map_name: str | None = None,
    filled_settings: list[tuple[str, Any, str]] | None = None,
) -> list[str]:
    """Render the QC report for one run and save as HTML. Returns its run-level notes.

    sqm_label is the run this report covers, as a BIDS stem (``sub-01_task-rest``). It picks
    the SQM record and the intermediate stage files off disk, and it goes into every figure's
    name so several runs of one subject share one ``figures/`` folder without overwriting
    each other. Passing None falls back to ``sub-<subject>``.

    roi_map_name is the ROI mapping's label, which the figures drawn over ROIs carry so they
    name the same segmentation their tables do.

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
    # the run is in every figure's name rather than in a directory of its own, so one
    # subject's runs share this folder without the same panel being called the same thing
    # in two places. A run with no label falls back to the subject, as out_path does.
    fig_name = figure_namer(sqm_label or f"sub-{subject}")

    # the run's own nirs/, session level included; the report itself sits above sessions
    ses = entity_of(sqm_label, "ses") if sqm_label else None
    nirs_dir = out_path.parent / (f"ses-{ses}" if ses else "") / "nirs"
    record            = _load_record(nirs_dir, sqm_label, subject, errors)
    windowed_section  = record.get("windowed") or None
    raw_before_motion = _load_stage_raw(nirs_dir, sqm_label, "sci", subject, errors)
    raw_after_motion  = _load_stage_raw(nirs_dir, sqm_label, "motcorrected", subject, errors)
    raw_errts         = _load_stage_raw(nirs_dir, sqm_label, "errts", subject, errors)
    imu               = _load_imu(nirs_dir, sqm_label, subject, errors)
    # the haemo chain as it exists on disk, in the order it was written; a stage missing here
    # gets no line
    psd_stages        = [
        (f"desc-{desc}", raw)
        for desc in ("filtered", "resampled", "errts")
        if (raw := _load_stage_raw(nirs_dir, sqm_label, desc, subject, errors)) is not None
    ] or None
    # measured under every label, but the views and pages are named by a slug of it; two
    # labels sharing one would overwrite each other, so the record keeps both and none is drawn
    if by_condition:
        try:
            refuse_colliding_labels(list(record.get("by_condition") or {}))
        except StageError as exc:
            errors.append(f"Per-condition pages: {exc}")
            logger.error("sub-%s | %s", subject, exc)
            by_condition = False
    sci_vars          = _section_sci(
                            raw_intensity, sci_scores, bad_channels, config,
                            windowed_section, subject, errors, figures_dir, fig_name)
    motion_vars       = _section_motion(
                            raw_long, raw_gvtd, gvtd_set, gvtd_blocks,
                            sci_scores, config, segments, subject, errors,
                            figures_dir, fig_name, windowed=windowed_section,
                            raw_before_motion=raw_before_motion,
                            raw_after_motion=raw_after_motion,
                            condition_spans=_record_windows(
                                record.get("by_condition") or {}) if by_condition else [],
                            imu=imu)
    motion_det_figs   = _motion_detail_figures(
                            raw_before_motion, raw_after_motion, subject, errors,
                            segments=segments,
                            corrected_by_set=motion_vars.get("corrected_by_set"),
                            spike_by_set=motion_vars.get("spike_by_set"),
                            gvtd_blocks=gvtd_blocks, imu=imu)
    # every condition's window goes into the run's own files, which is what lets the
    # condition pages point at them with a fragment instead of getting copies
    motion_det_vars   = _section_motion_detail(
                            motion_det_figs, subject, errors, figures_dir, fig_name,
                            condition_spans=_record_windows(record.get("by_condition") or {})
                            if by_condition else [])
    haemo_vars        = _section_haemo(raw_haemo, config, subject, errors, figures_dir,
                                       fig_name, l_freq=l_freq, h_freq=h_freq,
                                       raw_errts=raw_errts, psd_stages=psd_stages,
                                       record=record, sep_bands=sep_bands, mode=mode)
    carpet_stages = _carpet_stages(raw_haemo, psd_stages)
    carpet_vars = _section_stage_carpets(carpet_stages, roi_map, raw_gvtd, None,
                                         subject, errors, figures_dir, fig_name,
                                         raw_gvtd_after=raw_after_motion,
                                         gvtd_blocks=gvtd_blocks)
    # the trial window every epoch figure averages over. None on the config means the
    # report's own default, so an unset flag draws exactly what it always drew
    epoch_tmin = _EPOCH_TMIN if getattr(config, "epoch_tmin", None) is None else config.epoch_tmin
    epoch_tmax = _EPOCH_TMAX if getattr(config, "epoch_tmax", None) is None else config.epoch_tmax

    # --epoch-chunk-duration cuts the long annotations up once, here, rather than in each
    # figure: the epoch sections, the event timeline and the per-trial scoring then all read
    # one set of trials. A raw with no long annotation, or no flag, comes back unchanged.
    chunk = getattr(config, "epoch_chunk_duration", None)
    if chunk:
        n_events = len(raw_intensity.annotations)
        raw_intensity = chunk_annotations(raw_intensity, chunk)
        raw_haemo = chunk_annotations(raw_haemo, chunk)
        if after_haemo is not None:
            after_haemo = chunk_annotations(after_haemo, chunk)
        # said only when a block was long enough to be cut
        if len(raw_intensity.annotations) > n_events:
            _note(notes, subject, section_note("caveat.chunked_trials", chunk=chunk))

    # the "Raw Signal" section is the recording before anything was done to it, so its
    # figures come off desc-sci rather than the corrected desc-preproc the rest of the
    # report is built on. Same stage as the SCI/PSP windows and the SNR/CV numbers below.
    raw_haemo_uncorr  = _uncorrected_haemo(raw_before_motion, config, subject, errors)
    channel_det_vars  = _section_channel_detail(
                            raw_haemo_uncorr if raw_haemo_uncorr is not None else raw_haemo,
                            subject, errors, figures_dir, fig_name,
                            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
                            resp=(config.resp_l_freq, config.resp_h_freq),
                            sep_bands=sep_bands)
    channel_det_vars["channel_detail_stage"] = (
        "desc-sci" if raw_haemo_uncorr is not None else "desc-preproc")
    psd_det_vars      = _section_psd_detail(raw_haemo, subject, errors, figures_dir,
                                            fig_name, l_freq=l_freq, h_freq=h_freq,
                                            cardiac=(config.cardiac_l_freq, config.cardiac_h_freq),
                                            resp=(config.resp_l_freq, config.resp_h_freq),
                                            psd_stages=psd_stages, sep_bands=sep_bands)
    # graded by the windowed SCI, as each condition page's copy of this figure is
    sci_win_scores    = (((record.get("per_channel") or {}).get("raw") or {})
                         .get("sci_win_per_channel") or {})
    brain_vars        = _section_brain(
                            sci_win_scores, bad_channels, coords_head, good_mask, raw_intensity,
                            subject, errors, figures_dir, fig_name,
                            ch_names_brain=ch_names_brain,
                            sci_threshold=resolve_cutoffs(config)["sci"])
    # every figure in the epoch section on the denoised (bandpassed, pre-regression) haemo so
    # drift/noise is gone and the task response is intact; fall back to preproc only if no
    # post-processing ran. The unfiltered preproc would leave cardiac ripple on a curve read
    # for its shape, and make the figures under one heading describe two different stages.
    epoch_haemo       = after_haemo if after_haemo is not None else raw_haemo
    epoch_skip        = _no_epoch_reason(raw_haemo, epoch_tmin, epoch_tmax,
                                         single_trial=epoch_single_trial)
    # the window note only matters to figures that get drawn; asked before the skip it would
    # tell a reader to widen a window for a section that is not there
    if epoch_skip is None and getattr(config, "epoch_tmin", None) is None:
        outruns = _epoch_window_mismatch(raw_haemo, epoch_tmax)
        if outruns is not None:
            _note(notes, subject, section_note("caveat.epoch_window", tmin=epoch_tmin,
                                               tmax=epoch_tmax, outruns=outruns))

    if epoch_skip is not None:
        _note(notes, subject, section_note("caveat.epoch_skipped", reason=epoch_skip))
        epoch_vars       = {"epoch_preview_path": None, "epoch_preview_h": 0}
        trial_image_vars = {"trial_image_pairs": [], "trial_image_roi_pairs": []}
        topomap_vars     = {"evoked_topomap_path": None, "evoked_topomap_h": 0}
        trial_qc_vars    = {"trial_qc_path": None, "trial_qc_h": 0, "trial_qc_window": "",
                            "trial_qc_rows": []}
    else:
        epoch_vars        = _section_epoch_preview(epoch_haemo, subject, errors, figures_dir,
                                                   fig_name, epoch_tmin=epoch_tmin,
                                                   epoch_tmax=epoch_tmax,
                                                   sep_bands=sep_bands)
        trial_image_vars  = _section_trial_image(epoch_haemo, subject, errors, figures_dir,
                                                 fig_name, roi_map=roi_map,
                                                 roi_map_name=roi_map_name,
                                                 epoch_tmin=epoch_tmin,
                                                 epoch_tmax=epoch_tmax)
        topomap_vars      = _section_evoked_topomap(epoch_haemo, subject, errors, figures_dir,
                                                    fig_name, epoch_tmin=epoch_tmin,
                                                    epoch_tmax=epoch_tmax,
                                                    sep_bands=sep_bands)
        trial_qc_vars     = _section_trial_qc(raw_intensity, config, subject, errors,
                                              figures_dir, fig_name, notes)
    glm_vars          = _section_glm(design_matrix, glm_est, raw_haemo, subject, errors,
                                     figures_dir, fig_name, segments=segments)
    rest_vars         = _section_rest(alff_df, fc_df, subject, errors, figures_dir, fig_name,
                                      fc_hbr_df=fc_hbr_df,
                                      fc_seed=fc_seed, fc_roi=fc_roi, raw_haemo=raw_haemo,
                                      roi_map_name=roi_map_name, sep_bands=sep_bands)
    sqm_vars          = _section_sqm(sci_scores, bad_channels, subject, errors,
                                     out_dir=nirs_dir,
                                     sqm_label=sqm_label,
                                     sci_threshold=config.sci_threshold,
                                     psp_threshold=getattr(config, "psp_threshold", None))
    # a property of the high-pass, so stated on every run it filtered rather than measured
    if l_freq and any(lab == "desc-filtered" for lab, _ in (psd_stages or [])):
        _note(notes, subject, section_note("caveat.filter_edge", edge_s=round(1.0 / l_freq),
                                           l_freq=l_freq))

    unregistered = registration_note(registration_offset(raw_intensity))
    if unregistered:
        _note(notes, subject, unregistered)
    overlap = roi_overlap_note(roi_map)
    if overlap:
        _note(notes, subject, overlap)
    _note_separation(notes, subject, sqm_vars["sqm"], sqm_vars["channel_rows"],
                     short_channel_requested=bool(getattr(config, "short_channel", None)),
                     orphan_mm=separation_orphans(raw_intensity, sep_bands),
                     sep_bands=sep_bands)
    trigger_vars      = _section_trigger_timeline(raw_intensity, subject, errors,
                                                  figures_dir, fig_name)
    run_cutoffs       = resolve_cutoffs(config)
    ch_summary_vars   = _section_channel_summary(
                            sqm_vars["channel_rows"], subject, errors, figures_dir, fig_name,
                            sci_thresh=run_cutoffs["sci"], psp_thresh=run_cutoffs["psp"],
                            good_frac_thresh=run_cutoffs["good_frac"])
    cond_summary_vars = (_section_condition_summary(record, sci_scores, config, subject,
                                                    errors, figures_dir, fig_name)
                         if by_condition else
                         {"condition_summary_path": None, "condition_summary_h": 0})

    n_bad, n_total, bad_rate, badge_class = _bad_count(bad_channels, len(sci_scores))

    run_label_text = sqm_label or f"sub-{subject}"
    report_vars = dict(
        # the run names the page; that it is a QC report is what the reader opened. A
        # condition page appends its own label in `_condition_pages`, the way the hyper
        # post report does, so the two pages are told apart by their heading and their tab
        **page_vars(
            title=run_label_text,
            heading=run_label_text,
        ),
        **footer_vars(
            scope=f"sub-{subject}", errors=errors, notes=notes,
            nirs_dir=nirs_dir, label=sqm_label,
            provenance_path=provenance_path,
            methods=generate_methods_text(versions=versions, nirs_dir=nirs_dir,
                                          label=sqm_label),
            versions=versions,
        ),
        metric_summary=metric_summary,
        is_key_metric=is_key_metric,
        format_metric=format_metric,
        metric_class=metric_class,
        metric_label=metric_label,
        with_unit=with_unit,
        imu_stat_keys=IMU_STAT_KEYS,
        od_split_columns=measured_columns(
            OD_SPLIT_COLUMNS, sqm_vars["sqm_all"], sqm_vars["sqm_long"],
            sqm_vars["sqm_short"]),
        motion_split_columns=measured_columns(
            MOTION_SPLIT_COLUMNS, *(sqm_vars["motion_sets"] or {}).values()),
        subject=subject,
        run_label=sqm_label,
        run_entities={k: v for k, v in entities_of(sqm_label or "").items() if v},
        index_href=(report_name(f"sub-{subject}", desc="index") if sqm_label else None),
        run_command=run_command,
        # the numbers the page's prose and labels quote, from the constants that set them
        epoch_tmin=epoch_tmin, spike_ch_frac=SPIKE_CH_FRAC, sci_window_s=SCI_WINDOW_S,
        psp_window_s=PSP_WINDOW_S, cv_window_s=CV_WINDOW_S, gvtd_band=GVTD_MOTION_BAND,
        filled_by=_filled_by(filled_settings or [], mode),
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
        **cond_summary_vars,
        **carpet_vars,
        gvtd_set=gvtd_set,
        # the set GVTD was actually measured on, so the note says so on a per-condition page
        # too
        gvtd_channel_set=gvtd_set,
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
                              glm_vars, rest_vars, carpet_vars, cond_summary_vars),
                config=config, subject=subject,
                out_path=out_path, out_dir=nirs_dir, sqm_label=sqm_label,
                figures_dir=figures_dir, sci_scores=sci_scores, errors=errors,
                # closures rather than another ten parameters: both panels take a long
                # arg list that already exists here, and only the namer, the slice and
                # the view span differ per condition
                remake_sci=lambda cond_name, windowed_slice, sci_pc: _section_sci(
                    raw_intensity, sci_pc, bad_channels, config, windowed_slice,
                    subject, errors, figures_dir, cond_name),
                remake_motion=lambda cond_name, slug, span: {
                    **_section_motion(
                        raw_long, raw_gvtd, gvtd_set, gvtd_blocks, sci_scores, config,
                        segments, subject, errors, figures_dir, cond_name,
                        windowed=windowed_section,
                        raw_before_motion=raw_before_motion,
                        raw_after_motion=raw_after_motion,
                        skip_carpet=True, window=span),
                    **_condition_carpet(motion_vars, slug),
                },
                # the condition's own SCI over the run's rejections: a condition page grades
                # its stretch but rejects only what the run rejected
                remake_brain=lambda cond_name, sci_pc: _section_brain(
                    sci_pc, bad_channels, coords_head, good_mask,
                    raw_intensity, subject, errors, figures_dir, cond_name,
                    ch_names_brain=ch_names_brain, sci_threshold=resolve_cutoffs(config)["sci"]),
                remake_motion_detail=lambda slug: _condition_motion_detail(
                    motion_det_vars.get("motion_detail_pairs") or [], slug),
                remake_denoise_carpet=lambda cond_name, span: _section_stage_carpets(
                    carpet_stages, roi_map, raw_gvtd, span,
                    subject, errors, figures_dir, cond_name,
                    raw_gvtd_after=raw_after_motion, gvtd_blocks=gvtd_blocks),
                # --epoch-single-trial waives the "one row is not a comparison" floor
                # here as well, for the same reason it waives it on the epoch section
                remake_trial_qc=lambda cond_name, span: _condition_trial_qc(
                    trial_qc_vars.get("trial_qc_rows") or [], span,
                    subject, errors, figures_dir, cond_name,
                    min_trials=1 if epoch_single_trial else 2,
                    window=trial_qc_vars.get("trial_qc_window") or ""),
                # a run with nothing to epoch has no trial images on its own page either,
                # and the pass costs one figure per HbO channel per condition
                remake_trial_images=None if epoch_skip is not None else (
                    lambda spans: _section_condition_trial_images(
                        epoch_haemo, spans, subject, errors, figures_dir, fig_name,
                        roi_map=roi_map, roi_map_name=roi_map_name,
                        epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                        min_trials=1 if epoch_single_trial else 2)),
                remake_cropped=lambda cond_name, span: _cropped_sections(
                    span, cond_name, raw_haemo=raw_haemo, epoch_haemo=epoch_haemo,
                    raw_haemo_uncorr=raw_haemo_uncorr, raw_errts=raw_errts,
                    psd_stages=psd_stages, record=record, config=config,
                    l_freq=l_freq, h_freq=h_freq, sep_bands=sep_bands,
                    epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                    epoch_single_trial=epoch_single_trial, mode=mode,
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
# Nothing else belongs. The spectra and the correlation panels would all look like the
# condition's, which is why they are rebuilt per condition instead.
#
# Everything not listed has to carry the condition in its own `cond-` entity, which
# `_figure_leaks` reads off the name rather than listing the panels: a per-panel list would
# also pass a *different* condition's figure, worse than a run-wide one because the page
# would look per-condition and be the wrong condition. A `#slug` fragment counts as that
# name: the carpet and the per-channel motion figures hold every condition's window in one
# file, since their traces do not vary by condition and only the axes do.
_CONDITION_PAGE_DESCS = frozenset({"provenance", "timeseries", "heatmap", "trigger"})


def _figure_leaks(page: dict, label_slug: str) -> "list[str]":
    """Whole-run figures that survived onto a condition page, by value rather than by name.

    `_blanked` empties the section variables it is given, so a figure passed to the template
    some other way slips through it; checking the assembled values catches it.
    """
    leaks = []
    for key, value in page.items():
        # the per-channel panels arrive as a list of {pair, path, h}, so a check over plain
        # strings alone would not see the figures there are the most of
        for item in (value if isinstance(value, list) else [value]):
            path = item.get("path") if isinstance(item, dict) else item
            if not isinstance(path, str) or "figures/" not in path:
                continue
            name, _, fragment = path.rsplit("/", 1)[-1].partition("#")
            # a run-wide file addressed at one condition's window, which is how the motion
            # figures reach this page: the same traces, so one file carries every view
            if fragment == label_slug:
                continue
            entities = parse_path(name)
            # every per-condition figure carries the condition it is of
            if entities.get("condition") == label_slug:
                continue
            if entities.get("desc") in _CONDITION_PAGE_DESCS:
                continue
            leaks.append(f"{key}={name}")
    return leaks


def _cropped_sections(
    span: "tuple[float, float]",
    fig_name,
    *,
    raw_haemo, epoch_haemo, raw_haemo_uncorr, raw_errts, psd_stages, record, config,
    l_freq, h_freq, sep_bands, epoch_tmin, epoch_tmax, subject, errors, figures_dir,
    epoch_single_trial=False, mode=None,
) -> dict:
    """The panels that are safe to rebuild on a cropped copy, over one condition.

    Cropping is the right move for exactly these and the wrong one for the carpet and the
    SCI/PSP panel, and the line between them is whether the panel filters. It does not here:
    the haemoglobin timeseries are drawn as they are, the HbO-HbR correlation is a
    correlation over whatever samples it is given, and ``compute_psd`` is Welch, which
    segments and tapers but does not band-pass. So a cropped condition carries no filter
    edge that the whole run would not have had. A shorter span costs frequency resolution
    and averages fewer Welch segments, which makes the
    spectrum noisier and leaves it unbiased.

    Noisier up to a point. Below ``PSD_NFFT_CAP`` samples the cut lands on a coarser grid
    than the run instead, which is a different measurement rather than a noisier view of the
    same one. That is where the record stops writing the band scalars, so the figures stop
    there too.

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
    # the same floor the record holds the band scalars to, so a page cannot show a spectrum
    # for a number the record refused to write
    from nirspipe.qc.subject.condition_views import PSD_NFFT_CAP
    n_fft_floor = min(PSD_NFFT_CAP, len(raw_haemo.times))
    psd_ok = len(haemo.times) >= n_fft_floor
    if not psd_ok:
        logger.info("sub-%s | %s holds %d samples against a transform of %d; no spectra",
                    subject, fig_name.condition or "the cut", len(haemo.times),
                    n_fft_floor)
    out: dict = {"psd_too_short": not psd_ok}
    # the one panel handed a span rather than a cropped recording: it holds the stage list,
    # so only it can band-limit the whole run before cutting, which is the order its stage
    # comparison needs. See its docstring.
    out.update(_section_haemo(raw_haemo, config, subject, errors, figures_dir, fig_name,
                              l_freq=l_freq, h_freq=h_freq, raw_errts=raw_errts,
                              psd_stages=psd_stages, record=record, sep_bands=sep_bands,
                              crop=span, psd=psd_ok, mode=mode))
    if psd_ok:
        out.update(_section_psd_detail(haemo, subject, errors, figures_dir, fig_name,
                                       l_freq=l_freq, h_freq=h_freq, psd_stages=stages,
                                       sep_bands=sep_bands, **bands))
    # the bare span, unlike the epoch panels below: nothing in this section epochs,
    # so the pad would only show the neighbouring condition's last seconds
    out.update(_section_channel_detail(crop(raw_haemo_uncorr) or haemo, subject, errors,
                                       figures_dir, fig_name,
                                       sep_bands=sep_bands, **bands))
    # the same question the run's own page asks, re-asked on the crop: a condition page holds
    # one condition, so a design of one block per condition leaves it a single event and
    # nothing here to average
    epoch_src = crop(epoch_haemo, epoch_pad) or haemo
    if _no_epoch_reason(epoch_src, epoch_tmin, epoch_tmax,
                        single_trial=epoch_single_trial) is None:
        out.update(_section_epoch_preview(epoch_src, subject, errors,
                                          figures_dir, fig_name, epoch_tmin=epoch_tmin,
                                          epoch_tmax=epoch_tmax,
                                          sep_bands=sep_bands))
        out.update(_section_evoked_topomap(epoch_src, subject, errors, figures_dir,
                                           fig_name, epoch_tmin=epoch_tmin,
                                           epoch_tmax=epoch_tmax, sep_bands=sep_bands))
    else:
        out.update({"epoch_preview_path": None, "epoch_preview_h": 0,
                    "evoked_topomap_path": None, "evoked_topomap_h": 0})
    return out


def condition_page_name(run_label: str, condition: str, desc: "str | None" = None) -> str:
    """One condition's page of a run's report, from the condition's own label.

    ``("sub-01_task-rest", "game 1")`` -> ``"sub-01_task-rest_cond-game1_report.html"``
    ``("sub-01_task-rest", "game 1", desc="raw")``
        -> ``"sub-01_task-rest_cond-game1_desc-raw_report.html"``

    Module level and not inside the writer, because the subject index looks these up on
    disk: the two ends came apart once already and every per-condition link went dead. The
    raw viewer's pages take ``desc="raw"``, as its run page does, and the same slug its
    figures and URL fragments carry.
    """
    return report_name(run_label, condition=_pair_fname(condition), desc=desc)


def _condition_timeline(report_vars: dict) -> dict:
    """The run's event timeline, kept whole on a condition page.

    Run-wide and legitimate on every page, like the design matrix and for the same reason:
    it is the one panel that shows a condition in the context of the others, and what it is
    read for cannot be sliced. Its use is catching a condition that stopped being delivered
    partway through or a block that was started twice, and a timeline cut to one condition
    can show neither, since both are visible only against the conditions around them.

    It carries no per-condition figure, so nothing is recomputed: the same file the run's
    page points at. ``_CONDITION_PAGE_DESCS`` lets it past the leak check.
    """
    return {
        "trigger_timeline_path": report_vars.get("trigger_timeline_path"),
        "trigger_timeline_h": report_vars.get("trigger_timeline_h") or 0,
    }


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


def _record_windows(by_condition: dict) -> "list[tuple[str, float, float]]":
    """``(label, t0, t1)`` per annotated condition, in the record's own order.

    One expression, because the run's figures are handed these spans before the condition
    pages are built and the two must be the same windows.
    """
    return [(label, *entry["window_s"]) for label, entry in by_condition.items()]


def _windowed_slice(record: dict, windows: list, label: str) -> dict:
    """The record's ``windowed`` section with its matrices cut to one condition's columns.

    Only the six keys the SCI/PSP/CV panel reads. The rest of that section is spans and
    channel-averaged series measured over the run, and handing those to a per-condition
    panel would put run-wide stripes over per-condition columns.
    """
    windowed = record.get("windowed") or {}
    out: dict = {}
    for matrix_key, times_key in (("sci_matrix", "sci_times"), ("psp_matrix", "psp_times"),
                                  ("cv_matrix", "cv_times")):
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
    config: "PrepConfig",
    subject: str,
    out_path: Path,
    out_dir: Path | None,
    sqm_label: str | None,
    figures_dir: Path,
    sci_scores: dict,
    errors: list,
    remake_sci=None,
    remake_motion=None,
    remake_brain=None,
    remake_motion_detail=None,
    remake_denoise_carpet=None,
    remake_cropped=None,
    remake_trial_qc=None,
    remake_trial_images=None,
) -> None:
    """One subject-report page per annotated condition, read out of the quality record.

    Every number on these pages comes from the record's ``by_condition`` section, which
    :func:`~nirspipe.qc.subject.sqm_record.condition_sections` wrote. Nothing is measured here;
    a run whose record predates that section gets no pages rather than a second, possibly
    disagreeing, copy of the numbers.

    A panel gets here one of four ways, and which one is a property of the panel:

    - **read from the record**: the scalar panel and the channel table. The SCI/PSP/CV panel
      still cuts the stored matrices to this condition's columns, being a figure over the
      same windows those scalars were averaged over
    - **measured over the run, narrowed to the condition**: the GVTD carpet, the per-channel
      motion figures and the denoising carpet. Each derives something run-wide from what it
      is handed -- a filtered GVTD, a threshold, a per-channel z-scale -- so a cut recording
      would give every condition a scale no other condition could be read against. The
      per-channel motion figures are narrowed without being written again: the run's file
      holds every condition's window and this page links to one by URL fragment
    - **rebuilt on a cropped copy**: the haemoglobin panels, the spectra, the per-channel
      detail, the epoch preview and the topography. Safe because none of them filters; see
      :func:`_cropped_sections`
    - **rebuilt from the condition's own verdict**: the 3D quality views and the optode flat
      map, which carry that condition's SCI and its own rejected channels

    The trial half is a fifth way and the reason it took so long to build. Both panels have
    rows that are trials rather than windows, so they need the trials *inside* a condition,
    and a condition window is built from one annotation: on a block design that annotation
    is the only event in it and the panels would hold one row. They fill in on a blocked
    event-related design, where a block names the condition and shorter events sit inside it.
    The per-trial table is sliced out of the run's, and the trial images are drawn in one
    run-wide pass so every condition page shares a colour scale; neither is recomputed per
    page. A page with nothing to show says which of the two reasons applies.

    The event timeline is not among the blanked panels; it is kept whole, see
    :func:`_condition_timeline`.

    The GLM needs nothing rebuilt: one model is fitted over the whole recording and each
    condition is a column of it.

    The channel set is the run's throughout, since one set has to serve every condition. The
    *verdict* is not: each page screens on its own stretch.
    """
    if out_dir is None or sqm_label is None:
        logger.warning("sub-%s | no quality record location; no per-condition pages", subject)
        return
    record = read_record(_sqm_record_path(out_dir, sqm_label))
    by_condition = record.get("by_condition") or {}
    if not by_condition:
        logger.info("sub-%s | the record carries no by_condition section; no per-condition "
                    "pages", subject)
        return
    cutoffs = resolve_cutoffs(config)
    # the windows the record was written against, so the panels that still slice a matrix
    # here cut the same columns the stored scalars were averaged over
    windows = _record_windows(by_condition)

    blanked = _blanked(section_vars)
    # the run and its conditions are one set of pages, so each carries the whole strip and
    # reaches any other in a click, as a dyad's condition pages do
    def _page_name(label: "str | None") -> str:
        if label is None:
            return out_path.name
        return condition_page_name(out_path.stem.removesuffix("_report"), label)

    nav_pages = [(None, "Whole run")] + [(lab, lab) for lab in by_condition]

    # one pass for every condition, so the panels land on a shared colour scale; per page it
    # would be one scale each and the pages are read against each other
    trial_images = remake_trial_images(windows) if remake_trial_images is not None else {}
    for label, entry in by_condition.items():
        sliced = entry.get("per_channel") or {}
        # the header counts this condition's own assessment; Status stays the run's
        cond_bad = sorted(entry.get("bad_channels") or ())
        # the long set, as the run's own scalar panel is, so the two pages compare
        scalars = condition_verdict_view(entry)
        haemo_by_set = entry.get("haemo_by_set") or {}
        t0, t1 = entry["window_s"]
        span = (t0, t1)
        slug = _pair_fname(label)
        # one namer per condition, bound to it, so every panel below writes this page's own
        # file without anybody appending a slug to a bare panel name
        cond_name = figure_namer(sqm_label, slug)
        cropped: dict = {}
        if remake_cropped is not None:
            cropped = remake_cropped(cond_name, span)
        rows = _condition_channel_rows(record, entry, sci_scores,
                                       set(report_vars.get("bad_channels") or ()))
        cells = format_rows(rows, cutoffs["sci"], psp_threshold=cutoffs["psp"])
        summary = _section_channel_summary(
            rows, subject, errors, figures_dir, cond_name, cutoffs["sci"],
            psp_thresh=cutoffs["psp"], good_frac_thresh=cutoffs["good_frac"])
        # the SCI/PSP panel over this condition's columns: a real slice, since the figure
        # is handed its matrices and derives nothing from a recording
        panels: dict = {}
        if remake_sci is not None:
            panels.update(remake_sci(cond_name, _windowed_slice(record, windows, label),
                                     sliced.get("sci_win_per_channel") or {}))
        # the bad-segment zoom over this condition's own flagged segments, and the carpet
        # narrowed to it without being written again: measured over the run, viewed over it
        if remake_motion is not None:
            panels.update(remake_motion(cond_name, slug, span))
        # the denoising carpet is narrowed the same way but written again here, being a PNG
        # with no view to pick. The carpet and the per-channel motion figures are not: the
        # run's files carry every condition's window and this page addresses one by fragment
        if remake_brain is not None:
            panels.update(remake_brain(cond_name, sliced.get("sci_win_per_channel") or {}))
        if remake_motion_detail is not None:
            panels.update(remake_motion_detail(slug))
        if remake_denoise_carpet is not None:
            panels.update(remake_denoise_carpet(cond_name, span))
        # the trial half: rows are this condition's own trials, taken from the run's table
        # and from the run-wide image pass rather than measured again
        if remake_trial_qc is not None:
            panels.update(remake_trial_qc(cond_name, span))
        panels.update(trial_images.get(label) or {})
        panels.update(cropped)

        od_by_set = entry.get("od_by_set") or {}
        cond_motion_sets = entry.get("motion_by_set") or {}
        # The GLM is already per condition and needs nothing rebuilt: one model is fitted
        # over the whole recording and each condition is a contrast of it, so this page
        # keeps its own activation figure out of the set the run rendered. The design
        # matrix stays whole, because every condition is a column of that one model and a
        # per-condition version of it would be a different model.
        panels.update(_condition_glm(report_vars, label))
        panels.update(_condition_timeline(report_vars))
        page = {
            **report_vars, **blanked, **summary, **panels,
            "nav_links": [{"label": text, "href": _page_name(lab),
                           "current": lab == label} for lab, text in nav_pages],
            **_failing_summary(cond_bad, report_vars.get("n_total") or 0),
            "sqm": scalars,
            "channel_rows": rows,
            "channel_cells": cells,
            "channel_blocks": separation_blocks(cells, separation_bands(config)),
            # a condition has no whole-run SCI: that estimate has no windows to select from
            "channel_columns": channel_columns(("separation", "sci_whole")),
            "sqm_all": od_by_set.get("all") or {},
            "sqm_long": od_by_set.get("long") or {},
            "sqm_short": od_by_set.get("short") or {},
            "hb_all": haemo_by_set.get("all") or {},
            "hb_long": haemo_by_set.get("long") or {},
            "hb_short": haemo_by_set.get("short") or {},
            "sqm_split": bool(od_by_set.get("short")),
            "hb_split": bool(haemo_by_set.get("short")),
            "motion_sets": cond_motion_sets,
            "motion_split": bool(cond_motion_sets.get("long")
                                 and cond_motion_sets.get("short")),
            # WHOLE_RUN_ONLY_COLUMNS says which are dropped rather than left blank in all
            # three rows; low-frequency drift goes with them, measuring the span it is shown
            # rather than the recording
            "od_split_columns": CONDITION_OD_SPLIT_COLUMNS,
            "motion_split_columns": tuple(
                (key, text) for key, text in MOTION_SPLIT_COLUMNS
                if key not in WHOLE_RUN_ONLY_COLUMNS),
            # `page_heading` and `page_title` are what the shell reads; a `heading` key
            # here reached nothing, so every condition page carried the run's own title
            "page_heading": f"{report_vars['page_heading']}  \u00b7  {label}",
            "page_title": f"{report_vars['page_title']}  \u00b7  {label}",
            "condition_label": label,
            # what keeps this page's rating keys out of the run's; see the template's `_rk`
            "condition_slug": slug,
            "index_href": out_path.name,
        }
        leaks = _figure_leaks(page, _pair_fname(label))
        if leaks:
            # loud rather than silent: a run-wide figure under per-condition numbers reads
            # as that condition's, and nothing on the page would say otherwise
            logger.warning("sub-%s | condition %s still points at run-wide figures (%s); "
                           "they are being dropped", subject, label, ", ".join(leaks))
            page = {**page, **{k.split("=")[0]: None for k in leaks}}
        out = out_path.with_name(_page_name(label))
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
        mne_path = out_path.with_name(
            report_name(out_path.stem.removesuffix("_report"), desc="mne"))
        report.save(str(mne_path), overwrite=True, open_browser=False, verbose=False)
        logger.info("sub-%s | MNE report saved: %s", subject, mne_path)
    except Exception as e:
        errors.append(f"MNE report: {e}")
        logger.warning("sub-%s | MNE report failed: %s", subject, e)


