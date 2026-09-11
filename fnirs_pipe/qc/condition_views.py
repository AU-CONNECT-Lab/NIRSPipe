"""One QC view per condition, sliced out of the whole-run pass.

A recording preprocessed whole holds its conditions as annotations, so a per-condition view
is a column selection out of the windowed metrics the run already computed, never a cut of
the recording. The difference is not only cost. A condition cut into its own file is
filtered against its own two edges and lands on a window grid starting at its own onset, so
its numbers are comparable neither with the other conditions nor with the run. See
:func:`~fnirs_pipe.qc.metrics.windowed.condition_window_means`.

Each view screens on its own stretch, so a channel coupled through one condition and loose
through another is named in the one it was loose in. That is what a per-condition page is
for. It is a *view*, not what the run did: the recording was processed under the run's
verdict, and the run's own page carries it, one click away through the run index.
"""

from __future__ import annotations

import re

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.condition_views")

# what a per-condition view can honestly fill, because each has a windowed series behind it
SLICEABLE = ("sci_per_channel", "psp_per_channel", "good_frac_per_channel",
             "cv_per_channel", "snr_per_channel")

# and what it therefore has to drop, so no column mixes two time scopes, and a column added
# later shows a gap instead of a whole-run number under a condition's heading.
# `hbo_hbr_corr_per_channel` is dropped but recoverable: `with_condition_corr` puts back a
# value measured on the cut, and a failed recompute leaves dashes rather than the run's.
UNSLICEABLE = ("cp_per_channel", "temporal_derivative_variance",
               "hbo_hbr_corr_per_channel", "cnr_per_channel",
               # the spike mask is per sample, but only its whole-run share per channel is
               # stored, so there is nothing on disk to count over one condition's window
               "spike_pct_per_channel")

# the scalars a condition can be given, in print order. Short of the run's list on purpose:
# flat channels, mean amplitude and the spike count have no windowed series to slice, and
# the GVTD threshold is a mode of the whole run's histogram by definition, so a condition
# is counted against the run's line rather than given one of its own.
#
# The four GVTD means come off the stored per-window series and are therefore the corrected
# file, the stage that series is measured on; the spike and correction-footprint shares come
# off spans found on the uncorrected file. Two stages in one list, which is why the report
# names the stage on each row rather than printing nine bare numbers.
COND_SCALAR_KEYS = ("sci_mean", "good_frac_mean", "psp_mean",
                    "gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95",
                    "cv_mean", "snr_mean",
                    "gvtd_pct_above_thresh", "gvtd_num_above_thresh",
                    "spike_pct_frames", "spike_num_frames",
                    "motion_corrected_pct", "motion_corrected_num",
                    "motion_corrected_n_segments",
                    "channel_retention_rate")

# mne's compute_psd caps n_fft here, so a shorter cut lands on a coarser grid than the run.
# One floor for the record and the report, or a page shows a figure for a number it refused.
PSD_NFFT_CAP = 2048


def span_share(spans, t0: float, t1: float) -> "float | None":
    """Share of ``[t0, t1)`` covered by ``(onset, duration)`` spans, or None without a window.

    ::

        [(5.0, 2.0), (20.0, 4.0)], 0.0, 10.0  ->  0.2

    The spans come from a boolean the whole run set, so restricting them to a window counts
    the run's own verdict over that stretch rather than re-deciding it there.
    """
    if t1 <= t0:
        return None
    covered = 0.0
    for onset, duration in spans or []:
        covered += max(0.0, min(t1, float(onset) + float(duration)) - max(t0, float(onset)))
    return covered / (t1 - t0)


def condition_stem(stem: str, label: str | None, index: int) -> str:
    """A per-condition file's stem, by the rule ``fnirs-prep crop`` already uses.

    ::

      condition_stem("sub-01_task-full_desc-raw_nirs", "game1", 1)
      -> "sub-01_task-game1_desc-raw_nirs"
      condition_stem("sub-01_task-full_desc-raw_nirs", None, 3)
      -> "sub-01_task-full_desc-raw_nirs_seg-03"

    A labelled segment takes the label as its ``task-`` entity and an unlabelled one appends
    ``seg-NN``, which is what :func:`fnirs_pipe.pipeline.crop.crop_recording` does and for
    the reason recorded there: the segments become separate tasks of a valid dataset rather
    than one task nothing can tell apart.

    Unlike crop, the label here comes from an annotation rather than from a table the user
    wrote, so it is reduced to the alphanumerics a BIDS entity allows. ``condition_windows``
    numbers a repeated description ``desc#1``, ``desc#2``, and a ``#`` cannot go in a
    filename.
    """
    safe = re.sub(r"[^a-zA-Z0-9]", "", label or "")
    if not safe:
        return f"{stem}_seg-{index:02d}"
    return re.sub(r"task-[^_]+", f"task-{safe}", stem)


def condition_stems(stem: str, labels: "list[str]") -> "list[str]":
    """:func:`condition_stem` over a run's labels, with collisions broken by index.

    Stripping the non-alphanumerics can map two labels onto one stem ("game-1" and
    "game 1"), which would leave two views writing to one file and the second silently
    winning. A collision keeps the first and suffixes the rest.
    """
    out: list[str] = []
    seen: set[str] = set()
    for i, label in enumerate(labels, start=1):
        cand = condition_stem(stem, label, i)
        if cand in seen:
            logger.warning("condition %r collides with an earlier label on %s; "
                           "suffixing it", label, cand)
            cand = f"{cand}_seg-{i:02d}"
        seen.add(cand)
        out.append(cand)
    return out


def slice_record(record_view: dict, sliced: "dict[str, dict[str, float]]") -> dict:
    """A copy of the record whose sliceable per-channel metrics come from one condition.

    ``sliced`` is ``{"sci_per_channel": {ch: v}, ...}`` for this condition. Every section's
    per-channel dict is rebuilt from it, restricted to the channels that section already
    described, so a short channel's row stays in the short section and a long channel's in
    the long one; :func:`~fnirs_pipe.qc.channel_table.channel_rows` reads the sections and
    would otherwise put every channel in both.

    The unsliceable keys are dropped rather than carried over. Scalars are left alone: the
    caller replaces those it has condition values for.
    """
    per_channel = record_view.get("per_channel") or {}
    out_pc: dict[str, dict] = {}
    for section, metrics in per_channel.items():
        kept = {k: v for k, v in (metrics or {}).items()
                if k not in SLICEABLE and k not in UNSLICEABLE}
        for key in SLICEABLE:
            had = (metrics or {}).get(key)
            if not had:
                continue
            values = sliced.get(key) or {}
            kept[key] = {ch: values[ch] for ch in had if ch in values}
        out_pc[section] = kept
    return {**{k: v for k, v in record_view.items() if k != "per_channel"},
            "per_channel": out_pc}


def with_condition_corr(record: dict, corr_per_channel: "dict[str, float]") -> dict:
    """The record with ``preproc``'s HbO-HbR correlation replaced by the condition's own.

    ``channel_rows`` reads that column off the whole-file ``preproc`` section, so that is
    where a value measured on the cut has to land. An empty dict leaves the column dashed.
    """
    if not corr_per_channel:
        return record
    per_channel = {**(record.get("per_channel") or {})}
    per_channel["preproc"] = {**(per_channel.get("preproc") or {}),
                              "hbo_hbr_corr_per_channel": dict(corr_per_channel)}
    return {**record, "per_channel": per_channel}


def condition_haemo_scalars(haemo, errts, n_fft_floor: int, bands: dict) -> dict:
    """Scalars a condition can be given by recomputing on its own cut, not by slicing.

    ``haemo`` and ``errts`` are already cropped to the condition. None of these filters, so a
    cut carries no edge the whole run would not have had; ``compute_psd`` is Welch, which
    segments and tapers. What a cut does change is the spectrum's resolution, and only once
    the cut is shorter than the transform: below ``n_fft_floor`` samples the band metrics are
    computed over a coarser frequency grid than the run's and are left out instead.

    ::

        3053 samples, floor 2048  ->  band metrics included
        1200 samples, floor 2048  ->  band metrics absent

    Returns the scalars plus ``hbo_hbr_corr_per_channel``, measured on the same cut so the
    caller does not pay for it twice.
    """
    from fnirs_pipe.qc.metrics.haemo import (
        _retention_metrics, _spectral_metrics, gcor_metrics, haemo_quality_metrics,
    )

    out: dict = {}
    # gcor's before/after is stored under _prereg / _postreg rather than as an _errts suffix,
    # and those two stages are exactly this cut's haemo and errts, so it fills the same pair
    # the run's page prints instead of a lone number beside it
    for raw, suffix, gcor_suffix in ((haemo, "", "_prereg"), (errts, "_errts", "_postreg")):
        if raw is None:
            continue
        gcor = gcor_metrics(raw)
        out.update({f"{k}{gcor_suffix}": v for k, v in gcor.items()})
        if suffix == "":
            out.update(gcor)          # the plain key too, for a run with no errts stage
        quality = haemo_quality_metrics(raw)
        if "hbo_hbr_corr_mean" in quality:
            out[f"hbo_hbr_corr_mean{suffix}"] = quality["hbo_hbr_corr_mean"]
        # rides along because the caller already has it; `metric_rows` reads a fixed key
        # list, so a per-channel dict here never reaches the scalar panel
        if suffix == "" and quality.get("hbo_hbr_corr_per_channel"):
            out["hbo_hbr_corr_per_channel"] = quality["hbo_hbr_corr_per_channel"]
    if haemo is not None:
        out.update(_retention_metrics(haemo))
        # the band metrics live at preproc only: after the bandpass they measure the filter
        if len(haemo.times) >= n_fft_floor:
            out.update(_spectral_metrics(haemo, bands["cardiac"][0], bands["cardiac"][1],
                                         bands["resp"][0], bands["resp"][1]))
    return out


def zoom_to_condition(figure, t0: float, t1: float):
    """A time-axis figure viewing only one condition, without recomputing it.

    ::

      zoom_to_condition(carpet_figure_dict, 543.7, 1443.7)

    This is how the carpet, the GVTD trace and the raw timeseries go per condition. Handing
    a cropped recording to the figure builders instead would recompute what they derive
    internally, and all three of those derivations are run-wide on purpose:
    ``carpet_gvtd_figure`` filters GVTD at 0.01-0.5 Hz, z-scores each channel, and picks a
    threshold off the distribution. On a 900 s piece the 0.01 Hz filter is the same
    mismatch the old 0.02 Hz workaround existed for, each condition would get its own
    per-channel mean and SD so no two carpets could be read against each other, and each
    would get its own threshold line. That figure's own docstring makes this argument for
    the corrected-versus-uncorrected pair; it holds the same way across conditions.

    So the run is measured once and the view is narrowed. Every x axis in the layout is set,
    because the carpet is stacked subplots sharing a time axis and leaving one unset would
    show a panel at a different span from the one above it.

    Takes either a plotly figure or its dict form, since the two report paths hold different
    ones: the raw viewer inlines figure dicts, and the subject report keeps plotly objects
    to save as files. One function rather than an ``update_xaxes`` call at one call site and
    this loop at the other, which would be two spellings of one decision and free to drift.
    A figure object is narrowed in place and returned; a dict is returned narrowed as a copy,
    because the run's own dict is written out as well.
    """
    if hasattr(figure, "update_xaxes"):
        figure.update_xaxes(range=[float(t0), float(t1)], autorange=False)
        return figure
    layout = dict(figure.get("layout") or {})
    axes = [k for k in layout if k == "xaxis" or k.startswith("xaxis")]
    if not axes:
        logger.warning("figure has no x axis to narrow to %s-%s s", t0, t1)
        return figure
    for key in axes:
        layout[key] = {**(layout[key] or {}), "range": [float(t0), float(t1)],
                       "autorange": False}
    return {**figure, "layout": layout}


def condition_payloads(
    payload: dict,
    *,
    record_view: dict,
    sci_scores: "dict[str, float]",
    bad_channels: "set[str]",
    od_ch_names: "list[str]",
    channel_pairs: "list[str] | None",
    series: dict,
    windows: "list[tuple[str, float, float]]",
    good_frac_by_condition: "dict[str, dict[str, float]]",
    sci_threshold: float,
    cutoffs: "dict[str, float]",
    gvtd_series: "tuple | None" = None,
) -> "list[tuple[str, dict]]":
    """One viewer payload per condition, built by slicing the run's own pass.

    ``payload`` is the run's payload, whose parts that do not vary by condition (the optode
    layout, the separation notes, the per-channel figure paths) are carried over untouched.
    Everything that does vary is rebuilt from ``series``, the windowed matrices
    :func:`~fnirs_pipe.qc.metrics.windowed.attach_windowed_series` returned, and from
    ``good_frac_by_condition``, which the run already counted per condition.

    The two time-axis figures are narrowed rather than redrawn, see
    :func:`zoom_to_condition`. The PSD is dropped: it is one spectrum rather than a
    time-by-frequency matrix, so there is nothing to slice, and recomputing it on a cut
    condition is the thing this module exists to avoid. What a per-condition report actually
    wants from it, whether the cardiac peak survived, is the sliced PSP row.

    Each payload carries three notes, because each is a way a reader could be misled: what
    the view is, that the rejected channels are the run's verdict and not this condition's,
    and that the columns with no windowed series behind them are absent rather than zero.
    """
    from fnirs_pipe.qc.boilerplate.vocabulary import metric_rows
    from fnirs_pipe.qc.channel_table import (
        channel_rows, format_rows, heatmap_args, pair_rows, separation_blocks,
    )
    from fnirs_pipe.qc.figures import build_sci_psp_figure, channel_quality_heatmap
    from fnirs_pipe.qc.metrics.windowed import condition_window_means

    def _per_channel(matrix, times, window) -> "dict[str, float]":
        if matrix is None or times is None:
            return {}
        vals = condition_window_means(matrix, times, [window]).get(window[0])
        if vals is None:
            return {}
        return {ch: float(vals[i]) for i, ch in enumerate(od_ch_names) if i < len(vals)}

    out: list[tuple[str, dict]] = []
    for window in windows:
        label = window[0]
        sci_pc = _per_channel(series.get("sci_matrix"), series.get("sci_times"), window)
        psp_pc = _per_channel(series.get("psp_matrix"), series.get("psp_times"), window)
        frac_pc = good_frac_by_condition.get(label) or {}
        if not (sci_pc or psp_pc or frac_pc):
            logger.warning("condition %s has no windowed metric inside it; no view written",
                           label)
            continue

        cond_record = slice_record(record_view, {
            "sci_per_channel": sci_pc,
            "psp_per_channel": psp_pc,
            "good_frac_per_channel": frac_pc,
        })
        rows = channel_rows(cond_record, sci_scores, bad_channels)
        pair_cells = format_rows(pair_rows(rows, channel_pairs or None), sci_threshold,
                                 name_key="pair", psp_threshold=cutoffs["psp"])
        scalars = {
            "sci_mean":       _mean_or_none(sci_pc.values()),
            "psp_mean":       _mean_or_none(psp_pc.values()),
            "good_frac_mean": _mean_or_none(frac_pc.values()),
            "gvtd_mean":      _gvtd_mean(gvtd_series, window),
        }

        d = dict(payload)
        d["sqm"] = {
            "rows": metric_rows(scalars, COND_SCALAR_KEYS, skip_missing=True),
            # no All/Long/Short split: it would need the same slice per channel set, and the
            # run's split is one file away in the full report
            "split": {},
            "channel_set": "every channel",
        }
        d["channels"] = {
            "pairs":  pair_cells,
            "blocks": separation_blocks(pair_cells),
            # carried over: a separation is a property of the montage, not of a condition
            "notes":  (payload.get("channels") or {}).get("notes") or [],
        }
        d["ts"] = _narrowed_ts(payload.get("ts"), window)
        d["carpet_gvtd"] = _narrowed(payload.get("carpet_gvtd"), window)
        d["psd"] = {}
        d["trial_qc"] = {}
        d["sci_psp"] = _condition_sci_psp(build_sci_psp_figure, sci_pc, psp_pc,
                                          bad_channels, sci_threshold, series, window)
        d["ch_summary"] = {
            "figure": channel_quality_heatmap(sci_thresh=sci_threshold,
                                              **heatmap_args(rows)).to_dict(),
        }
        d["notes"] = list(payload.get("notes") or []) + [
            f"This view describes {label} only. Its numbers are sliced out of the whole "
            f"recording's windowed pass, not measured on a cut of it, so they sit on the "
            f"same window grid and the same filter as every other condition and as the run.",
            "The verdict here is this condition's own, screened on its windows against the "
            "run's line. The recording was processed under the run's verdict, not this one; "
            "the run's page carries it.",
            "The PSD is absent rather than zero: it is one spectrum rather than a matrix, "
            "so a per-condition value would have to be recomputed on a cut. The full report "
            "has it for the run.",
        ]
        out.append((label, d))
    return out


def _mean_or_none(values) -> "float | None":
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _gvtd_mean(gvtd_series, window) -> "float | None":
    """The condition's GVTD, sliced off the run's windowed trace.

    ``gvtd_series`` is ``(per_window, centre_times)``. The 0.01-0.5 Hz filter behind that
    trace ran once over the whole recording, which is why this slices rather than
    recomputing: a 900 s piece filtered at 0.01 Hz is the same length mismatch the old
    0.02 Hz high-pass workaround existed for.
    """
    from fnirs_pipe.qc.metrics.windowed import condition_window_means
    if not gvtd_series or gvtd_series[0] is None or gvtd_series[1] is None:
        return None
    val = condition_window_means(gvtd_series[0], gvtd_series[1], [window]).get(window[0])
    return None if val is None else float(val)


def _narrowed(inline: "dict | None", window) -> dict:
    if not inline or not inline.get("figure"):
        return {}
    return {**inline, "figure": zoom_to_condition(inline["figure"], window[1], window[2])}


def _narrowed_ts(inline: "dict | None", window) -> dict:
    """The raw-signal panel narrowed, with its own bounds and markers moved to match.

    ``t_start`` / ``t_end`` travel in this payload for the GUI, which reads them rather than
    the figure's axis, so leaving them at the run's bounds would give two views of one
    condition different spans.
    """
    if not inline:
        return {}
    out = _narrowed(inline, window)
    if not out:
        return {}
    markers = [m for m in (inline.get("markers") or [])
               if window[1] <= float(m.get("onset", -1)) <= window[2]]
    return {**out, "t_start": float(window[1]), "t_end": float(window[2]),
            "markers": markers}


def _condition_sci_psp(build, sci_pc, psp_pc, bad_channels, sci_threshold, series, window):
    """The SCI/PSP panel over one condition's columns, a real slice of both matrices.

    Unlike the carpet, this figure derives nothing internally: it is handed the matrices and
    the per-channel scalars, so both are replaced by the condition's. Passing the
    condition's matrices with the run's scalars would put a per-condition heatmap beside a
    whole-run bar, which is the mistake this pairing exists to prevent.
    """
    import numpy as np

    from fnirs_pipe.qc.metrics.windowed import _in_scope
    matrix, times = series.get("sci_matrix"), series.get("sci_times")
    if matrix is None or times is None:
        return {}
    centers = np.asarray(times)
    if centers.ndim == 2 and centers.shape[1] == 2:
        centers = centers.mean(axis=1)
    keep = _in_scope(centers, [window])
    if not keep.any():
        return {}
    psp_matrix, psp_times = series.get("psp_matrix"), series.get("psp_times")
    fig = build(
        sci_pc, psp_pc, bad_channels, sci_threshold,
        sci_matrix=np.asarray(matrix)[:, keep],
        sci_win_times=np.asarray(times)[keep],
        psp_matrix=None if psp_matrix is None else np.asarray(psp_matrix)[:, keep],
        psp_win_times=None if psp_times is None else np.asarray(psp_times)[keep],
    )
    return {"figure": fig.to_dict()}


def condition_slices_from_record(
    record: dict,
    ch_names: "list[str]",
    windows: "list[tuple[str, float, float]]",
    sci_cutoff: float,
    psp_cutoff: float,
) -> "dict[str, dict[str, dict[str, float]]]":
    """Per-condition per-channel metrics read out of a quality record, nothing recomputed.

    ::

      {"game1": {"sci_per_channel": {...}, "psp_per_channel": {...},
                 "good_frac_per_channel": {...}}, ...}

    The record's ``windowed`` section already stores the channel-by-window SCI and PSP
    matrices and their window bounds, so a condition is a column selection out of what the
    run measured once. This is the whole reason the subject report can go per condition
    without touching the recording again.

    The coupled-window share is rebuilt here rather than read, because the record stores
    only its whole-run value. It goes through
    :func:`~fnirs_pipe.qc.metrics.windowed.coupled_mask_from_matrices`, the same AND the
    screening applied, so a condition's share cannot disagree with the verdict the run was
    screened by. Both cutoffs have to be the run's own; passing anything else produces a
    number no channel was judged against.

    Returns an empty dict when the record predates the stored matrices, which is the honest
    answer: the values cannot be recovered from the whole-run scalars.
    """
    from fnirs_pipe.qc.metrics.windowed import (
        condition_window_means, coupled_mask_from_matrices,
    )

    windowed = record.get("windowed") or {}
    sci_matrix, psp_matrix = windowed.get("sci_matrix"), windowed.get("psp_matrix")
    sci_times, psp_times = windowed.get("sci_times"), windowed.get("psp_times")
    cv_matrix, cv_times = windowed.get("cv_matrix"), windowed.get("cv_times")
    if not sci_matrix or sci_times is None:
        logger.warning("the quality record carries no windowed matrices, so no "
                       "per-condition view can be built from it")
        return {}

    def _named(vals) -> "dict[str, float]":
        return {ch: float(vals[i]) for i, ch in enumerate(ch_names) if i < len(vals)}

    out: dict[str, dict[str, dict[str, float]]] = {}
    sci_by_cond = condition_window_means(sci_matrix, sci_times, windows)
    psp_by_cond = ({} if not psp_matrix or psp_times is None
                   else condition_window_means(psp_matrix, psp_times, windows))
    mask = coupled_mask_from_matrices(sci_matrix, psp_matrix, sci_cutoff, psp_cutoff) \
        if psp_matrix else None
    frac_by_cond = ({} if mask is None
                    else condition_window_means(mask.astype(float), sci_times, windows))
    # records written before CV was windowed have no matrix; those pages leave it out
    cv_by_cond = ({} if not cv_matrix or cv_times is None
                  else condition_window_means(cv_matrix, cv_times, windows))

    for window in windows:
        label = window[0]
        if label not in sci_by_cond:
            continue
        out[label] = {
            "sci_per_channel": _named(sci_by_cond[label]),
            "psp_per_channel": _named(psp_by_cond[label]) if label in psp_by_cond else {},
            "good_frac_per_channel": (_named(frac_by_cond[label])
                                      if label in frac_by_cond else {}),
            "cv_per_channel": _named(cv_by_cond[label]) if label in cv_by_cond else {},
            # 1/CV, as the run's own pair is, so the two cannot disagree on one page
            "snr_per_channel": ({ch: 1.0 / v for ch, v in _named(cv_by_cond[label]).items()
                                 if v} if label in cv_by_cond else {}),
        }
    return out


def condition_scalars(sliced: "dict[str, dict[str, float]]",
                      gvtd_means: "dict[str, float] | None" = None,
                      *,
                      shares: "dict[str, float | None] | None" = None,
                      n_frames: "dict[str, int | None] | None" = None,
                      n_segments: "dict[str, int] | None" = None,
                      retention: "float | None" = None) -> "dict[str, float | None]":
    """The scalars a condition's panel prints, averaged over the channels it has.

    Only those with a windowed series or a stored span list behind them, see
    :data:`COND_SCALAR_KEYS`. A key the condition has no values for comes back None, which
    the metric registry prints as a dash rather than as a zero.

    ``gvtd_means`` are per-window series already averaged over the condition's columns;
    ``shares``, ``n_frames`` and ``n_segments`` are the same span lists counted three ways,
    because the report prints a fraction, a frame count and a segment count off one boolean.
    """
    gvtd_means = gvtd_means or {}
    shares = shares or {}
    n_frames = n_frames or {}
    n_segments = n_segments or {}
    out: "dict[str, float | None]" = {
        "sci_mean":       _mean_or_none((sliced.get("sci_per_channel") or {}).values()),
        "psp_mean":       _mean_or_none((sliced.get("psp_per_channel") or {}).values()),
        "good_frac_mean": _mean_or_none((sliced.get("good_frac_per_channel") or {}).values()),
        "cv_mean":        _mean_or_none((sliced.get("cv_per_channel") or {}).values()),
        "snr_mean":       _mean_or_none((sliced.get("snr_per_channel") or {}).values()),
        "channel_retention_rate": retention,
    }
    for key in ("gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95"):
        out[key] = gvtd_means.get(key)
    for key in ("gvtd_pct_above_thresh", "spike_pct_frames", "motion_corrected_pct"):
        out[key] = shares.get(key)
    for key in ("gvtd_num_above_thresh", "spike_num_frames", "motion_corrected_num"):
        out[key] = n_frames.get(key)
    out["motion_corrected_n_segments"] = n_segments.get("motion_corrected_n_segments")
    return out


def condition_set_scalars(
    sliced: "dict[str, dict[str, float]]",
    bad_channels: "set[str]",
    long_names: "list[str]",
    short_names: "list[str]",
) -> "dict[str, dict[str, float | None]]":
    """The condition's optical-density averages over each channel set, for the split table.

    ::

        sliced["sci_per_channel"] over 44 channels, 28 long and 16 short
        -> {"all": {...}, "long": {...}, "short": {...}}

    Every metric here is a mean over the row's channels, which is the whole of what a set
    means for them. GVTD is not one of them and does not belong here: it is an RMS *across*
    channels, so its sets are three separate measurements rather than three groupings of one,
    and it is reported in the motion panel where its before and after sit side by side.

    A set with no channels comes back with every value None rather than being left out, so
    the table keeps its three rows on a montage that has only long channels.
    """
    members = {"all": None, "long": set(long_names), "short": set(short_names)}
    out: dict = {}
    for set_name, keep in members.items():
        def _mean(metric: str, keep=keep) -> "float | None":
            values = sliced.get(metric) or {}
            return _mean_or_none([v for ch, v in values.items()
                                  if keep is None or ch in keep])

        frac = sliced.get("good_frac_per_channel") or {}
        in_set = [ch for ch in frac if keep is None or ch in keep]
        retention = (None if not in_set else
                     1.0 - sum(1 for ch in in_set if ch in bad_channels) / len(in_set))
        row = {
            "channel_retention_rate": retention,
            "sci_mean":       _mean("sci_per_channel"),
            "good_frac_mean": _mean("good_frac_per_channel"),
            "psp_mean":       _mean("psp_per_channel"),
            "snr_mean":       _mean("snr_per_channel"),
            "cv_mean":        _mean("cv_per_channel"),
        }
        out[set_name] = row
    return out

def span_counts(spans, t0: float, t1: float) -> "tuple[float | None, int]":
    """``span_share`` again, with the number of spans that touch the window.

    ::

        [(5.0, 2.0), (20.0, 4.0)], 0.0, 10.0  ->  (0.2, 1)

    The report prints a fraction and a segment count off the same boolean, and counting the
    spans here keeps the two from being derived in two places.
    """
    share = span_share(spans, t0, t1)
    n = sum(1 for onset, duration in (spans or [])
            if min(t1, float(onset) + float(duration)) > max(t0, float(onset)))
    return share, n
