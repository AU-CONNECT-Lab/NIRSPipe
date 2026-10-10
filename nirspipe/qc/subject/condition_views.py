"""One QC view per condition, sliced out of the whole-run pass.

A recording preprocessed whole holds its conditions as annotations, so a per-condition view
is a column selection out of the windowed metrics the run already computed, never a cut of
the recording, so its numbers stay on the run's filter and window grid. See
:func:`~nirspipe.qc.metrics.windowed.condition_window_means`.

Each view screens on its own stretch, so a channel coupled through one condition and loose
through another is named in the one it was loose in. That is what a per-condition page is
for. It is a *view*, not what the run did: the recording was processed under the run's
verdict, and the run's own page carries it, one click away through the run index.
"""

from __future__ import annotations

import math

import numpy as np

from nirspipe.io.naming import parse_path
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.metrics.imu import IMU_STAT_KEYS
from nirspipe.utils.logging import get_logger

logger = get_logger("qc.condition_views")

# what a per-condition view can honestly fill, because each has a windowed series behind it
SLICEABLE = ("sci_win_per_channel", "psp_per_channel",
             "good_frac_per_channel", "cv_per_channel", "snr_per_channel")

# and what it therefore has to drop, so no column mixes two time scopes, and a column added
# later shows a gap instead of a whole-run number under a condition's heading.
# `hbo_hbr_corr_per_channel` is dropped but recoverable: `with_condition_corr` puts back a
# value measured on the cut, and a failed recompute leaves dashes rather than the run's.
UNSLICEABLE = ("cp_per_channel", "temporal_derivative_variance",
               # the whole-run SCI is one correlation over the recording, with no windows
               "sci_per_channel",
               "hbo_hbr_corr_per_channel", "cnr_per_channel",
               # the spike mask is per sample, but only its whole-run share per channel is
               # stored, so there is nothing on disk to count over one condition's window
               "spike_pct_per_channel",
               # both vary over the recording and neither has a windowed series stored
               "mean_amp_per_channel", "motion_corrected_frac_per_channel")

# Keys that describe the montage rather than the recording, so the run's value is also the
# condition's. Named rather than left out of both lists above, which is the same silence a
# metric that should have been sliced would pass through.
TIME_INVARIANT = ("ch_dist_per_channel",)

# the scalars a condition can be given, in print order. Short of the run's list on purpose:
# flat channels, mean amplitude and the spike count have no windowed series to slice, and
# the GVTD threshold is a mode of the whole run's histogram by definition, so a condition
# is counted against the run's line rather than given one of its own.
#
# The two GVTD values come off the stored per-window series and are therefore the corrected
# file, the stage that series is measured on; the spike and correction-footprint shares come
# off spans found on the uncorrected file. Two stages in one list, which is why the report
# names the stage on each row rather than printing nine bare numbers.
COND_SCALAR_KEYS = ("sci_win_mean", "good_frac_mean", "psp_mean",
                    "gvtd_filt_mean", "gvtd_filt_p95",
                    "cv_mean", "snr_mean",
                    "gvtd_pct_above_thresh", "gvtd_num_above_thresh",
                    "spike_pct_frames", "spike_num_frames",
                    "motion_corrected_pct", "motion_corrected_num",
                    "motion_corrected_n_segments",
                    "channel_retention_rate", *IMU_STAT_KEYS)

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


def slice_record(record_view: dict, sliced: "dict[str, dict[str, float]]") -> dict:
    """A copy of the record whose sliceable per-channel metrics come from one condition.

    ``sliced`` is ``{"sci_win_per_channel": {ch: v}, ...}`` for this condition. Every section's
    per-channel dict is rebuilt from it, restricted to the channels that section already
    described, so a short channel's row stays in the short section and a long channel's in
    the long one; :func:`~nirspipe.qc.common.channel_table.channel_rows` reads the sections and
    would otherwise put every channel in both.

    The unsliceable keys are dropped rather than carried over. Scalars are left alone: the
    caller replaces those it has condition values for.
    """
    per_channel = record_view.get("per_channel") or {}
    unclassified = {k for metrics in per_channel.values() for k in (metrics or {})
                    if k not in SLICEABLE and k not in UNSLICEABLE
                    and k not in TIME_INVARIANT}
    if unclassified:
        logger.warning("per-channel %s is in none of the three lists, so a condition page "
                       "is showing the whole run's value for it",
                       ", ".join(sorted(unclassified)))
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


def with_condition_corr(record: dict, corr_per_channel: "dict[str, float]",
                        section: str = "preproc") -> dict:
    """The record with ``section``'s HbO-HbR correlation replaced by the condition's own.

    ``channel_rows`` reads that column off the whole-file ``preproc`` section, or
    ``rawhaemo`` on a raw-only record, so that is where a value measured on the cut has to
    land. An empty dict leaves the column dashed.
    """
    if not corr_per_channel:
        return record
    per_channel = {**(record.get("per_channel") or {})}
    per_channel[section] = {**(per_channel.get(section) or {}),
                            "hbo_hbr_corr_per_channel": dict(corr_per_channel)}
    return {**record, "per_channel": per_channel}


def condition_haemo_scalars(haemo, errts, n_fft_floor: int, bands: dict,
                            filtered=None) -> dict:
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
    from nirspipe.qc.metrics.haemo import (
        _retention_metrics, _spectral_metrics, gcor_metrics, haemo_quality_metrics,
    )

    out: dict = {}
    # `filtered` on the before side, not `haemo`, so the pair isolates the regression; the
    # run's own page pairs the same two stages under these names
    for raw, suffix, gcor_suffix in ((haemo, "", None),
                                     (filtered, None, "_prereg"),
                                     (errts, "_errts", "_postreg")):
        if raw is None:
            continue
        gcor = gcor_metrics(raw)
        if gcor_suffix:
            out.update({f"{k}{gcor_suffix}": v for k, v in gcor.items()})
        if suffix == "":
            out.update(gcor)          # the plain key too, for a run with no errts stage
        if suffix is None:
            continue
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

      zoom_to_condition(carpet_figure_dict, 600.0, 1500.0)

    This is how the carpet, the GVTD trace and the raw timeseries go per condition. A cropped
    recording would make the figure builders re-derive what they compute run-wide (the GVTD
    filter, each channel's z-scale, the threshold), so the run is measured once and the view
    is narrowed. Every x axis in the layout is set,
    because the carpet is stacked subplots sharing a time axis and leaving one unset would
    show a panel at a different span from the one above it.

    Takes either a plotly figure or its dict form, since the two report paths hold different
    ones: the raw viewer inlines figure dicts, and the subject report keeps plotly objects
    to save as files. One function for both, so the two paths cannot drift.
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


_SCALE_NOTE = "qc-scale-note"


def _trace_x(trace):
    """A trace's timestamps, whether it carries them or a start and a step."""
    y = getattr(trace, "y", None)
    if y is None:
        return None
    x = getattr(trace, "x", None)
    if x is not None:
        return np.asarray(x, dtype=float)
    x0, dx = getattr(trace, "x0", None), getattr(trace, "dx", None)
    if x0 is None or dx is None:
        return None
    return float(x0) + float(dx) * np.arange(len(y))


def rescale_y_to_window(figure, t0: float, t1: float):
    """Re-fit every y axis of a time-axis figure to what is inside ``t0``-``t1``.

    ::

      rescale_y_to_window(motion_detail_fig, 22.4, 322.4)

    The companion of :func:`zoom_to_condition`, which narrows the view along time and leaves
    the y axes pinned to the whole run, where a quiet condition's row reads as a flat line.
    The row labels carry this window's maximum and the run's beside it, so a reader is told
    which scale they are on.

    Shaded spans follow the new range instead of setting it: they are frames drawn to the
    row's height, not measurements. A row with no trace inside the window is left alone.
    """
    if not hasattr(figure, "update_yaxes"):
        return figure
    spec = window_view_spec(figure, t0, t1)
    # the caller narrows one figure object once per condition, so the last condition's notes
    # have to go before this one's are written or every page carries every page's
    figure.layout.annotations = tuple(
        a for a in (figure.layout.annotations or ()) if a.name != _SCALE_NOTE)
    for key, (floor, top) in spec["y"].items():
        figure.layout[key].update(range=[floor, top], autorange=False)
    for band in spec["bands"]:
        tr = figure.data[band["i"]]
        floor, top = band["lo"], band["hi"]
        tr.y = [None if v is None else (floor if v <= floor else top) for v in tr.y]
    for note in spec["notes"]:
        figure.add_annotation(
            x=0.996, xref=f"{note['xref']} domain", y=0.97, yref=f"{note['yref']} domain",
            text=note["text"], showarrow=False, xanchor="right", yanchor="top",
            font=dict(size=7, color="#98a2ad"), name=_SCALE_NOTE,
        )
    return figure


def window_view_spec(figure, t0: float, t1: float) -> dict:
    """What :func:`rescale_y_to_window` would do to ``figure``, measured but not applied.

    ::

      window_view_spec(motion_detail_fig, 22.4, 322.4)
      -> {"x": [22.4, 322.4],
          "y": {"yaxis2": (0.0, 0.0031)},
          "bands": [{"i": 0, "lo": 0.0, "hi": 0.0031}],
          "notes": [{"yref": "y2", "xref": "x", "text": "max 0.0031 · run 0.017"}]}

    A figure written once per channel rather than once per condition carries a table of
    these and lets the page pick one by URL fragment, so the numbers have to come out
    separately from the figure they were measured on. ``rescale_y_to_window`` applies one,
    which is what keeps the saved-per-condition and the picked-at-load paths from drifting:
    there is one measurement and two ways of spending it.
    """
    out = {"x": [float(t0), float(t1)], "y": {}, "bands": [], "notes": []}
    if not hasattr(figure, "update_yaxes"):
        return out
    lines, bands = {}, {}
    for i, tr in enumerate(figure.data):
        axis = getattr(tr, "yaxis", None) or "y"
        group = bands if getattr(tr, "fill", None) == "toself" else lines
        group.setdefault(axis, []).append((i, tr))

    for axis, indexed in lines.items():
        traces = [tr for _i, tr in indexed]
        lo, hi, run_hi = np.inf, -np.inf, -np.inf
        x_axis = "x"
        for tr in traces:
            x = _trace_x(tr)
            if x is None:
                continue
            x_axis = getattr(tr, "xaxis", None) or x_axis
            y = np.asarray(tr.y, dtype=float)
            y = y[np.isfinite(y)]
            if y.size:
                run_hi = max(run_hi, float(y.max()))
            inside = np.asarray(tr.y, dtype=float)[(x >= t0) & (x <= t1)]
            inside = inside[np.isfinite(inside)]
            if inside.size:
                lo, hi = min(lo, float(inside.min())), max(hi, float(inside.max()))
        if not np.isfinite(lo) or hi <= lo:
            continue
        # a row pinned to zero (a magnitude: GVTD, |dOD/dt|) keeps that floor; an autoranged
        # row (OD, which goes negative) gets a floor fitted to the window
        key = "yaxis" + axis[1:]
        was = figure.layout[key].range if key in figure.layout else None
        if was is not None and float(was[0]) == 0.0:
            floor = 0.0
        else:
            floor = lo - 0.04 * (hi - lo)
        top = hi + 0.1 * (hi - floor)
        if key in figure.layout:
            out["y"][key] = (floor, top)
        for i, _band in bands.get(axis, []):
            out["bands"].append({"i": i, "lo": floor, "hi": top})
        # which scale the row is on, since it is no longer the one the run's page draws
        if np.isfinite(run_hi) and run_hi > hi:
            out["notes"].append({
                "yref": axis, "xref": x_axis,
                "text": f"max {hi:.3g} \u00b7 run {run_hi:.3g}",
            })
    return out


# the annotation a view fills in with its scale note. One slot per row, named after the row,
# so a note cannot land on a neighbour when the figure gains an annotation later.
SCALE_SLOT = "qc-scale-"


def apply_carpet_window(figure, t0: float, t1: float):
    """Narrow a carpet to one condition in place: time axis, GVTD rows, and their labels.

    The companion of :func:`carpet_window_spec`, which measures the same thing and hands it
    over instead. Both exist for the reason ``rescale_y_to_window`` and ``window_view_spec``
    both do: the saved file is written once and a page picks a view out of it at load, so
    the applied and the picked path have to be one measurement or they drift. Nothing in the
    report calls this; the browser test that proves the two paths render alike does.
    """
    spec = carpet_window_spec(figure, t0, t1)
    zoom_to_condition(figure, t0, t1)
    for key, (floor, top) in spec["y"].items():
        figure.layout[key].update(range=[floor, top], autorange=False)
    for band in spec["bands"]:
        tr = figure.data[band["i"]]
        floor, top = band["lo"], band["hi"]
        tr.y = [None if v is None else (floor if v <= floor else top) for v in tr.y]
    by_name = {a.name: a for a in (figure.layout.annotations or ()) if a.name}
    for note in spec["notes"]:
        if note["name"] in by_name:
            by_name[note["name"]].text = note["text"]
    return figure


def carpet_view_table(fig, spans: "list[tuple[str, float, float]]") -> "dict | None":
    """Each condition's view of a carpet: its window, and its GVTD rows re-fitted to it.

    The heatmaps keep the run's colour scale and only the line rows move;
    :func:`carpet_window_spec` says why, and computes each view.
    """
    from nirspipe.qc.common.figure_io import _pair_fname

    if not spans or not hasattr(fig, "add_annotation"):
        return None
    out = {}
    for label, t0, t1 in spans:
        spec = carpet_window_spec(fig, t0, t1)
        out[_pair_fname(label)] = {
            "x": spec["x"],
            "y": {key: [floor, top] for key, (floor, top) in spec["y"].items()},
            "bands": spec["bands"],
            "ann": {note["name"]: note["text"] for note in spec["notes"]},
        }
    return out


def condition_view_table(fig, spans: "list[tuple[str, float, float]]") -> "dict | None":
    """Every condition's view of one run-wide figure, keyed by slug for a page to pick.

    ::

      condition_view_table(detail_fig, [("video", 60.0, 300.0)])
      -> {"video": {"x": [60.0, 300.0], "y": {...}, "bands": [...], "ann": {...}}}

    The result is what ``_HASH_VIEW_JS`` spends: the file is written once and a condition
    page addresses it as ``…/figure.html#video``, so the run's figure and a condition's view
    of it cannot disagree.

    The scale notes go on as *empty* annotations first, one per row any condition writes one
    on, because a relayout cannot append an annotation. A row no condition annotates gets no
    slot, and a condition reaching the run's own maximum leaves its slot empty.

    ``spans`` must be on the same axis the figure was drawn on. Most figures here are drawn
    on the data axis; a caller whose figure is on the original recording's axis shifts the
    spans by ``first_time`` before calling.
    """
    if not spans or not hasattr(fig, "add_annotation"):
        return None
    from nirspipe.qc.common.figure_io import _pair_fname

    specs = [(_pair_fname(label), window_view_spec(fig, t0, t1)) for label, t0, t1 in spans]
    placed: set = set()
    for _slug, spec in specs:
        for note in spec["notes"]:
            if note["yref"] in placed:
                continue
            placed.add(note["yref"])
            fig.add_annotation(
                x=0.996, xref=f"{note['xref']} domain", y=0.97,
                yref=f"{note['yref']} domain", text="", showarrow=False,
                xanchor="right", yanchor="top", font=dict(size=7, color="#98a2ad"),
                name=f"{SCALE_SLOT}{note['yref']}")
    return {
        slug: {
            "x": spec["x"],
            "y": {key: [floor, top] for key, (floor, top) in spec["y"].items()},
            "bands": spec["bands"],
            "ann": {f"{SCALE_SLOT}{n['yref']}": n["text"] for n in spec["notes"]},
        }
        for slug, spec in specs
    }


# figures a condition page keeps whole, because they describe the run rather than any one
# condition. The optode layout is the montage; the event timeline is the run's schedule.
CONDITION_PAGE_FIGURES = ("layout", "trigger")


def figure_leaks(figure_paths: dict, slug: str) -> "list[str]":
    """Whole-run figures that survived onto a condition page, by value rather than by name.

    ::

      {"psd": {"src": "figures/sub-01_desc-rawpsd_nirs.html"}}, "video"  ->  ["psd"]

    A figure reaches a condition page one of three ways, and each leaves a mark: rewritten
    for the condition (its ``cond-`` entity is the slug), addressed at the condition's window
    (``…#<slug>``), or kept whole on purpose (:data:`CONDITION_PAGE_FIGURES`). Anything else
    is the run's figure sitting under this condition's numbers with nothing to say so.

    Checking the assembled values rather than a list of keys also catches a figure added
    later.
    """
    leaks = []
    for key, value in (figure_paths or {}).items():
        src = value.get("src") if isinstance(value, dict) else value
        if not isinstance(src, str) or "figures/" not in src:
            continue
        if key in CONDITION_PAGE_FIGURES:
            continue
        name, _, fragment = src.rsplit("/", 1)[-1].partition("#")
        if fragment == slug:
            continue
        if parse_path(name).get("condition") == slug:
            continue
        leaks.append(key)
    return leaks


def carpet_window_spec(figure, t0: float, t1: float) -> dict:
    """The carpet's GVTD and IMU rows re-fitted to one condition, its heatmaps left alone.

    ::

      carpet_window_spec(carpet_fig, 22.4, 322.4)
      -> {"x": [22.4, 322.4],
          "y": {"yaxis2": (0.0, 0.0061), "yaxis3": (0.0, 0.0061)},
          "bands": [{"i": 1, "lo": 0.0, "hi": 0.0061}, ...],
          "notes": [{"name": "gvtd-stat-long", "text": "max 2.36e-03 ..."}, ...]}

    Two things differ from :func:`window_view_spec`.

    **One top for every GVTD row, by the figure's own rule.** ``gvtd_y_top`` is called on
    this window's samples, so the view is drawn the way the panel is drawn: a 99.5th
    percentile rather than a maximum, and never below the threshold rule. Long and short
    share the number, so the difference between the two rows stays visible.

    **The heatmaps are not touched.** Their colour is a z-score against a per-channel mean
    and SD taken over the whole run from the uncorrected side, and both carpets use those
    same two numbers, so one colour means one deviation on every page. The lines print the
    scale they are on in the rewritten stat labels below.
    """
    from nirspipe.qc.figures.common.motion_panel import (
        GVTD_STAT_SLOT, IMU_SLOT, _gvtd_stat_label, gvtd_y_top, imu_y_top,
    )

    out = {"x": [float(t0), float(t1)], "y": {}, "bands": [], "notes": []}
    if not hasattr(figure, "update_yaxes"):
        return out

    # each IMU row, where there is one, takes its own top by its own rule: not a GVTD unit
    for a in (figure.layout.annotations or ()):
        if not (a.name or "").startswith(IMU_SLOT):
            continue
        axis = str(a.yref).removesuffix(" domain")
        cut = []
        for tr in figure.data:
            x = _trace_x(tr)
            if (getattr(tr, "yaxis", None) or "y") == axis and x is not None:
                cut.append(np.asarray(tr.y, dtype=float)[(x >= t0) & (x <= t1)])
        cut = np.concatenate(cut) if cut else np.empty(0)
        if cut.size:
            out["y"]["yaxis" + axis[1:]] = (0.0, imu_y_top(cut))

    # the GVTD rows are the ones carrying a named stat label; the strip has none and the
    # heatmaps are not scatter traces at all
    # the label is anchored to the row's domain, so its yref reads "y2 domain"
    rows = {str(a.yref).removesuffix(" domain"): a.name[len(GVTD_STAT_SLOT):]
            for a in (figure.layout.annotations or ())
            if a.name and a.name.startswith(GVTD_STAT_SLOT) and not a.name.endswith("-after")}
    if not rows:
        return out

    inside, per_row = [], {}
    for i, tr in enumerate(figure.data):
        axis = getattr(tr, "yaxis", None) or "y"
        if axis not in rows:
            continue
        if getattr(tr, "fill", None) == "toself":
            out["bands"].append({"i": i, "axis": axis})
            continue
        x, y = _trace_x(tr), np.asarray(tr.y, dtype=float)
        if x is None:
            continue
        cut = y[(x >= t0) & (x <= t1)]
        cut = cut[np.isfinite(cut)]
        if cut.size:
            inside.append(cut)
            per_row.setdefault((axis, str(getattr(tr, "name", ""))), cut)
    if not inside:
        return out

    top = gvtd_y_top(inside, [_threshold_of(figure, a) for a in rows])
    for axis in rows:
        key = "yaxis" + axis[1:]
        if key in figure.layout:
            out["y"][key] = (0.0, top)
    for band in out["bands"]:
        band["lo"], band["hi"] = 0.0, top
        band.pop("axis")

    for (axis, name), cut in per_row.items():
        suffix = "-after" if name == "after" else ""
        out["notes"].append({
            "name": f"{GVTD_STAT_SLOT}{rows[axis]}{suffix}",
            "text": _gvtd_stat_label(cut, _threshold_of(figure, axis),
                                     prefix="corrected" if suffix else ""),
        })
    return out


def _threshold_of(figure, axis: str) -> "float | None":
    """The dashed threshold rule on one GVTD row, or None if it carries no line."""
    for shape in (figure.layout.shapes or ()):
        if getattr(shape, "yref", None) == axis and getattr(shape, "y0", None) is not None:
            if shape.y0 == shape.y1:
                return float(shape.y0)
    return None


def condition_payloads(
    payload: dict,
    *,
    record: dict,
    by_condition: dict,
    sci_scores: "dict[str, float]",
    bad_channels: "set[str]",
    channel_pairs: "list[str] | None",
    series: dict,
    sci_threshold: float,
    cutoffs: "dict[str, float]",
    trial_rows: "list | None" = None,
    save_figure=None,
    remake_psd=None,
    remake_epoch=None,
    trial_images=None,
    save_stack=None,
    remake_hbo_hbr_corr=None,
    sep_bands=None,
) -> "list[tuple[str, dict]]":
    """One viewer payload per condition, read out of the quality record.

    Every number here comes from ``by_condition``, which
    :func:`~nirspipe.qc.subject.sqm_record.raw_condition_sections` wrote; nothing is measured. A
    record with no such section gets no pages rather than a second copy of the numbers free
    to disagree with the first.

    ``payload`` is the run's payload, whose parts that do not vary by condition (the optode
    layout, the separation notes) are carried over untouched. A figure gets here one of
    three ways, and which one is a property of the figure:

    - **rewritten for the condition**: the SCI/PSP panel and the channel-quality grid. Both
      are handed their data and derive nothing from a recording, so a real slice is correct
      and ``save_figure`` writes each under a name carrying the condition's slug.
    - **addressed at the condition's window**: the carpet, the per-channel detail and the
      per-channel motion figure. Each
      derives something run-wide from what it is handed, a filtered GVTD, a per-channel
      z-scale, a colour range, so a cut would give every condition a scale no other one can
      be read against. The run's file carries every window and this page asks for one by URL
      fragment.
    - **taken out of the run's own pass**: the trial images. One run-wide pass drew every
      condition's, so the pages share a colour scale.
    - **rebuilt on a cropped copy**: the spectrum and the grand mean, through ``remake_psd``
      and ``remake_epoch``. There is nothing
      to slice, it being one spectrum rather than a time-by-frequency matrix, and nothing in
      a Welch estimate reads outside the samples it is handed. A condition too short for the
      transform gets no spectrum rather than one on a coarser grid than the run's. The
      HbO-HbR correlation panel too, through ``remake_hbo_hbr_corr``, for the reason the
      record measures its numbers on the cut.

    The event timeline is the run's own figure, left whole: it is the schedule the whole
    recording ran to.
    The per-trial table is cut to the trials whose onset falls inside the window; nothing is
    rescored, a trial's SQM reading nothing outside its own crop.

    Each payload carries three notes, because each is a way a reader could be misled: what
    the view is, that its rejected channels are the run's while the header counts the ones
    failing on this condition's windows alone, and that the columns with no windowed series
    behind them are absent rather than zero.
    """
    from nirspipe.qc.boilerplate.vocabulary import metric_rows
    from nirspipe.qc.common.channel_table import (
        CONDITION_OD_SPLIT_COLUMNS, HAEMO_SPLIT_COLUMNS, MOTION_SPLIT_COLUMNS,
        WHOLE_RUN_ONLY_COLUMNS,
        channel_rows, format_rows, heatmap_args, pair_rows, separation_blocks, split_table,
    )
    from nirspipe.qc.common.figure_io import _pair_fname
    from nirspipe.qc.figures import build_sci_psp_figure, channel_quality_heatmap

    od_cols = CONDITION_OD_SPLIT_COLUMNS
    motion_cols = tuple((k, t) for k, t in MOTION_SPLIT_COLUMNS
                        if k not in WHOLE_RUN_ONLY_COLUMNS)

    out: list[tuple[str, dict]] = []
    for label, entry in by_condition.items():
        sliced = entry.get("per_channel") or {}
        cond_bad = set(entry.get("bad_channels") or ())
        scalars = entry.get("scalars") or {}
        od_by_set = entry.get("od_by_set") or {}
        motion_by_set = entry.get("motion_by_set") or {}
        t0, t1 = entry["window_s"]
        window = (label, float(t0), float(t1))
        slug = _pair_fname(label)

        # Status is the run's rejection; the condition's own assessment is the header count
        rows = channel_rows(
            with_condition_corr(slice_record(record, sliced),
                                sliced.get("hbo_hbr_corr_per_channel"), section="rawhaemo"),
            sci_scores, bad_channels)
        pair_cells = format_rows(pair_rows(rows, channel_pairs or None), sci_threshold,
                                 name_key="pair", psp_threshold=cutoffs["psp"])
        # split on the montage, not on whether a Short entry exists: one is returned for
        # every montage, all None where there are no short channels
        has_short = bool(scalars.get("n_short_channels"))
        split = split_table([
            ("All",   len(rows),                        od_by_set.get("all") or {},   False),
            ("Long",  scalars.get("n_long_channels"),   od_by_set.get("long") or {},  True),
            ("Short", scalars.get("n_short_channels"),  od_by_set.get("short") or {}, False),
        ], od_cols) if has_short else {}

        motion_split = split_table([
            ("All",   len(rows),                       motion_by_set.get("all") or {},   False),
            ("Long",  scalars.get("n_long_channels"),  motion_by_set.get("long") or {},  True),
            ("Short", scalars.get("n_short_channels"), motion_by_set.get("short") or {}, False),
        ], motion_cols) if has_short and motion_by_set.get("short") else {}

        # the run page's rows: All alone on a montage with nothing to split, where Long would
        # repeat it
        haemo_by_set = entry.get("haemo_by_set") or {}
        haemo_sets = (("All", len(rows), "all", False),
                      ("Long", scalars.get("n_long_channels"), "long", True),
                      ("Short", scalars.get("n_short_channels"), "short", False))
        haemo_rows = [(name, n, haemo_by_set[key], colour)
                      for name, n, key, colour in (haemo_sets if has_short else haemo_sets[:1])
                      if haemo_by_set.get(key)]
        haemo_split = split_table(haemo_rows, HAEMO_SPLIT_COLUMNS) if haemo_rows else {}

        d = dict(payload)
        # what the page is, for the panels that stay whole and have to say why
        d["condition_label"] = label
        # the run's summary describes the run's verdict; this page screens on its own
        # stretch, so it carries its own count and says which window it is
        n_total = len(rows)
        bad_rate = 100 * len(cond_bad) / n_total if n_total else 0.0
        d["summary"] = {
            **(payload.get("summary") or {}),
            "run": label,
            "n_bad": len(cond_bad),
            "n_total": n_total,
            "bad_rate": bad_rate,
            "badge_class": ("badge-green" if bad_rate < 10
                            else "badge-yellow" if bad_rate < 30 else "badge-red"),
            "scope": section_note("raw.condition_scope", label=label, t0=t0, t1=t1),
        }
        # the flat list keeps only what neither table covers, so a number is printed once.
        # Derived from the two column lists rather than written out again: a column added to
        # either table leaves the list on its own.
        covered = (({k for k, _ in od_cols} if split else set())
                   | ({k for k, _ in motion_cols} if motion_split else set()))
        flat_keys = tuple(k for k in COND_SCALAR_KEYS if k not in covered)
        d["sqm"] = {
            "rows": metric_rows(scalars, flat_keys, skip_missing=True, condition=True),
            "split": split,
            "motion_split": motion_split,
            "haemo_split": haemo_split,
            # every channel, and the three sets are in the tables. Not "no short channels":
            # the Short rows below say whether the montage has them
            "channel_set": "every channel",
        }
        d["channels"] = {
            "pairs":  pair_cells,
            "blocks": separation_blocks(pair_cells, sep_bands),
            # carried over: a separation is a property of the montage, not of a condition
            "notes":  (payload.get("channels") or {}).get("notes") or [],
        }
        d["ts"] = _narrowed_ts(payload.get("ts"), window)
        # not read by the viewer and heavy to carry: it draws these from `figure_paths`
        d["carpet_gvtd"] = {}
        d["psd"] = {}
        d["evoked_topo"] = {}

        # the run's, until this page writes its own; carried over it would be every
        # condition's trials under one condition's heading
        d["trial_images"] = []
        paths = dict(payload.get("figure_paths") or {})
        paths.pop("psd", None)
        paths.pop("evoked_topo", None)
        paths.pop("hbo_hbr_corr", None)
        for key in ("carpet", "ch_detail_template", "motion_detail_template"):
            entry_path = paths.get(key)
            if isinstance(entry_path, dict) and entry_path.get("src"):
                paths[key] = {**entry_path, "src": f"{entry_path['src']}#{slug}"}
            elif isinstance(entry_path, str):
                paths[key] = f"{entry_path}#{slug}"
        if save_figure is not None:
            sci_fig = _condition_sci_psp(
                build_sci_psp_figure, sliced.get("sci_win_per_channel") or {},
                sliced.get("psp_per_channel") or {}, sliced.get("cv_per_channel") or {},
                bad_channels, sci_threshold, cutoffs["psp"], series, window)
            for key, fig in (("sci_psp", sci_fig),
                             ("ch_summary", channel_quality_heatmap(
                                 sci_thresh=sci_threshold, psp_thresh=cutoffs["psp"],
                                 good_frac_thresh=cutoffs["good_frac"],
                                 **heatmap_args(rows)))):
                saved = save_figure(key, slug, fig) if fig is not None else None
                if saved:
                    paths[key] = saved
                else:
                    paths.pop(key, None)
            trial = _condition_trial_figure(trial_rows, window, slug, save_figure)
            paths.pop("trial_qc", None)
            d["trial_qc"] = {}
            if trial:
                paths["trial_qc"], d["trial_qc"] = trial
            psd_fig = remake_psd(float(t0), float(t1)) if remake_psd else None
            paths.pop("psd", None)
            if psd_fig is not None:
                paths["psd"] = save_figure("psd", slug, psd_fig)
            hb_fig = (remake_hbo_hbr_corr(float(t0), float(t1))
                      if remake_hbo_hbr_corr else None)
            if hb_fig is not None:
                paths["hbo_hbr_corr"] = save_figure("hbo_hbr_corr", slug, hb_fig)
            epoch_fig = remake_epoch(float(t0), float(t1)) if remake_epoch else None
            paths.pop("epoch_mean", None)
            if epoch_fig is not None:
                paths["epoch_mean"] = save_figure("epoch_mean", slug, epoch_fig)
            # this condition's rows out of the run-wide pass, never redrawn: one shared scale
            d["trial_images"] = [
                saved for pair, figs in ((trial_images or {}).get(label) or [])
                for saved in [save_stack("trialimg", slug, pair, figs)] if saved
            ] if save_stack else []
        # the trial images are figure paths outside `figure_paths`, so they are checked too
        leaks = figure_leaks(
            {**paths, **{f"trial_image[{e['pair']}]": e for e in d["trial_images"]}}, slug)
        if leaks:
            # loud, not silent: a run-wide figure here would read as this condition's
            logger.warning("condition %s still points at run-wide figures (%s); they are "
                           "being dropped", label, ", ".join(leaks))
            for key in leaks:
                if key.startswith("trial_image["):
                    d["trial_images"] = []
                else:
                    paths.pop(key, None)
        d["figure_paths"] = paths

        d["notes"] = list(payload.get("notes") or []) + [
            section_note("raw.condition_sliced", label=label),
            section_note("raw.condition_verdict"),
            section_note("raw.condition_spectrum"),
        ]
        out.append((label, d))
    return out


def _condition_trial_figure(trial_rows, window, slug, save_figure):
    """The run's per-trial table cut to the trials whose onset falls in one condition.

    Nothing is rescored: a trial's SQM is measured on a crop of its own window and reads
    nothing outside it, so a condition's rows are the run's rows. A block design gets
    nothing here, its condition window holding only the annotation that defines it.
    """
    from nirspipe.qc.figures import trial_quality_heatmap

    t0, t1 = window[1], window[2]
    keep = [(lab, sqm) for onset, lab, sqm in (trial_rows or []) if t0 < onset < t1]
    if len(keep) < 2:
        return None
    fig = trial_quality_heatmap([lab for lab, _ in keep], [sqm for _, sqm in keep])
    if fig is None:
        return None
    saved = save_figure("trial_qc", slug, fig)
    return (saved, {"n_trials": len(keep)}) if saved else None


def _mean_or_none(values) -> "float | None":
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _narrowed(inline: "dict | None", window) -> dict:
    if not inline or not inline.get("figure"):
        return {}
    return {**inline, "figure": zoom_to_condition(inline["figure"], window[1], window[2])}


def slice_time_traces(figure: dict, t0: float, t1: float) -> dict:
    """A figure dict holding only the samples inside ``[t0, t1]``, scales untouched.

    ::

      n traces of 4427 points, 3602.4 to 3902.5 s  ->  n traces of ~1300

    The companion of :func:`zoom_to_condition`. Narrowing sets the axis and leaves the run's
    samples in the trace, which is what a figure deriving something internally needs; it also means Plotly's own
    double-click, which autoranges over the data it holds, opens the whole run. Slicing
    removes that, so it is right wherever the y values do not depend on which samples are
    present. The raw-signal panel qualifies: each channel was z-scored and offset over the
    whole recording before the figure was built, so both numbers are already baked in and
    dropping samples moves nothing.

    Marker rectangles drawn against the data axis are dropped when they fall outside, since
    a shape at 22 s would pull an autorange back to the start of the recording. The striped
    channel bands are on ``paper`` and stay.
    """
    data = []
    for trace in figure.get("data") or []:
        x = trace.get("x")
        if not isinstance(x, list):
            data.append(trace)
            continue
        keep = [i for i, v in enumerate(x) if v is not None and t0 <= float(v) <= t1]
        cut = {**trace, "x": [x[i] for i in keep]}
        for key in ("y", "customdata", "text", "hovertext"):
            seq = trace.get(key)
            if isinstance(seq, list) and len(seq) == len(x):
                cut[key] = [seq[i] for i in keep]
        data.append(cut)

    layout = dict(figure.get("layout") or {})
    shapes = layout.get("shapes")
    if isinstance(shapes, list):
        layout["shapes"] = [
            sh for sh in shapes
            if sh.get("xref") != "x"
            or (float(sh.get("x1", t1)) >= t0 and float(sh.get("x0", t0)) <= t1)
        ]
    for key in [k for k in layout if k == "xaxis" or k.startswith("xaxis")]:
        layout[key] = {**(layout[key] or {}), "range": [float(t0), float(t1)]}
    return {**figure, "data": data, "layout": layout}


def _narrowed_ts(inline: "dict | None", window) -> dict:
    """The raw-signal panel cut to one condition, with its bounds and markers to match.

    ``t_start`` / ``t_end`` travel in this payload for the GUI, which reads them rather than
    the figure's axis, so leaving them at the run's bounds would give two views of one
    condition different spans.
    """
    if not inline or not inline.get("figure"):
        return {}
    markers = [m for m in (inline.get("markers") or [])
               if window[1] <= float(m.get("onset", -1)) <= window[2]]
    return {**inline,
            "figure": slice_time_traces(inline["figure"], window[1], window[2]),
            "t_start": float(window[1]), "t_end": float(window[2]),
            "markers": markers}


def _condition_sci_psp(build, sci_pc, psp_pc, cv_pc, bad_channels, sci_threshold,
                       psp_threshold, series, window):
    """The SCI/PSP panel over one condition's columns, a real slice of both matrices.

    Unlike the carpet, this figure derives nothing internally: it is handed the matrices and
    the per-channel scalars, so both are replaced by the condition's and a per-condition
    heatmap never sits beside a whole-run bar.

    Returns the figure itself, or None when no window of the run's grid falls inside this
    condition, so the caller can write it to a file of its own.
    """
    from nirspipe.qc.metrics.windowed import _in_scope, window_centers

    def _cut(matrix_key, times_key):
        """One metric's matrix and window times, cut on that metric's own grid."""
        matrix, times = series.get(matrix_key), series.get(times_key)
        if matrix is None or times is None:
            return None, None
        keep = _in_scope(window_centers(np.asarray(times)), [window])
        if not keep.any():
            return None, None
        return np.asarray(matrix)[:, keep], np.asarray(times)[keep]

    sci_matrix, sci_times = _cut("sci_matrix", "sci_times")
    if sci_matrix is None:
        return None
    psp_matrix, psp_times = _cut("psp_matrix", "psp_times")
    cv_matrix, cv_times = _cut("cv_matrix", "cv_times")
    return build(
        sci_pc, psp_pc, bad_channels, sci_threshold, psp_threshold=psp_threshold,
        window_s=series.get("qc_window_s"),
        sci_matrix=sci_matrix, sci_win_times=sci_times,
        psp_matrix=psp_matrix, psp_win_times=psp_times,
        cv_per_channel=cv_pc, cv_matrix=cv_matrix, cv_win_times=cv_times,
    )


def condition_slices_from_record(
    record: dict,
    ch_names: "list[str]",
    windows: "list[tuple[str, float, float]]",
    sci_cutoff: float,
    psp_cutoff: float,
    spans=(),
) -> "dict[str, dict[str, dict[str, float]]]":
    """Per-condition per-channel metrics read out of a quality record, nothing recomputed.

    ::

      {"game1": {"sci_win_per_channel": {...}, "psp_per_channel": {...},
                 "good_frac_per_channel": {...}}, ...}

    The record's ``windowed`` section already stores the channel-by-window SCI and PSP
    matrices and their window bounds, so a condition is a column selection out of what the
    run measured once.

    The coupled-window share is rebuilt here rather than read, because the record stores
    only its whole-run value. It goes through
    :func:`~nirspipe.qc.metrics.windowed.coupled_mask_from_matrices`, the same AND the
    screening applied, so a condition's share cannot disagree with the verdict the run was
    screened by. Both cutoffs have to be the run's own; passing anything else produces a
    number no channel was judged against.

    ``spans`` are the run's ``BAD_`` spans as ``(start, stop)``: a window touching one is left
    out of the share, as the screening leaves it out, while the SCI, PSP and CV means keep
    every window, as the run's own means do.

    Returns an empty dict when the record carries no stored matrices: the values cannot be
    recovered from the whole-run scalars.
    """
    from nirspipe.qc.metrics.windowed import (
        condition_window_means, coupled_mask_from_matrices, windows_touching,
    )

    windowed = record.get("windowed") or {}
    sci_matrix, psp_matrix = windowed.get("sci_matrix"), windowed.get("psp_matrix")
    sci_times, psp_times = windowed.get("sci_times"), windowed.get("psp_times")
    cv_matrix, cv_times = windowed.get("cv_matrix"), windowed.get("cv_times")
    if not sci_matrix or sci_times is None:
        logger.warning("the quality record carries no windowed matrices, so no "
                       "per-condition view can be built from it")
        return {}

    # a channel with no value in the condition is left out, as the whole run leaves it out
    def _named(vals) -> "dict[str, float]":
        return {ch: float(vals[i]) for i, ch in enumerate(ch_names)
                if i < len(vals) and np.isfinite(vals[i])}

    out: dict[str, dict[str, dict[str, float]]] = {}
    sci_by_cond = condition_window_means(sci_matrix, sci_times, windows)
    psp_by_cond = ({} if not psp_matrix or psp_times is None
                   else condition_window_means(psp_matrix, psp_times, windows))
    mask = coupled_mask_from_matrices(sci_matrix, psp_matrix, sci_cutoff, psp_cutoff) \
        if psp_matrix else None
    if mask is not None and len(spans):
        mask = mask.astype(float)
        mask[:, windows_touching(sci_times, spans)] = np.nan
    frac_by_cond = ({} if mask is None
                    else condition_window_means(mask.astype(float), sci_times, windows))
    # a record with no CV matrix leaves CV off its pages
    cv_by_cond = ({} if not cv_matrix or cv_times is None
                  else condition_window_means(cv_matrix, cv_times, windows))

    for window in windows:
        label = window[0]
        if label not in sci_by_cond:
            continue
        out[label] = {
            "sci_win_per_channel": _named(sci_by_cond[label]),
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
        # windowed by construction: a condition is a column selection out of sci_matrix,
        # and the whole-run sci_mean has no slice of itself to give
        "sci_win_mean":   _mean_or_none((sliced.get("sci_win_per_channel") or {}).values()),
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

        sliced["sci_win_per_channel"] over a montage of long and short channels
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
            "sci_win_mean":   _mean("sci_win_per_channel"),
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
