"""Plotly figure builders for the hyperscanning group-level raw QC report."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.common.figure_io import extract_markers as _extract_markers
from fnirs_pipe.qc.metrics.hyper import (  # noqa: F401  (re-exported for the panels)
    NULL_ALPHA_PCT, _ch_kept_by_member, dyad_status, sci_of,
)
from fnirs_pipe.qc.figures.common._utils import (CONDITION_PALETTE, PSD_NFFT,
                                          TIMELINE_ROW_PX,
                                          decimate as _decimate, physio_bands, timeline_axes,
                                          timeline_row_bands, timeline_row_traces)
# imported rather than restated, so the dyad and subject reports draw the same heads
from fnirs_pipe.qc.figures.common.head_map import (
    head_axes as _head_axes, head_ground as _head_ground,
    head_glyph as _head_glyph,
)
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.qc.figures.common.motion_panel import (
    carpet_z, _maxpool_xy, _px_rows, _span_polygons, add_carpet, carpet_coloraxis,
)
from fnirs_pipe.qc.metrics import (
    GVTD_MOTION_BAND, gvtd_channel_blocks, gvtd_timetrace, spike_segments,
)

logger = get_logger("qc.figures.hyper")

_MAX_TS_PTS = 4000

_SUB_COLORS   = ["#8e44ad", "#e67e22", "#16a085", "#f39c12",
                  "#2c3e50", "#1abc9c", "#c0392b", "#34495e"]
_COND_PALETTE = CONDITION_PALETTE
_COND_DASHES  = ["solid", "dash", "dot", "dashdot", "longdash"]

_GOOD_COLOR = "#C5E0B3"
_MIX_COLOR  = "#FFD966"
_NA_COLOR   = "#D3D3D3"
_BAD_COLOR  = "#F8786E"

# colors for the physiological band annotations (frequencies come from physio_bands)
_PSD_BAND_COLORS = {
    "Mayer":   "rgba(52,152,219,0.10)",
    "Resp":    "rgba(39,174,96,0.08)",
    "Cardiac": "rgba(231,76,60,0.08)",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _cond_colors(descriptions: list[str]) -> dict[str, str]:
    return {d: _COND_PALETTE[i % len(_COND_PALETTE)] for i, d in enumerate(descriptions)}


def _psd_band_shapes(cardiac=None) -> tuple[list[dict], list[dict]]:
    # resp omitted: the hyper raw pipeline has no respiration band input
    bands = physio_bands(cardiac=cardiac, resp=None)
    shapes = [dict(type="rect", xref="x", yref="paper",
                   x0=x0, x1=x1, y0=0, y1=1,
                   fillcolor=_PSD_BAND_COLORS.get(name, "rgba(120,120,120,0.08)"),
                   line=dict(width=0)) for name, x0, x1 in bands]
    annots = [dict(x=(x0 + x1) / 2, y=0.97, xref="x", yref="paper",
                   text=name, showarrow=False,
                   font=dict(size=8, color="#666")) for name, x0, x1 in bands]
    return shapes, annots


# ---------------------------------------------------------------------------
# Figure: the conditions, before and after alignment
# ---------------------------------------------------------------------------

def build_alignment_timeline(
    raw_raws: "dict[str, mne.io.Raw] | None",
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
) -> "go.Figure | None":
    """Every member's annotated blocks, on its own clock above and the shared one below.

    The second row is the shared clock every later panel reads.

    One row per member in each panel. None when nothing is annotated.
    """
    rows = [("Each recording on its own clock", raw_raws),
            ("After alignment, on the shared clock", aligned_raws)]
    rows = [(title, raws) for title, raws in rows if raws]
    if not rows:
        return None

    descs = list(dict.fromkeys(
        b["desc"] for _title, raws in rows for sid in subject_ids
        if sid in raws for b in _blocks(raws[sid])))
    if not descs:
        return None
    colours = _cond_colors(descs)

    fig = make_subplots(rows=len(rows), cols=1, vertical_spacing=0.22,
                        subplot_titles=[t for t, _r in rows])
    seen: set[str] = set()
    for r, (_title, raws) in enumerate(rows, start=1):
        for yi, sid in enumerate(subject_ids):
            for b in _blocks(raws.get(sid)):
                show = b["desc"] not in seen
                seen.add(b["desc"])
                fig.add_trace(go.Bar(
                    x=[max(b["duration"], 1.0)], y=[yi], base=[b["onset"]],
                    orientation="h", width=0.42,
                    marker=dict(color=colours[b["desc"]], opacity=0.9, cornerradius=3,
                                line=dict(width=0)),
                    name=b["desc"], legendgroup=b["desc"], showlegend=show,
                    hovertemplate=(f"<b>{b['desc']}</b><br>{sid}<br>"
                                   "onset %{base:.2f} s<br>"
                                   f"duration {b['duration']:.2f} s<extra></extra>"),
                ), row=r, col=1)
        fig.update_yaxes(tickvals=list(range(len(subject_ids))), ticktext=subject_ids,
                         range=[len(subject_ids) - 0.4, -0.6], showgrid=False,
                         tickfont=dict(size=10), row=r, col=1)
        fig.update_xaxes(gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9),
                         row=r, col=1)
    fig.update_xaxes(title_text="Time (s)", title_font=dict(size=10), row=len(rows), col=1)
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=170 * len(rows), plot_bgcolor="white", barmode="overlay",
                      margin=dict(l=96, r=24, t=70, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


def _blocks(raw: "mne.io.Raw | None") -> list[dict]:
    """Annotated blocks on the recording's own zero, BAD spans dropped.

    ``crop`` moves ``first_samp`` and leaves annotation onsets on the original clock, so the
    shared-clock time is ``onset - first_time``. Without that subtraction an aligned timeline
    is drawn identical to the unaligned one.
    """
    if raw is None:
        return []
    t0 = float(raw.first_time)
    return [{"desc": str(a["description"]), "onset": float(a["onset"]) - t0,
             "duration": float(a["duration"] or 0.0)}
            for a in raw.annotations
            if not str(a["description"]).upper().startswith("BAD")]


# ---------------------------------------------------------------------------
# Figure: shared usable time
# ---------------------------------------------------------------------------

# (key, axis label, whether the row is divided by each member's own median). SCI and PSP are
# what the carpet's mask is made of; CV catches baseline shifts and signal loss they miss.
_SERIES_ROWS = (("sci", "SCI (10 s)", False), ("psp", "PSP (10 s)", False),
                ("cv", "CV (10 s)", False))
_LEAD_COLOURS = ("#3498db", "#e67e22", "#16a085", "#8e44ad")


def _row_title(fig, row: int, text: str, x: float = -0.055) -> None:
    """A series row's name, at a fixed distance from the plot rather than from its ticks.

    Plotly pushes a y-axis title left by the width of that axis's tick labels, so rows whose
    labels are ``0.8`` and ``1`` and ``0.01`` put their titles at three different depths and
    the column of names comes out ragged. An annotation on the paper x axis, centred on the
    row's own domain, lands in the same place whatever the ticks say.
    """
    fig.add_annotation(x=x, xref="paper", xanchor="center",
                       y=0.5, yref=f"y{row if row > 1 else ''} domain", yanchor="middle",
                       text=text, textangle=-90, showarrow=False,
                       font=dict(size=9.5, color="#34495e"))


def build_usable_time(
    grid: dict,
    subject_ids: list[str],
    series: dict[str, dict],
    conditions: "dict[str, tuple[float, float]] | None" = None,
    cutoffs: "dict[str, float] | None" = None,
) -> "go.Figure | None":
    """Which pairs the dyad could use, when, and what went wrong where it could not.

    A channel is usable by the dyad only while it is coupled in **both** members at the same
    moment, so the dyad's usable time is the intersection of the two and not the smaller of
    them: two members can each keep 13 of 14 pairs and share only 12, if the pair they each
    lose is a different one. That intersection is the carpet at the bottom.

    Over it, each member's long-channel means for the metrics the mask is made of, so the
    panel says why a pair went and not only that it did. A condition bar names the blocks
    and dotted rules carry their edges down through the series.

    None when the grid holds no pair.
    """
    pairs = list(grid.get("long_pairs") or grid["pairs"])
    if not pairs:
        return None
    t = np.asarray(grid["t"], dtype=float)
    # long channels only, the set every dyad measure runs on
    keep = [i for i, name in enumerate(grid["pairs"]) if name in set(pairs)]
    status = dyad_status(grid, subject_ids)[keep]
    lost = (status != 2).mean(axis=1)
    order = np.argsort(lost)[::-1]
    labels = [f"{pairs[i]}   {lost[i] * 100:.0f}%" if lost[i] else pairs[i] for i in order]

    have = [(key, label, scaled) for key, label, scaled in _SERIES_ROWS
            if any(key in (series.get(sid) or {}) for sid in subject_ids)]
    conditions = conditions or {}
    bar_rows = 1 if conditions else 0
    n_rows = bar_rows + len(have) + 1
    bar_h, series_h, carpet_h = 34, 86, 26 * len(pairs)
    total = bar_h * bar_rows + series_h * len(have) + carpet_h
    heights = ([bar_h / total] * bar_rows + [series_h / total] * len(have)
               + [carpet_h / total])
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=heights)

    if bar_rows:
        colours = _cond_colors(list(conditions))
        for name, (a, b) in conditions.items():
            fig.add_trace(go.Bar(
                x=[b - a], y=[0], base=[a], orientation="h", width=0.55,
                marker=dict(color=colours[name], opacity=0.9, cornerradius=3,
                            line=dict(width=0)),
                name=name, legendgroup=name, showlegend=True,
                hovertemplate=f"<b>{name}</b><br>%{{base:.0f}} to {b:.0f} s<extra></extra>",
            ), row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[-0.5, 0.5],
                         row=1, col=1)

    lines = dict(zip(subject_ids, _LEAD_COLOURS))
    for r, (key, label, scaled) in enumerate(have, start=bar_rows + 1):
        for sid in subject_ids:
            s = series.get(sid) or {}
            if key not in s:
                continue
            y = np.asarray(s[key], dtype=float)
            if scaled:
                mid = float(np.nanmedian(y))
                y = y / mid if np.isfinite(mid) and mid else y
            fig.add_trace(go.Scatter(
                x=np.asarray(s["t"]), y=y, mode="lines", name=sid,
                legendgroup=sid, showlegend=(r == bar_rows + 1),
                line=dict(color=lines[sid], width=1.3), opacity=0.95,
                hovertemplate=f"t=%{{x:.0f}}s<br>{label} %{{y:.4g}}<extra></extra>",
            ), row=r, col=1)
        line = 1.0 if scaled else (cutoffs or {}).get(key)
        if line is not None:
            fig.add_hline(y=float(line), line_color="#adb5bd", line_width=1,
                          line_dash="dot", row=r, col=1)
        _row_title(fig, r, label)

    fig.add_trace(go.Heatmap(
        z=status[order], x=t, y=labels,
        colorscale=[[0, _BAD_COLOR], [0.33, _BAD_COLOR], [0.33, _MIX_COLOR],
                    [0.66, _MIX_COLOR], [0.66, _GOOD_COLOR], [1, _GOOD_COLOR]],
        zmin=0, zmax=2, showscale=False, ygap=2,
        hovertemplate="<b>%{y}</b><br>t=%{x:.0f}s<br>%{z:.0f} member(s) coupled<extra></extra>",
    ), row=n_rows, col=1)
    # the key a discrete heatmap cannot carry itself
    for colour, lab in ((_GOOD_COLOR, "coupled in both"), (_MIX_COLOR, "one member only"),
                        (_BAD_COLOR, "neither")):
        fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name=lab,
                             marker=dict(color=colour, line=dict(width=0)),
                             showlegend=True, hoverinfo="skip"), row=n_rows, col=1)

    for a, b in conditions.values():
        for edge in (a, b):
            fig.add_vline(x=edge, line_color="#e3e8ec", line_width=1, line_dash="dot")

    fig.update_xaxes(gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=False, autorange="reversed", tickfont=dict(size=9.5),
                     row=n_rows, col=1)
    fig.update_xaxes(title_text="Time on the shared clock (s)", title_font=dict(size=10),
                     row=n_rows, col=1)
    fig.update_layout(height=96 + total, plot_bgcolor="white", barmode="overlay",
                      margin=dict(l=112, r=24, t=48, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


# ---------------------------------------------------------------------------
# Figure: motion, and whether the two moved together
# ---------------------------------------------------------------------------

# The motion panel is two figures, before and after correction, on the same rows and axes.

# Colour is the member, and nothing else: the channel set is the row, before/after the figure.
_MEMBER_COLOURS = ["#4c72b0", "#c44e52", "#55a868", "#8172b3"]
# The shared floor under both traces; grey, since it belongs to the pair and not to either.
_TOGETHER_FILL = "rgba(120,120,130,0.30)"
_SPIKE_BOTH = "rgba(245,158,11,0.9)"

_MOTION_ROW_PX = 84
_MOTION_CARPET_PX = 240
_SPIKE_ROW_PX = 26
# Headroom over the 99.5th percentile of every trace on a set's rows; each row's own maximum
# stays printed in the margin.
_MOTION_CAP_PCTL = 99.5
_MOTION_HEADROOM = 1.35
# Spike runs closer than this many seconds are drawn as one mark.
SPIKE_MERGE_S = 2.0


def _merge_spans(spans, gap: float = SPIKE_MERGE_S):
    """Join spans whose gap is under ``gap`` seconds.

    ::

        [(1.0, 0.2), (1.9, 0.3), (40.0, 0.1)], gap 2  ->  [(1.0, 1.2), (40.0, 0.1)]
    """
    out: list[list[float]] = []
    for onset, duration in sorted((float(o), float(d)) for o, d in spans):
        if out and onset - (out[-1][0] + out[-1][1]) < gap:
            out[-1][1] = max(out[-1][0] + out[-1][1], onset + duration) - out[-1][0]
        else:
            out.append([onset, duration])
    return [(o, d) for o, d in out]


def _spans_to_mask(spans, t: np.ndarray) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    for onset, duration in spans:
        m |= (t >= onset) & (t <= onset + duration)
    return m


def _mask_to_spans(mask: np.ndarray, t: np.ndarray):
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.view(np.int8), [0]])))
    return [(float(t[a]), float(t[min(b, len(t) - 1)] - t[a]))
            for a, b in zip(edges[::2], edges[1::2])]


def motion_series(
    intensity_raws: dict[str, "mne.io.Raw"],
    after_raws: "dict[str, mne.io.Raw] | None",
    subject_ids: list[str],
    sep_bands=None,
) -> dict:
    """Everything the two motion figures draw, measured once off the aligned recordings.

    ::

      -> {"subject_ids": [...], "t": (39610,), "sets": ["long", "short"],
          "stages": ["before", "after"],
          "series": {"before": {"long": [("sub-01", (39610,)), ...]}, ...},
          "spikes_both": {"before": [(412.0, 3.1), ...]},
          "carpets": {"before": [("sub-01", z, t, labels, spans)]},
          "y_tops": {"long": 11.4}, "divisors": {("sub-01", "long"): 4.1e-04}}

    **Each member is divided by its own before-median, and the corrected traces are divided
    by that same number.** GVTD is an RMS of optical-density derivatives in the recording's
    own units, so two members' raw traces share no scale; at x its own median, 1.0 is that
    member's usual level for both of them. Dividing the corrected trace by its *own* median
    would divide out the shrinkage the second figure shows.

    One y range per channel set, shared by that set's before and after rows, which is what
    makes the correction readable as a drop. Long and short do not share one: each pair of
    (member, set) is divided by its own median, so a "x median" on the long channels is not
    the same quantity as one on the short.

    Spikes are kept only where **every** member was spiking at once; a member spiking alone
    shows on the usable-time carpet.

    Returns ``{}`` when no member carries usable optical density.
    """
    have = [sid for sid in subject_ids if sid in intensity_raws]
    if not have:
        return {}

    stages: list[str] = ["before"]
    series: dict[str, dict[str, list]] = {"before": {}}
    carpets: dict[str, list] = {"before": []}
    spikes: dict[str, dict[str, np.ndarray]] = {"before": {}}
    divisors: dict[tuple, float] = {}
    sets: list[str] = []
    t = None

    for sid in have:
        raw = intensity_raws[sid]
        try:
            od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        except Exception:
            logger.warning("%s: optical density failed; no motion row", sid, exc_info=True)
            continue
        sfreq = float(od.info["sfreq"])
        times = od.times
        if t is None or len(times) - 1 < len(t):
            t = times[1:]

        after_od = (after_raws or {}).get(sid)
        blocks = [(name, [c for c in names if c in od.ch_names])
                  for name, names in gvtd_channel_blocks(raw, sep_bands)]
        blocks = [(name, names) for name, names in blocks if names]
        ordered = [c for _, names in blocks for c in names]
        od_data = od.get_data(picks=ordered)
        after_data = _matched_after(after_od, ordered, od_data.shape, sfreq, sid)

        start = 0
        carpet_spans = []
        for name, names in blocks:
            rows = slice(start, start + len(names))
            carpet_spans.append((name, start, start + len(names) - 1))
            start += len(names)
            if name not in sets:
                sets.append(name)
            g = gvtd_timetrace(od_data[rows], sfreq, *GVTD_MOTION_BAND)
            mid = float(np.nanmedian(g))
            mid = mid if np.isfinite(mid) and mid > 0 else 1.0
            divisors[(sid, name)] = mid
            series["before"].setdefault(name, []).append((sid, g / mid))
            if after_data is not None:
                if "after" not in stages:
                    stages.append("after")
                    series["after"], carpets["after"], spikes["after"] = {}, [], {}
                after_g = gvtd_timetrace(after_data[rows], sfreq, *GVTD_MOTION_BAND)
                series["after"].setdefault(name, []).append((sid, after_g / mid))

        # spikes on the canonical set only: the test is ">= 10% of *these* channels"
        canonical = blocks[0][1]
        for stage, source in (("before", od), ("after", after_od)):
            if stage not in stages or source is None:
                continue
            try:
                picked = source.copy().pick([c for c in canonical if c in source.ch_names])
                spikes[stage][sid] = _spans_to_mask(spike_segments(picked), t)
            except Exception:
                logger.warning("%s: spike spans failed on the %s file", sid, stage,
                               exc_info=True)

        z, t_carpet, stats = carpet_z(od_data, times)
        carpets["before"].append((sid, z, t_carpet, ordered, carpet_spans))
        if after_data is not None:
            carpets["after"].append(
                (sid, carpet_z(after_data, times, stats=stats)[0], t_carpet,
                 ordered, carpet_spans))

    if not sets or t is None:
        return {}

    # a stage only one member has is worse than no stage: the reader would be handed a
    # "both at once" floor drawn from a single trace
    if "after" in stages and any(
            len(series["after"].get(name) or []) != len(series["before"].get(name) or [])
            for name in sets):
        logger.warning("only some members have a motion-corrected file; dropping the "
                       "corrected figure rather than drawing one member in it")
        stages.remove("after")

    # aligned recordings are one length by construction, but a member read from a different
    # stage can be a sample short; one length here keeps the pointwise minimum defined
    n = min(len(y) for stage in stages for v in series[stage].values() for _, y in v)
    n = min(n, len(t))
    t = t[:n]
    for stage in stages:
        series[stage] = {name: [(sid, y[:n]) for sid, y in v]
                         for name, v in series[stage].items()}

    # one range a set, over both stages, so the after figure's rows are not rescaled to
    # their own smaller numbers and the drop disappears
    y_tops = {}
    for name in sets:
        flat = [y for stage in stages for _, y in series[stage].get(name, [])]
        y_tops[name] = (float(np.nanpercentile(np.concatenate(flat), _MOTION_CAP_PCTL))
                        * _MOTION_HEADROOM) if flat else 1.0

    spikes_both = {}
    for stage in stages:
        lanes = [m[:n] for m in spikes.get(stage, {}).values() if m is not None]
        # every member, not any; one member's spikes show on the usable-time carpet
        both = (np.logical_and.reduce(lanes)
                if len(lanes) == len(have) and len(lanes) > 1
                else np.zeros(n, dtype=bool))
        spikes_both[stage] = _merge_spans(_mask_to_spans(both, t))

    return {"subject_ids": have, "t": t, "sets": sets, "stages": stages,
            "series": series, "spikes_both": spikes_both, "carpets": carpets,
            "y_tops": y_tops, "divisors": divisors}


def _matched_after(after_od, ch_names, shape, sfreq, sid):
    """The corrected recording over ``ch_names``, or None if it does not line up.

    A near miss is worse than nothing: GVTD over a different channel set would show in the
    second figure as an effect of the correction.
    """
    if after_od is None:
        return None
    if not set(ch_names) <= set(after_od.ch_names):
        logger.warning("%s: the corrected file is missing channels the panel draws; "
                       "no after figure", sid)
        return None
    if abs(float(after_od.info["sfreq"]) - sfreq) > 1e-6:
        logger.warning("%s: the corrected file is at a different sampling rate; "
                       "no after figure", sid)
        return None
    data = after_od.get_data(picks=ch_names)
    if data.shape != shape:
        logger.warning("%s: the corrected file has a different length; no after figure", sid)
        return None
    return data


def build_motion_panel(
    motion: dict,
    stage: str = "before",
    conditions: "dict[str, tuple[float, float]] | None" = None,
) -> "go.Figure | None":
    """One stage of the motion panel: a GVTD row per channel set, both members in each.

    ::

      build_motion_panel(motion, "after", conditions)  ->  figure

    Simultaneous motion is drawn as the pointwise minimum of the two traces, filled to the
    axis: high only where both are high, and needing no threshold to be picked.

    Under the rows sits the spike strip, marking only the spans where every member was
    spiking at once, and under that each member's z-scored optical-density carpet, drawn by
    the subject report's own :func:`~fnirs_pipe.qc.figures.common.motion_panel.add_carpet` so the
    dyad's image and the member's own cannot drift apart.

    Row titles and the run's numbers sit in the left margin rather than inside the panels,
    where a noisy recording's data would cover them.

    Returns None for a stage the dyad has no data for.
    """
    if not motion or stage not in motion.get("stages", []):
        return None
    sids = motion["subject_ids"]
    t = motion["t"]
    sets = [s for s in motion["sets"] if motion["series"][stage].get(s)]
    if not sets:
        return None

    spans_both = motion["spikes_both"].get(stage) or []
    carpets = motion["carpets"].get(stage) or []
    has_cond = bool(conditions)
    has_spikes = bool(spans_both)
    n_rows = int(has_cond) + len(sets) + int(has_spikes) + len(carpets)

    # the spike strip sits directly under the condition bar, above the traces, as one band
    heights = (([TIMELINE_ROW_PX - 10] if has_cond else [])
               + ([_SPIKE_ROW_PX] if has_spikes else [])
               + [_MOTION_ROW_PX] * len(sets)
               + [_MOTION_CARPET_PX] * len(carpets))
    vspace = 0.022
    row_heights, total_px = _px_rows(heights, vspace, chrome_px=150)
    titles = ([""] * (n_rows - len(carpets))
              + [f"{sid} carpet ({stage})" for sid, *_ in carpets])
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, row_heights=row_heights,
                        vertical_spacing=vspace, subplot_titles=titles)
    for ann in fig.layout.annotations:
        ann.yshift = 3

    ri = 1
    if has_cond:
        colours = _cond_colors(list(conditions))
        for name, (a, b) in conditions.items():
            fig.add_trace(go.Bar(
                x=[b - a], y=[0], base=[a], orientation="h", width=0.55,
                marker=dict(color=colours[name], opacity=0.9, line=dict(width=0)),
                name=name, legendgroup=name, showlegend=True,
                hovertemplate=f"<b>{name}</b><br>%{{base:.0f}} to {b:.0f} s<extra></extra>",
            ), row=1, col=1)
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[-0.5, 0.5],
                         row=1, col=1)
        ri = 2

    if has_spikes:
        xs, ys = _span_polygons(spans_both, 0.15, 0.85)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=_SPIKE_BOTH, mode="lines",
            line=dict(width=0), hoverinfo="skip", name="both spiking",
            legendgroup="spikes", showlegend=True), row=ri, col=1)
        fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False,
                         zeroline=False, row=ri, col=1)
        _margin_label(fig, ri, "<b>both spiking</b>")
        ri += 1

    shown: set[str] = set()
    for name in sets:
        entries = motion["series"][stage][name]
        y_top = motion["y_tops"][name]

        if len(entries) > 1:
            mins = np.minimum.reduce([y for _, y in entries])
            t_ds, m_ds = _maxpool_xy(t, mins)
            fig.add_trace(go.Scatter(
                x=t_ds, y=m_ds, mode="lines", fill="tozeroy", fillcolor=_TOGETHER_FILL,
                line=dict(width=0), name="both at once", legendgroup="both",
                showlegend="both" not in shown,
                hovertemplate="t=%{x:.0f}s<br>both >= %{y:.2f}x<extra></extra>",
            ), row=ri, col=1)
            shown.add("both")

        for sid, y in entries:
            t_ds, y_ds = _maxpool_xy(t, y)
            fig.add_trace(go.Scatter(
                x=t_ds, y=y_ds, mode="lines", name=sid, legendgroup=sid,
                showlegend=sid not in shown, opacity=0.9,
                line=dict(color=_MEMBER_COLOURS[sids.index(sid) % len(_MEMBER_COLOURS)],
                          width=1.5),
                hovertemplate=f"<b>{sid}</b><br>t=%{{x:.0f}}s<br>"
                              "%{y:.2f}x its own median<extra></extra>",
            ), row=ri, col=1)
            shown.add(sid)

        # 1.0 is where this member usually sits, by construction; it is a reference and not
        # a cutoff, and nothing on the page is counted against it
        fig.add_hline(y=1.0, row=ri, col=1,
                      line=dict(color="#c8cfd6", width=1, dash="dot"))
        fig.update_yaxes(range=[0, y_top], tickfont=dict(size=8), gridcolor="#eef1f4",
                         zeroline=False, row=ri, col=1)
        _margin_label(fig, ri, f"<b>GVTD {name}</b>")
        ri += 1

    for sid, z, t_carpet, labels, spans in carpets:
        add_carpet(fig, ri, z, t_carpet, labels, spans)
        ri += 1

    for a, b in (conditions or {}).values():
        for edge in (a, b):
            fig.add_vline(x=edge, line_color="#e3e8ec", line_width=1, line_dash="dot")

    fig.update_xaxes(gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9))
    fig.update_xaxes(range=[float(t[0]), float(t[-1])])
    fig.update_xaxes(title_text="Time on the shared clock (s)", title_font=dict(size=10),
                     row=n_rows, col=1)
    fig.update_layout(height=total_px, plot_bgcolor="white", barmode="overlay",
                      coloraxis=carpet_coloraxis(y=0.2),
                      margin=dict(l=112, r=24, t=56, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.012,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


def _margin_label(fig, row: int, text: str) -> None:
    """A row's name, parked in the left margin clear of the data.

    The row is named through ``yref`` rather than through ``row=``: passing the latter makes
    plotly rewrite ``xref`` to that subplot's x axis, and an axis-referenced annotation
    sitting at x=0 on an axis that starts at 0.1 s is clipped away silently.
    """
    # xshift clears the tick labels, which sit between the axis and the margin
    fig.add_annotation(x=0, xref="paper", xshift=-36,
                       y=0.5, yref=f"y{row if row > 1 else ''} domain",
                       text=text, showarrow=False, xanchor="right",
                       yanchor="middle", align="right", font=dict(size=10))


# ---------------------------------------------------------------------------
# Figure: where on the head
# ---------------------------------------------------------------------------

_HEAD_SCALE = [[0.0, _BAD_COLOR], [0.5, _MIX_COLOR], [1.0, _GOOD_COLOR]]


def _pair_values(geo, scope, by_pair: dict, reduce_fn) -> "np.ndarray":
    """One value per channel this head draws, NaN where the dyad grid has no such pair."""
    return np.array([reduce_fn(by_pair[name]) if name in by_pair else np.nan
                     for name in geo[scope]["names"]])


def build_head_by_condition(
    geo_by_sub: dict,
    subject_ids: list[str],
    grid: dict,
    conditions: dict,
) -> "go.Figure | None":
    """One head per member per block, coloured by the share of its windows that coupled.

    The aggregated view: which part of whose cap went and in which block. Complements the
    usable-time carpet, which is channel by time with no geometry.
    """
    names = list(conditions)
    if not names or not geo_by_sub:
        return None
    t = np.asarray(grid["t"], dtype=float)
    fig = make_subplots(rows=len(subject_ids), cols=len(names),
                        subplot_titles=names + [""] * (len(names) * (len(subject_ids) - 1)),
                        horizontal_spacing=0.015, vertical_spacing=0.04)
    for ri, sid in enumerate(subject_ids, start=1):
        geo = geo_by_sub.get(sid)
        if geo is None:
            continue
        by_pair = dict(zip(grid["pairs"], np.asarray(grid["ok"][sid], dtype=float)))
        for ci, cond in enumerate(names, start=1):
            a, b = conditions[cond]
            sel = (t >= a) & (t <= b)
            _head_ground(fig, geo, ri, ci)
            for scope in ("long", "short"):
                if scope not in geo:
                    continue
                vals = _pair_values(geo, scope, by_pair, lambda v, s=sel: v[s].mean())
                _head_glyph(fig, geo, scope, vals, ri, ci, "Coupled<br>windows", 0.0, 1.0,
                            bar=(ri == 1 and ci == len(names) and scope == "long"),
                            colorscale=_HEAD_SCALE, sid=sid)
    _head_axes(fig, geo_by_sub, len(subject_ids), len(names), row_labels=subject_ids)
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    # a head per member per block, so the grid is as wide as the blocks and only as tall as
    # the members; the height is what decides how big each head is drawn
    fig.update_layout(height=230 * len(subject_ids) + 46, plot_bgcolor="white",
                      showlegend=False, margin=dict(l=104, r=78, t=46, b=14))
    return fig


def build_head_slider(
    geo_by_sub: dict,
    subject_ids: list[str],
    grid: dict,
    series_by_sub: dict,
    conditions: "dict | None" = None,
    step: int = 3,
    cmin: float = 0.6,
    cmax: float = 1.0,
) -> "go.Figure | None":
    """One head per member, the slider running the whole recording on the shared clock.

    The instantaneous view, coloured by that window's SCI rather than by a share. Every frame
    names the block it lands in, so a position on the slider says what the dyad was doing and
    not only when.

    ``step`` decimates the frames. Each one carries a colour per marker per member, so the
    page grows with the frame count.
    """
    if not geo_by_sub:
        return None
    t = np.asarray(grid["t"], dtype=float)
    frames_at = list(range(0, len(t), max(1, step)))
    if not frames_at:
        return None
    conditions = conditions or {}

    def task_at(when: float) -> str:
        for name, (a, b) in conditions.items():
            if a <= when <= b:
                return name
        return "between blocks"

    def caption(k: int) -> dict:
        """The block the slider sits in, as the figure's title.

        A frame that hands back ``layout.annotations`` overwrites the layout's own array
        **by index**, so a one-element caption lands on subplot title 0 and the real caption,
        sitting at index 2, never updates: the first head loses its name and the caption is
        frozen at the first frame. A title has no index to collide with.
        """
        return dict(text=f"<b>{task_at(t[k])}</b>"
                         f"<span style='color:#8a949e'> &nbsp;t = {t[k]:.0f} s</span>",
                    x=0.5, xanchor="center", y=0.98, yanchor="top",
                    font=dict(size=13, color="#34495e"))

    fig = make_subplots(rows=1, cols=len(subject_ids), subplot_titles=list(subject_ids),
                        horizontal_spacing=0.04)
    drawn: list[tuple] = []
    for ci, sid in enumerate(subject_ids, start=1):
        geo = geo_by_sub.get(sid)
        sci = (series_by_sub.get(sid) or {}).get("per_pair_sci")
        if geo is None or sci is None:
            continue
        _head_ground(fig, geo, 1, ci)
        for scope in ("long", "short"):
            if scope not in geo:
                continue
            vals = _pair_values(geo, scope, sci, lambda v, k=frames_at[0]: v[k])
            i = _head_glyph(fig, geo, scope, vals, 1, ci, "SCI<br>(10 s)", cmin, cmax,
                            bar=(ci == len(subject_ids) and scope == "long"),
                            colorscale=_HEAD_SCALE, sid=sid)
            drawn.append((sid, scope, i))
    if not drawn:
        return None

    fig.frames = [
        go.Frame(name=f"{t[k]:.0f}",
                 data=[go.Scatter(marker=dict(color=np.repeat(
                     _pair_values(geo_by_sub[sid], scope,
                                  series_by_sub[sid]["per_pair_sci"],
                                  lambda v, kk=k: v[kk]),
                     geo_by_sub[sid][scope]["per_pair"]).astype(np.float32)))
                       for sid, scope, _i in drawn],
                 traces=[i for _s, _sc, i in drawn],
                 layout=dict(title=caption(k)))
        for k in frames_at
    ]
    _head_axes(fig, geo_by_sub, 1, len(subject_ids))
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(
        height=460, plot_bgcolor="white", showlegend=False,
        margin=dict(l=20, r=78, t=78, b=76),
        title=caption(frames_at[0]),
        sliders=[dict(active=0, y=0, yanchor="top", pad=dict(t=30, b=4),
                      currentvalue=dict(visible=False), font=dict(size=9),
                      steps=[dict(method="animate", label=fr.name,
                                  args=[[fr.name],
                                        dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))])
                             for fr in fig.frames])])
    return fig


# ---------------------------------------------------------------------------
# Figure: screening synchrony against its null
# ---------------------------------------------------------------------------

def build_screening_strip(coherence_df: "pd.DataFrame") -> "go.Figure | None":
    """Each window's coherence as its rank inside its own surrogate null, one row per window.

    **Raw coherence cannot share an axis across windows.** The estimator's floor sits near
    1/(number of Welch segments), and that count falls with the window. A value's percentile
    inside the null drawn for *that* window puts every window on one axis with one line to
    clear.

    One pale dot per channel, a diamond for the channel mean, and the top 5% shaded.

    Expects the frame :func:`~fnirs_pipe.pipeline.hyper.coherence.screening_coherence` returns.
    None when it is empty.
    """
    if coherence_df is None or coherence_df.empty:
        return None
    windows = list(dict.fromkeys(coherence_df["window"]))
    rows = list(reversed(windows))
    jitter = np.random.default_rng(0)
    # each block in the colour panel 2 and 3 give it, so a reader carries one mapping across
    # the page; the whole run is not a block and stays neutral
    blocks = [w for w in windows if w != "whole run"]
    colours = {**_cond_colors(blocks), "whole run": "#7f8c8d"}

    fig = go.Figure()
    fig.add_vrect(x0=NULL_ALPHA_PCT, x1=100, fillcolor="#3498db", opacity=0.07, line_width=0)
    fig.add_vline(x=NULL_ALPHA_PCT, line_color="#adb5bd", line_width=1, line_dash="dot")
    for i, name in enumerate(rows):
        sub = coherence_df[coherence_df["window"] == name]
        pct = sub["percentile"].to_numpy(dtype=float)
        colour = colours.get(name, "#7f8c8d")
        fig.add_trace(go.Scatter(
            x=pct, y=i + jitter.uniform(-0.13, 0.13, len(pct)), mode="markers",
            name=str(name), legendgroup=str(name), showlegend=False,
            customdata=sub["ch_name"].tolist(),
            # saturated only above the line; the rest are drawn as grey spread
            marker=dict(size=8,
                        color=[colour if p >= NULL_ALPHA_PCT else "#d7dde2" for p in pct],
                        opacity=[0.95 if p >= NULL_ALPHA_PCT else 0.75 for p in pct],
                        line=dict(width=0.6, color="#fff")),
            hovertemplate=("<b>%{customdata}</b><br>" + str(name)
                           + "<br>%{x:.1f}th percentile of its null<extra></extra>"),
        ))
    # the window's own rank (channels pooled, ranked once), not the mean of its channels' ranks
    means = [float(coherence_df[coherence_df["window"] == n]["window_percentile"].iloc[0])
             for n in rows]
    fig.add_trace(go.Scatter(
        x=means, y=list(range(len(rows))), mode="markers", name="channel mean",
        customdata=rows, showlegend=False,
        marker=dict(size=13, symbol="diamond",
                    color=[colours.get(n, "#7f8c8d") for n in rows],
                    line=dict(width=1.2, color="#fff")),
        hovertemplate="%{customdata}<br>mean at the %{x:.1f}th percentile<extra></extra>"))

    fig.update_xaxes(title_text="Percentile inside its own phase-scrambled null",
                     title_font=dict(size=10), range=[-2, 102], dtick=25,
                     gridcolor="#f5f5f5", zeroline=False, tickfont=dict(size=9))
    fig.update_yaxes(tickvals=list(range(len(rows))), ticktext=rows, showgrid=False,
                     range=[-0.6, len(rows) - 0.4], tickfont=dict(size=9))
    fig.update_layout(height=110 + 40 * len(rows), plot_bgcolor="white",
                      margin=dict(l=110, r=24, t=52, b=48),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    return fig


# ---------------------------------------------------------------------------
# Figure: trigger timeline
# ---------------------------------------------------------------------------

def build_trigger_timeline(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    xlabel: str = "Time (s) [aligned]",
) -> go.Figure | None:
    per_sub: dict[str, list[dict]] = {}
    all_descs: list[str] = []
    for sid in subject_ids:
        raw = aligned_raws.get(sid)
        mkrs = _extract_markers(raw) if raw else []
        per_sub[sid] = mkrs
        for m in mkrs:
            if m["description"] not in all_descs:
                all_descs.append(m["description"])

    if not any(per_sub.values()):
        return None

    colors = _cond_colors(all_descs)
    traces: list[go.BaseTraceType] = []
    seen: set[str] = set()

    for sub_idx, sid in enumerate(subject_ids):
        by_desc: dict[str, list[dict]] = {}
        for m in per_sub[sid]:
            by_desc.setdefault(m["description"], []).append(m)
        for desc, events in by_desc.items():
            traces += timeline_row_traces(events, sub_idx, colors.get(desc, "#999"),
                                          desc, desc not in seen, f"<br>{sid}")
            seen.add(desc)

    # the legend stays here, unlike the single-recording timeline: a row is a member and
    # the conditions sharing it are told apart by colour alone
    xaxis, yaxis = timeline_axes([f"sub-{s}" for s in subject_ids])
    xaxis["title"] = xlabel
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=xaxis, yaxis=yaxis,
            shapes=timeline_row_bands(len(subject_ids)),
            plot_bgcolor="white", paper_bgcolor="white",
            # overlay, or plotly groups each subject's bars and shifts them off their row
            barmode="overlay",
            height=max(110, len(subject_ids) * TIMELINE_ROW_PX + 88),
            margin=dict(l=80, r=20, t=10, b=44),
            legend=dict(font=dict(size=9), orientation="h", y=-0.35),
            hovermode="closest",
        ),
    )


# ---------------------------------------------------------------------------
# Figure: HbO + HbR signal overlay (2-row subplot, channel switching via select)
# ---------------------------------------------------------------------------

def build_signal_overlay(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    markers_list: list[dict],
    cond_colors_: dict[str, str],
) -> go.Figure | None:
    if not aligned_raws:
        return None
    ref_raw = aligned_raws.get(subject_ids[0])
    if ref_raw is None:
        return None
    hbo_picks = mne.pick_types(ref_raw.info, fnirs="hbo")
    if not len(hbo_picks):
        return None

    ch_pairs = [ref_raw.ch_names[p].rsplit(" ", 1)[0] for p in hbo_picks]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.06,
        subplot_titles=["HbO", "HbR"],
    )

    # HbO traces (row 1) then HbR traces (row 2): n_ch * n_subs each
    for row_idx, hb_type in enumerate(["hbo", "hbr"], start=1):
        for ch_idx, pair in enumerate(ch_pairs):
            ch_name = f"{pair} {hb_type}"
            for sub_idx, sid in enumerate(subject_ids):
                raw = aligned_raws.get(sid)
                if raw is None or ch_name not in raw.ch_names:
                    t_vals, y_vals = [], []
                else:
                    pick = raw.ch_names.index(ch_name)
                    arr, times = _decimate(
                        raw.get_data(picks=[pick]), raw.times, _MAX_TS_PTS
                    )
                    t_vals = times.tolist()
                    y_vals = (arr[0] * 1e6).tolist()
                fig.add_trace(go.Scatter(
                    x=t_vals, y=y_vals,
                    name=sid, mode="lines",
                    line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.3),
                    visible=(ch_idx == 0),
                    showlegend=False,
                    legendgroup=sid,
                ), row=row_idx, col=1)

    # Ghost traces for stable subject legend
    for sub_idx, sid in enumerate(subject_ids):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=sid,
            line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.5),
            showlegend=True, visible=True,
            legendgroup=sid,
        ), row=1, col=1)

    mkr_shapes = []
    for m in markers_list:
        color = cond_colors_.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            mkr_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(width=0), layer="below",
            ))

    fig.update_layout(
        shapes=mkr_shapes,
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=60, r=15, t=30, b=40),
        legend=dict(font=dict(size=9)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_xaxes(title_text="Time (s) [aligned]", row=2, col=1)
    fig.update_yaxes(title_text="µmol/L", gridcolor="#eeeeee")

    return fig


def build_signal_overlay_pair(
    aligned_raws: dict[str, mne.io.Raw],
    subject_ids: list[str],
    pair: str,
    markers_list: list[dict],
    cond_colors_: dict[str, str],
) -> go.Figure | None:
    """Single-channel-pair signal overlay (multi-subject), for iframe per-channel pages."""
    if not aligned_raws:
        return None

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5], vertical_spacing=0.06,
        subplot_titles=["HbO", "HbR"],
    )

    for row_idx, hb_type in enumerate(["hbo", "hbr"], start=1):
        ch_name = f"{pair} {hb_type}"
        for sub_idx, sid in enumerate(subject_ids):
            raw = aligned_raws.get(sid)
            if raw is None or ch_name not in raw.ch_names:
                continue
            pick = raw.ch_names.index(ch_name)
            arr, times = _decimate(
                raw.get_data(picks=[pick]), raw.times, _MAX_TS_PTS
            )
            fig.add_trace(go.Scatter(
                x=times.tolist(), y=(arr[0] * 1e6).tolist(),
                name=sid, mode="lines",
                line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.3),
                showlegend=(row_idx == 1),
                legendgroup=sid,
            ), row=row_idx, col=1)

    mkr_shapes = []
    for m in markers_list:
        color = cond_colors_.get(m["description"], "#f39c12")
        if m["duration"] > 0.1:
            mkr_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=m["onset"], x1=m["onset"] + m["duration"], y0=0, y1=1,
                fillcolor=_hex_to_rgba(color, 0.08),
                line=dict(width=0), layer="below",
            ))

    fig.update_layout(
        shapes=mkr_shapes,
        plot_bgcolor="white", paper_bgcolor="white",
        height=400, margin=dict(l=60, r=15, t=30, b=40),
        legend=dict(font=dict(size=9)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_xaxes(title_text="Time (s) [aligned]", row=2, col=1)
    fig.update_yaxes(title_text="µmol/L", gridcolor="#eeeeee")
    return fig


# ---------------------------------------------------------------------------
# Figure: per-channel PSD (HbO, multi-subject)
# ---------------------------------------------------------------------------

def build_psd(
    aligned_raws: dict[str, mne.io.Raw],
    ch_pair: str,
    subject_ids: list[str],
    cardiac: "tuple[float, float] | None" = None,
) -> go.Figure | None:
    hbo_name = f"{ch_pair} hbo"
    traces: list[go.BaseTraceType] = []

    for sub_idx, sid in enumerate(subject_ids):
        raw = aligned_raws.get(sid)
        if raw is None or hbo_name not in raw.ch_names:
            continue
        pick = raw.ch_names.index(hbo_name)
        arr = raw.get_data(picks=[pick])
        sfreq = raw.info["sfreq"]
        psds, freqs = mne.time_frequency.psd_array_welch(
            arr, sfreq, n_fft=min(PSD_NFFT, arr.shape[1]), verbose=False)
        psd = psds[0]
        fmax = min(2.0, float(sfreq / 2))
        mask = freqs <= fmax
        traces.append(go.Scatter(
            x=freqs[mask].tolist(), y=psd[mask].tolist(),
            name=sid, mode="lines",
            line=dict(color=_SUB_COLORS[sub_idx % len(_SUB_COLORS)], width=1.8),
        ))

    if not traces:
        return None

    shapes, annots = _psd_band_shapes(cardiac)
    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(title="Frequency (Hz)", range=[0, 2], gridcolor="#eeeeee"),
            yaxis=dict(title="Power (HbO)", type="log", gridcolor="#eeeeee"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=200, margin=dict(l=60, r=15, t=22, b=38),
            legend=dict(font=dict(size=9), orientation="h",
                        x=1, xanchor="right", y=1.0, yanchor="bottom"),
            shapes=shapes, annotations=annots,
        ),
    )


# ---------------------------------------------------------------------------
# Figure: channel quality summary (group status, channels on x-axis)
# ---------------------------------------------------------------------------

def build_channel_summary(
    sqm_data: dict[str, dict],
    subject_ids: list[str],
    sci_threshold: float,
) -> go.Figure | None:
    ch_set: set[str] = set()
    for sid in subject_ids:
        for k in sci_of(sqm_data, sid):
            ch_set.add(k.rsplit(" ", 1)[0] if " " in k else k)

    if not ch_set:
        return None

    ch_names = sorted(ch_set)
    colors, hover_texts = [], []

    for ch in ch_names:
        statuses = _ch_kept_by_member(ch, sqm_data, subject_ids, sci_threshold)
        known    = [s for s in statuses if s is not None]
        sci_vals = []
        for sid in subject_ids:
            sci_d = sci_of(sqm_data, sid)
            val   = sci_d.get(f"{ch} hbo") or sci_d.get(ch)
            sci_vals.append(f"{sid}: {val:.3f}" if val is not None else f"{sid}: N/A")

        if not known:
            c, status = _NA_COLOR, "N/A"
        elif all(known):
            c, status = _GOOD_COLOR, "all good"
        elif not any(known):
            c, status = _BAD_COLOR, "all bad"
        else:
            c, status = _MIX_COLOR, "mixed"

        colors.append(c)
        hover_texts.append(f"<b>{ch}</b> ({status})<br>" + "<br>".join(sci_vals))

    traces: list[go.BaseTraceType] = [
        go.Scatter(
            x=ch_names, y=[0] * len(ch_names),
            mode="markers",
            marker=dict(symbol="square", size=16, color=colors,
                        line=dict(width=0.8, color="#bbb")),
            hovertext=hover_texts, hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )
    ]
    for color, label in [(_GOOD_COLOR, "All good"), (_MIX_COLOR, "Mixed"),
                          (_BAD_COLOR, "All bad"), (_NA_COLOR, "N/A")]:
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color=color,
                        line=dict(width=1, color="#bbb")),
            name=label, showlegend=True,
        ))

    return go.Figure(
        data=traces,
        layout=go.Layout(
            xaxis=dict(tickfont=dict(size=8), showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=[-1, 1]),
            plot_bgcolor="white", paper_bgcolor="white",
            height=120, margin=dict(l=20, r=20, t=8, b=60),
            legend=dict(font=dict(size=9), orientation="h", y=-0.55),
        ),
    )


# ---------------------------------------------------------------------------
# Compute: group-level SQM scalars
# ---------------------------------------------------------------------------

