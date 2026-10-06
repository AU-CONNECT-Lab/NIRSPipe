import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.metrics import CV_PASS, PSP_PASS, SCI_PASS, SNR_PASS
from fnirs_pipe.qc.metrics._helpers import GOOD_FRAC_PASS
from fnirs_pipe.qc.metrics.windowed import window_centers
from fnirs_pipe.utils.logging import get_logger

from fnirs_pipe.qc.figures.common._utils import AXIS_TEXT_COLOR
from fnirs_pipe.qc.boilerplate.vocabulary import higher_is_better, format_metric

logger = get_logger("qc.figures.sci_psp")

_GOOD_COLOR = "#C5E0B3"
_BAD_COLOR  = "#F8786E"
_MAX_TS_PTS = 3000
_SPACING    = 3

# '#D3D3D3',  # NA - light gray
# '#FFD966',  # other - yellow

def binary_heatmap_figure(
    ch_names: list[str],
    good_mask: np.ndarray,
    metric_label: str = "Pass/Fail",
) -> go.Figure:
    n_ch = len(ch_names)
    colors = [_GOOD_COLOR if g else _BAD_COLOR for g in good_mask]

    fig = go.Figure(go.Scatter(
        x=[0] * n_ch,
        y=[i * _SPACING for i in range(n_ch)],
        mode="markers",
        marker=dict(
            symbol="square",
            size=12,
            color=colors,
            line=dict(width=0.5, color="#aaa"),
        ),
        text=[f"{n}: {'PASS' if g else 'FAIL'}" for n, g in zip(ch_names, good_mask)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 0.5]),
        yaxis=dict(
            tickvals=[i * _SPACING for i in range(n_ch)],
            ticktext=ch_names,
            autorange="reversed",
            tickfont=dict(size=9),
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=30, b=40),
    )
    return fig


_MISSING_COLOR = "#D3D3D3"


def _channel_metric_rows(sci_thresh, cv_thresh, snr_thresh, psp_thresh, good_frac_thresh):
    """(row label, heatmap_args key, hover format, pass test) for every channel-grid row.

    Status has no key and no test: it is read off ``is_bad``, the screening verdict.
    """
    return [
        ("Status",  None,               None,  None),
        ("Coupled", "good_frac_per_ch", ".3f", lambda v: v >= good_frac_thresh),
        ("SCI",     "sci_per_ch",       ".3f", lambda v: v >= sci_thresh),
        ("CV",      "cv_per_ch",        ".3f", lambda v: v <= cv_thresh),
        ("PSP",     "psp_per_ch",       ".3f", lambda v: v >= psp_thresh),
        ("SNR",     "snr_per_ch",       ".1f", lambda v: v >= snr_thresh),
    ]


def _channel_cell(metric, value, fmt, check, ch) -> tuple[str, str]:
    """Colour and hover for one cell; ``value`` is the bad flag on the Status row."""
    if metric == "Status":
        if value is None:
            return _MISSING_COLOR, f"{ch} · Status: —"
        return (_BAD_COLOR if value else _GOOD_COLOR), f"{ch} · Status: {'BAD' if value else 'OK'}"
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _MISSING_COLOR, f"{ch} · {metric}: —"
    return (_GOOD_COLOR if check(value) else _BAD_COLOR), f"{ch} · {metric}: {value:{fmt}}"


def _channel_xaxis(ch_names: list[str]) -> dict:
    n_ch = len(ch_names)
    return dict(
        tickvals=[i * _SPACING for i in range(n_ch)],
        ticktext=ch_names,
        tickangle=-45, tickfont=dict(size=9, color=AXIS_TEXT_COLOR),
        showgrid=False, zeroline=False,
        range=[-_SPACING, n_ch * _SPACING],
    )


def _long_short_divider(fig: go.Figure, split_at: "int | None", n_ch: int) -> bool:
    """Draw the long/short divider and name both sides; False when there is no split."""
    if split_at is None or not 0 < split_at < n_ch:
        return False
    # the line sits in the gap between the last long column and the first short one
    x_div = (split_at - 0.5) * _SPACING
    fig.add_vline(x=x_div, line_dash="dot", line_color="#888", line_width=1)
    # pixels above the plot rather than a paper fraction, so a tall grid does not lift them
    # out of the margin
    for centre, text in (((split_at - 1) / 2, "long"),
                         ((split_at + n_ch - 1) / 2, "short")):
        fig.add_annotation(x=centre * _SPACING, y=1.0, yshift=14, xref="x", yref="paper",
                           text=text, showarrow=False, font=dict(size=10, color="#666"))
    return True


def channel_quality_heatmap(
    ch_names: list[str],
    is_bad: list[bool],
    sci_per_ch: dict[str, float],
    cv_per_ch: dict[str, float],
    snr_per_ch: dict[str, float],
    psp_per_ch: dict[str, float],
    good_frac_per_ch: "dict[str, float] | None" = None,
    sci_thresh: float = SCI_PASS,
    cv_thresh: float = CV_PASS,
    snr_thresh: float = SNR_PASS,
    psp_thresh: float = PSP_PASS,
    good_frac_thresh: float = GOOD_FRAC_PASS,
    split_at: int | None = None,
) -> go.Figure:
    """Square-marker grid: channels on x-axis, metrics on y-axis. Green=pass, red=fail, gray=missing.

    ``split_at`` is the index the short channels start at, given when the caller has
    already ordered ``ch_names`` long block first; it draws the divider and names the two
    blocks.

    Status is the screening verdict and Coupled is the row that produces it, which is why it
    sits directly under: a channel can fail on its coupled-window share with every average
    below it comfortable. The rows under
    it are drawn against the cutoffs in :mod:`fnirs_pipe.qc.metrics._helpers` and none of
    them prunes.
    """
    lookups = {"good_frac_per_ch": good_frac_per_ch or {}, "sci_per_ch": sci_per_ch,
               "cv_per_ch": cv_per_ch, "snr_per_ch": snr_per_ch, "psp_per_ch": psp_per_ch}
    specs = _channel_metric_rows(sci_thresh, cv_thresh, snr_thresh, psp_thresh,
                                 good_frac_thresh)
    n_ch  = len(ch_names)
    n_met = len(specs)

    xs, ys, colors, hover = [], [], [], []
    for m_idx, (metric, key, fmt, chk) in enumerate(specs):
        for i, ch in enumerate(ch_names):
            value = is_bad[i] if key is None else lookups[key].get(ch)
            color, text = _channel_cell(metric, value, fmt, chk, ch)
            xs.append(i * _SPACING)
            ys.append(m_idx * _SPACING)
            colors.append(color)
            hover.append(text)

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(symbol="square", size=12, color=colors,
                    line=dict(width=0)),
        text=hover,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        xaxis=_channel_xaxis(ch_names),
        yaxis=dict(
            tickvals=[m * _SPACING for m in range(n_met)],
            ticktext=[spec[0] for spec in specs],
            autorange="reversed",
            tickfont=dict(size=10, color=AXIS_TEXT_COLOR),
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=20, t=20, b=100),
        height=260,
    )
    if _long_short_divider(fig, split_at, n_ch):
        # the extra top margin is what the raised labels sit in
        fig.update_layout(margin=dict(l=70, r=20, t=52, b=100))
    return fig


# the pitch of one condition row, in pixels; the figure sets its own height from it
_CONDITION_ROW_PX = 18


def condition_quality_heatmap(
    conditions: "list[tuple[str, dict]]",
    sci_thresh: float = SCI_PASS,
    cv_thresh: float = CV_PASS,
    snr_thresh: float = SNR_PASS,
    psp_thresh: float = PSP_PASS,
    good_frac_thresh: float = GOOD_FRAC_PASS,
) -> "go.Figure | None":
    """Every condition's channel grid, regrouped: one block per metric, one row per condition.

    ``conditions`` pairs each label with the :func:`heatmap_args` dict its own page draws
    from, and the pass rule is :func:`channel_quality_heatmap`'s, so a cell here is that
    page's cell. Grouping by metric puts one channel's conditions in a column, which is the
    comparison this figure exists for.

    Example: [("rest", args_a), ("task", args_b)] -> six blocks (Status ... SNR), each two
    rows deep, channels across.

    Columns follow the first condition's order. Returns None under two conditions, where
    the run's own grid already says everything.
    """
    if len(conditions) < 2:
        return None
    first = conditions[0][1]
    ch_names = first["ch_names"]
    n_ch = len(ch_names)
    specs = _channel_metric_rows(sci_thresh, cv_thresh, snr_thresh, psp_thresh,
                                 good_frac_thresh)
    # one empty row above each block carries the metric's name
    depth = len(conditions) + 1

    xs, ys, colors, hover, tickvals, ticktext = [], [], [], [], [], []
    for m_idx, (metric, key, fmt, chk) in enumerate(specs):
        top = m_idx * depth + 1
        for c_idx, (label, args) in enumerate(conditions):
            y = (top + c_idx) * _SPACING
            tickvals.append(y)
            ticktext.append(label)
            lookup = (dict(zip(args["ch_names"], args["is_bad"])) if key is None
                      else args.get(key) or {})
            for i, ch in enumerate(ch_names):
                color, text = _channel_cell(metric, lookup.get(ch), fmt, chk, ch)
                xs.append(i * _SPACING)
                ys.append(y)
                colors.append(color)
                hover.append(f"{label} · {text}")

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(symbol="square", size=12, color=colors, line=dict(width=0)),
        text=hover, hovertemplate="%{text}<extra></extra>", showlegend=False,
    ))
    for m_idx, (metric, *_rest) in enumerate(specs):
        # in the label column, on the block's empty top row, so it reads as a group heading
        fig.add_annotation(x=0, xref="paper", xanchor="right", xshift=-6,
                           y=m_idx * depth * _SPACING, yref="y",
                           text=f"<b>{metric}</b>", showarrow=False,
                           font=dict(size=10, color=AXIS_TEXT_COLOR))
    n_rows = len(specs) * depth
    left = max(70, 7 * max(len(label) for label, _ in conditions) + 20)
    top_px = 52 if _long_short_divider(fig, first.get("split_at"), n_ch) else 20
    fig.update_layout(
        xaxis=_channel_xaxis(ch_names),
        yaxis=dict(tickvals=tickvals, ticktext=ticktext,
                   tickfont=dict(size=9, color=AXIS_TEXT_COLOR),
                   showgrid=False, zeroline=False,
                   range=[(n_rows - 0.5) * _SPACING, -0.5 * _SPACING]),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=left, r=20, t=top_px, b=100),
        height=top_px + 100 + n_rows * _CONDITION_ROW_PX,
    )
    return fig


# key and row label only. The hover format and which end is the better one come from the
# metric registry, which is also where the reports read them, so the two cannot disagree.
# Labels stay local because a heatmap row is a few characters wide and the registry's are
# sentences.
#
# Every key here must carry a direction in the registry. Colour is relative within a row --
# the worse end of what this recording actually did -- so a descriptive metric with no better
# end has no worse end either and cannot be drawn; _trial_metric_specs drops it and says so.
_TRIAL_METRICS = [
    ("sci_mean",               "SCI"),
    ("psp_mean",               "PSP"),
    ("cv_mean",                "CV"),
    ("snr_mean",               "SNR"),
    ("gvtd_filt_mean",         "GVTD"),
    ("channel_retention_rate", "Retention"),
]


def _trial_metric_specs() -> list[tuple[str, str, bool]]:
    """_TRIAL_METRICS resolved against the registry, as (key, label, higher_is_better)."""
    specs = []
    for key, label in _TRIAL_METRICS:
        higher = higher_is_better(key)
        if higher is None:
            logger.warning("%s has no direction in the metric registry; "
                           "leaving it out of the per-trial heatmap", key)
            continue
        specs.append((key, label, higher))
    return specs


# the grid pitch, in pixels. Named rather than derived because the figure sets its own
# width from it, which is what keeps a cell square on a run of five trials
_TRIAL_CELL_PX = 28


def trial_quality_heatmap(
    trial_labels: list[str],
    trial_sqms: list[dict],
) -> go.Figure | None:
    """Trial x metric heatmap: which trials stand out, on which metric.

    Each metric row is scaled to its own min-max across trials, because SCI (0 to 1) and SNR
    (tens to hundreds) share no range and one colour scale over both would flatten SCI to a
    single shade. Colour is therefore relative within a row: red marks the worse end of what
    this recording actually did, not a threshold anyone crossed. Hover carries the real value.

    Example: 40 trials whose SCI holds near 0.9 except trials 12 and 13 gives a mostly uniform
    SCI row with two red cells.

    Returns None when no metric survives on any trial (nothing to draw).
    """
    present = [m for m in _trial_metric_specs()
               if any(isinstance(s.get(m[0]), (int, float)) for s in trial_sqms)]
    if not present or not trial_labels:
        return None

    z, text = [], []
    for key, label, better_high in present:
        raw_vals = [s.get(key) for s in trial_sqms]
        vals = [float(v) if isinstance(v, (int, float)) else None for v in raw_vals]
        good = [v for v in vals if v is not None]
        lo, hi = min(good), max(good)
        span = hi - lo
        if span == 0:
            # a metric that never moved has no worse end to point at; a whole row of red
            # would read as 40 bad trials rather than as a steady one
            z.append([None if v is None else 0.5 for v in vals])
        else:
            z.append([None if v is None
                      else ((v - lo) / span if better_high else (hi - v) / span)
                      for v in vals])
        text.append([f"{t} \u00b7 {label}: {format_metric(key, v)}"
                     for t, v in zip(trial_labels, vals)])

    n_trials, n_met = len(trial_labels), len(present)
    xs = [i * _SPACING for _ in range(n_met) for i in range(n_trials)]
    ys = [m * _SPACING for m in range(n_met) for _ in range(n_trials)]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(symbol="square", size=12,
                    color=[v if v is not None else np.nan for row in z for v in row],
                    colorscale=[[0.0, _BAD_COLOR], [0.5, "#FFD966"], [1.0, _GOOD_COLOR]],
                    cmin=0.0, cmax=1.0, showscale=False, line=dict(width=0)),
        text=[t for row in text for t in row],
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    ))
    # square markers on a fixed pitch, the form the channel quality grid above this panel
    # uses, so the two line up at the same left edge and carry the same size of cell. The
    # width is the figure's own: stretched to the page, five trials would come out as five bands
    _MARGIN = dict(l=70, r=20, t=20, b=100)
    fig.update_layout(
        xaxis=dict(tickvals=[i * _SPACING for i in range(n_trials)], ticktext=trial_labels,
                   tickangle=-45, tickfont=dict(size=8, color=AXIS_TEXT_COLOR),
                   showgrid=False, zeroline=False,
                   range=[-_SPACING, n_trials * _SPACING]),
        yaxis=dict(tickvals=[m * _SPACING for m in range(n_met)],
                   ticktext=[m[1] for m in present],
                   tickfont=dict(size=10, color=AXIS_TEXT_COLOR),
                   showgrid=False, zeroline=False,
                   range=[n_met * _SPACING, -_SPACING]),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=_MARGIN,
        width=_MARGIN["l"] + _MARGIN["r"] + (n_trials + 1) * _TRIAL_CELL_PX,
        height=_MARGIN["t"] + _MARGIN["b"] + (n_met + 1) * _TRIAL_CELL_PX,
    )
    return fig


def lollipop_scores_figure(
    ch_names: list[str],
    mean_scores: np.ndarray,
    colors: list[str],
    threshold: float,
    metric_label: str = "Score",
) -> go.Figure:
    n_ch = len(ch_names)
    stem_x, stem_y = [], []
    for i, val in enumerate(mean_scores):
        stem_x += [0.0, float(val), None]
        stem_y += [i * _SPACING, i * _SPACING, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y, mode="lines",
        line=dict(color="#aaaaaa", width=1.5),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=mean_scores.tolist(),
        y=[i * _SPACING for i in range(n_ch)],
        mode="markers",
        marker=dict(size=8, color=colors, line=dict(width=0.5, color="#333")),
        customdata=ch_names,
        showlegend=False,
        hovertemplate="%{customdata}: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line=dict(dash="dash", color="#888", width=1))
    fig.update_layout(
        xaxis_title=metric_label,
        yaxis=dict(autorange="reversed", showticklabels=False),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ---- The windowed strip, and the lollipop beside it ----
# Here rather than with the raw-intensity figures: only the subject pages draw it,
# and the lollipop it ends with is this module's.

def build_sci_psp_figure(
    sci_scores: dict[str, float],
    psp_per_channel: dict[str, float],
    bad_channels: set[str],
    sci_threshold: float = SCI_PASS,
    psp_threshold: float = PSP_PASS,
    sci_matrix: np.ndarray | None = None,
    sci_win_times: np.ndarray | None = None,
    psp_matrix: np.ndarray | None = None,
    psp_win_times: np.ndarray | None = None,
    cv_per_channel: dict[str, float] | None = None,
    cv_matrix: np.ndarray | None = None,
    cv_win_times: np.ndarray | None = None,
    cv_threshold: float = CV_PASS,
) -> go.Figure:
    """Channel quality over time: one heatmap row per metric, its channel mean beside it.

    Every row is measured on the uncorrected optical density and on one window grid, so a
    column in the CV row is the same stretch of recording as the column above it in SCI.

    CV carries SNR rather than getting a row of its own: the record stores SNR as 1/CV
    exactly, so a second row would be the same numbers reflected, and the hover prints both.
    CV is also the one row where low is good, which is why it takes the reversed scale.

    The lollipop beside each row is that row averaged along time, so the dot and the strip
    are one measurement, rather than the record's scalar of the same name -- which for SCI is
    the whole-run correlation and not the windowed one the strip draws.

    Without the windowed matrices this falls back to the lollipop-only pair.
    """
    ch_names = list(sci_scores.keys())
    sci_arr  = np.array([sci_scores.get(ch, 0.0) for ch in ch_names])
    has_psp  = bool(psp_per_channel)
    psp_arr  = np.array([psp_per_channel.get(ch, 0.0) for ch in ch_names]) if has_psp else None

    has_matrices = (
        sci_matrix is not None and sci_win_times is not None
        and psp_matrix is not None and psp_win_times is not None
        and has_psp
    )

    if not has_matrices:
        sci_colors = [
            "#e74c3c" if ch in bad_channels
            else ("#27ae60" if sci_scores.get(ch, 0.0) >= sci_threshold else "#f39c12")
            for ch in ch_names
        ]
        if not has_psp:
            return lollipop_scores_figure(ch_names, sci_arr, sci_colors, sci_threshold, "SCI")
        psp_colors = ["#27ae60" if psp_per_channel.get(ch, 0.0) >= psp_threshold
                      else "#e74c3c" for ch in ch_names]
        fig_sci = lollipop_scores_figure(ch_names, sci_arr, sci_colors, sci_threshold, "SCI")
        fig_psp = lollipop_scores_figure(ch_names, psp_arr,  psp_colors, psp_threshold, "PSP")
        n_ch = len(ch_names)
        combined = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                 horizontal_spacing=0.08, subplot_titles=["SCI", "PSP"])
        for t in fig_sci.data: combined.add_trace(t, row=1, col=1)
        for t in fig_psp.data: combined.add_trace(t, row=1, col=2)
        combined.update_yaxes(tickvals=[i * _SPACING for i in range(n_ch)],
                               ticktext=ch_names, autorange="reversed",
                               tickfont=dict(size=9), row=1, col=1)
        combined.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=2)
        combined.update_layout(height=min(max(300, n_ch * 14 + 100), 700),
                                margin=dict(l=110, r=30, t=40, b=40),
                                plot_bgcolor="white", paper_bgcolor="white")
        return combined

    # ---- one spec per heatmap row ----
    # (short name, heat title, mean title, matrix, window times, channel means,
    #  threshold, higher is better, hover line, colour range)
    #
    # Every row pins its range with the threshold at mid-scale, so the line is the same
    # colour on every recording and anything far past it saturates. An open `zmid` scale
    # widens symmetrically to the farthest window, which ran PSP's bar below zero and SCI's
    # above one. SCI tops out at 1, so its range ends there; PSP and CV start at 0.
    # the mean beside a row is that row's own mean, not the scalar of the same name: the
    # record's sci_mean is the whole-run correlation and its psp_mean and cv_mean are pinned
    # to 10 s. Here the dot is the row, averaged along time.
    def _row_mean(matrix) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.asarray(matrix, dtype=float), axis=1)

    rows = [
        ("SCI", "SCI (windowed)", "Row mean", sci_matrix, sci_win_times,
         _row_mean(sci_matrix), sci_threshold, True, "SCI=%{z:.3f}",
         (2 * sci_threshold - 1.0, 1.0)),
        ("PSP", "PSP (windowed)", "Row mean", psp_matrix, psp_win_times,
         _row_mean(psp_matrix), psp_threshold, True, "PSP=%{z:.3f}",
         (0.0, 2 * psp_threshold)),
    ]
    if cv_matrix is not None and cv_win_times is not None:
        rows.append(("CV", "CV (windowed)", "Row mean", cv_matrix, cv_win_times,
                     _row_mean(cv_matrix), cv_threshold, False,
                     "CV=%{z:.4f}<br>SNR=%{customdata:.1f}",
                     (0.0, 2 * cv_threshold)))

    n_ch   = len(ch_names)
    n_rows = len(rows)
    row_h  = min(max(220, n_ch * 14 + 80), 480)

    fig = make_subplots(
        rows=n_rows, cols=2,
        shared_yaxes=True,
        column_widths=[0.875, 0.125],
        row_heights=[1.0 / n_rows] * n_rows,
        vertical_spacing=0.06,
        horizontal_spacing=0.02,
        subplot_titles=[t for row in rows for t in (row[1], row[2])],
    )

    for i, (name, _heat_title, _mean_title, matrix, win_times, means,
            threshold, higher_better, hover, zrange) in enumerate(rows, start=1):
        centers = window_centers(win_times)
        z = np.asarray(matrix, dtype=float)
        # SNR is 1/CV by construction, so the row that has it hands it to the hover rather
        # than repeating the same matrix reflected as a fourth row
        customdata = None
        if not higher_better:
            with np.errstate(divide="ignore", invalid="ignore"):
                customdata = np.where(z > 0, 1.0 / z, np.nan)
        fig.add_trace(go.Heatmap(
            z=z, x=centers.tolist(), y=ch_names,
            customdata=customdata,
            colorscale="RdYlGn" if higher_better else "RdYlGn_r",
            zmin=zrange[0], zmax=zrange[1],
            colorbar=dict(title=name, thickness=10,
                          len=0.88 / n_rows, y=1.0 - (i - 0.5) / n_rows, x=1.01),
            hovertemplate="Ch: %{y}<br>t=%{x:.1f}s<br>" + hover + "<extra></extra>",
            name=name,
        ), row=i, col=1)

        # the screening verdict only colours the row it is read off; the others colour
        # against their own cutoff, so a channel rejected on SCI is not painted red in CV
        colors = []
        for ch, v in zip(ch_names, means):
            if name == "SCI" and ch in bad_channels:
                colors.append("#e74c3c")
            elif not np.isfinite(v):
                colors.append("#D3D3D3")
            elif (v >= threshold) if higher_better else (v <= threshold):
                colors.append("#27ae60")
            else:
                colors.append("#f39c12" if higher_better else "#e74c3c")

        lx, ly = [], []
        for ch, v in zip(ch_names, means):
            lx += [0.0, float(v) if np.isfinite(v) else 0.0, None]
            ly += [ch, ch, None]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                                 line=dict(color="#aaa", width=1.2),
                                 showlegend=False, hoverinfo="skip"), row=i, col=2)
        fig.add_trace(go.Scatter(x=np.asarray(means, dtype=float).tolist(), y=ch_names,
                                 mode="markers",
                                 marker=dict(size=7, color=colors,
                                             line=dict(width=0.5, color="#333")),
                                 showlegend=False,
                                 hovertemplate="%{y}: %{x:.3f}<extra></extra>"), row=i, col=2)
        fig.add_vline(x=threshold, line_dash="dash", line_color="#888",
                      line_width=1, row=i, col=2)

        fig.update_yaxes(autorange="reversed", tickfont=dict(size=9), row=i, col=1)
        fig.update_yaxes(autorange="reversed", showticklabels=False, row=i, col=2)

    fig.update_xaxes(title_text="Time (s)", gridcolor="#eee", row=n_rows, col=1)
    fig.update_xaxes(title_text="Score", row=n_rows, col=2)
    fig.update_layout(
        height=row_h * n_rows + 80,
        margin=dict(l=120, r=110, t=50, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig
