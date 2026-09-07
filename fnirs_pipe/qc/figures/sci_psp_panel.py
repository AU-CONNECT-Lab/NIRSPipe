import numpy as np
import plotly.graph_objects as go

from fnirs_pipe.qc.metrics import CV_PASS, PSP_PASS, SCI_PASS, SNR_PASS
from fnirs_pipe.utils.logging import get_logger

from ._utils import AXIS_TEXT_COLOR

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


def channel_quality_heatmap(
    ch_names: list[str],
    is_bad: list[bool],
    sci_per_ch: dict[str, float],
    cv_per_ch: dict[str, float],
    snr_per_ch: dict[str, float],
    psp_per_ch: dict[str, float],
    sci_thresh: float = SCI_PASS,
    cv_thresh: float = CV_PASS,
    snr_thresh: float = SNR_PASS,
    psp_thresh: float = PSP_PASS,
    split_at: int | None = None,
) -> go.Figure:
    """Square-marker grid: channels on x-axis, metrics on y-axis. Green=pass, red=fail, gray=missing.

    ``split_at`` is the index the short channels start at, given when the caller has
    already ordered ``ch_names`` long block first; it draws the divider and names the two
    blocks. The two are pruned by the same threshold but answer different questions, so a
    reader needs to know which side of the line a column is on.

    Status is the screening verdict. The metric rows below it are drawn against the
    cutoffs in :mod:`fnirs_pipe.qc.metrics._helpers`; CV and SNR prune nothing.
    """
    _MISSING = "#D3D3D3"
    metrics = ["Status", "SCI", "CV", "PSP", "SNR"]
    n_ch  = len(ch_names)
    n_met = len(metrics)

    specs = [
        (None,       None,   None),
        (sci_per_ch, ".3f",  lambda v: v >= sci_thresh),
        (cv_per_ch,  ".3f",  lambda v: v <= cv_thresh),
        (psp_per_ch, ".3f",  lambda v: v >= psp_thresh),
        (snr_per_ch, ".1f",  lambda v: v >= snr_thresh),
    ]

    xs, ys, colors, hover = [], [], [], []
    for m_idx, (metric, (lookup, fmt, chk)) in enumerate(zip(metrics, specs)):
        for i, ch in enumerate(ch_names):
            xs.append(i * _SPACING)
            ys.append(m_idx * _SPACING)
            if metric == "Status":
                bad = is_bad[i]
                colors.append(_BAD_COLOR if bad else _GOOD_COLOR)
                hover.append(f"{ch} — Status: {'BAD' if bad else 'OK'}")
            else:
                v = lookup.get(ch)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    colors.append(_MISSING)
                    hover.append(f"{ch} — {metric}: —")
                else:
                    colors.append(_GOOD_COLOR if chk(v) else _BAD_COLOR)
                    hover.append(f"{ch} — {metric}: {v:{fmt}}")

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
        xaxis=dict(
            tickvals=[i * _SPACING for i in range(n_ch)],
            ticktext=ch_names,
            tickangle=-45, tickfont=dict(size=9, color=AXIS_TEXT_COLOR),
            showgrid=False, zeroline=False,
            range=[-_SPACING, n_ch * _SPACING],
        ),
        yaxis=dict(
            tickvals=[m * _SPACING for m in range(n_met)],
            ticktext=metrics,
            autorange="reversed",
            tickfont=dict(size=10, color=AXIS_TEXT_COLOR),
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=20, t=20, b=100),
        height=260,
    )
    if split_at is not None and 0 < split_at < n_ch:
        # the line sits in the gap between the last long column and the first short one
        x_div = (split_at - 0.5) * _SPACING
        fig.add_vline(x=x_div, line_dash="dot", line_color="#888", line_width=1)
        for centre, text in (((split_at - 1) / 2, "long"),
                             ((split_at + n_ch - 1) / 2, "short")):
            fig.add_annotation(x=centre * _SPACING, y=1.10, xref="x", yref="paper",
                               text=text, showarrow=False, font=dict(size=10, color="#666"))
        # the extra top margin is what the raised labels sit in
        fig.update_layout(margin=dict(l=70, r=20, t=52, b=100))
    return fig


# key and row label only. The hover format and which end is the better one come from the
# metric registry, which is also where the reports read them: a row that said SNR was better
# low here and better high there would be a contradiction inside one report. Labels stay
# local because a heatmap row is a few characters wide and the registry's are sentences.
#
# Every key here must carry a direction in the registry. Colour is relative within a row --
# the worse end of what this recording actually did -- so a descriptive metric with no better
# end has no worse end either and cannot be drawn; _trial_metric_specs drops it and says so.
_TRIAL_METRICS = [
    ("sci_mean",               "SCI"),
    ("psp_mean",               "PSP"),
    ("cv_mean",                "CV"),
    ("snr_mean",               "SNR"),
    ("gvtd_mean",              "GVTD"),
    ("gvtd_filt_mean",         "GVTD band"),
    ("channel_retention_rate", "Retention"),
]


def _trial_metric_specs() -> list[tuple[str, str, bool]]:
    """_TRIAL_METRICS resolved against the registry, as (key, label, higher_is_better)."""
    from fnirs_pipe.qc.boilerplate.vocabulary import higher_is_better

    specs = []
    for key, label in _TRIAL_METRICS:
        higher = higher_is_better(key)
        if higher is None:
            logger.warning("%s has no direction in the metric registry; "
                           "leaving it out of the per-trial heatmap", key)
            continue
        specs.append((key, label, higher))
    return specs


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
    SCI row with two red cells, which is the whole point of looking per trial rather than at
    the recording mean.

    Returns None when no metric survives on any trial (nothing to draw).
    """
    from fnirs_pipe.qc.boilerplate.vocabulary import format_metric

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
        text.append([f"{label}: {format_metric(key, v)}" for v in vals])

    fig = go.Figure(go.Heatmap(
        z=z, text=text,
        x=trial_labels,
        y=[m[1] for m in present],
        colorscale=[[0.0, _BAD_COLOR], [0.5, "#FFD966"], [1.0, _GOOD_COLOR]],
        showscale=False,
        hovertemplate="%{x}<br>%{text}<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=8), showgrid=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=20, t=20, b=90),
        height=60 + 26 * len(present),
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
