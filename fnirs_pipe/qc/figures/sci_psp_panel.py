import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_GOOD_COLOR = "#C5E0B3"
_BAD_COLOR  = "#F8786E"
_MAX_TS_PTS = 3000
_SPACING    = 3

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
