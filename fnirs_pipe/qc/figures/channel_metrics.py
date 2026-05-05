"""Lollipop chart for scalar per-channel QC metrics (tSNR, CV, HbO-HbR corr, etc.)."""

import numpy as np
import plotly.graph_objects as go

_GOOD_COLOR    = "#3498db"
_BAD_COLOR     = "#e74c3c"
_NEUTRAL_COLOR = "#7f8c8d"


def lollipop_chart(
    ch_names: list[str],
    values: np.ndarray | list[float],
    title: str,
    xaxis_title: str = "Value",
    threshold: float | None = None,
    good_above: bool = True,
    sort_by_value: bool = True,
) -> go.Figure:
    """Horizontal lollipop chart for a scalar per-channel metric.

    Args:
        ch_names:      Channel names.
        values:        Scalar value per channel.
        title:         Figure title.
        xaxis_title:   X-axis label.
        threshold:     Optional pass/fail threshold; adds a vertical dashed line
                       and colours dots accordingly.
        good_above:    If True, values >= threshold are good (blue); else reversed.
        sort_by_value: Sort channels by value (ascending) for readability.
    """
    values = np.asarray(values, dtype=float)
    n = len(ch_names)

    if sort_by_value:
        order = np.argsort(values)
        ch_names = [ch_names[i] for i in order]
        values = values[order]

    if threshold is not None:
        if good_above:
            colors = [_GOOD_COLOR if v >= threshold else _BAD_COLOR for v in values]
        else:
            colors = [_GOOD_COLOR if v < threshold else _BAD_COLOR for v in values]
    else:
        colors = [_NEUTRAL_COLOR] * n

    stem_x, stem_y = [], []
    for i, val in enumerate(values):
        stem_x += [0.0, float(val), None]
        stem_y += [i, i, None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stem_x, y=stem_y,
        mode="lines",
        line=dict(color="#cccccc", width=1.5),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=values.tolist(), y=list(range(n)),
        mode="markers",
        marker=dict(size=9, color=colors, line=dict(width=0.5, color="#333")),
        text=ch_names,
        hovertemplate="<b>%{text}</b><br>%{x:.4f}<extra></extra>",
        showlegend=False,
    ))

    if threshold is not None:
        fig.add_vline(
            x=threshold,
            line=dict(dash="dash", color="#888", width=1),
            annotation_text=f"threshold={threshold}",
            annotation_position="top right",
            annotation_font_size=10,
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis=dict(
            tickvals=list(range(n)),
            ticktext=ch_names,
            autorange="reversed",
            tickfont=dict(size=9),
        ),
        height=max(350, n * 18 + 100),
        margin=dict(l=120, r=30, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig
