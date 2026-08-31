import numpy as np
import plotly.graph_objects as go

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
    sci_thresh: float = 0.75,
    cv_thresh: float = 0.5,
    snr_thresh: float = 20.0,
    psp_thresh: float = 0.1,
) -> go.Figure:
    """Square-marker grid: channels on x-axis, metrics on y-axis. Green=pass, red=fail, gray=missing."""
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
            tickangle=-45, tickfont=dict(size=9),
            showgrid=False, zeroline=False,
            range=[-_SPACING, n_ch * _SPACING],
        ),
        yaxis=dict(
            tickvals=[m * _SPACING for m in range(n_met)],
            ticktext=metrics,
            autorange="reversed",
            tickfont=dict(size=10),
            showgrid=False, zeroline=False,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=20, t=20, b=100),
        height=260,
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
