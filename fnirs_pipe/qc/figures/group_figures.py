"""Plotly figure builders for group-level QC reports (individual and hyper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _tukey_fences(values: np.ndarray, k: float = 1.5) -> tuple[float, float]:
    """Return (lower, upper) Tukey fences for outlier detection."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (-np.inf, np.inf)
    q1 = np.quantile(finite, 0.25)
    q3 = np.quantile(finite, 0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)


def detect_outliers(
    df: pd.DataFrame, metric_cols: list[str], k: float = 1.5,
) -> dict[str, list[str]]:
    """Return {bids_name: [metric_col, ...]} for rows beyond Tukey fences."""
    out: dict[str, list[str]] = {}
    for col in metric_cols:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        lo, hi = _tukey_fences(vals, k=k)
        mask = (vals < lo) | (vals > hi)
        for i, is_out in enumerate(mask):
            if is_out and np.isfinite(vals[i]):
                bids_name = str(df.iloc[i, 0])
                out.setdefault(bids_name, []).append(col)
    return out


def build_heatmap(df: pd.DataFrame, metric_cols: list[str]) -> go.Figure | None:
    """Subject × metric z-score heatmap. Colour = number of SDs from median."""
    rows = df.iloc[:, 0].astype(str).tolist()
    if not rows or not metric_cols:
        return None

    z = np.full((len(rows), len(metric_cols)), np.nan)
    for j, col in enumerate(metric_cols):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        finite = vals[np.isfinite(vals)]
        if finite.size < 2:
            continue
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        scale = 1.4826 * mad if mad > 0 else np.std(finite, ddof=0)
        if scale == 0:
            continue
        z[:, j] = (vals - med) / scale

    fig = go.Figure(go.Heatmap(
        z=z, x=metric_cols, y=rows,
        colorscale="RdBu", zmid=0, zmin=-3, zmax=3,
        hovertemplate="<b>%{y}</b><br>%{x}: z=%{z:.2f}<extra></extra>",
        colorbar=dict(title="z<br>(robust)", thickness=12),
    ))
    fig.update_layout(
        height=max(360, 22 * len(rows) + 160),
        margin=dict(l=240, r=20, t=30, b=120),
        plot_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-45, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def build_time_subject_heatmap(
    rows: list[dict], metric_field: str, times_field: str, title: str,
) -> go.Figure | None:
    """Heatmap of one windowed metric across subjects and time.

    `rows` is a list of dicts (one per subject) each containing the metric
    array and matching `*_times_s` array. Rows missing the metric are skipped.
    Different subjects may have different time bases; they are kept on the
    union time axis (cells beyond a subject's recording = NaN).
    """
    series = []
    for r in rows:
        vals  = r.get(metric_field)
        times = r.get(times_field)
        if vals is None or times is None or not len(vals):
            continue
        series.append((r["bids_name"], np.asarray(times, dtype=float), np.asarray(vals, dtype=float)))
    if not series:
        return None

    all_times = np.unique(np.concatenate([t for _, t, _ in series]))
    z = np.full((len(series), len(all_times)), np.nan)
    for i, (_, t, v) in enumerate(series):
        idx = np.searchsorted(all_times, t)
        z[i, idx] = v

    subjects = [s for s, _, _ in series]
    fig = go.Figure(go.Heatmap(
        z=z, x=all_times.tolist(), y=subjects,
        colorscale="RdYlGn", zauto=True,
        hovertemplate="<b>%{y}</b><br>t=%{x:.0f}s<br>" + metric_field + "=%{z:.3f}<extra></extra>",
        colorbar=dict(title=metric_field, thickness=12),
    ))
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=13)),
        height=max(320, 22 * len(subjects) + 160),
        margin=dict(l=240, r=20, t=50, b=60),
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="Time (s)", gridcolor="#eeeeee", automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def build_boxplot_per_metric(
    df: pd.DataFrame, metric_cols: list[str],
) -> go.Figure | None:
    """One subplot per metric: boxplot + all-subject scatter; outliers labelled."""
    if not metric_cols or df.empty:
        return None

    from plotly.subplots import make_subplots

    n = len(metric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=metric_cols,
        vertical_spacing=0.08, horizontal_spacing=0.08,
    )

    rows = df.iloc[:, 0].astype(str).tolist()
    for idx, col in enumerate(metric_cols):
        r = idx // ncols + 1
        c = idx % ncols + 1
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        finite_mask = np.isfinite(vals)
        finite_vals = vals[finite_mask]
        if finite_vals.size == 0:
            continue

        lo, hi = _tukey_fences(finite_vals, k=1.5)
        is_out = (vals < lo) | (vals > hi)
        colors = ["#c0392b" if (o and f) else "#3498db"
                  for o, f in zip(is_out, finite_mask)]

        fig.add_trace(go.Box(
            y=finite_vals, name="", boxpoints=False,
            line=dict(color="#7f8c8d"), fillcolor="rgba(189,195,199,0.3)",
            hoverinfo="skip", showlegend=False,
        ), row=r, col=c)
        rng = np.random.default_rng(seed=idx)
        fig.add_trace(go.Scatter(
            x=rng.uniform(-0.18, 0.18, size=len(rows)),
            y=vals, mode="markers",
            marker=dict(color=colors, size=7,
                        line=dict(width=0.5, color="#2c3e50")),
            text=rows, hovertemplate="<b>%{text}</b><br>%{y}<extra></extra>",
            showlegend=False,
        ), row=r, col=c)

    fig.update_layout(
        height=max(280 * nrows, 320),
        margin=dict(l=40, r=20, t=40, b=20),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig
