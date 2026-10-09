"""Panels for the cohort test `fnirs-hyper-groupnull` writes: real dyads against their null.

Both read the tables the command already wrote and compute nothing that is not on them
except a mean and its interval. ``byoccasion`` gives each occasion's real value and the mean
of its own draws; ``cohort`` gives each level's lift and its two tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

_REAL, _NULL, _INK = "#2c3e50", "#95a5a6", "#c0392b"
ALPHA = 0.05


def whole_level(frame: pd.DataFrame) -> pd.DataFrame:
    """The whole-brain rows, over every pairing where the null was drawn crossed."""
    whole = frame[frame["granularity"] == "whole"]
    pairings = "all" if (whole["pairings"] == "all").any() else "homologous"
    return whole[whole["pairings"] == pairings]


def p_column(frame: pd.DataFrame) -> str:
    """The corrected p where the run asked for one, else the raw p."""
    corrected = [c for c in frame.columns if c.startswith("p_")]
    return corrected[0] if corrected else "p"


def _mean_ci(values: np.ndarray) -> "tuple[float, float]":
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float(np.mean(values)) if values.size else np.nan, np.nan
    half = stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return float(values.mean()), float(half)


def build_occasion_panels(byoccasion: pd.DataFrame, value: str, axis_title: str,
                          captions: "dict[str, str] | None" = None) -> "go.Figure | None":
    """Per condition, each occasion's real value over its null mean, then the two paired.

    ::

      byoccasion whole-brain rows, 3 conditions x 12 occasions
        -> 3 rows: occasions along x on the left, null-to-real segments on the right

    A real value at or below its null mean is a hollow point. ``captions`` adds a line per
    condition, which is where the cohort test's result goes.
    """
    frame = whole_level(byoccasion)
    if frame.empty:
        return None
    conditions = list(dict.fromkeys(frame["condition"]))
    captions = captions or {}
    titles = []
    for cond in conditions:
        titles += [f"{cond}  {captions.get(cond, '')}".strip(), ""]
    fig = make_subplots(rows=len(conditions), cols=2, column_widths=[0.76, 0.24],
                        horizontal_spacing=0.03, vertical_spacing=0.5 / max(len(conditions), 1),
                        shared_yaxes=True, subplot_titles=titles)
    for r, cond in enumerate(conditions, start=1):
        part = frame[frame["condition"] == cond].sort_values("occasion")
        occ = part["occasion"].astype(str).tolist()
        real = part[value].to_numpy(dtype=float)
        null = part["null_mean"].to_numpy(dtype=float)
        above = real > null
        legend = r == 1
        fig.add_trace(go.Scatter(x=occ, y=null, mode="lines+markers", name="null mean",
                                 legendgroup="null", showlegend=legend,
                                 line=dict(color=_NULL, width=1.2), marker=dict(size=5),
                                 hovertemplate="%{x}<br>null mean %{y:.4f}<extra></extra>"),
                      row=r, col=1)
        fig.add_trace(go.Scatter(
            x=occ, y=real, mode="lines+markers", name="real dyad", legendgroup="real",
            showlegend=legend, line=dict(color=_REAL, width=0.8),
            marker=dict(size=7, color=[_REAL if a else "white" for a in above],
                        line=dict(color=_REAL, width=1.2)),
            hovertemplate="%{x}<br>real %{y:.4f}<extra></extra>"), row=r, col=1)
        for n, v in zip(null, real):
            fig.add_trace(go.Scatter(x=[0, 1], y=[n, v], mode="lines", showlegend=False,
                                     line=dict(color=_REAL, width=0.6), opacity=0.35,
                                     hoverinfo="skip"), row=r, col=2)
        for x, values, colour in ((-0.3, null, _NULL), (1.3, real, _REAL)):
            mean, half = _mean_ci(values)
            fig.add_trace(go.Scatter(
                x=[x], y=[mean], mode="markers", showlegend=False,
                marker=dict(size=8, color=colour),
                error_y=dict(type="data", array=[half if np.isfinite(half) else 0.0],
                             color=colour, thickness=1.4, width=0),
                hovertemplate=f"mean %{{y:.4f}} ± {half:.4f} (95% CI)<extra></extra>"),
                row=r, col=2)
        fig.update_xaxes(tickvals=[0, 1], ticktext=["null", "real"], range=[-0.6, 1.6],
                         row=r, col=2)
        fig.update_xaxes(type="category", tickfont=dict(size=8), row=r, col=1)
        fig.update_yaxes(title_text=axis_title, title_font=dict(size=10), row=r, col=1)
    fig.update_annotations(font=dict(size=11, color="#34495e"), xanchor="left", x=0.0)
    fig.update_layout(height=230 * len(conditions) + 60, plot_bgcolor="white",
                      margin=dict(l=70, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1,
                                  xanchor="right", font=dict(size=10)))
    return fig


def _significant(rows: pd.DataFrame, p_col: str) -> np.ndarray:
    return (rows[p_col] < ALPHA).to_numpy() if p_col in rows.columns else np.zeros(len(rows), bool)


def build_region_lift(cohort: pd.DataFrame, value_label: str, test: str = "paired") -> "go.Figure | None":
    """Each region's lift over its null, a dot where ``test`` clears ``ALPHA``.

    ::

      region pairs L>L, L>R, R>L, R>R in 2 conditions  ->  two 2 x 2 matrices
      regions L, R homologous only in 2 conditions      ->  one regions x conditions grid

    Lift is signed, so the scale is centred on zero and spans the largest lift either way.
    The corrected p is read where the run asked for one.
    """
    roi = cohort[(cohort["granularity"] == "roi") & (cohort["test"] == test)]
    if roi.empty:
        return None
    p_col = p_column(roi)
    crossed = roi[roi["pairings"] == "all"]
    lifts = np.abs(roi["lift"].to_numpy(dtype=float))
    lifts = lifts[np.isfinite(lifts)]
    reach = float(lifts.max()) if lifts.size and lifts.max() > 0 else 1.0
    scale = dict(colorscale="RdBu_r", zmin=-reach, zmax=reach, zmid=0.0)
    colorbar = dict(title=dict(text=f"lift ({value_label})", side="right"), thickness=12)
    if not crossed.empty:
        conditions = list(dict.fromkeys(crossed["condition"]))
        pairs = crossed["level"].str.split(">", expand=True)
        regions = list(dict.fromkeys(pairs[0].tolist() + pairs[1].tolist()))
        fig = make_subplots(rows=1, cols=len(conditions), subplot_titles=conditions,
                            horizontal_spacing=0.04)
        for c, cond in enumerate(conditions, start=1):
            part = crossed[crossed["condition"] == cond]
            a, b = part["level"].str.split(">", expand=True).T.to_numpy()
            z = np.full((len(regions), len(regions)), np.nan)
            z[[regions.index(x) for x in a], [regions.index(y) for y in b]] = part["lift"]
            fig.add_trace(go.Heatmap(z=z, x=regions, y=regions, showscale=c == 1,
                                     colorbar=colorbar, **scale,
                                     hovertemplate="%{y} > %{x}<br>lift %{z:.4f}<extra></extra>"),
                          row=1, col=c)
            hit = _significant(part, p_col)
            fig.add_trace(go.Scatter(x=list(b[hit]), y=list(a[hit]), mode="markers",
                                     showlegend=False, marker=dict(size=7, color=_INK),
                                     hoverinfo="skip"), row=1, col=c)
            fig.update_yaxes(autorange="reversed", showticklabels=c == 1, row=1, col=c)
        height = 120 + 46 * len(regions)
    else:
        part = roi[roi["pairings"] == "homologous"]
        conditions = list(dict.fromkeys(part["condition"]))
        regions = list(dict.fromkeys(part["level"]))
        z = np.full((len(regions), len(conditions)), np.nan)
        for _, row in part.iterrows():
            z[regions.index(row["level"]), conditions.index(row["condition"])] = row["lift"]
        fig = go.Figure(go.Heatmap(z=z, x=conditions, y=regions, colorbar=colorbar, **scale,
                                   hovertemplate="%{y}, %{x}<br>lift %{z:.4f}<extra></extra>"))
        hit = _significant(part, p_col)
        fig.add_trace(go.Scatter(x=list(part["condition"][hit]), y=list(part["level"][hit]),
                                 mode="markers", showlegend=False,
                                 marker=dict(size=7, color=_INK), hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        height = 120 + 40 * len(regions)
    fig.update_layout(height=max(height, 260), plot_bgcolor="white",
                      margin=dict(l=90, r=20, t=50, b=40))
    return fig
