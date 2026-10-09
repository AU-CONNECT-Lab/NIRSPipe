"""Panels for the cohort-level hyperscanning report: one mark per dyad, not per channel.

Kept apart from ``group_figures``, whose panels are the individual cohort's robust-z strip
and boxes over a metric table. Nothing here is a z-score: a dyad is read against the share
of its own recording it could use. The two modules share the report shell and the row order convention and nothing else.

Every panel takes the same ``rows``, one dict per dyad-task, and the same ``order``, so a
dyad sits on the same line down the page and is read across the panels rather than looked up
in each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.figures.hyper.hyper_figures import (
    _BAD_COLOR, _GOOD_COLOR, _MIX_COLOR,
)
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.group_hyper_figures")

# How far round a full share sweeps on a dial, and where it starts; the gap holds the labels.
_DIAL_SWEEP, _DIAL_START = 320.0, 20.0
_DIAL_ACCENTS = ("#4f9aa8", "#2f6f8f", "#1f5673", "#7a9e7e", "#b07d62", "#8c6f9e")

# dials drawn, one per dyad from the worst down
N_DIALS = 6


def cohort_order(rows: list[dict]) -> list[str]:
    """Dyad labels worst first, by shared usable time. The row order of every panel."""
    return [r["label"] for r in
            sorted(rows, key=lambda r: r["usable"].get("usable_window_frac", 1.0))]


def _by_label(rows: list[dict]) -> dict:
    return {r["label"]: r for r in rows}


# ---- Shared usable time ----

def build_usable_bars(rows: list[dict], order: list[str]) -> "go.Figure | None":
    """One bar per dyad, split into coupled in both / one member / neither.

    The overview panel, carrying both channels and time, a cell being one channel pair in one
    window. The three colours are the ones the dyad's own usable-time carpet uses.

    None when no dyad carries the split.
    """
    by_label = _by_label(rows)
    parts = (("usable_window_frac", "coupled in both", _GOOD_COLOR),
             ("one_member_frac", "one member only", _MIX_COLOR),
             ("neither_frac", "neither", _BAD_COLOR))
    if not any(p[0] in by_label[l]["usable"] for l in order for p in parts[1:]):
        return None

    fig = go.Figure()
    for key, name, colour in parts:
        fig.add_trace(go.Bar(
            # a share the record lacks draws no bar rather than a zero one
            x=[None if (v := by_label[label]["usable"].get(key)) is None else v * 100
               for label in order],
            y=order, orientation="h", name=name,
            marker=dict(color=colour, line=dict(width=0.5, color="#fff")),
            hovertemplate=f"<b>%{{y}}</b><br>{name}: %{{x:.1f}}%<extra></extra>"))

    shares = [by_label[label]["usable"].get("usable_window_frac") for label in order]
    shares = [s for s in shares if s is not None]
    if shares:
        median = float(np.median(shares)) * 100
        fig.add_vline(x=median, line_color="#6b7784", line_width=1, line_dash="dot",
                      annotation_text=f"cohort median {median:.0f}%",
                      annotation_font=dict(size=9.5), annotation_position="top")
    fig.update_layout(barmode="stack", height=110 + 24 * len(order), plot_bgcolor="white",
                      margin=dict(l=150, r=24, t=52, b=46), bargap=0.3,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1, font=dict(size=10)))
    fig.update_xaxes(title_text="Share of pair-windows", title_font=dict(size=10),
                     ticksuffix="%", range=[0, 100], dtick=25, gridcolor="#f5f5f5",
                     tickfont=dict(size=9), zeroline=False)
    fig.update_yaxes(tickfont=dict(size=9), showgrid=False, autorange="reversed")
    return fig


# ---- Where the time went ----

def _frame(rows: list[dict], order: list[str], key: str) -> pd.DataFrame:
    """``dyad x column`` of usable shares, off each row's per-pair or per-condition dict."""
    by_label = _by_label(rows)
    return pd.DataFrame({label: by_label[label].get(key) or {} for label in order}).T


def _dot_trace(frame: pd.DataFrame, showscale: bool,
               colorbar_x: "float | None" = None) -> go.Scatter:
    """One marker per cell, coloured by the share.

    The ramp runs pale for a lost pair and dark for a kept one, so the rare low cell is the
    bright dot in a dark field.
    """
    x, y, values = [], [], []
    for label in frame.index:
        for column in frame.columns:
            value = frame.loc[label, column]
            if pd.isna(value):
                continue
            x.append(str(column))
            y.append(str(label))
            values.append(float(value) * 100)
    return go.Scatter(
        x=x, y=y, mode="markers", showlegend=False,
        marker=dict(size=13, color=values, colorscale="YlGnBu", cmin=0, cmax=100,
                    showscale=showscale, line=dict(width=0),
                    colorbar=dict(ticksuffix="%", thickness=10, len=0.8, outlinewidth=0,
                                  tickfont=dict(size=9), tickvals=[0, 50, 100],
                                  **({} if colorbar_x is None else {"x": colorbar_x}))),
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{marker.color:.0f}% usable<extra></extra>")


def build_pair_field(rows: list[dict], order: list[str]) -> "go.Figure | None":
    """dyad x channel pair, one dot per cell, worst column first.

    The finding no single dyad page can carry: a column pale down the whole cohort is the cap
    or the optode rather than the dyad. Columns are ordered by the cohort's own mean.

    None when the dyads share no pair, rather than a field whose columns are not the same
    pair down the page. A cohort on two montages keeps the pairs the two have in common.
    """
    frame = _frame(rows, order, "by_pair")
    frame = frame.dropna(axis=1, how="any")
    if frame.empty or not len(frame.columns):
        logger.info("the dyads share no channel pair; the pair field is not drawn")
        return None
    frame = frame[frame.mean().sort_values().index]

    fig = go.Figure(_dot_trace(frame, True))
    fig.update_layout(height=96 + 24 * len(order), plot_bgcolor="white",
                      margin=dict(l=150, r=24, t=34, b=58))
    fig.update_xaxes(tickfont=dict(size=9), showgrid=False, ticks="")
    fig.update_yaxes(tickfont=dict(size=9), showgrid=True, gridcolor="#f4f6f8", ticks="",
                     autorange="reversed")
    return fig


def _add_dial(fig, row: int, col: int, label: str, rings: list[str], values: list[float],
              accent: str) -> None:
    """One dyad's dial: a ring per entry of ``rings``, the arc its share of the circle.

    A value of None is a block the dyad lacks: its ring is drawn empty and named so.
    """
    for j, (name, value) in enumerate(zip(rings, values)):
        r_ring = len(rings) - j              # outermost ring is the first condition
        fig.add_trace(go.Barpolar(
            r=[0.55], base=[r_ring - 0.275], theta=[_DIAL_START + _DIAL_SWEEP / 2],
            width=[_DIAL_SWEEP], marker=dict(color="#eef1f4", line=dict(width=0)),
            showlegend=False, hoverinfo="skip"), row=row, col=col)
        if value is None:
            fig.add_trace(go.Scatterpolar(
                r=[r_ring], theta=[0], mode="text", text=[f"{name} (no data)"],
                textfont=dict(size=8, color="#5d6b78"), showlegend=False,
                hoverinfo="skip"), row=row, col=col)
            continue
        sweep = max(float(value) * _DIAL_SWEEP, 0.8)
        fig.add_trace(go.Barpolar(
            r=[0.55], base=[r_ring - 0.275], theta=[_DIAL_START + sweep / 2],
            width=[sweep], marker=dict(color=accent, line=dict(width=0)),
            customdata=[[label, name, float(value) * 100]], showlegend=False,
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}"
                           "<br>%{customdata[2]:.0f}% usable<extra></extra>")),
            row=row, col=col)
        fig.add_trace(go.Scatterpolar(
            r=[r_ring], theta=[0], mode="text", text=[name],
            textfont=dict(size=8, color="#5d6b78"), showlegend=False,
            hoverinfo="skip"), row=row, col=col)


def build_condition_dials(rows: list[dict], order: list[str]) -> "go.Figure | None":
    """The condition field on the left, the dyads it turns up as dials on the right.

    The field is the same share cut by block, a share rather than absolute seconds.

    None when no dyad carries per-condition shares.
    """
    frame = _frame(rows, order, "by_cond").dropna(axis=1, how="all")
    if frame.empty or not len(frame.columns):
        return None

    by_label = _by_label(rows)
    worst = order[:N_DIALS]
    rings = [*frame.columns, "overall"]
    n_cols = min(3, max(1, len(worst)))
    n_rows = max(1, -(-len(worst) // n_cols))

    fig = make_subplots(
        rows=max(n_rows, 1), cols=n_cols + 1,
        column_widths=[0.31, *[0.69 / n_cols] * n_cols],
        vertical_spacing=0.12, horizontal_spacing=0.04,
        specs=[[{"type": "xy", "rowspan": n_rows}, *[{"type": "polar"}] * n_cols],
               *[[None, *[{"type": "polar"}] * n_cols] for _ in range(n_rows - 1)]],
        subplot_titles=["per condition", *worst])

    fig.add_trace(_dot_trace(frame, True, colorbar_x=0.30), row=1, col=1)
    for k, label in enumerate(worst):
        values = [None if pd.isna(frame.loc[label, c]) else float(frame.loc[label, c])
                  for c in frame.columns]
        overall = by_label[label]["usable"].get("usable_window_frac")
        values.append(None if overall is None else float(overall))
        _add_dial(fig, k // n_cols + 1, k % n_cols + 2, label, rings, values,
                  _DIAL_ACCENTS[k % len(_DIAL_ACCENTS)])

    fig.update_layout(height=max(340, 110 * n_rows + 24 * len(order) // 2 + 200),
                      margin=dict(l=150, r=30, t=54, b=54),
                      plot_bgcolor="white", paper_bgcolor="white")
    fig.update_annotations(font=dict(size=10.5, color="#34495e"))
    fig.update_xaxes(tickfont=dict(size=9), showgrid=False, ticks="")
    fig.update_yaxes(tickfont=dict(size=9), showgrid=True, gridcolor="#f4f6f8", ticks="",
                     autorange="reversed")
    fig.update_polars(
        hole=0.12, bgcolor="white",
        radialaxis=dict(range=[0, len(rings) + 0.6], showticklabels=False, ticks="",
                        showgrid=False, showline=False),
        angularaxis=dict(direction="clockwise", rotation=90, showticklabels=False,
                         ticks="", showgrid=False, showline=False))
    return fig
