"""Correlations among a run's nuisance series and with its global signals, the fNIRS counterpart of fMRIPrep's confounds panel."""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from nirspipe.pipeline.chromophores import chroma_of
from nirspipe.pipeline.glm_censored import kept_frames
from nirspipe.qc.figures.common._utils import CHROMA_STYLE
from nirspipe.qc.figures.common.matrix_map import CORRELATION_SCALE
from nirspipe.qc.metrics import long_short_channels
from nirspipe.qc.metrics.gvtd import GVTD_MOTION_BAND, gvtd_channel_picks, gvtd_timetrace

# the design columns that are nuisance rather than task or intercept
NUISANCE_PREFIXES = ("short_ch_", "aux_", "drift_")
# as fMRIPrep's panel, the global signals included
MAX_SERIES = 20
_IMU_NAMES = {"gyro": "gyro speed", "accel": "accel jerk"}
# the bars' share of white mixed into the chromophore colours, so they sit beside the heatmap
_BAR_TINT = 0.3
GLOBAL_NAMES = {c: f"global {CHROMA_STYLE[c][0]}" for c in ("hbo", "hbr")}


def nuisance_series(haemo: mne.io.BaseRaw, design: "pd.DataFrame | None" = None,
                    od: "mne.io.BaseRaw | None" = None, imu: "dict | None" = None,
                    sep_bands=None) -> pd.DataFrame:
    """The global signals and every nuisance series on ``haemo``'s frames, the ones left out dropped.

    ``haemo`` is the series the confound regression was fitted on; ``design`` its design
    table, whose ``censored`` column names the frames left out (without one, frames under a
    ``BAD_`` span are). The global signals are the mean of the kept long channels per
    chromophore. GVTD, band-passed to the motion band over ``od``'s long channels, and the
    IMU traces are interpolated onto ``haemo``'s samples.
    """
    times = haemo.times
    keep = (design["censored"].to_numpy() == 0 if design is not None and "censored" in design
            else kept_frames(haemo))
    columns: dict[str, np.ndarray] = {}
    long_names = set(long_short_channels(haemo, sep_bands)[0]) or set(haemo.ch_names)
    kept = [ch for ch in haemo.ch_names if ch in long_names and ch not in haemo.info["bads"]]
    for chroma, name in GLOBAL_NAMES.items():
        picks = [ch for ch in kept if chroma_of(ch) == chroma]
        if picks:
            columns[name] = haemo.get_data(picks=picks).mean(axis=0)
    if design is not None:
        for col in design.columns:
            if str(col).startswith(NUISANCE_PREFIXES):
                columns[str(col)] = design[col].to_numpy(dtype=float)
    if od is not None:
        picks = gvtd_channel_picks(od, sep_bands)[0]
        trace = gvtd_timetrace(od.get_data(picks=picks), float(od.info["sfreq"]),
                               *GVTD_MOTION_BAND)
        columns["GVTD"] = np.interp(times, od.times[1:], trace)
    for key, trace in (imu or {}).items():
        columns[_IMU_NAMES.get(key, key)] = np.interp(times, trace.t, trace.y)
    frame = pd.DataFrame(columns).loc[keep]
    return frame.loc[:, frame.std() > 0]


def _tint(hex_colour: str, white: float) -> str:
    """``hex_colour`` with ``white`` of white mixed in: ``_tint("#000000", 0.5)`` -> ``"#808080"``."""
    rgb = [int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c + (255 - c) * white):02x}" for c in rgb)


def nuisance_correlation_figure(frame: pd.DataFrame, max_series: int = MAX_SERIES
                                ) -> "go.Figure | None":
    """Left the correlations among the series, right each one's ``|r|`` with each global signal, as fMRIPrep draws them.

    The series kept are those with the largest ``|r|`` with either global signal, the global
    signals among them, up to ``max_series``. The heatmap keeps them in ``frame``'s order,
    one family beside the next; the bars run strongest first. None without a global signal
    and one series.
    """
    globals_ = [name for name in GLOBAL_NAMES.values() if name in frame.columns]
    if not globals_ or frame.shape[1] < 2:
        return None
    corr = frame.corr()
    strength = corr[globals_].abs().max(axis=1).sort_values(ascending=False, kind="stable")
    ranked = [c for c in strength.index if c not in globals_][:max_series - len(globals_)]
    order = [c for c in frame.columns if c in globals_ or c in ranked]
    shown = corr.loc[order, order].to_numpy(copy=True)
    np.fill_diagonal(shown, np.nan)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], horizontal_spacing=0.2)
    # white gaps between cells, the grid fMRIPrep's heatmap has
    fig.add_trace(go.Heatmap(
        z=shown, x=order, y=order, zmin=-1, zmax=1, zmid=0, colorscale=CORRELATION_SCALE,
        xgap=1, ygap=1,
        colorbar=dict(title=dict(text="r", side="right"), x=0.415, len=0.75, thickness=10),
        hovertemplate="%{y} / %{x}: r = %{z:.2f}<extra></extra>"), row=1, col=1)
    for chroma, name in GLOBAL_NAMES.items():
        if name not in globals_:
            continue
        fig.add_trace(go.Bar(
            x=ranked, y=corr.loc[ranked, name].abs(), name=name,
            marker_color=_tint(CHROMA_STYLE[chroma][1], _BAR_TINT),
            hovertemplate="%{x}: |r| = %{y:.2f}<extra>" + name + "</extra>"), row=1, col=2)
    # square cells: the y axis takes the x axis' scale and the panel shrinks to fit
    fig.update_yaxes(autorange="reversed", scaleanchor="x", constrain="domain", row=1, col=1)
    fig.update_xaxes(tickangle=-90, constrain="domain", row=1, col=1)
    fig.update_xaxes(tickangle=-90, title_text="Nuisance series", row=1, col=2)
    fig.update_yaxes(range=[0, 1], title_text="Magnitude of correlation with the global signal",
                     showgrid=True, gridcolor="#eceff1", row=1, col=2)
    fig.update_layout(height=640, barmode="group", plot_bgcolor="white", bargap=0.25,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.0,
                                  yanchor="bottom"),
                      margin=dict(l=40, r=20, t=40, b=40))
    return fig
