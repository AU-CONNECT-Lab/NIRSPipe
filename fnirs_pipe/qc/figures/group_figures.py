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


# Only scale-homogeneous metrics share a chart, so the y-axis stays in real units.
# Each metric in a group gets its own colour (subgroup).
_METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("Coupling & cardiac (0-1)",
     ["sci_mean", "psp_mean", "cp_mean",
      "channel_retention_rate", "pct_data_retained"]),
    ("GVTD amplitude",
     ["gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95", "gvtd_thresh"]),
    ("Motion fraction", ["gvtd_pct_above_thresh"]),
    ("Spike / motion counts", ["gvtd_num_above_thresh", "spike_count"]),
    ("Coefficient of variation", ["cv_mean", "cv_mean_760", "cv_mean_850"]),
    ("Intensity SNR", ["snr_mean"]),
    ("Mean amplitude", ["mean_amp_mean"]),
    ("HbO-HbR correlation", ["hbo_hbr_corr_mean"]),
    ("Global correlation", ["gcor_hbo", "gcor_hbr"]),
    ("Low-freq drift", ["lowfreq_drift_amplitude_hbo", "lowfreq_drift_amplitude_hbr"]),
    ("Residual physiology power", ["residual_cardiac_power", "residual_resp_power"]),
    ("Channel distance (m)", ["ch_dist_mean", "ch_dist_min", "ch_dist_max"]),
]


def _bare_metric(col: str) -> str:
    """Column name without its section prefix: ``raw_long_snr_mean`` -> ``snr_mean``.

    Quality records are sectioned, so the group table's columns are ``section_metric``
    while _METRIC_GROUPS above names bare metrics. Longest section first, or ``raw``
    would match a ``raw_long_`` column and leave ``long_snr_mean`` behind.
    """
    from fnirs_pipe.qc.sqm_record import SECTIONS

    for section in sorted(SECTIONS, key=len, reverse=True):
        if col.startswith(f"{section}_"):
            return col[len(section) + 1:]
    return col


def group_metrics(metric_cols: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Split metric_cols into (groups, ordered_flat) following _METRIC_GROUPS; leftovers -> 'Other'.

    Matching ignores the section prefix, so ``raw_sci_mean`` and ``raw_long_sci_mean``
    both land in the coupling group while staying separate columns.
    """
    by_metric: dict[str, list[str]] = {}
    for col in metric_cols:
        by_metric.setdefault(_bare_metric(col), []).append(col)

    groups: list[tuple[str, list[str]]] = []
    assigned: set[str] = set()
    for title, keys in _METRIC_GROUPS:
        present = [col for k in keys for col in by_metric.get(k, [])]
        if present:
            groups.append((title, present))
            assigned.update(present)
    leftover = [c for c in metric_cols if c not in assigned]
    if leftover:
        groups.append(("Other", leftover))
    ordered = [c for _, ks in groups for c in ks]
    return groups, ordered


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


_PALETTE = [
    "#3498db", "#e67e22", "#2ecc71", "#9b59b6",
    "#e74c3c", "#1abc9c", "#f1c40f", "#34495e",
]


def _group_box(title: str, keys: list[str], df: pd.DataFrame, rows: list[str]) -> go.Figure | None:
    """One chart: real-value box + jittered strip per metric, coloured per metric."""
    k = len(keys)
    fig = go.Figure()
    drawn = 0
    for xi, col in enumerate(keys):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        color = _PALETTE[xi % len(_PALETTE)]
        fig.add_trace(go.Box(
            x=[xi] * int(finite.sum()), y=vals[finite], name=col, width=0.5,
            boxpoints=False, line=dict(color=color, width=1.2),
            fillcolor="rgba(0,0,0,0)", hoverinfo="skip", showlegend=False,
        ))
        rng = np.random.default_rng(seed=abs(hash(col)) % (2**32))
        fig.add_trace(go.Scatter(
            x=xi + rng.uniform(-0.16, 0.16, size=len(rows)), y=vals,
            mode="markers",
            marker=dict(color=color, size=7, opacity=0.85,
                        line=dict(width=0.5, color="#2c3e50")),
            customdata=rows,
            hovertemplate="<b>%{customdata}</b><br>" + col + "=%{y:.4g}<extra></extra>",
            showlegend=False,
        ))
        drawn += 1
    if drawn == 0:
        return None

    fig.update_xaxes(
        tickvals=list(range(k)), ticktext=keys, range=[-0.6, k - 0.4],
        tickangle=-30 if k > 1 else 0, gridcolor="#f2f2f2",
    )
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False)
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=12)),
        width=150 + 78 * k, height=320,
        margin=dict(l=58, r=14, t=34, b=70),
        boxmode="overlay", plot_bgcolor="white",
    )
    return fig


def build_grouped_boxes(
    df: pd.DataFrame, metric_cols: list[str],
) -> list[tuple[str, go.Figure]]:
    """Flow of small real-value charts, one per scale-homogeneous group.

    Each point carries its bids_name (customdata) so the report can link to it."""
    if not metric_cols or df.empty:
        return []
    groups, _ = group_metrics(metric_cols)
    rows = df.iloc[:, 0].astype(str).tolist()
    out: list[tuple[str, go.Figure]] = []
    for title, keys in groups:
        fig = _group_box(title, keys, df, rows)
        if fig is not None:
            out.append((title, fig))
    return out
