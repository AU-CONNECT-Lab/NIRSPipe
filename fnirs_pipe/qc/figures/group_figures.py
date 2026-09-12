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


# Only scale-homogeneous metrics share a chart, so the y-axis stays in real units. Keys are
# bare metric names: which stage and which channel set a column came from is carried by the
# chart's x label and its colour, not by a key of its own.
_METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("Coupling & retention (0-1)",
     ["sci_mean", "cp_mean", "good_frac_mean", "channel_retention_rate",
      "snr_pass_rate", "pct_data_retained"]),
    ("Peak spectral power", ["psp_mean"]),
    ("Coefficient of variation", ["cv_mean", "cv_mean_760", "cv_mean_850"]),
    ("Intensity SNR", ["snr_mean"]),
    ("Mean amplitude", ["mean_amp_mean"]),
    ("GVTD amplitude",
     ["gvtd_mean", "gvtd_p95", "gvtd_filt_mean", "gvtd_filt_p95",
      "gvtd_thresh", "gvtd_thresh_applied", "gvtd_censor_thresh"]),
    ("GVTD in SD units", ["gvtd_vstd_mean", "gvtd_vstd_p95"]),
    ("Motion & spike fraction",
     ["gvtd_pct_above_thresh", "spike_pct", "spike_pct_frames",
      "gvtd_censor_pct", "motion_corrected_pct", "motion_corrected_frac_mean"]),
    ("Motion & spike counts",
     ["gvtd_num_above_thresh", "spike_count", "spike_num_frames",
      "gvtd_censor_n_spans", "motion_corrected_num", "motion_corrected_n_segments"]),
    ("Retained recording (s)", ["gvtd_censor_retained_s"]),
    ("HbO-HbR correlation", ["hbo_hbr_corr_mean"]),
    ("Global correlation", ["gcor_hbo", "gcor_hbr"]),
    ("Low-freq drift", ["lowfreq_drift_amplitude_hbo", "lowfreq_drift_amplitude_hbr"]),
    ("Physiology band power",
     ["cardiac_band_power_hbo", "cardiac_band_power_hbr",
      "resp_band_power_hbo", "resp_band_power_hbr"]),
    ("Physiology band fraction",
     ["cardiac_band_frac_hbo", "cardiac_band_frac_hbr",
      "resp_band_frac_hbo", "resp_band_frac_hbr"]),
    ("Contrast-to-noise", ["cnr_hbo_mean", "cnr_hbr_mean"]),
    ("Channel & epoch counts",
     ["n_channels", "n_long_channels", "n_short_channels", "n_bad",
      "n_flat_channels", "cnr_n_epochs", "gvtd_censor_n_epochs"]),
    ("Channel distance (m)", ["ch_dist_mean", "ch_dist_min", "ch_dist_max"]),
    ("Separation (mm)", ["sep_short_max_mm", "sep_long_min_mm", "sep_long_max_mm"]),
]

# Settings the record stores beside its metrics. A distribution of a number the run was told
# to use, rather than one it measured, says nothing: they stay in the table and out of the
# figures.
_SETTING_METRICS = frozenset(
    {"qc_window_s", "gvtd_censor_n_std", "gvtd_censor_min_epoch_s"})

# ---- Channel sets ----

# One colour per channel set, shared by every chart so the legend means the same thing on all
# of them. The set is the section's `_long` / `_short` suffix; a section without one measured
# every channel.
_CHANNEL_SETS = ("all", "long", "short")
_SET_COLOURS = {"all": "#34495e", "long": "#3498db", "short": "#e67e22"}


def _split_column(col: str) -> tuple[str, str, str]:
    """Split a group-table column into (stage, channel set, metric).

    ::

        "raw_long_sci_mean"  ->  ("raw", "long", "sci_mean")
        "errts_gcor_hbo"     ->  ("errts", "all", "gcor_hbo")

    Longest section first, or ``raw`` would match a ``raw_long_`` column and leave
    ``long_sci_mean`` behind.
    """
    from fnirs_pipe.qc.sqm_record import SECTIONS

    for section in sorted(SECTIONS, key=len, reverse=True):
        if col.startswith(f"{section}_"):
            stage, _, suffix = section.rpartition("_")
            if suffix in ("long", "short"):
                return stage, suffix, col[len(section) + 1:]
            return section, "all", col[len(section) + 1:]
    return "", "all", col


def _bare_metric(col: str) -> str:
    """Column name without its section prefix: ``raw_long_snr_mean`` -> ``snr_mean``."""
    return _split_column(col)[2]


def group_metrics(metric_cols: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Split metric_cols into (groups, ordered_flat) following _METRIC_GROUPS; leftovers -> 'Other'.

    Matching ignores the section prefix, so ``raw_sci_mean`` and ``raw_long_sci_mean``
    both land in the coupling group while staying separate columns. Settings are dropped
    rather than grouped, so neither the ordering nor the charts carry them.
    """
    by_metric: dict[str, list[str]] = {}
    for col in metric_cols:
        bare = _bare_metric(col)
        if bare not in _SETTING_METRICS:
            by_metric.setdefault(bare, []).append(col)

    groups: list[tuple[str, list[str]]] = []
    assigned: set[str] = set()
    for title, keys in _METRIC_GROUPS:
        present = [col for k in keys for col in by_metric.get(k, [])]
        if present:
            groups.append((title, present))
            assigned.update(present)
    leftover = [c for cols in by_metric.values() for c in cols if c not in assigned]
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
    """One chart: real-value box + jittered strip, x = metric, colour = channel set.

    The three channel sets of a metric sit side by side over one x position rather than at
    three x positions of their own, so a metric's all / long / short read as one comparison
    and the chart is a third as wide. The stage is dropped from the x label when the group
    only holds one, which is what a cohort measured by `prep-raw` alone looks like.
    """
    cats: list[tuple[str, str]] = []          # (stage, metric), in the group's order
    columns: dict[tuple[str, str], dict[str, str]] = {}   # cat -> {channel set: column}
    for col in keys:
        stage, cset, metric = _split_column(col)
        cat = (stage, metric)
        if cat not in columns:
            cats.append(cat)
            columns[cat] = {}
        columns[cat][cset] = col

    sets = [s for s in _CHANNEL_SETS if any(s in c for c in columns.values())]
    one_stage = len({stage for stage, _ in cats}) == 1
    labels = [metric if one_stage else f"{stage} {metric}" for stage, metric in cats]
    k = len(cats)
    slot = 0.8 / max(len(sets), 1)

    fig = go.Figure()
    drawn = 0
    for j, cset in enumerate(sets):
        offset = (j - (len(sets) - 1) / 2) * slot
        box_x: list[float] = []
        box_y: list[float] = []
        pt_x: list[float] = []
        pt_y: list[float] = []
        pt_meta: list[list[str]] = []
        for i, cat in enumerate(cats):
            col = columns[cat].get(cset)
            if col is None:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            box_x += [i + offset] * int(finite.sum())
            box_y += vals[finite].tolist()
            rng = np.random.default_rng(seed=abs(hash(col)) % (2**32))
            pt_x += (i + offset
                     + rng.uniform(-slot * 0.22, slot * 0.22, size=len(rows))).tolist()
            pt_y += vals.tolist()
            pt_meta += [[r, col] for r in rows]
        if not box_y:
            continue
        drawn += 1
        colour = _SET_COLOURS[cset]
        fig.add_trace(go.Box(
            x=box_x, y=box_y, width=slot * 0.68, boxpoints=False,
            line=dict(color=colour, width=1.2), fillcolor="rgba(0,0,0,0)",
            hoverinfo="skip", showlegend=False, legendgroup=cset,
        ))
        fig.add_trace(go.Scatter(
            x=pt_x, y=pt_y, mode="markers", name=cset, legendgroup=cset,
            marker=dict(color=colour, size=7, opacity=0.85,
                        line=dict(width=0.5, color="#2c3e50")),
            customdata=pt_meta,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}=%{y:.4g}<extra></extra>",
            showlegend=len(sets) > 1,
        ))
    if drawn == 0:
        return None

    fig.update_xaxes(
        tickvals=list(range(k)), ticktext=labels, range=[-0.6, k - 0.4],
        tickangle=-30 if k > 1 else 0, gridcolor="#f2f2f2",
    )
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False)
    fig.update_layout(
        title=dict(text=f"{title} ({cats[0][0]})" if one_stage and cats[0][0] else title,
                   x=0.02, xanchor="left", font=dict(size=12)),
        width=150 + (46 + 26 * len(sets)) * k, height=340,
        margin=dict(l=58, r=14, t=34, b=70),
        boxmode="overlay", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    font=dict(size=10)),
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
