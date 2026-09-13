"""Plotly figure builders for cohort-level QC reports (individual and hyper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fnirs_pipe.qc.figures.common._utils import CONDITION_PALETTE, LONG_COLOR, SHORT_COLOR


# Only scale-homogeneous metrics share a chart, so the y-axis stays in real units. Keys are
# bare metric names: which stage and which channel set a column came from is carried by the
# chart's x label and its colour, not by a key of its own.
_METRIC_GROUPS: list[tuple[str, list[str]]] = [
    ("Coupling & retention (0-1)",
     ["sci_win_mean", "sci_mean", "cp_mean", "good_frac_mean", "channel_retention_rate",
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
    from fnirs_pipe.qc.subject.sqm_record import SECTIONS

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


# ---- Deviation from the cohort ----

# Mean |z| over every metric, above which a run is called out. One number per run rather than
# one test per metric: the group table carries 85 columns, and fencing each of them separately
# and flagging a run that trips any one flags nearly every run in a large cohort whatever its
# quality. A fence on the scores themselves masks instead: four poor runs in a cohort of twenty
# pull the fence up over their own heads.
SCORE_THRESHOLD = 2.0

# One colour per run picked out of the pale mass, the same colour in every panel.
_HIGHLIGHT_COLOURS = ["#c0392b", "#8e44ad", "#d35400"]


def _robust_z(values: np.ndarray) -> np.ndarray:
    """Distance from the median in robust SDs; all-NaN when the column has no spread.

    ::

        [0.90, 0.94, 0.98]  ->  [-0.67, 0.0, 0.67]
        [28, 28, 28]        ->  [nan, nan, nan]

    A metric every run agreed on has no distance to report, and saying so with NaN keeps it
    out of the picture instead of drawing it as average.
    """
    out = np.full(np.shape(values), np.nan, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return out
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    scale = 1.4826 * mad if mad > 0 else np.std(finite)
    if scale == 0:
        return out
    return (values - median) / scale


def deviation_scores(df: pd.DataFrame, metric_cols: list[str]) -> np.ndarray:
    """Mean |z| per run over every metric that has any spread."""
    columns = [c for c in metric_cols if c in df.columns]
    if not columns:
        return np.full(len(df), np.nan)
    stack = np.array([_robust_z(pd.to_numeric(df[c], errors="coerce").to_numpy())
                      for c in columns])
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.abs(stack), axis=0)


def detect_outliers(
    df: pd.DataFrame, metric_cols: list[str], threshold: float = SCORE_THRESHOLD,
) -> dict[str, list[str]]:
    """{bids_name: [worst metrics]} for runs whose mean |z| is past the threshold.

    The metric list is what put the run there, worst first, so the reader gets the reason
    beside the verdict rather than a name alone.
    """
    columns = [c for c in metric_cols if c in df.columns]
    if not columns:
        return {}
    stack = np.array([_robust_z(pd.to_numeric(df[c], errors="coerce").to_numpy())
                      for c in columns])
    with np.errstate(invalid="ignore"):
        scores = np.nanmean(np.abs(stack), axis=0)

    out: dict[str, list[str]] = {}
    for i, score in enumerate(scores):
        if not np.isfinite(score) or score <= threshold:
            continue
        per_metric = np.abs(stack[:, i])
        order = np.argsort(-np.where(np.isfinite(per_metric), per_metric, -np.inf))
        worst = [columns[j] for j in order[:6]
                 if np.isfinite(per_metric[j]) and per_metric[j] > 2]
        out[str(df.iloc[i, 0])] = worst
    return out


def _short_run_labels(runs: list[str]) -> list[str]:
    """Run labels with the BIDS fields every run shares taken off.

    ::

        ["sub-01_task-full", "sub-02_task-full"]  ->  ["sub-01", "sub-02"]

    An axis with one tick per run has no room for the part of the name that is the same on
    all of them. The full name stays in the hover.
    """
    if len(runs) < 2:
        return list(runs)
    fields = [r.split("_") for r in runs]
    keep = [i for i in range(max(len(f) for f in fields))
            if len({f[i] if i < len(f) else "" for f in fields}) > 1]
    if not keep:
        return list(runs)
    return ["_".join(f[i] for i in keep if i < len(f)) for f in fields]


def build_deviation_strip(
    df: pd.DataFrame, metric_cols: list[str],
) -> "tuple[go.Figure | None, list[str]]":
    """One row per metric, one dot per run at its robust z, colour = channel set.

    This is the report's overview, in place of a subject x metric heatmap. A heatmap of
    z-scores has two failure modes this does not: at a small cohort every cell takes one of
    two colours, and at any cohort the metric names have to be read sideways. Here the metric
    names are horizontal, the cohort's spread is the width of its dot cloud, and a run sitting
    apart is apart on the axis rather than a shade darker.

    Returns (figure, dropped): a metric every run agreed on has no z to plot and is named in
    the report's notes instead.
    """
    runs = df.iloc[:, 0].astype(str).tolist()
    if not runs or not metric_cols:
        return None, []

    groups, _ = group_metrics(metric_cols)
    entries: list[dict] = []
    index: dict[tuple[str, str], dict] = {}
    for title, keys in groups:
        for col in keys:
            stage, channel_set, metric = _split_column(col)
            entry = index.get((stage, metric))
            if entry is None:
                entry = {"group": title, "stage": stage, "metric": metric, "z": {}}
                index[(stage, metric)] = entry
                entries.append(entry)
            entry["z"][channel_set] = _robust_z(
                pd.to_numeric(df[col], errors="coerce").to_numpy())

    dropped = [f'{e["stage"]}_{e["metric"]}' if e["stage"] else e["metric"]
               for e in entries if not any(np.isfinite(z).any() for z in e["z"].values())]
    kept = [e for e in entries if any(np.isfinite(z).any() for z in e["z"].values())]
    if not kept:
        return None, dropped

    one_stage = len({e["stage"] for e in kept}) == 1
    labels = [e["metric"] if one_stage else f'{e["stage"]} {e["metric"]}' for e in kept]
    sets = [s for s in _CHANNEL_SETS if any(s in e["z"] for e in kept)]
    jitter = np.random.default_rng(0)
    many = len(runs) > 12

    fig = go.Figure()
    for i, entry in enumerate(kept):
        for j, channel_set in enumerate(sets):
            z = entry["z"].get(channel_set)
            if z is None:
                continue
            offset = (j - (len(sets) - 1) / 2) * (0.7 / max(len(sets), 1))
            fig.add_trace(go.Scatter(
                x=z, y=i + offset + jitter.uniform(-0.06, 0.06, len(z)), mode="markers",
                name=channel_set, legendgroup=channel_set, showlegend=(i == 0),
                marker=dict(color=_SET_COLOURS[channel_set], size=5 if many else 7,
                            opacity=0.45 if many else 0.8,
                            line=dict(width=0.4, color="#fff")),
                customdata=runs,
                hovertemplate=("<b>%{customdata}</b><br>" + entry["metric"]
                               + " z=%{x:.2f}<extra></extra>"),
            ))
    fig.add_vrect(x0=-2, x1=2, fillcolor="#3498db", opacity=0.06, line_width=0)
    fig.add_vline(x=0, line_color="#adb5bd", line_width=1)
    for i in range(1, len(kept)):
        if kept[i]["group"] != kept[i - 1]["group"]:
            fig.add_hline(y=i - 0.5, line_width=1, line_color="#eef0f3")

    fig.update_yaxes(tickvals=list(range(len(kept))), ticktext=labels,
                     range=[len(kept) - 0.5, -0.5], gridcolor="#f7f8f9",
                     tickfont=dict(size=10))
    fig.update_xaxes(title_text="robust z (0 = cohort median)", gridcolor="#f0f0f0",
                     zeroline=False)
    fig.update_layout(height=110 + 34 * len(kept), plot_bgcolor="white",
                      margin=dict(l=150, r=20, t=40, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                                  font=dict(size=10)))
    return fig, dropped


# ---- Windowed series, per channel set ----

# The windowed metrics that are stored as a channel x window matrix: (key, label, matrix
# field, the per_channel field whose key order the matrix rows follow, window centres).
_WINDOW_MATRICES = [
    ("sci", "SCI (windowed)", "sci_matrix", "sci_per_channel", "sci_window_times_s"),
    ("psp", "PSP (windowed)", "psp_matrix", "psp_per_channel", "psp_window_times_s"),
    ("cv",  "CV (windowed)",  "cv_matrix",  "cv_per_channel",  "cv_window_times_s"),
]

# GVTD is stored as a series per channel set already. The bare key is the long channels,
# following `spike_spans_s`; `_short` and `_all` name the others.
_GVTD_SETS = {"long": "gvtd_per_window", "short": "gvtd_per_window_short",
              "all": "gvtd_per_window_all"}

# The trend a reader looks for in a windowed metric is slower than one QC window, so each
# series is smoothed over this many seconds and then sampled at that same step. Drawing 200
# runs at full resolution is a quarter of a million samples for a picture of a grey mass.
SMOOTH_S = 60.0


def _set_rows(row: dict, per_channel_field: str) -> dict[str, list[int]]:
    """Row indices of each channel set for a stored channel x window matrix.

    The matrix carries no channel names; its rows follow the recording's channel order, which
    is the key order of ``per_channel.raw.<field>``. Verified rather than assumed: splitting
    the matrices this way reproduces the stored per-set psp_mean and cv_mean exactly.
    """
    per_channel = row.get("per_channel") or {}
    names = list((per_channel.get("raw") or {}).get(per_channel_field) or {})
    if not names:
        return {}
    sets = {"all": list(range(len(names)))}
    for channel_set, section in (("long", "raw_long"), ("short", "raw_short")):
        members = set((per_channel.get(section) or {}).get(per_channel_field) or {})
        rows = [i for i, name in enumerate(names) if name in members]
        if rows:
            sets[channel_set] = rows
    return sets


def _window_series(row: dict, key: str) -> dict[str, tuple]:
    """{channel set: (times, values)} for one windowed metric of one run."""
    windowed = row
    if key == "gvtd":
        times = np.asarray(windowed.get("gvtd_window_times_s") or [], dtype=float)
        out = {}
        for channel_set, field in _GVTD_SETS.items():
            values = windowed.get(field)
            if values is not None and len(values) == times.size and times.size:
                out[channel_set] = (times, np.asarray(values, dtype=float))
        return out

    spec = next((m for m in _WINDOW_MATRICES if m[0] == key), None)
    if spec is None:
        return {}
    _key, _label, matrix_field, per_channel_field, times_field = spec
    matrix = windowed.get(matrix_field)
    times = np.asarray(windowed.get(times_field) or [], dtype=float)
    if matrix is None or not times.size:
        return {}
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != times.size:
        return {}
    out = {}
    for channel_set, rows in _set_rows(row, per_channel_field).items():
        if rows:
            with np.errstate(invalid="ignore"):
                out[channel_set] = (times, np.nanmean(matrix[rows], axis=0))
    return out


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling median, the edges held at their own value rather than dropped."""
    if window < 2:
        return values
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(values.size)])


def _bundle(x, ys, colour: str = "#c8d0d8", width: float = 0.5, opacity: float = 0.45):
    """Every run as one trace, the runs separated by a None in the data.

    Plotly slows to a crawl somewhere past a thousand traces, and a cohort of 200 drawn a
    line at a time reaches that on one panel. No hover: a line inside a grey mass cannot be
    pointed at, and a name per sample doubles the file.
    """
    xs: list = []
    values: list = []
    for y in ys:
        xs.extend(list(x) + [None])
        values.extend(list(y) + [None])
    return go.Scatter(x=xs, y=values, mode="lines", line=dict(color=colour, width=width),
                      opacity=opacity, showlegend=False, connectgaps=False, hoverinfo="skip")


def _stacked_series(rows: list[dict], key: str, channel_set: str) -> tuple:
    """(times, matrix) of one metric and channel set over the cohort, on a shared grid.

    Each run is smoothed and then sampled at the smoothing step. Runs binned differently land
    on the union of their grids, with the gaps left as NaN rather than interpolated.
    """
    series = []
    for row in rows:
        got = _window_series(row, key).get(channel_set)
        if got is None:
            continue
        times, values = got
        window = max(1, int(round(SMOOTH_S / float(row.get("qc_window_s") or 10.0))))
        series.append((times[::window], _smooth(values, window)[::window]))
    if not series:
        return None, None
    grid = np.unique(np.concatenate([t for t, _ in series]))
    stack = np.full((len(series), grid.size), np.nan)
    for i, (times, values) in enumerate(series):
        stack[i, np.searchsorted(grid, times)] = values
    return grid, stack


def build_window_grid(
    rows: list[dict], highlight: "list[str] | None" = None,
    metrics: tuple = ("sci", "cv", "gvtd"),
) -> "go.Figure | None":
    """Metric x channel-set grid over time: every run pale, the cohort's band and median over.

    Rows are metrics and columns are channel sets, sharing the y axis along a row so the three
    sets are read against each other. Only the short channels degrading is a coupling story
    and the long ones going with them is a movement story, and nothing else in the report
    separates the two.

    ``highlight`` names runs that keep a line of their own, one colour each, the same colour
    in every cell.
    """
    from plotly.subplots import make_subplots

    names = [str(r.get("bids_name", "")) for r in rows]
    picked = [(key, label) for key, label, *_rest in _WINDOW_MATRICES if key in metrics]
    if "gvtd" in metrics:
        picked.append(("gvtd", "GVTD (windowed)"))
    panels = [(key, label) for key, label in picked
              if any(_window_series(r, key) for r in rows)]
    if not panels:
        return None

    sets = [s for s in _CHANNEL_SETS
            if any(s in _window_series(r, key) for r in rows for key, _l in panels)]
    marked = [i for i, name in enumerate(names) if name in set(highlight or [])]
    many = len(rows) > 12

    fig = make_subplots(
        rows=len(panels), cols=len(sets), shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.055, horizontal_spacing=0.025,
        subplot_titles=[s if r == 0 else "" for r in range(len(panels)) for s in sets],
    )
    for r, (key, label) in enumerate(panels, start=1):
        for c, channel_set in enumerate(sets, start=1):
            grid, stack = _stacked_series(rows, key, channel_set)
            if grid is None:
                continue
            rest = [i for i in range(stack.shape[0]) if i not in marked]
            if rest:
                fig.add_trace(_bundle(grid, stack[rest], opacity=0.3 if many else 0.5),
                              row=r, col=c)
            with np.errstate(invalid="ignore"):
                lo, mid, hi = (np.nanpercentile(stack, q, axis=0) for q in (25, 50, 75))
            fig.add_trace(go.Scatter(
                x=np.concatenate([grid, grid[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                fill="toself", mode="lines", line=dict(width=0),
                fillcolor="rgba(52,152,219,0.18)", hoverinfo="skip", showlegend=False,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=grid, y=mid, mode="lines", line=dict(color="#2471a3", width=1.8),
                name="cohort median", legendgroup="median",
                showlegend=(r == 1 and c == 1),
                hovertemplate="t=%{x:.0f}s<br>median %{y:.4g}<extra></extra>",
            ), row=r, col=c)
            for k, i in enumerate(marked):
                if i >= stack.shape[0]:
                    continue
                fig.add_trace(go.Scatter(
                    x=grid, y=stack[i], mode="lines",
                    line=dict(color=_HIGHLIGHT_COLOURS[k % len(_HIGHLIGHT_COLOURS)],
                              width=1.3),
                    name=_short_run_labels(names)[i], legendgroup=names[i],
                    showlegend=(r == 1 and c == 1),
                    hovertemplate="t=%{x:.0f}s<br>%{y:.4g}<extra></extra>",
                ), row=r, col=c)
        fig.update_yaxes(title_text=label, title_font=dict(size=10), row=r, col=1)

    fig.update_xaxes(gridcolor="#f5f5f5", tickfont=dict(size=9))
    for c in range(1, len(sets) + 1):
        fig.update_xaxes(title_text="Time (s)", title_font=dict(size=10),
                         row=len(panels), col=c)
    fig.update_yaxes(gridcolor="#f0f0f0", tickfont=dict(size=9))
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=165 * len(panels) + 110, plot_bgcolor="white",
                      margin=dict(l=70, r=20, t=80, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                  font=dict(size=10)))
    return fig


def _rgba(hex_colour: str, alpha: float) -> str:
    """``"#e74c3c", 0.1 -> "rgba(231,76,60,0.1)"``, for a band drawn under a line."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def condition_colours(conditions: list[str]) -> dict[str, str]:
    """One colour per condition, by the rule the run report's markers already follow.

    ``raw_figures.condition_colors`` cycles :data:`CONDITION_PALETTE` over the descriptions
    in the order they first appear, and a condition keeps the colour it wears there, so a
    band on this page and a marker on the run report are the same colour for the same block.
    """
    return {c: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, c in enumerate(conditions)}


def build_condition_timeline(
    row: dict, windows: "list[tuple[str, float, float]]",
    metrics: tuple = ("sci", "cv", "gvtd"),
) -> "go.Figure | None":
    """One run's windowed metrics over time, the conditions shaded behind them.

    The cohort's :func:`build_window_grid` splits the channel sets into columns because it
    is drawing many runs; a subject page draws one, so the sets share a panel and the columns
    are spent on nothing. Long and short only: ``all`` is a blend of the two and lands
    between them, a third line for no third answer.

    Smoothed over :data:`SMOOTH_S` and sampled at that step for the reason recorded there,
    which also takes a 390-window run down to 65 points a line.

    Colours come from the run report rather than from ``_SET_COLOURS``. That palette puts the
    long channels on the blue ``_utils.HBR_COLOR`` uses, so a long-channel line here and an
    HbR trace one click away would be the same blue; ``LONG_COLOR`` and ``SHORT_COLOR`` are
    the separation split every raw-level view already wears.
    """
    from plotly.subplots import make_subplots

    picked = [(key, label) for key, label, *_rest in _WINDOW_MATRICES if key in metrics]
    if "gvtd" in metrics:
        picked.append(("gvtd", "GVTD"))
    panels = [(key, label) for key, label in picked if _window_series(row, key)]
    if not panels:
        return None

    step = max(1, int(round(SMOOTH_S / float(row.get("qc_window_s") or 10.0))))
    set_colour = {"long": LONG_COLOR, "short": SHORT_COLOR}
    colours = condition_colours([label for label, _t0, _t1 in windows])

    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.045)
    for r, (key, label) in enumerate(panels, start=1):
        series = _window_series(row, key)
        for channel_set in ("long", "short"):
            if channel_set not in series:
                continue
            times, values = series[channel_set]
            fig.add_trace(go.Scatter(
                x=times[::step], y=_smooth(values, step)[::step], mode="lines",
                name=channel_set, legendgroup=channel_set, showlegend=(r == 1),
                line=dict(color=set_colour[channel_set], width=2.0),
                hovertemplate=f"{label} {channel_set}<br>t=%{{x:.0f}}s %{{y:.4g}}<extra></extra>",
            ), row=r, col=1)
        fig.update_yaxes(title_text=label, title_font=dict(size=10), row=r, col=1)

    for name, t0, t1 in windows:
        fig.add_vrect(x0=t0, x1=t1, fillcolor=_rgba(colours[name], 0.10), line_width=0,
                      layer="below", row="all", col=1)
        # named once, on the top panel: a label per panel is the same word four times
        fig.add_annotation(x=(t0 + t1) / 2, y=1.0, yref="y domain", text=name,
                           showarrow=False, yanchor="bottom", row=1, col=1,
                           font=dict(size=9, color=colours[name]))

    fig.update_xaxes(gridcolor="#f5f5f5", tickfont=dict(size=9))
    fig.update_xaxes(title_text="Time (s)", title_font=dict(size=10), row=len(panels), col=1)
    fig.update_yaxes(gridcolor="#f0f0f0", tickfont=dict(size=9))
    fig.update_layout(height=170 * len(panels) + 90, plot_bgcolor="white",
                      margin=dict(l=70, r=20, t=60, b=45),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                                  font=dict(size=10)))
    return fig


# ---- Per condition ----

# What the per-condition panels report, in this order. Each is a key of a condition's
# `scalars` block.
_CONDITION_METRICS = [
    ("sci_win_mean", "SCI"), ("psp_mean", "PSP"), ("cv_mean", "CV"), ("snr_mean", "SNR"),
    ("gvtd_mean", "GVTD"), ("gvtd_pct_above_thresh", "GVTD above threshold"),
]

def condition_names(rows: list[dict]) -> list[str]:
    """Conditions the cohort has, in the order the first run that carries them wrote them."""
    names: list[str] = []
    for row in rows:
        for condition in (row.get("by_condition") or {}):
            if condition not in names:
                names.append(condition)
    return names


def _condition_matrix(rows: list[dict], conditions: list[str], metric: str) -> np.ndarray:
    """runs x conditions of one metric, NaN where a run does not carry that condition."""
    out = np.full((len(rows), len(conditions)), np.nan)
    for i, row in enumerate(rows):
        by_condition = row.get("by_condition") or {}
        for j, condition in enumerate(conditions):
            scalars = (by_condition.get(condition) or {}).get("scalars") or {}
            value = scalars.get(metric)
            if isinstance(value, (int, float)):
                out[i, j] = float(value)
    return out


def build_condition_panels(
    rows: list[dict], highlight: "list[str] | None" = None,
) -> "go.Figure | None":
    """One panel per metric: x is the condition, one line per run, the median over them.

    The windows collapsed into the blocks the run was designed around, which is the form the
    question takes: did this run get worse where everyone got worse, or on its own.
    """
    from plotly.subplots import make_subplots

    conditions = condition_names(rows)
    if len(conditions) < 2:
        return None
    names = [str(r.get("bids_name", "")) for r in rows]
    short = _short_run_labels(names)
    marked = [i for i, name in enumerate(names) if name in set(highlight or [])]
    many = len(rows) > 12

    panels = [(metric, label) for metric, label in _CONDITION_METRICS
              if np.isfinite(_condition_matrix(rows, conditions, metric)).any()]
    if not panels:
        return None

    cols = 3
    grid_rows = (len(panels) + cols - 1) // cols
    fig = make_subplots(rows=grid_rows, cols=cols, subplot_titles=[l for _m, l in panels],
                        vertical_spacing=0.13, horizontal_spacing=0.07)
    for i, (metric, _label) in enumerate(panels):
        r, c = divmod(i, cols)
        values = _condition_matrix(rows, conditions, metric)
        rest = [k for k in range(len(rows)) if k not in marked]
        if rest:
            fig.add_trace(_bundle(conditions, values[rest], width=0.5 if many else 0.7,
                                  opacity=0.35 if many else 0.6), row=r + 1, col=c + 1)
        for k, idx in enumerate(marked):
            fig.add_trace(go.Scatter(
                x=conditions, y=values[idx], mode="lines+markers",
                line=dict(color=_HIGHLIGHT_COLOURS[k % len(_HIGHLIGHT_COLOURS)], width=1.5),
                marker=dict(size=4), name=short[idx], legendgroup=names[idx],
                showlegend=(i == 0),
                hovertemplate=f"<b>{names[idx]}</b><br>%{{x}}: %{{y:.4g}}<extra></extra>",
            ), row=r + 1, col=c + 1)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(values, axis=0)
        fig.add_trace(go.Scatter(
            x=conditions, y=median, mode="lines+markers",
            line=dict(color="#2471a3", width=2.2), marker=dict(size=5, color="#2471a3"),
            name="cohort median", legendgroup="median", showlegend=(i == 0),
            hovertemplate="%{x}: median %{y:.4g}<extra></extra>",
        ), row=r + 1, col=c + 1)

    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9), gridcolor="#f7f8f9")
    fig.update_yaxes(tickfont=dict(size=9), gridcolor="#f0f0f0")
    fig.update_annotations(font=dict(size=11))
    fig.update_layout(height=230 * grid_rows + 70, plot_bgcolor="white",
                      margin=dict(l=55, r=20, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                  font=dict(size=10)))
    return fig


def _condition_heatmap(panels: "list[tuple[str, np.ndarray]]", conditions: list[str],
                       labels: list[str], row_label: str) -> "go.Figure":
    """One heatmap per metric over a shared y axis, colour = robust z inside a condition.

    ``panels`` are ``(title, values)`` with values a ``len(labels) x len(conditions)`` array
    in real units; the z-scoring is down a condition's own column, so a cell says how far
    that row sat from the others *in that condition* rather than how the conditions rank.

    Two pictures are this one with a different y axis: runs against conditions over a cohort,
    and channels against conditions inside one run. Sharing the renderer is what keeps them
    reading the same way, since a reader who learned the colour on one meets it on the other.

    The y labels sit on the left edge only. The panels share the axis, and repeating the
    names over every panel's cells is what made this unreadable.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True,
                        subplot_titles=[t for t, _v in panels], horizontal_spacing=0.012)
    for i, (_title, values) in enumerate(panels):
        z = np.column_stack([_robust_z(values[:, j]) for j in range(len(conditions))])
        fig.add_trace(go.Heatmap(
            z=z, x=conditions, y=labels, coloraxis="coloraxis", xgap=1, ygap=1,
            text=[[f"{v:.4g}" for v in row] for row in values],
            hovertemplate=(f"<b>{row_label} %{{y}}</b><br>%{{x}}<br>"
                           "value %{text}<br>z=%{z:.2f}<extra></extra>"),
        ), row=1, col=i + 1)

    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_yaxes(showticklabels=True, tickfont=dict(size=9), row=1, col=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9))
    fig.update_annotations(font=dict(size=11, color="#6c757d"))
    fig.update_layout(height=150 + 14 * len(labels), width=180 * len(panels) + 130,
                      plot_bgcolor="white", margin=dict(l=110, r=20, t=70, b=90),
                      coloraxis=dict(colorscale="RdBu", cmid=0, cmin=-3, cmax=3,
                                     colorbar=dict(title="z", thickness=12, len=0.8)))
    return fig


def build_condition_matrix(
    rows: list[dict], order: "list[str] | None" = None,
) -> "go.Figure | None":
    """run x condition, one panel per metric, colour = robust z inside that condition.

    Sorted worst-first when an order is given, so the runs a reader is looking for are at the
    top.
    """
    conditions = condition_names(rows)
    if len(conditions) < 2:
        return None
    names = [str(r.get("bids_name", "")) for r in rows]
    index = list(range(len(rows)))
    if order:
        rank = {name: i for i, name in enumerate(order)}
        index.sort(key=lambda i: rank.get(names[i], len(order)))
    labels = [_short_run_labels(names)[i] for i in index]

    panels = [(label, _condition_matrix(rows, conditions, metric)[index])
              for metric, label in _CONDITION_METRICS
              if np.isfinite(_condition_matrix(rows, conditions, metric)).any()]
    if not panels:
        return None
    return _condition_heatmap(panels, conditions, labels, "run")


# The per-channel field behind each condition panel, where one exists. GVTD has none: it is
# an RMS across channels by definition, and so is its above-threshold share, which is why
# those two panels belong to the profile alone and this picture is shorter by two.
_CONDITION_PER_CHANNEL = {
    "sci_win_mean": "sci_per_channel",
    "psp_mean":     "psp_per_channel",
    "cv_mean":      "cv_per_channel",
    "snr_mean":     "snr_per_channel",
}

# Appended after them. `good_frac` has no panel in the profile and belongs here because it
# is the line a channel is actually rejected on, which is what a condition's kept count is.
_EXTRA_CHANNEL_PANELS = (("good_frac_per_channel", "Coupled windows"),)


def _channel_matrix(by_condition: dict, conditions: list[str],
                    field: str) -> "tuple[list[str], np.ndarray]":
    """``(channel names, channels x conditions)`` of one per-channel field, NaN where absent.

    The names come from the first condition that carries the field, which is the recording's
    channel order; a condition missing a channel leaves that cell NaN rather than shifting
    the rows under it.
    """
    names: list[str] = []
    for condition in conditions:
        stored = ((by_condition.get(condition) or {}).get("per_channel") or {}).get(field)
        if stored:
            names = list(stored)
            break
    if not names:
        return [], np.zeros((0, len(conditions)))
    values = np.full((len(names), len(conditions)), np.nan)
    for j, condition in enumerate(conditions):
        stored = ((by_condition.get(condition) or {}).get("per_channel") or {}).get(field)
        for i, name in enumerate(names):
            value = (stored or {}).get(name)
            if isinstance(value, (int, float)):
                values[i, j] = float(value)
    return names, values


def build_channel_condition_matrix(by_condition: dict) -> "go.Figure | None":
    """channel x condition for one run, the panels and order of ``build_condition_panels``.

    The cohort's matrix asks which *run* moved in a condition; a subject has one run per
    page, so the same question there is which *channel* moved, and the answer is the only one
    a group page cannot give. The panel list is derived from ``_CONDITION_METRICS`` rather
    than written again, so the two cannot name different metrics under the same heading.

    Channels are ordered by how far they move across conditions on the first panel, worst
    first, and every panel keeps that order so one row reads across all of them.
    """
    conditions = list(by_condition)
    if len(conditions) < 2:
        return None

    fields = [(_CONDITION_PER_CHANNEL[metric], label)
              for metric, label in _CONDITION_METRICS if metric in _CONDITION_PER_CHANNEL]
    fields += list(_EXTRA_CHANNEL_PANELS)

    found = [(label, *_channel_matrix(by_condition, conditions, field))
             for field, label in fields]
    found = [(label, names, values) for label, names, values in found
             if names and np.isfinite(values).any()]
    if not found:
        return None

    order_names, order_values = found[0][1], found[0][2]
    with np.errstate(invalid="ignore"):
        spread = np.nanmax(
            np.abs(order_values - np.nanmedian(order_values, axis=1, keepdims=True)), axis=1)
    labels = [order_names[i] for i in np.argsort(-np.nan_to_num(spread))]

    panels = []
    for label, names, values in found:
        index = [names.index(n) for n in labels if n in names]
        panels.append((label, values[index]))
    return _condition_heatmap(panels, conditions, labels, "channel")


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
