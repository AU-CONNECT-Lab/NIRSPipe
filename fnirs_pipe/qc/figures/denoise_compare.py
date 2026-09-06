"""What each pipeline step did: one small panel per metric, plus the per-channel detail.

Small multiples rather than one shared axis, because the metrics here run on unrelated
scales and a shared axis would either flatten the correlations or hide the band powers.
Each panel carries its own range and prints its values, so no comparison between panels is
implied.

Direction is marked only on the whole chain, never on an intermediate step: a step can move
a metric for reasons that are not quality, and colouring that step would print the wrong
conclusion on the figure. A panel with ``lower_better`` unset gets no direction at all, for
rows where movement is not itself good or bad.

``stage_metrics_figure`` renders whatever panels it is handed, so the same figure serves
both the haemoglobin chain and the optical-density one; the ``*_stage_panels`` builders
below turn each chain's numbers into panels.
"""

from typing import NamedTuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK, SUBTLE, RULE = "#1f2933", "#8b95a1", "#e3e8ee"
WORSE, BETTER, FLAT = "#b0413e", "#3f7d5a", "#c9d1d9"


class Panel(NamedTuple):
    """One mini panel: a title, one value per stage, and how to read it."""
    title: str
    values: list
    lower_better: "bool | None" = None   # None: movement carries no verdict
    primary: bool = True                 # False greys the title
    unit: str = ""


# key, panel title, whether a smaller value is the better one
_DENOISE_QUALITY = [
    ("hbo_hbr_corr_mean", "HbO–HbR r", True),
    ("gcor_hbo", "GCOR HbO", True),
    ("gcor_hbr", "GCOR HbR", True),
    ("resp_band_power_hbo", "resp power HbO, in band", True),
    ("cnr_hbo_mean", "CNR HbO", False),
    ("cnr_hbr_mean", "CNR HbR", True),
]
_DENOISE_FILTER_CHECK = [
    ("drift_band_power_hbo", "drift power HbO"),
    ("cardiac_band_power_hbo", "cardiac power HbO"),
]

_MOTION_COUPLING = [
    ("sci_mean", "SCI"),
    ("psp_mean", "PSP"),
]
# key, label, whether a smaller value is the better one, display scale. The last two are
# stored as fractions despite the pct in their names, so they are scaled here to be read
# as the percentages their labels promise.
_MOTION_ARTIFACT = [
    ("gvtd_filt_mean", "GVTD, motion band", True, 1.0),
    ("gvtd_filt_p95", "GVTD p95, motion band", True, 1.0),
    ("gvtd_pct_above_thresh", "% of run above GVTD threshold", True, 100.0),
    ("spike_pct", "% of samples spiking", True, 100.0),
]


def _fmt(v, unit: str = "") -> str:
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    if unit == "%":
        return f"{v:.1f}%"
    return f"{v:.3f}" if abs(v) >= 1e-3 else f"{v:.1e}"


def denoise_stage_panels(metrics: dict) -> "list[Panel]":
    """Panels for the haemoglobin chain, from :func:`comparable_stage_metrics` output.

    Ordered quality first, then the share of variance still present, then the two band
    powers that only say whether the filter ran. Those last are greyed: sitting entirely
    outside the analysis passband, they fall by the filter's stopband attenuation whatever
    the data did, so their one use is catching a filter that was not configured as asked.
    """
    panels = [Panel(title, metrics["quality"][key], lower_better)
              for key, title, lower_better in _DENOISE_QUALITY
              if key in metrics["quality"]]
    remaining = metrics.get("variance_remaining") or []
    if any(isinstance(v, (int, float)) for v in remaining):
        panels.append(Panel("variance remaining",
                            [None if v is None else v * 100 for v in remaining],
                            None, True, "%"))
    panels += [Panel(title, metrics["removed"][key], None, False)
               for key, title in _DENOISE_FILTER_CHECK if key in metrics["removed"]]
    return panels


def motion_stage_panels(record: dict, sections: "list[tuple[str, str]]") -> "list[Panel]":
    """Panels for the optical-density chain, read from the record.

    ``sections`` is ``[(stage label, record section), ...]``. Nothing is recomputed here:
    both stages are unfiltered optical density, so unlike the haemoglobin chain their
    stored numbers already sit in the same band and may be compared as they are.

    SCI and PSP get no direction. They measure optode coupling, which the correction is not
    supposed to change, so the reading is whether they came back where they started, and a
    verdict on a small movement would be noise dressed as a finding.
    """
    def series(key: str) -> list:
        return [(record.get(section) or {}).get(key) for _, section in sections]

    panels = [Panel(label, series(key), None, False)
              for key, label in _MOTION_COUPLING
              if any(isinstance(v, (int, float)) for v in series(key))]
    for key, label, lower_better, scale in _MOTION_ARTIFACT:
        values = series(key)
        if not any(isinstance(v, (int, float)) for v in values):
            continue
        scaled = [None if not isinstance(v, (int, float)) else v * scale for v in values]
        panels.append(Panel(label, scaled, lower_better, True,
                            "%" if scale != 1.0 else ""))
    return panels


def stage_metrics_figure(
    labels: list[str],
    panels: "list[Panel]",
    n_cols: int = 3,
) -> "go.Figure | None":
    """One mini panel per metric across ``labels``, values printed on the points."""
    if len(labels) < 2 or not panels:
        return None

    n_rows = -(-len(panels) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=[p.title for p in panels],
                        vertical_spacing=0.24 if n_rows > 1 else 0.1,
                        horizontal_spacing=0.08)

    for i, panel in enumerate(panels):
        row, col = divmod(i, n_cols)
        values = panel.values
        good = None
        if panel.lower_better is not None and all(isinstance(v, (int, float)) for v in values):
            good = (values[-1] < values[0]) if panel.lower_better else (values[-1] > values[0])
        color = INK if good is None else (BETTER if good else WORSE)
        fig.add_trace(go.Scatter(
            x=labels, y=values, mode="lines+markers+text",
            text=[_fmt(v, panel.unit) for v in values],
            textposition="top center", textfont=dict(size=10, color=SUBTLE),
            line=dict(color=color, width=2), marker=dict(size=7, color=color),
            showlegend=False,
            hovertemplate=f"{panel.title}<br>%{{x}}: %{{y:.4g}}<extra></extra>"),
            row=row + 1, col=col + 1)
        present = [v for v in values if isinstance(v, (int, float))]
        if present:
            lo, hi = min(present), max(present)
            # headroom for the printed values, and a usable range when the line is flat
            pad = (hi - lo) * 0.45 or abs(hi) * 0.3 or 1.0
            fig.update_yaxes(range=[lo - pad, hi + pad * 1.6], row=row + 1, col=col + 1)
        if not panel.primary:
            fig.layout.annotations[i].update(font=dict(color=SUBTLE))

    fig.update_xaxes(tickfont=dict(size=9, color=SUBTLE), showgrid=False, linecolor=RULE,
                     range=[-0.35, len(labels) - 0.65])
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     linecolor="white")
    for annotation in fig.layout.annotations:
        annotation.update(font=dict(size=11, color=annotation.font.color or INK))
    fig.update_layout(
        height=190 * n_rows + 30, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=30, r=30, t=40, b=25))
    return fig


def stage_slope_figure(
    per_stage: "list[dict[str, float]]",
    labels: list[str],
    *,
    title: str,
    y_title: str,
    higher_is_better: bool = True,
    flat_tol: float = 0.05,
) -> "go.Figure | None":
    """One line per channel across the stages, with the mean drawn over the top.

    ``per_stage`` is one ``{channel: value}`` per stage, aligned with ``labels``; only
    channels present at every stage are drawn. Channels the chain left worse are drawn
    heavy so they can be picked out of the bundle.
    """
    if len(per_stage) < 2 or not per_stage[0]:
        return None
    chans = [ch for ch in per_stage[0] if all(ch in d for d in per_stage)]
    if not chans:
        return None
    values = np.array([[d[ch] for d in per_stage] for ch in chans], dtype=float)
    net = values[:, -1] - values[:, 0]

    fig = go.Figure()
    for ch, row, delta in zip(chans, values, net):
        improved = delta > 0 if higher_is_better else delta < 0
        worse = abs(delta) > flat_tol and not improved
        fig.add_trace(go.Scatter(
            x=labels, y=row, mode="lines+markers", name=ch,
            line=dict(color=WORSE if worse else (BETTER if abs(delta) > flat_tol else FLAT),
                      width=2.4 if worse else 1.4),
            marker=dict(size=5), showlegend=False,
            hovertemplate=f"{ch}<br>%{{x}}: %{{y:.3f}}<extra></extra>"))
    means = values.mean(axis=0)
    fig.add_trace(go.Scatter(
        x=labels, y=means, mode="lines+markers+text",
        text=[f"{v:.3f}" for v in means], textposition="bottom center",
        textfont=dict(size=10, color=INK),
        line=dict(color=INK, width=3), marker=dict(size=8),
        showlegend=False, hovertemplate="mean %{x}: %{y:.3f}<extra></extra>"))

    n_worse = int(sum(abs(d) > flat_tol and not (d > 0 if higher_is_better else d < 0)
                      for d in net))
    fig.update_layout(
        title=dict(text=f"{title}<br><span style='font-size:11px;color:#666'>{n_worse} of "
                        f"{len(chans)} channels ended worse than they started, drawn "
                        "heavy</span>", font=dict(size=15, color=INK)),
        xaxis=dict(range=[-0.2, len(labels) - 0.8], showgrid=False, linecolor=RULE),
        yaxis=dict(title=y_title, gridcolor="#f2f4f7", zeroline=False),
        height=420, margin=dict(l=70, r=40, t=70, b=40),
        plot_bgcolor="white", paper_bgcolor="white")
    return fig
