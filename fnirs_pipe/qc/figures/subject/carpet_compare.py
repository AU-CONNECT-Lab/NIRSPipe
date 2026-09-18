"""Stage-by-stage haemoglobin carpet, on the motion panel's time axis and its greyscale."""

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.figures.common.motion_panel import (
    CARPET_Z, _GVTD_LINE, _LINE_MAX_PTS, _THRESH_RULE, _maxpool_xy, _px_rows,
    add_carpet, carpet_coloraxis, carpet_z,
)
from fnirs_pipe.utils import is_optical_density

_CARPET_ROW_PX = 190
_GVTD_ROW_PX = 90
_VSPACE = 0.035
_ROI_SEAM = "#ffffff"


def _stage_channels(stages, chromophore: str) -> list[str]:
    """Channels of one chromophore every stage still carries, in the first stage's order."""
    if not stages:
        return []
    first = [c for c in stages[0][1].ch_names if c.endswith(f" {chromophore}")]
    for _, raw in stages[1:]:
        have = set(raw.ch_names)
        first = [c for c in first if c in have]
    return first


def _roi_order(names: list[str], roi_map: "dict | None"):
    """Row order grouping channels by ROI, plus [(label, first row, last row), ...].

    ``roi_map`` is {ROI label: [channel names]}, matched on the full name then the S-D base.
    Unclaimed channels go last under "unassigned"; no map means montage order and no groups.
    """
    if not roi_map or not names:
        return np.arange(len(names)), []
    # index the S-D base too, so an HbO-named map also places the HbR rows
    ch_to_roi: dict = {}
    for label, chans in roi_map.items():
        for ch in chans:
            ch_to_roi[ch] = label
            ch_to_roi.setdefault(ch.rsplit(" ", 1)[0], label)
    labels = [str(ch_to_roi.get(c, ch_to_roi.get(c.rsplit(" ", 1)[0], "unassigned")))
              for c in names]
    ordered = sorted(set(labels))
    if "unassigned" in ordered:  # keep it last rather than wherever it sorts
        ordered = [l for l in ordered if l != "unassigned"] + ["unassigned"]
    code_of = {name: i for i, name in enumerate(ordered)}
    codes = np.array([code_of[l] for l in labels])
    order = np.argsort(codes, kind="stable")
    groups = []
    for i, label in enumerate(ordered):
        rows = np.where(codes[order] == i)[0]
        if rows.size:
            groups.append((label, int(rows[0]), int(rows[-1])))
    return order, groups


def _gvtd_row(raw_gvtd: mne.io.Raw):
    """(times, trace, threshold) for the motion row, or None with nothing to draw."""
    from fnirs_pipe.qc.metrics import (
        GVTD_MOTION_BAND, GVTD_N_STD, gvtd_threshold, gvtd_timetrace,
    )

    od = raw_gvtd if is_optical_density(raw_gvtd) else \
        mne.preprocessing.nirs.optical_density(raw_gvtd.copy(), verbose=False)
    data = od.get_data()
    if not data.size:
        return None
    g = gvtd_timetrace(data, float(od.info["sfreq"]), *GVTD_MOTION_BAND)
    return od.times[:len(g)], g, gvtd_threshold(g, n_std=GVTD_N_STD)


def _cut(t: np.ndarray, xlim) -> slice:
    """Columns inside ``xlim``, or everything when there is no span."""
    if xlim is None:
        return slice(None)
    i0 = int(np.searchsorted(t, float(xlim[0])))
    i1 = int(np.searchsorted(t, float(xlim[1])))
    return slice(i0, max(i1, i0 + 1))


def carpet_compare_figure(
    stages: "list[tuple[str, mne.io.Raw]]",
    chromophore: str = "hbo",
    roi_map: "dict | None" = None,
    raw_gvtd: "mne.io.Raw | None" = None,
    xlim: "tuple[float, float] | None" = None,
    z_threshold: float = CARPET_Z,
) -> "go.Figure | None":
    """One carpet per stage under a shared GVTD row, all on one time axis.

    ``stages`` is ``[(label, raw), ...]`` in pipeline order. Every carpet is z-scored by the
    first stage's per-channel mean and SD, and each later block's title carries the median of
    its own SD against it. ``raw_gvtd`` is intensity or optical density; omitting it drops the
    motion row. ``xlim`` (t0, t1) cuts the columns drawn, not the z-scoring, which stays
    whole-run so a condition's carpet is on the run's greyscale.

    Returns None when no channel of ``chromophore`` survives in every stage.
    """
    names = _stage_channels(stages, chromophore)
    if not names:
        return None
    order, groups = _roi_order(names, roi_map)
    ordered_names = [names[i] for i in order]

    data = [raw.get_data(picks=ordered_names) for _, raw in stages]
    ref_sd = data[0].std(axis=1)
    ref_sd[ref_sd == 0] = 1.0
    titles = [f"{label}  ·  SD {np.median(d.std(axis=1) / ref_sd):.2f}× {stages[0][0]}"
              if i else label
              for i, (label, d) in enumerate(zip((s[0] for s in stages), data))]

    gvtd = _gvtd_row(raw_gvtd) if raw_gvtd is not None else None
    n_rows = len(stages) + (1 if gvtd else 0)
    heights = ([_GVTD_ROW_PX] if gvtd else []) + [_CARPET_ROW_PX] * len(stages)
    row_heights, total_px = _px_rows(heights, _VSPACE, chrome_px=140)
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=_VSPACE,
        subplot_titles=([""] if gvtd else []) + titles,
    )
    for ann in fig.layout.annotations:
        ann.yshift = 3

    if gvtd:
        t_g, g, thresh = gvtd
        keep = _cut(t_g, xlim)
        t_ds, g_ds = _maxpool_xy(t_g[keep], g[keep], _LINE_MAX_PTS)
        fig.add_trace(go.Scatter(
            x=t_ds, y=g_ds, mode="lines", name="GVTD",
            line=dict(color=_GVTD_LINE, width=1.5),
            hovertemplate="t=%{x:.1f}s<br>GVTD=%{y:.2e}<extra></extra>",
        ), row=1, col=1)
        if thresh is not None:
            fig.add_hline(y=thresh, line=dict(color=_THRESH_RULE, width=1.2, dash="dash"),
                          row=1, col=1)
        fig.update_yaxes(title_text="GVTD", title_font_size=10, row=1, col=1)

    first_row = 2 if gvtd else 1
    stats = None
    for i, ((_, raw), d) in enumerate(zip(stages, data)):
        z, t_ds, first_stats = carpet_z(d, raw.times, z_threshold, stats=stats)
        if stats is None:
            stats = first_stats
        keep = _cut(t_ds, xlim)
        add_carpet(fig, first_row + i, z[:, keep], t_ds[keep], ordered_names, [])
        _roi_marks(fig, first_row + i, groups, len(ordered_names))

    fig.update_layout(
        coloraxis=carpet_coloraxis(z_threshold, y=0.42),
        height=total_px, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=90 if groups else 60, r=30, t=60, b=50),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom",
                    font=dict(size=11)),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    return fig


def _roi_marks(fig, row: int, groups: list, n_ch: int) -> None:
    """ROI names down the left edge of one carpet, with a white seam between groups."""
    if len(groups) < 2:
        return
    fig.update_yaxes(
        showticklabels=True, tickmode="array",
        tickvals=[(i0 + i1) / 2 for _, i0, i1 in groups],
        ticktext=[label for label, _, _ in groups],
        tickfont=dict(size=9), ticks="", row=row, col=1,
    )
    for _, i0, _ in groups[1:]:
        edge = 1.0 - i0 / n_ch
        fig.add_shape(type="line", xref="x domain", yref="y domain",
                      x0=0, x1=1, y0=edge, y1=edge,
                      line=dict(color=_ROI_SEAM, width=2), row=row, col=1)
