"""Stage-by-stage haemoglobin carpet, on the motion panel's time axis and its greyscale."""

import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fnirs_pipe.qc.figures.common._utils import HBO_COLOR, HBR_COLOR
from fnirs_pipe.qc.figures.common.motion_panel import (
    CARPET_Z, _GVTD_LINE, _LINE_MAX_PTS, _SEAM, _THRESH_RULE, _maxpool_xy, _px_rows,
    carpet_coloraxis, carpet_z,
)
from fnirs_pipe.utils import is_optical_density

_CARPET_ROW_PX = 190
_CHROMO_COLOR = {"hbo": HBO_COLOR, "hbr": HBR_COLOR}
_STAGE_LINE = ("#b0b7bf", "#2f4858")   # the stage before denoising, then the one drawn
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


def _haemo_gvtd(stages, reference, chromos):
    """[(label, times, trace), ...] of the channel-equalised GVTD of each haemo stage.

    Standardised per channel first, so HbO and HbR contribute equally instead of the trace
    following whichever has the larger units.
    """
    from fnirs_pipe.qc.metrics import GVTD_MOTION_BAND, gvtd_timetrace

    out = []
    for label, raw in ([reference] if reference else []) + list(stages):
        picks = [c for c in raw.ch_names if c.rsplit(" ", 1)[-1] in chromos]
        if not picks:
            continue
        g = gvtd_timetrace(raw.get_data(picks=picks), float(raw.info["sfreq"]),
                           *GVTD_MOTION_BAND, standardize_channels=True)
        out.append((label, raw.times[:len(g)], g))
    return out if len(out) > 1 else []


def _cut(t: np.ndarray, xlim) -> slice:
    """Columns inside ``xlim``, or everything when there is no span."""
    if xlim is None:
        return slice(None)
    i0 = int(np.searchsorted(t, float(xlim[0])))
    i1 = int(np.searchsorted(t, float(xlim[1])))
    return slice(i0, max(i1, i0 + 1))


def carpet_compare_figure(
    stages: "list[tuple[str, mne.io.Raw]]",
    chromophore: "str | tuple[str, ...]" = ("hbo", "hbr"),
    roi_map: "dict | None" = None,
    raw_gvtd: "mne.io.Raw | None" = None,
    xlim: "tuple[float, float] | None" = None,
    z_threshold: float = CARPET_Z,
    reference: "tuple[str, mne.io.Raw] | None" = None,
) -> "go.Figure | None":
    """A carpet per stage in ``stages``, under a shared GVTD row on one time axis.

    Each carpet is z-scored by its own per-channel mean and SD. ``reference`` is a stage that
    is not drawn and only supplies the SD each title is quoted against. ``raw_gvtd`` is
    intensity or optical density; omitting it drops the motion row. ``xlim`` (t0, t1) cuts the
    columns drawn, not the z-scoring, which stays whole-run so a condition's carpet is on the
    run's greyscale.

    ``chromophore`` may be one name or several; each gets its own block. Returns None when
    no channel of any of them survives in every stage.
    """
    chromos = (chromophore,) if isinstance(chromophore, str) else tuple(chromophore)
    ref_label, ref_raw = reference if reference else stages[0]

    rows = []   # one drawn row per stage: its title and the chromophore blocks stacked in it
    for label, raw in stages:
        blocks, ratios = [], []
        for chromo in chromos:
            names = _stage_channels(list(stages) + ([reference] if reference else []), chromo)
            if not names:
                continue
            order, groups = _roi_order(names, roi_map)
            ordered = [names[i] for i in order]
            d = raw.get_data(picks=ordered)
            ref_sd = ref_raw.get_data(picks=ordered).std(axis=1)
            ref_sd[ref_sd == 0] = 1.0
            blocks.append((chromo, ordered, d, groups))
            ratios.append((chromo, float(np.median(d.std(axis=1) / ref_sd))))
        if not blocks:
            continue
        ratio_text = "   ".join(f"{c.upper()} SD {r:.2f}× {ref_label}" for c, r in ratios)
        first = not reference and label == stages[0][0]
        rows.append((label if first else f"{label}  ·  {ratio_text}", raw, blocks))
    if not rows:
        return None

    stage_gvtd = _haemo_gvtd(stages, reference, chromos)
    gvtd = _gvtd_row(raw_gvtd) if raw_gvtd is not None else None
    n_rows = len(rows) + (1 if gvtd else 0) + (1 if stage_gvtd else 0)
    block_px = _CARPET_ROW_PX * max(len(r[2]) for r in rows)
    heights = ([_GVTD_ROW_PX] if stage_gvtd else []) + ([_GVTD_ROW_PX] if gvtd else [])         + [block_px] * len(rows)
    row_heights, total_px = _px_rows(heights, _VSPACE, chrome_px=140)
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights, vertical_spacing=_VSPACE,
        subplot_titles=([""] * ((1 if stage_gvtd else 0) + (1 if gvtd else 0))
                        + [r[0] for r in rows]),
    )
    for ann in fig.layout.annotations:
        ann.yshift = 3

    if stage_gvtd:
        for k, (label, t_s, g_s) in enumerate(stage_gvtd):
            keep = _cut(t_s, xlim)
            t_ds, g_ds = _maxpool_xy(t_s[keep], g_s[keep], _LINE_MAX_PTS)
            fig.add_trace(go.Scatter(
                x=t_ds, y=g_ds, mode="lines", name=label,
                line=dict(color=_STAGE_LINE[min(k, len(_STAGE_LINE) - 1)],
                          width=1.3 if k else 1.1),
                opacity=1.0 if k else 0.75,
                hovertemplate=f"{label}<br>t=%{{x:.1f}}s<br>%{{y:.2f}}<extra></extra>",
            ), row=1, col=1)
        fig.update_yaxes(title_text="stage GVTD", title_font_size=10, row=1, col=1)

    if gvtd:
        gvtd_row = 2 if stage_gvtd else 1
        t_g, g, thresh = gvtd
        keep = _cut(t_g, xlim)
        t_ds, g_ds = _maxpool_xy(t_g[keep], g[keep], _LINE_MAX_PTS)
        fig.add_trace(go.Scatter(
            x=t_ds, y=g_ds, mode="lines", name="GVTD",
            line=dict(color=_GVTD_LINE, width=1.5),
            hovertemplate="t=%{x:.1f}s<br>GVTD=%{y:.2e}<extra></extra>",
        ), row=gvtd_row, col=1)
        if thresh is not None:
            fig.add_hline(y=thresh, line=dict(color=_THRESH_RULE, width=1.2, dash="dash"),
                          row=gvtd_row, col=1)
        fig.update_yaxes(title_text="motion GVTD", title_font_size=10, row=gvtd_row, col=1)

    first_row = 1 + (1 if stage_gvtd else 0) + (1 if gvtd else 0)
    has_roi = False
    for i, (_, raw, blocks) in enumerate(rows):
        row = first_row + i
        zs, labels, spans, roi_ticks = [], [], [], []
        for chromo, ordered, d, groups in blocks:
            z, t_ds, _ = carpet_z(d, raw.times, z_threshold)
            spans.append((chromo, len(labels), len(labels) + len(ordered) - 1))
            roi_ticks += [(label, len(labels) + (i0 + i1) / 2) for label, i0, i1 in groups]                 if len(groups) > 1 else []
            labels += ordered
            zs.append(z)
        keep = _cut(t_ds, xlim)
        z = np.vstack(zs)[:, keep]
        fig.add_trace(go.Heatmap(
            z=z, x=t_ds[keep], y=labels, coloraxis="coloraxis",
            hovertemplate="%{y}<br>t=%{x:.1f}s<br>z=%{z:.2f}<extra></extra>",
            showlegend=False,
        ), row=row, col=1)
        fig.update_yaxes(showticklabels=False, autorange="reversed", row=row, col=1)
        _chromo_marks(fig, row, spans, len(labels))
        if roi_ticks:
            has_roi = True
            fig.update_yaxes(showticklabels=True, tickmode="array",
                             tickvals=[v for _, v in roi_ticks],
                             ticktext=[l for l, _ in roi_ticks],
                             tickfont=dict(size=9), ticks="", row=row, col=1)

    fig.update_layout(
        coloraxis=carpet_coloraxis(z_threshold, y=0.42),
        height=total_px, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=90 if has_roi else 60, r=30, t=60, b=50),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.02, yanchor="bottom",
                    font=dict(size=11)),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    return fig


def _chromo_marks(fig, row: int, spans: list, n_ch: int) -> None:
    """A colour bar per chromophore down the left edge, and a white seam between them."""
    if len(spans) < 2:
        return
    last = len(spans) - 1
    for k, (name, i0, i1) in enumerate(spans):
        y0 = 1.0 - (i1 + 1) / n_ch + (_SEAM / 2 if k < last else 0.0)
        y1 = 1.0 - i0 / n_ch - (_SEAM / 2 if k else 0.0)
        fig.add_shape(type="rect", xref="x domain", yref="y domain",
                      x0=-0.012, x1=-0.004, y0=y0, y1=y1,
                      fillcolor=_CHROMO_COLOR.get(name, _GVTD_LINE), line=dict(width=0),
                      row=row, col=1)
        if k:
            edge = y1 + _SEAM / 2
            fig.add_shape(type="line", xref="x domain", yref="y domain",
                          x0=0, x1=1, y0=edge, y1=edge,
                          line=dict(color=_ROI_SEAM, width=2), row=row, col=1)
