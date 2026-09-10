"""2-D optode flat map: source / detector positions coloured by SCI."""

import re

import mne
import numpy as np
import plotly.graph_objects as go

from fnirs_pipe.qc.metrics import SCI_PASS

from ._utils import head_outline
from .raw_figures import sci_color, sci_legend


def optode_layout_static(
    raw: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    sci_threshold: float = SCI_PASS,
) -> str | None:
    """Matplotlib static optode flat map. Returns base64 PNG or None.

    ``bad_channels`` was accepted and ignored until 2026-09-09, which made this an SCI map
    rather than a quality map: a channel rejected on its coupled-window share was drawn
    green beside a table calling it BAD. A rejected pair is red now whatever its SCI.
    """
    import base64
    import io
    import matplotlib.pyplot as plt

    picks = mne.pick_types(raw.info, fnirs=True)
    if not len(picks):
        return None

    _ch_re = re.compile(r"(S\d+)[_\s]+(D\d+)", re.IGNORECASE)
    pair_sci: dict[str, list[float]] = {}
    pair_xy: dict[str, tuple] = {}

    for idx in picks:
        ch = raw.info["chs"][idx]
        name = raw.info["ch_names"][idx]
        loc = ch["loc"]
        src_xyz, det_xyz = loc[3:6], loc[6:9]
        if np.any(np.isnan(src_xyz)) or np.any(np.isnan(det_xyz)):
            continue
        if np.allclose(src_xyz, 0) and np.allclose(det_xyz, 0):
            continue
        m = _ch_re.search(name)
        if m is None:
            continue
        pair_id = f"{m.group(1).upper()}_{m.group(2).upper()}"
        sci = sci_scores.get(name)
        if sci is None:
            base = name.split(" ")[0]
            sci = sci_scores.get(base + " hbo") or sci_scores.get(base + " hbr")
        if sci is not None:
            pair_sci.setdefault(pair_id, []).append(sci)
        pair_xy[pair_id] = (
            (float(src_xyz[0]), float(src_xyz[1])),
            (float(det_xyz[0]), float(det_xyz[1])),
        )

    if not pair_xy:
        return None

    # matched on the source-detector pair, since screening names intensity channels and
    # this figure draws one line per pair
    bad_pairs = set()
    for name in bad_channels or []:
        m = _ch_re.search(str(name))
        if m is not None:
            bad_pairs.add(f"{m.group(1).upper()}_{m.group(2).upper()}")

    def _color(sci, rejected=False):
        if rejected:
            return sci_color(None, sci_threshold, rejected=True)
        return "#95a5a6" if sci is None else sci_color(sci, sci_threshold)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    for pair_id, (s, d) in pair_xy.items():
        vals = pair_sci.get(pair_id)
        c = _color(float(np.mean(vals)) if vals else None, pair_id in bad_pairs)
        ax.plot([s[0], d[0]], [s[1], d[1]], color=c, lw=2.0, zorder=1)

    sources: dict = {}
    detectors: dict = {}
    for pair_id, (s, d) in pair_xy.items():
        sources[pair_id.split("_")[0]] = s
        detectors[pair_id.split("_")[1]] = d

    all_xs = [v[0] for v in list(sources.values()) + list(detectors.values())]
    all_ys = [v[1] for v in list(sources.values()) + list(detectors.values())]
    head_outline(ax, all_xs, all_ys)

    if sources:
        sx_v, sy_v = zip(*sources.values())
        ax.scatter(sx_v, sy_v, s=55, c="#e74c3c", edgecolors="#922b21", lw=1.2, zorder=3, label="Source")
        for sid, (x, y) in sources.items():
            ax.text(x, y, sid, fontsize=6, ha="center", va="bottom", color="#922b21", zorder=4)
    if detectors:
        dx_v, dy_v = zip(*detectors.values())
        ax.scatter(dx_v, dy_v, s=45, c="#2980b9", edgecolors="#1a5276", lw=1.2, zorder=3, label="Detector")
        for did, (x, y) in detectors.items():
            ax.text(x, y, did, fontsize=6, ha="center", va="top", color="#1a5276", zorder=4)

    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.set_title(f"Optode flat map — {sci_legend(sci_threshold)}", fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def optode_layout_figure(
    raw: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    sci_threshold: float = SCI_PASS,
) -> go.Figure | None:
    """Top-down 2-D projection of sources (red) and detectors (blue).

    A rejected channel's line is red; the rest are coloured green→yellow→red by the mean SCI
    of that source-detector pair. Useful for diagnosing whether a bad region is due
    to a single bad source or detector rather than an individual channel.

    Returns None if no valid channel positions are found.
    """
    picks = mne.pick_types(raw.info, fnirs=True)
    if not len(picks):
        return None

    # group channels by (src_id, det_id) pair parsed from channel name
    _ch_re = re.compile(r"(S\d+)[_\s]+(D\d+)", re.IGNORECASE)

    pair_sci: dict[str, list[float]] = {}
    pair_xy: dict[str, tuple[tuple, tuple]] = {}

    for idx in picks:
        ch = raw.info["chs"][idx]
        name = raw.info["ch_names"][idx]
        loc = ch["loc"]
        src_xyz = loc[3:6]
        det_xyz = loc[6:9]

        if np.any(np.isnan(src_xyz)) or np.any(np.isnan(det_xyz)):
            continue
        if np.allclose(src_xyz, 0) and np.allclose(det_xyz, 0):
            continue

        m = _ch_re.search(name)
        if m is None:
            continue
        pair_id = f"{m.group(1).upper()}_{m.group(2).upper()}"

        sci = sci_scores.get(name)
        if sci is None:
            # try matching without chromophore suffix
            base = name.split(" ")[0]
            sci = sci_scores.get(base + " hbo") or sci_scores.get(base + " hbr")
        if sci is not None:
            pair_sci.setdefault(pair_id, []).append(sci)

        pair_xy[pair_id] = (
            (float(src_xyz[0]), float(src_xyz[1])),
            (float(det_xyz[0]), float(det_xyz[1])),
        )

    if not pair_xy:
        return None

    bad_pairs = set()
    for name in bad_channels or []:
        m = _ch_re.search(str(name))
        if m is not None:
            bad_pairs.add(f"{m.group(1).upper()}_{m.group(2).upper()}")

    def _sci_color(mean_sci: float | None, rejected: bool = False) -> str:
        if rejected:
            return sci_color(None, sci_threshold, rejected=True)
        return "#95a5a6" if mean_sci is None else sci_color(mean_sci, sci_threshold)

    fig = go.Figure()

    # channel lines
    for pair_id, (src_xy, det_xy) in pair_xy.items():
        sci_vals = pair_sci.get(pair_id)
        mean_sci = float(np.mean(sci_vals)) if sci_vals else None
        color = _sci_color(mean_sci, pair_id in bad_pairs)
        sci_str = f"{mean_sci:.3f}" if mean_sci is not None else "N/A"
        fig.add_trace(go.Scatter(
            x=[src_xy[0], det_xy[0]],
            y=[src_xy[1], det_xy[1]],
            mode="lines",
            line=dict(width=2.5, color=color),
            name=pair_id,
            showlegend=False,
            hovertemplate=f"{pair_id}<br>SCI = {sci_str}<extra></extra>",
        ))

    # collect unique sources and detectors from pair positions
    sources: dict[str, tuple] = {}
    detectors: dict[str, tuple] = {}
    for pair_id, (src_xy, det_xy) in pair_xy.items():
        src_id = pair_id.split("_")[0]
        det_id = pair_id.split("_")[1]
        sources[src_id] = src_xy
        detectors[det_id] = det_xy

    if sources:
        sx, sy, slabels = zip(*[(v[0], v[1], k) for k, v in sorted(sources.items())])
        fig.add_trace(go.Scatter(
            x=list(sx), y=list(sy),
            mode="markers+text",
            marker=dict(size=13, color="#e74c3c",
                        line=dict(width=1.5, color="#922b21")),
            text=list(slabels),
            textposition="top center",
            textfont=dict(size=8),
            name="Source",
            hovertemplate="%{text}<extra>Source</extra>",
        ))

    if detectors:
        dx, dy, dlabels = zip(*[(v[0], v[1], k) for k, v in sorted(detectors.items())])
        fig.add_trace(go.Scatter(
            x=list(dx), y=list(dy),
            mode="markers+text",
            marker=dict(size=11, color="#2980b9",
                        line=dict(width=1.5, color="#1a5276")),
            text=list(dlabels),
            textposition="bottom center",
            textfont=dict(size=8),
            name="Detector",
            hovertemplate="%{text}<extra>Detector</extra>",
        ))

    # head outline: circle + nose + ears derived from optode bounding box
    all_xs = [xy[0] for xy in sources.values()] + [xy[0] for xy in detectors.values()]
    all_ys = [xy[1] for xy in sources.values()] + [xy[1] for xy in detectors.values()]
    if all_xs and all_ys:
        cx = (max(all_xs) + min(all_xs)) / 2
        cy = (max(all_ys) + min(all_ys)) / 2
        r = max(
            max(abs(x - cx) for x in all_xs),
            max(abs(y - cy) for y in all_ys),
        ) * 1.18

        fig.add_shape(
            type="circle",
            x0=cx - r, y0=cy - r, x1=cx + r, y1=cy + r,
            line=dict(color="#aaa", width=1.5),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )

        # nose: small triangle at top (+y = anterior)
        nw, nh = r * 0.06, r * 0.10
        fig.add_trace(go.Scatter(
            x=[cx - nw, cx, cx + nw, cx - nw],
            y=[cy + r - nh * 0.3, cy + r + nh, cy + r - nh * 0.3, cy + r - nh * 0.3],
            mode="lines", fill="toself",
            line=dict(color="#aaa", width=1),
            fillcolor="#ddd",
            showlegend=False, hoverinfo="skip",
        ))

        # ears: left (-x) and right (+x)
        ew, eh = r * 0.07, r * 0.12
        for ex0, ex1 in ((cx - r - ew, cx - r + ew * 0.3),
                         (cx + r - ew * 0.3, cx + r + ew)):
            fig.add_shape(
                type="circle",
                x0=ex0, y0=cy - eh, x1=ex1, y1=cy + eh,
                line=dict(color="#aaa", width=1),
                fillcolor="#ddd",
                layer="below",
            )

    fig.update_layout(
        title=f"Optode flat map — channel {sci_legend(sci_threshold)}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   constrain="domain"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(font=dict(size=10)),
    )
    return fig
