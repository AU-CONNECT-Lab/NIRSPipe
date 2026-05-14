"""Circular connectivity (connectogram) figure.

fc_connectogram():  within-subject FC connectogram from a channel × channel DataFrame.
                    HbO and HbR are drawn as separate circles and stacked vertically.

# TODO: hyper variant — inter-brain connectogram (Sub1 left semicircle, Sub2 right semicircle).
"""

from __future__ import annotations

import base64
import io
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.figures.connectogram")

_GROUP_COLORS = [
    "#2980b9", "#c0392b", "#27ae60", "#8e44ad",
    "#d35400", "#16a085", "#f39c12", "#2c3e50",
    "#1abc9c", "#e74c3c", "#3498db", "#9b59b6",
]


def _source_groups(ch_names: list[str]) -> dict[str, str]:
    """Auto-group channels by source label (S1, S2, …) from names like 'S1-D1 hbo'."""
    groups: dict[str, str] = {}
    for ch in ch_names:
        m = re.match(r"(S\d+)", ch)
        groups[ch] = m.group(1) if m else "other"
    return groups


def _single_circle(
    fc_mat: np.ndarray,
    ch_names: list[str],
    groups: dict[str, str],
    threshold: float,
    n_lines: int | None,
    title: str,
) -> bytes:
    """Render one connectogram circle and return PNG bytes."""
    from mne.viz.circle import _plot_connectivity_circle, circular_layout

    unique_groups = list(dict.fromkeys(groups.get(c, "other") for c in ch_names))
    sorted_ch = sorted(ch_names, key=lambda c: unique_groups.index(groups.get(c, "other")))

    fc_sorted = fc_mat[
        np.ix_(
            [ch_names.index(c) for c in sorted_ch],
            [ch_names.index(c) for c in sorted_ch],
        )
    ].copy()
    np.fill_diagonal(fc_sorted, 0.0)
    if n_lines is None:
        fc_sorted[np.abs(fc_sorted) < threshold] = 0.0

    boundaries = [0]
    prev = groups.get(sorted_ch[0], "other")
    for i, ch in enumerate(sorted_ch[1:], 1):
        g = groups.get(ch, "other")
        if g != prev:
            boundaries.append(i)
            prev = g

    node_angles = circular_layout(
        sorted_ch, sorted_ch,
        start_pos=90,
        group_boundaries=boundaries,
        group_sep=8,
    )

    color_map = {g: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(unique_groups)}
    node_colors = [mcolors.to_rgba(color_map[groups.get(c, "other")]) for c in sorted_ch]
    display_names = [c.replace(" hbo", "").replace(" hbr", "") for c in sorted_ch]

    fig, ax = _plot_connectivity_circle(
        fc_sorted,
        display_names,
        node_angles=node_angles,
        node_colors=node_colors,
        n_lines=n_lines,
        colormap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        colorbar=True,
        colorbar_size=0.12,
        colorbar_pos=(-0.2, 0.1),
        title=title,
        facecolor="white",
        textcolor="#2c3e50",
        node_edgecolor="white",
        linewidth=1.5,
        fontsize_names=7,
        padding=1.0,
        show=False,
    )
    fig.set_size_inches(8, 8)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fc_connectogram(
    fc_df: pd.DataFrame,
    groups: dict[str, str] | None = None,
    threshold: float = 0.3,
    n_lines: int | None = None,
    title: str = "FC Connectogram",
) -> str:
    """Return base64 PNG with HbO (top) and HbR (bottom) connectograms stacked vertically.

    Args:
        fc_df:      Square channel × channel Pearson r DataFrame (from compute_fc).
        groups:     Optional mapping channel → group label for colour-coding nodes.
                    Defaults to grouping by source label (S1, S2, …).
        threshold:  Minimum |r| to draw a connection. Weaker edges are hidden.
        n_lines:    If set, draw only the top-N strongest connections (overrides threshold).
        title:      Base title; " — HbO" / " — HbR" is appended automatically.
    """
    from PIL import Image

    all_ch = fc_df.columns.tolist()
    hbo_ch = [c for c in all_ch if c.endswith(" hbo")]
    hbr_ch = [c for c in all_ch if c.endswith(" hbr")]

    if not hbo_ch and not hbr_ch:
        raise ValueError("fc_df has no recognised HbO/HbR channels")

    if groups is None:
        groups = _source_groups(all_ch)

    fc_mat = fc_df.to_numpy(dtype=float).copy()
    np.fill_diagonal(fc_mat, 0.0)

    pngs: list[bytes] = []
    for subset, label in ((hbo_ch, "HbO"), (hbr_ch, "HbR")):
        if not subset:
            continue
        idx = [all_ch.index(c) for c in subset]
        sub_mat = fc_mat[np.ix_(idx, idx)]
        pngs.append(_single_circle(sub_mat, subset, groups, threshold, n_lines,
                                   f"{title} — {label}"))

    if len(pngs) == 1:
        return base64.b64encode(pngs[0]).decode()

    imgs = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    h = max(im.height for im in imgs)
    combined = Image.new("RGB", (sum(im.width for im in imgs), h), (255, 255, 255))
    x = 0
    for im in imgs:
        combined.paste(im, (x, 0))
        x += im.width

    buf = io.BytesIO()
    combined.save(buf, format="png", dpi=(300, 300))
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
