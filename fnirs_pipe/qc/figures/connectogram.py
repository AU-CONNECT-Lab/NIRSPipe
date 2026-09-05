"""Circular connectivity (connectogram) figure.

::

  fc_connectogram():    within-subject FC connectogram from a channel × channel DataFrame.
                        HbO and HbR are drawn as separate circles and stacked vertically.
  isc_connectogram():   inter-brain connectogram, Sub1 on left semicircle, Sub2 on right.
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
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def fc_connectogram(
    fc_df: pd.DataFrame,
    fc_hbr_df: pd.DataFrame | None = None,
    groups: dict[str, str] | None = None,
    threshold: float = 0.3,
    n_lines: int | None = None,
    title: str = "FC Connectogram",
) -> str:
    """Return base64 PNG with one connectogram per chromophore present, side by side.

    Args:
        fc_df:      Square channel × channel Pearson r DataFrame (from compute_fc).
                    compute_fc returns one matrix per chromophore, so this is the HbO one;
                    a single matrix holding both is also accepted and split by suffix.
        fc_hbr_df:  The HbR matrix, when the two are supplied separately.
        groups:     Optional mapping channel → group label for colour-coding nodes.
                    Defaults to grouping by source label (S1, S2, …).
                    Rejected channels are left off the circle: compute_fc blanks their row,
                    so they carry no edge.
        threshold:  Minimum ``|r|`` to draw a connection. Weaker edges are hidden.
        n_lines:    If set, draw only the top-N strongest connections (overrides threshold).
        title:      Base title; " — HbO" / " — HbR" is appended automatically.
    """
    from PIL import Image

    subsets: list[tuple[list[str], np.ndarray, str]] = []
    for frame in (fc_df, fc_hbr_df):
        if frame is None or frame.empty:
            continue
        all_ch = frame.columns.tolist()
        fc_mat = frame.to_numpy(dtype=float).copy()
        # a rejected channel is an all-NaN row: it has no edge to draw, and a node kept for
        # it would sit on the circle claiming the ring has one more channel than it measured
        drawable = ~np.all(np.isnan(fc_mat), axis=1)
        np.fill_diagonal(fc_mat, 0.0)
        for suffix, label in ((" hbo", "HbO"), (" hbr", "HbR")):
            names = [c for i, c in enumerate(all_ch) if c.endswith(suffix) and drawable[i]]
            if not names:
                continue
            idx = [all_ch.index(c) for c in names]
            subsets.append((names, fc_mat[np.ix_(idx, idx)], label))

    if not subsets:
        raise ValueError("fc_df has no recognised HbO/HbR channels")

    if groups is None:
        groups = _source_groups([c for names, _, _ in subsets for c in names])

    pngs: list[bytes] = []
    for names, sub_mat, label in subsets:
        pngs.append(_single_circle(sub_mat, names, groups, threshold, n_lines,
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


def isc_connectogram(
    isc_mat: np.ndarray,
    ch_names: list[str],
    subject_ids: tuple[str, str],
    threshold: float = 0.3,
    n_lines: int | None = None,
    diagonal_only: bool = False,
    title: str = "Inter-brain ISC Connectogram",
) -> str:
    """Return base64 PNG of inter-brain connectogram.

    Sub1 channels occupy the left semicircle; Sub2 the right.

    Args:
        isc_mat:       n × n ISC matrix from compute_isc().
        ch_names:      Channel labels (n,), without HbO/HbR type suffix.
        subject_ids:   (sub1_id, sub2_id) used as group colour labels.
        threshold:     Minimum ``|ISC|`` to draw an arc (ignored when n_lines set).
        n_lines:       Draw only the top-N strongest arcs.
        diagonal_only: If True (default), draw only same-channel arcs
                       (sub1_ch_i ↔ sub2_ch_i). Keeps the plot readable.
        title:         Figure title.
    """
    from mne.viz.circle import _plot_connectivity_circle, circular_layout

    n = len(ch_names)
    if n == 0:
        raise ValueError("ch_names is empty")

    sub1_id, sub2_id = subject_ids[0], subject_ids[1]

    # internal unique names for layout; display names are shown in the figure
    uniq_sub1 = [f"{c}__1" for c in ch_names]
    uniq_sub2 = [f"{c}__2" for c in ch_names]
    all_uniq  = uniq_sub1 + uniq_sub2
    display_names = list(ch_names) + list(ch_names)

    # 2n × 2n connectivity: only cross-brain arcs filled
    conn = np.zeros((2 * n, 2 * n))
    block = np.diag(np.diag(isc_mat)) if diagonal_only else isc_mat.copy()
    if n_lines is None:
        block[np.abs(block) < threshold] = 0.0
    conn[:n, n:] = block
    conn[n:, :n] = block.T

    # sub1 on left half, sub2 on right half
    node_angles = circular_layout(
        all_uniq, all_uniq,
        start_pos=90,
        group_boundaries=[0, n],
        group_sep=12,
    )

    groups = {c: sub1_id for c in uniq_sub1}
    groups.update({c: sub2_id for c in uniq_sub2})
    # purple / green — avoids confusion with HbO (red) / HbR (blue)
    color_map   = {sub1_id: "#8e44ad", sub2_id: "#27ae60"}
    node_colors = [mcolors.to_rgba(color_map[groups[c]]) for c in all_uniq]

    fig, _ = _plot_connectivity_circle(
        conn,
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
        fontsize_names=6,
        padding=1.0,
        show=False,
    )
    fig.set_size_inches(10, 10)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
