"""Resting-state QC figures.

alff_falff_figure():  per-channel ALFF and fALFF bar charts (HbO / HbR colour-coded).
fc_matrix_figure():   functional connectivity heatmaps, HbO and HbR as separate subplots.

# TODO: project ALFF/fALFF onto brain surface via mne_nirs when montage/head coords available.
# TODO: add ROI-to-ROI FC heatmap (atlas parcellation, e.g. via nilearn NiftiLabelsMasker analog).
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fnirs_pipe.utils.logging import get_logger

from ._utils import HBO_COLOR, HBR_COLOR

logger = get_logger("qc.figures.rest")

_HBO_COLOR = HBO_COLOR
_HBR_COLOR = HBR_COLOR
_MEAN_LINE_COLOR = "#555555"


def alff_falff_figure(
    alff_df: pd.DataFrame,
    title: str = "ALFF and fALFF per Channel",
) -> str:
    """Return base64 PNG of ALFF / fALFF bar charts.

    alff_df must have columns: channel, alff, falff.
    HbO channels (name ends with ' hbo') are drawn in red; HbR in blue.
    """
    channels = alff_df["channel"].tolist()
    alff_vals = alff_df["alff"].to_numpy(dtype=float)
    falff_vals = alff_df["falff"].to_numpy(dtype=float)

    is_hbo = np.array([ch.endswith(" hbo") for ch in channels])
    colors = [_HBO_COLOR if h else _HBR_COLOR for h in is_hbo]
    x = np.arange(len(channels))

    fig, (ax_a, ax_f) = plt.subplots(
        2, 1,
        figsize=(max(6.0, len(channels) * 0.22), 7),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.12)

    for ax, vals, ylabel, row_title in (
        (ax_a, alff_vals,  "ALFF",  "ALFF"),
        (ax_f, falff_vals, "fALFF", "fALFF"),
    ):
        ax.bar(x, vals, color=colors, edgecolor="none", alpha=0.85)

        hbo_mean = np.nanmean(vals[is_hbo]) if is_hbo.any() else None
        hbr_mean = np.nanmean(vals[~is_hbo]) if (~is_hbo).any() else None
        if hbo_mean is not None:
            ax.axhline(hbo_mean, color=_HBO_COLOR, lw=1.0, ls="--", alpha=0.7,
                       label=f"HbO mean = {hbo_mean:.4f}")
        if hbr_mean is not None:
            ax.axhline(hbr_mean, color=_HBR_COLOR, lw=1.0, ls="--", alpha=0.7,
                       label=f"HbR mean = {hbr_mean:.4f}")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(row_title, fontsize=10, pad=4)
        ax.legend(fontsize=7, frameon=False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", lw=0.6)
        ax.set_axisbelow(True)

    step = max(1, len(channels) // 30)
    ax_f.set_xticks(x[::step])
    ax_f.set_xticklabels(
        [channels[i] for i in range(0, len(channels), step)],
        rotation=45, ha="right", fontsize=6,
    )
    ax_f.set_xlabel("Channel", fontsize=9)

    # shared legend for colour meaning
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=_HBO_COLOR, label="HbO"),
        Patch(facecolor=_HBR_COLOR, label="HbR"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False,
               bbox_to_anchor=(1.0, 1.0))

    fig.suptitle(title, fontsize=11, y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fc_matrix_figure(
    fc_df: pd.DataFrame,
    title: str = "Functional Connectivity (Pearson r)",
) -> str:
    """Return base64 PNG of FC heatmaps, HbO and HbR as separate subplots.

    fc_df is a square channel × channel correlation DataFrame (from compute_fc).
    Diagonal is set to NaN so self-correlations are not shown.
    """
    ch_names = fc_df.columns.tolist()
    fc_mat = fc_df.to_numpy(dtype=float).copy()
    np.fill_diagonal(fc_mat, np.nan)

    hbo_idx = [i for i, c in enumerate(ch_names) if c.endswith(" hbo")]
    hbr_idx = [i for i, c in enumerate(ch_names) if c.endswith(" hbr")]

    hbo_names = [ch_names[i] for i in hbo_idx]
    hbr_names = [ch_names[i] for i in hbr_idx]
    hbo_mat = fc_mat[np.ix_(hbo_idx, hbo_idx)]
    hbr_mat = fc_mat[np.ix_(hbr_idx, hbr_idx)]

    def _square_size(n: int) -> float:
        return max(3.0, min(n * 0.14, 8.0))

    sq_hbo = _square_size(len(hbo_names))
    sq_hbr = _square_size(len(hbr_names))
    fig_w = sq_hbo + sq_hbr + 2.0
    fig_h = max(sq_hbo, sq_hbr)

    fig, (ax_o, ax_r) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [sq_hbo, sq_hbr]},
    )
    fig.subplots_adjust(wspace=0.4)

    for ax, mat, names, label, sq in (
        (ax_o, hbo_mat, hbo_names, "HbO", sq_hbo),
        (ax_r, hbr_mat, hbr_names, "HbR", sq_hbr),
    ):
        im = ax.imshow(mat, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
        n = len(names)
        step = max(1, n // 20)
        idxs = list(range(0, n, step))
        ax.set_xticks(idxs)
        ax.set_xticklabels([names[i] for i in idxs], fontsize=6,
                           rotation=45, ha="right")
        ax.set_yticks(idxs)
        ax.set_yticklabels([names[i] for i in idxs], fontsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r", pad=0.02)
        ax.set_title(f"FC — {label}", fontsize=10, pad=6)

    fig.suptitle(title, fontsize=11, y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
