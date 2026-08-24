"""Resting-state QC figures.

alff_falff_figure():    per-channel ALFF and fALFF bar charts (HbO / HbR colour-coded).
fc_matrix_figure():     functional connectivity heatmaps, HbO and HbR as separate subplots.
fc_seed_topo_figure():  seed-to-whole-brain correlations drawn on the optode flat map.

# TODO: project ALFF/fALFF onto brain surface via mne_nirs when montage/head coords available.
# TODO: add ROI-to-ROI FC heatmap (atlas parcellation, e.g. via nilearn NiftiLabelsMasker analog).
"""

from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from fnirs_pipe.utils.logging import get_logger

from ._utils import HBO_COLOR, HBR_COLOR, head_outline

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
    fc_hbr_df: pd.DataFrame | None = None,
    title: str = "Functional Connectivity (Pearson r)",
) -> str:
    """Return base64 PNG of FC heatmaps, one panel per chromophore present.

    compute_fc returns one matrix per chromophore, so HbO arrives in fc_df and HbR in
    fc_hbr_df. A single matrix holding both is also accepted and split by channel suffix.
    Diagonal is set to NaN so self-correlations are not shown.
    """
    panels: list[tuple[list[str], np.ndarray, str]] = []
    for frame in (fc_df, fc_hbr_df):
        if frame is None or frame.empty:
            continue
        names = frame.columns.tolist()
        mat = frame.to_numpy(dtype=float).copy()
        np.fill_diagonal(mat, np.nan)
        for suffix, label in ((" hbo", "HbO"), (" hbr", "HbR")):
            idx = [i for i, c in enumerate(names) if c.endswith(suffix)]
            if idx:
                panels.append(([names[i] for i in idx], mat[np.ix_(idx, idx)], label))

    if not panels:
        raise ValueError("fc_df has no recognised HbO/HbR channels")

    def _square_size(n: int) -> float:
        return max(3.0, min(n * 0.14, 8.0))

    sizes = [_square_size(len(names)) for names, _, _ in panels]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(sum(sizes) + 2.0, max(sizes)),
        gridspec_kw={"width_ratios": sizes},
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.4)

    for ax, (names, mat, label) in zip(axes[0], panels):
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


def _channel_endpoints(raw: mne.io.Raw) -> "dict[str, tuple[tuple[float, float], tuple[float, float]]]":
    """{channel name: ((source x, y), (detector x, y))} for every channel with usable positions.

    e.g. "S1_D1 hbo" -> ((-0.031, 0.088), (-0.012, 0.093)). Channels whose montage carries no
    optode coordinates report all-zero or NaN locations and are left out, which is what makes
    an empty return the signal that no flat map can be drawn at all.
    """
    ends: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for idx in mne.pick_types(raw.info, fnirs=True, exclude=[]):
        loc = raw.info["chs"][idx]["loc"]
        src, det = loc[3:6], loc[6:9]
        if np.any(np.isnan(src)) or np.any(np.isnan(det)):
            continue
        if np.allclose(src, 0) and np.allclose(det, 0):
            continue
        ends[raw.info["ch_names"][idx]] = (
            (float(src[0]), float(src[1])), (float(det[0]), float(det[1])),
        )
    return ends


def fc_seed_topo_figure(
    raw: mne.io.Raw,
    seed_df: pd.DataFrame,
    seed_hbr_df: pd.DataFrame | None = None,
    title: str = "Seed-to-whole-brain connectivity (Pearson r)",
) -> str | None:
    """Return base64 PNG of one flat map per seed ROI, or None if the montage has no positions.

    compute_fc_seed returns an ROI x channel frame per chromophore, so HbO arrives in seed_df
    and HbR in seed_hbr_df. Each channel is drawn as its source-to-detector segment on the
    optode flat map, coloured by that seed's correlation with it, on the same RdBu_r / +-1
    scale fc_matrix_figure uses so the two figures can be read against each other.

    Three states are distinguishable on purpose, because confusing them is the mistake this
    figure exists to avoid:

    - an ordinary channel, coloured by r;
    - a channel **inside the seed**, grey and thicker. Its value is NaN rather than zero, and
      grey says "no claim made here" where a blue line would say "no connection";
    - a **rejected** channel, drawn at low alpha. It still carries a real correlation, but the
      channel was excluded upstream.

    Short channels are not drawn: they measure extracerebral signal, so a correlation with
    them is not a connectivity claim. Channels are drawn weakest first, so strong connections
    are never hidden under weak ones.
    """
    from fnirs_pipe.qc.quantitative_metrics import long_short_channels

    panels = [(f, lab) for f, lab in ((seed_df, "HbO"), (seed_hbr_df, "HbR"))
              if f is not None and not f.empty]
    if not panels:
        return None

    ends = _channel_endpoints(raw)
    if not ends:
        logger.warning("seed topography skipped: montage carries no optode positions")
        return None

    long_names, _ = long_short_channels(raw)
    drawable = set(ends) & (set(long_names) or set(ends))   # no split at all -> draw everything
    bads = set(raw.info["bads"])

    rois = list(dict.fromkeys([r for frame, _ in panels for r in frame.index]))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#aaa")
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # one head for every panel, sized from all drawable channels rather than from whichever
    # subset a given panel happens to draw
    head_x = [c for ch in drawable for c in (ends[ch][0][0], ends[ch][1][0])]
    head_y = [c for ch in drawable for c in (ends[ch][0][1], ends[ch][1][1])]

    n_rows, n_cols = len(rois), len(panels)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.0 + 1.4, n_rows * 3.0), squeeze=False,
    )

    for i, roi in enumerate(rois):
        for j, (frame, label) in enumerate(panels):
            ax = axes[i][j]
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"{roi} — {label}", fontsize=9, pad=4)
            head_outline(ax, head_x, head_y)
            if roi not in frame.index:
                ax.text(0.5, 0.5, "no good channel", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="#888")
                continue

            row = frame.loc[roi]
            cells = [(ch, float(row[ch])) for ch in row.index if ch in drawable]
            # weakest first so a strong connection is never hidden under a weak one; the
            # seed's own NaN channels sort last and sit on top, which is where they belong
            cells.sort(key=lambda c: (np.isnan(c[1]), abs(c[1]) if not np.isnan(c[1]) else 0.0))

            for ch, v in cells:
                (sx, sy), (dx, dy) = ends[ch]
                in_seed = np.isnan(v)
                ax.plot([sx, dx], [sy, dy],
                        color=cmap(norm(np.ma.masked_invalid([v])))[0],
                        lw=2.5 if in_seed else 2.0,
                        alpha=0.35 if ch in bads else 1.0, zorder=1,
                        solid_capstyle="round")

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, shrink=0.6,
        label="Pearson r", pad=0.02,
    )
    fig.suptitle(title, fontsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
