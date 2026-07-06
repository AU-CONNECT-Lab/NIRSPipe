"""HbO–HbR correlation panel.

Left panel:  channel × channel Pearson r heatmap with HbO/HbR block divider.
Right panel: per-pair HbO–HbR r, sorted worst→best, colour-coded by quality.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from ._utils import HBO_COLOR, HBR_COLOR


def hbo_hbr_correlation_panel(
    raw_haemo: mne.io.Raw,
    title: str = "HbO–HbR Signal Quality",
) -> str:
    """Return base64 PNG of the correlation panel."""
    hbo_picks = mne.pick_types(raw_haemo.info, fnirs="hbo")
    hbr_picks = mne.pick_types(raw_haemo.info, fnirs="hbr")
    hbo_names = [raw_haemo.ch_names[i] for i in hbo_picks]
    hbr_names = [raw_haemo.ch_names[i] for i in hbr_picks]
    hbo_data  = raw_haemo.get_data(picks=hbo_picks)
    hbr_data  = raw_haemo.get_data(picks=hbr_picks)
    n_hbo = len(hbo_names)

    all_data  = np.vstack([hbo_data, hbr_data])
    all_names = hbo_names + hbr_names
    n_ch      = len(all_names)
    corr      = np.corrcoef(all_data).astype(float)
    np.fill_diagonal(corr, np.nan)

    hbo_map = {n.replace(" hbo", ""): hbo_data[i] for i, n in enumerate(hbo_names)}
    hbr_map = {n.replace(" hbr", ""): hbr_data[i] for i, n in enumerate(hbr_names)}
    pair_items = [
        (k, float(np.corrcoef(hbo_map[k], hbr_map[k])[0, 1]))
        for k in hbo_map if k in hbr_map
    ]
    pair_items.sort(key=lambda x: -x[1])   # worst (positive r) at top
    bar_labels = [p[0] for p in pair_items]
    bar_vals   = [p[1] for p in pair_items]
    bar_colors = [
        "#e74c3c" if v >= 0 else ("#f39c12" if v > -0.3 else "#27ae60")
        for v in bar_vals
    ]
    n_pairs = len(bar_labels)

    # left panel is a square: side = capped to [4, 7] inches
    sq = max(4.0, min(n_ch * 0.13, 7.0))
    # right panel width scales with label length, height matches sq
    bar_w = max(3.0, n_pairs * 0.08)
    fig, (ax_c, ax_b) = plt.subplots(
        1, 2,
        figsize=(sq + bar_w + 1.0, sq),
        gridspec_kw={"width_ratios": [sq, bar_w]},
    )
    fig.subplots_adjust(wspace=0.45)

    # ---- left: channel correlation matrix ----
    im = ax_c.imshow(corr, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1,
                     interpolation="nearest")
    ax_c.axhline(n_hbo - 0.5, color="#888", lw=0.6, ls="--")
    ax_c.axvline(n_hbo - 0.5, color="#888", lw=0.6, ls="--")

    step = max(1, n_ch // 20)
    idxs = list(range(0, n_ch, step))
    ax_c.set_xticks(idxs)
    ax_c.set_xticklabels([all_names[i] for i in idxs], fontsize=6,
                         rotation=45, ha="right")
    ax_c.set_yticks(idxs)
    ax_c.set_yticklabels([all_names[i] for i in idxs], fontsize=6)

    # block type labels on y-axis
    ax_c.text(-n_ch * 0.04, n_hbo / 2 - 0.5, "HbO",
              ha="right", va="center", fontsize=8, fontweight="bold",
              color=HBO_COLOR, clip_on=False)
    ax_c.text(-n_ch * 0.04, n_hbo + len(hbr_names) / 2 - 0.5, "HbR",
              ha="right", va="center", fontsize=8, fontweight="bold",
              color=HBR_COLOR, clip_on=False)

    for spine in ax_c.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax_c, shrink=0.65, label="Pearson r", pad=0.02)
    ax_c.set_title("Channel correlation matrix", fontsize=10, pad=6)

    # ---- right: per-pair HbO–HbR r (sorted) ----
    y_pos = np.arange(n_pairs)
    ax_b.barh(y_pos, bar_vals, color=bar_colors, alpha=0.85, height=0.72,
              edgecolor="none")
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(bar_labels, fontsize=7)
    ax_b.invert_yaxis()
    ax_b.axvline(x=0,    color="#444",     lw=0.8)
    ax_b.axvline(x=-0.3, color="#27ae60",  lw=0.8, ls="--", alpha=0.7)
    ax_b.set_xlabel("HbO–HbR Pearson r", fontsize=9)
    ax_b.set_title("Per-pair HbO–HbR r\n(green = r \u2264 \u22120.3, sorted worst\u2192best)",
                   fontsize=9)
    ax_b.grid(axis="x", color="#eeeeee", lw=0.6)
    ax_b.set_xlim(-1.1, 1.1)
    ax_b.set_axisbelow(True)

    fig.suptitle(title, fontsize=11, y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
