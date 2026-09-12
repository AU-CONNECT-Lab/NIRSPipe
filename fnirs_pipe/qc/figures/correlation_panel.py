"""HbO–HbR correlation panel.

Top row:    one channel × channel Pearson r heatmap per stage, before and after denoising,
            on one colour scale so the pair subtracts by eye.
Bottom row: per-pair HbO–HbR r as a dumbbell, before → after, grouped by source-detector
            separation.

The two stages used to be two separate figures, each with its own bar chart sorted by its
own r, so the same channel sat at a different height in each and comparing one channel
across the step meant finding it twice in two differently ordered lists. One figure, one
channel order, and the change is the length of a connector.

Separation is what the grouping is for. The HbO–HbR anticorrelation that the −0.3
threshold tests for is a property of cortical haemodynamics, so it says nothing about a
short channel, which only ever sees scalp. Sorted together the short channels land at one
end of the list and read as the worst channels on the montage when they are simply not
being asked the same question.

Both panels read their colour off one RdBu_r scale, so a shade means the same r whether it
is a matrix cell or a dot. The verdict is the dashed −0.3 rule rather than a third hue: the
green/amber/red fills this replaced separated amber from green by ΔE 5.8 under protanopia,
so the one distinction the panel exists to make was unreadable to a red-green colourblind
reader. A single threshold rule keeps its green, having nothing to be confused against.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import mne
import numpy as np

# A rotated channel label needs about this much axis to stay legible at 5.5 pt. The cap
# used to be a constant 60, which was fine for one wide heatmap and unreadable once two of
# them split the width: 44 labels at 4.2 pt is a grey smear, not a label.
_TICK_LABEL_PITCH_IN = 0.11

# The separation groups, in the order they are drawn. "mid" is the 10-15 mm gap that
# long_short_channels leaves unclaimed; it is usually empty.
_GROUP_ORDER = ["long", "mid", "short"]
_GROUP_LABEL = {
    "long":  "long channels",
    "mid":   "mid-range channels",
    "short": "short channels",
}

# RdBu_r for both panels: a white midpoint rather than a tinted one, which is what keeps
# a correlation matrix reading as "nothing here" in the middle instead of beige. The dots
# below take their fill from the same map, so one colour means one r across the figure.
_DIVERGING = plt.get_cmap("RdBu_r")
_NORM = Normalize(-1.0, 1.0)

_MUTED, _GRID, _BASELINE = "#888888", "#eeeeee", "#444444"
_SURFACE = "#ffffff"
# a white fill at r near zero would vanish on a white surface, so the marks carry a ring
_MARK_EDGE = "#9aa0a6"

_R_THRESHOLD = -0.3


def _pair_key(ch_name: str) -> str:
    return ch_name.rsplit(" ", 1)[0]


def _pair_group(raw_haemo: mne.io.Raw, sep_bands=None) -> "dict[str, str]":
    """Map each S-D pair to ``long`` / ``mid`` / ``short``.

    ``long_short_channels`` returns names with the chromophore suffix and leaves the
    10-15 mm band in neither list, so::

        ["S1_D1 hbo", "S1_D8 hbo"] -> {"S1_D1": "long", "S1_D8": "short"}
    """
    from fnirs_pipe.qc.metrics import long_short_channels

    long_names, short_names = long_short_channels(raw_haemo, sep_bands)
    groups = {_pair_key(n): "mid" for n in raw_haemo.ch_names}
    groups.update({_pair_key(n): "long" for n in long_names})
    groups.update({_pair_key(n): "short" for n in short_names})
    return groups


def _stage(raw: mne.io.Raw, order: "list[str]"):
    """One stage's correlation matrix and per-pair r, on a channel order fixed elsewhere.

    ``order`` is the before stage's channel order, so both stages index the same rows and
    a cell at (i, j) means the same pair of channels in both heatmaps.
    """
    idx = {n: i for i, n in enumerate(raw.ch_names)}
    picks = [idx[n] for n in order if n in idx]
    if len(picks) != len(order):
        return None, None
    data = raw.get_data(picks=picks)
    n = len(order)
    corr = np.corrcoef(data).astype(float)
    np.fill_diagonal(corr, np.nan)
    # Lower triangle only: the matrix is symmetric, so the upper half is the same values
    # read the other way round. The HbO x HbR block, which is what the panel is for,
    # survives once in the lower left.
    corr[np.triu_indices(n, k=1)] = np.nan

    by_name = dict(zip(order, data))
    pair_r = {}
    for name in order:
        k = _pair_key(name)
        hbo, hbr = f"{k} hbo", f"{k} hbr"
        if k not in pair_r and hbo in by_name and hbr in by_name:
            pair_r[k] = float(np.corrcoef(by_name[hbo], by_name[hbr])[0, 1])
    return corr, pair_r


def _draw_heatmap(ax, corr, all_names, n_hbo, groups, hbo_names, hbr_names,
                  title, show_yticks, max_labels):
    n_ch = len(all_names)
    im = ax.imshow(corr, aspect="equal", cmap=_DIVERGING, norm=_NORM,
                   interpolation="nearest")

    def _divider(pos: float, **style) -> None:
        # an L hugging the diagonal, since a full-width rule would run out over the blank
        # upper triangle and read as part of the plot
        ax.plot([-0.5, pos], [pos, pos], **style)
        ax.plot([pos, pos], [pos, n_ch - 0.5], **style)

    _divider(n_hbo - 0.5, color="#888", lw=0.6, ls="--")
    # separation boundaries inside each chromophore block, lighter than the HbO/HbR one
    for offset, names in [(0, hbo_names), (n_hbo, hbr_names)]:
        seen = [groups[_pair_key(n)] for n in names]
        for i in range(1, len(seen)):
            if seen[i] != seen[i - 1]:
                _divider(offset + i - 0.5, color="#ccc", lw=0.5, ls=":")

    # Label every channel while they still fit; past that, subsample and mark the
    # unlabelled cells with minor ticks so the rows stay countable
    step = 1 if n_ch <= max_labels else int(np.ceil(n_ch / max_labels))
    idxs = list(range(0, n_ch, step))
    tick_fs = 5.5
    ax.set_xticks(idxs)
    ax.set_xticklabels([all_names[i] for i in idxs], fontsize=tick_fs,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(idxs)
    # No HbO / HbR block labels: every tick label already ends in the chromophore, and
    # the divider marks where one block stops. The labels were the third telling.
    ax.set_yticklabels([all_names[i] for i in idxs] if show_yticks else [],
                       fontsize=tick_fs)
    if step > 1:
        ax.set_xticks(np.arange(n_ch), minor=True)
        ax.set_yticks(np.arange(n_ch), minor=True)
        ax.tick_params(which="minor", length=1.5, width=0.4, color="#bbbbbb")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, pad=6)
    return im


def _draw_dumbbell(ax, groups, r_before, r_after, show_headers):
    """Per-pair r, one x position per pair, before as an open ring and after filled.

    Ordered best→worst inside each group on the *before* value, so one order serves both
    stages and the −0.3 crossing happens once along the row.
    """
    xs, labels, before, after = [], [], [], []
    spans = []
    x = 0.0
    for g in _GROUP_ORDER:
        keys = sorted((k for k in r_before if groups.get(k) == g),
                      key=lambda k: r_before[k])
        if not keys:
            continue
        start = x
        for k in keys:
            xs.append(x)
            labels.append(k)
            before.append(r_before[k])
            after.append(r_after.get(k) if r_after else None)
            x += 1.0
        spans.append((start, x - 1.0, _GROUP_LABEL[g]))
        x += 1.4          # a gap between groups, so the grouping needs no rule
    if not xs:
        return

    xs = np.asarray(xs, dtype=float)
    before = np.asarray(before, dtype=float)
    paired = r_after is not None and all(v is not None for v in after)
    tip = np.asarray(after, dtype=float) if paired else before

    ax.axhline(0.0, color=_BASELINE, lw=0.9, zorder=1)
    ax.axhline(_R_THRESHOLD, color="#27ae60", lw=0.9, ls="--", alpha=0.7, zorder=1)
    ax.text(xs[-1] + 0.9, _R_THRESHOLD, "r = −0.3", color="#27ae60", fontsize=7,
            alpha=0.9, va="bottom", ha="right")

    if paired:
        ax.vlines(xs, before, tip, color=_MUTED, lw=1.4, alpha=0.55, zorder=2)
        ax.scatter(xs, before, s=26, facecolor=_SURFACE, edgecolor=_MUTED, lw=1.2,
                   zorder=3, label="before denoising")
    # the surface ring keeps a mark near the neutral midpoint visible on a white surface
    ax.scatter(xs, tip, s=48, c=_DIVERGING(_NORM(tip)), edgecolor=_MARK_EDGE, lw=0.8,
               zorder=4, label="after denoising" if paired else "HbO–HbR r")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    # headroom above r = 1 for the group headers, which would otherwise land on the marks
    ax.set_ylim(-1.12, 1.34)
    ax.set_xlim(-1.0, xs[-1] + 1.0)
    ax.grid(axis="y", color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, labelsize=7)
    ax.set_ylabel("HbO–HbR r", fontsize=9)
    if show_headers:
        for xa, xb, name in spans:
            ax.annotate(name, xy=((xa + xb) / 2, 1.16), fontsize=7.5, color="#555",
                        ha="center", va="center", fontweight="bold")
    if paired:
        # above the axes: at lower right it sat on the short channels, which is exactly
        # where this panel puts its most positive values
        ax.legend(frameon=False, fontsize=7.5, loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.4,
                  borderaxespad=0.0, columnspacing=1.2)
    # pad clears the group headers, which sit just inside the top of the axes
    ax.set_title("Per-pair HbO–HbR r, best→worst"
                 + (" within group" if show_headers else ""),
                 fontsize=10, pad=8, loc="left")


def hbo_hbr_correlation_panel(
    raw_haemo: mne.io.Raw,
    title: str = "HbO–HbR Signal Quality",
    sep_bands=None,
    raw_after: "mne.io.Raw | None" = None,
) -> str:
    """Return base64 PNG of the correlation panel.

    ``raw_after`` is the denoised recording. Given it, the panel carries both stages: two
    heatmaps on one colour scale and a dumbbell per pair. Left out, or not matching the
    before stage's channels, it degrades to the one-stage panel: one heatmap and one dot
    per pair, which is what a run with no denoising and what a condition page both get.
    """
    groups = _pair_group(raw_haemo, sep_bands)
    rank = {g: i for i, g in enumerate(_GROUP_ORDER)}

    def _by_separation(picks):
        # stable, so the montage's own channel order survives inside each group
        return sorted(picks, key=lambda i: rank[groups[_pair_key(raw_haemo.ch_names[i])]])

    hbo_picks = _by_separation(mne.pick_types(raw_haemo.info, fnirs="hbo"))
    hbr_picks = _by_separation(mne.pick_types(raw_haemo.info, fnirs="hbr"))
    hbo_names = [raw_haemo.ch_names[i] for i in hbo_picks]
    hbr_names = [raw_haemo.ch_names[i] for i in hbr_picks]
    all_names = hbo_names + hbr_names
    n_hbo, n_ch = len(hbo_names), len(all_names)

    corr_b, r_before = _stage(raw_haemo, all_names)
    corr_a, r_after = (_stage(raw_after, all_names) if raw_after is not None
                       else (None, None))
    stages = [(corr_b, "before denoising")]
    if corr_a is not None:
        stages.append((corr_a, "after denoising"))

    show_headers = len({groups.get(k) for k in r_before}) > 1

    # Two square heatmaps side by side already make a wide top row, so the figure lands
    # near 1.5:1 without padding either panel: the report scales it to the page width and
    # a squarer figure eats the viewport height at no gain in detail.
    # Two squares side by side add width twice as fast as height, so a *larger* square is
    # what makes this figure wider rather than taller: shrinking it leaves the fixed chrome
    # (two rows of rotated tick labels and two titles) as a bigger share of the height and
    # the panel comes out squarer. Measured on a 44-channel montage, the layout floors at
    # about 1160 px of screen height in a 1600 px column and does not improve past sq 5.2.
    sq = max(3.4, min(n_ch * 0.118, 5.2))
    dumb_h = 1.50
    fig = plt.figure(figsize=(sq * len(stages) + 0.55, sq + dumb_h))
    gs = fig.add_gridspec(
        2, len(stages) + 1,
        height_ratios=[sq, dumb_h],
        width_ratios=[sq] * len(stages) + [sq * 0.032],
        # 0.78 put a 2.5 in band of nothing between the two rows, a third of the height
        hspace=0.62, wspace=0.06,
    )
    # the heatmap axes is roughly three quarters of the square it is allotted
    max_labels = max(10, int(sq * 0.76 / _TICK_LABEL_PITCH_IN))
    im = last_ax = None
    for col, (corr, stage_title) in enumerate(stages):
        last_ax = fig.add_subplot(gs[0, col])
        im = _draw_heatmap(last_ax, corr, all_names, n_hbo, groups, hbo_names, hbr_names,
                           stage_title, show_yticks=(col == 0), max_labels=max_labels)
    # One colour bar for both heatmaps, which is the point of drawing them on one scale.
    cax = fig.add_subplot(gs[0, len(stages)])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    _draw_dumbbell(fig.add_subplot(gs[1, :len(stages)]), groups, r_before, r_after,
                   show_headers)

    fig.suptitle(title, fontsize=11, y=0.99)

    buf = io.BytesIO()
    # 200 rather than 300: the report scales this to the page width, so the extra pixels
    # were never displayed and cost half the file
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=_SURFACE)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
