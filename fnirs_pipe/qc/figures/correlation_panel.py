"""HbO–HbR correlation panel.

Left panel:  channel × channel Pearson r heatmap with HbO/HbR block divider.
Right panel: per-pair HbO–HbR r, grouped by source-detector separation and sorted
             worst→best inside each group.

Separation is what the grouping is for. The HbO–HbR anticorrelation that the −0.3
threshold tests for is a property of cortical haemodynamics, so it says nothing about a
short channel, which only ever sees scalp. Sorted together the short channels land at the
top of the list and read as the worst channels on the montage when they are simply not
being asked the same question.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from ._utils import HBO_COLOR, HBR_COLOR

_MAX_TICK_LABELS = 60

# The separation groups, in the order they are drawn. "mid" is the 10-15 mm gap that
# long_short_channels leaves unclaimed; it is usually empty.
_GROUP_ORDER = ["long", "mid", "short"]
_GROUP_LABEL = {
    "long":  "long channels",
    "mid":   "mid-range channels",
    "short": "short channels",
}
_NEUTRAL_COLOR = "#95a5a6"


def _pair_key(ch_name: str) -> str:
    return ch_name.rsplit(" ", 1)[0]


def _pair_group(raw_haemo: mne.io.Raw) -> "dict[str, str]":
    """Map each S-D pair to ``long`` / ``mid`` / ``short``.

    ``long_short_channels`` returns names with the chromophore suffix and leaves the
    10-15 mm band in neither list, so::

        ["S1_D1 hbo", "S1_D8 hbo"] -> {"S1_D1": "long", "S1_D8": "short"}
    """
    from fnirs_pipe.qc.quantitative_metrics import long_short_channels

    long_names, short_names = long_short_channels(raw_haemo)
    groups = {_pair_key(n): "mid" for n in raw_haemo.ch_names}
    groups.update({_pair_key(n): "long" for n in long_names})
    groups.update({_pair_key(n): "short" for n in short_names})
    return groups


def hbo_hbr_correlation_panel(
    raw_haemo: mne.io.Raw,
    title: str = "HbO–HbR Signal Quality",
) -> str:
    """Return base64 PNG of the correlation panel."""
    groups = _pair_group(raw_haemo)
    rank = {g: i for i, g in enumerate(_GROUP_ORDER)}

    def _by_separation(picks):
        # stable, so the montage's own channel order survives inside each group
        return sorted(picks, key=lambda i: rank[groups[_pair_key(raw_haemo.ch_names[i])]])

    hbo_picks = _by_separation(mne.pick_types(raw_haemo.info, fnirs="hbo"))
    hbr_picks = _by_separation(mne.pick_types(raw_haemo.info, fnirs="hbr"))
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

    hbo_map = {_pair_key(n): hbo_data[i] for i, n in enumerate(hbo_names)}
    hbr_map = {_pair_key(n): hbr_data[i] for i, n in enumerate(hbr_names)}
    pair_r = {
        k: float(np.corrcoef(hbo_map[k], hbr_map[k])[0, 1])
        for k in hbo_map if k in hbr_map
    }

    # ---- lay the bar rows out group by group ----
    # A header row carries the group name in place of a pair name, so the grouping needs
    # no extra text placed over the bars. Headers are dropped only when every pair is a
    # long channel, the case where the quality colours speak for the whole list.
    grouped = [
        (g, sorted([k for k in pair_r if groups[k] == g], key=lambda k: -pair_r[k]))
        for g in _GROUP_ORDER
    ]
    grouped = [(g, keys) for g, keys in grouped if keys]
    show_headers = not (len(grouped) == 1 and grouped[0][0] == "long")

    rows: list[tuple[float, str, str]] = []   # (y, kind, label)
    bar_y, bar_vals, bar_colors = [], [], []
    long_span: list[float] = []
    y = 0.0
    for g, keys in grouped:
        if show_headers:
            if rows:
                y += 0.7
            rows.append((y, "header", _GROUP_LABEL[g]))
            y += 1.0
        for k in keys:
            v = pair_r[k]
            rows.append((y, "bar", k))
            bar_y.append(y)
            bar_vals.append(v)
            bar_colors.append(
                ("#e74c3c" if v >= 0 else ("#f39c12" if v > -0.3 else "#27ae60"))
                if g == "long" else _NEUTRAL_COLOR
            )
            if g == "long":
                long_span.append(y)
            y += 1.0
    n_rows = len(rows)

    # left panel is a square: side = capped to [4, 7] inches
    sq = max(4.0, min(n_ch * 0.13, 7.0))
    # right panel width scales with label length, height matches sq
    bar_w = max(3.0, n_rows * 0.08)
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

    # separation boundaries inside each chromophore block, lighter than the HbO/HbR one
    for offset, names in [(0, hbo_names), (n_hbo, hbr_names)]:
        seen = [groups[_pair_key(n)] for n in names]
        for i in range(1, len(seen)):
            if seen[i] != seen[i - 1]:
                ax_c.axhline(offset + i - 0.5, color="#ccc", lw=0.5, ls=":")
                ax_c.axvline(offset + i - 0.5, color="#ccc", lw=0.5, ls=":")

    # Label every channel while they still fit; past that, subsample and mark the
    # unlabelled cells with minor ticks so the rows stay countable
    step  = 1 if n_ch <= _MAX_TICK_LABELS else int(np.ceil(n_ch / _MAX_TICK_LABELS))
    idxs  = list(range(0, n_ch, step))
    tick_fs = 6 if n_ch <= 40 else 4.5
    ax_c.set_xticks(idxs)
    ax_c.set_xticklabels([all_names[i] for i in idxs], fontsize=tick_fs,
                         rotation=45, ha="right")
    ax_c.set_yticks(idxs)
    ax_c.set_yticklabels([all_names[i] for i in idxs], fontsize=tick_fs)
    if step > 1:
        ax_c.set_xticks(np.arange(n_ch), minor=True)
        ax_c.set_yticks(np.arange(n_ch), minor=True)
        ax_c.tick_params(which="minor", length=1.5, width=0.4, color="#bbbbbb")

    # block type labels, placed outside the tick labels (x in axes fraction, y in rows)
    block_tr = ax_c.get_yaxis_transform()
    for label, centre, color in [("HbO", n_hbo / 2 - 0.5, HBO_COLOR),
                                 ("HbR", n_hbo + len(hbr_names) / 2 - 0.5, HBR_COLOR)]:
        ax_c.text(-0.24, centre, label, transform=block_tr,
                  ha="right", va="center", fontsize=9, fontweight="bold",
                  color=color, clip_on=False)

    for spine in ax_c.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax_c, shrink=0.65, label="Pearson r", pad=0.02)
    ax_c.set_title("Channel correlation matrix"
                   + (" (grouped by separation)" if show_headers else ""),
                   fontsize=10, pad=6)

    # ---- right: per-pair HbO–HbR r (grouped, sorted) ----
    ax_b.barh(bar_y, bar_vals, color=bar_colors, alpha=0.85, height=0.72,
              edgecolor="none")
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    ax_b.set_yticks([r[0] for r in rows])
    ax_b.set_yticklabels([r[2] for r in rows], fontsize=7)
    for tick, (_, kind, _) in zip(ax_b.get_yticklabels(), rows):
        if kind == "header":
            tick.set_fontweight("bold")
            tick.set_color("#555")
            tick.set_fontsize(7.5)
    ax_b.set_ylim(-0.8, (rows[-1][0] if rows else 0) + 0.8)
    ax_b.invert_yaxis()
    ax_b.axvline(x=0, color="#444", lw=0.8)
    if long_span:
        ax_b.plot([-0.3, -0.3], [min(long_span) - 0.5, max(long_span) + 0.5],
                  color="#27ae60", lw=0.8, ls="--", alpha=0.7)
    ax_b.set_xlabel("HbO–HbR Pearson r", fontsize=9)
    subtitle = ("sorted worst→best within group\n"
                "(green = r ≤ −0.3, long channels only)" if show_headers else
                "(green = r ≤ −0.3, sorted worst→best)")
    ax_b.set_title(f"Per-pair HbO–HbR r\n{subtitle}", fontsize=9)
    ax_b.grid(axis="x", color="#eeeeee", lw=0.6)
    ax_b.set_xlim(-1.1, 1.1)
    ax_b.set_axisbelow(True)

    fig.suptitle(title, fontsize=11, y=1.01)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
