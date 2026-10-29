"""2-D optode flat map: channels coloured by SCI, optodes by the share of their long channels rejected."""

import re

import mne
import numpy as np

from nirspipe.io.optode_space import placement_of
from nirspipe.qc.common.channel_table import optode_rejections
from nirspipe.qc.common.figure_io import fig_png_b64
from nirspipe.qc.metrics._helpers import long_short_channels
from nirspipe.utils import pair_of

from nirspipe.qc.figures.common._utils import head_outline
from nirspipe.qc.figures.common.topomap import _projected_optodes
from nirspipe.qc.figures.subject.brain_views import _lookup_sci
from nirspipe.qc.figures.subject.raw_figures import (
    NO_SHARE_COLOR, SHARE_RAMP, sci_color, sci_legend,
)

_OUTLINE_KEYS = ("head", "nose", "ear_left", "ear_right")

_NO_SHARE_EDGE = "#98a4ae"
_OPTODE_EDGE = "#444444"


def _mne_head(ax, outlines: dict, xs, ys) -> None:
    """MNE's head, nose and ears, with limits that hold them and every optode."""
    for key in _OUTLINE_KEYS:
        if outlines.get(key) is not None:
            ax.plot(*outlines[key], color="#aaa", lw=1.5, zorder=0)
    all_x = list(xs) + [v for k in _OUTLINE_KEYS if outlines.get(k) is not None
                        for v in outlines[k][0]]
    all_y = list(ys) + [v for k in _OUTLINE_KEYS if outlines.get(k) is not None
                        for v in outlines[k][1]]
    pad = 0.04 * max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)


def optode_layout_static(
    raw: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    sci_threshold: float,
    sep_bands=None,
) -> str | None:
    """Matplotlib static optode flat map. Returns base64 PNG or None.

    A pair in ``bad_channels`` is red whatever its SCI, so this is a quality map rather than
    an SCI map: a channel rejected on its passing-window share is never drawn green beside a
    table calling it BAD.

    Each source (square) and detector (circle) is filled by the share of its long channels
    rejected, white for none to the rejected red for all, so an optode that lost every
    channel stands apart from losses spread over the montage. An optode only short channels
    use is a small grey mark. ``sep_bands`` is the run's separations; None takes the package
    defaults.

    Optodes sit where MNE's topomap projection puts them, inside MNE's head outline, so this
    map and the report's head maps describe the same head. A montage with no fitted sphere
    falls back to plain x/y inside a circle fitted to the optodes; one no reading placed is
    drawn as laid out, with no head.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D

    projected = _projected_optodes(raw)
    if projected is None:
        return None
    opt_xy, pairs, outlines = projected

    _ch_re = re.compile(r"(S\d+)[_\s]+(D\d+)", re.IGNORECASE)
    pair_sci: dict[str, list[float]] = {}
    for idx in mne.pick_types(raw.info, fnirs=True):
        name = raw.info["ch_names"][idx]
        m = _ch_re.search(name)
        sci = _lookup_sci(sci_scores, name)
        if m is not None and sci is not None:
            pair_sci.setdefault(f"{m.group(1).upper()}_{m.group(2).upper()}", []).append(sci)

    # matched on the source-detector pair, since screening names intensity channels and
    # this figure draws one line per pair
    bad_pairs = set()
    for name in bad_channels or []:
        m = _ch_re.search(str(name))
        if m is not None:
            bad_pairs.add(f"{m.group(1).upper()}_{m.group(2).upper()}")

    long_pairs = {pair_of(n).upper() for n in long_short_channels(raw, sep_bands)[0]}
    status = {f"{s}_{d}".upper(): {"separation": "long" if f"{s}_{d}".upper() in long_pairs
                                   else "short",
                                   "is_bad": f"{s}_{d}".upper() in bad_pairs}
              for s, d in pairs}
    shares = {optode: bad / listed
              for optode, (bad, listed) in optode_rejections(status).items()}

    def _color(sci, rejected=False):
        if rejected:
            return sci_color(None, sci_threshold, rejected=True)
        return "#95a5a6" if sci is None else sci_color(sci, sci_threshold)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    for s, d in pairs:
        pair_id = f"{s}_{d}".upper()
        vals = pair_sci.get(pair_id)
        c = _color(float(np.mean(vals)) if vals else None, pair_id in bad_pairs)
        ax.plot([opt_xy[s][0], opt_xy[d][0]], [opt_xy[s][1], opt_xy[d][1]],
                color=c, lw=2.0, zorder=1)

    sources = [n for n in opt_xy if n.upper().startswith("S")]
    detectors = [n for n in opt_xy if n.upper().startswith("D")]

    xs = [float(xy[0]) for xy in opt_xy.values()]
    ys = [float(xy[1]) for xy in opt_xy.values()]
    if outlines:
        _mne_head(ax, outlines, xs, ys)
    elif placement_of(raw).placed:
        head_outline(ax, xs, ys)

    cmap = LinearSegmentedColormap.from_list("rejected_share", SHARE_RAMP)
    norm = Normalize(0.0, 1.0)
    for names, marker, va in ((sources, "s", "bottom"), (detectors, "o", "top")):
        valued = [n for n in names if n.upper() in shares]
        bare = [n for n in names if n.upper() not in shares]
        if valued:
            ax.scatter([opt_xy[n][0] for n in valued], [opt_xy[n][1] for n in valued],
                       s=60, marker=marker, c=[shares[n.upper()] for n in valued],
                       cmap=cmap, norm=norm, edgecolors=_OPTODE_EDGE, linewidths=1.0, zorder=3)
        if bare:
            ax.scatter([opt_xy[n][0] for n in bare], [opt_xy[n][1] for n in bare],
                       s=22, marker=marker, c=NO_SHARE_COLOR, edgecolors=_NO_SHARE_EDGE,
                       linewidths=0.8, zorder=2)
        for n in names:
            x, y = opt_xy[n]
            ax.text(x, y, n, fontsize=6, ha="center", va=va, zorder=4,
                    color=_OPTODE_EDGE if n.upper() in shares else _NO_SHARE_EDGE)

    handles = [Line2D([], [], ls="", marker=m, ms=7, mfc="white", mec=_OPTODE_EDGE, label=label)
               for m, label, names in (("s", "Source", sources), ("o", "Detector", detectors))
               if names]
    if any(n.upper() not in shares for n in opt_xy):
        handles.append(Line2D([], [], ls="", marker="o", ms=4.5, mfc=NO_SHARE_COLOR,
                              mec=_NO_SHARE_EDGE, label="Short channels only"))
    # top right, outside the head, which reaches the axes' corner
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False)
    ax.set_title(f"Optode flat map: {sci_legend(sci_threshold)}", fontsize=8)

    if shares:
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                           shrink=0.4, aspect=16, pad=0.02)
        bar.set_ticks([0.0, 0.5, 1.0], labels=["0%", "50%", "100%"])
        bar.set_label("Optode: long channels rejected", fontsize=7)
        bar.ax.tick_params(labelsize=7, width=0.5)
        bar.outline.set_linewidth(0.5)

    return fig_png_b64(fig)
