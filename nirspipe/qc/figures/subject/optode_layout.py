"""2-D optode flat map: source / detector positions coloured by SCI."""

import re

import mne
import numpy as np

from nirspipe.qc.common.figure_io import fig_png_b64

from nirspipe.qc.figures.common._utils import head_outline
from nirspipe.qc.figures.common.topomap import _projected_optodes
from nirspipe.qc.figures.subject.brain_views import _lookup_sci
from nirspipe.qc.figures.subject.raw_figures import sci_color, sci_legend

_OUTLINE_KEYS = ("head", "nose", "ear_left", "ear_right")


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
) -> str | None:
    """Matplotlib static optode flat map. Returns base64 PNG or None.

    A pair in ``bad_channels`` is red whatever its SCI, so this is a quality map rather than
    an SCI map: a channel rejected on its coupled-window share is never drawn green beside a
    table calling it BAD.

    Optodes sit where MNE's topomap projection puts them, inside MNE's head outline, so this
    map and the report's head maps describe the same head. A montage with no fitted sphere
    falls back to plain x/y inside a circle fitted to the optodes.
    """
    import matplotlib.pyplot as plt

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

    sources = {n: xy for n, xy in opt_xy.items() if n.upper().startswith("S")}
    detectors = {n: xy for n, xy in opt_xy.items() if n.upper().startswith("D")}

    xs = [float(xy[0]) for xy in opt_xy.values()]
    ys = [float(xy[1]) for xy in opt_xy.values()]
    if outlines:
        _mne_head(ax, outlines, xs, ys)
    else:
        head_outline(ax, xs, ys)

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
    ax.set_title(f"Optode flat map: {sci_legend(sci_threshold)}", fontsize=8)

    return fig_png_b64(fig)
