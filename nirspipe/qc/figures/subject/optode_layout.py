"""2-D optode flat map: source / detector positions coloured by SCI."""

import re

import mne
import numpy as np

from nirspipe.qc.common.figure_io import fig_png_b64

from nirspipe.qc.figures.common._utils import head_outline
from nirspipe.qc.figures.subject.brain_views import _lookup_sci
from nirspipe.qc.figures.subject.raw_figures import sci_color, sci_legend


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
    """
    import matplotlib.pyplot as plt

    picks =mne.pick_types(raw.info, fnirs=True)
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
        sci = _lookup_sci(sci_scores, name)
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
    ax.set_title(f"Optode flat map: {sci_legend(sci_threshold)}", fontsize=8)

    return fig_png_b64(fig)
