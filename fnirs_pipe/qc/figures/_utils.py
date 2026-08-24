"""Shared color constants and helpers used across QC figure modules."""

import numpy as np

HBO_COLOR      = "#e74c3c"
HBR_COLOR      = "#3498db"
HBO_MEAN_COLOR = "#c0392b"   # darker HbO for bold mean line
HBR_MEAN_COLOR = "#1a5276"   # darker HbR for bold mean line

# Qualitative palette for condition/trigger colors (cycled by index).
CONDITION_PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]


def decimate(arr: np.ndarray, times: np.ndarray, max_pts: int):
    """Uniformly subsample columns of arr (and times) to at most max_pts for display."""
    if len(times) <= max_pts:
        return arr, times
    step = max(1, len(times) // max_pts)
    return arr[:, ::step], times[::step]


def physio_bands(cardiac=None, resp=None):
    """PSD annotation bands as (name, f_lo, f_hi). Single source so every PSD figure agrees.

    Mayer is a fixed visual reference (no metric counterpart). ``cardiac`` / ``resp`` take
    the CLI ``(l_freq, h_freq)`` so the shading matches the bands the metrics integrate over;
    pass None to omit that band entirely rather than guess a default (e.g. the hyper pipeline
    has no respiration band).
    """
    bands = [("Mayer", 0.07, 0.13)]
    if resp is not None:
        bands.append(("Resp", *resp))
    if cardiac is not None:
        bands.append(("Cardiac", *cardiac))
    return bands


def head_outline(ax, xs, ys):
    """Draw the head circle, ears and nose around a set of optode x/y, and set the limits.

    One definition so every flat map in the report has the same head. The circle is centred
    on the optode bounding box, not on the origin, because montage coordinates are in head
    space and a frontal-only cap sits well off centre::

        xs = [-0.05, 0.05], ys = [0.0, 0.08]  ->  centre (0, 0.04), radius 0.047

    Returns ``(cx, cy, r)`` so the caller can place anything else relative to the head.
    """
    import matplotlib.patches as mpatches

    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    # 1.18 leaves the outermost optode just inside the scalp rather than on it
    r = max(max(abs(x - cx) for x in xs), max(abs(y - cy) for y in ys)) * 1.18

    ax.add_patch(mpatches.Circle((cx, cy), r, fill=False, edgecolor="#aaa", lw=1.5, zorder=0))
    nw, nh = r * 0.06, r * 0.10
    ax.fill([cx - nw, cx, cx + nw, cx - nw],
            [cy + r - nh * 0.3, cy + r + nh, cy + r - nh * 0.3, cy + r - nh * 0.3],
            color="#ddd", edgecolor="#aaa", lw=1, zorder=0)
    for ex in (cx - r, cx + r):
        ax.add_patch(mpatches.Ellipse((ex, cy), r * 0.14, r * 0.24,
                                      fc="#ddd", ec="#aaa", lw=1, zorder=0))

    # bounds relative to the head so the ears and nose are never clipped
    pad = r * 0.25
    ax.set_xlim(cx - r - r * 0.14 - pad, cx + r + r * 0.14 + pad)
    ax.set_ylim(cy - r - pad, cy + r + r * 0.12 + pad)
    return cx, cy, r
