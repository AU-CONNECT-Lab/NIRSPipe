"""The conventions every matrix panel in the reports is drawn with.

Which colour scale a quantity takes, which rows and columns a matrix keeps, and how a
cell's number is printed on top of it. One copy, so a red cell cannot mean +1 in one panel
and -1 in the next.

Layout stays with each figure: a lower-triangle channel matrix with separation dividers, a
pair of square per-chromophore matrices and a cross-brain grid are different pictures. What
they must not disagree about is what a colour means.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Diverging and centred on zero, red at +1, because a correlation has a meaningful middle.
# An unsigned bounded quantity such as a coherence has none and takes a sequential scale.
CORRELATION_SCALE = "RdBu_r"
COHERENCE_SCALE = "viridis"

def scale_color(value: float, cmap: str, vmin: float, vmax: float) -> str:
    """The colour a scale gives one value, as ``rgb(r,g,b)``.

    ``scale_color(0.0, "RdBu_r", -1, 1) -> the scale's midpoint colour``

    For anything that has to match a heatmap's fill without being one: the ink a printed
    cell value takes, or a chord coloured by the same quantity the matrix beside it is.
    """
    from plotly.colors import sample_colorscale

    t = (float(value) - vmin) / ((vmax - vmin) or 1.0)
    return sample_colorscale(cmap, [float(np.clip(t, 0.0, 1.0))])[0]


def kept_axes(mats) -> tuple[np.ndarray, np.ndarray]:
    """Which rows and which columns hold a finite value in any of ``mats``.

    ::

      [[0.4, nan], [nan, nan]]  ->  rows [True, False], columns [True, False]

    A figure draws only these: a site no value was computed for is left off the axis rather
    than drawn as an empty row. The tables on disk keep every site.
    """
    finite = np.any([np.isfinite(np.asarray(m, dtype=float)) for m in mats], axis=0)
    return finite.any(axis=1), finite.any(axis=0)


def cell_values(fig, z, row_labels, col_labels, *, cmap, vmin, vmax, row, col,
                fmt: str = ".2f", strip_zero: bool = True) -> None:
    """Print every cell's value on the heatmap, in ink the cell's own colour can carry.

    ::

      a cell at .69 -> white text;  the same number at .12 -> black

    **Every cell gets its number**, so a matrix on a fixed scale whose values sit close
    together stays readable.

    The ink is chosen from **the cell's own colour** and not from its value, so one rule
    serves sequential and diverging scales alike. Two text traces rather than one, because a
    heatmap takes a single ``textfont`` for the whole grid.
    """
    n = max(len(row_labels), len(col_labels), 1)
    size = float(np.clip(150.0 / n, 5.0, 14.0))
    groups: dict[str, list] = {"white": [[], [], []], "#111111": [[], [], []]}
    for i, r in enumerate(row_labels):
        for j, c in enumerate(col_labels):
            v = z[i, j]
            if not np.isfinite(v):
                continue
            rgb = scale_color(v, cmap, vmin, vmax)
            red, green, blue = (float(x) for x in rgb[rgb.index("(") + 1:-1].split(","))
            ink = "white" if (0.299 * red + 0.587 * green + 0.114 * blue) < 140 else "#111111"
            xs, ys, ts = groups[ink]
            xs.append(c)
            ys.append(r)
            text = f"{v:{fmt}}"
            ts.append(text.replace("0.", ".") if strip_zero else text)
    for ink, (xs, ys, texts) in groups.items():
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, text=texts, mode="text", hoverinfo="skip", showlegend=False,
            textfont=dict(size=size, color=ink),
        ), row=row, col=col)
