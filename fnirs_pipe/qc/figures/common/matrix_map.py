"""The conventions every matrix panel in the reports is drawn with.

Which colour scale a quantity takes, what a cell with no value looks like, and how a cell's
number is printed on top of it. Three figures used to carry their own copy of the
correlation scale alone, so a red cell could have meant +1 in one panel and -1 in the next.

Layout stays with each figure: a lower-triangle channel matrix with separation dividers, a
pair of square per-chromophore matrices and a cross-brain grid are different pictures. What
they must not disagree about is what a colour means.
"""

from __future__ import annotations

import numpy as np

# Diverging and centred on zero, red at +1, because a correlation has a meaningful middle.
# An unsigned bounded quantity such as a coherence has none and takes a sequential scale.
CORRELATION_SCALE = "RdBu_r"
COHERENCE_SCALE = "viridis"

# A cell no value was computed for: grey, because it is not a value and no point on the
# scale should be able to stand for it. Light enough not to read as a value of its own, and
# still clear of the palest end of either scale, which is very nearly white.
BLANK_CELL = "#d5d5d5"


def scale_color(value: float, cmap: str, vmin: float, vmax: float) -> str:
    """The colour a scale gives one value, as ``rgb(r,g,b)``.

    ``scale_color(0.0, "RdBu_r", -1, 1) -> the scale's midpoint colour``

    For anything that has to match a heatmap's fill without being one: the ink a printed
    cell value takes, or a chord coloured by the same quantity the matrix beside it is.
    """
    from plotly.colors import sample_colorscale

    t = (float(value) - vmin) / ((vmax - vmin) or 1.0)
    return sample_colorscale(cmap, [float(np.clip(t, 0.0, 1.0))])[0]


def matrix_ground(fig, n_rows: int, n_cols: int, row: int, col: int,
                  color: str = BLANK_CELL, triangle: bool = False) -> None:
    """The blank field a matrix is painted over, so an absent cell is grey and not the page.

    Painted under the heatmap rather than left to show the panel through. No gap between
    cells, since a border in that colour would draw a missing cell where there is none.

    ``triangle`` grounds the lower half alone, for a symmetric matrix drawn as a triangle: a
    full square would ground the half that was masked on purpose and it would read as a
    montage of missing cells.
    """
    if not triangle:
        fig.add_shape(type="rect", x0=-0.5, x1=n_cols - 0.5, y0=-0.5, y1=n_rows - 0.5,
                      fillcolor=color, line=dict(width=0), layer="below", row=row, col=col)
        return
    # the staircase the kept cells make, as one closed path
    steps = []
    for i in range(n_rows):
        steps.append(f"L{i + 0.5},{i - 0.5}")
        steps.append(f"L{i + 0.5},{i + 0.5}")
    fig.add_shape(type="path", layer="below", row=row, col=col,
                  path=f"M-0.5,-0.5 {' '.join(steps)} L-0.5,{n_rows - 0.5} Z",
                  fillcolor=color, line=dict(width=0))


def cell_values(fig, z, row_labels, col_labels, *, cmap, vmin, vmax, row, col,
                fmt: str = ".2f", strip_zero: bool = True) -> None:
    """Print every cell's value on the heatmap, in ink the cell's own colour can carry.

    ::

      a cell at .69 -> white text;  the same number at .12 -> black

    **Every cell gets its number.** A fixed colour scale is what lets two panels be compared
    by eye, and the cost is that a matrix whose values all sit near 0.25 renders as one flat
    square; the printed value is what makes such a matrix readable at all.

    The ink is chosen from **the cell's own colour** and not from its value: a sequential
    scale is pale at its low end and dark at its high one while a diverging one is dark at
    both ends and pale in the middle, so any rule written against the value serves one of
    them and fails the other. Asking the scale and taking the luminance serves both, and any
    scale added later. Two text traces rather than one, because a heatmap takes a single
    ``textfont`` for the whole grid and that is the one thing this needs per cell.
    """
    import plotly.graph_objects as go

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
