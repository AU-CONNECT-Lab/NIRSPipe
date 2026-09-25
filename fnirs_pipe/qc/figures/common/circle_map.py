"""The circle the inter-brain connectogram is drawn on.

Where a node sits, how a chord between two of them bows, and how a node's label is set
against the ring. The geometry is kept apart from the figure that fills it.

Groups are separated by blank circle rather than by a drawn rule. A dyad circle has two
groups, the two members. **Nodes carry no colour of their own**: the chords need the colour
for their value.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Degrees of blank circle between groups: the two members face each other across the split.
DYAD_GAP = 12.0

NODE_INK = "#8d9aa6"
LABEL_INK = "#2c3e50"


def ring_angles(counts, gap: float = DYAD_GAP, start: float = 90.0) -> np.ndarray:
    """Where each node sits, in degrees, counterclockwise from ``start``.

    ``counts`` is the size of each group in order::

        ring_angles([3, 3], gap=12)  ->  the first group over the left semicircle,
                                         the second coming back up the right

    Counterclockwise from the top, so with two groups the first runs down the left side and
    the second comes back up the right, which puts the two homologous ends of a montage
    facing each other across the split.
    """
    counts = [int(c) for c in counts if int(c) > 0]
    total = sum(counts)
    if not total:
        return np.zeros(0)
    step = (360.0 - len(counts) * gap) / total
    out, k = [], 0
    for g, size in enumerate(counts):
        for _ in range(size):
            out.append(start + gap / 2 + (k + 0.5) * step + g * gap)
            k += 1
    return np.asarray(out, dtype=float)


def bezier(p0, p2, n: int = 40) -> "tuple[np.ndarray, np.ndarray]":
    """A chord from p0 to p2 bowed toward the middle of the circle.

    Quadratic, with the centre as the control point, which is what makes a pairing between
    two nodes that face each other read as a straight line across and one between two
    neighbours as a shallow arc: how far a chord bows is then how far apart its ends are.
    """
    t = np.linspace(0.0, 1.0, n)[:, None]
    return tuple(((1 - t) ** 2 * np.asarray(p0) + t ** 2 * np.asarray(p2)).T)


def node_arc(fig, theta: float, half_width: float, hover: str, row: int, col: int,
             colour: str = NODE_INK, radius: float = 1.0, width: float = 9) -> None:
    """One node, as a thick arc segment of the ring rather than a dot.

    An arc spans the share of the circle the node owns, so a group of them reads as a block.
    """
    a = np.linspace(theta - half_width, theta + half_width, 12)
    fig.add_trace(go.Scatter(
        x=np.cos(a) * radius, y=np.sin(a) * radius, mode="lines",
        line=dict(color=colour, width=width), showlegend=False,
        hovertemplate=f"{hover}<extra></extra>",
    ), row=row, col=col)


def radial_label(fig, theta: float, radius: float, text: str, row: int, col: int,
                 size: float = 9, colour: str = LABEL_INK) -> None:
    """A label set along the radius, flipped on the left half so it is never upside down."""
    deg = np.rad2deg(theta) % 360.0
    flip = 90.0 < deg < 270.0
    fig.add_annotation(
        x=float(np.cos(theta) * radius), y=float(np.sin(theta) * radius),
        text=text, showarrow=False, font=dict(size=size, color=colour),
        textangle=-(deg - 180.0) if flip else -deg,
        xanchor="right" if flip else "left", yanchor="middle",
        xref=f"x{col}" if col > 1 else "x", yref=f"y{col}" if col > 1 else "y",
    )


def circle_axes(fig, row: int, col: int, span: float = 1.16) -> None:
    """A square, invisible frame around the unit circle, with room for the radial labels."""
    n = col if col > 1 else ""
    fig.update_xaxes(visible=False, range=[-span, span], row=row, col=col)
    fig.update_yaxes(visible=False, range=[-span, span], scaleanchor=f"x{n}", scaleratio=1,
                     row=row, col=col)
