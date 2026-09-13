"""Provenance flow diagram: which file each output came from, and via which step."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fnirs_pipe.qc.provenance import Node

# ---- Palette (shared with the interface DAG, fnirs_pipe/interface/pages/analysis.py) ----
# Signal domain -> outline colour. Reading left to right the colour tracks the domain
# change: intensity -> optical density -> haemoglobin -> analysis output. Boxes are drawn
# unfilled, so the outline carries the whole signal, as it does in the interface.
_COLORS = {
    "input":      "#9CA3AF",
    "od":         "#E67E22",
    "haemo":      "#8E44AD",
    "derivative": "#2980B9",
}

# Outline for a node whose sidecar outlived its file. Red is reserved for this, so it is
# never a domain colour: anything red in the diagram is a problem.
_MISSING = "#C0392B"

_CANVAS = "#FAFAFA"
_LABEL_TEXT = "#374151"
_STEP_TEXT = "#6B7280"
_STATE_TEXT = "#9CA3AF"

# Box shape and spacing follow the interface stylesheet, scaled up for three text lines:
# there the node is 140x44 px with 80 px between ranks and 35 px between rows.
_BOX_W, _BOX_H = 2.8, 1.10
_X_GAP, _Y_GAP = 4.4, 1.95
_PAD_X, _PAD_Y = 0.55, 0.55
_SCALE = 0.52          # inches per data unit


def _fit(text: str, base: float, max_chars: int) -> float:
    """Shrink the font rather than let a long line run outside the box.

    Measured on the longest line, so wrapped text is not shrunk for its total length.
    """
    longest = max((len(line) for line in text.split("\n")), default=0)
    return base if longest <= max_chars else base * max_chars / longest


def provenance_figure(nodes: dict[str, Node], title: str | None = None):
    """Layered left-to-right DAG of the run's outputs. Returns a matplotlib Figure.

    Nodes are placed by depth (longest path from a root), so an arrow never points
    backwards. Each box shows the output name, the step and settings that produced it,
    and the shape of the data at that point.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if not nodes:
        return None

    columns: dict[int, list[Node]] = {}
    for node in sorted(nodes.values(), key=lambda n: (n.depth, n.label)):
        columns.setdefault(node.depth, []).append(node)

    pos: dict[str, tuple[float, float]] = {}
    for depth, column in columns.items():
        # centre each column vertically so the trunk of the graph stays level
        offset = (len(column) - 1) / 2
        for i, node in enumerate(column):
            pos[node.key] = (depth * _X_GAP, (offset - i) * _Y_GAP)

    span_x = (max(columns)) * _X_GAP + _BOX_W + 2 * _PAD_X
    span_y = (max(len(c) for c in columns.values()) - 1) * _Y_GAP + _BOX_H + 2 * _PAD_Y
    fig, ax = plt.subplots(figsize=(span_x * _SCALE, span_y * _SCALE + 0.35))
    ax.set_facecolor(_CANVAS)

    for node in nodes.values():
        # a QC record names every stage it measured, so drawing its edges lays the whole
        # chain over itself. It keeps its depth, and its second line says what it measured
        if node.checkpoint:
            continue
        x, y = pos[node.key]
        # the edge takes the colour of the node it feeds, as the interface stylesheet does
        colour = _COLORS.get(node.domain, _COLORS["derivative"])
        for src in node.sources:
            if src not in pos:
                continue
            sx, sy = pos[src]
            ax.annotate(
                "", xy=(x - _BOX_W / 2, y), xytext=(sx + _BOX_W / 2, sy),
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.5,
                                mutation_scale=11, shrinkA=0, shrinkB=0),
            )

    for node in nodes.values():
        x, y = pos[node.key]
        colour = _COLORS.get(node.domain, _COLORS["derivative"])
        # a sidecar whose file was deleted takes a dashed red outline, the one place red
        # appears, so it is not mistaken for output
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - _BOX_W / 2, y - _BOX_H / 2), _BOX_W, _BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor="none", edgecolor=_MISSING if node.missing else colour,
            linestyle="--" if node.missing else "-", linewidth=2.0, zorder=2,
        ))
        # three lines: what it is, how it was made, and the shape of the data left behind
        ax.text(x, y + 0.30, node.label, ha="center", va="center",
                fontsize=_fit(node.label, 9.5, 16), fontweight="bold", color=_LABEL_TEXT,
                zorder=3)
        # the SQM checkpoints are named after their step, so printing it again under the
        # label would spend the line on a word already there
        step = node.step if node.step != node.label else ""
        made = " ".join(p for p in (step, node.detail) if p)
        if made:
            ax.text(x, y + 0.01, made, ha="center", va="center",
                    fontsize=_fit(made, 6.5, 26), color=_STEP_TEXT, zorder=3)
        # the state line is the shape of the data left behind; there is none to describe
        note = "file missing" if node.missing else node.state
        if note:
            ax.text(x, y - 0.28, note, ha="center", va="center",
                    fontsize=_fit(note, 6.0, 28),
                    color=_MISSING if node.missing else _STATE_TEXT, zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - _BOX_W / 2 - _PAD_X, max(xs) + _BOX_W / 2 + _PAD_X)
    ax.set_ylim(min(ys) - _BOX_H / 2 - _PAD_Y, max(ys) + _BOX_H / 2 + _PAD_Y)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold", color=_LABEL_TEXT, pad=8)

    fig.tight_layout()
    return fig
