"""Provenance flow diagram: which file each output came from, and via which step."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fnirs_pipe.qc.provenance import Node

# Signal domain -> (fill, edge). Reading left to right the colour tracks the domain
# change: intensity -> optical density -> haemoglobin -> analysis output.
_COLORS = {
    "input":      ("#ecf0f1", "#7f8c8d"),
    "od":         ("#fdebd0", "#e67e22"),
    "haemo":      ("#fadbd8", "#c0392b"),
    "derivative": ("#d6eaf8", "#2874a6"),
}

_BOX_W, _BOX_H = 2.4, 1.02
_X_GAP, _Y_GAP = 3.6, 1.5


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

    n_cols = max(columns) + 1
    height = max(len(c) for c in columns.values())
    fig, ax = plt.subplots(figsize=(2.0 + n_cols * 1.55, 1.4 + height * 1.05))

    for node in nodes.values():
        x, y = pos[node.key]
        for src in node.sources:
            if src not in pos:
                continue
            sx, sy = pos[src]
            ax.annotate(
                "", xy=(x - _BOX_W / 2, y), xytext=(sx + _BOX_W / 2, sy),
                arrowprops=dict(arrowstyle="-|>", color="#95a5a6", lw=1.2,
                                shrinkA=0, shrinkB=0,
                                connectionstyle="arc3,rad=0.06"),
            )

    for node in nodes.values():
        x, y = pos[node.key]
        fill, edge = _COLORS.get(node.domain, _COLORS["derivative"])
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - _BOX_W / 2, y - _BOX_H / 2), _BOX_W, _BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=2,
        ))
        # three lines: what it is, how it was made, and the shape of the data left behind
        ax.text(x, y + 0.28, node.label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="#2c3e50", zorder=3)
        # the SQM checkpoints are named after their step, so printing it again under the
        # label would spend the line on a word already there
        step = node.step if node.step != node.label else ""
        made = " ".join(p for p in (step, node.detail) if p)
        if made:
            ax.text(x, y + 0.02, made, ha="center", va="center",
                    fontsize=_fit(made, 6.5, 24), color="#7f8c8d", zorder=3)
        if node.state:
            ax.text(x, y - 0.26, node.state, ha="center", va="center",
                    fontsize=_fit(node.state, 6.0, 26), color="#95a5a6", zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - _BOX_W, max(xs) + _BOX_W)
    ax.set_ylim(min(ys) - _BOX_H, max(ys) + _BOX_H)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, color="#2c3e50", pad=8)

    ax.legend(
        handles=[mpatches.Patch(facecolor=f, edgecolor=e, label=name)
                 for name, (f, e) in _COLORS.items()],
        loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4,
        frameon=False, fontsize=7,
    )
    fig.tight_layout()
    return fig
