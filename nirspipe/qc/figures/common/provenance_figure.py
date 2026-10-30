"""Provenance flow diagram: which file each output came from, and via which step."""

from __future__ import annotations

import math
from pathlib import Path

from nirspipe.qc.common.provenance import Node, scan, simplify, to_mermaid

# ---- Palette (shared with the interface DAG, nirspipe/interface/pages/analysis.py) ----
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
_X_GAP, _Y_GAP = 3.9, 1.95
_PAD_X, _PAD_Y = 0.55, 0.55
_SCALE = 0.52          # inches per data unit
# A graph taller than it is wide wraps each long column into a grid of this many rows
_MAX_ROWS = 5
_SUB_GAP = 3.4         # between the sub-columns of a wrapped column
# drawn at this resolution and shown at 110 px per inch (srcset 2.727x in _footer.html.j2)
_DPI = 300


def _fit(text: str, base: float, max_chars: int) -> float:
    """Shrink the font rather than let a long line run outside the box.

    Measured on the longest line, so wrapped text is not shrunk for its total length.
    """
    longest = max((len(line) for line in text.split("\n")), default=0)
    return base if longest <= max_chars else base * max_chars / longest


def _by_parents(column: list[Node]) -> list[list[Node]]:
    """The column's nodes grouped by their parents, each group where its first node stood."""
    groups: dict[tuple, list[Node]] = {}
    for node in column:
        groups.setdefault(tuple(sorted(node.sources)), []).append(node)
    return list(groups.values())


def _layout(columns: dict[int, list[Node]]) -> tuple[dict, list[list[str]]]:
    """Each node's (x, y), and the blocks of the columns that wrapped, as lists of keys.

    ::

      3 inputs into 20 outputs (taller than wide)  ->  the 20 in a grid of 4 x 5
      a nine-step chain (wider than tall)          ->  one row, as before

    Only a tall graph wraps: a wrapped column adds width, which a wide graph already
    squeezed to the page cannot spare. A wrapped column is one block per set of parents,
    each starting a row of its own, so no arrow into one block crosses another.
    """
    width = max(columns) * _X_GAP + _BOX_W
    height = (max(len(c) for c in columns.values()) - 1) * _Y_GAP + _BOX_H
    pos: dict[str, tuple[float, float]] = {}
    blocks: list[list[str]] = []
    x = 0.0
    for depth in sorted(columns):
        column = columns[depth]
        k = math.ceil(len(column) / _MAX_ROWS) if height > width else 1
        groups = _by_parents(column) if k > 1 else [column]
        rows = sum(math.ceil(len(group) / k) for group in groups)
        first = 0
        for group in groups:
            for i, node in enumerate(group):
                # a row at a time, so a short last row sits at the left like text; centred
                # vertically so the trunk of the graph stays level
                row, j = divmod(i, k)
                pos[node.key] = (x + j * _SUB_GAP, ((rows - 1) / 2 - first - row) * _Y_GAP)
            first += math.ceil(len(group) / k)
            if k > 1:
                blocks.append([n.key for n in group])
        x += _X_GAP + (k - 1) * _SUB_GAP
    return pos, blocks


def provenance_figure(nodes: dict[str, Node], title: str | None = None):
    """Layered left-to-right DAG of the run's outputs. Returns a matplotlib Figure.

    Nodes are placed by depth (longest path from a root), so an arrow never points
    backwards. Each box shows the output name, the step and settings that produced it,
    and the shape of the data at that point. A wrapped column (see :func:`_layout`) whose
    outputs share one set of parents takes one arrow per parent, into a bar down its left
    edge, rather than one into every box.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if not nodes:
        return None

    columns: dict[int, list[Node]] = {}
    last = max(n.depth for n in nodes.values())
    # a QC record goes last in its column: it draws no edge and joins no bundle. An output
    # whose sidecar names no parents joins the outputs rather than standing among the inputs
    def column_of(node: Node) -> int:
        return last if node.step is not None and not node.sources else node.depth

    for node in sorted(nodes.values(),
                       key=lambda n: (column_of(n), not n.sources, n.checkpoint, n.label)):
        columns.setdefault(column_of(node), []).append(node)

    pos, blocks = _layout(columns)
    in_block = {key: i for i, block in enumerate(blocks) for key in block}

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    span_x = max(xs) - min(xs) + _BOX_W + 2 * _PAD_X
    span_y = max(ys) - min(ys) + _BOX_H + 2 * _PAD_Y
    fig, ax = plt.subplots(figsize=(span_x * _SCALE, span_y * _SCALE + 0.35))
    ax.set_facecolor(_CANVAS)

    bundled: set = set()
    for node in nodes.values():
        # a QC record names every stage it measured, so drawing its edges lays the whole
        # chain over itself. It keeps its depth, and its second line says what it measured
        if node.checkpoint:
            continue
        x, y = pos[node.key]
        # the edge takes the colour of the node it feeds, as the interface stylesheet does
        colour = _COLORS.get(node.domain, _COLORS["derivative"])
        block = in_block.get(node.key)
        fed = ([k for k in blocks[block] if not nodes[k].checkpoint]
               if block is not None else [])
        # a block of one box takes its arrows straight in
        shared = len(fed) > 1 and len({tuple(sorted(nodes[k].sources)) for k in fed}) == 1
        for src in node.sources:
            if src not in pos:
                continue
            sx, sy = pos[src]
            end = (x - _BOX_W / 2, y)
            if shared:
                if (src, block) in bundled:
                    continue
                bundled.add((src, block))
                bar_x = min(pos[k][0] for k in fed) - _BOX_W / 2 - 0.35
                top = max(pos[k][1] for k in fed) + _BOX_H / 2
                bottom = min(pos[k][1] for k in fed) - _BOX_H / 2
                ax.plot([bar_x, bar_x], [bottom, top], color=colour, lw=2.0, zorder=1)
                end = (bar_x, min(max(sy, bottom), top))
            ax.annotate(
                "", xy=end, xytext=(sx + _BOX_W / 2, sy),
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


# ---- Writing the graph out ----
# Here, not beside `scan` and `simplify`: those build the graph and know nothing about
# pictures. This draws it and saves both outputs.

def provenance_title(label: str, mode: "str | None") -> str:
    return label + (f"  |  mode: {mode}" if mode else "")


def write_provenance(
    nirs_dir: Path,
    out_dir: Path,
    fig_name,
    title: str | None = None,
    label: str | None = None,
) -> list[Path]:
    """Render the graph for nirs_dir into out_dir as a PNG and its mermaid source.

    ``fig_name`` is a :func:`~nirspipe.qc.common.figure_io.figure_namer`, so both files
    are named the way every other figure in the tree is and carry the run they are of.

    ``label`` restricts the graph to one BIDS run; see :func:`scan`.

    Both outputs are drawn off :func:`simplify`, so the picture and the mermaid source are
    the same graph.

    Returns the files written, empty if nirs_dir holds no provenance sidecars.
    """
    nodes = scan(nirs_dir, label=label)
    if not nodes:
        return []
    nodes = simplify(nodes)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = provenance_figure(nodes, title=title)
    if fig is not None:
        png = out_dir / fig_name("provenance", extension=".png")
        fig.savefig(png, dpi=_DPI, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        written.append(png)

    mmd = out_dir / fig_name("provenance", extension=".mmd")
    mmd.write_text(to_mermaid(nodes), encoding="utf-8")
    written.append(mmd)
    return written
