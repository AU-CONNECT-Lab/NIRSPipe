"""Rebuild the file-level provenance graph from the JSON sidecars on disk.

The sidecars are the manifest: each one names the step that produced its file and
the files it came from, so the graph is recovered from the run's own output rather
than from anything held in memory. Any past run can be re-read.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ENTITY_RE = re.compile(r"^(sub|ses|task|run|acq|desc|group)-(.+)$")

# Signal domain per node, used to colour the diagram.
_DOMAIN = {
    "od": "od", "sci": "od", "motcorrected": "od",
    "preproc": "haemo", "filtered": "haemo", "resampled": "haemo",
    "errts": "haemo", "errtsbroad": "haemo",
}


@dataclass
class Node:
    key: str                                   # filename without extension
    label: str                                 # short name shown in the diagram
    step: str | None = None                    # transformation that produced it
    params: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)   # keys of parent nodes
    domain: str = "derivative"
    depth: int = 0


def _key(path: str | Path) -> str:
    return Path(path).name.split(".")[0]


def _label(key: str) -> str:
    """Short display name: the BIDS suffix, qualified by desc- when both carry meaning.

    sub-01_task-tapping_desc-preproc_nirs -> "preproc"   (generic 'nirs' suffix, desc wins)
    sub-01_task-tapping_desc-hbo_fc       -> "fc (hbo)"  (both informative)
    sub-01_task-tapping_alff              -> "alff"      (no desc)
    sub-01_task-tapping_design_matrix     -> "design_matrix"  (multi-token suffix)
    design_matrix                         -> "design_matrix"  (no BIDS entities at all)
    """
    entities: dict[str, str] = {}
    rest: list[str] = []
    for token in key.split("_"):
        if m := _ENTITY_RE.match(token):
            entities[m.group(1)] = m.group(2)
        else:
            rest.append(token)

    if not entities:
        return key
    suffix = "_".join(rest)
    desc = entities.get("desc")
    if suffix in ("", "nirs"):
        return desc or suffix or key
    return f"{suffix} ({desc})" if desc else suffix


def _domain_of(label: str, is_root: bool) -> str:
    if is_root:
        return "input"
    return _DOMAIN.get(label, "derivative")


def scan(nirs_dir: Path) -> dict[str, Node]:
    """Read every sidecar under nirs_dir and return the graph keyed by filename stem."""
    nodes: dict[str, Node] = {}

    for sidecar in sorted(Path(nirs_dir).glob("*.json")):
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            continue
        if "step" not in meta:                 # not a provenance sidecar
            continue
        key = _key(sidecar)
        nodes[key] = Node(
            key=key,
            label=_label(key),
            step=meta.get("step"),
            params=meta.get("parameters") or {},
            sources=[_key(s) for s in (meta.get("Sources") or [])],
        )

    # Sources may name files outside nirs_dir (the BIDS input); add them as roots.
    for node in list(nodes.values()):
        for src in node.sources:
            if src not in nodes:
                nodes[src] = Node(key=src, label=_label(src))

    for node in nodes.values():
        node.domain = _domain_of(node.label, is_root=not node.sources and node.step is None)

    _assign_depth(nodes)
    return nodes


def _assign_depth(nodes: dict[str, Node]) -> None:
    """Longest path from a root, so every node sits to the right of all its parents."""
    resolved: dict[str, int] = {}

    def depth(key: str, seen: frozenset[str] = frozenset()) -> int:
        if key in resolved:
            return resolved[key]
        node = nodes.get(key)
        parents = [s for s in (node.sources if node else []) if s in nodes and s not in seen]
        d = 1 + max((depth(p, seen | {key}) for p in parents), default=-1)
        resolved[key] = d
        return d

    for key in nodes:
        nodes[key].depth = depth(key)


def write_provenance(
    nirs_dir: Path,
    out_dir: Path,
    stem: str,
    title: str | None = None,
) -> list[Path]:
    """Render the graph for nirs_dir into out_dir as <stem>.png and <stem>.mmd.

    Returns the files written, empty if nirs_dir holds no provenance sidecars.
    """
    from fnirs_pipe.qc.figures.provenance_figure import provenance_figure

    nodes = scan(nirs_dir)
    if not nodes:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = provenance_figure(nodes, title=title)
    if fig is not None:
        png = out_dir / f"{stem}.png"
        fig.savefig(png, dpi=200, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        written.append(png)

    mmd = out_dir / f"{stem}.mmd"
    mmd.write_text(to_mermaid(nodes), encoding="utf-8")
    written.append(mmd)
    return written


def to_mermaid(nodes: dict[str, Node]) -> str:
    """Render the graph as a mermaid flowchart, for docs and markdown viewers."""
    def ident(key: str) -> str:
        return re.sub(r"\W", "_", key)      # mermaid ids allow no dashes or dots

    ordered = sorted(nodes.values(), key=lambda n: (n.depth, n.label))
    lines = ["flowchart LR"]
    lines += [f'    {ident(n.key)}["{n.label}"]' for n in ordered]
    for node in ordered:
        for src in node.sources:
            arrow = f"-- {node.step} -->" if node.step else "-->"
            lines.append(f"    {ident(src)} {arrow} {ident(node.key)}")
    return "\n".join(lines)
