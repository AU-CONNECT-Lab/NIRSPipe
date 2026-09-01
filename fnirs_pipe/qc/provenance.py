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
    data: dict[str, Any] = field(default_factory=dict)  # as recorded, before _state_line
    sources: list[str] = field(default_factory=list)   # keys of parent nodes
    domain: str = "derivative"
    depth: int = 0
    detail: str = ""                           # settings the step used, e.g. "0.01-0.5 Hz"
    state: str = ""                            # shape of the data here, e.g. "40/56 ch · 2 Hz"
    missing: bool = False                      # the sidecar is here, the file it describes is not
    checkpoint: bool = False                   # a QC record, measured off the chain rather than on it


def _key(path: str | Path) -> str:
    return Path(path).name.split(".")[0]


def _is_checkpoint(data: dict[str, Any]) -> bool:
    """Whether this sidecar describes a QC record rather than a signal file.

    A QC record measures several stages and holds the numbers itself, so it is neither
    produced by one parent nor read by anything downstream. ``n_metrics`` marks it; the
    same field tells `_state_line` to print a metric count instead of a data shape.
    """
    return data.get("n_metrics") is not None


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


# Metric family shown on an SQM node. A key whose leading token says nothing on its own
# gets an explicit name here; every other key falls back to the token before the first "_".
_SQM_FAMILIES = {
    "channel_retention": "retention",
    "pct_data": "retention",
    "ch_dist": "distance",
    "mean_amp": "amplitude",
    "hbo_hbr": "hbo/hbr",
}


def _wrap_terms(terms: list[str], width: int = 24, max_lines: int = 3) -> str:
    """Join terms onto at most max_lines lines of width, eliding the rest with an ellipsis."""
    import textwrap

    lines = textwrap.wrap(" ".join(terms), width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " …"
    return "\n".join(lines)


def _sqm_detail(metrics: list[str], width: int = 24, max_lines: int = 3) -> str:
    """The metric families a checkpoint computed, wrapped to fit inside a node box.

    ["sci_mean", "ch_dist_min", "gvtd_p95"] -> "sci distance gvtd"

    The full key list stays in the sidecar; only the families are drawn, or a 27-metric
    checkpoint would need a paragraph.
    """
    families: list[str] = []
    for key in metrics:
        name = next((v for k, v in _SQM_FAMILIES.items() if key.startswith(k)),
                    key.split("_")[0])
        if name not in families:
            families.append(name)

    return _wrap_terms(families, width, max_lines)


def _step_detail(step: str | None, params: dict[str, Any], data: dict[str, Any] | None = None) -> str:
    """The settings a step actually used, short enough to sit under the node label.

    bandpass     + {high_pass: 0.01, low_pass: 0.5} -> "0.01-0.5 Hz"
    beer_lambert + {dpf: [6.0]}                     -> "dpf 6.0"
    unknown step, or the keys are absent            -> ""

    Sidecars carry the whole config, so each step names only the keys that describe it.
    """
    def pick(*keys: str) -> Any:
        for k in keys:
            if (v := params.get(k)) is not None:
                return v
        return None

    if step == "bandpass":
        lo, hi = pick("high_pass", "l_freq"), pick("low_pass", "h_freq")
        if lo is not None and hi is not None:
            return f"{lo:g}-{hi:g} Hz"
        if hi is not None:
            return f"<{hi:g} Hz"
        return f">{lo:g} Hz" if lo is not None else ""

    if step in ("resample", "glm_residuals_broadband"):
        v = pick("sfreq", "resample_sfreq")
        return f"{v:g} Hz" if v is not None else ""

    if step == "beer_lambert":
        dpf = pick("dpf")
        if isinstance(dpf, (list, tuple)):
            dpf = ", ".join(f"{d:g}" for d in dpf)
        return f"dpf {dpf}" if dpf is not None else ""

    if step == "sci_pruning":
        thr = pick("sci_threshold")
        return f"thr {thr:g}" if thr is not None else ""

    if step == "motion_correction":
        return str(pick("motion_correction") or "")

    if step in ("sqm", "sqm_raw"):
        # the sectioned record names the stages it measured; the older per-checkpoint
        # sidecars name the metrics they computed
        data = data or {}
        if data.get("sections"):
            return _wrap_terms(list(data["sections"]))
        return _sqm_detail(data.get("metrics") or [])

    if step in ("glm_residuals", "glm_fit", "design_matrix"):
        bits = [pick("hrf_model"), pick("noise_model")]
        if (drift := pick("drift_model")) is not None:
            bits.append(f"{drift} drift")
        return " / ".join(str(b) for b in bits if b)

    return ""


def _state_line(data: dict[str, Any]) -> str:
    """Shape of the data at this node: "40/56 ch · 2 Hz · 595 s". Empty for older sidecars."""
    if not data:
        return ""
    if (n_metrics := data.get("n_metrics")) is not None:
        return f"{n_metrics} metrics"          # a QC checkpoint holds numbers, not signal
    bits: list[str] = []
    n, bad = data.get("n_channels"), data.get("n_bad") or 0
    if n is not None:
        bits.append(f"{n - bad}/{n} ch" if bad else f"{n} ch")
    if (sfreq := data.get("sfreq")) is not None:
        bits.append(f"{sfreq:g} Hz")
    if (dur := data.get("duration_s")) is not None:
        bits.append(f"{dur:g} s")
    return " · ".join(bits)


def _domain_of(label: str, is_root: bool) -> str:
    if is_root:
        return "input"
    return _DOMAIN.get(label, "derivative")


def _file_is_gone(sidecar: Path, data: dict[str, Any]) -> bool:
    """Whether the file this sidecar describes has been deleted out from under it.

    Nothing in the package removes a sidecar when its output is deleted, and switching a
    tree from one mode to another leaves the old mode's sidecars behind. Reading them as
    ordinary nodes makes the graph claim steps the run did not perform, so they are found
    here and drawn as missing rather than silently believed.

    The SQM record is the exception and is never missing: it is a JSON that holds the
    numbers itself, so it has no companion by design. ``n_metrics`` is what marks it, the
    same field `_state_line` uses to tell a QC checkpoint from a signal file.
    """
    if _is_checkpoint(data):
        return False
    stem = sidecar.name[: -len(".json")]
    return not any(p.suffix != ".json" for p in sidecar.parent.glob(f"{stem}.*"))


def scan(nirs_dir: Path, label: str | None = None) -> dict[str, Node]:
    """Read the sidecars under nirs_dir and return the graph keyed by filename stem.

    ``label`` is a BIDS run stem (``sub-01_task-rest``). Without it every run of the subject
    lands in one graph, which for a five-task subject is five disjoint chains of identically
    named nodes drawn on top of each other. With it the graph is that run alone.
    """
    nodes: dict[str, Node] = {}

    pattern = f"{label}_*.json" if label else "*.json"
    for sidecar in sorted(Path(nirs_dir).glob(pattern)):
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            continue
        if "step" not in meta:                 # not a provenance sidecar
            continue
        key = _key(sidecar)
        params = meta.get("parameters") or {}
        data = meta.get("data") or {}
        nodes[key] = Node(
            key=key,
            label=_label(key),
            step=meta.get("step"),
            params=params,
            data=data,
            sources=[_key(s) for s in (meta.get("Sources") or [])],
            detail=_step_detail(meta.get("step"), params, data),
            state=_state_line(data),
            missing=_file_is_gone(sidecar, data),
            checkpoint=_is_checkpoint(data),
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
    label: str | None = None,
) -> list[Path]:
    """Render the graph for nirs_dir into out_dir as <stem>.png and <stem>.mmd.

    ``label`` restricts the graph to one BIDS run; see :func:`scan`.

    Returns the files written, empty if nirs_dir holds no provenance sidecars.
    """
    from fnirs_pipe.qc.figures.provenance_figure import provenance_figure

    nodes = scan(nirs_dir, label=label)
    if not nodes:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig = provenance_figure(nodes, title=title)
    if fig is not None:
        png = out_dir / f"{stem}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight")
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
    lines = ["flowchart LR", "    classDef missing stroke:#b03a2e,stroke-dasharray: 4 3"]
    for n in ordered:
        note = "file missing" if n.missing else n.state
        text = f"{n.label}<br/><small>{note}</small>" if note else n.label
        lines.append(f'    {ident(n.key)}["{text}"]' + (":::missing" if n.missing else ""))
    for node in ordered:
        # a QC record names every stage it measured, which would put an edge from the whole
        # chain onto one node and bury the chain itself. Its own text says what it measured
        if node.checkpoint:
            continue
        for src in node.sources:
            # a wrapped detail is drawn on several lines in the PNG; a mermaid edge label
            # is one line, and a raw newline would break the arrow
            edge = " ".join(p for p in (node.step, node.detail.replace("\n", " ")) if p)
            arrow = f"-- {edge} -->" if edge else "-->"
            lines.append(f"    {ident(src)} {arrow} {ident(node.key)}")
    return "\n".join(lines)
