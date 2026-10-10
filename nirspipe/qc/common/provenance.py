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
import copy

from nirspipe.io.derivatives import entity_of

_ENTITY_RE = re.compile(r"^([a-z]+)-([A-Za-z0-9]+)$")

# entities that say whose file it is, which every node of one graph shares
_IDENTITY = frozenset({"sub", "ses", "task", "acq", "run", "group"})

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
    """Short display name: the suffix and the entity values that set this file apart.

    ::

      sub-01_task-rest_desc-preproc_nirs                   -> "preproc"
      sub-01_task-rest_chromo-hbo_stat-pearson_relmat      -> "relmat pearson hbo"
      group-G1_task-rest_chromo-hbo_cond-game1_stat-isc_relmat -> "relmat isc hbo (game1)"
      sub-01_task-rest_design                              -> "design"
      design_matrix                                        -> "design_matrix"

    On the generic ``nirs`` suffix the desc is the name, the stage the signal is at. The
    measure comes before the slices so a collapsed box's shared prefix names it. The
    condition goes in parentheses and nowhere else, because :func:`_variant_line` reads a
    label's head as what the siblings of a collapsed box differ by and counts the
    conditions apart from it.
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
    cond = entities.pop("cond", None)
    qualifiers = [v for k, v in sorted(entities.items(), key=lambda kv: kv[0] not in
                                       ("stat", "desc")) if k not in _IDENTITY]
    if suffix in ("", "nirs"):
        head = entities.get("desc") or suffix or key
    else:
        head = " ".join([suffix, *qualifiers])
    return f"{head} ({cond})" if cond else head


def _step_detail(step: str | None, params: dict[str, Any]) -> str:
    """The settings a step actually used, short enough to sit under the node label.

    bandpass     + {high_pass: 0.01, low_pass: 0.5} -> "0.01-0.5 Hz"
    beer_lambert + {dpf: [6.0]}                     -> "dpf 6.0"
    sci_pruning  + {sci_threshold: 0.75}            -> "SCI 0.75 + PSP 0.1, coupled ≥ 0.75"
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
        # the two lines a window clears together and the share of windows that keeps a
        # channel: the share rejects, so the SCI line alone would name the wrong rule
        if pick("sci_threshold") is None:
            return ""
        # local: qc.metrics pulls in mne, and the GUI imports this module at start-up
        from nirspipe.qc.metrics import resolve_cutoffs
        lines = resolve_cutoffs(sci=pick("sci_threshold"), psp=pick("psp_threshold"),
                                good_frac=pick("min_good_frac"))
        return f"SCI {lines['sci']:g} + PSP {lines['psp']:g}, coupled ≥ {lines['good_frac']:g}"

    if step == "motion_correction":
        return str(pick("motion_correction") or "")

    if step in ("sqm", "sqm_raw"):
        # the record covers every stage and every metric family, so listing either fills the
        # box with a paragraph. The count on the state line says how much of it there is
        return "quantitative QC metrics"

    if step in ("glm_residuals", "glm_fit", "design_matrix"):
        # an HRF with no conditions to convolve is a config default, not a setting the step used
        bits = [None if params.get("conditions") == [] else pick("hrf_model"), pick("noise_model")]
        if (drift := pick("drift_model")) is not None:
            bits.append(f"{drift} drift")
        # the confounds regressed out, which in denoise mode are the whole of the step
        if (short := pick("short_channel")) not in (None, "none"):
            bits.append(f"short {short}")
        if aux := pick("aux_regressors"):
            bits.append(f"{len(aux)} aux")
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
            detail=_step_detail(meta.get("step"), params),
            state=_state_line(data),
            missing=_file_is_gone(sidecar, data),
            checkpoint=_is_checkpoint(data),
        )

    # Sources may name files outside nirs_dir (the BIDS input); add them as roots.
    for node in list(nodes.values()):
        for src in node.sources:
            if src not in nodes:
                # a root has no sidecar to describe it, so the box would carry a bare name
                nodes[src] = Node(key=src, label=_label(src), detail="raw data")

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


# ---- Drawing the graph: collapsing repeats ----

# A step that produced this many outputs from one set of inputs is drawn as one box; fewer
# are drawn side by side.
_COLLAPSE_MIN = 3


def _common_prefix(labels: list[str]) -> str:
    """The name a set of sibling labels shares, cut at a separator.

    ::

      ["relmat isc hbo", "relmat isc hbo (baseline)", "relmat isc hbr"] -> "relmat isc"

    A raw character-wise prefix of that set is ``relmat isc hb``, half a word, so where the
    labels carry on past the prefix it backs off to the last separator inside it.
    """
    prefix = labels[0]
    for label in labels[1:]:
        while not label.startswith(prefix):
            prefix = prefix[:-1]
    if any(len(lab) > len(prefix) and lab[len(prefix)] not in " -_(" for lab in labels):
        prefix = re.split(r"[ \-_(](?!.*[ \-_(])", prefix)[0]
    return prefix.rstrip(" -_(")


def _variant_line(members: list[Node], prefix: str) -> str:
    """What the members of a collapsed box differ by, on one line.

    ::

      12 relmat isc outputs -> "hbo, hbr  ·  whole run + 5 conditions"

    Two axes, because those are the two a run repeats a step over: what is left of each
    label once the shared prefix is off, and the ``cond-`` entity. A member without one is
    the whole run, so it is named rather than counted with the windows.
    """
    suffixes: list[str] = []
    descs: list[str] = []
    whole_run = False
    for node in members:
        tail = node.label.split(" (")[0]
        tail = tail[len(prefix):].lstrip(" -_") or tail
        if tail not in suffixes:
            suffixes.append(tail)
        cond = entity_of(node.key, "cond")
        if cond is None:
            whole_run = True
        elif cond not in descs:
            descs.append(cond)

    bits = [", ".join(suffixes)]
    if descs:
        windows = f"{len(descs)} condition{'s' if len(descs) > 1 else ''}"
        bits.append(f"whole run + {windows}" if whole_run else windows)
    return "  ·  ".join(b for b in bits if b)


def _qualify_roots(nodes: dict[str, Node]) -> None:
    """Add the subject to a root's label where two roots would otherwise read the same.

    A hyperscanning graph has one input per member and both are ``errts``, so unqualified
    the two boxes are indistinguishable and the arrows out of them say nothing. A
    single-subject graph has no such collision and keeps the bare name.
    """
    roots = [n for n in nodes.values() if n.step is None]
    seen: dict[str, int] = {}
    for node in roots:
        seen[node.label] = seen.get(node.label, 0) + 1
    for node in roots:
        if seen[node.label] > 1 and (sub := entity_of(node.key, "sub")):
            node.label = f"{node.label} ({sub})"


def simplify(nodes: dict[str, Node]) -> dict[str, Node]:
    """The graph as it is drawn: repeats collapsed, ambiguous roots named.

    ::

      2 inputs + 12 relmat isc + 4 relmat wtc -> 2 inputs + 1 relmat isc box + 4

    Siblings are collapsed when they share a step *and* a source set, which is what makes
    the merged box's arrows the members' own arrows rather than an approximation of them.
    Anything downstream of a member is repointed at the box, so no edge is left dangling.

    A drawing step, not a reading one: :func:`scan` still returns every node, which is what
    the interface DAG and the methods text are built on.
    """
    groups: dict[tuple, list[Node]] = {}
    for node in nodes.values():
        if node.step is None:
            continue
        groups.setdefault((node.step, tuple(sorted(node.sources)), node.missing),
                          []).append(node)

    out: dict[str, Node] = {}
    merged_into: dict[str, str] = {}
    for (step, sources, missing), members in groups.items():
        if len(members) < _COLLAPSE_MIN:
            continue
        members.sort(key=lambda n: n.label)
        prefix = _common_prefix([n.label for n in members]) or step.replace("_", "-")
        key = f"{step}__collapsed"
        out[key] = Node(
            key=key, label=prefix, step=step, sources=list(sources),
            detail=members[0].detail, state=_variant_line(members, prefix),
            missing=missing, domain=members[0].domain, depth=members[0].depth,
            checkpoint=members[0].checkpoint,
        )
        for node in members:
            merged_into[node.key] = key

    for key, node in nodes.items():
        if key in merged_into:
            continue
        node = copy.copy(node)
        node.sources = sorted({merged_into.get(s, s) for s in node.sources})
        out[key] = node

    _qualify_roots(out)
    return out




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
