"""Every derivative name this package writes, built from and parsed against one table.

The table is ``fnirs_pipe/data/fnirs_pipe_bids_config.json``: the entities this package adds
to the ones pybids already knows, and the path patterns each kind of output follows. Names
used to be assembled by hand in forty-odd modules, which is how five orthogonal dimensions
ended up hyphen-stacked into a single token that only a regex could take apart again.

Two functions, and they are inverses:

    derivative_path(out, "relmat", ".tsv", group="G1", task="rest", statistic="wtc")
        -> out/group-G1/nirs/group-G1_task-rest_stat-wtc_relmat.tsv
    parse_path(that)
        -> {"group": "G1", "task": "rest", "statistic": "wtc", ...}

:func:`derivative_path` does not create directories. The caller decides when a name becomes
a file, which is what lets a dry run print names without leaving a tree behind.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_CONFIG_FILE = Path(__file__).parent.parent / "data" / "fnirs_pipe_bids_config.json"


@lru_cache(maxsize=1)
def config():
    """The merged pybids ``Config``: its own entities, its derivatives ones, then ours.

    Ours go last so a name we declare would win, though none currently collides: pybids
    already knows ``seg-``, ``label-`` and ``desc-``, and its ``datatype`` pattern already
    matches ``nirs/``. Redeclaring one is not merely redundant, it breaks BIDSLayout, whose
    entity table refuses two rows of the same name.
    """
    from bids.layout import Config

    spec = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    merged: dict[str, str] = {}
    for builtin in ("bids", "derivatives"):
        merged.update({k: v.pattern for k, v in Config.load(builtin).entities.items()})
    merged.update({e["name"]: e["pattern"] for e in spec["entities"]})
    return Config(
        name=spec["name"],
        entities=[{"name": name, "pattern": pattern} for name, pattern in merged.items()],
        default_path_patterns=spec["default_path_patterns"],
    )


def bids_label(text: str, fallback: str = "custom") -> str:
    """Anything a user typed, reduced to what BIDS allows in an entity value.

    ``"left PFC (dorsal)"`` -> ``"leftPFCdorsal"``;  ``"__"`` -> ``"custom"``

    A label is alphanumeric by definition, so a name carrying a space, a hyphen or an
    underscore would either be rejected or, worse, read as a second entity.
    """
    kept = "".join(ch for ch in str(text) if ch.isalnum())
    return kept or fallback


def layout_config() -> list[str]:
    """What to hand ``BIDSLayout(config=...)`` so it can index a tree this package wrote.

    Names rather than the merged ``Config`` above: BIDSLayout stores its configuration in
    sqlite, so it takes config *names or paths* and refuses an object. It also refuses two
    entities of one name, which is why this package's file redeclares none of the builtins.
    """
    return ["bids", "derivatives", str(_CONFIG_FILE)]


def derivative_path(output_dir, suffix: str, extension: str, **entities) -> Path:
    """Where one output goes, given what it is and which entities distinguish it.

    ``(out, "qc", ".json", subject="01", task="rest", desc="preproc")``
        -> ``out/sub-01/nirs/sub-01_task-rest_desc-preproc_qc.json``

    Entities whose value is None are dropped, so a caller can pass ``session=None`` without
    branching. Raises ValueError when no pattern fits, which is the failure worth having:
    silently returning a name outside the scheme is how the old hand-built strings drifted.
    """
    from bids.layout.writing import build_path

    given = {key: value for key, value in entities.items() if value is not None}
    relative = build_path({**given, "suffix": suffix, "extension": extension},
                          config().default_path_patterns)
    if relative is None:
        raise ValueError(
            f"no path pattern fits suffix={suffix!r} extension={extension!r} with "
            f"{sorted(given)}; add the suffix to the pattern it belongs to in "
            f"{_CONFIG_FILE.name}, rather than building the name by hand"
        )
    return Path(output_dir) / relative


def figure_name(label: str, desc: str, *, suffix: str = "nirs",
                extension: str = ".html", **entities) -> str:
    """One figure's filename. ``desc`` names the panel, ``suffix`` names what it draws.

    ``("sub-01_task-rest", "carpet")``        -> ``"sub-01_task-rest_desc-carpet_nirs.html"``
    ``("sub-01_task-rest", "detail", channel="S1D1")``
        -> ``"sub-01_task-rest_chan-S1D1_desc-detail_nirs.html"``

    The label is in the name because a subject's runs share one ``figures/`` folder. They
    used to be kept apart by a subdirectory per run, which meant the same panel was called
    the same thing in two places and neither name said which run it was.
    """
    carried = _label_entities(label)
    return derivative_path("", suffix, extension, datatype="figures", desc=desc,
                           **carried, **entities).name


def _label_entities(label: str) -> dict:
    """The entities a run or dyad label carries, for a caller that has only the label."""
    from fnirs_pipe.io.derivatives import entity_of

    return {key: entity_of(label, short)
            for key, short in (("subject", "sub"), ("group", "group"),
                               ("session", "ses"), ("task", "task"), ("run", "run"))}


def report_name(label: str, *, desc: "str | None" = None,
                condition: "str | None" = None, pairing: "str | None" = None) -> str:
    """One report page's filename, from the run label the QC code passes around.

    ``("sub-01_task-rest")``              -> ``"sub-01_task-rest_report.html"``
    ``("sub-01_task-rest", desc="raw")``  -> ``"sub-01_task-rest_desc-raw_report.html"``
    ``("group-G1_task-rest", condition="game1")``
        -> ``"group-G1_task-rest_cond-game1_report.html"``

    The QC code carries a label rather than a set of entities, so the entities are read back
    out of it. Four spellings used to coexist here (`_qc.html`, `_desc-raw_nirs.html`,
    `_qc_mne.html` and a subject index shaped like a run report), and `.html` on a `nirs`
    suffix claimed to be a snirf's sidecar.
    """
    return derivative_path("", "report", ".html", condition=condition, desc=desc,
                           pairing=pairing, **_label_entities(label)).name


def parse_path(path) -> dict:
    """Read the entities back out of a name :func:`derivative_path` built.

    ``"group-G1/nirs/group-G1_task-rest_stat-wtc_relmat.tsv"``
        -> ``{"group": "G1", "task": "rest", "statistic": "wtc", "suffix": "relmat", ...}``

    Give it a path relative to the output directory. An absolute one still parses, but any
    entity-looking fragment in the leading directories is read as an entity too.
    """
    from bids.layout import parse_file_entities

    return parse_file_entities(str(path), config=[config()])
