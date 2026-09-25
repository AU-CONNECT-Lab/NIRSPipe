"""Write BIDS Derivatives output structure."""

import json
import os
import re
import time
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from typing import Any

from fnirs_pipe.exceptions import MissingDerivativesError
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe import __version__

logger = get_logger("io.derivatives")

def carry_entities(source_entities: dict[str, str] | None) -> dict[str, str]:
    """Keep only task/run from source entities so output filenames mirror the input."""
    return {k: v for k, v in (source_entities or {}).items() if k in ("task", "run")}


def build_output_path(
    output_dir: Path,
    subject: str,
    entities: dict[str, str],
    suffix: str,
    extension: str,
    session: str | None = None,
) -> Path:
    """A derivative's path, with its directory already created.

    ::

      entities={"task": "rest", "desc": "preproc"}, suffix="nirs", extension=".snirf"
        -> <out>/sub-01/nirs/sub-01_task-rest_desc-preproc_nirs.snirf

    The name comes from :mod:`fnirs_pipe.io.naming`, which reads it off the one config every
    reader parses against. This is the writing half: it exists so a caller that is about to
    write a file does not have to remember to create the folder, and so a caller that only
    wants a name can ask naming directly and leave no tree behind.
    """
    from fnirs_pipe.io.naming import derivative_path

    path = derivative_path(output_dir, suffix, extension,
                           subject=subject, session=session, **entities)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def channel_decisions_path(
    output_dir: Path, subject: str, task: str | None = None, session: str | None = None,
) -> Path:
    """Where the raw QC page keeps a run's per-channel keep/drop decisions.

    Re-exported from :mod:`fnirs_pipe.io.naming`, which builds it.
    """
    from fnirs_pipe.io.naming import channel_decisions_path as _path

    return _path(output_dir, subject, task, session)


def subject_report_dir(output_dir: Path, subject_id: str) -> Path:
    """A subject's own folder, holding their HTML reports. The mirror of ``group-<id>/``."""
    folder = output_dir / f"sub-{subject_id.removeprefix('sub-')}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def group_report_dir(output_dir: Path, group_id: str) -> Path:
    """A group's own folder, holding its HTML reports. The mirror of ``sub-<id>/``."""
    folder = output_dir / f"group-{group_id}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def group_data_dir(output_dir: Path, group_id: str, session: str | None = None) -> Path:
    """A group's tables, sidecars and quality record: ``group-<id>/[ses-<s>/]nirs``.

    The mirror of a subject's ``sub-<id>/[ses-<s>/]nirs``.
    """
    folder = group_report_dir(output_dir, group_id)
    if session:
        folder = folder / f"ses-{session}"
    folder = folder / "nirs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def group_output_path(
    output_dir: Path,
    group_id: str,
    entities: dict,
    suffix: str,
    extension: str,
    session: str | None = None,
) -> Path:
    """A dyad derivative's path, with its directory already created.

    ::

      entities={"task": "rest", "statistic": "wtc"}, suffix="relmat", extension=".tsv"
        -> <out>/group-G1/nirs/group-G1_task-rest_stat-wtc_relmat.tsv

    The mirror of :func:`build_output_path` for the group side. It builds the whole name
    rather than handing out a prefix to append to, so each dimension stays its own entity.
    """
    from fnirs_pipe.io.naming import derivative_path

    path = derivative_path(output_dir, suffix, extension,
                           group=group_id, session=session, **entities)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def data_state(raw: Any) -> dict[str, Any]:
    """Shape of the signal as written, so a sidecar records what the step left behind.

    A 56-channel 10 Hz recording with 16 channels marked bad ->
        {"n_channels": 56, "n_bad": 16, "sfreq": 10.0, "duration_s": 595.2}

    Cheap: everything here is already on the Raw being written.
    """
    return {
        "n_channels": len(raw.ch_names),
        "n_bad": len(raw.info.get("bads") or []),
        "sfreq": round(float(raw.info["sfreq"]), 4),
        "duration_s": round(float(raw.n_times) / float(raw.info["sfreq"]), 1),
    }


def write_sidecar_json(out_path: Path, provenance: dict[str, Any]) -> None:
    """Write a JSON sidecar next to out_path (same name, .json extension).

    provenance should include:
      - pipeline_version
      - step (e.g. 'od_conversion')
      - parameters (dict of relevant config values)
      - Sources (BIDS field: list of source paths as strings)
      - data (channel count, sampling rate, duration; see data_state)
      - timestamp (ISO-8601, auto-added if missing)
    """
    provenance.setdefault(
        "timestamp", datetime.now(timezone.utc).isoformat()
    )
    # a .tsv.gz would otherwise get a .tsv.json sidecar, which nothing would find
    name = out_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    sidecar_path = out_path.with_name(name).with_suffix(".json")
    sidecar_path.write_text(json.dumps(provenance, indent=2))



def entity_of(path: "Path | str", name: str) -> str | None:
    """Read one BIDS entity back out of a filename, or None when it carries no such key.

    ``entity_of("sub-01_task-rest_desc-preproc_nirs.snirf", "desc")`` -> ``"preproc"``

    The one place the package parses an entity out of a name, so a rename cannot leave one
    reader matching and another silently finding nothing. A label is alphanumeric by BIDS
    definition, so the match ends at the underscore or dot that follows it.
    """
    # ^ as well as _, so the leading sub- or group- is readable too
    m = re.search(rf"(?:^|_){name}-([A-Za-z0-9]+)", getattr(path, "name", path))
    return m.group(1) if m else None


def select_one_run(
    candidates: list[Path],
    *,
    what: str,
    subject_id: str,
    task: str,
    session: str | None = None,
    run: str | None = None,
) -> Path:
    """Narrow a list of candidate recordings to exactly one, or say why it cannot.

    ``[sub-01_ses-a_x, sub-01_ses-b_x], session="a"`` -> the ses-a one
    ``[sub-01_ses-a_x, sub-01_ses-b_x], session=None`` -> raises, naming a and b

    One recording per group member is what every inter-brain metric assumes, so ambiguity is
    refused rather than resolved to the first match.
    """
    for name, value in (("ses", session), ("run", run)):
        if value is not None:
            candidates = [p for p in candidates if entity_of(p, name) == value]

    if not candidates:
        raise MissingDerivativesError(
            f"No {what} for {subject_id} task-{task}"
            + (f" ses-{session}" if session else "") + (f" run-{run}" if run else "") + "."
        )
    if len(candidates) == 1:
        return candidates[0]

    def _spread(name: str) -> str:
        found = sorted({v for p in candidates if (v := entity_of(p, name))})
        return f"{name}: {', '.join(found)}" if len(found) > 1 else ""

    varies = ", ".join(x for x in (_spread("ses"), _spread("run")) if x)
    raise MissingDerivativesError(
        f"{len(candidates)} candidates for {subject_id} task-{task} {what}: "
        f"{', '.join(p.name for p in candidates)}. "
        + (f"They differ by {varies}. Add a session or run column to the group CSV to say "
           "which one to use." if varies
           else "They cannot be told apart by session or run; remove the duplicates.")
    )


def subject_nirs_dirs(
    output_dir: Path, subject_id: str, session: str | None = None,
) -> list[Path]:
    """Every ``nirs/`` a subject's derivatives can sit in, session level included.

    ::

      sub-01/nirs and sub-01/ses-a/nirs on disk, session=None -> both
      the same tree, session="a"                              -> the ses-a one only

    :func:`derivatives_path` writes a session to its own folder, so a subject recorded over
    two sessions has no ``sub-01/nirs`` at all. Readers that build the path by hand find
    nothing there and report the subject as missing. Only directories that exist are
    returned, so a caller that gets an empty list is looking at a subject with no output.
    """
    subject = output_dir / f"sub-{subject_id.removeprefix('sub-')}"
    candidates = ([subject / f"ses-{session}" / "nirs"] if session
                  else [subject / "nirs", *sorted(subject.glob("ses-*/nirs"))])
    return [d for d in candidates if d.is_dir()]


def find_preproc_snirf(
    output_dir: Path, subject_id: str, task: str, desc: str = "preproc",
    session: str | None = None, run: str | None = None,
) -> Path:
    """Locate the desc-{desc} snirf for *subject_id* under *output_dir*.

    Every pipeline step writes one snirf per desc, so desc is what selects a stage:
    "preproc" is Beer-Lambert output, "filtered" the bandpassed one, "errts" the GLM
    residual. Raises MissingDerivativesError if the directory or the file is absent.

    Searched across :func:`subject_nirs_dirs`, so a session either names its folder or,
    unnamed, has every session's folder offered to ``select_one_run`` at once. That is what
    lets a multi-session tree raise "which session" rather than "no such directory".
    """
    nirs_dirs = subject_nirs_dirs(output_dir, subject_id, session)
    if not nirs_dirs:
        raise MissingDerivativesError(
            "Derivatives directory not found: "
            f"{output_dir / subject_id / (f'ses-{session}/nirs' if session else 'nirs')}")

    def _glob(pattern: str) -> list[Path]:
        return sorted((p for d in nirs_dirs for p in d.glob(pattern)), key=lambda p: p.name)

    candidates = _glob(f"{subject_id}_*task-{task}_*desc-{desc}_nirs.snirf")
    if not candidates:
        # only when this stage carries no task entity anywhere: a subject who has the stage
        # for other tasks but not this one is missing data, and handing back another task's
        # recording would analyse the wrong condition without saying so
        untasked = _glob(f"{subject_id}_*desc-{desc}_nirs.snirf")
        if untasked and not any("_task-" in p.name for p in untasked):
            candidates = untasked
    if not candidates:
        available = sorted({found for p in _glob(f"{subject_id}_*_nirs.snirf")
                            if (found := entity_of(p, "desc"))})
        tasks = sorted({found for p in _glob(f"{subject_id}_*desc-{desc}_nirs.snirf")
                        if (found := entity_of(p, "task"))})
        where = ", ".join(str(d) for d in nirs_dirs)
        raise MissingDerivativesError(
            f"No desc-{desc} snirf found for {subject_id} (task={task}) in {where}. "
            f"Available desc: {', '.join(available) if available else 'none'}. "
            + (f"desc-{desc} exists for task: {', '.join(tasks)}. " if tasks else "")
            + "Run fnirs-pipe preprocessing first."
        )
    return select_one_run(candidates, what=f"desc-{desc} snirf", subject_id=subject_id,
                          task=task, session=session, run=run)


# Windows refuses to rename over a file anything else has open, so a reader that happens to be
# mid-read costs a retry rather than the run.
_REPLACE_TRIES = 5
_REPLACE_WAIT_S = 0.05


# What a reader should not hold against the tree: the reports and their figures are for
# people, and BIDS says nothing about either.
#
# No trailing slash on the two directories: the validator matches nothing with `figures/`.
#
# The records that are a JSON with no data file beside them: the quality records, the
# human ratings and the channel decisions. A validator reads any such JSON as a sidecar
# whose data file is missing. Named one by one, never `*.json`, so every real sidecar and
# every table stays checked.
JSON_ONLY_DESCS = ("sqm", "sqmraw", "rating", "rawrating", "rawdecision")
_BIDSIGNORE = ("*.html", "logs", "figures",
               *(f"*_desc-{desc}_qc.json" for desc in JSON_ONLY_DESCS))


def write_bidsignore(output_dir: Path) -> None:
    """Register the paths BIDS has no say over, so a validator skips rather than flags them.

    Reports, logs, the figures inside them, and the JSON-only records. Every table and every
    file with a sidecar stays on the validator's books, dyad tables included.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".bidsignore").write_text("\n".join(_BIDSIGNORE) + "\n", encoding="utf-8")


def _source_dataset(source: Path) -> dict:
    """One SourceDatasets entry for the tree this output was computed from.

    Reads the source's own description so the entry carries which tool and which version
    wrote it, which is the thing a reader cannot recover from the dyad tables themselves.
    """
    entry: dict = {"URL": source.resolve().as_uri()}
    try:
        desc = json.loads((source / "dataset_description.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return entry
    made_by = (desc.get("GeneratedBy") or [{}])[0]
    if made_by.get("Version"):
        entry["Version"] = made_by["Version"]
    if made_by.get("Name"):
        entry["Name"] = made_by["Name"]
    return entry


def write_dataset_description(
    output_dir: Path, *, name: str = "fnirs-pipe output",
    generated_by: str = "fnirs-pipe", source: "Path | None" = None,
) -> None:
    """Write dataset_description.json for the derivatives dataset.

    Every run writes this, so two started against one output directory write the same path.
    The content is fixed for one run: a run that finds it already correct leaves it alone,
    which is what keeps concurrent runs off each other rather than the rename. Where it does
    have to be written it goes to a temporary first, since the plain write truncates and a
    reader in between sees an empty file and reports the tree as bad BIDS. A reader can still
    be told the path is busy at the instant it flips, which Windows offers no way around, but
    that is a retry rather than a tree that looks invalid.

    ``source`` is the tree this one was computed from. It is what lets a reader of a dyad
    result recover which preprocessing produced its inputs, and it has no correct value while
    a tool writes back into the tree it read.
    """
    desc = {
        "Name": name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": generated_by, "Version": __version__}],
    }
    if source is not None:
        desc["SourceDatasets"] = [_source_dataset(Path(source))]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset_description.json"
    text = json.dumps(desc, indent=2)
    try:
        if path.read_text() == text:
            return
    except OSError:
        pass

    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text)
        for attempt in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_TRIES - 1:
                    raise
                time.sleep(_REPLACE_WAIT_S)
    finally:
        # a successful replace has already consumed it; anything else must not leave it behind
        tmp.unlink(missing_ok=True)

