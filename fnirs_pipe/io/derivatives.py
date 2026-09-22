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
    """Construct a BIDS-Derivatives compliant output path.

    All intermediate and final files land in output_dir (no separate work_dir).
    The desc entity distinguishes processing steps, e.g.::

      desc-od, desc-sci, desc-motcorrected, desc-preproc

    Examples::

      sub-01/nirs/sub-01_desc-od_nirs.snirf              (no session)
      sub-01/ses-wave1/nirs/sub-01_ses-wave1_desc-od_nirs.snirf
    """
    parts = [f"sub-{subject}"]
    if session:
        parts.append(f"ses-{session}")

    folder = output_dir.joinpath(*parts, "nirs")
    folder.mkdir(parents=True, exist_ok=True)

    entity_order = ["sub", "ses", "task", "run", "desc"]
    all_entities = {"sub": subject}
    if session:
        all_entities["ses"] = session
    all_entities.update(entities)

    filename_parts = [
        f"{key}-{all_entities[key]}"
        for key in entity_order
        if key in all_entities
    ]
    filename = "_".join(filename_parts) + f"_{suffix}{extension}"
    return folder / filename


def channel_decisions_path(
    output_dir: Path, subject: str, task: str | None = None, session: str | None = None,
) -> Path:
    """Where the raw QC page keeps a run's per-channel keep/drop decisions.

    ``(out, "01", task="rest")`` -> ``out/sub-01_task-rest_raw_channel_decisions.json``

    One function because four call sites built this name by hand: the rating server, the
    dyad rating server, the Hyper Preparation page and the Data Prep page. They agreed on
    the parts by convention alone, and the Hyper Preparation one left the session out, so
    on a two-session tree it reads a path the others never write. That call still passes no
    session because the page holds none; the mismatch is now in one place instead of four.
    """
    parts = [f"sub-{str(subject).removeprefix('sub-')}"]
    if session:
        parts.append(f"ses-{session}")
    if task:
        parts.append(f"task-{task}")
    return Path(output_dir) / ("_".join(parts) + "_raw_channel_decisions.json")


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

    The mirror of a subject's ``sub-<id>/[ses-<s>/]nirs``. These used to sit loose in the
    derivatives root, where a study of 25 dyads over 5 tasks put two thousand files
    between the reader and the subject folders.
    """
    folder = group_report_dir(output_dir, group_id)
    if session:
        folder = folder / f"ses-{session}"
    folder = folder / "nirs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


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
      - data (channel count, sampling rate, duration — see data_state)
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

    The one place the package parses an entity out of a name. It used to be three separate
    regexes in three modules, which is how a rename can leave two of them reading and the
    third silently finding nothing. A label is alphanumeric by BIDS definition, so the
    match ends at the underscore or dot that follows it.
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

    One recording per group member is what every inter-brain metric assumes, and the two
    places that used to pick one both took the first match in sorted order. A subject with
    two sessions or two runs of the same task then had one of them silently analysed and the
    other silently dropped, with nothing in the output saying which. Ambiguity is refused
    here instead: naming the session or the run is a decision only the caller can make.
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


def write_dataset_description(output_dir: Path) -> None:
    """Write dataset_description.json for the derivatives dataset.

    Every run writes this, so two started against one output directory write the same path.
    The content is fixed: a run that finds it already correct leaves it alone, which is what
    keeps concurrent runs off each other rather than the rename. Where it does have to be
    written it goes to a temporary first, since the plain write truncates and a reader in
    between sees an empty file and reports the tree as bad BIDS. A reader can still be told
    the path is busy at the instant it flips, which Windows offers no way around, but that is
    a retry rather than a tree that looks invalid.
    """
    from fnirs_pipe import __version__

    desc = {
        "Name": "fnirs-pipe output",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fnirs-pipe", "Version": __version__}],
    }
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

