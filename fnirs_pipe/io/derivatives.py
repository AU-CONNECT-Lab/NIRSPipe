"""Write BIDS Derivatives output structure."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fnirs_pipe.exceptions import MissingDerivativesError
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("io.derivatives")

# same entity pattern read_snirf parses the stage back out of
_DESC_RE = re.compile(r"_desc-([A-Za-z0-9]+)[_.]")


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



def _entity(name: str, path: Path) -> str | None:
    m = re.search(rf"_{name}-([A-Za-z0-9]+)", path.name)
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
            candidates = [p for p in candidates if _entity(name, p) == value]

    if not candidates:
        raise MissingDerivativesError(
            f"No {what} for {subject_id} task-{task}"
            + (f" ses-{session}" if session else "") + (f" run-{run}" if run else "") + "."
        )
    if len(candidates) == 1:
        return candidates[0]

    def _spread(name: str) -> str:
        found = sorted({v for p in candidates if (v := _entity(name, p))})
        return f"{name}: {', '.join(found)}" if len(found) > 1 else ""

    varies = ", ".join(x for x in (_spread("ses"), _spread("run")) if x)
    raise MissingDerivativesError(
        f"{len(candidates)} candidates for {subject_id} task-{task} {what}: "
        f"{', '.join(p.name for p in candidates)}. "
        + (f"They differ by {varies}. Add a session or run column to the group CSV to say "
           "which one to use." if varies
           else "They cannot be told apart by session or run; remove the duplicates.")
    )


def find_preproc_snirf(
    output_dir: Path, subject_id: str, task: str, desc: str = "preproc",
    session: str | None = None, run: str | None = None,
) -> Path:
    """Locate the desc-{desc} snirf for *subject_id* under *output_dir*.

    Every pipeline step writes one snirf per desc, so desc is what selects a stage:
    "preproc" is Beer-Lambert output, "filtered" the bandpassed one, "errts" the GLM
    residual. Raises MissingDerivativesError if the directory or the file is absent.
    """
    nirs_dir = output_dir / subject_id / "nirs"
    if not nirs_dir.exists():
        raise MissingDerivativesError(f"Derivatives directory not found: {nirs_dir}")

    candidates = sorted(nirs_dir.glob(f"{subject_id}_task-{task}_*desc-{desc}_nirs.snirf"))
    if not candidates:
        # only when this stage carries no task entity anywhere: a subject who has the stage
        # for other tasks but not this one is missing data, and handing back another task's
        # recording would analyse the wrong condition without saying so
        untasked = sorted(nirs_dir.glob(f"{subject_id}_*desc-{desc}_nirs.snirf"))
        if untasked and not any("_task-" in p.name for p in untasked):
            candidates = untasked
    if not candidates:
        available = sorted({m.group(1) for p in nirs_dir.glob(f"{subject_id}_*_nirs.snirf")
                            if (m := _DESC_RE.search(p.name))})
        tasks = sorted({m.group(1) for p in nirs_dir.glob(f"{subject_id}_*desc-{desc}_nirs.snirf")
                        if (m := re.search(r"_task-([A-Za-z0-9]+)", p.name))})
        raise MissingDerivativesError(
            f"No desc-{desc} snirf found for {subject_id} (task={task}) in {nirs_dir}. "
            f"Available desc: {', '.join(available) if available else 'none'}. "
            + (f"desc-{desc} exists for task: {', '.join(tasks)}. " if tasks else "")
            + "Run fnirs-pipe preprocessing first."
        )
    return select_one_run(candidates, what=f"desc-{desc} snirf", subject_id=subject_id,
                          task=task, session=session, run=run)


def write_dataset_description(output_dir: Path) -> None:
    """Write dataset_description.json for the derivatives dataset."""
    desc = {
        "Name": "fnirs-pipe output",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fnirs-pipe"}],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset_description.json"
    path.write_text(json.dumps(desc, indent=2))

