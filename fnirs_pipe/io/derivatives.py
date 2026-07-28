"""Write BIDS Derivatives output structure."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fnirs_pipe.exceptions import MissingDerivativesError


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


def write_sidecar_json(out_path: Path, provenance: dict[str, Any]) -> None:
    """Write a JSON sidecar next to out_path (same name, .json extension).

    provenance should include:
      - pipeline_version
      - step (e.g. 'od_conversion')
      - parameters (dict of relevant config values)
      - Sources (BIDS field: list of source paths as strings)
      - timestamp (ISO-8601, auto-added if missing)
    """
    provenance.setdefault(
        "timestamp", datetime.now(timezone.utc).isoformat()
    )
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(provenance, indent=2))



def find_preproc_snirf(output_dir: Path, subject_id: str, task: str) -> Path:
    """Locate the desc-preproc snirf for *subject_id* under *output_dir*.

    Raises MissingDerivativesError if the derivatives directory or file is absent.
    """
    nirs_dir = output_dir / subject_id / "nirs"
    if not nirs_dir.exists():
        raise MissingDerivativesError(f"Derivatives directory not found: {nirs_dir}")

    candidates = sorted(nirs_dir.glob(f"{subject_id}_task-{task}_*desc-preproc_nirs.snirf"))
    if not candidates:
        candidates = sorted(nirs_dir.glob(f"{subject_id}_*desc-preproc_nirs.snirf"))
    if not candidates:
        raise MissingDerivativesError(
            f"No desc-preproc snirf found for {subject_id} (task={task}) in {nirs_dir}. "
            "Run fnirs-pipe preprocessing first."
        )
    return candidates[0]


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

