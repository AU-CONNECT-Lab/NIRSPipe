"""A --keep-spans table written from condition names: one row per occurrence of each."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nirspipe.io.snirf import read_snirf
from nirspipe.utils import is_marker
from nirspipe.utils.logging import get_logger

logger = get_logger("pipeline.keep_spans")

# table column -> pybids entity, in the table's column order
_LABEL_COLUMNS = {"session": "session", "task": "task", "run": "run"}


def condition_spans(
    bids_dir: Path,
    sub: str,
    conditions: list[str],
    duration: float | None = None,
    ses: str | None = None,
    task: str | None = None,
    run: str | None = None,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Every occurrence of ``conditions`` in one subject's selected recordings, as --keep-spans rows.

    ::

      sub-01 task-mixed: "talk" at 20 s for 150 s, "listen" at 190 s, conditions ["talk"]
      ->  [{"participant_id": "sub-01", "task": "mixed", "onset": 20.0, "duration": 150.0}]

    Onsets are on each recording's data axis, the axis --keep-spans reads. ``duration`` sets
    every occurrence's length; without it a zero-length occurrence is refused. A recording
    holding none of the conditions is refused, naming it.
    """
    from nirspipe.io.bids import get_layout

    layout = get_layout(bids_dir, validate=validate)
    query = {"subject": sub, "extension": ".snirf",
             **{k: v for k, v in (("session", ses), ("task", task), ("run", run)) if v}}
    files = sorted(layout.get(**query), key=lambda f: f.path)
    if not files:
        raise FileNotFoundError(f"No SNIRF found for sub-{sub} ses={ses} task={task} run={run}")

    rows: list[dict[str, Any]] = []
    seen: dict[tuple, str] = {}
    for f in files:
        name = Path(f.path).name
        entities = f.get_entities()
        labels = {"participant_id": f"sub-{sub}",
                  **{column: str(entities[entity]) for column, entity in _LABEL_COLUMNS.items()
                     if entities.get(entity) is not None}}
        # the table names a recording by these labels only, so another entity (acq) is lost
        key = tuple(sorted(labels.items()))
        if key in seen:
            logger.warning("%s and %s share every label the table has a column for; the rows "
                           "of each will apply to both", seen[key], name)
        seen[key] = name

        raw = read_snirf(f.path, preload=False, verbose="error")
        found = []
        for onset, length, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                       raw.annotations.description):
            if not is_marker(desc) or str(desc) not in conditions:
                continue
            onset = float(onset) - float(raw.first_time)
            if duration is None and float(length) <= 0:
                raise ValueError(f"{name}: condition {str(desc)!r} at {onset:g} s has no "
                                 f"duration; give --duration")
            found.append({**labels, "onset": onset,
                          "duration": float(length) if duration is None else float(duration)})
        if not found:
            raise ValueError(f"{name} holds none of the conditions {', '.join(conditions)}")
        rows += sorted(found, key=lambda row: row["onset"])
    return rows
