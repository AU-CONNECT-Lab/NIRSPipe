"""Stage lineage carried on mne.io.Raw objects.

Each transformation stamps its output with the pipeline stage that produced it,
so downstream consumers can check what they were handed at runtime instead of
trusting a parameter name.

Stored in raw.info["temp"]. Survives copy / filter / resample / crop, but not a
SNIRF round trip: readers re-stamp from the desc- entity in the filename.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fnirs_pipe.exceptions import StageError

if TYPE_CHECKING:
    import mne

_KEY = "fnirs_pipe_lineage"


@dataclass(frozen=True)
class Lineage:
    stage: str                        # desc- entity of the output, e.g. "preproc"
    step: str                         # transformation that produced it, e.g. "beer_lambert"
    source: str | None = None         # stage of the input, "raw" if the input was unstamped
    params: dict[str, Any] = field(default_factory=dict)


def stamp(
    raw: mne.io.Raw,
    stage: str,
    step: str,
    source: mne.io.Raw | None = None,
    **params: Any,
) -> mne.io.Raw:
    """Record the producing stage on raw. Returns raw so calls can be chained.

    Safe when source is raw itself (in-place steps): the source stage is read
    before the new stamp overwrites it.
    """
    if source is None:
        src = None
    else:
        prev = lineage_of(source)
        src = prev.stage if prev else "raw"
    raw.info["temp"] = {**(raw.info.get("temp") or {}), _KEY: Lineage(stage, step, src, params)}
    return raw


def lineage_of(raw: mne.io.Raw) -> Lineage | None:
    return (raw.info.get("temp") or {}).get(_KEY)


def stage_of(raw: mne.io.Raw) -> str | None:
    lin = lineage_of(raw)
    return lin.stage if lin else None


def require_stage(raw: mne.io.Raw, *allowed: str) -> None:
    """Guard for computations that are only valid at certain pipeline stages."""
    if (st := stage_of(raw)) not in allowed:
        raise StageError(f"requires stage in {allowed}, got {st!r}")
