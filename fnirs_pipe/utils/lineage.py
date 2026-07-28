"""Provenance: stage lineage on Raw objects, and a manifest of the files written.

Each transformation stamps its output with the pipeline stage that produced it,
so downstream consumers can check what they were handed at runtime instead of
trusting a parameter name.

Stored in raw.info["temp"]. Survives copy / filter / resample / crop, but not a
SNIRF round trip: readers re-stamp from the desc- entity in the filename.

Recorder turns those stamps into file-level provenance: it remembers which path
each stage was written to, so an output's Sources can be resolved from the stamp
rather than threaded through by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


class Recorder:
    """Tracks which file each pipeline stage was written to, within one run.

    ``register_input`` seeds a file that was read rather than produced, so the
    first output of a run can still name its source.
    """

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._stage_paths: dict[str, str] = {}

    def register_input(self, path: Path | str, raw: mne.io.Raw) -> None:
        if (st := stage_of(raw)) is not None:
            self._stage_paths[st] = Path(path).as_posix()

    def sources_of(self, raw: mne.io.Raw) -> list[str]:
        """Paths this object's input stage was written to, for the BIDS Sources field."""
        lin = lineage_of(raw)
        src = self._stage_paths.get(lin.source) if lin and lin.source else None
        return [src] if src else []

    def written(self, path: Path, raw: mne.io.Raw) -> Path:
        lin = lineage_of(raw)
        self.entries.append({
            "path": Path(path).as_posix(),
            "stage": lin.stage if lin else None,
            "step": lin.step if lin else None,
            "sources": self.sources_of(raw),
            "params": dict(lin.params) if lin else {},
        })
        if lin:
            self._stage_paths[lin.stage] = Path(path).as_posix()
        return path

    @property
    def last(self) -> Path | None:
        return Path(self.entries[-1]["path"]) if self.entries else None
