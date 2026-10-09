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

from collections.abc import Iterable

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fnirs_pipe.exceptions import StageError

if TYPE_CHECKING:
    import mne

_KEY = "fnirs_pipe_lineage"
_RESERVED = ("raw", "stage", "step", "source", "path")


@dataclass(frozen=True)
class Lineage:
    stage: str                        # desc- entity of the output, e.g. "preproc"
    step: str                         # transformation that produced it, e.g. "beer_lambert"
    source: str | None = None         # stage of the input, "raw" if the input was unstamped
    params: dict[str, Any] = field(default_factory=dict)
    path: str | None = None           # file this was read from, when it came off disk


def stamp(
    raw: mne.io.Raw,
    stage: str,
    step: str,
    source: mne.io.Raw | None = None,
    path: str | None = None,
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
    raw.info["temp"] = {
        **(raw.info.get("temp") or {}),
        _KEY: Lineage(stage, step, src, params, path),
    }
    return raw


def restamp(raw: mne.io.Raw, **params: Any) -> mne.io.Raw:
    """Raw's own stamp again with ``params`` replaced, for a step that moves what it recorded.

    ::

      aligned stamp {offset_s: 12.0, duration_s: 400.0} + restamp(offset_s=32.0, duration_s=360.0)
    """
    lin = lineage_of(raw)
    if lin is None:
        return raw
    raw.info["temp"] = {**(raw.info.get("temp") or {}),
                        _KEY: replace(lin, params={**lin.params, **params})}
    return raw


def carried_params(raw: mne.io.Raw) -> dict[str, Any]:
    """Params of raw's stamp, for a re-stamp that has to keep them.

    Refuses a param sharing a name with stamp()'s own arguments: expanded into that call
    it would bind twice, and Python raises before the body can say which key did it.
    """
    params = (lin.params if (lin := lineage_of(raw)) else None) or {}
    if clash := sorted(set(params) & set(_RESERVED)):
        raise StageError(
            f"lineage params {clash} share a name with an argument of stamp(); "
            "rename them at the step that stamped them")
    return params


def path_from(raw: mne.io.Raw) -> str | None:
    """File this object was read from, or None if it was produced in memory."""
    lin = lineage_of(raw)
    return lin.path if lin else None


def paths_from(raws: "Iterable[mne.io.Raw]") -> list[str]:
    """Files these objects were read from, in order, skipping any produced in memory."""
    return [p for p in (path_from(raw) for raw in raws) if p]


def lineage_of(raw: "mne.io.Raw | None") -> Lineage | None:
    # a member the group is missing has no recording and so no stamp, which the callers
    # already read as "not aligned" rather than as an error
    return (raw.info.get("temp") or {}).get(_KEY) if raw is not None else None


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
        # stamp() calls an unstamped source "raw", so key it the same way: dropping the
        # path instead leaves every output of the run with an empty Sources
        self._stage_paths[stage_of(raw) or "raw"] = Path(path).as_posix()

    def _nearest(self) -> str | None:
        """Most recent file this run knows about.

        Not every stage reaches disk: rest mode filters in memory and never writes
        desc-filtered, so an output whose source stage has no file still needs an
        ancestor to point at. Stage paths are inserted in pipeline order, so the
        last one is the nearest persisted ancestor.
        """
        return next(reversed(self._stage_paths.values()), None)

    def sources_of(self, raw: mne.io.Raw) -> list[str]:
        """Nearest persisted ancestor of raw, for the BIDS Sources field."""
        lin = lineage_of(raw)
        src = self._stage_paths.get(lin.source) if lin and lin.source else None
        src = src or self._nearest()
        return [src] if src else []

    def path_of(self, raw: mne.io.Raw) -> str | None:
        """File raw itself was written to, or its nearest persisted ancestor."""
        st = stage_of(raw)
        return (self._stage_paths.get(st) if st else None) or self._nearest()

    def written(self, path: Path, raw: mne.io.Raw) -> Path:
        lin = lineage_of(raw)
        if lin and lin.stage in self._stage_paths:
            raise StageError(
                f"stage {lin.stage!r} already written to {self._stage_paths[lin.stage]}; "
                f"a second file cannot share it without breaking source resolution"
            )
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
