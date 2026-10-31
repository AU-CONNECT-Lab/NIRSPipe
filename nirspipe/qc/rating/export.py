"""Turn the rating viewers' files into tables the pipeline reads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import entity_of
from nirspipe.utils import pair_of
from nirspipe.utils.logging import get_logger

logger = get_logger("qc.rating.export")

CHANNELS_NAME = "desc-ratedchannels_qc.tsv"
VERDICTS_NAME = "desc-ratedpages_qc.tsv"
SPANS_NAME = "desc-ratedspans_qc.tsv"

# the dyad viewer's key: one verdict for every run of the member's task and session
HYPER_KEY = "_hyper"

_CHANNEL_COLUMNS = ["participant_id", "session", "task", "acq", "run", "rater",
                    "bad_channels", "keep_channels"]
_SPAN_COLUMNS = ["participant_id", "session", "task", "acq", "run", "rater", "onset",
                 "duration"]
_VERDICT_COLUMNS = ["page", "rater", "key", "verdict", "note"]

_RATED = ("*_desc-rawdecision_qc.json", "*_desc-rawspans_qc.json", "*_desc-rating_qc.json",
          "*_desc-rawrating_qc.json")


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("unreadable, skipped: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _pairs(states: dict, wanted: str) -> list[str]:
    return sorted({pair_of(ch) for ch, state in states.items() if state == wanted})


def _rater_of(path: Path) -> str:
    return entity_of(path.name, "rater") or ""


def _files(trees: list[Path], patterns: tuple[str, ...], rater: "str | None") -> list[Path]:
    """The trees' files of these kinds, tree by tree; only ``rater``'s when one is named."""
    return [path for tree in trees
            for path in sorted(p for pattern in patterns for p in tree.rglob(pattern))
            if rater is None or _rater_of(path) == rater]


def raters(trees: list[Path]) -> list[str]:
    """Every rater whose files the trees hold, ``""`` for files saved without ``--rater``."""
    return sorted({_rater_of(path) for path in _files(trees, _RATED, None)})


def channel_rows(trees: list[Path], rater: "str | None" = None) -> tuple[list[dict], list[str]]:
    """One ``--bad-channels`` row per rated run, and the pairs rated both ways for one recording.

    ::

      sub-01_task-rest_desc-rawdecision_qc.json:
      {"sub-01_task-rest_run-01": {"S1_D1 hbo": "bad", "S2_D2 hbo": "good"}}
      -> {"participant_id": "sub-01", "session": "", "task": "rest", "acq": "", "run": "01",
          "rater": "", "bad_channels": "S1_D1", "keep_channels": "S2_D2"}

    A ``_hyper`` entry leaves ``acq`` and ``run`` blank, so it holds for every run of that
    task and session. ``rater`` is the file's ``rater-`` entity, blank without one.
    """
    rows, conflicts = [], []
    for path in _files(trees, ("*_desc-rawdecision_qc.json",), rater):
        base = {"participant_id": f"sub-{entity_of(path.name, 'sub')}",
                "session": entity_of(path.name, "ses") or "",
                "task": entity_of(path.name, "task") or ""}
        data = {k: v for k, v in _read(path).items() if isinstance(v, dict)}
        hyper = data.get(HYPER_KEY, {})
        for key, states in data.items():
            # only the run page may decide: a decision applies to the whole run
            if key != HYPER_KEY and entity_of(key, "cond"):
                raise StageError(f"{path}: '{key}' was saved from a condition page, but a "
                                 "channel decision applies to the whole run. Delete that "
                                 "key and decide on the run page.")
            bad, keep = _pairs(states, "bad"), _pairs(states, "good")
            # a run's row and the dyad row both match that run, so they must agree
            against = {} if key == HYPER_KEY else hyper
            clash = ((set(bad) & set(_pairs(against, "good")))
                     | (set(keep) & set(_pairs(against, "bad"))) | (set(bad) & set(keep)))
            conflicts += [f"{base['participant_id']} {key}: {p}" for p in sorted(clash)]
            if bad or keep:
                run = "" if key == HYPER_KEY else (entity_of(key, "run") or "")
                acq = "" if key == HYPER_KEY else (entity_of(key, "acq") or "")
                rows.append({**base, "acq": acq, "run": run, "rater": _rater_of(path),
                             "bad_channels": ",".join(bad), "keep_channels": ",".join(keep)})
    return rows, conflicts


def verdict_rows(trees: list[Path], rater: "str | None" = None) -> list[dict]:
    """Every FINAL verdict a rating file holds, one row per page and key, for reading."""
    rows = []
    for path in _files(trees, ("*_desc-rating_qc.json", "*_desc-rawrating_qc.json"), rater):
        data = _read(path)
        notes = data.get("notes") or {}
        for key, verdict in (data.get("ratings") or {}).items():
            if "final" in key.lower() and verdict not in (None, "", "NA"):
                rows.append({"page": data.get("stem") or path.stem, "rater": _rater_of(path),
                             "key": key, "verdict": verdict, "note": notes.get(key, "")})
    return rows


def span_rows(trees: list[Path], rater: "str | None" = None) -> tuple[list[dict], bool]:
    """One ``--bad-spans`` row per marked stretch, and whether any span file was found.

    ::

        {"sub-01_task-rest_run-01": [[12.0, 15.5]]} in sub-01_task-rest_desc-rawspans_qc.json
            -> {"participant_id": "sub-01", "session": "", "task": "rest", "acq": "", "run": "01",
                "rater": "", "onset": 12.0, "duration": 3.5}
    """
    rows, found = [], False
    for path in _files(trees, ("*_desc-rawspans_qc.json",), rater):
        found = True
        base = {"participant_id": f"sub-{entity_of(path.name, 'sub')}",
                "session": entity_of(path.name, "ses") or "",
                "task": entity_of(path.name, "task") or ""}
        for key, spans in _read(path).items():
            for start, stop in spans:
                rows.append({**base, "acq": entity_of(key, "acq") or "",
                             "run": entity_of(key, "run") or "", "rater": _rater_of(path),
                             "onset": start, "duration": round(stop - start, 3)})
    return rows, found


def export_spans(trees: list[Path], out_dir: Path, rater: "str | None" = None) -> "Path | None":
    """Write the marked stretches as a --bad-spans table, or nothing where none were ever marked."""
    rows, found = span_rows(trees, rater)
    if not found:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / SPANS_NAME
    pd.DataFrame(rows, columns=_SPAN_COLUMNS).to_csv(path, sep="\t", index=False)
    return path


def export_ratings(trees: list[Path], out_dir: Path,
                   rater: "str | None" = None) -> tuple[Path, Path, list[str]]:
    """Write the channel table and the verdict list into ``out_dir``; return both paths and any conflicts."""
    channels, conflicts = channel_rows(trees, rater)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels_path, verdicts_path = out_dir / CHANNELS_NAME, out_dir / VERDICTS_NAME
    pd.DataFrame(channels, columns=_CHANNEL_COLUMNS).to_csv(channels_path, sep="\t", index=False)
    pd.DataFrame(verdict_rows(trees, rater), columns=_VERDICT_COLUMNS).to_csv(
        verdicts_path, sep="\t", index=False)
    return channels_path, verdicts_path, conflicts
