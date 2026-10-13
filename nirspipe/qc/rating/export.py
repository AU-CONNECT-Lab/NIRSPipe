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

# the dyad viewer's key: one verdict for every run of the member's task and session
HYPER_KEY = "_hyper"

_CHANNEL_COLUMNS = ["participant_id", "session", "task", "run", "bad_channels", "keep_channels"]


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("unreadable, skipped: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _pairs(states: dict, wanted: str) -> list[str]:
    return sorted({pair_of(ch) for ch, state in states.items() if state == wanted})


def channel_rows(trees: list[Path]) -> tuple[list[dict], list[str]]:
    """One ``--bad-channels`` row per rated run, and the pairs rated both ways for one recording.

    ::

      sub-01_task-rest_desc-rawdecision_qc.json:
      {"sub-01_task-rest_run-01": {"S1_D1 hbo": "bad", "S2_D2 hbo": "good"}}
      -> {"participant_id": "sub-01", "session": "", "task": "rest", "run": "01",
          "bad_channels": "S1_D1", "keep_channels": "S2_D2"}

    A ``_hyper`` entry leaves ``run`` blank, so it holds for every run of that task and session.
    """
    rows, conflicts = [], []
    for tree in trees:
        for path in sorted(tree.rglob("*_desc-rawdecision_qc.json")):
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
                    rows.append({**base, "run": run, "bad_channels": ",".join(bad),
                                 "keep_channels": ",".join(keep)})
    return rows, conflicts


def verdict_rows(trees: list[Path]) -> list[dict]:
    """Every FINAL verdict a rating file holds, one row per page and key, for reading."""
    rows = []
    for tree in trees:
        for path in sorted([*tree.rglob("*_desc-rating_qc.json"),
                            *tree.rglob("*_desc-rawrating_qc.json")]):
            data = _read(path)
            notes = data.get("notes") or {}
            for key, verdict in (data.get("ratings") or {}).items():
                if "final" in key.lower() and verdict not in (None, "", "NA"):
                    rows.append({"page": data.get("stem") or path.stem, "key": key,
                                 "verdict": verdict, "note": notes.get(key, "")})
    return rows


def export_ratings(trees: list[Path], out_dir: Path) -> tuple[Path, Path, list[str]]:
    """Write the channel table and the verdict list into ``out_dir``; return both paths and any conflicts."""
    channels, conflicts = channel_rows(trees)
    out_dir.mkdir(parents=True, exist_ok=True)
    channels_path, verdicts_path = out_dir / CHANNELS_NAME, out_dir / VERDICTS_NAME
    pd.DataFrame(channels, columns=_CHANNEL_COLUMNS).to_csv(channels_path, sep="\t", index=False)
    pd.DataFrame(verdict_rows(trees), columns=["page", "key", "verdict", "note"]).to_csv(
        verdicts_path, sep="\t", index=False)
    return channels_path, verdicts_path, conflicts
