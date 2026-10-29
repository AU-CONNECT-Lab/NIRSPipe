"""One delimiter rule for every table a user hands the package, one format for every table it writes."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from nirspipe.utils.logging import get_logger

logger = get_logger("io.tables")

# Extensions that name their own delimiter. Anything else is sniffed.
_BY_EXTENSION = {".tsv": "\t", ".csv": ","}

_CANDIDATES = ("\t", ",", ";", "|")

# how BIDS codes a missing or non-applicable value in a table
NA = "n/a"

# a condition label is text: read as numbers, a label "1" would come back as 1.0
LABEL_DTYPES = {"condition": str}

# the condition a whole-run row carries in a table that also holds conditions
WHOLE_RUN = "whole_run"


def write_tsv(frame: pd.DataFrame, path, **kwargs):
    """Write a table as BIDS reads one: tab-separated, missing values ``n/a``, no index unless asked."""
    frame.to_csv(path, sep="\t", na_rep=NA, **{"index": False, **kwargs})
    return path


def column_entries(columns, descriptions: dict, units: "dict | None" = None) -> dict:
    """BIDS sidecar entries, Description and Units, for the given columns that either dict names."""
    out = {}
    for column in columns:
        entry = {"Description": descriptions[column]} if column in descriptions else {}
        if units and units.get(column):
            entry["Units"] = units[column]
        if entry:
            out[column] = entry
    return out


def _sniff_delimiter(path: Path) -> str:
    """Pick a delimiter off the header line of a table whose extension does not say.

    Counts each candidate in the first line and takes the most frequent, so
    ``group_id,subject_id,task`` gives ``,`` and ``onset\\tduration`` gives ``\\t``.
    Falls back to whitespace when the header holds no candidate but does hold a space
    (``onset duration``), and to tab for a single-column file, where nothing separates
    anything and the choice cannot matter.
    """
    with path.open(encoding="utf-8", errors="replace") as fh:
        header = fh.readline()

    counts = {d: header.count(d) for d in _CANDIDATES}
    best = max(counts, key=lambda d: counts[d])
    if counts[best]:
        return best
    return r"\s+" if re.search(r"\S\s+\S", header.strip()) else "\t"


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a user-supplied table, whatever its extension and delimiter.

    Every entry point that accepts a table from the user (events, segments, pairs,
    participants) goes through here, so a `.csv` of events works where a `.tsv` does and
    nobody has to convert a file to satisfy one call site. Extra keyword arguments reach
    ``pandas.read_csv`` unchanged.

    This is for reading only. What the package writes back out stays BIDS-correct:
    `participants.tsv` and `*_events.tsv` are tab-separated regardless of what came in.
    """
    path = Path(path)
    sep = _BY_EXTENSION.get(path.suffix.lower()) or _sniff_delimiter(path)
    return pd.read_csv(path, sep=sep, engine="python", **kwargs)


def read_tsv_or_none(path: Path, lost: str) -> "pd.DataFrame | None":
    """A tab-separated table the package wrote, or None when absent or unreadable.

    ``lost`` says what an unreadable one costs the caller, for the warning.
    """
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep="	", dtype=LABEL_DTYPES)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("unreadable table, %s: %s (%s)", lost, path.name, exc)
        return None


def join_whole_run(whole: "list[pd.DataFrame]",
                   by_condition: "list[pd.DataFrame]") -> "pd.DataFrame | None":
    """The whole run's rows, ``condition`` whole_run, and every condition's as one table.

    ::

      whole [chromophore, label, coherence], by_condition [chromophore, condition, label, ...]
        -> [chromophore, condition, label, coherence], whole-run rows first
    """
    # with no condition rows the table keeps no condition column, which says as much
    whole = [w.assign(condition=WHOLE_RUN) for w in whole] if by_condition else list(whole)
    frames = whole + list(by_condition)
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    if by_condition:
        # the condition rows' column order, which keeps `condition` up front
        front = list(by_condition[0].columns)
        table = table[front + [c for c in table.columns if c not in front]]
    return table


def split_whole_run(frame: "pd.DataFrame | None",
                    ) -> "tuple[pd.DataFrame | None, pd.DataFrame | None]":
    """A table's whole-run rows and its condition rows, each None when it has none.

    ::

      condition [whole_run, whole_run, game, rest]  ->  (2 rows, no condition column), (2 rows)
      no condition column                           ->  (every row, None)
    """
    if frame is None:
        return None, None
    if "condition" not in frame.columns:
        return (frame if not frame.empty else None), None
    windowed = frame["condition"] != WHOLE_RUN
    whole = frame[~windowed].drop(columns="condition")
    by_condition = frame[windowed]
    return (whole if not whole.empty else None), (by_condition if not by_condition.empty
                                                  else None)
