"""One delimiter rule for every table a user hands the package."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("io.tables")

# Extensions that name their own delimiter. Anything else is sniffed.
_BY_EXTENSION = {".tsv": "\t", ".csv": ","}

_CANDIDATES = ("\t", ",", ";", "|")


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
        return pd.read_csv(path, sep="	")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("unreadable table, %s: %s (%s)", lost, path.name, exc)
        return None
