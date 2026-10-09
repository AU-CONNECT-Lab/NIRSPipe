"""One finished run of the group commands on the dyad fingerprint: where everything went, and the truth."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from html.parser import HTMLParser
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from fnirs_pipe.io.snirf import read_snirf
from tests._dyad_fingerprint import COUPLING_BAND, TASK, GroupTruth

BAND = COUPLING_BAND


@dataclass
class Groups:
    root: Path
    bids: Path
    deriv: Path
    hyper: Path
    truth: GroupTruth
    # every call to a spied builder: (args, kwargs, return value)
    calls: dict = field(default_factory=dict, repr=False)
    _stages: dict = field(default_factory=dict, repr=False)

    # ---- the group tree ----
    def gdir(self, group: str) -> Path:
        return self.hyper / f"group-{group}"

    def page(self, group: str, pair: "tuple[str, str] | None" = None,
             cond: str | None = None, raw: bool = False) -> Path:
        middle = f"_pair-{pair[0].replace('-', '')}x{pair[1].replace('-', '')}" if pair else ""
        middle += f"_cond-{cond}" if cond else ""
        suffix = "_desc-raw_report.html" if raw else "_report.html"
        return self.gdir(group) / f"group-{group}_task-{TASK}{middle}{suffix}"

    def figure(self, group: str, name: str) -> Path:
        """``name`` is everything after the task entity, e.g. ``desc-rawalignment_nirs.html``."""
        return self.gdir(group) / "figures" / f"group-{group}_task-{TASK}_{name}"

    def table(self, group: str, name: str) -> pd.DataFrame:
        return pd.read_csv(self.gdir(group) / "nirs" / f"group-{group}_task-{TASK}_{name}",
                           sep="\t")

    def record(self, group: str) -> dict:
        path = self.gdir(group) / "nirs" / f"group-{group}_task-{TASK}_desc-sqm_qc.json"
        return json.loads(path.read_text())

    # ---- each member's own derivatives ----
    def stage_path(self, sid: str, desc: str) -> Path:
        return self.deriv / sid / "nirs" / f"{sid}_task-{TASK}_desc-{desc}_nirs.snirf"

    def read(self, sid: str, desc: str) -> mne.io.Raw:
        """A member's stage file with its rejection marks, on its own clock."""
        key = (sid, desc)
        if key not in self._stages:
            self._stages[key] = read_snirf(self.stage_path(sid, desc), verbose="error")
        return self._stages[key]

    def sidecar(self, sid: str, desc: str) -> dict:
        return json.loads(self.stage_path(sid, desc).with_suffix(".json").read_text())

    def member_record(self, sid: str) -> dict:
        from fnirs_pipe.qc.subject.record_io import read_record
        return read_record(self.deriv / sid / "nirs" / f"{sid}_task-{TASK}_desc-sqm_qc.json")

    # ---- the channel coherence maps, which are PNGs: their builder's inputs ----
    def map_inputs(self, path: Path) -> tuple:
        """The arguments the builder was handed for the PNG written at ``path``."""
        if "png" not in self._stages:
            self._stages["png"] = {hashlib.sha1(base64.b64decode(b64)).digest(): (args, kwargs)
                                   for args, kwargs, b64 in self.calls.get("build_wtc_channel", [])
                                   if b64}
        found = self._stages["png"].get(hashlib.sha1(path.read_bytes()).digest())
        assert found, f"no captured call drew {path.name}"
        return found


def wtc_tables(groups: Groups, group: str) -> pd.DataFrame:
    """The channel band means of the whole run and every condition, one frame."""
    whole = groups.table(group, "stat-wtc_relmat.tsv").assign(condition="")
    cond = groups.table(group, "cond-all_stat-wtc_relmat.tsv")
    return pd.concat([whole, cond], ignore_index=True)


def band_mean(wtc: np.ndarray, coi: np.ndarray, freqs: np.ndarray, band=BAND) -> float:
    """The coherence of one map over the band, outside the cone, as the tables take it."""
    rows = (freqs >= band[0]) & (freqs <= band[1])
    with np.errstate(divide="ignore"):
        edge = np.where(np.asarray(coi) > 1e-10, 1.0 / np.asarray(coi), np.inf)
    keep = freqs[rows][:, None] >= edge[None, :]
    return float(np.nanmean(np.where(keep, np.asarray(wtc)[rows], np.nan)))


class _Tables(HTMLParser):
    """Every <table> on a page as rows of cell text, keyed by the heading last seen before it."""

    def __init__(self):
        super().__init__()
        self.tables: list[tuple[str, list[list[str]]]] = []
        self._heading, self._in_heading = "", False
        self._rows: "list | None" = None
        self._cell: "list | None" = None

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag in ("h2", "h3") or (tag == "p" and "qm-sub" in cls):
            self._in_heading, self._heading = True, ""
        elif tag == "table":
            self._rows = []
        elif tag == "tr" and self._rows is not None:
            self._rows.append([])
        elif tag in ("td", "th") and self._rows is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("h2", "h3", "p") and self._in_heading:
            self._in_heading = False
        elif tag in ("td", "th") and self._cell is not None:
            self._rows[-1].append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "table" and self._rows is not None:
            self.tables.append((self._heading.strip(), self._rows))
            self._rows = None

    def handle_data(self, data):
        if self._in_heading:
            self._heading += data
        if self._cell is not None:
            self._cell.append(data)


def html_tables(path: Path) -> list[tuple[str, list[list[str]]]]:
    parser = _Tables()
    parser.feed(Path(path).read_text(encoding="utf-8"))
    return parser.tables


def table_under(path: Path, heading: str) -> list[list[str]]:
    """The first table after a heading that starts with ``heading``."""
    return next(rows for head, rows in html_tables(path) if head.startswith(heading))


_ENTITY = re.compile(r"([a-z]+)-([^_]+)")


def entities(path: Path) -> dict:
    return dict(_ENTITY.findall(path.name.split("_desc-")[0]))


def js_var(html: str, name: str):
    """The JSON literal the page assigns to ``NAME`` (``var NAME =`` or ``window.NAME =``)."""
    for match in re.finditer(rf"(?:var|const|let|window\.)\s*{name}\s*=\s*", html):
        try:
            value, _ = json.JSONDecoder().raw_decode(html, match.end())
        except json.JSONDecodeError:
            continue                 # an assignment from another variable, not the literal
        return value
    raise AssertionError(f"no {name} literal on the page")
