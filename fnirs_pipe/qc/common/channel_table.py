"""Per-channel QC rows, assembled once for every view that prints them.

The subject report, the raw viewer and the GUI all answer the same question channel by
channel: what did this channel score, was it kept, and which separation block is it in.
Each of them used to assemble that answer itself, from different halves of the record, so
the three disagreed: the raw viewer read one metric where the report read five, the GUI
looked up SCI under a key the raw recording does not use and printed a dash for every
channel, and only the report knew about the long/short split at all.

Everything here reads an SQM record and returns plain dicts. Nothing computes a metric and
nothing renders: the record is the single measurement, and a view that wants a number it
does not carry is asking for a metric that was never stored.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.channel_table")

# the per-channel families that decide whether a channel was measured by a section at all
_PER_CHANNEL_KEYS = (
    "sci_per_channel", "psp_per_channel", "good_frac_per_channel",
    "snr_per_channel", "cv_per_channel",
)

# the sections whose per-channel dicts describe the long channels, merged in this order so
# the split values win over the whole-file ones. It mirrors the scalar merge in the subject
# report's SQM section; a raw-only record simply carries none of the later ones.
_LONG_MERGE_SECTIONS = ("motion", "preproc", "preproc_long", "censor")


def _pair_of(ch: str) -> str:
    """Channel name without its wavelength or chromophore suffix.

    "S1_D1 760" -> "S1_D1",  "S1_D1 hbo" -> "S1_D1"

    Raw intensity names its channels by wavelength and haemoglobin names them by
    chromophore, so a metric stored under one naming is looked up from the other only
    through here.
    """
    return re.sub(r"\s+(\d+|hbo|hbr)$", "", ch, flags=re.IGNORECASE)


def _long_per_channel(record: dict) -> dict:
    """The per-channel dicts that describe the long channels, merged into one.

    ``raw_long`` where the montage was split, ``raw`` where it was not, with the later
    sections layered on top so a metric measured after Beer-Lambert is found here too.
    """
    per_channel = record.get("per_channel") or {}
    raw_key = "raw_long" if "raw_long" in record else "raw"
    merged: dict = {}
    for key in (raw_key, *_LONG_MERGE_SECTIONS):
        merged.update(per_channel.get(key) or {})
    return merged


def channel_rows(
    record: dict,
    sci_scores: dict[str, float],
    bad_channels: Iterable[str],
) -> list[dict[str, Any]]:
    """One row per channel: its scores, its verdict, and which separation block it is in.

    Rows follow the order of ``sci_scores``, which is acquisition order, and every value is
    read from the record rather than recomputed, so a table and the stored numbers cannot
    disagree.

    A short channel's scores come from ``raw_short`` and a long channel's from the long
    sections, because the two are measured separately: averaging a short channel's coupling
    in with the long ones lifts every score, and reading a short channel's row off the long
    section would leave it blank. A channel in neither range falls back to the whole-montage
    ``raw`` section, which measured it even though no split claimed it; its scores are the
    same per-channel numbers either way, and only the scalar means they feed differ.
    ``corr`` is the exception and comes from the whole-file ``preproc`` section, so it is
    filled for short channels too.
    """
    per_channel = record.get("per_channel") or {}
    short_pc = per_channel.get("raw_short") or {}
    long_pc = per_channel.get("raw_long") or {}
    whole_pc = per_channel.get("raw") or {}
    merged_pc = _long_per_channel(record)
    # the whole-file section, not the long split: this column is the one short channels have
    corr_pc = (per_channel.get("preproc") or {}).get("hbo_hbr_corr_per_channel") or {}
    bad = set(bad_channels)

    def _named(pc: dict) -> set:
        return {ch for key in _PER_CHANNEL_KEYS for ch in (pc.get(key) or {})}

    short_names, long_names = _named(short_pc), _named(long_pc)

    def separation_of(ch: str) -> str:
        """Which block a channel belongs to, or "" when the record was never split.

        The two separation ranges do not meet, so a channel at 12 mm is in neither list.
        Those get their own name rather than falling into the long block, where a row of
        dashes would read as a long channel whose metrics failed.
        """
        if not long_names:
            return ""
        if ch in short_names:
            return "short"
        return "long" if ch in long_names else "unclassified"

    def section_for(sep: str) -> dict:
        if sep == "short":
            return short_pc
        # an unclassified channel is in neither split but the whole-montage section
        # measured it, and without those scores its rejection prints no reason
        return whole_pc if sep == "unclassified" else merged_pc

    def row_of(ch: str) -> dict[str, Any]:
        sep = separation_of(ch)
        pc = section_for(sep)

        def value_of(key: str):
            return (pc.get(key) or {}).get(ch)

        return {
            "name":       ch,
            "sci":        value_of("sci_per_channel"),
            "sci_win":    value_of("sci_win_per_channel"),
            "psp":        value_of("psp_per_channel"),
            "good_frac":  value_of("good_frac_per_channel"),
            "snr":        value_of("snr_per_channel"),
            "cv":         value_of("cv_per_channel"),
            "spike":      value_of("spike_pct_per_channel"),
            "corr":       corr_pc.get(_pair_of(ch)),
            "is_bad":     ch in bad,
            "separation": sep,
        }

    return [row_of(ch) for ch in sci_scores]


def pair_rows(rows: list[dict], pairs: list[str] | None = None) -> list[dict[str, Any]]:
    """The same rows collapsed to one per source-detector pair.

    Raw intensity carries two channels per pair, one per wavelength, and SCI is a property
    of the pair rather than of either wavelength. A view that lists pairs (the channel
    decisions tables) needs one row each, and looking a pair's score up by guessing at a
    suffix is what made the GUI print a dash for every channel::

        rows for "S1_D1 760" and "S1_D1 850"  ->  one row, pair "S1_D1"

    ``pairs`` fixes the order and the membership when the caller already has a pair list
    from the haemoglobin file; without it the pairs come out in the order they appear.
    A pair with no row is still returned, empty, so the caller's list stays intact.
    """
    by_pair: dict[str, list[dict]] = {}
    for row in rows:
        by_pair.setdefault(_pair_of(row["name"]), []).append(row)

    def _first(group: list[dict], key: str):
        for row in group:
            if row.get(key) is not None:
                return row[key]
        return None

    order = pairs if pairs is not None else list(by_pair)
    out = []
    for pair in order:
        group = by_pair.get(pair, [])
        out.append({
            "pair":       pair,
            "sci":        _first(group, "sci"),
            "psp":        _first(group, "psp"),
            "snr":        _first(group, "snr"),
            "cv":         _first(group, "cv"),
            "spike":      _first(group, "spike"),
            "corr":       _first(group, "corr"),
            # either wavelength failing is the pair failing, which is how screening treats it
            "is_bad":     any(row["is_bad"] for row in group),
            "separation": group[0]["separation"] if group else "",
        })
    return out


def _neither_range_title(sep_bands=None) -> str:
    from fnirs_pipe.qc.metrics import unclaimed_separations
    return f"Neither range ({unclaimed_separations(sep_bands)})"


def separation_blocks(rows: list[dict], sep_bands=None) -> list[tuple[str, list[dict]]]:
    """Rows grouped by separation, long first, empty blocks dropped.

    A single block back means the montage is of one kind (or was never split), which is the
    case where a view prints one plain table with no group headers.
    """
    groups = [("Long channels", [r for r in rows if r.get("separation") in ("long", "")]),
              ("Short channels", [r for r in rows if r.get("separation") == "short"]),
              (_neither_range_title(sep_bands),
               [r for r in rows if r.get("separation") == "unclassified"])]
    return [(title, block) for title, block in groups if block]


def heatmap_args(rows: list[dict]) -> dict[str, Any]:
    """Arguments for :func:`channel_quality_heatmap`, ordered the way a table reads.

    Long block first so the grid and the table above it list channels in the same order,
    with a stable sort so acquisition order survives inside each block. ``split_at`` is the
    divider between long and short, and it names its two sides, so it is only set when there
    are exactly two: a montage with channels in neither range has three blocks, and the
    table is where those get named.
    """
    order = {"long": 0, "": 0, "short": 1, "unclassified": 2}
    ordered = sorted(rows, key=lambda r: order.get(r.get("separation"), 2))
    n_long = sum(1 for r in ordered if r.get("separation") == "long")
    n_odd = sum(1 for r in ordered if r.get("separation") == "unclassified")
    return {
        "ch_names":   [r["name"] for r in ordered],
        "is_bad":     [r["is_bad"] for r in ordered],
        # the row that decides Status, so the grid can say why a channel was rejected
        "good_frac_per_ch": {r["name"]: r["good_frac"] for r in ordered
                             if r.get("good_frac") is not None},
        # the windowed estimator, matching every other row here and the screening itself
        "sci_per_ch": {r["name"]: v for r in ordered
                       if (v := r.get("sci_win") or r.get("sci")) is not None},
        "cv_per_ch":  {r["name"]: r["cv"] for r in ordered if r.get("cv") is not None},
        "snr_per_ch": {r["name"]: r["snr"] for r in ordered if r.get("snr") is not None},
        "psp_per_ch": {r["name"]: r["psp"] for r in ordered if r.get("psp") is not None},
        "split_at":   n_long if not n_odd and 0 < n_long < len(ordered) else None,
    }


def registration_note(offset: "tuple[float, float] | None") -> "str | None":
    """What to say when the optode positions were never registered to the head.

    ``offset`` is :func:`fnirs_pipe.qc.metrics.registration_offset`' output, and None means
    there is nothing to say. One wording for both views, as the separation notes are.

    Worth its own note rather than a figure caption because the figures it invalidates are
    the ones a reader trusts on sight: a cloud of optodes drawn beside a brain reads as a
    montage that reaches an unusual part of it, not as a montage in the wrong frame.
    """
    if offset is None:
        return None
    reach, scalp = offset
    return (
        f"The optode positions are not registered to this recording's head coordinates: "
        f"they sit a median of {reach:.0f} mm from the head centre while the fiducials put "
        f"the scalp at {scalp:.0f} mm. Everything drawn from positions, the 3-D views, the "
        f"flat maps and the topographies, is therefore not anatomical, and no channel can "
        f"be placed on a brain region. Separations are measured between optodes rather "
        f"than against the head, so the long / short split, the screening and every metric "
        f"built on them are unaffected. Fixing it needs the digitised nasion and "
        f"preauricular points the recording was taken with, or a standard montage put in "
        f"their place; neither can be recovered from the file itself."
    )


def separation_notes(
    scalars: dict,
    rows: list[dict],
    short_channel_requested: bool = False,
    sep_bands=None,
    orphan_mm: "dict[str, float] | None" = None,
) -> list[str]:
    """What to say when the montage could not be split the way the metrics assume it was.

    Three cases, and they are worth telling apart. No channel in either range means the
    recording carries no registered optode positions, so every distance reads as zero and
    the metrics fall back to the whole montage. Channels in neither range is a real montage
    with real positions that happens to use separations the two ranges leave out; those
    channels are screened and scored like any other but sit in none of the split scalars.
    The third is a run that asked for
    short-channel regression and had no short channel to build it from, which the pipeline
    treats as a warning and carries on past.

    ``orphan_mm`` is :func:`separation_orphans`' output, name -> mm. Given, the second note
    also says where those channels actually sit and which bound would take them in, which
    is the difference between a reader knowing the gap exists and knowing what to do about
    it: the gap is a package default, the separations are this montage's.

    Returns the notes in the order they should be printed, empty when the split was clean.
    The caller decides where they go: the subject report files them as run notes, the raw
    views print them under the table.
    """
    notes: list[str] = []
    n_long, n_short = scalars.get("n_long_channels"), scalars.get("n_short_channels")
    if n_long == 0 and n_short == 0:
        notes.append(
            "No channel fell in either separation range, which is what a recording with "
            "no registered optode positions looks like. The quantitative metrics are over "
            "every channel rather than long channels only, and the per-channel table is "
            "not grouped. Anything that needs positions, including short-channel "
            "regression and the topographies, is unavailable for this run.")
        return notes

    n_odd = sum(1 for r in rows if r.get("separation") == "unclassified")
    if n_odd:
        from fnirs_pipe.qc.metrics import unclaimed_separations
        where = ""
        if orphan_mm:
            import math
            lo, hi = min(orphan_mm.values()), max(orphan_mm.values())
            span = f"{lo:.1f} mm" if hi - lo < 0.05 else f"{lo:.1f} to {hi:.1f} mm"
            # ceil for the short bound, floor for the long one: a bound has to reach past
            # every orphan to take them all in, and rounding the other way excludes one
            where = (f" Theirs sit at {span}, so --short-max-dist {math.ceil(hi)} would make "
                     f"them short channels and --long-min-dist {math.floor(lo)} would make "
                     f"them long; which is right depends on how deep this montage's short "
                     f"end reaches, which the separations alone do not settle.")
        notes.append(
            f"{n_odd} channel(s) sit at a separation the long and short ranges leave out "
            f"({unclaimed_separations(sep_bands)}). They were screened and their row in the "
            f"per-channel table carries the whole-montage scores, but no split claims them, "
            f"so they are in none of the long or short scalar metrics.{where}")

    if short_channel_requested:
        short_rows = [r for r in rows if r.get("separation") == "short"]
        if not n_short:
            notes.append(
                "Short-channel regression was requested but this montage carries no short "
                "channel, so it did not run and no systemic signal was regressed out.")
        elif short_rows and all(r["is_bad"] for r in short_rows):
            notes.append(
                f"Short-channel regression was requested but all {len(short_rows)} short "
                f"channels were rejected, so it did not run. Their scores are in the "
                f"per-channel table.")
    return notes


# The scalar metric each per-channel column is the same quantity as, which is where its
# number format comes from. A per-channel SCI printed to three decimals in one view and two
# in the next is the drift this mapping exists to stop.
_COLUMN_METRIC = {
    "sci":   "sci_mean",
    "psp":   "psp_mean",
    "snr":   "snr_mean",
    "cv":    "cv_mean",
    "spike": "spike_pct",
    "corr":  "hbo_hbr_corr_mean",
}

# ---- Columns ----
# Column order and headers for every view that prints per-channel rows. This is the only
# place they are written: the subject report, the raw viewer, the GUI and the CSV all take
# their columns from here through :func:`channel_columns`, so a column added once is added
# everywhere and cannot come out in a different order in the next table down.
#
# ``separation`` is last because most views drop it: they group by separation instead, and
# the block header already says which side a row is on.
CHANNEL_COLUMNS = (
    ("name",       "Channel"),
    ("status",     "Status"),
    ("sci",        "SCI"),
    ("psp",        "PSP"),
    ("snr",        "SNR (intensity)"),
    ("cv",         "CV"),
    ("spike",      "Spike % (exp.)"),
    ("corr",       "HbO–HbR corr"),
    ("separation", "Separation"),
)


def channel_columns(drop: "Sequence[str]" = ()) -> list[tuple[str, str]]:
    """:data:`CHANNEL_COLUMNS` without the columns a view does not print.

    channel_columns(("corr", "separation")) -> [("name", "Channel"), ..., ("spike", ...)]

    A view drops a column rather than listing the ones it wants, so a new column reaches it
    by default and the lists cannot silently fall out of step.
    """
    dropped = set(drop)
    return [(key, label) for key, label in CHANNEL_COLUMNS if key not in dropped]


# Every column a table prints, minus the derived Status, plus the three raw values no table
# has room for: the coupled-window share the verdict rests on, the verdict itself as a flag,
# and the criterion it failed.
CSV_FIELDS = (*(key for key, _ in channel_columns(("status",))),
              "good_frac", "is_bad", "reason")


# The optical-density metrics the "one file, three channel sets" table prints, in column
# order. Shared because the subject report and the raw viewer both draw that table and a
# column added to one and not the other is a difference a reader reads as a finding.
OD_SPLIT_COLUMNS = (
    ("channel_retention_rate", "Channel retention"),
    ("sci_win_mean",           "Mean SCI (10 s)"),
    ("sci_mean",               "Mean SCI (whole run)"),
    ("good_frac_mean",         "Coupled windows"),
    ("psp_mean",               "Mean PSP (10 s)"),
    ("snr_mean",               "Mean SNR (10 s)"),
    ("cv_mean",                "Mean CV (10 s)"),
    ("mean_amp_mean",          "Mean amplitude"),
    ("spike_pct",              "Spike share"),
    # a sum over the set's channels, so it is read against the row's channel count rather
    # than against another row
    ("spike_count",            "Spike count"),
)

# The motion table's columns, a separate table rather than more of the one above because
# nearly all of these measure a *set* rather than group per-channel numbers: GVTD is an RMS
# across channels, and the frame counts ask how many of *these* channels were flagged at
# once, so each set carries its own trace, its own cutoff and its own bar. Read those down a
# column. "Corrected per channel" is the exception, a mean over the set, and does compare
# across rows. Here rather than in either template, so the raw viewer and the subject report
# cannot end up listing different metrics for the same recording.
MOTION_SPLIT_COLUMNS = (
    ("gvtd_mean",             "GVTD mean"),
    ("gvtd_p95",              "GVTD p95"),
    ("gvtd_filt_mean",        "GVTD mean 0.01-0.5 Hz"),
    ("gvtd_filt_p95",         "GVTD p95 0.01-0.5 Hz"),
    ("gvtd_pct_above_thresh", "GVTD % motion"),
    ("gvtd_num_above_thresh", "GVTD motion frames"),
    ("gvtd_thresh",           "GVTD threshold"),
    ("spike_pct_frames",      "Spike % frames"),
    ("spike_num_frames",      "Spike frames"),
    # "per channel" is the average share of each channel the correction altered; "% frames"
    # is the share of timepoints where it altered at least a tenth of the set. Different
    # questions, which is why both are here under names that do not read as the same one
    ("motion_corrected_frac_mean", "Corrected per channel"),
    ("motion_corrected_pct",       "Corrected % frames"),
    ("motion_corrected_num",       "Corrected frames"),
    ("motion_corrected_n_segments", "Corrected segments"),
)

# What a per-condition page drops from the two lists above rather than leaving blank in all
# three rows: none of these has a windowed series to slice, and the GVTD threshold is a mode
# of the whole run's histogram by definition, so a condition is counted against the run's
# line. Shared so the subject report's condition pages and the raw viewer's drop the same.
WHOLE_RUN_ONLY_COLUMNS = frozenset({
    "mean_amp_mean", "spike_pct", "spike_count", "gvtd_thresh",
    "motion_corrected_frac_mean",
    # one estimate over the whole recording, with no windowed series to take a condition's
    # columns out of. The 10 s estimate beside it is the one a condition can have
    "sci_mean",
})


def _cell(key: str, scalars: dict, colour: bool) -> dict:
    """One table cell, with the after value beside it where the row carries one."""
    from fnirs_pipe.qc.boilerplate.vocabulary import format_metric, metric_class

    before, after = scalars.get(key), scalars.get(f"{key}_post")
    cell = {"value": format_metric(key, before),
            "cls": metric_class(key, after if after is not None else before) if colour else ""}
    if after is not None:
        cell["post"] = format_metric(key, after)
    return cell


def measured_columns(
    columns: "Sequence[tuple[str, str]]", *scalar_sets: dict,
) -> "tuple[tuple[str, str], ...]":
    """The columns at least one channel set has a value for.

    ::

      measured_columns([("gvtd_mean", "GVTD"), ("motion_corrected_frac_mean", "Corrected")],
                       {"gvtd_mean": 0.004}, {"gvtd_mean": 0.009})
      -> (("gvtd_mean", "GVTD"),)

    A column nothing measured is dropped rather than printed as a row of dashes: the
    correction footprint is there only when a correction ran.
    """
    return tuple((key, label) for key, label in columns
                 if any((s or {}).get(key) is not None for s in scalar_sets))


def split_table(
    channel_sets: list[tuple[str, Any, dict, bool]],
    columns: "Sequence[tuple[str, str]]" = OD_SPLIT_COLUMNS,
) -> dict[str, Any]:
    """One recording measured over several channel sets, as a table ready to print.

    ``channel_sets`` is ``[(name, n_channels, scalars, colour), ...]`` -- normally All, Long
    and Short. ``colour`` says whether that row's cells carry a verdict: only the long
    channels get one, because a short channel's coupling is high by construction and the
    published cutoffs were never set for it, so colouring its row would call a number good
    against a threshold that does not apply::

        split_table([("Long", 40, {"sci_mean": 0.81}, True)], [("sci_mean", "Mean SCI")])
        -> columns: [{"key": "sci_mean", "label": "Mean SCI", ...}]
           rows:    [{"name": "Long", "n": 40, "cells": [{"value": "0.810", "cls": "qm-ok"}]}]
    """
    from fnirs_pipe.qc.boilerplate.vocabulary import is_key_metric, metric_summary

    columns = measured_columns(columns, *(s for _, _, s, _ in channel_sets))
    return {
        "columns": [{"key": key, "label": label, "tip": metric_summary(key),
                     "key_metric": is_key_metric(key)} for key, label in columns],
        "rows": [{
            "name": name,
            "n": n_channels,
            # `post` is the same metric measured after a processing step, present only where
            # the caller merged one in under a `_post` suffix. The verdict colour follows the
            # after value: it is the state the data is left in.
            "cells": [_cell(key, scalars, colour) for key, _ in columns],
        } for name, n_channels, scalars, colour in channel_sets],
    }


def _failed_criteria(row: dict, cutoffs: dict[str, float]) -> list[str]:
    """Which screening criteria this row's own scores fail, or ["manual"].

    _failed_criteria({"sci": 0.95, "psp": 0.02}, {"sci": 0.8, "psp": 0.1}) -> ["PSP"]

    Derived rather than stored: the record already carries every criterion's per-channel
    score, so asking it again costs nothing and cannot disagree with the screening that
    produced the verdict. A rejected channel that fails nothing was rejected by hand, which
    is only claimed when every criterion actually has a score to judge it by -- an
    unmeasured criterion is not evidence of a manual rejection.
    """
    from fnirs_pipe.qc.metrics import CRITERIA

    failed, scored = [], True
    for c in CRITERIA:
        # a criterion that rejects nothing is not a reason a channel went
        if not c.screens:
            continue
        value = row.get(c.name)
        if value is None:
            scored = False
            continue
        # one throwaway channel, so the criterion's own comparison decides rather than a
        # second copy of it here
        if c.failures({"_": value}, cutoffs.get(c.name)):
            failed.append(c.label)
    if failed:
        return failed
    return ["manual"] if scored else []


def format_rows(
    rows: list[dict],
    sci_threshold: float | None = None,
    *,
    name_key: str = "name",
    psp_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Rows with the numbers already formatted and the cells already classed.

    format_rows([{"name": "S1_D1 760", "sci": 0.412, "corr": None, "is_bad": True, ...}], 0.8)
    -> [{"name": "S1_D1 760", "status": "BAD", "status_cls": "bad",
         "sci": "0.412", "sci_cls": "bad", "corr": "\u2014", "corr_cls": "", ...}]

    Formatting lives here rather than in each renderer because two of the three cannot use
    the Python registry directly: the raw viewer prints in JavaScript and the GUI builds
    Dash components, so both get their strings from here instead of reimplementing the
    formats and the SCI cutoff. Classes are the subject report's ``.qm-table`` names
    (``bad``/``good`` for status, ``bad`` for a below-threshold SCI, ``neg``/``pos`` for the
    correlation's sign), so one stylesheet rule covers all three views.

    ``sci_threshold`` is the run's own ``--sci-threshold``, not the registry's cutoff for
    the SCI mean: this column marks the channels that were actually pruned, so colouring it
    against anything else would show a verdict the run did not reach. ``psp_threshold`` is
    the run's other line, and is here for the same reason: the reason printed beside a
    rejected channel is re-derived from its scores, so a run that moved the PSP line has to
    hand that line over or the reason comes out naming the wrong criterion.
    ``name_key`` is ``"pair"`` for rows that came through :func:`pair_rows`.

Status names the criterion a rejected channel failed. Screening is a union, so a channel
    can be BAD with a passing SCI cell; without the reason printed beside it that reads as a
    contradiction rather than as a PSP failure.
    """
    from fnirs_pipe.qc.boilerplate.vocabulary import format_metric
    from fnirs_pipe.qc.metrics import SCI_PASS, resolve_cutoffs

    if sci_threshold is None:
        sci_threshold = SCI_PASS
    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold)
    out = []
    for row in rows:
        why = _failed_criteria(row, cutoffs) if row["is_bad"] else []
        formatted: dict[str, Any] = {
            "name":       row.get(name_key),
            "status":     ("BAD (" + "/".join(why) + ")" if why
                           else "BAD" if row["is_bad"] else "OK"),
            "status_cls": "bad" if row["is_bad"] else "good",
            "reason":     "/".join(why),
            "separation": row.get("separation") or "",
            "is_bad":     row["is_bad"],
        }
        for column, metric in _COLUMN_METRIC.items():
            value = row.get(column)
            formatted[column] = format_metric(metric, value)
            formatted[f"{column}_cls"] = ""
        if row.get("sci") is not None and row["sci"] < sci_threshold:
            formatted["sci_cls"] = "bad"
        if row.get("corr") is not None:
            formatted["corr_cls"] = "neg" if row["corr"] < 0 else "pos"
        out.append(formatted)
    return out


def save_channel_csv(rows: list[dict], label: str, out_dir: Path,
                     sci_threshold: float | None = None,
                     psp_threshold: float | None = None) -> None:
    """Per-channel metrics for one run. The name carries the run's entities, or a subject
    with several tasks would keep only whichever ran last.

    ``reason`` names the criterion a rejected channel failed, so the CSV answers "why did
    this channel go" without the reader re-deriving it from the score columns.
    """
    if not rows:
        return
    reasons = {r["name"]: r["reason"]
               for r in format_rows(rows, sci_threshold, psp_threshold=psp_threshold)}
    rows = [{**r, "reason": reasons.get(r["name"], "")} for r in rows]
    out_path = Path(out_dir) / f"{label}_channel_metrics.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%s | channel metrics CSV saved: %s", label, out_path)
