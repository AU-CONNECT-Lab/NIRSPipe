"""Per-channel QC rows, assembled once for every view that prints them.

The subject report, the raw viewer and the GUI all answer the same question channel by
channel: what did this channel score, was it kept, and which separation block is it in.
Assembled here once, so the three cannot read different halves of the record and disagree.

Everything here reads an SQM record and returns plain dicts. Nothing computes a metric and
nothing renders: the record is the single measurement, and a view that wants a number it
does not carry is asking for a metric that was never stored.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from nirspipe.io.optode_space import placement_of
from nirspipe.io.tables import column_entries, write_tsv
from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.qc.boilerplate.vocabulary import (
    CHANNEL_QC_COLUMNS, MISSING_VALUE, condition_label, metric_label,
)
from nirspipe.qc.metrics._helpers import bands_from_record, in_long_band, in_short_band
from nirspipe.qc.metrics.coupling import CV_WINDOW_S, PSP_WINDOW_S, SCI_WINDOW_S
from nirspipe.utils import bare_roi_map, pair_of, pair_order, roi_overlaps
from nirspipe.utils.logging import get_logger

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


def uncorrected_view(record: dict) -> dict:
    """``record`` as raw QC reads it: :func:`channel_rows` takes the correlation before the correction."""
    per_channel = record.get("per_channel") or {}
    return {**record, "per_channel": {k: v for k, v in per_channel.items() if k != "preproc"}}


def channel_rows(
    record: dict,
    sci_scores: dict[str, float],
    bad_channels: Iterable[str],
) -> list[dict[str, Any]]:
    """One row per channel: its scores, its verdict, and which separation block it is in.

    Rows go by source then detector number, a pair's wavelengths side by side
    (:func:`channel_order`), and every value is read from the record rather than recomputed,
    so a table and the stored numbers cannot disagree.

    A short channel's scores come from ``raw_short`` and a long channel's from the long
    sections, because the two are measured separately: averaging a short channel's coupling
    in with the long ones lifts every score, and reading a short channel's row off the long
    section would leave it blank. A channel in neither range falls back to the whole-montage
    ``raw`` section, which measured it even though no split claimed it; its scores are the
    same per-channel numbers either way, and only the scalar means they feed differ.
    ``corr`` is the exception and comes from the whole-file ``preproc`` section, so it is
    filled for short channels too. Raw QC's views leave ``preproc`` out, through
    :func:`uncorrected_view`, and read ``rawhaemo``, the uncorrected side. ``corr_post`` is the same correlation on
    the regression residual (``errts``), None where the record has none.
    """
    per_channel = record.get("per_channel") or {}
    short_pc = per_channel.get("raw_short") or {}
    long_pc = per_channel.get("raw_long") or {}
    whole_pc = per_channel.get("raw") or {}
    merged_pc = _long_per_channel(record)
    # the whole-file section, not the long split: this column is the one short channels have
    corr_pc = (per_channel.get("preproc") or per_channel.get("rawhaemo")
               or {}).get("hbo_hbr_corr_per_channel") or {}
    corr_post_pc = (per_channel.get("errts") or {}).get("hbo_hbr_corr_per_channel") or {}
    share_pc = (per_channel.get("short_regression") or {}).get("short_share_per_channel") or {}
    bad = set(bad_channels)

    def _named(pc: dict) -> set:
        return {ch for key in _PER_CHANNEL_KEYS for ch in (pc.get(key) or {})}

    short_names, long_names = _named(short_pc), _named(long_pc)
    measured_short = set(short_names)
    dist = whole_pc.get("ch_dist_per_channel") or {}
    bands = bands_from_record(record.get("raw") or {})
    if not short_names:
        # a short set rejected whole has no section of its own; its separations still say
        # which channels it held
        short_names = {ch for ch, d in dist.items() if d is not None and in_short_band(d / 1e3, bands)}
    if not long_names:
        # an all-long montage writes no long section, nor does a long set rejected whole
        long_names = {ch for ch, d in dist.items() if d is not None and in_long_band(d / 1e3, bands)}

    def separation_of(ch: str) -> str:
        """Which block a channel belongs to, or "" when no channel has a separation in either range.

        The two separation ranges do not meet, so a channel at 12 mm is in neither list.
        Those get their own name rather than falling into the long block, where a row of
        dashes would read as a long channel whose metrics failed.
        """
        if not long_names and not short_names:
            return ""
        if ch in short_names:
            return "short"
        return "long" if ch in long_names else "unclassified"

    def section_for(sep: str, ch: str) -> dict:
        if sep == "short":
            # the whole-montage section measured a short channel its own section left out
            return short_pc if ch in measured_short else whole_pc
        # an unclassified channel is in neither split but the whole-montage section
        # measured it, and without those scores its rejection prints no reason
        return whole_pc if sep == "unclassified" else merged_pc

    def row_of(ch: str) -> dict[str, Any]:
        sep = separation_of(ch)
        pc = section_for(sep, ch)

        def value_of(key: str):
            return (pc.get(key) or {}).get(ch)

        return {
            "name":       ch,
            "sci_win":    value_of("sci_win_per_channel"),
            "sci_whole":  value_of("sci_per_channel"),
            "psp":        value_of("psp_per_channel"),
            "good_frac":  value_of("good_frac_per_channel"),
            "snr":        value_of("snr_per_channel"),
            "cv":         value_of("cv_per_channel"),
            "spike":      value_of("spike_pct_per_channel"),
            "saturated":  value_of("saturated_per_channel"),
            "corr":       corr_pc.get(pair_of(ch)),
            "corr_post":  corr_post_pc.get(pair_of(ch)),
            "short_share_hbo": share_pc.get(f"{pair_of(ch)} hbo"),
            "short_share_hbr": share_pc.get(f"{pair_of(ch)} hbr"),
            "is_bad":     ch in bad,
            "separation": sep,
        }

    return sorted((row_of(ch) for ch in sci_scores), key=lambda r: channel_order(r["name"]))


def channel_order(ch: str) -> tuple:
    """A sort key: source, then detector number, then wavelength.

    ["S2_D1 850", "S10_D1 760", "S2_D1 760"] -> ["S2_D1 760", "S2_D1 850", "S10_D1 760"]
    """
    wavelength = str(ch).rsplit(" ", 1)[1] if " " in str(ch) else ""
    return pair_order(ch), len(wavelength), wavelength


# The per-channel columns measured on each wavelength on its own; the rest are a pair's
# (SCI and PSP correlate the two wavelengths, HbO-HbR corr is after Beer-Lambert).
WAVELENGTH_COLUMNS = ("snr", "cv", "spike", "saturated")


def pair_wavelengths(rows: list[dict]) -> list[str]:
    """The wavelengths behind the rows, in the order :func:`pair_rows` keeps them.

    ["S1_D1 850", "S1_D1 760"] -> ["760", "850"]
    """
    return sorted({str(r["name"]).rsplit(" ", 1)[1] for r in rows if " " in str(r["name"])},
                  key=lambda w: (len(w), w))


def pair_rows(rows: list[dict], pairs: list[str] | None = None, *,
              by_wavelength: bool = False) -> list[dict[str, Any]]:
    """The same rows collapsed to one per source-detector pair.

    Raw intensity carries two channels per pair, one per wavelength, and SCI is a property
    of the pair rather than of either wavelength. A view that lists pairs (the channel
    decisions tables) needs one row each::

        rows for "S1_D1 760" and "S1_D1 850"  ->  one row, pair "S1_D1"

    ``pairs`` fixes the membership when the caller already has a pair list from the
    haemoglobin file; either way the pairs come out by source then detector number. A pair
    with no row is still returned, empty, so the caller's list stays intact.

    ``by_wavelength`` keeps both values of each :data:`WAVELENGTH_COLUMNS` column as a
    tuple in :func:`pair_wavelengths` order, ``snr: (23.7, 13.7)``, instead of the first.
    """
    by_pair: dict[str, list[dict]] = {}
    for row in rows:
        by_pair.setdefault(pair_of(row["name"]), []).append(row)
    wavelengths = pair_wavelengths(rows) if by_wavelength else []

    def _first(group: list[dict], key: str):
        if key in WAVELENGTH_COLUMNS and wavelengths:
            by_wl = {str(r["name"]).rsplit(" ", 1)[-1]: r.get(key) for r in group}
            return tuple(by_wl.get(w) for w in wavelengths)
        for row in group:
            if row.get(key) is not None:
                return row[key]
        return None

    order = sorted(pairs if pairs is not None else by_pair, key=pair_order)
    out = []
    for pair in order:
        group = by_pair.get(pair, [])
        out.append({
            "pair":       pair,
            "sci_win":    _first(group, "sci_win"),
            "sci_whole":  _first(group, "sci_whole"),
            "psp":        _first(group, "psp"),
            "snr":        _first(group, "snr"),
            "cv":         _first(group, "cv"),
            "spike":      _first(group, "spike"),
            "saturated":  _first(group, "saturated"),
            "corr":       _first(group, "corr"),
            "corr_post":  _first(group, "corr_post"),
            "short_share": (_first(group, "short_share_hbo"), _first(group, "short_share_hbr")),
            # the share a pair is rejected on, so its Status can name the criterion
            "good_frac":  min((row["good_frac"] for row in group
                               if row.get("good_frac") is not None), default=None),
            # either wavelength failing is the pair failing, which is how screening treats it
            "is_bad":     any(row["is_bad"] for row in group),
            "separation": group[0]["separation"] if group else "",
        })
    return out


def _neither_range_title(sep_bands=None) -> str:
    from nirspipe.qc.metrics import unclaimed_separations
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
    with a stable sort so :func:`channel_rows`' order survives inside each block. ``split_at`` is the
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
        "sci_per_ch": {r["name"]: r["sci_win"] for r in ordered if r.get("sci_win") is not None},
        "cv_per_ch":  {r["name"]: r["cv"] for r in ordered if r.get("cv") is not None},
        "snr_per_ch": {r["name"]: r["snr"] for r in ordered if r.get("snr") is not None},
        "psp_per_ch": {r["name"]: r["psp"] for r in ordered if r.get("psp") is not None},
        "split_at":   n_long if not n_odd and 0 < n_long < len(ordered) else None,
    }


def template_note(population: "str | None", template: "str | None") -> "str | None":
    """The note a child run on the adult template gets, or None."""
    from nirspipe.io.head_template import FSAVERAGE

    if population == "child" and (template or FSAVERAGE) == FSAVERAGE:
        return section_note("caveat.child_template")
    return None


def bad_count(bad: "list[str]", n_total: int) -> "tuple[int, int, float, str]":
    """The summary box's count, total, share in percent and badge colour."""
    rate = 100 * len(bad) / n_total if n_total > 0 else 0.0
    badge = "badge-green" if rate < 10 else "badge-yellow" if rate < 30 else "badge-red"
    return len(bad), n_total, rate, badge


def summary_cells(n_bad: int, n_total: int, good_frac: "float | None",
                  heart_rate: "float | None", duration_s: "float | None") -> dict:
    """The run summary's printed cells, formatted here once so both reports round a half alike.

    ::

      (5, 40, 0.85, 1.2, 391.4) -> {"bad": "5/40 (12%)", "frac": "85% of windows",
                                    "heart_rate": "1.20 Hz (72 bpm)", "duration": "391 s"}
    """
    return {
        "bad": f"{n_bad}/{n_total} ({100 * n_bad / n_total:.0f}%)" if n_total else None,
        "frac": None if good_frac is None else f"{good_frac * 100:.0f}% of windows",
        "heart_rate": (None if heart_rate is None
                       else f"{heart_rate:.2f} Hz ({heart_rate * 60:.0f} bpm)"),
        "duration": None if duration_s is None else f"{duration_s:.0f} s",
    }


def heart_rate_note(heart_rate: "float | None", cardiac: "tuple[float, float]") -> "str | None":
    """What to say when the heart rate lies outside the cardiac band, or None."""
    if heart_rate is None or cardiac[0] <= heart_rate <= cardiac[1]:
        return None
    return section_note("caveat.heart_rate_outside_band", hz=heart_rate, bpm=60 * heart_rate,
                        lo=cardiac[0], hi=cardiac[1])


def placement_note(raw) -> "str | None":
    """What to say when no reading placed the optodes on their template, read as the brain views read it.

    None for a placed run, and for one with no positions at all, which the separation notes
    already name. One wording for both views, as the separation notes are.
    """
    locs = np.array([ch["loc"][3:9] for ch in raw.info["chs"]], dtype=float)
    if not np.nan_to_num(locs).any():
        return None
    placement = placement_of(raw)
    if placement.placed:
        return None
    return section_note("caveat.unplaced", reason=placement.note)


def roi_overlap_note(roi_map: "dict[str, list[str]] | None") -> "str | None":
    """What to say when a channel sits in two ROIs, or None. One wording for both reports."""
    shared = roi_overlaps(roi_map or {})
    if not shared:
        return None
    channels = ", ".join(f"{ch} ({', '.join(rois)})" for ch, rois in shared.items())
    return section_note("caveat.roi_overlap", n=len(shared), channels=channels)


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
    short-channel regression and had every short channel rejected, which the pipeline
    treats as a warning and carries on past.

    ``orphan_mm`` is :func:`separation_orphans`' output, name -> mm. Given, the second note
    also says where those channels actually sit and which bound would take them in.

    Returns the notes in the order they should be printed, empty when the split was clean.
    The caller decides where they go: the subject report files them as run notes, the raw
    views print them under the table.
    """
    notes: list[str] = []
    n_long, n_short = scalars.get("n_long_channels"), scalars.get("n_short_channels")
    if n_long == 0 and n_short == 0:
        notes.append(section_note("caveat.no_separation"))
        return notes

    odd = [r for r in rows if r.get("separation") == "unclassified"]
    # a pair, not each of its wavelengths: the raw viewer hands over one row per wavelength
    n_odd = len({r.get("pair") or (pair_of(r["name"]) if r.get("name") else i)
                 for i, r in enumerate(odd)})
    if n_odd:
        from nirspipe.qc.metrics import separation_bands, unclaimed_separations
        where = ""
        short_max, long_min = (sep_bands or separation_bands())[:2]
        # the two flags can only take in channels between the ranges; one past --long-max-dist
        # or at zero separation is not made short or long by moving them
        if orphan_mm and all(1000 * short_max < mm < 1000 * long_min
                             for mm in orphan_mm.values()):
            import math
            lo, hi = min(orphan_mm.values()), max(orphan_mm.values())
            span = f"{lo:.1f} mm" if hi - lo < 0.05 else f"{lo:.1f} to {hi:.1f} mm"
            # ceil for the short bound, floor for the long one: a bound has to reach past
            # every orphan to take them all in, and rounding the other way excludes one
            where = " " + section_note("caveat.unclassified_where", span=span,
                                       short_max=math.ceil(hi), long_min=math.floor(lo))
        notes.append(section_note("caveat.unclassified", n=n_odd,
                                  ranges=unclaimed_separations(sep_bands), where=where))

    if short_channel_requested:
        short_rows = [r for r in rows if r.get("separation") == "short"]
        if short_rows and all(r["is_bad"] for r in short_rows):
            notes.append(section_note("caveat.short_regression_all_bad", n=len(short_rows)))
    return notes


# The scalar metric each per-channel column is the same quantity as, which is where its
# number format comes from. A per-channel SCI printed to three decimals in one view and two
# in the next is the drift this mapping exists to stop.
_COLUMN_METRIC = {
    "sci_win":   "sci_win_mean",
    "sci_whole": "sci_mean",
    "psp":   "psp_mean",
    "snr":   "snr_mean",
    "cv":    "cv_mean",
    "spike": "spike_pct",
    "saturated": "saturated_pct",
    "corr":  "hbo_hbr_corr_mean",
    "short_share": "short_share_hbo",
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
    # read off the run's channel table, so a view without one drops it
    ("region",     "Region (aparc)"),
    ("status",     "Status"),
    # the windowed SCI is the one read; the whole-run correlation beside it is context
    ("sci_win",    f"SCI ({SCI_WINDOW_S:g} s)"),
    ("sci_whole",  "SCI (whole run)"),
    ("psp",        f"PSP ({PSP_WINDOW_S:g} s)"),
    ("snr",        f"SNR μ/σ ({CV_WINDOW_S:g} s)"),
    ("cv",         f"CV ({CV_WINDOW_S:g} s)"),
    ("spike",      "Spike % (exp.)"),
    ("saturated",  "Saturated %"),
    ("corr",       "HbO–HbR corr"),
    # what the short-channel regressors carried, HbO then HbR; only a run that regressed has it
    ("short_share", "Short-channel share"),
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


# Every column a table prints, minus the derived Status and the region the run's channel
# table already holds, the short-channel share split by chromophore, plus the three raw
# values no table has room for: the passing-window
# share the verdict rests on, the verdict itself as a flag, and the criterion it failed.
CSV_FIELDS = (*(key for key, _ in channel_columns(("status", "region", "short_share"))),
              "short_share_hbo", "short_share_hbr", "good_frac", "is_bad", "reason")
# raw QC regresses nothing, so its table has no short-channel share
RAW_CSV_FIELDS = tuple(f for f in CSV_FIELDS if not f.startswith("short_share_"))


# The optical-density metrics the "one file, three channel sets" table prints, in column
# order. Shared because the subject report and the raw viewer both draw that table and a
# column added to one and not the other is a difference a reader reads as a finding.
OD_SPLIT_COLUMNS = tuple((key, metric_label(key)) for key in (
    "channel_retention_rate", "sci_win_mean", "sci_mean", "good_frac_mean", "psp_mean",
    "heart_rate", "snr_mean", "cv_mean", "mean_amp_mean", "n_saturated_channels", "spike_pct",
    # a sum over the set's channels, so it is read against the row's channel count rather
    # than against another row
    "spike_count",
))

# The motion table's columns, a separate table rather than more of the one above because
# nearly all of these measure a *set* rather than group per-channel numbers: GVTD is an RMS
# across channels, and the frame counts ask how many of *these* channels were flagged at
# once, so each set carries its own trace, its own cutoff and its own bar. Read those down a
# column. "Corrected per channel" is the exception, a mean over the set, and does compare
# across rows. Here rather than in either template, so the raw viewer and the subject report
# cannot end up listing different metrics for the same recording.
MOTION_SPLIT_COLUMNS = tuple((key, metric_label(key)) for key in (
    "gvtd_filt_mean", "gvtd_filt_p95", "gvtd_pct_above_thresh", "gvtd_num_above_thresh",
    "gvtd_thresh", "spike_pct_frames", "spike_num_frames",
    # the average share of each channel the correction altered, then the share of timepoints
    # where it altered at least a tenth of the set: two questions, two registry labels
    "motion_corrected_frac_mean", "motion_corrected_pct", "motion_corrected_num",
    "motion_corrected_n_segments",
))

# optical-density metrics no set table has a column for; templates read it as `od_loose_keys`
OD_LOOSE_KEYS = ("cp_mean", "n_flat_channels")

# A share and the count it is a share of, printed in one cell as `12.9% (1098)` so the count
# needs no row of its own. Templates read it as the global `count_beside`.
COUNT_BESIDE = {
    "spike_pct": "spike_count",
    "gvtd_pct_above_thresh": "gvtd_num_above_thresh",
    "spike_pct_frames": "spike_num_frames",
    "motion_corrected_pct": "motion_corrected_num",
}

# The raw viewer's haemoglobin table, on its run page and its condition pages. The subject
# report prints this metric among its own haemoglobin rows instead.
HAEMO_SPLIT_COLUMNS = (
    ("hbo_hbr_corr_mean", "HbO–HbR corr"),
)

# What a per-condition page drops from the two lists above rather than leaving blank in all
# three rows: none of these has a windowed series to slice, and the GVTD threshold is a mode
# of the whole run's histogram by definition, so a condition is counted against the run's
# line. Shared so the subject report's condition pages and the raw viewer's drop the same.
WHOLE_RUN_ONLY_COLUMNS = frozenset({
    "mean_amp_mean", "n_saturated_channels", "spike_pct", "spike_count", "gvtd_thresh",
    "motion_corrected_frac_mean",
    # one estimate over the whole recording, with no windowed series to take a condition's
    # columns out of. The 10 s estimate beside it is the one a condition can have
    "sci_mean",
    # one spectrum over the whole recording
    "heart_rate",
})

# what a condition page's set table prints: the run's columns a condition can have, each
# named for what it is on a stretch
CONDITION_OD_SPLIT_COLUMNS = tuple((key, condition_label(key) or text)
                                   for key, text in OD_SPLIT_COLUMNS
                                   if key not in WHOLE_RUN_ONLY_COLUMNS)


def _cell(key: str, scalars: dict, colour: bool, count_key: "str | None" = None) -> dict:
    """One table cell, with the after value beside it where the row carries one.

    ``count_key`` prints that count after the value, either side: ``7.8% (34330)``.
    """
    from nirspipe.qc.boilerplate.vocabulary import format_metric, metric_class

    def shown(suffix: str) -> str:
        text = format_metric(key, scalars.get(key + suffix))
        count = scalars.get(count_key + suffix) if count_key else None
        return text if count is None else f"{text} ({format_metric(count_key, count)})"

    before, after = scalars.get(key), scalars.get(f"{key}_post")
    cell = {"value": shown(""),
            "cls": metric_class(key, after if after is not None else before) if colour else ""}
    if after is not None:
        cell["post"] = shown("_post")
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
    *,
    fold_counts: bool = False,
) -> dict[str, Any]:
    """One recording measured over several channel sets, as a table ready to print.

    ``channel_sets`` is ``[(name, n_channels, scalars, colour), ...]`` -- normally All, Long
    and Short. ``colour`` says whether that row's cells carry a verdict: only the long
    channels get one, because a short channel's coupling is high by construction and the
    cutoffs do not apply to it::

        split_table([("Long", 40, {"sci_mean": 0.81}, True)], [("sci_mean", "Mean SCI")])
        -> columns: [{"key": "sci_mean", "label": "Mean SCI", ...}]
           rows:    [{"name": "Long", "n": 40, "cells": [{"value": "0.810", "cls": "qm-ok"}]}]

    ``fold_counts`` prints each :data:`COUNT_BESIDE` count inside its share's cell and drops
    the count's own column.
    """
    from nirspipe.qc.boilerplate.vocabulary import is_key_metric, metric_summary, with_unit

    columns = measured_columns(columns, *(s for _, _, s, _ in channel_sets))
    counted = {}
    if fold_counts:
        keys = {key for key, _ in columns}
        counted = {share: count for share, count in COUNT_BESIDE.items()
                   if share in keys and count in keys}
        columns = tuple((key, label) for key, label in columns if key not in counted.values())
    return {
        "columns": [{"key": key, "label": with_unit(key, label), "tip": metric_summary(key),
                     "key_metric": is_key_metric(key)} for key, label in columns],
        "rows": [{
            "name": name,
            "n": n_channels,
            # `post` is the same metric measured after a processing step, present only where
            # the caller merged one in under a `_post` suffix. The verdict colour follows the
            # after value: it is the state the data is left in.
            "cells": [_cell(key, scalars, colour, counted.get(key)) for key, _ in columns],
        } for name, n_channels, scalars, colour in channel_sets],
    }


def metric_set_table(
    sets: "list[tuple[str, Any]]",
    groups: "list[tuple[str, dict, list[dict]]]",
) -> dict[str, Any]:
    """A run's metrics as one table: a metric per row, a channel set per column.

    ``sets`` is ``[(name, n_channels), ...]`` in column order. ``groups`` is
    ``[(title, split, loose), ...]``: ``split`` a :func:`split_table` result (empty when the
    montage did not split), ``loose`` rows from ``metric_rows`` that belong to no channel set,
    printed in the first column with the others blank. A share and its count print in one
    cell, ``7.8% (34330)``, whether they arrive as two loose rows or as two columns of a split
    that was not folded::

        metric_set_table([("All", 52), ("Long", 44)],
                         [("Optical density", split, [{"key": "cp_mean", ...}])])
        -> {"sets": ["All", "Long"], "counts": [52, 44],
            "groups": [{"title": "Optical density",
                        "rows": [{"label": "Mean SCI (10 s)", "cells": [{...}, {...}]}, ...]}]}
    """
    names = [name for name, _ in sets]
    blank = {"value": "", "cls": ""}
    out = []
    for title, split, loose in groups:
        rows = []
        by_set = {r["name"]: r["cells"] for r in (split or {}).get("rows") or []}
        columns = (split or {}).get("columns") or []
        where = {col["key"]: j for j, col in enumerate(columns)}
        counted = {where[c] for share, c in COUNT_BESIDE.items() if share in where and c in where}

        def cell(name: str, j: int, key: str) -> dict:
            if name not in by_set:
                return {"value": MISSING_VALUE, "cls": ""}
            got = dict(by_set[name][j])
            c = where.get(COUNT_BESIDE.get(key, ""))
            # a count with nothing measured adds nothing, so a missing pair prints one dash
            if c is not None and by_set[name][c]["value"] != MISSING_VALUE:
                count = by_set[name][c]
                got["value"] = f"{got['value']} ({count['value']})"
                if "post" in got and "post" in count:
                    got["post"] = f"{got['post']} ({count['post']})"
            return got

        for j, col in enumerate(columns):
            if j in counted:
                continue
            rows.append({"label": col["label"], "tip": col["tip"], "key_metric": col["key_metric"],
                         "cells": [cell(n, j, col["key"]) for n in names]})
        by_key = {r["key"]: r for r in loose}
        for r in loose:
            if r["key"] in COUNT_BESIDE.values() and any(
                    share in by_key and count == r["key"] for share, count in COUNT_BESIDE.items()):
                continue
            count = by_key.get(COUNT_BESIDE.get(r["key"], ""))
            value = r["value"] if count is None else f"{r['value']} ({count['value']})"
            rows.append({"label": r["label"], "tip": r["tip"], "key_metric": r["key_metric"],
                         "cells": [{"value": value, "cls": r["cls"]}] + [blank] * (len(names) - 1)})
        if rows:
            out.append({"title": title, "rows": rows})
    return {"sets": names, "counts": [n for _, n in sets], "groups": out}


def _failed_criteria(row: dict, cutoffs: dict[str, float]) -> list[str]:
    """Which screening criteria this row's own scores fail, or ["manual"].

    _failed_criteria({"sci": 0.95, "psp": 0.02}, {"sci": 0.8, "psp": 0.1}) -> ["PSP"]

    Derived rather than stored: the record already carries every criterion's per-channel
    score, so asking it again costs nothing and cannot disagree with the screening that
    produced the verdict. A rejected channel that fails nothing was rejected by hand, which
    is only claimed when every criterion actually has a score to judge it by -- an
    unmeasured criterion is not evidence of a manual rejection.
    """
    from nirspipe.qc.metrics import CRITERIA

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
    sci_threshold: "float | None",
    *,
    name_key: str = "name",
    psp_threshold: float | None = None,
    regions: "dict[str, str] | None" = None,
) -> list[dict[str, Any]]:
    """Rows with the numbers already formatted and the cells already classed.

    ::

      format_rows([{"name": "S1_D1 760", "sci_win": 0.412, "corr": None, "is_bad": True, ...}], 0.8)
      -> [{"name": "S1_D1 760", "status": "BAD", "status_cls": "bad",
           "sci_win": "0.412", "sci_win_cls": "bad", "corr": "\u2014", "corr_cls": "", ...}]

    Formatting lives here rather than in each renderer because the raw viewer prints in
    JavaScript and cannot use the Python registry, so it gets its strings from here instead
    of reimplementing the formats and the SCI cutoff. Classes are the subject report's
    ``.qm-table`` names (``bad``/``good`` for status, ``bad`` for a below-threshold SCI,
    ``neg``/``pos`` for the correlation's sign), so one stylesheet rule covers both views.

    ``sci_threshold`` is the run's own ``--sci-threshold``, not the registry's cutoff for
    the SCI mean: this column marks the channels that were actually pruned, so colouring it
    against anything else would show a verdict the run did not reach. ``psp_threshold`` is
    the run's other line, and is here for the same reason: the reason printed beside a
    rejected channel is re-derived from its scores, so a run that moved the PSP line has to
    hand that line over or the reason comes out naming the wrong criterion.
    ``sci_threshold`` None means no line was recorded, and the SCI cell stays uncoloured.
    ``name_key`` is ``"pair"`` for rows that came through :func:`pair_rows`. ``regions`` is
    :func:`~nirspipe.io.optode_space.recorded_regions`, pair -> region, for the Region column.

    Status is ``BAD`` for a channel rejected on its passing-window share, the reason every
    table's note states once, and ``BAD (manual)`` for one rejected by hand. ``reason`` keeps
    the full list for the channel TSV.
    """
    from nirspipe.qc.boilerplate.vocabulary import format_metric
    from nirspipe.qc.metrics import CRITERIA, resolve_cutoffs

    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold)
    passing = next(c.label for c in CRITERIA if c.name == "good_frac")
    out = []
    for row in rows:
        why = _failed_criteria(row, cutoffs) if row["is_bad"] else []
        # the passing-window rule is the usual reason and the notes name it once
        shown = [w for w in why if w != passing]
        formatted: dict[str, Any] = {
            "name":       row.get(name_key),
            "region":     (regions or {}).get(pair_of(row.get(name_key))) or MISSING_VALUE,
            "status":     ("BAD (" + "/".join(shown) + ")" if shown
                           else "BAD" if row["is_bad"] else "OK"),
            "status_cls": "bad" if row["is_bad"] else "good",
            "reason":     "/".join(why),
            "separation": row.get("separation") or "",
            "is_bad":     row["is_bad"],
            "sci_win_value": row.get("sci_win"),
        }
        for column, metric in _COLUMN_METRIC.items():
            value = row.get(column)
            # a pair's two wavelengths, from pair_rows(by_wavelength=True); none measured is one dash
            if isinstance(value, tuple) and all(v is None for v in value):
                value = None
            if isinstance(value, tuple):
                formatted[column] = " / ".join(format_metric(metric, v) for v in value)
            else:
                formatted[column] = format_metric(metric, value)
            formatted[f"{column}_cls"] = ""
        if (sci_threshold is not None and row.get("sci_win") is not None
                and row["sci_win"] < sci_threshold):
            formatted["sci_win_cls"] = "bad"
        # the denoised side beside it where the record has one, coloured by the result
        post = row.get("corr_post")
        if post is not None:
            formatted["corr"] = f"{formatted['corr']} → {format_metric('hbo_hbr_corr_mean', post)}"
            formatted["corr_denoised"] = True
        shown = post if post is not None else row.get("corr")
        if shown is not None:
            formatted["corr_cls"] = "neg" if shown < 0 else "pos"
        out.append(formatted)
    return out


# The per-channel quality table. Not `_channels.tsv`: that name is BIDS' own, for the
# optode pairing and channel properties of one recording, and this is scores and the
# reason a channel was rejected.
CHANNEL_METRICS_SUFFIX = "_desc-channel_qc.tsv"
# raw QC's, raw-prefixed like its record, so neither command's verdict replaces the other's
RAW_CHANNEL_METRICS_SUFFIX = "_desc-rawchannel_qc.tsv"
# best first, the order a reader wanting one verdict per run takes them in
CHANNEL_METRICS_SUFFIXES = (CHANNEL_METRICS_SUFFIX, RAW_CHANNEL_METRICS_SUFFIX)


def pair_status(path: Path) -> "dict[str, dict[str, Any]]":
    """Each source-detector pair a run's channel table lists, in its order, with its separation and whether either wavelength was rejected."""
    out: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for entry in csv.DictReader(fh, delimiter="\t"):
            pair = pair_of(entry.get("name") or "")
            if not pair:
                continue
            row = out.setdefault(pair, {"separation": entry.get("separation") or "n/a",
                                        "is_bad": False})
            row["is_bad"] |= (entry.get("is_bad") or "").strip().lower() == "true"
    return out


# the separations a pair counts as long under: a montage that never split writes none, which
# a channel table prints as n/a and every table blocks with the long channels
_LONG_SEPARATIONS = ("long", "", "n/a")


def long_kept_pairs(pairs: "dict[str, dict[str, Any]]") -> "set[str]":
    """The long pairs :func:`pair_status` rows list as kept; a pair with no separation counts as long."""
    return {pair for pair, row in pairs.items()
            if row["separation"] in _LONG_SEPARATIONS and not row["is_bad"]}


def region_coverage(pairs: "dict[str, dict[str, Any]]",
                    roi_map: "dict[str, list[str]]") -> "list[dict[str, Any]]":
    """Each ``--roi-mapping`` region's long channels in one recording, and which were rejected.

    ::

      region_coverage({"S1_D1": {"separation": "long", "is_bad": False},
                       "S1_D2": {"separation": "long", "is_bad": True},
                       "S1_D9": {"separation": "short", "is_bad": False}},
                      {"L": ["S1_D1", "S1_D2 hbo", "S1_D9", "S7_D7"]})
      -> [{"region": "L", "listed": 2, "kept": 1, "rejected": ["S1_D2"],
           "not_counted": {"S1_D9": "short", "S7_D7": "not in the recording"}}]

    ``pairs`` is :func:`pair_status`-shaped. A listed pair that is not a long channel of this
    recording is named under ``not_counted`` rather than counted, since no region output
    averages it.
    """
    out = []
    for region, listed_pairs in bare_roi_map(roi_map).items():
        long_pairs, rejected, not_counted = [], [], {}
        for pair in listed_pairs:
            status = pairs.get(pair)
            separation = (status or {}).get("separation")
            if status is None:
                not_counted[pair] = "not in the recording"
            elif separation not in _LONG_SEPARATIONS:
                not_counted[pair] = ("between the separation bands" if separation == "unclassified"
                                     else str(separation))
            else:
                long_pairs.append(pair)
                if status.get("is_bad"):
                    rejected.append(pair)
        out.append({"region": str(region), "listed": len(long_pairs),
                    "kept": len(long_pairs) - len(rejected), "rejected": rejected,
                    "not_counted": not_counted})
    return out


def region_table(rows: list[dict], roi_map: "dict[str, list[str]] | None") -> dict[str, Any]:
    """A run's region coverage from its :func:`channel_rows`: ``{"rows", "not_counted"}``, empty without a map.

    ``not_counted`` names every listed pair no region counted, with why, in one line.
    """
    if not roi_map:
        return {"rows": [], "not_counted": ""}
    coverage = region_coverage({r["pair"]: r for r in pair_rows(rows)}, roi_map)
    skipped = {pair: why for row in coverage for pair, why in row["not_counted"].items()}
    return {"rows": coverage,
            "not_counted": ", ".join(f"{pair} ({why})" for pair, why in skipped.items())}


def optode_rejections(pairs: "dict[str, dict[str, Any]]") -> "dict[str, tuple[int, int]]":
    """Each source's and detector's long pairs as ``(rejected, listed)``, from :func:`pair_status` rows.

    ::

      {"S1_D1": {"separation": "long", "is_bad": True},
       "S1_D2": {"separation": "long", "is_bad": False},
       "S1_D9": {"separation": "short", "is_bad": True}}
      -> {"S1": (1, 2), "D1": (1, 1), "D2": (0, 1)}

    Short pairs are left out, so an optode that only short pairs use has no entry. A pair
    with no separation at all (a montage that never split, ``n/a`` in the channel table)
    counts as long, as the tables block it.
    """
    out: dict[str, list[int]] = {}
    for pair, status in pairs.items():
        if status.get("separation") not in _LONG_SEPARATIONS:
            continue
        for optode in pair.split("_")[:2]:
            counts = out.setdefault(optode, [0, 0])
            counts[0] += bool(status.get("is_bad"))
            counts[1] += 1
    return {optode: (bad, listed) for optode, (bad, listed) in out.items()}


def save_channel_csv(rows: list[dict], label: str, out_dir: Path,
                     sci_threshold: "float | None",
                     psp_threshold: float | None = None,
                     suffix: str = CHANNEL_METRICS_SUFFIX,
                     fields: "Sequence[str]" = CSV_FIELDS) -> None:
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
    out_path = Path(out_dir) / (label + suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # an empty reason or separation is not applicable, which BIDS writes as n/a too
    table = pd.DataFrame(rows).reindex(columns=list(fields)).replace("", None)
    write_tsv(table, out_path)
    out_path.with_suffix(".json").write_text(
        json.dumps(column_entries(table.columns, CHANNEL_QC_COLUMNS), indent=2) + "\n",
        encoding="utf-8")
    logger.info("%s | channel metrics CSV saved: %s", label, out_path)
