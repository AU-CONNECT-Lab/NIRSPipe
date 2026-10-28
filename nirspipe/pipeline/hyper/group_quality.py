"""What a group's members are worth: rejected channels, screening, and the dyad's record.

Reads :mod:`~nirspipe.pipeline.hyper.group_io`; nothing reads this. Everything that reduces two
members to one answer applies the same rule: a dyad's channel is usable only while it is
passing in **both** of them.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import mne
import numpy as np
import pandas as pd

from nirspipe.utils import pair_of
from nirspipe.qc.common.channel_table import CHANNEL_METRICS_SUFFIX
from nirspipe.io.derivatives import find_preproc_snirf, group_output_path, read_json
from nirspipe.io.head_template import CORTEX_ATLASES
from nirspipe.io.optode_space import read_placement, recorded_coords_path, write_coords_sidecar
from nirspipe.io.groups import GroupEntry
from nirspipe.pipeline.hyper.group_io import _for_task, _hyper_sidecar, _member_sqm_files
from nirspipe.pipeline.prep_pipeline import _expand_bad_pairs
from nirspipe.utils.lineage import path_from
from nirspipe.utils.logging import get_logger
from nirspipe.io.tables import read_table, write_tsv

logger = get_logger("pipeline.group_quality")


# Which record sections make up each channel set's row in the dyad quality table, in the
# order they are merged. `raw` before `motion` before `preproc` follows the pipeline, and
# no two of them carry the same key, so the order is for reading rather than precedence.
# The whole-file sections are the "all" set: a record splits by separation and the
# unsuffixed section is the one measured over every channel.
_SET_SECTIONS = {
    "all":   ("raw", "motion", "preproc"),
    "long":  ("raw_long", "motion_long", "preproc_long"),
    "short": ("raw_short", "motion_short", "preproc_short"),
}


_BANDS_FIELDS = ("short_max_dist", "long_min_dist", "long_max_dist")


def write_group_bads(
    output_dir: Path,
    group: list[GroupEntry],
    sqm_by_subject: dict[str, dict],
    bads_scope: str,
    session: "str | None" = None,
) -> Path:
    """Write the rejected channels the inter-brain metrics actually excluded, one row each.

    The channel set is a decision that changes every coherence value, so it is written here
    rather than only logged. Columns: group_id, task, subject_id, channel, bads_scope,
    rejected_in. ``rejected_in`` lists the runs whose quality metrics rejected the channel,
    which under ``--bads-scope subject`` is how a condition that was clean on its own comes
    to lose a channel.
    """
    gid, task = group[0].group_id, group[0].task
    rows: list[dict] = []
    for entry in group:
        sqm = sqm_by_subject.get(entry.subject_id, {})
        sources = sqm.get("bad_channel_sources") or {}
        for channel in sqm.get("bad_channels") or []:
            rows.append({
                "group_id": gid, "task": task, "subject_id": entry.subject_id,
                "channel": channel, "bads_scope": bads_scope,
                "rejected_in": ";".join(sources.get(channel) or [task]),
            })

    out_path = group_output_path(output_dir, gid, {"task": task, "desc": "bad"},
                                 "qc", ".tsv", session=session)
    columns = ["group_id", "task", "subject_id", "channel", "bads_scope", "rejected_in"]
    write_tsv(pd.DataFrame(rows, columns=columns), out_path)
    _hyper_sidecar(out_path, "hyper_bads", [], bads_scope=bads_scope)
    logger.info("excluded channels saved: %s (%d rows)", out_path, len(rows))
    return out_path


def write_group_channel_coords(
    derivatives_dir: Path,
    output_dir: Path,
    group: list[GroupEntry],
    desc: str,
    session: "str | None" = None,
) -> Path:
    """Every member's channel positions on their own template in one table, with subject_id and space columns.

    Kept per member, since two caps need not sit alike and two members need not share a
    head (a parent and an infant); a figure wanting one position per channel averages
    members on one template. Each member's rows are prep's record for the run this group
    reads, and a member whose tree predates that record is refused. The name carries the
    space when every member shares it, and none otherwise.

    ::

      sub-A rows S1_D1 hbo ... + sub-B rows S1_D1 hbo ...
        -> group-G1_task-rest_space-fsaverage_channels.tsv: subject_id space name x y z aparc ...
    """
    frames, sources, placements = [], [], {}
    for entry in group:
        snirf = find_preproc_snirf(derivatives_dir, entry.subject_id, entry.task, desc=desc,
                                   session=entry.session, run=entry.run)
        placement = read_placement(snirf)
        placements[entry.subject_id] = placement.to_json()
        record = recorded_coords_path(snirf)
        frames.append(read_table(record).assign(subject_id=entry.subject_id,
                                                space=placement.space))
        sources.append(record)
    spaces = {p["space"] for p in placements.values()}
    shared = spaces.pop() if len(spaces) == 1 else None
    out_path = group_output_path(output_dir, group[0].group_id,
                                 {"task": group[0].task, "space": shared}, "channels", ".tsv",
                                 session=session)
    table = pd.concat(frames)
    regions = [c for c in CORTEX_ATLASES.values() if c in table.columns]
    write_tsv(table[["subject_id", "space", "name", "x", "y", "z", *regions]], out_path)
    write_coords_sidecar(out_path, sources, placements, shared, regions=regions)
    return out_path


def apply_group_bads(
    raws: dict[str, mne.io.Raw],
    sqm_by_subject: dict[str, dict],
) -> None:
    """Mark each subject's rejected channels bad on their Raw, in place.

    Not because the file lacks the marks: every derivative's sidecar carries
    ``bad_channels`` and ``read_snirf`` restores them. It is ``--bads-scope`` that needs
    this. A per-file sidecar can only name what that run rejected, while ``subject`` scope
    is the union over every run the subject has, which no single file knows. Marking the
    union here is what makes WTC, ISC and the report rest on the one channel set the scope
    chose, since ``long_channel_picks`` drops bads.

    The record names channels by wavelength ("S6_D5 760"), a haemoglobin Raw by chromophore
    ("S6_D5 hbo"). Both reduce to the S-D label, which is the key every inter-brain metric
    already matches on, so a pair rejected at either wavelength is marked at both
    chromophores.

    ::

      {"sub-A": raw}  + {"sub-A": {"bad_channels": ["S6_D5 760", "S6_D5 850"]}}
      ->  raw.info["bads"] == ["S6_D5 hbo", "S6_D5 hbr"]
    """
    for subject_id, raw in raws.items():
        labels = {pair_of(ch)
                  for ch in (sqm_by_subject.get(subject_id, {}).get("bad_channels") or [])}
        if not labels:
            continue
        marked = _expand_bad_pairs(raw, list(labels))
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(marked))
        logger.info("%s: %d channel(s) marked bad from the quality record: %s",
                    subject_id, len(marked), ", ".join(sorted(labels)))


def member_raw_record(output_dir: Path, entry: GroupEntry, raw: mne.io.Raw) -> "tuple[Path, dict]":
    """The member's raw QC record for the recording ``raw`` holds, as ``(path, record)``.

    Matched by the recording its Sources name rather than by label, so a member with several
    runs of a task gets the record of the run the pairs table chose. Refused when there is
    none, or when it predates records that carry the respiration band among their parameters.
    """
    from nirspipe.exceptions import StageError
    from nirspipe.qc.subject.record_io import read_record
    from nirspipe.qc.subject.sqm_record import RECORD_SUFFIXES

    source = Path(path_from(raw) or "").name
    for path in _member_sqm_files(output_dir, entry,
                                  f"{entry.subject_id}*{RECORD_SUFFIXES['sqmraw']}"):
        if any(Path(str(uri)).name == source for uri in read_json(path).get("Sources") or []):
            record = read_record(path)
            if "resp_l_freq" not in (record.get("parameters") or {}):
                raise StageError(f"{path.name} was written before raw QC records carried their "
                                 f"respiration band; run `nirspipe-qc ... participant` on "
                                 f"{entry.subject_id} again")
            return path, record
    raise StageError(f"{entry.subject_id}: no raw QC record for {source} under {output_dir}; "
                     f"run `nirspipe-qc ... participant` on it first")


def load_group_sqm_raw(
    output_dir: Path,
    group: list[GroupEntry],
    raws: dict[str, mne.io.Raw],
    records: "dict[str, tuple[Path, dict]]",
    session: "str | None" = None,
) -> dict[str, dict]:
    """Each member's raw QC record in the dyad page's shape, and the dyad's two quality tables.

    The verdict, the screening lines and the cardiac band are each member's own, as the
    participant level recorded them, so nothing is screened here and a parent-child dyad
    keeps two bands. ``records`` is :func:`member_raw_record` per member.

    Writes two TSVs under group-{gid}/nirs/, where a subject's own tables sit:
      group-{gid}_task-{task}_desc-subject_qc.tsv: one row per subject (scalars)
      group-{gid}_task-{task}_desc-channel_qc.tsv: one row per subject × channel

    Returns {subject_id: sqm_dict} for the HTML report, with ``screen_cutoffs``,
    ``screen_band`` and ``parameters`` beside the record's own numbers.
    """
    from nirspipe.qc.subject.sqm_record import raw_verdict_view

    gid  = group[0].group_id
    task = group[0].task

    sqm_data: dict[str, dict] = {}
    scalar_rows: list[dict] = []
    channel_rows: list[dict] = []
    members: dict[str, dict] = {}

    for entry in group:
        sid = entry.subject_id
        _path, record = records[sid]
        params = record["parameters"]
        cutoffs = {"sci": params["sci_threshold"], "psp": params.get("psp_threshold"),
                   "good_frac": params.get("min_good_frac")}

        sqm = raw_verdict_view(record)
        imu_sec = record.get("imu") or {}
        sqm.update(imu_sec)
        per_channel = record.get("per_channel") or {}
        per_all = per_channel.get("raw") or {}
        sci_cw = per_all.get("sci_per_channel") or {}
        sci_win = per_all.get("sci_win_per_channel") or {}
        bad_channels = list(record.get("bad_channels") or [])
        tsv_bads = _expand_bad_pairs(raws[sid], params.get("channels_tsv_bads") or [])

        # which channel set the row above describes, so a table read on its own says so
        sqm["channel_set"] = "long" if record.get("raw_long") else "all"
        sqm["sci_per_channel"] = {k: float(v) for ch, v in sci_cw.items()
                                  for k in (f"{pair_of(ch)} hbo", pair_of(ch))}
        sqm["sci_win_per_channel"] = _pairwise(sci_win)
        sqm["bad_channels"]    = bad_channels
        sqm["channels_tsv_pairs"] = sorted({pair_of(c) for c in tsv_bads})
        sqm["screen_windows"]  = _screen_windows(record, cutoffs)
        sqm["screen_cutoffs"]  = cutoffs
        sqm["screen_band"]     = (params["cardiac_l_freq"], params["cardiac_h_freq"])
        sqm["resp_band"]       = (params["resp_l_freq"], params["resp_h_freq"])
        sqm["screening_gap"]   = record.get("screening_gap")
        sqm["parameters"]      = params
        sqm["per_channel"]     = per_channel
        sqm["per_channel_all"]  = per_all
        sqm["per_channel_long"] = per_channel.get("raw_long") or per_all
        # the three sections side by side, the shape `load_group_sqm` gives the post report
        sqm["by_set"] = {name: {k: v for section in sections
                                for k, v in (record.get(section) or {}).items()}
                         for name, sections in _SET_SECTIONS.items()}
        sqm_data[sid] = sqm
        members[sid] = {k: params.get(k) for k in ("sci_threshold", "psp_threshold",
                                                   "min_good_frac", "cardiac_l_freq",
                                                   "cardiac_h_freq", "resp_l_freq",
                                                   "resp_h_freq")}

        # every scalar the record holds, not a whitelist that would leave SCI alone
        scalar_rows.append({
            "group_id":       gid,
            "subject_id":     sid,
            "task":           task,
            "channel_set":    sqm["channel_set"],
            "n_bad_channels": len(bad_channels),
            **{k: v for k, v in sqm.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)},
            # the unit each IMU number was read in, which differs between devices
            **{k: v for k, v in imu_sec.items() if isinstance(v, str)},
        })
        for ch, sci_val in sci_cw.items():
            channel_rows.append({
                "group_id":   gid,
                "subject_id": sid,
                "task":       task,
                "channel":    ch,
                # the subject table's names, windowed first
                "sci_win":    sci_win.get(ch),
                "sci_whole":  sci_val,
                "is_bad":     ch in bad_channels,
            })

    # each member's recording and the record its numbers were read from
    sources = [p for e in group for p in (path_from(raws[e.subject_id]),
                                          str(records[e.subject_id][0])) if p]
    # the members whose channels.tsv rejected something, named in both sidecars
    listed = {e.subject_id: names for e in group
              if (names := records[e.subject_id][1]["parameters"].get("channels_tsv_bads"))}
    tsv_params = {"members": members, **({"channels_tsv_bads": listed} if listed else {})}

    scalar_path = group_output_path(output_dir, gid, {"task": task, "desc": "subject"},
                                    "qc", ".tsv", session=session)
    write_tsv(pd.DataFrame(scalar_rows), scalar_path)
    _hyper_sidecar(scalar_path, "group_sqm_raw", sources, **tsv_params)

    # not `_channels.tsv`: that name is BIDS's own channel description, and this holds
    # one row per subject and channel of cross-subject quality
    channel_path = group_output_path(output_dir, gid, {"task": task, "desc": "channel"},
                                     "qc", ".tsv", session=session)
    write_tsv(pd.DataFrame(channel_rows), channel_path)
    _hyper_sidecar(channel_path, "group_sqm_raw_channels", sources, **tsv_params)

    return sqm_data


def load_group_sqm(
    output_dir: Path, group: list[GroupEntry], bads_scope: str = "run",
    scope_tasks: "list[str] | None" = None,
) -> dict[str, dict]:
    """Load per-subject SQM scalars and channel metrics from derivatives.

    A subject has one record and one channel-metrics CSV per run, and every entry names the
    task it belongs to, so the run's own files are the ones read. Reading all of them and
    letting the last win would analyse every task with the last task's rejected channels.

    ``bads_scope`` decides what counts as a bad channel:

    - ``"run"``: this task's own rejections, matching the rest of the metrics returned here.
    - ``"subject"``: the union over the subject's runs, so a channel rejected in any
      condition is rejected in all of them. Conditions then rest on the same channel set, at
      the cost of losing a channel everywhere because one segment was bad.

    ``scope_tasks`` bounds that union to the tasks the analysis covers, normally the ones the
    pairs table names. Without it the union is every ``desc-sci`` sidecar in the folder, so a
    subject who also sat a resting run, or a second experiment, loses channels here for a
    recording nobody asked about.

    A recording preprocessed whole is screened once, so the subject has one run per task and
    the union over it is itself: ``subject`` and ``run`` then name the same set, and a line in
    the log says so. They differ when the conditions were cropped to separate tasks before
    prep.

    Returns {subject_id: sqm_dict}. Alongside the flattened scalars each dict carries
    ``windowed`` (the record's channel-by-window matrices), ``channel_order`` (their row
    order), ``per_channel`` (the whole per-channel section, which the dyad's channel table
    prints), ``screen_cutoffs`` (the lines that run screened by) and ``by_condition`` (the
    record's per-condition entries), which is what a per-condition view of the dyad pages
    is built from.
    """
    if bads_scope not in ("run", "subject"):
        raise ValueError(f"bads_scope must be 'run' or 'subject', got {bads_scope!r}")

    from nirspipe.qc.subject.record_io import read_record
    from nirspipe.qc.subject.sqm_record import RECORD_SUFFIX, raw_verdict_view

    result: dict[str, dict] = {}
    for entry in group:
        sqm: dict = {}
        # the long-channel view is the one a quality judgement wants, with raw standing in
        # when the montage has no short channels to exclude
        records = _member_sqm_files(output_dir, entry, f"{entry.subject_id}*{RECORD_SUFFIX}")
        for record_path in _for_task(records, entry.task):
            try:
                record = read_record(record_path)
            except (OSError, json.JSONDecodeError):
                continue
            # Whole-file section underneath its long-channel split in both families, rather
            # than replaced by it: the split sections carry only what a channel set can be
            # measured on, so the run-level numbers (the montage counts, the recording's
            # duration) survive underneath while the long values win where both exist. The
            # raw family goes through raw_verdict_view, which is the same rule the dyad raw
            # report and the subject report read their scalars by.
            sqm.update(raw_verdict_view(record))
            sqm.update(record.get("motion") or {})
            sqm.update(record.get("preproc") or {})
            sqm.update(record.get("preproc_long") or {})
            # unsuffixed, as the subject report reads it: its keys collide with nothing
            sqm.update(record.get("imu") or {})
            # the channel-by-window matrices a per-condition view selects columns from,
            # nested under one key so they cannot collide with a scalar name
            sqm["windowed"] = record.get("windowed") or {}
            sqm["channel_order"] = list(sqm["windowed"].get("sci_channels") or [])
            # the record's whole per-channel section: the dyad's channel table prints the
            # same columns the subject report does, PSP, SNR, CV and the spike share among them
            per_ch = record.get("per_channel") or {}
            # every section, not only the all-channel one: a short channel's scores live in
            # `raw_short` and the dyad's channel table prints short rows too
            sqm["per_channel"] = per_ch
            sqm["per_channel_all"] = per_ch.get("raw") or {}
            # the windowed SCI the dyad pages read, keyed the way their lookups ask
            sqm["sci_win_per_channel"] = _pairwise(sqm["per_channel_all"]
                                                   .get("sci_win_per_channel") or {})
            # an all-long montage writes no `raw_long`, so the whole-file section stands in
            sqm["per_channel_long"] = per_ch.get("raw_long") or sqm["per_channel_all"]
            # the screening grid rebuilt from the matrices the record stored, the same shape
            # `load_group_sqm_raw` gives, so a dyad panel does not care which command
            # produced the members' numbers
            sqm["screen_windows"] = _screen_windows(record, sqm.get("screen_cutoffs"))
            # the per-condition numbers the record already holds, so the dyad pages read
            # them rather than cutting the matrices above a second time
            sqm["by_condition"] = record.get("by_condition") or {}
            # the same scalars split by channel set, for the dyad quality table's three columns
            sqm["by_set"] = {
                set_name: {k: v
                           for section in sections
                           for k, v in (record.get(section) or {}).items()}
                for set_name, sections in _SET_SECTIONS.items()
            }

        # Rejection is read from the desc-sci sidecars, which prep writes on every run and
        # which name every channel the run rejected whatever came after. The channel-metrics
        # CSV holds the same set, but the report writes that one, so a tree produced with
        # --no-report has the sidecars and no CSV.
        sidecars = _member_sqm_files(output_dir, entry,
                                     f"{entry.subject_id}*_desc-sci_nirs.json")
        csvs     = _member_sqm_files(output_dir, entry,
                                     f"{entry.subject_id}*{CHANNEL_METRICS_SUFFIX}")
        marks, read_bads = ((sidecars, _bad_from_sidecar) if sidecars
                            else (csvs, _bad_from_csv))
        kind = "desc-sci sidecar" if sidecars else "channel metrics CSV"

        run_marks = _for_task(marks, entry.task)
        if not marks:
            logger.warning(
                "%s has neither a desc-sci sidecar nor channel metrics, so no channel is "
                "rejected for task-%s and every bad channel enters the inter-brain metrics. "
                "Rerun nirspipe on this subject.", entry.subject_id, entry.task)
        elif not run_marks:
            logger.warning(
                "%s has a %s but none for task-%s, so no channel is rejected for it. "
                "Rerun nirspipe on this task.", entry.subject_id, kind, entry.task)

        for path in run_marks:
            if sidecars:
                scores = _sci_from_sidecar(path)
                if scores:
                    sqm["sci_per_channel"] = scores
            else:
                try:
                    ch_df = pd.read_csv(path, sep="\t")
                    if {"name", "sci_whole"}.issubset(ch_df.columns):
                        sqm["sci_per_channel"] = dict(
                            zip(ch_df["name"].astype(str),
                                pd.to_numeric(ch_df["sci_whole"], errors="coerce"))
                        )
                except Exception:
                    pass
            sqm["bad_channels"] = read_bads(path)
            if sidecars:
                sqm["screen_cutoffs"] = _screen_cutoffs(path)
            logger.info("%s task-%s: %d rejected channel(s) from %s (%s)",
                        entry.subject_id, entry.task, len(sqm["bad_channels"]), kind, path.name)

        # which run each rejection came from, so the union is reviewable
        sources: dict[str, list[str]] = {
            ch: [entry.task] for ch in (sqm.get("bad_channels") or [])
        }
        if bads_scope == "subject":
            in_scope = ([p for p in marks
                         if (m := re.search(r"_task-([A-Za-z0-9]+)", p.name))
                         and m.group(1) in scope_tasks]
                        if scope_tasks else marks)
            for path in in_scope:
                from_task = m.group(1) if (m := re.search(r"_task-([A-Za-z0-9]+)", path.name)) else entry.task
                for ch in read_bads(path):
                    if from_task not in sources.setdefault(ch, []):
                        sources[ch].append(from_task)
            sqm["bad_channels"] = sorted(sources)
            if len(in_scope) <= 1:
                # the union over one run is that run, so the flag did nothing
                logger.warning(
                    "%s: --bads-scope subject found %d run(s) in scope, so the union is "
                    "just that run's own rejections and the flag changed nothing. That is "
                    "what a whole-recording analysis looks like: prep screened the "
                    "recording once, so every condition already rests on one channel set. "
                    "The flag matters only where the conditions were cropped to separate "
                    "tasks before prep.", entry.subject_id, len(in_scope))
            else:
                logger.info("%s: --bads-scope subject unions %d channel(s) over %d run(s) "
                            "of %s", entry.subject_id, len(sources), len(in_scope), kind)
        sqm["bad_channel_sources"] = {ch: sorted(t) for ch, t in sources.items()}

        result[entry.subject_id] = sqm

    return result


def _screen_windows(record: dict, cutoffs: "dict | None") -> dict:
    """``{mask, centers, channel_order}`` off a stored record, empty when it has no matrices.

    The cutoffs are the run's own, since a record screened at a different SCI line has to be
    re-masked at that line; a run that recorded no SCI line is not re-masked at a guessed one,
    and one that recorded no PSP line is masked on SCI alone.
    """
    from nirspipe.qc.metrics.windowed import passing_mask_from_matrices, window_centers

    windowed = record.get("windowed") or {}
    lines = cutoffs or {}
    if lines.get("sci") is None:
        return {}
    sci = windowed.get("sci_matrix")
    psp = windowed.get("psp_matrix")
    psp_line = lines.get("psp")
    mask = passing_mask_from_matrices(sci, psp, float(lines["sci"]),
                                      None if psp_line is None else float(psp_line))
    times = windowed.get("sci_times")
    if mask is None or not times:
        return {}
    out = {"mask": mask,
           "centers": window_centers(np.asarray(times, dtype=float)),
           "sci": np.asarray(sci, dtype=float),
           **({"psp": np.asarray(psp, dtype=float)} if psp else {}),
           **({"cv": np.asarray(windowed["cv_matrix"], dtype=float)}
              if windowed.get("cv_matrix") else {}),
           "channel_order": list(windowed.get("sci_channels") or [])}
    # GVTD is an RMS across channels, so it has no matrix and rides along as one series.
    # Only a stored record carries it; the dyad's own pass does not measure motion, and a
    # panel that draws it has to cope with the row being absent rather than assume it.
    gvtd = windowed.get("gvtd_filt_per_window")
    if gvtd:
        out["gvtd"] = np.asarray(gvtd, dtype=float)
    return out


def _bad_from_sidecar(json_path: Path) -> list[str]:
    """The rejected channels prep recorded for a run, from its desc-sci sidecar.

    ``{"parameters": ..., "bad_channels": ["S6_D5 760", "S6_D5 850"]}`` -> that list.

    This is where rejection is decided: prep writes the sidecar after unioning the SCI
    detections with any manual --bad-channels, and writes it whether or not a report was
    asked for. The channel-metrics CSV carries the same set, but the report writes that one.
    """
    return list(read_json(json_path).get("bad_channels") or [])


def _sci_from_sidecar(json_path: Path) -> dict[str, float]:
    """The per-channel SCI prep recorded for a run, from the same sidecar."""
    scores = read_json(json_path).get("sci_scores") or {}
    return {str(k): float(v) for k, v in scores.items()}


def _screen_cutoffs(json_path: Path) -> dict:
    """The screening lines a run used, off its ``desc-sci`` sidecar.

    ``{"sci": 0.8, "psp": 0.1, "good_frac": 0.75}``, or an empty dict when the sidecar
    predates them. A per-condition view rebuilds the passing-window share from the stored
    matrices, and it has to apply the lines that subject was actually screened by, not the
    registry defaults or ``nirspipe-hyper``'s own ``--sci-threshold``, which only colours the
    dyad page.
    """
    params = read_json(json_path).get("parameters") or {}
    named = {"sci": params.get("sci_threshold"), "psp": params.get("psp_threshold"),
             "good_frac": params.get("min_good_frac")}
    return {k: float(v) for k, v in named.items() if v is not None}


def _bad_from_csv(csv_path: Path) -> list[str]:
    try:
        ch_df = pd.read_csv(csv_path, sep="	")
    except Exception:
        return []
    if "is_bad" not in ch_df.columns:
        return []
    return ch_df.loc[
        ch_df["is_bad"].astype(str).str.lower().isin({"true", "1"}), "name"
    ].tolist()


def _pairwise(per_wavelength: dict) -> dict[str, float]:
    """A per-channel score re-keyed the way every dyad lookup asks for it.

    ::

      {"S1_D1 760": 0.86, "S1_D1 850": 0.86} -> {"S1_D1": 0.86, "S1_D1 hbo": 0.86}

    The whole-run SCI is built with both spellings in ``load_group_sqm_raw``, because the pages ask
    for ``"<pair> hbo"`` and fall back to the bare pair. The windowed estimate comes out of
    the metric registry keyed by the recording's own channel names instead, so handing it
    over unchanged makes every lookup miss and every channel read as unmeasured, with
    nothing to say so. SCI is a property of the pair, both wavelength rows carrying one
    number, so averaging them is a no-op that also survives a montage where it is not.
    """
    by_pair: dict[str, list[float]] = {}
    for ch, value in per_wavelength.items():
        by_pair.setdefault(pair_of(ch), []).append(float(value))
    out: dict[str, float] = {}
    for pair, values in by_pair.items():
        out[pair] = out[f"{pair} hbo"] = sum(values) / len(values)
    return out


def resolve_group_bands(
    group: list[GroupEntry], group_sqm: dict[str, dict], override: dict | None = None,
) -> "tuple[float, float, float | None]":
    """The separation bands a dyad's inter-brain metrics run on, read off the members' records.

    ``nirspipe-hyper`` works on derivatives that prep has already split into long and short
    channels and stamped with the bands it split them by, so the bands are read back rather
    than taken again from the command line.
    ``nirspipe-qc``'s group level is not a caller: it takes each member's bands from that
    member's raw QC record.

    Two members are two prep runs, so they can disagree. **That is refused, not
    reconciled**: the bands also chose what short-channel regression removed from each
    member upstream, so a band derived from both would describe neither.

    ``override`` is the three flags as ``{"short_max_dist": ..., "long_min_dist": ...,
    "long_max_dist": ...}`` in metres, None for one left off. A value given wins and is
    logged as forced; one left off falls back to the *records*, not to the package default,
    so capping the long band does not silently re-assert the other two. An upper bound the
    records carry therefore cannot be switched off from the command line, which needs a
    re-run of prep.

    A member whose record carries no stamp reads back as the defaults. That is a guess
    rather than a fact, so it is warned about and does not count towards agreement.
    """
    from nirspipe.qc.metrics._helpers import (
        bands_from_record,
        bands_phrase,
        record_has_bands,
        separation_bands,
        validate_bands,
    )

    stamped: dict[str, tuple] = {}
    unstamped: list[str] = []
    for entry in group:
        label = f"{entry.subject_id} task-{entry.task}"
        sqm = group_sqm.get(entry.subject_id) or {}
        if record_has_bands(sqm):
            stamped[label] = bands_from_record(sqm)
        else:
            unstamped.append(label)

    if len(set(stamped.values())) > 1:
        spread = "\n".join(f"  {label}: {bands_phrase(bands)}"
                           for label, bands in sorted(stamped.items()))
        raise ValueError(
            "the members were prepped with different separation bands, so this dyad has no "
            f"one definition of a long channel:\n{spread}\n"
            "Re-run nirspipe on the odd one out so the two match, or pass "
            "--short-max-dist / --long-min-dist / --long-max-dist to force one set for this "
            "run. They are not reconciled for you: the bands also chose what short-channel "
            "regression removed from each member, so a band taken from both would describe "
            "neither."
        )

    # a stamped member's bands when there is one, else today's defaults; either way a guess
    # for the unstamped members, which is what the warning below is about
    from_records = next(iter(set(stamped.values())), separation_bands())

    if unstamped:
        logger.warning(
            "no separation bands stamped for %s, so %s is assumed for %s: the record has "
            "no raw section to read them from. "
            "Re-run nirspipe on it, or pass --short-max-dist / --long-min-dist / "
            "--long-max-dist to say what it was.",
            ", ".join(unstamped), bands_phrase(from_records),
            "it" if len(unstamped) == 1 else "them")

    forced = {k: v for k, v in (override or {}).items() if v is not None}
    if not forced:
        logger.info("separation bands from the members' records: %s",
                    bands_phrase(from_records))
        return validate_bands(from_records)

    merged = tuple(forced.get(field, current)
                   for field, current in zip(_BANDS_FIELDS, from_records))
    if stamped and merged != from_records:
        logger.warning(
            "separation bands forced to %s, overriding the %s the members' records stamp. "
            "The split the inter-brain metrics use now differs from the one prep applied, "
            "including which channels it regressed out as short.",
            bands_phrase(merged), bands_phrase(from_records))
    else:
        logger.info("separation bands forced to %s", bands_phrase(merged))
    return validate_bands(merged)
