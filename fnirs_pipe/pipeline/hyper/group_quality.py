"""What a group's members are worth: rejected channels, screening, and the dyad's record.

Reads :mod:`~fnirs_pipe.pipeline.hyper.group_io`; nothing reads this. Everything that reduces two
members to one answer applies the same rule: a dyad's channel is usable only while it is
coupled in **both** of them.
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import mne
import numpy as np
import pandas as pd

from fnirs_pipe.io.derivatives import group_data_dir
from fnirs_pipe.pipeline.hyper.group_io import (
    GroupEntry, _for_task, _hyper_sidecar, _member_sqm_files,
)
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

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
) -> Path:
    """Write the rejected channels the inter-brain metrics actually excluded, one row each.

    The channel set is a decision that changes every coherence value, and until now it was
    only visible in the log. Columns: group_id, task, subject_id, channel, bads_scope,
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

    out_path = group_data_dir(output_dir, gid) / f"group-{gid}_task-{task}_hyper-bads.tsv"
    columns = ["group_id", "task", "subject_id", "channel", "bads_scope", "rejected_in"]
    pd.DataFrame(rows, columns=columns).to_csv(out_path, sep="	", index=False)
    _hyper_sidecar(out_path, "hyper_bads", [], bads_scope=bads_scope)
    logger.info("excluded channels saved: %s (%d rows)", out_path, len(rows))
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
        labels = {ch.rsplit(" ", 1)[0]
                  for ch in (sqm_by_subject.get(subject_id, {}).get("bad_channels") or [])}
        if not labels:
            continue
        marked = [ch for ch in raw.ch_names if ch.rsplit(" ", 1)[0] in labels]
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(marked))
        logger.info("%s: %d channel(s) marked bad from the quality record: %s",
                    subject_id, len(marked), ", ".join(sorted(labels)))


def compute_group_sqm_raw(
    group: list[GroupEntry],
    raws: dict[str, mne.io.Raw],
    sci_threshold: float,
    output_dir: Path,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    psp_threshold: float | None = None,
    sep_bands=None,
    min_good_frac: float | None = None,
    screen_scope: str = "run",
) -> dict[str, dict]:
    """Compute raw-level SQM (SCI, bad channels) for each group member.

    Writes two TSVs under group-{gid}/nirs/, where a subject's own tables sit:
      group-{gid}_task-{task}_hyper-raw_sqm.tsv      — one row per subject (scalars)
      group-{gid}_task-{task}_hyper-raw_channels.tsv  — one row per subject × channel

    Returns {subject_id: sqm_dict} for use in the HTML report.

    The dict is the long-channel verdict, assembled by
    :func:`~fnirs_pipe.qc.subject.sqm_record.raw_verdict_view` from the same three sections the
    per-subject record holds. It used to be one all-channel pass, which put a subject's
    SCI, CV, SNR and GVTD in this table on a different channel set than the same subject's
    numbers in the individual reports and in `fnirs-hyper run`, so the two could not be
    read against each other.
    """
    from fnirs_pipe.qc.metrics import resolve_cutoffs, screen_channels, screening_scores
    from fnirs_pipe.qc.common.screen_scope import resolve_screen_scope
    from fnirs_pipe.qc.subject.sqm_record import raw_sections, raw_verdict_view

    cutoffs = resolve_cutoffs(sci=sci_threshold, psp=psp_threshold,
                              good_frac=min_good_frac)

    gid  = group[0].group_id
    task = group[0].task
    data_dir = group_data_dir(output_dir, gid)

    sqm_data: dict[str, dict] = {}
    scalar_rows: list[dict] = []
    channel_rows: list[dict] = []

    for entry in group:
        raw = raws[entry.subject_id]

        # raw_od stays None when the conversion itself failed, which is the one case where
        # nothing can be screened: everything below reads optical density
        raw_od: "mne.io.Raw | None" = None
        sci_cw = {ch: float("nan") for ch in raw.ch_names}
        try:
            raw_od  = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
            sci_arr = mne.preprocessing.nirs.scalp_coupling_index(
                raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
            sci_cw  = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
        except Exception:
            logger.warning("%s: SCI could not be measured", entry.subject_id, exc_info=True)

        sci_scores: dict[str, float] = {}
        for ch, val in sci_cw.items():
            pair = ch.rsplit(" ", 1)[0]
            sci_scores[f"{pair} hbo"] = val
            sci_scores[pair] = val

        bad_channels: list[str] = []
        screen: dict = {}
        screen_windows: dict = {}
        if raw_od is not None:
            # the coupled-window grid, kept rather than recounted: the dyad panels that draw
            # quality over time have to shade the windows the verdict was taken on, and a
            # second pass could disagree with it. `have` then stops the criterion table from
            # counting the same windows again.
            from fnirs_pipe.qc.metrics.windowed import coupled_windows
            counted = coupled_windows(
                raw_od, cardiac_l_freq, cardiac_h_freq, cutoffs["sci"], cutoffs["psp"],
                scope=resolve_screen_scope(raw, screen_scope))
            if counted["mask"] is not None:
                screen_windows = {k: counted[k] for k in
                                  ("mask", "centers", "sci", "psp", "channel_order")}
                # CV on the same grid, off raw intensity, which is what it is defined on:
                # after the optical-density conversion sigma/mu is no longer relative
                # brightness. SCI and PSP are blind to the shifts and dropouts it catches.
                try:
                    from fnirs_pipe.qc.metrics.windowed import compute_windowed_cv
                    cv_m, cv_t = compute_windowed_cv(raw)
                    if cv_m is not None and len(cv_t) == len(counted["centers"]):
                        screen_windows["cv"] = cv_m
                except Exception:
                    logger.warning("%s: windowed CV could not be measured",
                                   entry.subject_id, exc_info=True)
                # motion on the same window grid, which the screening pass does not measure:
                # SCI and PSP are blind to movement by construction, so without this row the
                # dyad panel can say a pair decoupled but never that the member moved
                try:
                    from fnirs_pipe.qc.metrics.gvtd import compute_windowed_gvtd
                    means, _p95, gvtd_t = compute_windowed_gvtd(raw_od)
                    if len(gvtd_t) == len(counted["centers"]):
                        screen_windows["gvtd"] = means
                    else:
                        logger.warning("%s: windowed GVTD landed on %d windows against the "
                                       "screening's %d; dropping the motion row rather than "
                                       "drawing it on the wrong axis", entry.subject_id,
                                       len(gvtd_t), len(counted["centers"]))
                except Exception:
                    logger.warning("%s: windowed GVTD could not be measured",
                                   entry.subject_id, exc_info=True)
            screen = screening_scores(raw_od, cardiac_l_freq, cardiac_h_freq,
                                      have={"sci": sci_cw,
                                            "good_frac": counted["fractions"]},
                                      cutoffs=cutoffs,
                                      scope=resolve_screen_scope(raw, screen_scope))
            bad_channels, _ = screen_channels(screen, cutoffs)

        try:
            sections, _per_channel = raw_sections(
                raw, sci_cw, bad_channels, cardiac_l_freq, cardiac_h_freq, sep_bands,
                screen.get("good_frac"))
            sqm = raw_verdict_view(sections)
        except Exception:
            logger.warning("%s: quality metrics failed", entry.subject_id, exc_info=True)
            sections, sqm, _per_channel = {}, {}, {}

        # which channel set the row above describes, so a table read on its own says so
        sqm["channel_set"] = "long" if sections.get("raw_long") else "all"
        sqm["sci_per_channel"] = sci_scores
        sqm["sci_win_per_channel"] = _pairwise(
            (sections.get("raw") or {}).get("sci_win_per_channel") or {})
        sqm["bad_channels"]    = bad_channels
        sqm["screen_windows"]  = screen_windows
        sqm["screen_cutoffs"]  = dict(cutoffs)
        sqm["per_channel"]     = _per_channel or {}
        sqm["per_channel_all"]  = (_per_channel or {}).get("raw") or {}
        sqm["per_channel_long"] = ((_per_channel or {}).get("raw_long")
                                   or sqm["per_channel_all"])
        sqm_data[entry.subject_id] = sqm

        # Every scalar the record holds, not a whitelist: a whitelist leaves SCI as the only
        # quality metric a hyperscanning study sees, and SCI is amplitude-invariant, so a run
        # with a collapsed cardiac pulse would read as fine
        scalar_rows.append({
            "group_id":       gid,
            "subject_id":     entry.subject_id,
            "task":           task,
            "channel_set":    sqm["channel_set"],
            "n_bad_channels": len(bad_channels),
            **{k: v for k, v in sqm.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)},
        })

        for ch, sci_val in sci_cw.items():
            channel_rows.append({
                "group_id":   gid,
                "subject_id": entry.subject_id,
                "task":       task,
                "channel":    ch,
                "sci":        sci_val,
                "is_bad":     ch in bad_channels,
            })

    stem = f"group-{gid}_task-{task}_hyper-raw"
    sources = [p for p in (path_from(raws[e.subject_id]) for e in group) if p]

    scalar_path = data_dir / f"{stem}_sqm.tsv"
    pd.DataFrame(scalar_rows).to_csv(scalar_path, sep="\t", index=False)
    _hyper_sidecar(scalar_path, "group_sqm_raw", sources,
                   sci_threshold=sci_threshold,
                   psp_threshold=cutoffs["psp"],
                   cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq)

    channel_path = data_dir / f"{stem}_channels.tsv"
    pd.DataFrame(channel_rows).to_csv(channel_path, sep="\t", index=False)
    _hyper_sidecar(channel_path, "group_sqm_raw_channels", sources,
                   sci_threshold=sci_threshold,
                   psp_threshold=cutoffs["psp"],
                   cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq)

    return sqm_data


def load_group_sqm(
    output_dir: Path, group: list[GroupEntry], bads_scope: str = "run",
    scope_tasks: "list[str] | None" = None,
) -> dict[str, dict]:
    """Load per-subject SQM scalars and channel metrics from derivatives.

    A subject has one record and one channel-metrics CSV per run, and every entry names the
    task it belongs to, so the run's own files are the ones read. Reading all of them and
    letting the last win, as this used to, meant a five-task subject had four tasks quietly
    analysed with a fifth task's rejected channels.

    ``bads_scope`` decides what counts as a bad channel:

    - ``"run"``: this task's own rejections, matching the rest of the metrics returned here.
    - ``"subject"``: the union over the subject's runs, so a channel rejected in any
      condition is rejected in all of them. Conditions then rest on the same channel set,
      which is what a comparison between them needs; the cost is losing a channel everywhere
      because one segment was bad.

    ``scope_tasks`` bounds that union to the tasks the analysis covers, normally the ones the
    pairs table names. Without it the union is every ``desc-sci`` sidecar in the folder, so a
    subject who also sat a resting run, or a second experiment, loses channels here for a
    recording nobody asked about.

    Which scope is worth using depends on how the tree was produced, and the two are not
    always different. A recording preprocessed whole is screened once, so the subject has one
    run per task and the union over it is itself: ``subject`` and ``run`` then name the same
    set, and a line in the log says so. They differ when the conditions were cropped to
    separate tasks before prep, which is the case ``subject`` was added for.

    Returns {subject_id: sqm_dict}. Alongside the flattened scalars each dict carries
    ``windowed`` (the record's channel-by-window matrices), ``channel_order`` (their row
    order), ``per_channel`` (the whole per-channel section, which the dyad's channel table
    prints), ``screen_cutoffs`` (the lines that run screened by) and ``by_condition`` (the
    record's per-condition entries), which is what a per-condition view of the dyad pages
    is built from.
    """
    if bads_scope not in ("run", "subject"):
        raise ValueError(f"bads_scope must be 'run' or 'subject', got {bads_scope!r}")

    from fnirs_pipe.qc.subject.sqm_record import raw_verdict_view, record_glob

    result: dict[str, dict] = {}
    for entry in group:
        sqm: dict = {}
        # the long-channel view is the one a quality judgement wants, with raw standing in
        # when the montage has no short channels to exclude
        records = _member_sqm_files(output_dir, entry, record_glob(f"{entry.subject_id}*"))
        for record_path in _for_task(records, entry.task):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
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
            # the channel-by-window matrices, kept whole rather than flattened: they are
            # what a per-condition view is a column selection out of, and without them the
            # dyad pages can only print the whole recording's numbers under a condition's
            # heading. Nested under one key so they cannot collide with a scalar name.
            sqm["windowed"] = record.get("windowed") or {}
            sqm["channel_order"] = list(
                ((record.get("per_channel") or {}).get("raw") or {})
                .get("sci_per_channel") or {})
            # the record's whole per-channel section, not only the SCI its row order is read
            # from: the dyad's channel table prints the same columns the subject report does,
            # and those come from PSP, SNR, CV and the spike share alongside it
            per_ch = record.get("per_channel") or {}
            # every section, not only the all-channel one: a short channel's scores live in
            # `raw_short` and the dyad's channel table prints short rows too
            sqm["per_channel"] = per_ch
            sqm["per_channel_all"] = per_ch.get("raw") or {}
            # the long section too, and it is not a nicety: every dyad measure runs on long
            # channels, and a `raw_long` is not written when the montage is all long, so the
            # whole-file section is the long one there rather than a missing answer
            sqm["per_channel_long"] = per_ch.get("raw_long") or sqm["per_channel_all"]
            # the same screening grid `compute_group_sqm_raw` keeps when it measures one
            # itself, rebuilt here from the matrices the record stored. One shape, so a dyad
            # panel does not care which command produced the members' numbers.
            sqm["screen_windows"] = _screen_windows(record, sqm.get("screen_cutoffs"))
            # the per-condition numbers the record already holds, so the dyad pages read
            # them rather than cutting the matrices above a second time
            sqm["by_condition"] = record.get("by_condition") or {}
            # The same scalars kept split by channel set instead of collapsed onto the long
            # view above. The dyad quality table prints the three sets side by side, the way
            # a subject report's own metrics section does, so a reader is not handed one set
            # and left to trust that it was the right one to judge the dyad on.
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
                                     f"{entry.subject_id}*_channel_metrics.csv")
        marks, read_bads = ((sidecars, _bad_from_sidecar) if sidecars
                            else (csvs, _bad_from_csv))
        kind = "desc-sci sidecar" if sidecars else "channel metrics CSV"

        run_marks = _for_task(marks, entry.task)
        if not marks:
            logger.warning(
                "%s has neither a desc-sci sidecar nor channel metrics, so no channel is "
                "rejected for task-%s and every bad channel enters the inter-brain metrics. "
                "Rerun fnirs-pipe on this subject.", entry.subject_id, entry.task)
        elif not run_marks:
            logger.warning(
                "%s has a %s but none for task-%s, so no channel is rejected for it. "
                "Rerun fnirs-pipe on this task.", entry.subject_id, kind, entry.task)

        for path in run_marks:
            if sidecars:
                scores = _sci_from_sidecar(path)
                if scores:
                    sqm["sci_per_channel"] = scores
            else:
                try:
                    ch_df = pd.read_csv(path)
                    if {"name", "sci"}.issubset(ch_df.columns):
                        sqm["sci_per_channel"] = dict(
                            zip(ch_df["name"].astype(str),
                                pd.to_numeric(ch_df["sci"], errors="coerce"))
                        )
                except Exception:
                    pass
            sqm["bad_channels"] = read_bads(path)
            if sidecars:
                sqm["screen_cutoffs"] = _screen_cutoffs(path)
            logger.info("%s task-%s: %d rejected channel(s) from %s (%s)",
                        entry.subject_id, entry.task, len(sqm["bad_channels"]), kind, path.name)

        # which run each rejection came from, so the union is reviewable rather than a
        # channel list with no explanation of why a clean condition lost a channel
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
                # loud, because it reads as a choice that was made and was not. The union
                # over one run is that run, so the flag did nothing; a caller who passed it
                # expecting conditions to be unioned is looking at a tree where they are
                # not separate runs, and the union they wanted is already what prep did
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

    The cutoffs are the run's own where it recorded them, since a record screened at a
    different SCI line has to be re-masked at that line and not at this package's default.
    """
    from fnirs_pipe.qc.metrics._helpers import PSP_PASS, SCI_PASS
    from fnirs_pipe.qc.metrics.windowed import coupled_mask_from_matrices, window_centers

    windowed = record.get("windowed") or {}
    lines = cutoffs or {}
    sci = windowed.get("sci_matrix")
    psp = windowed.get("psp_matrix")
    mask = coupled_mask_from_matrices(sci, psp, float(lines.get("sci", SCI_PASS)),
                                      float(lines.get("psp", PSP_PASS)))
    times = windowed.get("sci_times")
    if mask is None or not times:
        return {}
    out = {"mask": mask,
           "centers": window_centers(np.asarray(times, dtype=float)),
           "sci": np.asarray(sci, dtype=float), "psp": np.asarray(psp, dtype=float),
           **({"cv": np.asarray(windowed["cv_matrix"], dtype=float)}
              if windowed.get("cv_matrix") else {}),
           "channel_order": list(((record.get("per_channel") or {}).get("raw") or {})
                                 .get("sci_per_channel") or {})}
    # GVTD is an RMS across channels, so it has no matrix and rides along as one series.
    # Only a stored record carries it; the dyad's own pass does not measure motion, and a
    # panel that draws it has to cope with the row being absent rather than assume it.
    gvtd = windowed.get("gvtd_per_window")
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
    try:
        return list(json.loads(json_path.read_text(encoding="utf-8")).get("bad_channels") or [])
    except (OSError, json.JSONDecodeError):
        return []


def _sci_from_sidecar(json_path: Path) -> dict[str, float]:
    """The per-channel SCI prep recorded for a run, from the same sidecar."""
    try:
        scores = json.loads(json_path.read_text(encoding="utf-8")).get("sci_scores") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(k): float(v) for k, v in scores.items()}


def _screen_cutoffs(json_path: Path) -> dict:
    """The screening lines a run used, off its ``desc-sci`` sidecar.

    ``{"sci": 0.8, "psp": 0.1, "good_frac": 0.75}``, or an empty dict when the sidecar
    predates them. A per-condition view rebuilds the coupled-window share from the stored
    matrices, and it has to apply the lines that subject was actually screened by: the
    registry defaults would produce a share no channel of theirs was ever judged against,
    and ``fnirs-hyper``'s own ``--sci-threshold`` is a colouring threshold for the dyad
    page, not the one prep ran.
    """
    try:
        params = json.loads(json_path.read_text(encoding="utf-8")).get("parameters") or {}
    except (OSError, json.JSONDecodeError):
        return {}
    named = {"sci": params.get("sci_threshold"), "psp": params.get("psp_threshold"),
             "good_frac": params.get("min_good_frac")}
    return {k: float(v) for k, v in named.items() if v is not None}


def _bad_from_csv(csv_path: Path) -> list[str]:
    try:
        ch_df = pd.read_csv(csv_path)
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

    The whole-run SCI is built with both spellings a few lines above, because the pages ask
    for ``"<pair> hbo"`` and fall back to the bare pair. The windowed estimate comes out of
    the metric registry keyed by the recording's own channel names instead, so handing it
    over unchanged makes every lookup miss and every channel read as unmeasured, with
    nothing to say so. SCI is a property of the pair, both wavelength rows carrying one
    number, so averaging them is a no-op that also survives a montage where it is not.
    """
    by_pair: dict[str, list[float]] = {}
    for ch, value in per_wavelength.items():
        by_pair.setdefault(str(ch).rsplit(" ", 1)[0], []).append(float(value))
    out: dict[str, float] = {}
    for pair, values in by_pair.items():
        out[pair] = out[f"{pair} hbo"] = sum(values) / len(values)
    return out


def resolve_group_bands(
    group: list[GroupEntry], group_sqm: dict[str, dict], override: dict | None = None,
) -> "tuple[float, float, float | None]":
    """The separation bands a dyad's inter-brain metrics run on, read off the members' records.

    ``fnirs-hyper run`` works on derivatives that prep has already split into long and short
    channels and stamped with the bands it split them by, so being *told* the bands again on
    the command line is an invitation to type a number that does not match the one on disk.
    Reading them back makes that mismatch impossible rather than merely documented.
    ``fnirs-qc hyper-raw`` is not a caller: it reads BIDS raw data and computes the record
    itself, so there is nothing on disk to read back and its flags stay the source of truth.

    Two members are two prep runs, so they can disagree. **That is refused, not
    reconciled.** The bands did not only choose which channels are long, they chose what
    short-channel regression removed from each member upstream, so a band derived from both
    would be stamped on a table that neither member was processed with. Nothing is lost by
    refusing: the homologous channel set already intersects on its own, because the metrics
    pair by S-D label and a label only one member calls long is simply absent from the
    other's map.

    ``override`` is the three flags as ``{"short_max_dist": ..., "long_min_dist": ...,
    "long_max_dist": ...}`` in metres, None for one left off. A value given wins and is
    logged as forced; one left off falls back to the *records*, not to the package default,
    so capping the long band does not silently re-assert the other two. An upper bound the
    records carry therefore cannot be switched off from the command line, which needs a
    re-run of prep.

    A member whose record predates the stamp reads back as today's defaults. That is a guess
    rather than a fact, so it is warned about and does not count towards agreement.
    """
    from fnirs_pipe.qc.metrics._helpers import (
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
            "Re-run fnirs-pipe on the odd one out so the two match, or pass "
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
            "no separation bands stamped for %s, so %s is assumed for %s: the record "
            "predates the stamp and what it was prepped with is not recoverable from it. "
            "Re-run fnirs-pipe on it, or pass --short-max-dist / --long-min-dist / "
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
