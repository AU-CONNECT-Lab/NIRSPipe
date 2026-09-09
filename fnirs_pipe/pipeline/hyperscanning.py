"""Hyperscanning pipeline utilities: group IO, alignment, and signal metrics."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, StageError
from fnirs_pipe.io.derivatives import find_preproc_snirf, group_data_dir
from fnirs_pipe.io.snirf import read_snirf
from fnirs_pipe.io.tables import read_table
from fnirs_pipe.pipeline.synchrony import (  # noqa: F401  re-exported
    WTCResult,
    compute_pairwise_coherence,
    compute_wtc,
    compute_wtc_pseudo,
    roi_maps_from_channels,
    roi_mean_of_channels,
    window_result,
    wtc_band_mean,
)
from fnirs_pipe.utils import is_optical_density
from fnirs_pipe.utils.lineage import lineage_of, path_from, stage_of
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.hyperscanning")


def _hyper_sidecar(path: Path, step: str, sources: list[str], **params) -> None:
    from fnirs_pipe import __version__
    from fnirs_pipe.io.derivatives import write_sidecar_json
    write_sidecar_json(path, {
        "pipeline_version": __version__,
        "step": step,
        "Sources": sources,
        "parameters": params,
    })


# ---- Data management: group definition & IO ----


@dataclass
class GroupEntry:
    group_id: str
    subject_id: str
    task: str
    # optional CSV columns, present when (group_id, subject_id, task) alone does not name
    # one recording: a subject recorded over several sessions, or several runs of one task
    session: str | None = None
    run: str | None = None


def parse_group_csv(csv_path: Path) -> dict[tuple[str, str], list[GroupEntry]]:
    """Parse group CSV into {(group_id, task): [GroupEntry, ...]}."""
    try:
        df = read_table(csv_path, dtype=str)
    except Exception as exc:
        raise GroupCSVError(f"Cannot read CSV {csv_path}: {exc}") from exc

    missing = {"group_id", "subject_id", "task"} - set(df.columns)
    if missing:
        raise GroupCSVError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["group_id", "subject_id", "task"])
    if df.empty:
        raise GroupCSVError("CSV contains no valid rows after dropping NaN values")

    def optional(row, column: str) -> str | None:
        """A session or run label if the CSV carries one for this row, else None."""
        if column not in df.columns:
            return None
        value = str(row[column]).strip()
        return value or None if value.lower() not in ("nan", "none", "") else None

    result: dict[tuple[str, str], list[GroupEntry]] = {}
    for _, row in df.iterrows():
        key = (str(row["group_id"]).strip(), str(row["task"]).strip())
        entry = GroupEntry(
            group_id=str(row["group_id"]).strip(),
            subject_id=str(row["subject_id"]).strip(),
            task=str(row["task"]).strip(),
            session=optional(row, "session"),
            run=optional(row, "run"),
        )
        result.setdefault(key, []).append(entry)

    for (gid, task), members in result.items():
        if len(members) < 2:
            raise GroupCSVError(
                f"Group '{gid}' task '{task}' has only {len(members)} subject — need at least 2"
            )

    return result


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


def load_group_haemo(
    output_dir: Path,
    group: list[GroupEntry],
    desc: str = "preproc",
) -> dict[str, mne.io.Raw]:
    """Load one haemoglobin stage per subject, selected by its desc entity.

    "preproc" is Beer-Lambert output, the stage every montage has. Anything the post
    pipeline wrote is equally valid input here: "filtered", "resampled", "errts" (the
    confound-regression residual, which is what an inter-brain metric usually wants, since
    short-channel regression removes the systemic physiology two people in one room share).

    Returns {subject_id: raw_haemo}.
    Raises MissingDerivativesError if any SNIRF is absent, StageError if one holds optical
    density rather than concentration.
    """
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        snirf_path = find_preproc_snirf(output_dir, entry.subject_id, entry.task, desc=desc,
                                        session=entry.session, run=entry.run)
        raw = read_snirf(snirf_path, verbose=False)
        if is_optical_density(raw):
            raise StageError(
                f"{snirf_path.name} holds optical density, not haemoglobin concentration. "
                f"desc-{desc} is a pre-Beer-Lambert stage; pick one at or after it "
                "(preproc, filtered, resampled, errts, errtsbroad)."
            )
        result[entry.subject_id] = raw
    return result


def warn_outside_passband(raws: dict[str, mne.io.Raw], fmin: float, fmax: float) -> None:
    """Warn when a requested frequency range reaches past the bandpass the files record.

    Reported rather than enforced: a wider range is occasionally deliberate. The passband
    comes from the sidecar, so it is what the file went through rather than what was asked for.
    """
    for subject_id, raw in raws.items():
        lin = lineage_of(raw)
        params = (lin.params if lin else None) or {}
        low, high = params.get("high_pass"), params.get("low_pass")
        outside = []
        if low is not None and fmin < low:
            outside.append(f"{fmin} Hz is below its {low} Hz high-pass")
        if high is not None and fmax > high:
            outside.append(f"{fmax} Hz is above its {high} Hz low-pass")
        if outside:
            logger.warning("sub-%s | requested %s-%s Hz but %s. Those scales carry what the "
                           "filter removed, not signal",
                           subject_id, fmin, fmax, " and ".join(outside))


def unfiltered_stage_note(raws: dict[str, mne.io.Raw]) -> "str | None":
    """A sentence for the ISC panel when the files record no bandpass, else None.

    ISC is a whole-record zero-lag correlation and so has no frequency axis to keep drift
    out of. On an unfiltered stage it is dominated by the slowest component present, and two
    members recorded in one room drift together for instrumental and environmental reasons
    that are not neural. Excluding the short channels does not help: long channels carry the
    same drift. WTC is unaffected, since its band mean averages only the cells inside the
    requested band.

    ``--desc`` defaults to ``preproc``, which is Beer-Lambert output and is not bandpassed,
    so the default is the case this warns about.

    The passband comes from the sidecar through the lineage stamp, the same route
    :func:`warn_outside_passband` reads, so a file whose sidecar is missing looks the same as
    one that was never filtered. The wording says "record no bandpass" rather than "are
    unfiltered" for that reason.
    """
    unrecorded = []
    for subject_id, raw in sorted(raws.items()):
        lin = lineage_of(raw)
        if ((lin.params if lin else None) or {}).get("high_pass") is None:
            unrecorded.append(subject_id)
    if not unrecorded:
        return None
    stages = sorted({stage_of(raw) or "?" for raw in raws.values()})
    return (
        f"The files for {', '.join(unrecorded)} record no bandpass "
        f"(stage {', '.join(repr(s) for s in stages)}). A whole-record correlation has no "
        "frequency axis, so drift and systemic physiology enter it directly, and two members "
        "recorded together drift alike. The wavelet coherence panels are unaffected. Point "
        "--desc at a filtered stage (filtered, errts) to read these numbers as neural."
    )


def load_group_raw_bids(bids_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load raw CW-amplitude SNIRF from BIDS for each group member.

    Returns {subject_id: raw_intensity}.
    Raises MissingDerivativesError if no SNIRF is found for any member.
    """
    from fnirs_pipe.io.bids import get_layout, get_nirs_files

    from fnirs_pipe.io.derivatives import select_one_run

    layout = get_layout(bids_dir, validate=False)
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        sub_label = entry.subject_id.removeprefix("sub-")
        files = get_nirs_files(layout, subject=sub_label, session=entry.session,
                               task=entry.task)
        path = select_one_run(
            sorted(files), what="raw SNIRF in BIDS", subject_id=entry.subject_id,
            task=entry.task, session=entry.session, run=entry.run,
        )
        result[entry.subject_id] = read_snirf(path, verbose=False)
    return result


def _raw_to_haemo(raw: mne.io.Raw, dpf: list[float]) -> mne.io.Raw:
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
    # Single value applies to both wavelengths; a list gives one PPF per wavelength.
    ppf = dpf[0] if len(dpf) == 1 else dpf
    return mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)


# ---- Computation: raw-level QC ----


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
    :func:`~fnirs_pipe.qc.sqm_record.raw_verdict_view` from the same three sections the
    per-subject record holds. It used to be one all-channel pass, which put a subject's
    SCI, CV, SNR and GVTD in this table on a different channel set than the same subject's
    numbers in the individual reports and in `fnirs-hyper run`, so the two could not be
    read against each other.
    """
    from fnirs_pipe.qc.metrics import resolve_cutoffs, screen_channels, screening_scores
    from fnirs_pipe.qc.screen_scope import resolve_screen_scope
    from fnirs_pipe.qc.sqm_record import raw_sections, raw_verdict_view

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
        if raw_od is not None:
            screen = screening_scores(raw_od, cardiac_l_freq, cardiac_h_freq,
                                      have={"sci": sci_cw}, cutoffs=cutoffs,
                                      scope=resolve_screen_scope(raw, screen_scope))
            bad_channels, _ = screen_channels(screen, cutoffs)

        try:
            sections, _per_channel = raw_sections(
                raw, sci_cw, bad_channels, cardiac_l_freq, cardiac_h_freq, sep_bands,
                screen.get("good_frac"))
            sqm = raw_verdict_view(sections)
        except Exception:
            logger.warning("%s: quality metrics failed", entry.subject_id, exc_info=True)
            sections, sqm = {}, {}

        # which channel set the row above describes, so a table read on its own says so
        sqm["channel_set"] = "long" if sections.get("raw_long") else "all"
        sqm["sci_per_channel"] = sci_scores
        sqm["bad_channels"]    = bad_channels
        sqm_data[entry.subject_id] = sqm

        # Every scalar the record holds, not a whitelist: the three columns this used to
        # write left SCI as the only quality metric a hyperscanning study ever saw, and SCI
        # is amplitude-invariant, so a run with a collapsed cardiac pulse reads as fine
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


# ---- Data management: alignment & signal preprocessing ----


def align_recordings(
    raws: dict[str, mne.io.Raw],
    task: str,
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Align N recordings by the first shared annotation trigger.

    Crops each raw from its first occurrence of a common trigger description,
    then trims all to the same duration (shortest post-crop).

    Returns (aligned_raws, {subject_id: crop_offset_seconds}).
    Raises AlignmentError if no shared trigger exists across all subjects.
    """
    # Collect each subject's trigger descriptions, dropping BAD_* motion annotations.
    desc_sets: dict[str, set[str]] = {}
    for sub_id, raw in raws.items():
        desc_sets[sub_id] = {
            ann["description"]
            for ann in raw.annotations
            if not str(ann["description"]).upper().startswith("BAD")
        }

    if not any(desc_sets.values()):
        raise AlignmentError(
            "No non-BAD annotations found in any recording. "
            "Cannot align without shared trigger events."
        )

    # A trigger usable for alignment must be present in every subject.
    common: set[str] = set.intersection(*desc_sets.values()) if desc_sets else set()
    if not common:
        all_descs = {sub: sorted(d) for sub, d in desc_sets.items()}
        raise AlignmentError(
            f"No shared trigger descriptions across all participants. "
            f"Per-subject descriptions: {all_descs}. "
            "Note: alignment requires a shared hardware trigger."
        )

    # Offset = onset of each subject's earliest common trigger (sort by onset).
    offsets: dict[str, float] = {}
    for sub_id, raw in raws.items():
        for ann in sorted(raw.annotations, key=lambda a: float(a["onset"])):
            if ann["description"] in common:
                offsets[sub_id] = float(ann["onset"])
                break
        if sub_id not in offsets:
            raise AlignmentError(f"Subject {sub_id}: no common trigger found (unexpected state)")

    # Crop each recording to start at its trigger, then clip all to a shared length.
    aligned: dict[str, mne.io.Raw] = {}
    for sub_id, raw in raws.items():
        aligned[sub_id] = raw.copy().crop(tmin=offsets[sub_id])

    min_duration = min(r.times[-1] for r in aligned.values())
    for sub_id in list(aligned):
        aligned[sub_id].crop(tmax=min_duration)

    return aligned, offsets


def trim_to_shortest(
    raws: dict[str, mne.io.Raw],
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Trim all recordings to the shortest duration without trigger-based alignment.

    Use for resting-state data where no shared trigger exists.
    Returns (trimmed_raws, {subject_id: 0.0}).
    """
    min_duration = min(r.times[-1] for r in raws.values())
    trimmed = {sid: raw.copy().crop(tmax=min_duration) for sid, raw in raws.items()}
    offsets = {sid: 0.0 for sid in raws}
    return trimmed, offsets


def crop_aligned_window(
    raws: dict[str, mne.io.Raw],
    tstart: float | None,
    tend: float | None,
) -> dict[str, mne.io.Raw]:
    """Cut every aligned recording down to [tstart, tend] on the shared post-alignment clock.

    Runs after `align_recordings` / `trim_to_shortest`, where t=0 is the shared trigger
    (or the common start) and all recordings already have one length. That is what makes a
    single window valid for the whole group: the same [tstart, tend] names the same moment
    of the task in every subject, which it would not on the raw per-subject clocks.

    Example: a 600 s aligned dyad with tstart=60, tend=300 returns both recordings cut to
    the 240 s of task between them, so the coherence never sees the baseline or the wrap-up.

    `tend` past the end of the data is clipped rather than refused: recordings differ in
    length and an over-long window is a request for "to the end", not a mistake. A `tstart`
    at or past the end has no data to describe and raises.
    """
    if tstart is None and tend is None:
        return raws

    duration = min(float(r.times[-1]) for r in raws.values())
    t0 = 0.0 if tstart is None else float(tstart)
    t1 = duration if tend is None else min(float(tend), duration)
    if t0 >= duration:
        raise AlignmentError(
            f"--tstart {t0:g}s is at or past the aligned recording length ({duration:g}s)"
        )
    if t1 <= t0:
        raise AlignmentError(f"empty window: --tstart {t0:g}s is not before --tend {t1:g}s")
    if tend is not None and tend > duration:
        logger.warning("--tend %gs exceeds the aligned length %gs; using %gs",
                       tend, duration, duration)

    return {sid: raw.copy().crop(tmin=t0, tmax=t1) for sid, raw in raws.items()}


def normalize_raws(raws: dict[str, mne.io.Raw]) -> dict[str, mne.io.Raw]:
    """Z-score each channel independently per subject (mean=0, std=1 across time).

    Applied after alignment so all subjects share the same time axis.
    Channels with near-zero variance are left unchanged (divided by 1.0).

    Affects signal overlay display only; coherence and ISC values are scale-invariant
    and unchanged. SCI / bad-channel detection run on raw CW data before normalization.
    """
    result: dict[str, mne.io.Raw] = {}
    for sid, raw in raws.items():
        r = raw.copy()
        data = r.get_data()
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True)
        r._data[:] = (data - mu) / np.where(sd < 1e-12, 1.0, sd)
        result[sid] = r
    return result


# ---- Data management: derivatives IO ----


def _for_task(paths: list[Path], task: str) -> list[Path]:
    """The subset of paths belonging to one task.

    ``[sub-01_task-rest_x, sub-01_task-game_x], "rest"`` -> the rest one
    ``[sub-01_x], "rest"``                               -> that one (nothing is labelled)
    ``[sub-01_task-game_x], "rest"``                     -> nothing

    The second case covers derivatives with no task entity at all, where the single
    unlabelled file is the right answer. The third returns empty on purpose: substituting
    another task's file is the failure this function exists to prevent.
    """
    matched = [p for p in paths if f"_task-{task}_" in p.name]
    if matched or any("_task-" in p.name for p in paths):
        return matched
    return paths


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


def load_group_sqm(
    output_dir: Path, group: list[GroupEntry], bads_scope: str = "run",
) -> dict[str, dict]:
    """Load per-subject SQM scalars and channel metrics from derivatives.

    A subject has one record and one channel-metrics CSV per run, and every entry names the
    task it belongs to, so the run's own files are the ones read. Reading all of them and
    letting the last win, as this used to, meant a five-task subject had four tasks quietly
    analysed with a fifth task's rejected channels.

    ``bads_scope`` decides what counts as a bad channel:

    - ``"run"``: this task's own rejections, matching the rest of the metrics returned here.
    - ``"subject"``: the union over every run of the subject, so a channel rejected in any
      condition is rejected in all of them. Conditions then rest on the same channel set,
      which is what a comparison between them needs; the cost is losing a channel everywhere
      because one segment was bad.

    Returns {subject_id: sqm_dict}.
    """
    if bads_scope not in ("run", "subject"):
        raise ValueError(f"bads_scope must be 'run' or 'subject', got {bads_scope!r}")

    from fnirs_pipe.qc.sqm_record import raw_verdict_view

    result: dict[str, dict] = {}
    for entry in group:
        nirs_dir = output_dir / entry.subject_id / "nirs"

        sqm: dict = {}
        # the long-channel view is the one a quality judgement wants, with raw standing in
        # when the montage has no short channels to exclude
        records = sorted(nirs_dir.glob(f"{entry.subject_id}*_desc-sqm_nirs.json"))
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

        # Rejection is read from the desc-sci sidecars, which prep writes on every run and
        # which name every channel the run rejected whatever came after. The channel-metrics
        # CSV holds the same set, but the report writes that one, so a tree produced with
        # --no-report has the sidecars and no CSV and used to end up with nothing rejected
        # at all, silently.
        sidecars = sorted(nirs_dir.glob(f"{entry.subject_id}*_desc-sci_nirs.json"))
        csvs     = sorted(nirs_dir.glob(f"{entry.subject_id}*_channel_metrics.csv"))
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
            logger.info("%s task-%s: %d rejected channel(s) from %s (%s)",
                        entry.subject_id, entry.task, len(sqm["bad_channels"]), kind, path.name)

        # which run each rejection came from, so the union is reviewable rather than a
        # channel list with no explanation of why a clean condition lost a channel
        sources: dict[str, list[str]] = {
            ch: [entry.task] for ch in (sqm.get("bad_channels") or [])
        }
        if bads_scope == "subject":
            for path in marks:
                from_task = m.group(1) if (m := re.search(r"_task-([A-Za-z0-9]+)", path.name)) else entry.task
                for ch in read_bads(path):
                    if from_task not in sources.setdefault(ch, []):
                        sources[ch].append(from_task)
            sqm["bad_channels"] = sorted(sources)
            logger.info("%s: --bads-scope subject unions %d channel(s) over %d run(s) of %s",
                        entry.subject_id, len(sources), len(marks), kind)
        sqm["bad_channel_sources"] = {ch: sorted(t) for ch, t in sources.items()}

        result[entry.subject_id] = sqm

    return result

# ---- Separation bands ----

_BANDS_FIELDS = ("short_max_dist", "long_min_dist", "long_max_dist")


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
