"""Putting a group's recordings on one clock, and saying afterwards what was done.

:func:`alignment_params` is stamped onto every inter-brain table, because ``--no-align`` and
an alignment whose trigger sits at t=0 both leave every offset at zero and nothing else on
disk can tell the two apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from nirspipe.exceptions import AlignmentError
from nirspipe.io.auxiliary import ImuTrace
from nirspipe.io.snirf import write_snirf
from nirspipe.pipeline.hyper._helpers import _zscore_rows
from nirspipe.utils import is_marker
from nirspipe.utils.snirf_prep import annotations_to_df, bids_stem, copy_sidecars
from nirspipe.utils.lineage import lineage_of
from nirspipe.utils.lineage import path_from
from nirspipe.utils.lineage import restamp
from nirspipe.utils.lineage import stamp
from nirspipe.utils.logging import get_logger
from nirspipe.io.tables import write_tsv

logger = get_logger("pipeline.alignment")


# ---- Alignment ----
#
# The stage every inter-brain number is computed at. Both routes onto it stamp their
# outputs with it, which is what lets a consumer tell them apart: `--no-align` and an
# alignment whose shared trigger happens to sit at t=0 both leave every offset at zero,
# and they are not the same analysis.
ALIGN_STAGE = "aligned"

# What the source's stamp says about the data rather than about the step that made it, and
# so is still true of the aligned copy. The same keys `read_snirf` restores off the sidecar.
_CARRIED_KEYS = ("high_pass", "low_pass", "filter_method", "filter_order")


def _stamp_alignment(raw: mne.io.Raw, source: mne.io.Raw, step: str,
                     **params) -> mne.io.Raw:
    """Record which alignment produced one output, keeping the file it came from.

    ``stamp`` replaces the whole lineage entry, so the source's path has to be carried
    across; without it ``path_from`` returns None and every sidecar built from these
    objects loses its ``Sources``.

    The passband is carried too: it describes the data rather than this step, and every
    consumer downstream of the alignment sees only this stamp.
    """
    prev = lineage_of(source)
    carried = {k: v for k, v in ((prev.params if prev else None) or {}).items()
               if k in _CARRIED_KEYS and k not in params}
    return stamp(raw, ALIGN_STAGE, step, source=source, path=path_from(source),
                 **carried, **params)


def alignment_params(raws: dict[str, mne.io.Raw]) -> dict:
    """What put these recordings on one clock, for a sidecar beside a number read off them.

    ::

      {"aligned": True, "align_step": "align_recordings",
       "align_trigger": {"sub-01": "start", "sub-02": "start"},
       "align_offset_s": {"sub-01": 0.0, "sub-02": 22.4},
       "aligned_duration_s": 3900.0}

    Every inter-brain metric assumes the members share a time axis, and this is what says on
    disk whether they were put on one. ``aligned: null`` means the recordings
    reached the metric without going through either route, which is the case worth catching:
    a reader cannot tell it from a successful alignment by looking at the numbers.

    Read off the lineage stamps rather than passed in, so a sidecar cannot claim an
    alignment that did not run.
    """
    stamps = {sid: lineage_of(raw) for sid, raw in raws.items()}
    aligned = {sid: lin for sid, lin in stamps.items()
               if lin is not None and lin.stage == ALIGN_STAGE}
    if len(aligned) != len(stamps) or not aligned:
        logger.warning("these recordings carry no alignment stamp, so nothing can be "
                       "recorded about the clock the metrics were computed on")
        return {"aligned": None}

    steps = {lin.step for lin in aligned.values()}
    first = next(iter(aligned.values()))
    out: dict = {
        "aligned": bool(first.params.get("aligned")),
        # one word: two members aligned by different routes is not a state the callers produce
        "align_step": steps.pop() if len(steps) == 1 else sorted(steps),
        "align_offset_s": {sid: lin.params.get("offset_s") for sid, lin in aligned.items()},
        "aligned_duration_s": first.params.get("duration_s"),
    }
    triggers = {sid: lin.params.get("trigger") for sid, lin in aligned.items()}
    if any(triggers.values()):
        out["align_trigger"] = triggers
    return out


def write_aligned_member(raw_aligned: mne.io.Raw, snirf_path, out_dir, group_id: str):
    """One member's aligned recording, its events and its sidecar, beside each other.

    ::

      out_dir/sub-01_task-rest_nirs.snirf, _events.tsv, and _nirs.json gaining
      {"align_group": "G1", "align_step": "align_recordings", "align_trigger": "start",
       "align_offset_s": 22.4, "aligned_duration_s": 3900.0}

    The offset goes in the member's own sidecar. The copied BIDS sidecar is added to, not
    replaced: the fields a raw recording has to carry are still needed there. Read off the
    lineage stamp, like
    :func:`alignment_params`, so the sidecar cannot claim an alignment that did not run.
    """
    snirf_path, out_dir = Path(snirf_path), Path(out_dir)
    stem = bids_stem(snirf_path)
    sidecar = out_dir / f"{stem}_nirs.json"
    # the tree holds one copy per recording, so a member two groups share would carry
    # whichever group was written last, cut at the other group's offset
    try:
        earlier = json.loads(sidecar.read_text(encoding="utf-8")).get("align_group")
    except (OSError, json.JSONDecodeError):
        earlier = None
    if earlier is not None and earlier != group_id:
        raise AlignmentError(
            f"{stem} is already aligned for group {earlier}, and group {group_id} would "
            f"overwrite it; align groups that share a member into separate output trees")
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecars(snirf_path, stem, out_dir)

    out_snirf = out_dir / f"{stem}_nirs.snirf"
    write_snirf(raw_aligned, out_snirf)
    write_tsv(annotations_to_df(raw_aligned), out_dir / f"{stem}_events.tsv")

    lin = lineage_of(raw_aligned)
    if lin is None or lin.stage != ALIGN_STAGE:
        raise AlignmentError(f"{stem}: carries no alignment stamp, so it was not aligned")
    fields =json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    fields.update({
        "align_group": group_id,
        "align_step": lin.step,
        "align_trigger": lin.params.get("trigger"),
        "align_offset_s": round(float(lin.params.get("offset_s") or 0.0), 3),
        "aligned_duration_s": lin.params.get("duration_s"),
    })
    sidecar.write_text(json.dumps(fields, indent=2), encoding="utf-8")
    return out_snirf


def align_recordings(
    raws: dict[str, mne.io.Raw],
    task: str,
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Align N recordings by the first shared annotation trigger.

    Crops each raw from its first occurrence of a common trigger description,
    then trims all to the same duration (shortest post-crop).

    Returns (aligned_raws, {subject_id: crop_offset_seconds}). Each output carries an
    ``aligned`` lineage stamp naming the trigger and the offset it was cut at, which
    :func:`alignment_params` reads back for the sidecars.

    Raises AlignmentError if no shared trigger exists across all subjects.
    """
    # Collect each subject's trigger descriptions, dropping BAD_ spans and EDGE joins.
    desc_sets: dict[str, set[str]] = {}
    for sub_id, raw in raws.items():
        desc_sets[sub_id] = {
            ann["description"]
            for ann in raw.annotations
            if is_marker(ann["description"])
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
    triggers: dict[str, str] = {}
    for sub_id, raw in raws.items():
        for ann in sorted(raw.annotations, key=lambda a: float(a["onset"])):
            if ann["description"] in common:
                offsets[sub_id] = float(ann["onset"])
                triggers[sub_id] = str(ann["description"])
                break
        if sub_id not in offsets:
            raise AlignmentError(f"Subject {sub_id}: no common trigger found (unexpected state)")

    # Crop each recording to start at its trigger, then clip all to a shared length.
    aligned: dict[str, mne.io.Raw] = {}
    for sub_id, raw in raws.items():
        aligned[sub_id] = raw.copy().crop(tmin=offsets[sub_id])

    min_duration = min(r.times[-1] for r in aligned.values())
    for sub_id, raw in raws.items():
        aligned[sub_id].crop(tmax=min_duration)
        # stamped after both crops, so the duration recorded is the one the metrics saw
        _stamp_alignment(aligned[sub_id], raw, "align_recordings", aligned=True,
                         trigger=triggers[sub_id], offset_s=offsets[sub_id],
                         duration_s=float(min_duration))

    logger.info("aligned %d recording(s) on %r, offsets %s, %.1f s kept", len(aligned),
                sorted(set(triggers.values())),
                {s: round(o, 3) for s, o in offsets.items()}, min_duration)
    return aligned, offsets


def align_like(
    raws: dict[str, mne.io.Raw],
    aligned_raws: dict[str, mne.io.Raw],
) -> dict[str, mne.io.Raw]:
    """Cut a second copy of the same recordings onto the clock ``aligned_raws`` sit on.

    ::

        align_like({"sub-01": intensity}, {"sub-01": haemo_aligned})
          -> {"sub-01": intensity cropped to the same window}

    The dyad raw report converts to haemoglobin before aligning, so the aligned objects are
    no longer optical density and nothing downstream can take GVTD or a carpet off them.
    This brings the intensity copy onto the same window instead of aligning it a second
    time, so it cannot land on a trigger other than the one every other panel is drawn
    against.

    ``crop`` accumulates into ``first_samp``, so the shift already applied to a member is
    ``aligned.first_time - raw.first_time`` whatever produced it, trigger alignment, a plain
    trim, or a ``--tstart`` window on top. A member the aligned set does not carry, or one
    whose window runs past the end of this copy, is dropped rather than returned short.
    """
    out: dict[str, mne.io.Raw] = {}
    for sid, raw in raws.items():
        ref = aligned_raws.get(sid)
        if ref is None:
            continue
        tmin = _aligned_shift(raw, ref)
        tmax = tmin + float(ref.times[-1])
        if tmin < -1e-6 or tmax > float(raw.times[-1]) + 1e-6:
            logger.warning("%s: the aligned window (%.1f-%.1f s) is not inside this copy "
                           "(0-%.1f s); dropping it from the motion panel",
                           sid, tmin, tmax, float(raw.times[-1]))
            continue
        # the tolerance above lets float round-off past the last sample through; crop refuses it
        out[sid] = raw.copy().crop(tmin=max(tmin, 0.0), tmax=min(tmax, float(raw.times[-1])))
    return out


def aligned_offsets(
    raws: dict[str, mne.io.Raw], aligned_raws: dict[str, mne.io.Raw],
) -> dict[str, float]:
    """Seconds into each original recording at which its aligned copy starts.

    ::

      trigger at 22.4 s, then a window from 60 s  ->  {"sub-02": 82.4}

    The aligner's own offsets stop at the trigger. A window cut afterwards moves the shared
    clock's zero again, and this is the offset that converts after both.
    """
    return {sid: _aligned_shift(raws[sid], ref) for sid, ref in aligned_raws.items()
            if sid in raws}


# how far two members' copies of one trigger may sit apart and still be the same condition
_TRIGGER_JITTER_SAMPLES = 2.0


def onset_residuals(raws: dict[str, mne.io.Raw], subject_ids: list[str]) -> list[dict]:
    """Each block's onset on the shared clock, minus the first member's copy of it.

    ::

      ca at 30.0 s in sub-01 and 30.098 s in sub-02
        -> [{"condition": "ca", "subject_id": "sub-02", "residual_s": 0.098}]

    What alignment cannot remove: a trigger that landed late in one member, or two clocks
    drifting apart. Blocks are matched by description and by order of occurrence, a repeat
    labelled ``ca (2)``; a block the first member or this member lacks has no row.
    """
    def onsets(raw: mne.io.Raw) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for a in raw.annotations:
            if is_marker(a["description"]):
                out.setdefault(str(a["description"]), []).append(
                    float(a["onset"]) - float(raw.first_time))
        return {desc: sorted(times) for desc, times in out.items()}

    if not subject_ids or subject_ids[0] not in raws:
        return []
    ref = onsets(raws[subject_ids[0]])
    rows = []
    for sid in subject_ids[1:]:
        if sid not in raws:
            continue
        own = onsets(raws[sid])
        for desc, ref_times in ref.items():
            for i, (t_ref, t_own) in enumerate(zip(ref_times, own.get(desc, []))):
                rows.append({"condition": desc if len(ref_times) == 1 else f"{desc} ({i + 1})",
                             "subject_id": sid, "residual_s": t_own - t_ref})
    return rows


def _aligned_shift(raw: mne.io.Raw, aligned: mne.io.Raw) -> float:
    """Seconds into ``raw`` at which ``aligned`` starts; ``crop`` accumulates into first_samp."""
    return float(aligned.first_time) - float(raw.first_time)


def align_imu_like(
    imu: "dict[str, dict[str, ImuTrace]]",
    raws: dict[str, mne.io.Raw],
    aligned_raws: dict[str, mne.io.Raw],
) -> "dict[str, dict[str, ImuTrace]]":
    """Each member's IMU traces moved onto the clock ``aligned_raws`` sit on.

    ::

        member aligned 22.4 s in, gyro jolt at 30.0 s on its own clock
          -> the same jolt at 7.6 s on the shared clock

    ``imu`` is ``{sid: {sensor: ImuTrace}}`` on each member's own clock, zero at the first
    sample of ``raws[sid]``, the same assumption the aux regressors make. The shift is the one
    :func:`align_like` cuts the optical copy by, so the two cannot disagree. Samples outside
    the aligned window are dropped; a member the aligned set does not carry is left out.
    """
    out: dict[str, dict[str, ImuTrace]] = {}
    for sid, traces in imu.items():
        raw, ref = raws.get(sid), aligned_raws.get(sid)
        if raw is None or ref is None or not traces:
            continue
        shift, end = _aligned_shift(raw, ref), float(ref.times[-1])
        out[sid] = {}
        for sensor, trace in traces.items():
            t_shared = np.asarray(trace.t, dtype=float) - shift
            keep = (t_shared >= 0.0) & (t_shared <= end)
            out[sid][sensor] = ImuTrace(t_shared[keep], np.asarray(trace.y)[keep], trace.unit)
    return out


def trim_to_shortest(
    raws: dict[str, mne.io.Raw],
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Trim all recordings to the shortest duration without trigger-based alignment.

    Use for resting-state data where no shared trigger exists.
    Returns (trimmed_raws, {subject_id: 0.0}).

    The outputs carry an ``aligned`` stamp too, with ``aligned=False`` on it: the offsets
    are zero here and can legitimately be zero after a real alignment as well, so the flag
    is the only thing that tells a reader which of the two produced the numbers.
    """
    min_duration = min(r.times[-1] for r in raws.values())
    trimmed = {sid: _stamp_alignment(raw.copy().crop(tmax=min_duration), raw,
                                     "trim_to_shortest", aligned=False, offset_s=0.0,
                                     duration_s=float(min_duration))
               for sid, raw in raws.items()}
    offsets = {sid: 0.0 for sid in raws}
    logger.warning("no trigger alignment: %d recording(s) trimmed to %.1f s and assumed to "
                   "share a clock already", len(trimmed), min_duration)
    return trimmed, offsets


def crop_aligned_window(
    raws: dict[str, mne.io.Raw],
    tstart: float | None,
    tend: float | None,
) -> dict[str, mne.io.Raw]:
    """Actually cut every aligned recording down to ``[tstart, tend]``.

    For a metric with no frequency axis of its own, where a window carries no edge the whole
    record would not have had: the Welch coherence and the signal overlays of the raw dyad
    report. ``nirspipe-hyper`` does **not** use this. Its window goes through
    :func:`resolve_analysis_window` instead and is taken out of the wavelet transform, a cut
    stretch transformed alone having two edges and a cone of influence of its own.

    ``tend`` past the end of the data is clipped rather than refused: recordings differ in
    length and an over-long window is a request for "to the end", not a mistake.
    """
    window = resolve_analysis_window(raws, tstart, tend)
    if window is None:
        return raws
    t0, t1 = window
    cut = {sid: raw.copy().crop(tmin=t0, tmax=t1) for sid, raw in raws.items()}
    for raw in cut.values():
        lin = lineage_of(raw)
        if lin is not None and lin.stage == ALIGN_STAGE:
            # the cut moves the shared zero, so each member's offset onto it moves with it
            restamp(raw, offset_s=float(lin.params.get("offset_s") or 0.0) + t0,
                    duration_s=float(raw.times[-1]))
    return cut


def resolve_analysis_window(
    raws: dict[str, mne.io.Raw],
    tstart: float | None,
    tend: float | None,
) -> "tuple[float, float] | None":
    """Validate ``--tstart``/``--tend`` against the aligned recordings and return the window.

    Runs after `align_recordings` / `trim_to_shortest`, where t=0 is the shared trigger
    (or the common start) and all recordings already have one length. That is what makes a
    single window valid for the whole group: the same [tstart, tend] names the same moment
    of the task in every subject, which it would not on the raw per-subject clocks.

    Example: a 600 s aligned dyad with tstart=60, tend=300 gives ``(60.0, 300.0)``, and the
    coherence reported is the 240 s between them.

    **It returns a window; it does not cut.** The recordings stay whole and the window is
    taken out of the wavelet transform afterwards, which is the same route
    ``--wtc-by-condition`` takes (see :func:`~nirspipe.pipeline.hyper.wtc.window_result`).

    ``tend`` past the end of the data is clipped rather than refused: recordings differ in
    length and an over-long window is a request for "to the end", not a mistake. A ``tstart``
    at or past the end has no data to describe and raises.
    """
    if tstart is None and tend is None:
        return None

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
    logger.info("window %.1f-%.1f s of %.1f s", t0, t1, duration)
    return t0, t1


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
        r._data[:] = _zscore_rows(r.get_data())
        # what a figure reads to label the traces z rather than a concentration
        result[sid] = restamp(r, normalized=True)
    return result
