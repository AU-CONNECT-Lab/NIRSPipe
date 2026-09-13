"""Putting a group's recordings on one clock, and saying afterwards what was done.

:func:`alignment_params` is stamped onto every inter-brain table, because ``--no-align`` and
an alignment whose trigger sits at t=0 both leave every offset at zero and nothing else on
disk can tell the two apart.
"""

from __future__ import annotations

import mne
import numpy as np

from fnirs_pipe.exceptions import AlignmentError
from fnirs_pipe.utils.lineage import lineage_of
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.lineage import stamp
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.alignment")


# ---- Alignment ----
#
# The stage every inter-brain number is computed at. Both routes onto it stamp their
# outputs with it, which is what lets a consumer tell them apart: `--no-align` and an
# alignment whose shared trigger happens to sit at t=0 both leave every offset at zero,
# and they are not the same analysis.
ALIGN_STAGE = "aligned"


def _stamp_alignment(raw: mne.io.Raw, source: mne.io.Raw, step: str,
                     **params) -> mne.io.Raw:
    """Record which alignment produced one output, keeping the file it came from.

    ``stamp`` replaces the whole lineage entry, so the source's path has to be carried
    across; without it ``path_from`` returns None and every sidecar built from these
    objects loses its ``Sources``.
    """
    return stamp(raw, ALIGN_STAGE, step, source=source, path=path_from(source), **params)


def alignment_params(raws: dict[str, mne.io.Raw]) -> dict:
    """What put these recordings on one clock, for a sidecar beside a number read off them.

    ::

      {"aligned": True, "align_step": "align_recordings",
       "align_trigger": {"sub-01": "start", "sub-02": "start"},
       "align_offset_s": {"sub-01": 0.0, "sub-02": 22.4},
       "aligned_duration_s": 3900.0}

    Every inter-brain metric assumes the members share a time axis, and until this nothing
    on disk said whether they had been put on one. ``aligned: null`` means the recordings
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
        # one word rather than a set: two members aligned by different routes is not a
        # state the callers can produce, and saying so loudly beats writing a list
        "align_step": steps.pop() if len(steps) == 1 else sorted(steps),
        "align_offset_s": {sid: lin.params.get("offset_s") for sid, lin in aligned.items()},
        "aligned_duration_s": first.params.get("duration_s"),
    }
    triggers = {sid: lin.params.get("trigger") for sid, lin in aligned.items()}
    if any(triggers.values()):
        out["align_trigger"] = triggers
    return out


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
    time: a second pass would re-detect the trigger and could disagree with the one every
    other panel is drawn against.

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
        tmin = float(ref.first_time) - float(raw.first_time)
        tmax = tmin + float(ref.times[-1])
        if tmin < -1e-6 or tmax > float(raw.times[-1]) + 1e-6:
            logger.warning("%s: the aligned window (%.1f-%.1f s) is not inside this copy "
                           "(0-%.1f s); dropping it from the motion panel",
                           sid, tmin, tmax, float(raw.times[-1]))
            continue
        out[sid] = raw.copy().crop(tmin=max(tmin, 0.0), tmax=tmax)
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
    report. ``fnirs-hyper run`` does **not** use this. Its window goes through
    :func:`resolve_analysis_window` instead and is taken out of the wavelet transform, a cut
    stretch transformed alone having two edges and a cone of influence of its own.

    ``tend`` past the end of the data is clipped rather than refused: recordings differ in
    length and an over-long window is a request for "to the end", not a mistake.
    """
    window = resolve_analysis_window(raws, tstart, tend)
    if window is None:
        return raws
    t0, t1 = window
    return {sid: raw.copy().crop(tmin=t0, tmax=t1) for sid, raw in raws.items()}


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
    ``--wtc-by-condition`` takes and for the reason recorded in
    :func:`~fnirs_pipe.pipeline.synchrony.window_result`: a cut stretch transformed on its
    own has two edges of its own, and its cone of influence eats a share of the band that
    grows as the window shortens, so the coherence over a 300 s cut comes out higher than
    the same 300 s read out of the whole record. This used to cut, so it was the one entry in
    this pipeline still paying that cost. Numbers from before that change are not
    reproducible with it.

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
        data = r.get_data()
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True)
        r._data[:] = (data - mu) / np.where(sd < 1e-12, 1.0, sd)
        result[sid] = r
    return result
