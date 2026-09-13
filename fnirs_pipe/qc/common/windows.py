"""Condition windows and crop provenance, read off a recording's annotations.

Nothing here is hyperscanning. These three lived in ``hyper_report`` and the single-subject
reports imported them from there, which is how the ``first_time`` correction below came to
be applied on the group path and not on the dyad raw one: a fix follows the module it was
written in, not the meaning it belongs to. They sit in their own module so both paths reach
them without either importing the other's report builder.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne

from fnirs_pipe.qc.common.figure_io import extract_markers
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.windows")


def crop_provenance(raw: "mne.io.Raw") -> "dict | None":
    """What a cropped input's sidecar says about the cut, or None if it is a whole recording.

    ::

      a file cut to [3555, 3927] with a 47 s margin
        -> {"window": [3555.0, 3927.0], "analysis": [3602.4, 3902.5], "margin_s": 47.1}

    ``fnirs-prep crop`` records the span it wrote and, with ``--margin``, the narrower span
    the cut was made for. Nothing downstream had read either, so a segment and a whole
    recording were treated identically and the tool said nothing about it. That is the one
    way into edge-inflated coherence that a user cannot see, since the numbers look ordinary.

    The sidecar is read off disk rather than the lineage stamp, which carries only the filter
    keys across a SNIRF round trip.
    """
    path = path_from(raw)
    if not path:
        return None
    try:
        side = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    params = (side.get("parameters") or {})
    windows = params.get("crop_windows_s")
    if not windows:
        return None
    analysis = params.get("crop_analysis_windows_s")
    return {"window": windows[0] if len(windows) == 1 else windows,
            "analysis": (analysis[0] if analysis and len(analysis) == 1 else analysis),
            "margin_s": float(params.get("crop_margin_s") or 0.0),
            "n_windows": len(windows)}


def markers_on_data_axis(raw: "mne.io.Raw") -> list[dict]:
    """Non-BAD annotations with their onsets moved onto the data axis, which starts at zero.

    ::

      a raw cropped from 22.4 s, annotation "talk" at onset 3602.4  ->  onset 3580.0

    Annotations of a cropped recording still sit on the original recording's axis, with the
    offset held in ``first_time``, while the data axis and everything computed from it start
    at zero. :func:`~fnirs_pipe.pipeline.hyperscanning.align_recordings` crops every member
    from its first shared trigger, so raw annotation onsets are late by that trigger's onset
    against any figure or window drawn on the aligned clock.

    Every consumer of the aligned markers goes through this. The condition windows did the
    subtraction and the figures' block boundaries did not, which drew every boundary line on
    every coherence map late by that same offset. It is invisible on a tree whose input
    files each held one condition cropped to its own start, where the offset is zero.
    """
    origin = float(raw.first_time)
    markers = extract_markers(raw)
    if not origin:
        return markers
    return [{**m, "onset": float(m["onset"]) - origin} for m in markers]


def condition_windows(
    raw: "mne.io.Raw",
    min_duration: float,
) -> "list[tuple[str, float, float]]":
    """[(label, tstart, tend)] on the aligned clock, one window per task annotation.

    e.g. annotations "Video" at 60 s for 240 s and "Talk" at 300 s for 240 s give
    ``[("Video", 60.0, 300.0), ("Talk", 300.0, 540.0)]``.

    An annotation's own ``duration`` is the window when it has one. Many acquisition systems
    write triggers with a duration of zero, so a window with none runs from its onset to the
    next annotation's, and the last to the end of the recording. Which rule each window came
    from is logged, and the tail of a zero-duration run is as long as the recording happens
    to be rather than as long as the block was.

    A description that occurs more than once gets one window per occurrence, numbered
    ``desc#1``, ``desc#2``. They are not spliced into a single record: a wavelet transform
    reads a join between two non-adjacent segments as a step, which lands in the result as
    broadband coherence at the join.

    ``min_duration`` drops windows shorter than that, in seconds. One cycle of the lowest
    frequency asked for is the sensible floor, and it is why an event-related design with
    two-second trials yields nothing here: a window holding less than one cycle has no
    average of that frequency to report, however the coherence was computed. Pass 0.0 where
    the windows only split panels and no frequency has to fit inside one.
    """
    markers = sorted(markers_on_data_axis(raw), key=lambda m: m["onset"])
    if not markers:
        return []

    end = float(raw.times[-1])
    counts: dict[str, int] = {}
    for m in markers:
        counts[m["description"]] = counts.get(m["description"], 0) + 1
    seen: dict[str, int] = {}

    windows: list[tuple[str, float, float]] = []
    n_from_duration = 0
    for i, m in enumerate(markers):
        onset = float(m["onset"])
        if float(m["duration"]) > 0:
            stop = onset + float(m["duration"])
            n_from_duration += 1
        else:
            stop = float(markers[i + 1]["onset"]) if i + 1 < len(markers) else end
        stop = min(stop, end)

        desc = m["description"]
        if counts[desc] > 1:
            seen[desc] = seen.get(desc, 0) + 1
            label = f"{desc}#{seen[desc]}"
        else:
            label = desc

        if stop - onset < min_duration:
            logger.info("condition %s spans %.1fs, under the %.1fs floor: skipped",
                        label, stop - onset, min_duration)
            continue
        windows.append((label, onset, stop))

    logger.info("condition windows: %d kept, %d took their own duration and %d ran to the "
                "next trigger", len(windows), n_from_duration, len(markers) - n_from_duration)
    return windows
