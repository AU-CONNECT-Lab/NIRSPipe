"""Condition windows and crop provenance, read off a recording's annotations.

Shared by both report paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import mne

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.qc.common.figure_io import _pair_fname, extract_markers
from fnirs_pipe.utils.lineage import path_from
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.windows")


def crop_provenance(raw: "mne.io.Raw") -> "dict | None":
    """What a cropped input's sidecar says about the cut, or None if it is a whole recording.

    ::

      a file cut to [3555, 3927] with a 47 s margin
        -> {"window": [3555.0, 3927.0], "analysis": [3602.4, 3902.5], "margin_s": 47.1}

    ``fnirs-prep crop`` records the span it wrote and, with ``--margin``, the narrower span
    the cut was made for.

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

      a raw cropped from 22.4 s, annotation "task" at onset 3602.4  ->  onset 3580.0

    Annotations of a cropped recording still sit on the original recording's axis, with the
    offset held in ``first_time``, while the data axis and everything computed from it start
    at zero. :func:`~fnirs_pipe.pipeline.hyper.align_recordings` crops every member
    from its first shared trigger, so raw annotation onsets are late by that trigger's onset
    against any figure or window drawn on the aligned clock.

    Every consumer of the aligned markers goes through this.
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

    e.g. annotations "Rest" at 60 s for 240 s and "Task" at 300 s for 240 s give
    ``[("Rest", 60.0, 300.0), ("Task", 300.0, 540.0)]``.

    An annotation's own ``duration`` is the window when it has one. Many acquisition systems
    write triggers with a duration of zero, so a window with none runs from its onset to the
    next annotation's, and the last to the end of the recording. Which rule each window came
    from is logged, and the tail of a zero-duration run is as long as the recording happens
    to be rather than as long as the block was.

    A description that occurs more than once gets one window per occurrence, numbered
    ``desc#1``, ``desc#2``. They are not spliced into a single record: a wavelet transform
    reads a join between two non-adjacent segments as a step, which lands in the result as
    broadband coherence at the join.

    ``min_duration`` drops windows shorter than that, in seconds. Pass 0.0 where the windows
    only split panels and no frequency has to fit inside one.
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


def refuse_colliding_labels(labels, reserved: "tuple[str, ...]" = ()) -> None:
    """Raise when two condition labels would name one file, or one takes a reserved name.

    ::

      ["game 1", "game1"]            -> StageError: both are cond-game1
      ["all"], reserved=("all",)     -> StageError: cond-all is the table of every condition

    A label reaches a file name through a slug of its letters and digits, so two labels that
    differ only in the rest would write the same file, the later one replacing the earlier.
    """
    by_slug: dict[str, list[str]] = {}
    for label in labels:
        by_slug.setdefault(_pair_fname(str(label)), []).append(str(label))
    clashes = [names for names in by_slug.values() if len(names) > 1]
    if clashes:
        shown = "; ".join(" and ".join(repr(n) for n in names) for names in clashes)
        raise StageError(
            f"conditions {shown} would be written to one file, since a file name keeps only "
            "the letters and digits of a condition. Rename the annotations so that they "
            "differ in a letter or digit.")
    taken = sorted(set(by_slug) & set(reserved))
    if taken:
        raise StageError(
            f"a condition named {', '.join(repr(t) for t in taken)} takes a name the output "
            f"reserves: cond-{taken[0]} is the table holding every condition. Rename the "
            "annotation.")


def split_windows(
    windows: "list[tuple[str, float, float]]",
    length_s: float,
) -> "tuple[list[tuple[str, float, float]], dict[str, tuple[str, float]]]":
    """Cut each window into non-overlapping windows of one length, and say where each came from.

    ::

      [("rest", 0.0, 100.0), ("task", 200.0, 500.0)], length 100.0
        -> [("rest-w1", 0.0, 100.0), ("task-w1", 200.0, 300.0),
            ("task-w2", 300.0, 400.0), ("task-w3", 400.0, 500.0)],
           {"rest-w1": ("rest", 0.0), "task-w1": ("task", 0.0),
            "task-w2": ("task", 100.0), "task-w3": ("task", 200.0)}

    The remainder is dropped rather than kept short, since a window of another length would
    undo the split. A window shorter than ``length_s`` therefore yields nothing and is logged.

    The second return value is what a null needs. A draw cuts the stand-in at its own marker
    for the condition, not at the real dyad's clock, so for a window it needs the condition
    the window belongs to and how far into it the window starts. Returned explicitly rather
    than parsed back out of the label, which ``condition_windows`` may already have numbered
    ``desc#2``.
    """
    if length_s <= 0:
        raise ValueError(f"length_s must be positive, got {length_s}")

    out: list[tuple[str, float, float]] = []
    sources: dict[str, tuple[str, float]] = {}
    for label, tstart, tstop in windows:
        span = float(tstop) - float(tstart)
        n = int(span // length_s)
        if n < 1:
            logger.info("condition %s spans %.1fs, under the %.1fs window: no window fits",
                        label, span, length_s)
            continue
        for i in range(n):
            offset = i * length_s
            sub = f"{label}-w{i + 1}"
            out.append((sub, float(tstart) + offset, float(tstart) + offset + length_s))
            sources[sub] = (label, offset)
        left = span - n * length_s
        if left > 1.0:
            logger.info("condition %s: %d window(s) of %.1fs, %.1fs left over and dropped",
                        label, n, length_s, left)
    logger.info("equal-length windows: %d of %.1fs from %d condition(s)",
                len(out), length_s, len({s[0] for s in sources.values()}))
    return out, sources
