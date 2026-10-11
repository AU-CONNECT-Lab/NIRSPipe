"""``BAD_`` spans on a recording: the time every estimate leaves out, and why.

Two kinds, told apart by description alone: ``BAD_unselected`` is real data outside the
stretches the user named, and every other ``BAD_`` description is corrupted time.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np

from nirspipe.utils.logging import get_logger

logger = get_logger("utils.spans")

UNSELECTED = "BAD_unselected"
MANUAL = "BAD_manual"
INPUT_SOURCE = "input file"
# the flag behind each span the run itself adds; a span already on the input is the input's
_ADDED_BY = {UNSELECTED: "--keep-spans", MANUAL: "--bad-spans", "BAD_gvtd": "--gvtd-censor"}


def is_bad_span(description: Any) -> bool:
    """Whether an annotation is a ``BAD_`` span, by mne's own case-insensitive prefix rule."""
    return str(description).lower().startswith("bad")


def span_kind(description: Any) -> str:
    return "unselected" if str(description).lower() == UNSELECTED.lower() else "corrupted"


def bad_spans(raw: mne.io.BaseRaw) -> list[tuple[float, float, str]]:
    """Every ``BAD_`` annotation as ``(start, stop, description)`` on the data axis.

    ::

      "BAD_gvtd" at 12 s for 3 s, "task" at 20 s  ->  [(12.0, 15.0, "BAD_gvtd")]

    Clipped to the recording; a zero-length span covers no time and is left out.
    """
    origin = float(raw.first_time)
    end = raw.n_times / float(raw.info["sfreq"])
    out = []
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration,
                                     raw.annotations.description):
        if not is_bad_span(desc):
            continue
        start = max(0.0, float(onset) - origin)
        stop = min(float(onset) - origin + float(duration), end)
        if stop > start:
            out.append((start, stop, str(desc)))
    return out


def frames_under(raw: mne.io.BaseRaw, spans) -> np.ndarray:
    """Per frame, whether it lies under any ``(start, stop, ...)`` span on the data axis.

    ::

      10 Hz, a span 2.04-3.04 s  ->  True at samples 20 to 29

    Each edge is rounded to the nearest sample, as mne's ``reject_by_annotation`` rounds it,
    so these frames are the ones epochs and spectra leave out.
    """
    sfreq = float(raw.info["sfreq"])
    under = np.zeros(raw.n_times, dtype=bool)
    for start, stop, *_ in spans:
        under[max(int(np.round(start * sfreq)), 0):int(np.round(stop * sfreq))] = True
    return under


def merged(intervals) -> list[tuple[float, float]]:
    """``[(start, stop)]`` sorted with overlaps and touching ends joined.

    ::

      [(5, 9), (0, 2), (8, 12)]  ->  [(0, 2), (5, 12)]
    """
    out: list[list[float]] = []
    for start, stop in sorted((float(a), float(b)) for a, b in intervals):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], stop)
        else:
            out.append([start, stop])
    return [(a, b) for a, b in out]


def add_bad_spans(raw: mne.io.BaseRaw, spans, description: str) -> None:
    """Attach ``(onset, duration)`` spans on the data axis as annotations, in place."""
    if not spans:
        return
    origin = float(raw.first_time)
    # orig_time has to be the existing annotations': mne refuses to add two that disagree
    raw.set_annotations(raw.annotations + mne.Annotations(
        [origin + float(o) for o, _ in spans], [float(d) for _, d in spans],
        [description] * len(spans), orig_time=raw.annotations.orig_time))


def mark_unselected(raw: mne.io.BaseRaw, keep) -> mne.io.BaseRaw:
    """Mark everything outside the ``(onset, duration)`` stretches in ``keep`` as ``BAD_unselected``.

    ::

      a 600 s recording, keep [(100, 200), (250, 100)]
      ->  BAD_unselected at (0, 100) and (350, 250)

    No stretches means the recording is used whole and nothing is marked. A stretch ending
    past the recording is clipped with a log line; one starting outside it is refused.
    """
    if not keep:
        return raw
    end = raw.n_times / float(raw.info["sfreq"])
    stretches = _on_recording(keep, end, "--keep-spans")
    gaps, cursor = [], 0.0
    for start, stop in stretches:
        if start > cursor:
            gaps.append((cursor, start - cursor))
        cursor = stop
    if cursor < end:
        gaps.append((cursor, end - cursor))
    kept = sum(b - a for a, b in stretches)
    logger.info("--keep-spans: %.0f of %.0f s kept, %d stretch(es) marked %s",
                kept, end, len(gaps), UNSELECTED)
    add_bad_spans(raw, gaps, UNSELECTED)
    return raw


def mark_manual(raw: mne.io.BaseRaw, spans) -> mne.io.BaseRaw:
    """Mark the ``(onset, duration)`` stretches in ``spans`` as ``BAD_manual``: corrupted time the user named.

    ::

      a 600 s recording, spans [(100, 20), (110, 30)]  ->  BAD_manual at (100, 40)

    Ends and refusals as :func:`mark_unselected`'s; overlapping stretches become one span.
    """
    if not spans:
        return raw
    stretches = _on_recording(spans, raw.n_times / float(raw.info["sfreq"]), "--bad-spans")
    logger.info("--bad-spans: %.0f s in %d stretch(es) marked %s",
                sum(b - a for a, b in stretches), len(stretches), MANUAL)
    add_bad_spans(raw, [(a, b - a) for a, b in stretches], MANUAL)
    return raw


def _on_recording(spans, end: float, flag: str) -> list[tuple[float, float]]:
    """``(onset, duration)`` rows as merged ``(start, stop)``, each end clipped to ``end``."""
    stretches = []
    for onset, duration in spans:
        onset, stop = float(onset), float(onset) + float(duration)
        if not 0.0 <= onset < end:
            raise ValueError(f"{flag} onset {onset:g} s lies outside the recording "
                             f"(0 to {end:g} s)")
        if stop > end:
            logger.info("%s stretch %g-%g s clipped to the recording's end at %g s",
                        flag, onset, stop, end)
        stretches.append((onset, min(stop, end)))
    return merged(stretches)


def excluded_spans(raw: mne.io.BaseRaw, input_spans=(), known=()) -> list[dict[str, Any]]:
    """The sidecar's ``excluded_spans``: each ``BAD_`` span with its kind and where it came from.

    ::

      BAD_unselected (0, 100), BAD_gvtd (412.5, 3.1)
      ->  [{"onset": 0.0, "duration": 100.0, "description": "BAD_unselected",
            "kind": "unselected", "source": "--keep-spans"}, {... "source": "--gvtd-censor"}]

    ``input_spans`` is :func:`bad_spans` of the recording as read, so a span the input
    already carried is credited to the input file whatever its description. ``known`` is
    an upstream sidecar's ``excluded_spans``, whose source a matching span keeps.
    """
    given = {(round(a, 6), round(b, 6), d) for a, b, d in input_spans}
    recorded = {(round(k["onset"], 6), round(k["onset"] + k["duration"], 6), k["description"]):
                k["source"] for k in known}
    out = []
    for start, stop, desc in bad_spans(raw):
        key = (round(start, 6), round(stop, 6), desc)
        source = (INPUT_SOURCE if key in given
                  else recorded.get(key) or _ADDED_BY.get(desc, INPUT_SOURCE))
        out.append({"onset": start, "duration": stop - start, "description": desc,
                    "kind": span_kind(desc), "source": source})
    return out


def excluded_time(raw: mne.io.BaseRaw) -> dict[str, Any]:
    """The quality record's ``excluded`` section: the share of the run per kind, and what is kept.

    ::

      600 s, BAD_unselected over 300 s, BAD_gvtd over 30 s of which 10 s inside it
      ->  {"total_s": 600.0, "kept_s": 280.0,
           "unselected_frac": 0.5, "corrupted_frac": 0.05,
           "n_spans": 2}

    A stretch can be unselected and corrupted at once, so the two shares can add up to more
    than the excluded share; the kept seconds count each instant once.
    """
    total = raw.n_times / float(raw.info["sfreq"])
    spans = bad_spans(raw)

    def covered(kind=None) -> float:
        return sum(b - a for a, b in merged(
            (s, e) for s, e, d in spans if kind is None or span_kind(d) == kind))

    return {
        "total_s": total,
        "kept_s": total - covered(),
        "unselected_frac": covered("unselected") / total,
        "corrupted_frac": covered("corrupted") / total,
        "n_spans": len(spans),
    }
