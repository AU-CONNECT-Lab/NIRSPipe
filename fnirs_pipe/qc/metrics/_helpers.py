"""Pieces more than one metric family needs, and nothing that measures anything itself.

``_safe_metrics`` declares an output schema, ``_mask_to_segments`` turns a per-sample mask
into time spans, and ``long_short_channels`` is the package's one definition of what a long
channel is. All three are shared, so they sit below every family and import none of them.
"""

import functools

import mne
import numpy as np

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.metrics.helpers")


# the bands do not meet, deliberately, so long + short is not every channel on a montage
SHORT_MAX_DIST = 0.01   # m, <= this is a short channel
LONG_MIN_DIST  = 0.015  # m, >= this and <= LONG_MAX_DIST is a long channel
LONG_MAX_DIST  = 0.045  # m


def long_short_channels(raw: mne.io.Raw) -> "tuple[list[str], list[str]]":
    """Split channel names by source-detector separation, returning ``(long, short)``.

    One definition for the whole package, so the report, the prep-raw figures and the SQM
    record agree on which channels are which. Because the two ranges do not meet (see
    SHORT_MAX_DIST / LONG_MIN_DIST), the two lists need not cover every channel::

        distances 8, 12, 30, 50 mm  ->  long ["30mm"], short ["8mm"]

    A montage with no registered optode positions reports every distance as zero, which
    would make every channel short; that case is logged and yields no split at all.

    Bad channels stay in both lists. Separation is the only thing being asked about, and
    pick_types drops bads by default, which would leave every metric computed from these
    lists averaging over channels that were selected for being good.
    """
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    names = [raw.ch_names[i] for i in picks]
    long_names  = [ch for ch, d in zip(names, dists) if LONG_MIN_DIST <= d <= LONG_MAX_DIST]
    short_names = [ch for ch, d in zip(names, dists) if 0 < d <= SHORT_MAX_DIST]
    if not long_names and not short_names:
        logger.warning("no channel falls in either separation range; optode positions "
                       "are probably missing")
    return long_names, short_names


def _mean_or_none(values) -> "float | None":
    """Mean of a collection of values, or None if it is empty."""
    vals = list(values)
    return float(np.mean(vals)) if vals else None


def _safe_metrics(label: str, keys):
    """Wrap a metric function so it always returns a dict keyed by ``keys``.

    ``keys`` declares the output schema once: the wrapped function starts from that
    schema (all None) and overlays whatever it computes, so a failure logs and leaves
    every key None instead of each function hand-writing an all-None fallback.

    A key ending in ``*`` declares a family whose members are only known at runtime, one
    per wavelength for instance. Those cannot be pre-filled — on failure there is no way
    to know which members would have existed — so the wildcard documents them and keeps
    them from reading as an undeclared key. Anything else the function returns that the
    schema does not mention is logged, because it is present on success and absent on
    failure, which is exactly what this decorator exists to prevent.
    """
    fixed = tuple(k for k in keys if not k.endswith("*"))
    families = tuple(k[:-1] for k in keys if k.endswith("*"))

    def deco(fn):
        @functools.wraps(fn)
        def wrap(*args, **kwargs):
            base = dict.fromkeys(fixed)
            try:
                computed = fn(*args, **kwargs) or {}
            except Exception as exc:
                logger.warning("%s failed: %s", label, exc)
                return base
            undeclared = [k for k in computed
                          if k not in base and not k.startswith(families)]
            if undeclared:
                logger.warning("%s returned undeclared keys %s; they disappear when it "
                               "fails, so declare them (a '*' suffix marks a family)",
                               label, undeclared)
            base.update(computed)
            return base
        return wrap
    return deco


def _mask_to_segments(flagged: np.ndarray, times: np.ndarray) -> "list[tuple[float, float]]":
    """Collapse a per-sample bool mask into (onset, duration) time spans.

    Given which timepoints are flagged (motion / corrected / spike), return each
    contiguous run of True as a time interval, used to draw shaded bands on figures.
    A run of n samples is n sample periods long, so a single flagged sample is a span
    one period wide rather than a span of zero width that no figure can draw.

    Example: mask [F,T,T,F,T] at 1 Hz -> [(1.0, 2.0), (4.0, 1.0)].
    """
    if not flagged.any():
        return []
    # diff marks the edges: a 0->1 step opens a run, a 1->0 step closes it
    edges = np.diff(flagged.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)   # +1 undoes the one-sample offset diff introduces
    ends = list(np.where(edges == -1)[0] + 1)
    # a run touching the very start / end has no edge to detect, so add that boundary by hand
    if flagged[0]:
        starts.insert(0, 0)
    if flagged[-1]:
        ends.append(len(flagged))    # one past the last sample; closed below by extrapolation
    dt = float(times[-1] - times[-2]) if len(times) > 1 else 0.0
    def _end_time(e: int) -> float:
        return float(times[e]) if e < len(times) else float(times[-1]) + dt
    return [(float(times[s]), max(_end_time(e) - float(times[s]), 0.0)) for s, e in zip(starts, ends)]
