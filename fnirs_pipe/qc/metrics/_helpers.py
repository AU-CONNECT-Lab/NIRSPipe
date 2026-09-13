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


# ---- Separation bands ----
# The gap between the two is deliberate, so long + short is not every channel on a montage.
SHORT_MAX_DIST = 0.01   # m, <= this is a short channel
LONG_MIN_DIST  = 0.015  # m, >= this is a long channel
LONG_MAX_DIST  = None   # m, or None for no upper bound


Bands = "tuple[float, float, float | None]"

# Montage + bands pairs already named in the orphan warning below. `long_short_channels` is a
# pure function of both and is called once per figure, so a dyad run would otherwise print
# the same list a dozen times; the set is for log noise only and nothing reads it.
_ORPHANS_WARNED: "set[tuple]" = set()


def separation_bands(config: object | None = None) -> Bands:
    """``(short_max, long_min, long_max)`` in metres; ``long_max`` None means no upper bound.

    separation_bands(config_with_long_max_dist_0_045) -> (0.01, 0.015, 0.045)

    An accessor rather than three imports, so a caller cannot pick up two of the three and
    silently invent a fourth band. A config that does not carry a field, or carries None for
    it, keeps the default; passing None gives the defaults, which is what a caller with no
    run config has. ``long_max`` is the exception on both counts, since None is a value there
    rather than an absence, so :func:`bands_from_record` is what reads a stored one back.
    """
    values = []
    for field, default in (("short_max_dist", SHORT_MAX_DIST),
                           ("long_min_dist", LONG_MIN_DIST),
                           ("long_max_dist", LONG_MAX_DIST)):
        got = getattr(config, field, None) if config is not None else None
        values.append(default if got is None else float(got))
    return tuple(values)


def validate_bands(sep_bands: Bands) -> Bands:
    """Refuse a set of bands that does not describe two separated ranges.

    validate_bands((0.01, 0.005, None)) -> ValueError

    The three are independent flags, so nothing stops a caller moving one and leaving the
    others: overlapping bands would put a channel in both lists at once, and an upper bound
    under the lower one would empty the long list without saying why.
    """
    short_max, long_min, long_max = sep_bands
    if not short_max > 0 or not long_min > 0:
        raise ValueError(f"separations must be positive, got short_max={short_max}, "
                         f"long_min={long_min}")
    if short_max > long_min:
        raise ValueError(f"the short band ends at {short_max * 1e3:.1f} mm, past where the "
                         f"long band starts ({long_min * 1e3:.1f} mm); they would overlap")
    if long_max is not None and long_max <= long_min:
        raise ValueError(f"the long band's upper bound ({long_max * 1e3:.1f} mm) is not above "
                         f"its lower one ({long_min * 1e3:.1f} mm), so no channel is long")
    return sep_bands


def _long_band_phrase(long_min: float, long_max: "float | None") -> str:
    """``"long >= 15 mm"``, or ``"long 15-45 mm"`` when an upper bound is set."""
    if long_max is None:
        return f"long >= {long_min * 1e3:.0f} mm"
    return f"long {long_min * 1e3:.0f}-{long_max * 1e3:.0f} mm"


def bands_phrase(sep_bands: "Bands | None" = None) -> str:
    """The whole triple in one line, for a log message or an error.

    (0.01, 0.015, None) -> "short <= 10 mm, long >= 15 mm"
    """
    short_max, long_min, long_max = sep_bands if sep_bands is not None else separation_bands()
    return f"short <= {short_max * 1e3:.0f} mm, {_long_band_phrase(long_min, long_max)}"


def unclaimed_separations(sep_bands: "Bands | None" = None) -> str:
    """The separations in neither band: ``"10-15 mm"``, or ``"10-15 mm, or over 45 mm"``."""
    short_max, long_min, long_max = sep_bands if sep_bands is not None else separation_bands()
    gap = f"{short_max * 1e3:.0f}-{long_min * 1e3:.0f} mm"
    return gap if long_max is None else f"{gap}, or over {long_max * 1e3:.0f} mm"


# The record's `raw` section stamps the bands under these, in mm, beside n_long_channels.
# One tuple so the writer and `bands_from_record` cannot drift on a key name.
BANDS_RECORD_KEYS = ("sep_short_max_mm", "sep_long_min_mm", "sep_long_max_mm")


def bands_to_record(sep_bands: Bands) -> dict:
    """The bands as record scalars, in mm::

        (0.01, 0.015, None) -> {"sep_short_max_mm": 10.0, "sep_long_min_mm": 15.0,
                                "sep_long_max_mm": None}

    Stamped so a reader of an old record can tell which bands produced its split, which is
    otherwise unrecoverable once the defaults move.
    """
    return {k: (None if v is None else round(v * 1e3, 1))
            for k, v in zip(BANDS_RECORD_KEYS, sep_bands)}


def bands_from_record(scalars: dict) -> Bands:
    """The bands a stored record was split with, read back from its ``raw`` section.

    A key the record does not carry falls back to today's default, which is every record
    written before the bands were stamped. A key present and null is not missing: it is an
    upper bound deliberately switched off, so the two cases are told apart by presence.
    """
    defaults = separation_bands()
    out = []
    for key, default in zip(BANDS_RECORD_KEYS, defaults):
        if key not in scalars:
            out.append(default)
        else:
            value = scalars[key]
            out.append(None if value is None else float(value) / 1e3)
    return tuple(out)


def record_has_bands(scalars: dict) -> bool:
    """Whether a record stamps its own bands, rather than predating the stamp.

    {"sep_short_max_mm": 10.0, "sep_long_min_mm": 15.0, "sep_long_max_mm": None} -> True

    :func:`bands_from_record` falls back to today's defaults for a missing key, so a record
    written before the stamp existed reads back as whatever the defaults happen to be now.
    A caller comparing two records has to be able to tell that apart from two records that
    genuinely agree. All three keys or none: the writer always writes the three together,
    so a partial set is not a stamp.
    """
    return all(key in scalars for key in BANDS_RECORD_KEYS)


# Per-channel pass/fail lines. Apart from METRIC_DISPLAY, which holds the cutoffs for a
# channel *average*: a mean SNR of 20 over a montage is a different claim from one channel
# reading 20.
SCI_PASS = 0.8        # also the --sci-threshold default
PSP_PASS = 0.1
# Share of windows in which a channel must pass both lines above to be kept. SCI's 0.8 and
# PSP's 0.1 are the values their authors established; this one is not theirs. The tool that
# defines the rule leaves the percentage to the user and states no default, so 0.75 is
# borrowed from the one worked example that names a number.
GOOD_FRAC_PASS = 0.75
# CV is measured per channel name, and intensity names are per wavelength, so it is the
# per-wavelength CV and takes its threshold: 5% (Lloyd-Fox 2009). Alternatives are 7.5%
# (Hocke 2018) and 15% (Piper 2014), both on whole-channel CV. SNR is 1/CV by construction,
# so it is derived rather than written down, and the stored snr_pass_rate reads the same
# line: three numbers for one decision is how they drifted apart in the first place.
CV_PASS  = 0.05
SNR_PASS = 1.0 / CV_PASS


def long_short_channels(
    raw: mne.io.Raw, sep_bands: "Bands | None" = None,
) -> "tuple[list[str], list[str]]":
    """Split channel names by source-detector separation, returning ``(long, short)``.

    One definition for the whole package, so the report, the prep-raw figures and the SQM
    record agree on which channels are which. Because the two ranges do not meet (see
    SHORT_MAX_DIST / LONG_MIN_DIST), the two lists need not cover every channel::

        distances 8, 12, 30, 50 mm  ->  long ["30mm", "50mm"], short ["8mm"]

    ``sep_bands`` is this run's separations from :func:`separation_bands`; None takes the
    package defaults. Every caller in one run has to pass the same value, or the reports
    would describe a different montage than the regression used.

    A montage with no registered optode positions reports every distance as zero, which
    would make every channel short; that case is logged and yields no split at all.

    A channel in neither band is named in a warning rather than passed over. It takes part
    in no split section, no GVTD trace and no verdict, and the two bands not meeting is
    what makes that possible without anything on the page saying so.

    Bad channels stay in both lists. Separation is the only thing being asked about, and
    pick_types drops bads by default, which would leave every metric computed from these
    lists averaging over channels that were selected for being good.
    """
    short_max, long_min, long_max = sep_bands if sep_bands is not None else separation_bands()
    picks = mne.pick_types(raw.info, meg=False, fnirs=True, exclude=[])
    dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
    names = [raw.ch_names[i] for i in picks]
    long_names  = [ch for ch, d in zip(names, dists)
                   if long_min <= d and (long_max is None or d <= long_max)]
    short_names = [ch for ch, d in zip(names, dists) if 0 < d <= short_max]
    if not long_names and not short_names:
        logger.warning("no channel falls in either separation range; optode positions "
                       "are probably missing")
        return long_names, short_names
    claimed = set(long_names) | set(short_names)
    orphans = {ch: d for ch, d in zip(names, dists) if ch not in claimed}
    key = (tuple(sorted(orphans)), short_max, long_min, long_max)
    if orphans and key not in _ORPHANS_WARNED:
        _ORPHANS_WARNED.add(key)
        logger.warning(
            "%d channel(s) fall in neither separation band (short <= %.0f mm, %s) and so take "
            "part in no split section, no GVTD trace and no quality verdict: %s",
            len(orphans), short_max * 1e3, _long_band_phrase(long_min, long_max),
            ", ".join(f"{ch} {d * 1e3:.0f}mm" for ch, d in sorted(orphans.items())))
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


# ---- Which events a run can actually be epoched on ----
# The CNR metric and the subject report both ask this before building Epochs, so it sits
# below both rather than in qc.figures, where it was.

def epochable_events(raw, tmin: float, tmax: float):
    """Events that can actually be epoched over ``[tmin, tmax]``, as ``(events, event_id)``.

    ``BAD_`` annotations are censoring marks rather than stimuli and never enter the event
    set. What is left is kept only if its whole window lies inside the recording and clears
    the ``BAD_`` segments, which is the same test MNE applies before dropping an epoch.
    Applying it up front lets a caller skip the figure instead of building an empty Epochs
    and drawing nothing, which is what MNE reports as "All epochs were dropped"::

        markers at 0 s and 300 s in a 300 s run, tmin=-5, tmax=25  ->  (empty, {})

    That is the block design that marks only where a condition starts and ends: neither
    window fits, so the run has no trials to epoch even though it carries annotations.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    event_id = {k: v for k, v in event_id.items() if not str(k).upper().startswith("BAD")}
    empty = (np.empty((0, 3), dtype=int), {})
    if len(events) == 0 or not event_id:
        return empty
    events = events[np.isin(events[:, 2], list(event_id.values()))]

    sfreq = raw.info["sfreq"]
    first = int(round(tmin * sfreq))
    n_win = int(round(tmax * sfreq)) + 1 - first
    start = events[:, 0] + first - raw.first_samp
    keep = (start >= 0) & (start + n_win <= len(raw.times))

    for ann in raw.annotations:
        if not str(ann["description"]).upper().startswith("BAD"):
            continue
        onset = float(ann["onset"]) - raw.first_time
        keep &= ~((onset < (start + n_win) / sfreq) & (onset + float(ann["duration"]) > start / sfreq))

    events = events[keep]
    if len(events) == 0:
        return empty
    codes = set(events[:, 2].tolist())
    event_id = {k: v for k, v in event_id.items() if v in codes}
    return (events, event_id) if event_id else empty
