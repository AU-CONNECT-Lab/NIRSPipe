"""Per-run SQM record: one JSON per BIDS run, assembled after the pipeline has finished.

The pipeline writes each stage to disk; this reads those files back and computes every
metric once, at one named stage, through one writer. It works on any tree a past run left
behind.

Sections name the input a metric family was measured on. The metric names inside a
section are the names the metric functions return:

    raw        every channel of the original recording, the archival view
    raw_long   the same recording, long channels only
    raw_short  the same recording, short channels only: are the regressors trustworthy
    motion     what the motion correction repaired, from the OD either side of it
    preproc    Beer-Lambert output, before any filtering
    filtered   the same signal after the bandpass
    resampled  the same signal after the resample, when one ran
    errts      the confound-regression residual, the last denoising step there is

Every family is written three times, over every channel, the long ones and the short ones:
``raw`` / ``raw_long`` / ``raw_short``, ``preproc`` / ``preproc_long`` / ``preproc_short``,
and so on for each stage. One file seen through three channel sets, so the only thing that
differs within a trio is source-detector separation, and a ``_long`` number is only ever
compared against another ``_long`` one.

**Split a new metric three ways unless it cannot be.** A short channel sees scalp rather
than cortex, so an average over both sets describes neither. Two kinds of metric are the
exception:

    a sum over channels  ``spike_count``, which tracks how many channels a set has and
                     nothing else. It is the only metric of the motion families that stays
                     on ``all`` and ``long`` alone. GVTD, the spike frame counts and the
                     correction footprint all split in full: each set is its own
                     measurement rather than a regrouping of one, "at least a tenth of
                     *these* channels" being a different question per set.
    not channel-based  ``pct_data_retained`` is a share of the recording's duration. It is
                     the same number for every channel set, so it stays on the whole-file
                     section alone (see ``_WHOLE_FILE_KEYS``); repeating it under ``_long``
                     would name a quantity that does not exist.

Anything else, split it.

Bad channels: the Beer-Lambert conversion is the dividing line, never the section.

    raw / raw_long / raw_short / motion   intensity and OD, include them
    preproc / filtered / resampled / errts  haemoglobin, exclude them

A rejected channel is still part of what the machine recorded, so everything measured
before Beer-Lambert describes the recording as it arrived; after it the channel is out of
the analysis, and those metrics go through ``mne.pick_types``, which drops bads by
default. Excluding them earlier would make ``sci_mean`` circular, taken over channels
selected for good SCI. ``channel_retention_rate`` is what says how many were dropped.

Per-channel values live under ``per_channel`` so the sections stay scalar. Those are
always complete, every channel, whichever section they sit under.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np

from fnirs_pipe.io.auxiliary import (aux_table_units, find_aux_table, imu_traces,
                                     read_aux_table, table_channels)
from fnirs_pipe.io.derivatives import bids_uris, entity_of, read_json, resolve_bids_uri
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe import __version__
from fnirs_pipe.qc.common.channel_table import channel_rows, save_channel_csv
from fnirs_pipe.qc.subject.record_io import write_record
from fnirs_pipe.qc.subject.condition_views import (
    PSD_NFFT_CAP, condition_haemo_scalars, condition_scalars, condition_set_scalars,
    condition_slices_from_record, span_counts,
)

logger = get_logger("qc.sqm_record")

# the post-Beer-Lambert stages, in the order the pipeline writes them. Each is its own
# section named after the file it measured, which is the rule every other section follows.
_HAEMO_STAGES = ("filtered", "resampled", "errts")

# Which side of the bandpass a stage sits on, named beside the list it comes from so a new
# stage cannot land on the wrong side in silence. Crossing the line measures the filter.
PRE_BANDPASS_HAEMO_STAGE = "preproc"
POST_BANDPASS_HAEMO_STAGES = _HAEMO_STAGES

# Every haemoglobin stage is split by separation the same way the raw ones are, so a
# `_long` metric always has a `_long` counterpart to compare against, never an all-channel
# one. `_SPLIT_SUFFIXES` is the order they are written and read in.
_SPLIT_SUFFIXES = ("long", "short")
_HAEMO_SECTIONS = tuple(
    name for stage in ("preproc", *_HAEMO_STAGES)
    for name in (stage, *(f"{stage}_{s}" for s in _SPLIT_SUFFIXES))
)

SECTIONS = ("raw", "raw_long", "raw_short",
            "motion", "motion_long", "motion_short",
            "motion_post", "motion_post_long", "motion_post_short", "windowed",
            *_HAEMO_SECTIONS)

# Sections only some runs have. Kept out of SECTIONS, which means "every run writes this" and
# is asserted as such: censoring is opt-in and an IMU is on some devices only, so a record
# without either is correct, not incomplete.
# The group table still descends into these.
# The two records a run can leave behind, best first. `sqm` is what the pipeline writes,
# `sqmraw` what `fnirs-qc prep-raw` writes, measuring the original recording only. A run
# that saw both commands has both files. Both are sectioned; the shape is what is read,
# never the name.
SQM_DESCS = ("sqm", "sqmraw")

OPTIONAL_SECTIONS = ("censor", "imu")

# `pct_data_retained` measures the recording's duration, not its channels, so it is one
# number for every channel set. It stays on the whole-file section alone: repeating it
# under `_long` would name a long-channel quantity that does not exist.
_WHOLE_FILE_KEYS = ("pct_data_retained",)


def scan_runs(nirs_dir: Path) -> dict[str, dict[str, Path]]:
    """Map each BIDS run label to its derivatives: ``{label: {desc: path}}``.

    ``sub-01_task-tapping_desc-preproc_nirs.snirf`` becomes
    ``{"sub-01_task-tapping": {"preproc": <path>}}``, so a subject holding three tasks
    yields three runs rather than one that silently keeps the last.
    """
    runs: dict[str, dict[str, Path]] = {}
    for path in sorted(Path(nirs_dir).glob("*_desc-*_nirs.snirf")):
        stage = entity_of(path, "desc")
        if stage is None:
            continue
        label = path.name[: path.name.index("_desc-")]
        runs.setdefault(label, {})[stage] = path
    return runs


def entities_of(label: str) -> dict[str, str | None]:
    """Pull the BIDS entities back out of a run label, for the columns a database wants."""
    return {key: entity_of(label, key) for key in ("ses", "task", "run")}


# What record_path writes, for the readers that glob for it; one spelling for all of them.
# The suffix is `qc`, not `nirs`: a .json whose suffix is nirs is by BIDS definition the
# sidecar of a snirf, and there is no desc-sqm snirf for this to be the sidecar of.
RECORD_SUFFIXES = {desc: f"_desc-{desc}_qc.json" for desc in SQM_DESCS}
RECORD_SUFFIX = RECORD_SUFFIXES["sqm"]


def record_path(nirs_dir: Path, label: str) -> Path:
    return Path(nirs_dir) / (label + RECORD_SUFFIX)


def record_label(path: "Path | str") -> str:
    """The run label a record is named for: the inverse of :func:`record_path`.

    ``"sub-01_task-rest_desc-sqm_qc.json"`` -> ``"sub-01_task-rest"``
    """
    name = getattr(path, "name", path)
    return name[: name.index("_desc-")]


def _sidecar(path: Path) -> dict[str, Any]:
    return read_json(path.with_suffix(".json"))


def _bids_input(stages: dict[str, Path], bids_root: Path | None = None) -> Path | None:
    """The original recording: the OD file is the only derivative whose source is it.

    ``Sources`` holds a BIDS URI resolved through the tree's ``raw`` link, so trees moved
    together still find it. A raw dataset moved on its own leaves the name right, so fall
    back to finding it under ``bids_root``. Without that fallback the three ``raw*``
    sections vanish from every rebuilt record, with one warning to say why.
    """
    if "od" not in stages:
        return None
    sources = _sidecar(stages["od"]).get("Sources") or []
    if not sources:
        return None
    src = resolve_bids_uri(sources[0], stages["od"])
    if src.exists():
        return src
    if bids_root is not None:
        # BIDS names are unique within a dataset, so the first hit is the right one
        if (found := next(Path(bids_root).rglob(src.name), None)) is not None:
            logger.info("%s moved; using %s", src.name, found)
            return found
    logger.warning("the recorded input %s is gone%s", src,
                   "" if bids_root else " and no bids_root was given to search")
    return None


def _bands(stages: dict[str, Path]) -> dict[str, float] | None:
    """Band edges the run used, read back from any sidecar that recorded them."""
    keys = ("cardiac_l_freq", "cardiac_h_freq", "resp_l_freq", "resp_h_freq")
    for path in stages.values():
        params = _sidecar(path).get("parameters") or {}
        if all(params.get(k) is not None for k in keys):
            return {k: float(params[k]) for k in keys}
    return None


def _window_s(stages: dict[str, Path]) -> float | None:
    """QC window length the run used, read back from any sidecar that recorded it."""
    for path in stages.values():
        value = (_sidecar(path).get("parameters") or {}).get("qc_window_s")
        if value is not None:
            return float(value)
    return None


def _sci_scores(stages: dict[str, Path]) -> dict[str, float]:
    """Per-channel SCI from the sci sidecar, recomputed from the OD file if absent."""
    if "sci" in stages:
        scores = _sidecar(stages["sci"]).get("sci_scores")
        if scores:
            return {k: float(v) for k, v in scores.items()}
    # a tree with no stored scores still has the OD file to recompute them from
    source = stages.get("sci") or stages.get("od")
    if source is None:
        return {}
    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.qc.metrics import compute_sci_scores
    bands = _bands(stages) or {}
    if not bands:
        return {}
    scores, _ = compute_sci_scores(
        read_snirf(source), bands["cardiac_l_freq"], bands["cardiac_h_freq"])
    return scores


def _good_frac_scores(stages: dict[str, Path]) -> dict[str, float]:
    """Per-channel coupled-window share from the sci sidecar, or empty.

    Not recomputed when it is absent, unlike :func:`_sci_scores`: it depends on this run's
    ``--sci-threshold``, ``--psp-threshold``, ``--min-good-frac`` and ``--screen-scope``,
    and a guess would store a number no channel was judged by. A tree without the stored
    share has none, and the report leaves the row out.
    """
    if "sci" not in stages:
        return {}
    scores = _sidecar(stages["sci"]).get("good_frac_scores") or {}
    return {k: float(v) for k, v in scores.items()}


def _imu_of(stages: dict[str, Path]) -> "dict | None":
    """The run's IMU traces, from the aux table preprocessing wrote beside its stages."""
    table = find_aux_table(next(iter(stages.values()))) if stages else None
    if table is None:
        return None
    try:
        return imu_traces(*table_channels(read_aux_table(table)), aux_table_units(table)) or None
    except Exception:
        logger.warning("%s unreadable; no imu section", table, exc_info=True)
        return None


def _imu_slicer(imu: "dict | None"):
    """``imu_of(t0, t1)`` for one condition, or None for a recording without an IMU."""
    if not imu:
        return None
    from fnirs_pipe.qc.metrics import imu_scalars
    return lambda t0, t1: imu_scalars(imu, t0, t1)


def _split_scalars(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate the scalar metrics from the per-channel dicts and windowed lists."""
    scalars, nested = {}, {}
    for key, value in record.items():
        (nested if isinstance(value, (dict, list)) else scalars)[key] = value
    return scalars, nested


def _short_section(
    raw_intensity: mne.io.Raw,
    short_names: list[str],
    sci_scores: dict[str, float],
    bad_channels: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    good_frac_scores: "dict[str, float] | None" = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only the metrics that answer "are the short-channel regressors trustworthy".

    Coupling (SCI, PSP, the coupled-window share) and amplitude (SNR, CV) transfer to short
    channels, and so does the whole spike family: the mask is a per-channel MAD test, and the
    frame counts over it ask how many of *these* channels spiked at once, which the windowed
    half stores per set (``spike_spans_short_s``). ``spike_count`` is a sum over
    channels rather than a per-channel reading, so it counts this set's channels and not
    another's; it is printed beside the set's channel count for that reason. Drift does not
    transfer. GVTD does, in full, threshold included: it is an RMS
    across whatever channels it is given, so a short set is its own measurement rather than a
    subset of the long one, and its threshold is the mode of *its own trace over time*, which
    has as many samples as any other trace of the same recording.

    Each set's share is counted against its own set's threshold, so a short share and a long
    share are not two readings of one thing; the panel labels them.

    Every value here is a value and not a verdict. Nothing colours a short channel's row,
    because a short channel's coupling is high by construction; the long section is where
    the number is read against a line.
    """
    from fnirs_pipe.qc.metrics import (
        _intensity_metrics, _motion_metrics, _psp_metrics, _sci_metrics,
        _sci_win_metrics, _spike_metrics,
    )
    from fnirs_pipe.qc.metrics.coupling import _good_frac_metrics

    raw_short = raw_intensity.copy().pick(short_names)
    short_set = set(short_names)
    short_sci = {k: v for k, v in sci_scores.items() if k in short_set}
    short_bad = [c for c in bad_channels if c in short_set]
    short_frac = {k: v for k, v in (good_frac_scores or {}).items() if k in short_set}

    record: dict[str, Any] = {"n_channels": len(short_names), "n_bad": len(short_bad)}
    record.update(_sci_metrics(short_sci, short_bad))
    record.update(_sci_win_metrics(raw_short, cardiac_l_freq, cardiac_h_freq))
    record.update(_psp_metrics(raw_short, cardiac_l_freq, cardiac_h_freq))
    record.update(_good_frac_metrics(short_frac))
    intensity = _intensity_metrics(raw_short)
    record.update({k: v for k, v in intensity.items()
                   if k.startswith(("snr_", "cv_", "mean_amp_"))})
    record.update(_motion_metrics(raw_short))
    spike = _spike_metrics(raw_short)
    record.update({k: v for k, v in spike.items()
                   if k in ("spike_count", "spike_pct", "spike_pct_per_channel",
                            "spike_num_frames", "spike_pct_frames")})
    record["spike_pct_per_channel"] = record.get("spike_pct_per_channel") or {}
    return _split_scalars(record)


def _section_writer(sections: dict[str, Any], per_channel: dict[str, Any]):
    """A ``section(name, compute)`` that files one family and survives its own failure.

    A section that raises costs that section alone, never the rest of the record, which is
    what lets a partial recording still produce something readable.
    """
    def section(name: str, compute) -> None:
        try:
            scalars, nested = _split_scalars(compute())
        except Exception:
            logger.warning("%s: section failed", name, exc_info=True)
            return
        sections[name] = scalars
        per_channel[name] = nested
    return section


def raw_verdict_view(record: dict) -> dict:
    """The raw-family scalars a quality verdict is read off, as one flat dict.

    raw_verdict_view({"raw": {"sci_mean": 0.71, "duration_s": 600},
                      "raw_long": {"sci_mean": 0.86}})
    -> {"sci_mean": 0.86, "duration_s": 600}

    The long-channel split sits on top of the whole-file section rather than replacing it,
    because the split carries only what a channel set can be measured on and the run-level
    numbers (montage counts, duration) live underneath.

    One function because three readers need the same answer: the subject report, the dyad
    raw report and the dyad analysis. A flat legacy record has no sections and comes back
    unchanged.
    """
    view = {**(record.get("raw") or {})}
    view.update(record.get("raw_long") or {})
    return view


def fill_skipped_long_sections(record: dict) -> dict:
    """The record with every ``_long`` section an all-long montage skipped filled back in.

    fill_skipped_long_sections({"raw": {"cv_mean": 0.02, "n_long_channels": 40}})
    -> {"raw": {...}, "raw_long": {"cv_mean": 0.02, "n_long_channels": 40}}

    A ``_long`` section is not written when the long set is the whole file, because it would
    repeat the section above it. That is a storage rule and not a measurement one: on such a
    montage the whole-file section *is* the long-channel one. A reader comparing a cohort on
    ``raw_long_cv_mean`` would otherwise drop every all-long run rather than compare it, and
    drop it silently, since an absent section and a failed one look the same from outside.
    :func:`raw_verdict_view` is the same rule for one section at a time.

    Absence plus a positive long count is what identifies the case, and it is exact: a
    montage whose long channels are only part of it has the section written, and one with no
    long channels at all has nothing to fill from. The per-channel dicts are filled with it,
    so a filled record stays readable by everything that reads an unfilled one.
    """
    if not (record.get("raw") or {}).get("n_long_channels"):
        return record
    filled = dict(record)
    per_channel = dict(record.get("per_channel") or {})
    for name in SECTIONS:
        base = name.removesuffix("_long")
        if base == name:
            continue
        if name not in filled and isinstance(filled.get(base), dict):
            filled[name] = dict(filled[base])
        if name not in per_channel and isinstance(per_channel.get(base), dict):
            per_channel[name] = dict(per_channel[base])
    if per_channel:
        filled["per_channel"] = per_channel
    return filled


def raw_sections(
    raw_intensity: mne.io.Raw,
    sci_scores: dict[str, float],
    bad_channels: list[str],
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    sep_bands=None,
    good_frac_scores: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The three views of the original recording, as ``(sections, per_channel)``.

    One file measured over every channel, over the long ones, and over the short ones.
    Every writer of a run record goes through here, so a tree holds one record shape
    whichever command produced it and the group table needs no per-writer special case.

    ``raw_long`` is skipped when the long set is the whole montage, since it would repeat
    ``raw``; ``raw_short`` is skipped when there is no short channel. Both missing is
    therefore ambiguous on its own, which is what ``n_long_channels`` and
    ``n_short_channels`` on ``raw`` are for: two zeros means no registered optode
    positions, and any other pair means a montage of one kind.

    The separations that produced the split are stamped beside those counts, since the
    defaults can move and a reader of an old record cannot otherwise recover them.

    ``good_frac_scores`` is the coupled-window share the screening already counted, passed
    in rather than recounted: it is two windowed passes over the recording, and a caller
    that screened has it. None leaves the entry empty rather than paying for it again.
    """
    from fnirs_pipe.qc.metrics import compute_raw_sqm, long_short_channels
    from fnirs_pipe.qc.metrics._helpers import bands_to_record, separation_bands

    sections: dict[str, Any] = {}
    per_channel: dict[str, Any] = {}
    section = _section_writer(sections, per_channel)

    section("raw", lambda: compute_raw_sqm(
        raw_intensity, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq,
        good_frac_scores))

    sep_bands = sep_bands if sep_bands is not None else separation_bands()
    long_names, short_names = long_short_channels(raw_intensity, sep_bands)
    if "raw" in sections:
        sections["raw"]["n_long_channels"] = len(long_names)
        sections["raw"]["n_short_channels"] = len(short_names)
        sections["raw"].update(bands_to_record(sep_bands))
    if long_names and len(long_names) < len(raw_intensity.ch_names):
        def long_section():
            raw_long = raw_intensity.copy().pick(long_names)
            long_sci = {k: v for k, v in sci_scores.items() if k in set(long_names)}
            long_bad = [c for c in bad_channels if c in set(long_names)]
            long_frac = {k: v for k, v in (good_frac_scores or {}).items()
                         if k in set(long_names)}
            return compute_raw_sqm(
                raw_long, long_sci, long_bad, cardiac_l_freq, cardiac_h_freq, long_frac)
        section("raw_long", long_section)
    if short_names:
        # already returns the (scalars, nested) split, so it bypasses `section`
        try:
            sections["raw_short"], per_channel["raw_short"] = _short_section(
                raw_intensity, short_names, sci_scores, bad_channels,
                cardiac_l_freq, cardiac_h_freq, good_frac_scores)
        except Exception:
            logger.warning("raw_short: section failed", exc_info=True)
    return sections, per_channel


def haemo_sections(
    name: str,
    raw_haemo: mne.io.Raw,
    compute,
    sep_bands=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One haemoglobin file measured over every channel, the long ones, the short ones.

    The same three-way split as :func:`raw_sections`, and for the same reason: a short
    channel sees scalp, so a mean taken over both sets describes neither.

    ``compute`` maps a Raw to a flat metric dict, which is what lets the unfiltered stage
    (band power and drift included) and the filtered ones share this.

    A subset that turns out to be the whole file is skipped rather than written twice.
    """
    from fnirs_pipe.qc.metrics import long_short_channels

    sections: dict[str, Any] = {}
    per_channel: dict[str, Any] = {}
    section = _section_writer(sections, per_channel)

    section(name, lambda: compute(raw_haemo))

    long_names, short_names = long_short_channels(raw_haemo, sep_bands)
    for suffix, names in zip(_SPLIT_SUFFIXES, (long_names, short_names)):
        if not names or len(names) == len(raw_haemo.ch_names):
            continue
        def subset(names=names):
            picked = compute(raw_haemo.copy().pick(names))
            return {k: v for k, v in picked.items() if k not in _WHOLE_FILE_KEYS}
        section(f"{name}_{suffix}", subset)
    return sections, per_channel


def _motion_post_section(
    raw_od: mne.io.Raw,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    gvtd_thresh: "float | None" = None,
) -> dict[str, Any]:
    """Did the correction work, and did it cost anything.

    Two questions, not one. GVTD and the spike counts answer the first: both measure what
    the correction is there to remove, so a run that improved shows them fall against the
    same keys in ``raw``. SCI and PSP answer the second: they live in the cardiac band,
    above the frequencies motion correction touches, so they should come back unchanged. A
    drop means the correction ate physiology along with the artifact.

    ``gvtd_thresh`` is the matching ``raw*`` section's cutoff, so both sides are counted
    against one yardstick. See ``_motion_metrics``.

    Hand-picked rather than ``compute_raw_sqm``: this file is optical density, which would
    leave that function's intensity family empty.
    """
    from fnirs_pipe.qc.metrics import (
        _mean_or_none, _motion_metrics, _psp_metrics, _sci_win_metrics, _spike_metrics,
        compute_sci_scores,
    )

    record: dict[str, Any] = {}
    record.update(_motion_metrics(raw_od, thresh=gvtd_thresh))
    record.update(_spike_metrics(raw_od))
    # the mean over every channel, bads included, so it is comparable with `raw`
    scores, _ = compute_sci_scores(raw_od, cardiac_l_freq, cardiac_h_freq)
    record["sci_mean"] = _mean_or_none(scores.values())
    # per channel as well as the mean: the report pairs these against the pre-correction
    # scores channel by channel, and a mean cannot say which channel the correction cost
    record["sci_per_channel"] = {k: float(v) for k, v in scores.items()}
    record.update(_sci_win_metrics(raw_od, cardiac_l_freq, cardiac_h_freq))
    record.update(_psp_metrics(raw_od, cardiac_l_freq, cardiac_h_freq))
    return record


def motion_sections(
    before_od: "mne.io.Raw | None",
    after_od: "mne.io.Raw",
    section,
    windowed: dict[str, Any],
    raw_thresh,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    sep_bands=None,
) -> None:
    """The whole motion family, from the optical density either side of the correction.

    Two groups. ``motion*`` is the correction's footprint, how much of the recording it
    touched, and ``motion_post*`` is the corrected file measured again on the keys ``raw*``
    already carries, so a pair subtracts. Both split by separation, because the footprint's
    frame counts ask how many channels were corrected at once and that is a different
    measurement per set rather than the same one regrouped.

    ``raw_thresh(name)`` returns the ``raw*`` section's GVTD cutoff for the matching channel
    set, so before and after are counted against one yardstick. See
    :func:`_motion_post_section`.

    ``before_od`` None leaves the footprint half out and keeps the rest: measuring what the
    correction touched needs both sides, measuring the corrected file needs only the one.
    A tree carrying a corrected file but no pre-correction one still gets ``motion_post*``.

    Written in place through ``section``, the same writer the rest of the record uses, so a
    family that fails costs that family alone. One function because two callers assemble it:
    the pipeline's record reads the two files off disk, and ``fnirs-qc prep-raw`` corrects a
    copy in memory and never writes it.
    """
    from fnirs_pipe.qc.metrics import (
        long_short_channels, motion_corrected_segments, motion_correction_metrics,
    )

    if before_od is not None:
        section("motion", lambda: motion_correction_metrics(before_od, after_od))
        mc_long, mc_short = long_short_channels(before_od, sep_bands)
        if mc_long and len(mc_long) < len(before_od.ch_names):
            section("motion_long", lambda: motion_correction_metrics(
                before_od.copy().pick(mc_long), after_od.copy().pick(mc_long)))
        if mc_short:
            section("motion_short", lambda: motion_correction_metrics(
                before_od.copy().pick(mc_short), after_od.copy().pick(mc_short)))

        # `motion_long` counts the spans; this is where they are, for the figures that draw
        # them and for a condition counting the run's own boolean over its own stretch. The
        # long set under the plain key, the convention `spike_spans_s` follows: a condition's
        # share has to be the same measurement as the run's row above it, and the figures
        # draw this strip beside a spike row that is already the long set's.
        # One list per set, so a condition's three rows are three measurements the way the
        # run's are, rather than the long one under three headings.
        long_picks = (mc_long if mc_long and len(mc_long) < len(before_od.ch_names)
                      else None)
        try:
            for key, picks in (("motion_corrected_spans_s", long_picks),
                               ("motion_corrected_spans_short_s", mc_short or None),
                               ("motion_corrected_spans_all_s", None)):
                if key.endswith("_short_s") and picks is None:
                    continue
                windowed[key] = [
                    list(span) for span in motion_corrected_segments(
                        before_od if picks is None else before_od.copy().pick(picks),
                        after_od if picks is None else after_od.copy().pick(picks))]
        except Exception:
            logger.warning("windowed: correction spans failed", exc_info=True)

    section("motion_post", lambda: _motion_post_section(
        after_od, cardiac_l_freq, cardiac_h_freq, raw_thresh("raw")))
    post_long, post_short = long_short_channels(after_od, sep_bands)
    if post_long and len(post_long) < len(after_od.ch_names):
        section("motion_post_long", lambda: _motion_post_section(
            after_od.copy().pick(post_long), cardiac_l_freq, cardiac_h_freq,
            raw_thresh("raw_long")))
    if post_short:
        section("motion_post_short", lambda: _motion_post_section(
            after_od.copy().pick(post_short), cardiac_l_freq, cardiac_h_freq,
            raw_thresh("raw_short")))


def _cutoffs_from_sidecar(stages: dict[str, Path]) -> dict[str, float]:
    """The lines the run screened by, so a condition's verdict is on the same ones."""
    from fnirs_pipe.qc.metrics import resolve_cutoffs
    params = (_sidecar(stages["sci"]).get("parameters") or {}) if "sci" in stages else {}
    return resolve_cutoffs(sci=params.get("sci_threshold"),
                           psp=params.get("psp_threshold"),
                           good_frac=params.get("min_good_frac"))


def condition_sections(
    sections: dict[str, Any],
    raw_intensity: "mne.io.Raw",
    stages: dict[str, Path],
    *,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
    sep_bands=None,
    imu: "dict | None" = None,
) -> dict[str, Any]:
    """One entry per annotated condition, sliced from ``windowed`` and recomputed on a crop.

    Written whenever the recording carries conditions: ``--by-condition`` decides what a
    report shows, not what the record measures. Anything deriving a scale from the recording
    it is handed (the filtered GVTD trace, per-channel z-scores, a threshold off a
    distribution) stays whole-run and is absent here. ``{}`` when no annotation holds two
    screening windows.
    """
    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.qc.common.windows import condition_windows
    from fnirs_pipe.qc.metrics import long_short_channels
    from fnirs_pipe.qc.metrics.windowed import SCREEN_WINDOW_S

    windows = condition_windows(raw_intensity, min_duration=2 * SCREEN_WINDOW_S)
    if not windows:
        logger.info("no annotation holds two screening windows; no by_condition section")
        return {}

    def _read(desc: str):
        path = stages.get(desc)
        if path is None:
            return None
        try:
            return read_snirf(path)
        except Exception:
            logger.warning("%s unreadable; conditions get no %s metrics", path, desc,
                           exc_info=True)
            return None

    haemo, errts = _read("preproc"), _read("errts")
    # gcor's before side, the same stage the run's own page pairs against errts
    filtered = _read("filtered")
    bands = {"cardiac": (cardiac_l_freq, cardiac_h_freq),
             "resp": (resp_l_freq, resp_h_freq)}
    n_fft_floor = min(PSD_NFFT_CAP, len(haemo.times)) if haemo is not None else 0

    def haemo_of(t0, t1):
        return _condition_haemo(haemo, errts, t0, t1, n_fft_floor, bands, sep_bands,
                                condition_haemo_scalars, long_short_channels,
                                filtered=filtered)

    return _condition_entries(sections, raw_intensity, windows,
                              _cutoffs_from_sidecar(stages), sep_bands, haemo_of,
                              imu_of=_imu_slicer(imu))


def raw_condition_sections(
    sections: dict[str, Any],
    raw_intensity: "mne.io.Raw",
    windows: "list[tuple[str, float, float]]",
    cutoffs: dict[str, float],
    sep_bands=None,
    imu: "dict | None" = None,
) -> dict[str, Any]:
    """:func:`condition_sections` for a record written before the pipeline ran.

    Same section name, same entry shape, same assembler. What differs is forced by what a
    raw-only pass holds: there is no haemoglobin stage, so ``haemo_by_set`` and the
    correlation and CNR per-channel dicts have no input and their keys are left out rather
    than written as nulls. A reader tests for presence.

    The windows and the cutoffs are passed in rather than re-derived, because this caller
    screened the recording itself and already holds both. Re-deriving a cutoff is how a
    condition ends up measured against a line no channel was judged by.
    """
    return _condition_entries(sections, raw_intensity, windows, cutoffs, sep_bands,
                              imu_of=_imu_slicer(imu))


# Screening criteria a condition can be judged on: those with a windowed series to cut.
# One that starts screening without one is announced rather than silently skipped.
_CONDITION_SCREENABLE = frozenset({"good_frac"})


def _condition_entries(
    sections: dict[str, Any],
    raw_intensity: "mne.io.Raw",
    windows: "list[tuple[str, float, float]]",
    cutoffs: dict[str, float],
    sep_bands=None,
    haemo_of=None,
    imu_of=None,
) -> dict[str, Any]:
    """The ``by_condition`` entries themselves, for whichever writer holds the record.

    ``haemo_of(t0, t1)`` returns that condition's ``(haemo_by_set, per_channel)``; None is a
    pass with no haemoglobin stage, and leaves those keys out. ``imu_of(t0, t1)`` returns the
    condition's IMU summary, None on a recording without one. Everything else is read out
    of ``sections`` rather than measured, so the two writers cannot end up with different
    numbers for one recording.
    """
    from fnirs_pipe.qc.metrics import long_short_channels, screen_channels
    from fnirs_pipe.qc.metrics.screening import CRITERIA
    from fnirs_pipe.qc.metrics.windowed import condition_window_means

    unscreened = sorted(c.name for c in CRITERIA
                        if c.screens and c.name not in _CONDITION_SCREENABLE)
    if unscreened:
        logger.warning("by_condition: %s screens the run but has no windowed series to cut "
                       "to a condition, so a condition's verdict is now decided on less "
                       "than its run's (%s)",
                       ", ".join(unscreened), ", ".join(sorted(_CONDITION_SCREENABLE)))

    per_channel = sections.get("per_channel") or {}
    ch_names = list((per_channel.get("raw") or {}).get("sci_per_channel") or {})
    sliced_all = condition_slices_from_record(
        sections, ch_names, windows, cutoffs["sci"], cutoffs["psp"])
    if not sliced_all:
        return {}

    windowed = sections.get("windowed") or {}
    gvtd_times = windowed.get("gvtd_window_times_s") or []
    # measured on the corrected file where there is one, which is what the report says on
    # the rows it prints; on a raw-only record there is only the one stage
    _GVTD_SERIES = (("gvtd_mean", "gvtd_per_window"),
                    ("gvtd_p95", "gvtd_p95_per_window"),
                    ("gvtd_filt_mean", "gvtd_filt_per_window"),
                    ("gvtd_filt_p95", "gvtd_filt_p95_per_window"))

    def _gvtd_slices(suffix: str) -> dict:
        return {key: condition_window_means(windowed[series + suffix], gvtd_times, windows)
                for key, series in _GVTD_SERIES
                if windowed.get(series + suffix) and gvtd_times}

    # separate measurements rather than subsets, GVTD being an RMS across channels
    gvtd_sets = {"long": _gvtd_slices(""),
                 "short": _gvtd_slices("_short"),
                 "all": _gvtd_slices("_all")}
    # The run's booleans kept as spans, so a condition counts them over its own stretch,
    # against the run's own cutoff rather than one re-derived on a piece. Three families and
    # three sets each; the long set is under the plain key, the convention the writers
    # follow. A span list the writer had no input for is absent and its scalars come back
    # None, which is how a raw-only record carries no correction footprint.
    _SPAN_FAMILIES = (
        ("gvtd_above_spans", "gvtd_pct_above_thresh", "gvtd_num_above_thresh", None),
        ("spike_spans", "spike_pct_frames", "spike_num_frames", None),
        ("motion_corrected_spans", "motion_corrected_pct", "motion_corrected_num",
         "motion_corrected_n_segments"),
    )

    def _spans_of(stem: str, set_name: str) -> list:
        key = f"{stem}_s" if set_name == "long" else f"{stem}_{set_name}_s"
        return windowed.get(key) or []

    sfreq = float(raw_intensity.info["sfreq"])
    long_names, short_names = long_short_channels(raw_intensity, sep_bands)

    window_of = {w[0]: w for w in windows}
    out: dict[str, Any] = {}
    for label, sliced in sliced_all.items():
        t0, t1 = window_of[label][1], window_of[label][2]
        # this condition's own verdict on the run's line; the data was processed under the
        # run's, which the run's own section carries
        cond_frac = sliced.get("good_frac_per_channel") or {}
        cond_bad, _ = screen_channels({"good_frac": cond_frac}, cutoffs)
        retention = (1.0 - len(cond_bad) / len(cond_frac)) if cond_frac else None

        # the frame count is that share of this stretch's samples, not a second pass. The
        # long set fills the condition's own scalars as well, the run's rows being long.
        shares, n_frames, n_segments = {}, {}, {}
        span_counts_by_set: dict[str, dict] = {"all": {}, "long": {}, "short": {}}
        for stem, share_key, count_key, seg_key in _SPAN_FAMILIES:
            for set_name in span_counts_by_set:
                spans = _spans_of(stem, set_name)
                if not spans:
                    continue
                share, n_seg = span_counts(spans, t0, t1)
                counted = {share_key: share,
                           count_key: (None if share is None
                                       else int(round(share * (t1 - t0) * sfreq)))}
                if seg_key:
                    counted[seg_key] = n_seg
                span_counts_by_set[set_name].update(counted)
                if set_name == "long":
                    shares[share_key] = counted[share_key]
                    n_frames[count_key] = counted[count_key]
                    if seg_key:
                        n_segments[seg_key] = n_seg

        scalars = condition_scalars(
            sliced, {k: float(v[label]) for k, v in gvtd_sets["long"].items() if label in v},
            shares=shares, n_frames=n_frames, n_segments=n_segments, retention=retention)
        scalars["n_long_channels"] = len(long_names)
        scalars["n_short_channels"] = len(short_names)

        haemo_by_set, haemo_per_channel = haemo_of(t0, t1) if haemo_of else ({}, {})
        # the long set, matching what the run's own haemoglobin rows report
        scalars.update(haemo_by_set.get("long") or haemo_by_set.get("all") or {})
        if imu_of:
            scalars.update(imu_of(t0, t1))

        # no `_post` half: the GVTD series is measured on the corrected file
        motion_by_set = {
            name: {**{k: float(v[label]) for k, v in series.items() if label in v},
                   **{k: v for k, v in span_counts_by_set.get(name, {}).items()
                      if v is not None}}
            for name, series in gvtd_sets.items()
        }

        entry = {
            # unrounded, so a reader can pair these bounds back to the annotations they
            # came from
            "window_s": [float(t0), float(t1)],
            "bad_channels": cond_bad,
            "scalars": scalars,
            "od_by_set": condition_set_scalars(sliced, set(cond_bad), long_names,
                                               short_names),
            "motion_by_set": motion_by_set,
            "per_channel": {**sliced, **haemo_per_channel},
        }
        if haemo_by_set:
            entry["haemo_by_set"] = haemo_by_set
        out[label] = entry
    logger.info("by_condition: %d condition(s)", len(out))
    return out


def _condition_cnr(haemo, t0, t1, picks=None) -> dict:
    """CNR over one condition's own events, on a cut widened for the epoch windows.

    The bare span puts a block's onset at t=0 with no baseline before it, and mne drops the
    epoch for want of one. Widening can reach the next condition's onset, hence the filter:
    the events kept are this condition's, the extra samples only give them room.
    """
    from fnirs_pipe.qc.metrics.haemo import (
        CNR_BASELINE_S, CNR_RESPONSE_S, _cnr_metrics,
    )
    lo = max(0.0, float(t0) + min(0.0, CNR_BASELINE_S[0]))
    hi = min(float(haemo.times[-1]), float(t1) + max(0.0, CNR_RESPONSE_S[1]))
    if hi <= lo:
        return {}
    cut = haemo.copy().crop(lo, hi)
    # the window is on the data axis and an onset is not, so a recording cropped from
    # anywhere but zero needs the offset off before the two compare
    origin = float(haemo.first_time)
    tol = 0.5 / float(haemo.info["sfreq"])
    keep = [i for i, a in enumerate(cut.annotations)
            if float(t0) - tol <= float(a["onset"]) - origin <= float(t1) + tol]
    if not keep:
        return {}
    if len(keep) < len(cut.annotations):
        cut.set_annotations(cut.annotations[keep])
    if picks is not None:
        names = [c for c in picks if c in cut.ch_names]
        if not names:
            return {}
        cut = cut.pick(names)
    return _cnr_metrics(cut) or {}


def _condition_haemo(haemo, errts, t0, t1, n_fft_floor, bands, sep_bands,
                     condition_haemo_scalars, long_short_channels, filtered=None):
    """One condition's haemoglobin scalars per channel set, plus its per-channel dicts.

    Safe to crop because nothing here filters. The per-channel dicts are lifted out of the
    scalars, which flatten into a table of numbers.

    ``filtered`` is gcor's before side. Without it that pair spans the bandpass as well as
    the regression, which is not what the run's own page reports under the same names.
    """
    if haemo is None:
        return {}, {}

    def _cut(raw):
        if raw is None:
            return None
        lo, hi = max(0.0, float(t0)), min(float(raw.times[-1]), float(t1))
        return None if hi <= lo else raw.copy().crop(tmin=lo, tmax=hi)

    def _pick(raw, picks):
        if raw is None:
            return None
        names = [c for c in picks if c in raw.ch_names]
        return raw.copy().pick(names) if names else None

    haemo_cut, errts_cut, filt_cut = _cut(haemo), _cut(errts), _cut(filtered)
    if haemo_cut is None:
        return {}, {}
    by_set = {"all": condition_haemo_scalars(haemo_cut, errts_cut, n_fft_floor, bands,
                                             filtered=filt_cut)}
    long_names, short_names = long_short_channels(haemo_cut, sep_bands)
    for set_name, names in (("long", long_names), ("short", short_names)):
        picks = [c for c in names if c in haemo_cut.ch_names]
        if not picks:
            by_set[set_name] = {}
            continue
        row = condition_haemo_scalars(
            haemo_cut.copy().pick(picks), _pick(errts_cut, picks), n_fft_floor, bands,
            filtered=_pick(filt_cut, picks))
        # kept on the whole-file row alone, as the run's own sections keep it
        by_set[set_name] = {k: v for k, v in row.items() if k not in _WHOLE_FILE_KEYS}

    # CNR needs room either side of each onset, so it cuts the uncropped file itself
    def _scalars_of(cnr):
        return {k: v for k, v in cnr.items() if k != "cnr_per_channel"}

    cnr_all = _condition_cnr(haemo, t0, t1)
    by_set["all"].update(_scalars_of(cnr_all))
    for set_name, names in (("long", long_names), ("short", short_names)):
        if by_set.get(set_name):
            by_set[set_name].update(_scalars_of(_condition_cnr(haemo, t0, t1, names)))

    per_channel = {key: source[key]
                   for key, source in (("hbo_hbr_corr_per_channel", by_set["all"]),
                                       ("cnr_per_channel", cnr_all))
                   if source.get(key)}
    for scalars in by_set.values():
        scalars.pop("hbo_hbr_corr_per_channel", None)
    return by_set, per_channel


def compute_run_sections(
    stages: dict[str, Path],
    *,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
    qc_window_s: float = 10.0,
    bids_root: Path | None = None,
    sep_bands=None,
) -> dict[str, Any]:
    """Every SQM section for one run, keyed by section name, plus ``per_channel``.

    ``bids_root`` is where to look for the original recording when the path its sidecar
    recorded no longer resolves; without it a moved tree loses the ``raw*`` sections.
    """
    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.qc.metrics import (
        attach_windowed_series, compute_haemo_sqm, compute_prep_haemo_sqm,
        long_short_channels,
    )
    from fnirs_pipe.qc.metrics._helpers import separation_bands

    sep_bands = sep_bands if sep_bands is not None else separation_bands()

    sections: dict[str, Any] = {}
    per_channel: dict[str, Any] = {}

    section = _section_writer(sections, per_channel)

    try:
        sci_scores = _sci_scores(stages)
    except Exception:
        logger.warning("sci scores unavailable; sci metrics will be empty", exc_info=True)
        sci_scores = {}

    raw_intensity = None
    bids_input = _bids_input(stages, bids_root)
    if bids_input is None:
        logger.warning("no BIDS input resolvable from the od sidecar; raw sections skipped")
    else:
        try:
            raw_intensity = read_snirf(bids_input)
        except Exception:
            logger.warning("%s unreadable; raw sections skipped", bids_input, exc_info=True)

    # Everything time-indexed rather than channel-indexed: the per-window series and the
    # flagged spans. Its own section because `_split_scalars` files any list under
    # `per_channel`, which none of these are. Filled from several places below.
    windowed: dict[str, Any] = {}

    if raw_intensity is not None:
        # the input carries no marks of its own; the run's rejections come from the sci file
        bad_channels = list(_sidecar(stages["sci"]).get("bad_channels") or []) if "sci" in stages else []
        raw_intensity.info["bads"] = [c for c in bad_channels if c in raw_intensity.ch_names]

        raw_secs, raw_pc = raw_sections(
            raw_intensity, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq,
            sep_bands, _good_frac_scores(stages))
        sections.update(raw_secs)
        per_channel.update(raw_pc)

    # the head-movement record, where the recording carried one. Measured on the input's own
    # clock, the one the aux table was written on
    imu = _imu_of(stages)
    if imu and raw_intensity is not None:
        try:
            from fnirs_pipe.qc.metrics import imu_section
            sections["imu"] = imu_section(imu, raw_intensity, sep_bands)
        except Exception:
            logger.warning("imu section failed", exc_info=True)

    # written by the prep step rather than measured here: censoring is a decision the run
    # made, and re-deriving it would silently disagree with the marks already on the files
    if "sci" in stages:
        censor = _sidecar(stages["sci"]).get("gvtd_censor")
        if censor:
            sections["censor"] = censor

    # the spans the report draws on the carpet, on the channel set it draws them for, so the
    # figure reads them back instead of running the same detection again. Pre-correction by
    # definition, since the carpet is the uncorrected recording: the chain stops before
    # `desc-motcorrected`, whose spikes are the ones the correction failed to remove.
    spike_stage = stages.get("sci") or stages.get("od")
    if raw_intensity is not None or spike_stage is not None:
        try:
            from fnirs_pipe.qc.metrics import spike_segments
            spike_source = raw_intensity if raw_intensity is not None else read_snirf(spike_stage)
            spike_long, spike_short = long_short_channels(spike_source, sep_bands)
            # one list per channel set, because the test is ">= 10% of *these* channels
            # spiking" and the panel draws each set its own row: a span found on the short
            # channels is not a claim about the long ones. The long list keeps the plain key,
            # being the one the verdict, the detail figure and every older record read.
            for key, names in (("spike_spans_s", spike_long),
                               ("spike_spans_short_s", spike_short),
                               ("spike_spans_all_s", list(spike_source.ch_names))):
                if names:
                    picked = spike_source.copy().pick(names)
                elif key == "spike_spans_s":
                    picked = spike_source          # unsplit montage: every channel
                else:
                    continue
                windowed[key] = [list(span) for span in spike_segments(picked)]
        except Exception:
            logger.warning("windowed: spike spans failed", exc_info=True)

    # the same, for the samples above the run's GVTD threshold. Kept as spans rather than a
    # per-window share so a condition counts the run's own boolean over its own stretch
    gvtd_span_source = stages.get("motcorrected") or stages.get("sci") or stages.get("od")
    if gvtd_span_source is not None:
        try:
            from fnirs_pipe.qc.metrics import gvtd_above_segments
            span_raw = read_snirf(gvtd_span_source)
            windowed["gvtd_above_spans_s"] = [
                list(span) for span in gvtd_above_segments(span_raw, sep_bands)]
            # and one list per other set, each against its own threshold, following
            # `spike_spans_short_s`. Three sets, three traces, three cutoffs: one set's share
            # is not a second reading of another's, and the panel says so
            _, span_short = long_short_channels(span_raw, sep_bands)
            for key, span_picks in (("gvtd_above_spans_short_s", span_short),
                                    ("gvtd_above_spans_all_s", list(span_raw.ch_names))):
                if span_picks:
                    windowed[key] = [
                        list(span) for span in
                        gvtd_above_segments(span_raw, sep_bands, picks=span_picks)]
        except Exception:
            logger.warning("windowed: GVTD above-threshold spans failed", exc_info=True)

    # SCI, PSP and GVTD per window, all three on the same grid, but not off the same file.
    #
    # SCI and PSP come from the uncorrected OD, which is where the per-channel scores in
    # `raw` were taken, so both halves of the report's panel sit on one stage. GVTD keeps
    # the corrected file.
    sci_source  = stages.get("sci") or stages.get("od") or stages.get("motcorrected")
    gvtd_source = stages.get("motcorrected") or sci_source
    if sci_source is not None:
        try:
            raw_sci_od = read_snirf(sci_source)
            raw_gvtd_od = (raw_sci_od if gvtd_source == sci_source
                           else read_snirf(gvtd_source))
            series = attach_windowed_series(
                windowed, raw_sci_od, cardiac_l_freq, cardiac_h_freq, qc_window_s,
                gvtd_od=raw_gvtd_od, raw_intensity=raw_intensity, sep_bands=sep_bands)
            # on the GVTD file's clock and grid, so window i is the same stretch in both
            if imu:
                from fnirs_pipe.qc.metrics import imu_windowed
                windowed.update(imu_windowed(imu, raw_gvtd_od.times,
                                             float(raw_gvtd_od.info["sfreq"]), qc_window_s))
            # the channel by window matrices as well as the channel-averaged series: the
            # report's per-channel heatmap needs them, and it must not recompute
            for key in ("sci_matrix", "psp_matrix", "cv_matrix"):
                if series.get(key) is not None:
                    windowed[key] = np.asarray(series[key]).tolist()
            for key in ("sci_times", "psp_times", "cv_times",
                        "sci_channels", "psp_channels", "cv_channels"):
                if series.get(key) is not None:
                    windowed[key] = np.asarray(series[key]).tolist()
        except Exception:
            logger.warning("windowed: series failed", exc_info=True)

    # the OD either side of the motion step is on disk as desc-sci and desc-motcorrected,
    # so the correction's footprint is measurable here rather than only in memory
    if "motcorrected" in stages:
        mc_before = None
        try:
            mc_after = read_snirf(stages["motcorrected"])
            if "sci" in stages:
                mc_before = read_snirf(stages["sci"])
        except Exception:
            mc_after = None
            logger.warning("motion sections skipped; %s unreadable",
                           stages["motcorrected"], exc_info=True)
        if mc_after is not None:
            # each post section counts against the cutoff of the `raw` section on the same
            # channels, so a pair shares one yardstick and one channel set. Absent when the
            # matching raw section failed, and the post section then falls back to its own.
            motion_sections(mc_before, mc_after, section, windowed,
                            lambda name: (sections.get(name) or {}).get("gvtd_thresh"),
                            cardiac_l_freq, cardiac_h_freq, sep_bands)

    # One family per haemo stage the run actually wrote, each measured with the same
    # metrics as `preproc` so any of them subtracts against it. Band power and drift are
    # not among those metrics: past the bandpass they measure the filter, which is why
    # `compute_prep_haemo_sqm` is used for `preproc` alone. Every family is split by
    # separation, so a `_long` number is only ever compared with another `_long` one.
    haemo_computes = {
        "preproc": lambda r: compute_prep_haemo_sqm(
            r, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq),
        **{desc: (lambda r: compute_haemo_sqm(r)) for desc in _HAEMO_STAGES},
    }
    for desc, compute in haemo_computes.items():
        if desc not in stages:
            continue
        try:
            raw_haemo = read_snirf(stages[desc])
        except Exception:
            logger.warning("%s unreadable; its sections are skipped", stages[desc],
                           exc_info=True)
            continue
        haemo_secs, haemo_pc = haemo_sections(desc, raw_haemo, compute, sep_bands)
        sections.update(haemo_secs)
        per_channel.update(haemo_pc)

    if windowed:
        sections["windowed"] = windowed

    sections["per_channel"] = per_channel

    # last, because it reads the matrices and the per-channel dicts above rather than the
    # recordings. Written whether or not any report will ask for per-condition pages: what
    # is measured is the record's business, what is shown is the report's.
    if raw_intensity is not None:
        try:
            by_condition = condition_sections(
                sections, raw_intensity, stages,
                cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
                resp_l_freq=resp_l_freq, resp_h_freq=resp_h_freq, sep_bands=sep_bands,
                imu=imu)
        except Exception:
            logger.warning("by_condition section failed", exc_info=True)
            by_condition = {}
        if by_condition:
            sections["by_condition"] = by_condition
    return sections


def sqm_record_dict(sections: dict[str, Any], sources: list[str]) -> dict[str, Any]:
    """The on-disk record: provenance keys wrapped around the sections themselves.

    Shared by both writers so a record is the same shape whichever command made it. The
    provenance table lists a record's metric names prefixed the way the group table names
    its columns, so the two read as one vocabulary.
    """
    known = (*SECTIONS, *OPTIONAL_SECTIONS)
    metrics = [f"{s}_{k}" for s in known if isinstance(sections.get(s), dict)
               for k in sections[s]]
    return {
        "pipeline_version": __version__,
        "step": "sqm",
        "Sources": sources,
        "data": {
            "sections": [s for s in known if s in sections],
            "metrics": metrics,
            "n_metrics": len(metrics),
        },
        **sections,
    }


def write_run_sqm(
    nirs_dir: Path,
    label: str,
    stages: dict[str, Path],
    sections: dict[str, Any],
    bids_root: Path | None = None,
) -> Path:
    """Write ``<label>_desc-sqm_qc.json`` and its tables, provenance keys included.

    The provenance lives in the same file rather than a sidecar beside it: a sidecar for
    ``x.json`` would resolve to ``x.json`` itself. A top-level ``step`` is all the
    provenance graph needs to pick the file up.
    """
    sources = [p.as_posix() for p in stages.values()]
    bids_input = _bids_input(stages, bids_root)
    if bids_input is not None:
        sources.insert(0, bids_input.as_posix())

    out_path = record_path(nirs_dir, label)
    write_record(out_path, sqm_record_dict(sections, bids_uris(sources, out_path)))
    logger.info("SQM record -> %s", out_path)
    return out_path


def write_channel_table(
    nirs_dir: Path,
    label: str,
    stages: dict[str, Path],
    sections: dict[str, Any],
) -> None:
    """Write ``<label>_desc-channel_qc.tsv`` from the record and the sci sidecar, report or not."""
    sci = _sidecar(stages["sci"]) if "sci" in stages else {}
    params = sci.get("parameters") or {}
    scores = ((sections.get("per_channel") or {}).get("raw") or {}).get("sci_per_channel") or {}
    rows = channel_rows(sections, scores, sci.get("bad_channels") or [])
    save_channel_csv(rows, label, nirs_dir, params.get("sci_threshold"),
                     psp_threshold=params.get("psp_threshold"))


def build_sqm_records(
    nirs_dir: Path,
    *,
    cardiac_l_freq: float | None = None,
    cardiac_h_freq: float | None = None,
    resp_l_freq: float | None = None,
    resp_h_freq: float | None = None,
    qc_window_s: float | None = None,
    bids_root: Path | None = None,
    labels: "set[str] | None" = None,
    sep_bands=None,
) -> list[Path]:
    """Write one SQM record per run found under nirs_dir. Band edges default to the
    values the run's own sidecars recorded, so a past tree needs no arguments.

    ``labels`` restricts the work to those BIDS run stems, which is what a `--task-label`
    run wants: the tasks it did not touch keep the records they already had, computed with
    the settings they were computed under. Omit it to rebuild every run in the directory.

    ``bids_root`` rescues the ``raw*`` sections when the tree has been moved since the run:
    the sidecars name the original recording by an absolute path that no longer resolves,
    but the filename is still correct, so it is searched for there.

    ``sep_bands`` has to be the run's own, unlike the band edges above: nothing on disk
    records them, so left out they fall back to the package defaults and every ``*_long``
    and ``*_short`` metric describes a channel set the run did not use."""
    written: list[Path] = []
    for label, stages in scan_runs(Path(nirs_dir)).items():
        if labels is not None and label not in labels:
            continue
        bands = {
            "cardiac_l_freq": cardiac_l_freq, "cardiac_h_freq": cardiac_h_freq,
            "resp_l_freq": resp_l_freq, "resp_h_freq": resp_h_freq,
        }
        if any(v is None for v in bands.values()):
            recorded = _bands(stages)
            if recorded is None:
                logger.warning("%s: no band edges in the sidecars and none supplied; skipped", label)
                continue
            bands = {k: (v if v is not None else recorded[k]) for k, v in bands.items()}
        # a tree with no stored length falls back to the pipeline's default, so its series
        # are still binned on a known grid
        window_s = qc_window_s if qc_window_s is not None else (_window_s(stages) or 10.0)
        # each section guards itself, so what reaches here is fatal for this run only;
        # the remaining runs still get their records
        try:
            sections = compute_run_sections(
                stages, **bands, qc_window_s=window_s, bids_root=bids_root,
                sep_bands=sep_bands)
            written.append(
                write_run_sqm(Path(nirs_dir), label, stages, sections, bids_root))
        except Exception:
            logger.error("%s: SQM record could not be written", label, exc_info=True)
            continue
        try:
            write_channel_table(Path(nirs_dir), label, stages, sections)
        except Exception:
            logger.error("%s: channel table could not be written", label, exc_info=True)
    return written
