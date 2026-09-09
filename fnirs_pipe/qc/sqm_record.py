"""Per-run SQM record: one JSON per BIDS run, assembled after the pipeline has finished.

The pipeline writes each stage to disk; this reads those files back and computes every
metric once, at one named stage, through one writer. That is what stops the two
checkpoints from overwriting each other, and it works on any tree a past run left behind.

Sections name the input a metric family was measured on. The metric names inside a
section are the same names the metric functions have always returned:

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

**Split a new metric three ways unless it cannot be.** A short channel sits millimetres
from its source: it returns far more light, a far stronger pulse, and it sees scalp rather
than cortex, so an average over both sets describes neither. Two kinds of metric are the
exception, and both are exceptions for a reason that can be stated:

    montage-wide     GVTD, the spike counts, the motion-correction footprint. These
                     aggregate across channels, and a handful of scalp channels has no
                     reference distribution to read them against. A short-channel GVTD
                     would be a number without a meaning.
    not channel-based  ``pct_data_retained`` is a share of the recording's duration. It is
                     the same number for every channel set, so it stays on the whole-file
                     section alone (see ``_WHOLE_FILE_KEYS``); repeating it under ``_long``
                     would name a quantity that does not exist.

Anything else, split it. The metric that made this a rule was ``hbo_hbr_corr_mean``: the
HbO-HbR anticorrelation is a property of cortical haemodynamics, a short channel has none,
and mixing the two moved one of the numbers that decides whether a run looks usable.

Bad channels: the Beer-Lambert conversion is the dividing line, never the section.

    raw / raw_long / raw_short / motion   intensity and OD, include them
    preproc / filtered / resampled / errts  haemoglobin, exclude them

A rejected channel is still part of what the machine recorded, so everything measured
before Beer-Lambert describes the recording as it arrived; after it the channel is out of
the analysis, and those metrics go through ``mne.pick_types``, which drops bads by
default. Reading it the other way round gives ``sci_mean`` over channels that were
selected for having good SCI, which is circular and can never fall below the threshold.
``channel_retention_rate`` is what says how many were dropped.

Per-channel values live under ``per_channel`` so the sections stay scalar. Those are
always complete, every channel, whichever section they sit under.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mne
import numpy as np

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.sqm_record")

_DESC_RE = re.compile(r"_desc-([A-Za-z0-9]+)_nirs\.snirf$")

# the post-Beer-Lambert stages, in the order the pipeline writes them. Each is its own
# section named after the file it measured, which is the rule every other section follows.
# The `final` section this replaced named a position rather than an input: it meant
# "resampled, or filtered, or preproc, whichever exists", so the same key described the
# bandpassed file on one run and the unfiltered one on another, and a run that regressed
# confounds had its actual endpoint (`errts`) sitting outside it.
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

SECTIONS = ("raw", "raw_long", "raw_short", "motion", "motion_post",
            "motion_post_long", "motion_post_short", "windowed",
            *_HAEMO_SECTIONS)

# Sections only some runs have. Kept out of SECTIONS, which means "every run writes this" and
# is asserted as such: censoring is opt-in, so a record without it is correct, not incomplete.
# The group table still descends into these.
OPTIONAL_SECTIONS = ("censor",)

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
        match = _DESC_RE.search(path.name)
        if match is None:
            continue
        label = path.name[: path.name.index("_desc-")]
        runs.setdefault(label, {})[match.group(1)] = path
    return runs


def entities_of(label: str) -> dict[str, str | None]:
    """Pull the BIDS entities back out of a run label, for the columns a database wants."""
    return {
        key: (m.group(1) if (m := re.search(rf"_{key}-([A-Za-z0-9]+)", label)) else None)
        for key in ("ses", "task", "run")
    }


def record_path(nirs_dir: Path, label: str) -> Path:
    return Path(nirs_dir) / f"{label}_desc-sqm_nirs.json"


def _sidecar(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _bids_input(stages: dict[str, Path], bids_root: Path | None = None) -> Path | None:
    """The original recording: the OD file is the only derivative whose source is it.

    ``Sources`` holds the absolute path the run saw, so a tree that has been copied to
    another machine, or whose input has moved, names a file that is no longer there. The
    name is still right, so fall back to finding it under ``bids_root``. Without that
    fallback the three ``raw*`` sections vanish from every rebuilt record, with one
    warning to say why.
    """
    if "od" not in stages:
        return None
    sources = _sidecar(stages["od"]).get("Sources") or []
    if not sources:
        return None
    src = Path(sources[0])
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
    # pre-0.20 trees have no stored scores; the OD file still supports recomputing them
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

    Not recomputed when it is absent, unlike :func:`_sci_scores`. The share is a count
    against two thresholds over a scope, so recomputing it here would need this run's
    ``--sci-threshold``, ``--psp-threshold``, ``--min-good-frac`` and ``--screen-scope``,
    and guessing any of them would put a number in the record that no channel was actually
    judged by. A tree written before the share was stored has none, and the report says so
    by leaving the row out.
    """
    if "sci" not in stages:
        return {}
    scores = _sidecar(stages["sci"]).get("good_frac_scores") or {}
    return {k: float(v) for k, v in scores.items()}


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
    channels; GVTD, spikes and drift do not, because they aggregate over a montage and a
    handful of scalp channels has no reference distribution to read them against.

    The share is a value here and not a verdict. Nothing colours a short channel's row,
    because a short channel's coupling is high by construction and the published cutoffs
    were never set for it; the long section is where the number is read against a line.
    """
    from fnirs_pipe.qc.metrics import _intensity_metrics, _psp_metrics, _sci_metrics
    from fnirs_pipe.qc.metrics.coupling import _good_frac_metrics

    raw_short = raw_intensity.copy().pick(short_names)
    short_set = set(short_names)
    short_sci = {k: v for k, v in sci_scores.items() if k in short_set}
    short_bad = [c for c in bad_channels if c in short_set]
    short_frac = {k: v for k, v in (good_frac_scores or {}).items() if k in short_set}

    record: dict[str, Any] = {"n_channels": len(short_names), "n_bad": len(short_bad)}
    record.update(_sci_metrics(short_sci, short_bad))
    record.update(_psp_metrics(raw_short, cardiac_l_freq, cardiac_h_freq))
    record.update(_good_frac_metrics(short_frac))
    intensity = _intensity_metrics(raw_short)
    record.update({k: v for k, v in intensity.items()
                   if k.startswith(("snr_", "cv_", "mean_amp_"))})
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
    channel sees scalp, so a mean taken over both sets describes neither. Global
    correlation is the sharpest case, since short channels correlate strongly with one
    another and lift it by construction, but the HbO-HbR anticorrelation is the one that
    misleads most: it is a property of cortical haemodynamics, and a short channel has
    none, so mixing the two moves the number that decides whether a run looks usable.

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
) -> dict[str, Any]:
    """Did the correction work, and did it cost anything.

    Two questions, not one. GVTD and the spike counts answer the first: both measure what
    the correction is there to remove, so a run that improved shows them fall against the
    same keys in ``raw``. SCI and PSP answer the second: they live in the cardiac band,
    above the frequencies motion correction touches, so they should come back unchanged. A
    drop means the correction ate physiology along with the artifact, which is the failure
    mode wavelet correction has and TDDR largely does not.

    Hand-picked rather than ``compute_raw_sqm``: this file is optical density, so that
    would set the whole intensity family to None and add CP and channel distance on top,
    twenty-odd keys of which half would be empty.
    """
    from fnirs_pipe.qc.metrics import (
        _mean_or_none, _motion_metrics, _psp_metrics, _spike_metrics, compute_sci_scores,
    )

    record: dict[str, Any] = {}
    record.update(_motion_metrics(raw_od))
    record.update(_spike_metrics(raw_od))
    # the mean over every channel, bads included, so it is comparable with `raw`
    scores, _ = compute_sci_scores(raw_od, cardiac_l_freq, cardiac_h_freq)
    record["sci_mean"] = _mean_or_none(scores.values())
    # per channel as well as the mean: the report pairs these against the pre-correction
    # scores channel by channel, and a mean cannot say which channel the correction cost
    record["sci_per_channel"] = {k: float(v) for k, v in scores.items()}
    record.update(_psp_metrics(raw_od, cardiac_l_freq, cardiac_h_freq))
    return record


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
            # one list per separation class, because the test is ">= 10% of *these* channels
            # spiking" and the panel draws each class its own row: a span found on the short
            # channels is not a claim about the long ones. The long list keeps the plain key,
            # being the one the verdict, the detail figure and every older record read.
            for key, names in (("spike_spans_s", spike_long),
                               ("spike_spans_short_s", spike_short)):
                if names:
                    picked = spike_source.copy().pick(names)
                elif key == "spike_spans_s":
                    picked = spike_source          # unsplit montage: every channel, as before
                else:
                    continue
                windowed[key] = [list(span) for span in spike_segments(picked)]
        except Exception:
            logger.warning("windowed: spike spans failed", exc_info=True)

    # SCI, PSP and GVTD per window, all three on the same grid, but not off the same file.
    #
    # SCI and PSP come from the uncorrected OD, which is where the per-channel scores in
    # `raw` were taken. The report draws both halves of one panel from these, and reading
    # the windows off the corrected file put the heatmap a stage ahead of the lollipop
    # beside it. GVTD keeps the corrected file: it measures the movement the correction
    # exists to remove, so the corrected one is the informative stage for it.
    sci_source  = stages.get("sci") or stages.get("od") or stages.get("motcorrected")
    gvtd_source = stages.get("motcorrected") or sci_source
    if sci_source is not None:
        try:
            raw_sci_od = read_snirf(sci_source)
            raw_gvtd_od = (raw_sci_od if gvtd_source == sci_source
                           else read_snirf(gvtd_source))
            series = attach_windowed_series(
                windowed, raw_sci_od, cardiac_l_freq, cardiac_h_freq, qc_window_s,
                gvtd_od=raw_gvtd_od, raw_intensity=raw_intensity)
            # the channel by window matrices as well as the channel-averaged series: the
            # report's per-channel heatmap needs them, and it must not recompute
            for key in ("sci_matrix", "psp_matrix"):
                if series.get(key) is not None:
                    windowed[key] = np.asarray(series[key]).tolist()
            for key in ("sci_times", "psp_times"):
                if series.get(key) is not None:
                    windowed[key] = np.asarray(series[key]).tolist()
        except Exception:
            logger.warning("windowed: series failed", exc_info=True)

    # the OD either side of the motion step is on disk as desc-sci and desc-motcorrected,
    # so the correction's footprint is measurable here rather than only in memory
    if "sci" in stages and "motcorrected" in stages:
        from fnirs_pipe.qc.metrics import (
            motion_corrected_segments, motion_correction_metrics,
        )
        section("motion", lambda: motion_correction_metrics(
            read_snirf(stages["sci"]), read_snirf(stages["motcorrected"])))
        # `motion` counts the spans; this is where they are, for the figures that draw them
        try:
            windowed["motion_corrected_spans_s"] = [
                list(span) for span in motion_corrected_segments(
                    read_snirf(stages["sci"]), read_snirf(stages["motcorrected"]))
            ]
        except Exception:
            logger.warning("windowed: correction spans failed", exc_info=True)

    # the same OD-domain metrics as `raw`, measured on the corrected file. Same domain and
    # same units, so these subtract against `raw`; nothing across Beer-Lambert does.
    # Split by separation the same way `raw` is, so every post section has a `raw*` section
    # on the identical channel set to subtract against; mixing the two splits would compare
    # a long-channel GVTD against an all-channel one and read the difference as an effect
    # of the correction.
    if "motcorrected" in stages:
        try:
            raw_motcorr = read_snirf(stages["motcorrected"])
        except Exception:
            raw_motcorr = None
            logger.warning("%s unreadable; motion_post sections skipped",
                           stages["motcorrected"], exc_info=True)
        if raw_motcorr is not None:
            section("motion_post", lambda: _motion_post_section(
                raw_motcorr, cardiac_l_freq, cardiac_h_freq))
            post_long, post_short = long_short_channels(raw_motcorr, sep_bands)
            if post_long and len(post_long) < len(raw_motcorr.ch_names):
                section("motion_post_long", lambda: _motion_post_section(
                    raw_motcorr.copy().pick(post_long), cardiac_l_freq, cardiac_h_freq))
            if post_short:
                section("motion_post_short", lambda: _motion_post_section(
                    raw_motcorr.copy().pick(post_short), cardiac_l_freq, cardiac_h_freq))

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
    return sections


def sqm_record_dict(sections: dict[str, Any], sources: list[str]) -> dict[str, Any]:
    """The on-disk record: provenance keys wrapped around the sections themselves.

    Shared by both writers so a record is the same shape whichever command made it. The
    provenance table lists a record's metric names prefixed the way the group table names
    its columns, so the two read as one vocabulary.
    """
    from fnirs_pipe import __version__

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
    """Write ``<label>_desc-sqm_nirs.json``, provenance keys included.

    The provenance lives in the same file rather than a sidecar beside it: a sidecar for
    ``x.json`` would resolve to ``x.json`` itself. A top-level ``step`` is all the
    provenance graph needs to pick the file up.
    """
    sources = [p.as_posix() for p in stages.values()]
    bids_input = _bids_input(stages, bids_root)
    if bids_input is not None:
        sources.insert(0, bids_input.as_posix())

    record = sqm_record_dict(sections, sources)
    out_path = record_path(nirs_dir, label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("SQM record → %s", out_path)
    return out_path


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
) -> list[Path]:
    """Write one SQM record per run found under nirs_dir. Band edges default to the
    values the run's own sidecars recorded, so a past tree needs no arguments.

    ``labels`` restricts the work to those BIDS run stems, which is what a `--task-label`
    run wants: the tasks it did not touch keep the records they already had, computed with
    the settings they were computed under. Omit it to rebuild every run in the directory.

    ``bids_root`` rescues the ``raw*`` sections when the tree has been moved since the run:
    the sidecars name the original recording by an absolute path that no longer resolves,
    but the filename is still correct, so it is searched for there."""
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
        # a tree written before the length was stored falls back to the default the
        # pipeline has always used, so its series are still binned on a known grid
        window_s = qc_window_s if qc_window_s is not None else (_window_s(stages) or 10.0)
        # each section guards itself, so what reaches here is fatal for this run only;
        # the remaining runs still get their records
        try:
            sections = compute_run_sections(
                stages, **bands, qc_window_s=window_s, bids_root=bids_root)
            written.append(
                write_run_sqm(Path(nirs_dir), label, stages, sections, bids_root))
        except Exception:
            logger.error("%s: SQM record could not be written", label, exc_info=True)
    return written
