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
    final      the last haemo file the run produced (resampled > filtered > preproc)

The three ``raw*`` sections are one file seen through three channel sets, so the only
thing that differs between them is source-detector separation.

Bad channels: the Beer-Lambert conversion is the dividing line, never the section.

    raw / raw_long / raw_short / motion   intensity and OD, include them
    preproc / final                       haemoglobin, exclude them

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

# the haemo file the "final" section measures, best first
_FINAL_ORDER = ("resampled", "filtered", "preproc")

SECTIONS = ("raw", "raw_long", "raw_short", "motion", "motion_post",
            "motion_post_long", "motion_post_short", "windowed", "preproc", "final")


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
    from fnirs_pipe.qc.quantitative_metrics import compute_sci_scores
    bands = _bands(stages) or {}
    if not bands:
        return {}
    scores, _ = compute_sci_scores(
        read_snirf(source), bands["cardiac_l_freq"], bands["cardiac_h_freq"])
    return scores


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only the metrics that answer "are the short-channel regressors trustworthy".

    Coupling (SCI, PSP) and amplitude (SNR, CV) transfer to short channels; GVTD, spikes
    and drift do not, because they aggregate over a montage and a handful of scalp
    channels has no reference distribution to read them against.
    """
    from fnirs_pipe.qc.quantitative_metrics import _intensity_metrics, _psp_metrics, _sci_metrics

    raw_short = raw_intensity.copy().pick(short_names)
    short_sci = {k: v for k, v in sci_scores.items() if k in set(short_names)}
    short_bad = [c for c in bad_channels if c in set(short_names)]

    record: dict[str, Any] = {"n_channels": len(short_names), "n_bad": len(short_bad)}
    record.update(_sci_metrics(short_sci, short_bad))
    record.update(_psp_metrics(raw_short, cardiac_l_freq, cardiac_h_freq))
    intensity = _intensity_metrics(raw_short)
    record.update({k: v for k, v in intensity.items()
                   if k.startswith(("snr_", "cv_", "mean_amp_"))})
    return _split_scalars(record)


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
    from fnirs_pipe.qc.quantitative_metrics import (
        _mean_or_none, _motion_metrics, _psp_metrics, _spike_metrics, compute_sci_scores,
    )

    record: dict[str, Any] = {}
    record.update(_motion_metrics(raw_od))
    record.update(_spike_metrics(raw_od))
    # the mean over every channel, bads included, so it is comparable with `raw`
    scores, _ = compute_sci_scores(raw_od, cardiac_l_freq, cardiac_h_freq)
    record["sci_mean"] = _mean_or_none(scores.values())
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
) -> dict[str, Any]:
    """Every SQM section for one run, keyed by section name, plus ``per_channel``.

    ``bids_root`` is where to look for the original recording when the path its sidecar
    recorded no longer resolves; without it a moved tree loses the ``raw*`` sections.
    """
    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.qc.quantitative_metrics import (
        attach_windowed_series, compute_haemo_sqm, compute_prep_haemo_sqm, compute_raw_sqm,
        long_short_channels,
    )

    sections: dict[str, Any] = {}
    per_channel: dict[str, Any] = {}

    def section(name: str, compute) -> None:
        """Run one section. A failure costs that section alone, never the whole record."""
        try:
            scalars, nested = _split_scalars(compute())
        except Exception:
            logger.warning("%s: section failed", name, exc_info=True)
            return
        sections[name] = scalars
        per_channel[name] = nested

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

        section("raw", lambda: compute_raw_sqm(
            raw_intensity, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq))

        long_names, short_names = long_short_channels(raw_intensity)
        if long_names and len(long_names) < len(raw_intensity.ch_names):
            def raw_long_section():
                raw_long = raw_intensity.copy().pick(long_names)
                long_sci = {k: v for k, v in sci_scores.items() if k in set(long_names)}
                long_bad = [c for c in bad_channels if c in set(long_names)]
                return compute_raw_sqm(
                    raw_long, long_sci, long_bad, cardiac_l_freq, cardiac_h_freq)
            section("raw_long", raw_long_section)
        if short_names:
            # already returns the (scalars, nested) split, so it bypasses `section`
            try:
                sections["raw_short"], per_channel["raw_short"] = _short_section(
                    raw_intensity, short_names, sci_scores, bad_channels,
                    cardiac_l_freq, cardiac_h_freq)
            except Exception:
                logger.warning("raw_short: section failed", exc_info=True)

    # the spans the report draws on the carpet, on the channel set it draws them for, so the
    # figure reads them back instead of running the same detection again. Pre-correction by
    # definition, since the carpet is the uncorrected recording: the chain stops before
    # `desc-motcorrected`, whose spikes are the ones the correction failed to remove.
    spike_stage = stages.get("sci") or stages.get("od")
    if raw_intensity is not None or spike_stage is not None:
        try:
            from fnirs_pipe.qc.quantitative_metrics import spike_segments
            spike_source = raw_intensity if raw_intensity is not None else read_snirf(spike_stage)
            spike_long, _ = long_short_channels(spike_source)
            if spike_long:
                spike_source = spike_source.copy().pick(spike_long)
            windowed["spike_spans_s"] = [list(span) for span in spike_segments(spike_source)]
        except Exception:
            logger.warning("windowed: spike spans failed", exc_info=True)

    # SCI, PSP and GVTD per window, all three on the same grid.
    #
    # Read off the corrected OD when there is one. That is the signal Beer-Lambert actually
    # received, and it is what the subject report draws, so record and report cannot drift.
    od_source = stages.get("motcorrected") or stages.get("sci") or stages.get("od")
    if od_source is not None:
        try:
            series = attach_windowed_series(
                windowed, read_snirf(od_source), cardiac_l_freq, cardiac_h_freq, qc_window_s)
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
        from fnirs_pipe.qc.quantitative_metrics import (
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
            post_long, post_short = long_short_channels(raw_motcorr)
            if post_long and len(post_long) < len(raw_motcorr.ch_names):
                section("motion_post_long", lambda: _motion_post_section(
                    raw_motcorr.copy().pick(post_long), cardiac_l_freq, cardiac_h_freq))
            if post_short:
                section("motion_post_short", lambda: _motion_post_section(
                    raw_motcorr.copy().pick(post_short), cardiac_l_freq, cardiac_h_freq))

    if "preproc" in stages:
        section("preproc", lambda: compute_prep_haemo_sqm(
            read_snirf(stages["preproc"]), cardiac_l_freq, cardiac_h_freq,
            resp_l_freq, resp_h_freq))

    final_desc = next((d for d in _FINAL_ORDER if d in stages), None)
    if final_desc is not None:
        section("final", lambda: compute_haemo_sqm(read_snirf(stages[final_desc])))
        if "final" in sections:
            sections["final"]["stage"] = final_desc

    if windowed:
        sections["windowed"] = windowed

    sections["per_channel"] = per_channel
    return sections


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
    from fnirs_pipe import __version__

    sources = [p.as_posix() for p in stages.values()]
    bids_input = _bids_input(stages, bids_root)
    if bids_input is not None:
        sources.insert(0, bids_input.as_posix())

    # the provenance table lists a checkpoint's metric names; prefixed the way the group
    # table names its columns, so the two read as one vocabulary
    metrics = [f"{s}_{k}" for s in SECTIONS if isinstance(sections.get(s), dict)
               for k in sections[s]]
    record = {
        "pipeline_version": __version__,
        "step": "sqm",
        "Sources": sources,
        "data": {
            "sections": [s for s in SECTIONS if s in sections],
            "metrics": metrics,
            "n_metrics": len(metrics),
        },
        **sections,
    }
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
