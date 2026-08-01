"""Per-run SQM record: one JSON per BIDS run, assembled after the pipeline has finished.

The pipeline writes each stage to disk; this reads those files back and computes every
metric once, at one named stage, through one writer. That is what stops the two
checkpoints from overwriting each other, and it works on any tree a past run left behind.

Sections name the input a metric family was measured on. The metric names inside a
section are the same names the metric functions have always returned:

    raw       every channel of the original recording, the archival view
    raw_long  long channels only, the view a quality judgement should use
    short     the short-channel regressors: are they trustworthy
    preproc   Beer-Lambert output, before any filtering
    final     the last haemo file the run produced (resampled > filtered > preproc)

Per-channel values live under ``per_channel`` so the sections stay scalar.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mne

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.sqm_record")

_DESC_RE = re.compile(r"_desc-([A-Za-z0-9]+)_nirs\.snirf$")

# the haemo file the "final" section measures, best first
_FINAL_ORDER = ("resampled", "filtered", "preproc")

SECTIONS = ("raw", "raw_long", "short", "preproc", "final")


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


def _sidecar(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _bids_input(stages: dict[str, Path]) -> Path | None:
    """The original recording: the OD file is the only derivative whose source is it."""
    if "od" not in stages:
        return None
    sources = _sidecar(stages["od"]).get("Sources") or []
    if not sources:
        return None
    src = Path(sources[0])
    return src if src.exists() else None


def _bands(stages: dict[str, Path]) -> dict[str, float] | None:
    """Band edges the run used, read back from any sidecar that recorded them."""
    keys = ("cardiac_l_freq", "cardiac_h_freq", "resp_l_freq", "resp_h_freq")
    for path in stages.values():
        params = _sidecar(path).get("parameters") or {}
        if all(params.get(k) is not None for k in keys):
            return {k: float(params[k]) for k in keys}
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


def _long_short(raw: mne.io.Raw) -> tuple[list[str], list[str]]:
    """Channel names split by source-detector separation, using mne_nirs' definition."""
    from mne_nirs.channels import get_long_channels, get_short_channels
    try:
        long_names = list(get_long_channels(raw.copy()).ch_names)
    except Exception:
        long_names = list(raw.ch_names)
    try:
        short_names = list(get_short_channels(raw.copy()).ch_names)
    except Exception:
        short_names = []
    return long_names, short_names


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


def compute_run_sections(
    stages: dict[str, Path],
    *,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    resp_l_freq: float,
    resp_h_freq: float,
) -> dict[str, Any]:
    """Every SQM section for one run, keyed by section name, plus ``per_channel``."""
    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.qc.quantitative_metrics import (
        compute_haemo_sqm, compute_prep_haemo_sqm, compute_raw_sqm,
    )

    sections: dict[str, Any] = {}
    per_channel: dict[str, Any] = {}
    sci_scores = _sci_scores(stages)

    bids_input = _bids_input(stages)
    if bids_input is not None:
        raw_intensity = read_snirf(bids_input)
        # the input carries no marks of its own; the run's rejections come from the sci file
        bad_channels = list(_sidecar(stages["sci"]).get("bad_channels") or []) if "sci" in stages else []
        raw_intensity.info["bads"] = [c for c in bad_channels if c in raw_intensity.ch_names]

        scalars, nested = _split_scalars(compute_raw_sqm(
            raw_intensity, sci_scores, bad_channels, cardiac_l_freq, cardiac_h_freq))
        sections["raw"] = scalars
        per_channel["raw"] = nested

        long_names, short_names = _long_short(raw_intensity)
        if long_names and len(long_names) < len(raw_intensity.ch_names):
            raw_long = raw_intensity.copy().pick(long_names)
            long_sci = {k: v for k, v in sci_scores.items() if k in set(long_names)}
            long_bad = [c for c in bad_channels if c in set(long_names)]
            scalars, nested = _split_scalars(compute_raw_sqm(
                raw_long, long_sci, long_bad, cardiac_l_freq, cardiac_h_freq))
            sections["raw_long"] = scalars
            per_channel["raw_long"] = nested
        if short_names:
            scalars, nested = _short_section(
                raw_intensity, short_names, sci_scores, bad_channels,
                cardiac_l_freq, cardiac_h_freq)
            sections["short"] = scalars
            per_channel["short"] = nested
    else:
        logger.warning("no BIDS input resolvable from the od sidecar; raw sections skipped")

    if "preproc" in stages:
        raw_preproc = read_snirf(stages["preproc"])
        scalars, nested = _split_scalars(compute_prep_haemo_sqm(
            raw_preproc, cardiac_l_freq, cardiac_h_freq, resp_l_freq, resp_h_freq))
        sections["preproc"] = scalars
        per_channel["preproc"] = nested

    final_desc = next((d for d in _FINAL_ORDER if d in stages), None)
    if final_desc is not None:
        raw_final = read_snirf(stages[final_desc])
        scalars, nested = _split_scalars(compute_haemo_sqm(raw_final))
        sections["final"] = scalars
        sections["final"]["stage"] = final_desc
        per_channel["final"] = nested

    sections["per_channel"] = per_channel
    return sections


def write_run_sqm(
    nirs_dir: Path,
    label: str,
    stages: dict[str, Path],
    sections: dict[str, Any],
) -> Path:
    """Write ``<label>_desc-sqm_nirs.json``, provenance keys included.

    The provenance lives in the same file rather than a sidecar beside it: a sidecar for
    ``x.json`` would resolve to ``x.json`` itself. A top-level ``step`` is all the
    provenance graph needs to pick the file up.
    """
    from fnirs_pipe import __version__

    sources = [p.as_posix() for p in stages.values()]
    bids_input = _bids_input(stages)
    if bids_input is not None:
        sources.insert(0, bids_input.as_posix())

    record = {
        "pipeline_version": __version__,
        "step": "sqm",
        "Sources": sources,
        "data": {"sections": [s for s in SECTIONS if s in sections]},
        **sections,
    }
    out_path = Path(nirs_dir) / f"{label}_desc-sqm_nirs.json"
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
) -> list[Path]:
    """Write one SQM record per run found under nirs_dir. Band edges default to the
    values the run's own sidecars recorded, so a past tree needs no arguments."""
    written: list[Path] = []
    for label, stages in scan_runs(Path(nirs_dir)).items():
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
        try:
            sections = compute_run_sections(stages, **bands)
            written.append(write_run_sqm(Path(nirs_dir), label, stages, sections))
        except Exception:
            logger.warning("%s: SQM record failed", label, exc_info=True)
    return written
