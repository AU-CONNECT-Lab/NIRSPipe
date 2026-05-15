"""Raw SNIRF cropping: single-segment and multi-segment to BIDS derivatives."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.utils.snirf_prep import (
    annotations_to_df,
    bids_stem,
    copy_sidecars,
    deriv_nirs_dir,
    ensure_dataset_description,
    find_snirf,
    read_raw_snirf,
)

logger = get_logger("pipeline.crop")

_DERIV_NAME = "cropped"


def _write_segment(raw_seg, out_snirf: Path) -> None:
    import mne
    mne.export.export_raw(str(out_snirf), raw_seg, fmt="snirf", overwrite=True, verbose=False)
    events_path = out_snirf.parent / out_snirf.name.replace("_nirs.snirf", "_events.tsv")
    annotations_to_df(raw_seg).to_csv(events_path, sep="\t", index=False)


def _setup_deriv_dir(derivatives_dir: Path, sub: str, ses: str | None) -> Path:
    out_nirs_dir = deriv_nirs_dir(derivatives_dir, _DERIV_NAME, sub, ses)
    ensure_dataset_description(derivatives_dir / _DERIV_NAME, _DERIV_NAME, "fnirs-prep crop")
    out_nirs_dir.mkdir(parents=True, exist_ok=True)
    return out_nirs_dir


def _crop_raw(
    raw,
    out_nirs_dir: Path,
    stem: str,
    *,
    tmin: float | None,
    tmax: float | None,
    segments_df: pd.DataFrame | None,
    combine: bool,
) -> list[Path]:
    import mne

    if segments_df is not None:
        segs = [
            raw.copy().crop(tmin=row.onset, tmax=row.onset + row.duration)
            for _, row in segments_df.iterrows()
        ]
        if combine:
            out = out_nirs_dir / f"{stem}_nirs.snirf"
            _write_segment(mne.concatenate_raws(segs), out)
            logger.info("Written combined: %s", out)
            return [out]
        out_paths: list[Path] = []
        for i, seg in enumerate(segs, start=1):
            out = out_nirs_dir / f"{stem}_seg-{i:02d}_nirs.snirf"
            _write_segment(seg, out)
            logger.info("Written segment %d: %s", i, out)
            out_paths.append(out)
        return out_paths

    cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    out = out_nirs_dir / f"{stem}_nirs.snirf"
    _write_segment(cropped, out)
    logger.info("Written: %s", out)
    return [out]


def crop_snirf_from_path(
    snirf_path: Path,
    derivatives_dir: Path,
    sub: str,
    ses: str | None = None,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    segments_df: pd.DataFrame | None = None,
    combine: bool = False,
) -> list[Path]:
    """Crop a SNIRF given its path directly (no BIDS layout lookup).

    Public entry point shared by the interface. Returns list of written SNIRF paths.
    """
    stem = bids_stem(snirf_path)
    out_nirs_dir = _setup_deriv_dir(derivatives_dir, sub, ses)
    copy_sidecars(snirf_path, stem, out_nirs_dir)
    raw = read_raw_snirf(snirf_path)
    return _crop_raw(raw, out_nirs_dir, stem,
                     tmin=tmin, tmax=tmax, segments_df=segments_df, combine=combine)


def crop_snirf(
    bids_dir: Path,
    derivatives_dir: Path,
    sub: str,
    ses: str | None = None,
    task: str | None = None,
    run: str | None = None,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    segments_path: Path | None = None,
    combine: bool = False,
    validate: bool = False,
) -> list[Path]:
    """Crop a raw SNIRF via BIDS layout lookup and write to derivatives/cropped/.

    Single segment: use tmin/tmax.
    Multi-segment: use segments_path (TSV with onset/duration columns).
    combine=True concatenates multi-segment output into one file.

    Returns list of written SNIRF paths.
    """
    snirf_path = find_snirf(bids_dir, sub, ses, task, run, validate=validate)
    stem = bids_stem(snirf_path)
    out_nirs_dir = _setup_deriv_dir(derivatives_dir, sub, ses)
    copy_sidecars(snirf_path, stem, out_nirs_dir)
    raw = read_raw_snirf(snirf_path)

    segments_df = pd.read_csv(segments_path, sep="\t") if segments_path is not None else None
    return _crop_raw(raw, out_nirs_dir, stem,
                     tmin=tmin, tmax=tmax, segments_df=segments_df, combine=combine)
