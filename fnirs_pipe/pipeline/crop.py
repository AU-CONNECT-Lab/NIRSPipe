"""Raw SNIRF cropping: single-segment and multi-segment to BIDS derivatives."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from fnirs_pipe.io.snirf import write_snirf
from fnirs_pipe.io.tables import read_table
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.utils.snirf_prep import (
    annotations_to_df,
    bids_stem,
    copy_dataset_root,
    copy_sidecars,
    deriv_nirs_dir,
    ensure_dataset_description,
    find_snirf,
    read_raw_snirf,
)

logger = get_logger("pipeline.crop")

_DERIV_NAME = "cropped"


def _write_segment(raw_seg, out_snirf: Path) -> None:
    write_snirf(raw_seg, out_snirf)
    events_path = out_snirf.parent / out_snirf.name.replace("_nirs.snirf", "_events.tsv")
    annotations_to_df(raw_seg).to_csv(events_path, sep="\t", index=False)


def _setup_deriv_dir(derivatives_dir: Path, sub: str, ses: str | None) -> Path:
    out_nirs_dir = deriv_nirs_dir(derivatives_dir, _DERIV_NAME, sub, ses)
    ensure_dataset_description(derivatives_dir / _DERIV_NAME, _DERIV_NAME, "fnirs-prep crop")
    out_nirs_dir.mkdir(parents=True, exist_ok=True)
    return out_nirs_dir


def _crop_raw(
    raw,
    snirf_path: Path,
    out_nirs_dir: Path,
    stem: str,
    *,
    tmin: float | None,
    tmax: float | None,
    segments_df: pd.DataFrame | None,
    combine: bool,
) -> list[Path]:
    import mne

    def _write(seg, name: str) -> Path:
        out = out_nirs_dir / f"{name}_nirs.snirf"
        _write_segment(seg, out)
        copy_sidecars(snirf_path, stem, out_nirs_dir, name)
        return out

    if segments_df is not None:
        segs = [
            raw.copy().crop(tmin=row.onset, tmax=row.onset + row.duration)
            for _, row in segments_df.iterrows()
        ]
        if combine:
            out = _write(mne.concatenate_raws(segs), stem)
            logger.info("Written combined: %s", out)
            return [out]
        tasks = segments_df["task"] if "task" in segments_df.columns else None
        out_paths: list[Path] = []
        for i, seg in enumerate(segs, start=1):
            if tasks is None:
                name = f"{stem}_seg-{i:02d}"
            else:
                name = re.sub(r"task-[^_]+", f"task-{tasks.iloc[i - 1]}", stem)
            out = _write(seg, name)
            logger.info("Written segment %d: %s", i, out)
            out_paths.append(out)
        return out_paths

    out = _write(raw.copy().crop(tmin=tmin, tmax=tmax), stem)
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
    raw = read_raw_snirf(snirf_path)
    return _crop_raw(raw, snirf_path, out_nirs_dir, stem,
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
    Multi-segment: use segments_path (table with onset/duration columns).
    An optional `task` column names each segment, and its output takes that task entity
    instead of `_seg-NN`, which makes the segments separate tasks of a valid BIDS dataset
    rather than one task the pipeline cannot tell apart.
    combine=True concatenates multi-segment output into one file.

    Returns list of written SNIRF paths.
    """
    snirf_path = find_snirf(bids_dir, sub, ses, task, run, validate=validate)
    stem = bids_stem(snirf_path)
    out_nirs_dir = _setup_deriv_dir(derivatives_dir, sub, ses)
    copy_dataset_root(bids_dir, derivatives_dir / _DERIV_NAME)
    raw = read_raw_snirf(snirf_path)

    segments_df = read_table(segments_path) if segments_path is not None else None
    return _crop_raw(raw, snirf_path, out_nirs_dir, stem,
                     tmin=tmin, tmax=tmax, segments_df=segments_df, combine=combine)
