"""Marker editing utilities: export events.tsv and apply edits back to SNIRF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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

logger = get_logger("pipeline.edit_markers")

_DERIV_NAME = "marker_edited"


def _df_to_annotations(df: pd.DataFrame):
    import mne
    return mne.Annotations(
        onset=df["onset"].to_numpy(),
        duration=df["duration"].to_numpy(),
        description=df["trial_type"].to_numpy(dtype=str),
    )


def export_markers(
    bids_dir: Path,
    sub: str,
    out_dir: Path,
    ses: str | None = None,
    task: str | None = None,
    run: str | None = None,
    validate: bool = False,
) -> Path:
    """Export a run's events.tsv to out_dir for manual editing.

    Copies the BIDS sidecar if present; otherwise extracts from SNIRF annotations.
    Returns the path to the exported TSV.
    """
    import shutil

    snirf_path = find_snirf(bids_dir, sub, ses, task, run, validate=validate)
    stem = bids_stem(snirf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_events.tsv"

    events_src = snirf_path.parent / f"{stem}_events.tsv"
    if events_src.exists():
        shutil.copy2(events_src, out_path)
        logger.info("Copied events TSV from BIDS: %s", events_src)
    else:
        raw = read_raw_snirf(snirf_path)
        annotations_to_df(raw).to_csv(out_path, sep="\t", index=False)
        logger.info("Extracted events from SNIRF annotations: %s", snirf_path)

    return out_path


def apply_markers(
    bids_dir: Path,
    derivatives_dir: Path,
    sub: str,
    ses: str | None = None,
    task: str | None = None,
    run: str | None = None,
    *,
    tsv: Path | None = None,
    shift: float | None = None,
    set_duration: float | None = None,
    rename: list[str] | None = None,
    validate: bool = False,
) -> Path:
    """Apply marker edits and write to derivatives/marker_edited/.

    Exactly one of tsv/shift/set_duration/rename must be provided.
    Returns the path to the written SNIRF.
    """
    import mne

    snirf_path = find_snirf(bids_dir, sub, ses, task, run, validate=validate)
    stem = bids_stem(snirf_path)

    out_nirs_dir = deriv_nirs_dir(derivatives_dir, _DERIV_NAME, sub, ses)
    ensure_dataset_description(
        derivatives_dir / _DERIV_NAME, _DERIV_NAME, "fnirs-prep edit-markers"
    )
    out_nirs_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecars(snirf_path, stem, out_nirs_dir)

    raw = read_raw_snirf(snirf_path)

    if tsv is not None:
        df = pd.read_csv(tsv, sep="\t")
        raw.set_annotations(_df_to_annotations(df))

    elif shift is not None:
        a = raw.annotations
        raw.set_annotations(mne.Annotations(
            onset=np.maximum(0.0, a.onset + shift),
            duration=a.duration,
            description=a.description,
        ))

    elif set_duration is not None:
        a = raw.annotations
        raw.set_annotations(mne.Annotations(
            onset=a.onset,
            duration=np.full(len(a), set_duration),
            description=a.description,
        ))

    elif rename is not None:
        rename_map = dict(r.split(":", 1) for r in rename)
        a = raw.annotations
        raw.set_annotations(mne.Annotations(
            onset=a.onset,
            duration=a.duration,
            description=[rename_map.get(d, d) for d in a.description],
        ))

    out_snirf = out_nirs_dir / f"{stem}_nirs.snirf"
    mne.export.export_raw(str(out_snirf), raw, fmt="snirf", overwrite=True, verbose=False)
    annotations_to_df(raw).to_csv(out_nirs_dir / f"{stem}_events.tsv", sep="\t", index=False)

    logger.info("Written to %s", out_nirs_dir)
    return out_snirf
