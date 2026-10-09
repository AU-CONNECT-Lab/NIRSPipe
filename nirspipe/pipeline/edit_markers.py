"""Marker editing utilities: export events.tsv and apply edits back to SNIRF."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import mne

from nirspipe.io.auxiliary import write_aux_window
from nirspipe.io.derivatives import dataset_root_of, write_dataset_description
from nirspipe.io.snirf import write_snirf
from nirspipe.io.tables import read_table, write_tsv
from nirspipe.utils.logging import get_logger
from nirspipe.utils.snirf_prep import (
    annotations_to_df,
    bids_stem,
    copy_dataset_root,
    copy_sidecars,
    deriv_nirs_dir,
    find_snirf,
    read_raw_snirf,
)

logger = get_logger("pipeline.edit_markers")

_DERIV_NAME = "marker_edited"


def _df_to_annotations(df: pd.DataFrame):
    return mne.Annotations(
        onset=df["onset"].to_numpy(),
        duration=df["duration"].to_numpy(),
        description=df["trial_type"].to_numpy(dtype=str),
    )


def apply_markers_from_df(
    snirf_path: Path,
    derivatives_dir: Path,
    sub: str,
    ses: str | None,
    df: pd.DataFrame,
) -> Path:
    """Read snirf_path, replace annotations with df, write to derivatives/marker_edited/.

    Public entry point shared by the CLI (via apply_markers) and the interface callbacks.
    Returns the path to the written SNIRF.
    """
    stem = bids_stem(snirf_path)
    out_nirs_dir = deriv_nirs_dir(derivatives_dir, _DERIV_NAME, sub, ses)
    write_dataset_description(derivatives_dir / _DERIV_NAME, name=_DERIV_NAME,
                              generated_by="nirspipe-prep edit-markers",
                              source=dataset_root_of(snirf_path))
    out_nirs_dir.mkdir(parents=True, exist_ok=True)
    copy_sidecars(snirf_path, stem, out_nirs_dir)

    raw = read_raw_snirf(snirf_path)
    raw.set_annotations(_df_to_annotations(df))

    out_snirf = out_nirs_dir / f"{stem}_nirs.snirf"
    write_snirf(raw, out_snirf)
    # write_snirf goes through the Raw, which has nowhere to hold an aux channel, so the
    # output would carry none. Nothing is cut here, so the window is the whole recording.
    names = write_aux_window(snirf_path, out_snirf, [(0.0, float(raw.times[-1]))])
    if names:
        logger.info("Carried %d aux channels into %s", len(names), out_snirf.name)
    write_tsv(annotations_to_df(raw), out_nirs_dir / f"{stem}_events.tsv")

    logger.info("Written to %s", out_nirs_dir)
    return out_snirf


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
        write_tsv(annotations_to_df(raw), out_path)
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
    snirf_path = find_snirf(bids_dir, sub, ses, task, run, validate=validate)

    if tsv is not None:
        df = read_table(tsv)
    else:
        raw = read_raw_snirf(snirf_path)
        a = raw.annotations

        if shift is not None:
            df = pd.DataFrame({
                "onset":      np.maximum(0.0, a.onset + shift),
                "duration":   a.duration,
                "trial_type": a.description,
            })
        elif set_duration is not None:
            df = pd.DataFrame({
                "onset":      a.onset,
                "duration":   np.full(len(a), set_duration),
                "trial_type": a.description,
            })
        elif rename is not None:
            rename_map = dict(r.split(":", 1) for r in rename)
            df = pd.DataFrame({
                "onset":      a.onset,
                "duration":   a.duration,
                "trial_type": [rename_map.get(d, d) for d in a.description],
            })

    copy_dataset_root(bids_dir, derivatives_dir / _DERIV_NAME)
    return apply_markers_from_df(snirf_path, derivatives_dir, sub, ses, df)
