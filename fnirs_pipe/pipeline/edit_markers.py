"""Marker editing utilities: export events.tsv and apply edits back to SNIRF."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.io.bids import get_layout
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("pipeline.edit_markers")

_DERIV_NAME = "marker_edited"
_COPY_SUFFIXES = ["_nirs.json", "_channels.tsv", "_optodes.tsv", "_coordsystem.json"]


def _find_snirf(
    bids_dir: Path,
    sub: str,
    ses: str | None,
    task: str | None,
    run: str | None,
    validate: bool = False,
) -> Path:
    layout = get_layout(bids_dir, validate=validate)
    kwargs: dict = {"subject": sub, "extension": ".snirf"}
    if ses:  kwargs["session"] = ses
    if task: kwargs["task"] = task
    if run:  kwargs["run"] = run
    files = [Path(f.path) for f in layout.get(**kwargs)]
    if not files:
        raise FileNotFoundError(
            f"No SNIRF found for sub-{sub} ses={ses} task={task} run={run}"
        )
    if len(files) > 1:
        paths = "\n".join(f"  {f}" for f in files)
        raise ValueError(
            f"Multiple SNIRFs found; narrow down with --ses/--task/--run:\n{paths}"
        )
    return files[0]


def _bids_stem(snirf_path: Path) -> str:
    return snirf_path.name.replace("_nirs.snirf", "")


def _read_raw(snirf_path: Path):
    import mne
    return mne.io.read_raw_snirf(str(snirf_path), preload=True, verbose=False)


def _annotations_to_df(raw) -> pd.DataFrame:
    a = raw.annotations
    return pd.DataFrame({
        "onset": a.onset,
        "duration": a.duration,
        "trial_type": a.description,
    })


def _df_to_annotations(df: pd.DataFrame):
    import mne
    return mne.Annotations(
        onset=df["onset"].to_numpy(),
        duration=df["duration"].to_numpy(),
        description=df["trial_type"].to_numpy(dtype=str),
    )


def _deriv_nirs_dir(derivatives_dir: Path, sub: str, ses: str | None) -> Path:
    parts = [f"sub-{sub}"]
    if ses:
        parts.append(f"ses-{ses}")
    parts.append("nirs")
    return derivatives_dir / _DERIV_NAME / Path(*parts)


def _ensure_dataset_description(deriv_root: Path) -> None:
    desc_path = deriv_root / "dataset_description.json"
    if desc_path.exists():
        return
    deriv_root.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": _DERIV_NAME,
        "BIDSVersion": "1.7.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "fnirs-prep edit-markers"}],
    }
    desc_path.write_text(json.dumps(desc, indent=2))


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
    snirf_path = _find_snirf(bids_dir, sub, ses, task, run, validate=validate)
    stem = _bids_stem(snirf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_events.tsv"

    events_src = snirf_path.parent / f"{stem}_events.tsv"
    if events_src.exists():
        shutil.copy2(events_src, out_path)
        logger.info("Copied events TSV from BIDS: %s", events_src)
    else:
        raw = _read_raw(snirf_path)
        _annotations_to_df(raw).to_csv(out_path, sep="\t", index=False)
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

    snirf_path = _find_snirf(bids_dir, sub, ses, task, run, validate=validate)
    stem = _bids_stem(snirf_path)

    deriv_nirs_dir = _deriv_nirs_dir(derivatives_dir, sub, ses)
    _ensure_dataset_description(derivatives_dir / _DERIV_NAME)
    deriv_nirs_dir.mkdir(parents=True, exist_ok=True)

    for suffix in _COPY_SUFFIXES:
        src = snirf_path.parent / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, deriv_nirs_dir / src.name)

    raw = _read_raw(snirf_path)

    if tsv is not None:
        df = pd.read_csv(tsv, sep="\t")
        raw.set_annotations(_df_to_annotations(df))

    elif shift is not None:
        a = raw.annotations
        new_onsets = np.maximum(0.0, a.onset + shift)
        raw.set_annotations(
            mne.Annotations(onset=new_onsets, duration=a.duration, description=a.description)
        )

    elif set_duration is not None:
        a = raw.annotations
        raw.set_annotations(
            mne.Annotations(
                onset=a.onset,
                duration=np.full(len(a), set_duration),
                description=a.description,
            )
        )

    elif rename is not None:
        rename_map = dict(r.split(":", 1) for r in rename)
        a = raw.annotations
        new_desc = [rename_map.get(d, d) for d in a.description]
        raw.set_annotations(
            mne.Annotations(onset=a.onset, duration=a.duration, description=new_desc)
        )

    out_snirf = deriv_nirs_dir / f"{stem}_nirs.snirf"
    mne.export.export_raw(str(out_snirf), raw, fmt="snirf", overwrite=True, verbose=False)

    out_events = deriv_nirs_dir / f"{stem}_events.tsv"
    _annotations_to_df(raw).to_csv(out_events, sep="\t", index=False)

    logger.info("Written to %s", deriv_nirs_dir)
    return out_snirf
