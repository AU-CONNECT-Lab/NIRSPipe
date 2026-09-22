"""Shared helpers for prep-stage pipeline modules (crop, edit_markers, etc.)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

NIRS_SIDECAR_SUFFIXES = ["_nirs.json", "_channels.tsv"]
NIRS_MONTAGE_SUFFIXES = ["_optodes.tsv", "_coordsystem.json"]


def find_snirf(
    bids_dir: Path,
    sub: str,
    ses: str | None,
    task: str | None,
    run: str | None,
    validate: bool = False,
    desc: str | None = None,
) -> Path:
    """The one SNIRF matching the entities, raising if there is none or several.

    ``desc`` selects a pipeline stage inside a derivatives tree, e.g. "errts". Matched on
    the filename rather than through the layout, which is how the post-processing loop
    already finds its inputs: the layout is built without a derivatives config, so a
    ``desc=`` query returns nothing.
    """
    from fnirs_pipe.io.bids import get_layout
    layout = get_layout(bids_dir, validate=validate)
    kwargs: dict = {"subject": sub, "extension": ".snirf"}
    if ses:  kwargs["session"] = ses
    if task: kwargs["task"] = task
    if run:  kwargs["run"] = run
    files = [Path(f.path) for f in layout.get(**kwargs)]
    if desc:
        files = [f for f in files if f"desc-{desc}_" in f.name]
    if not files:
        raise FileNotFoundError(
            f"No SNIRF found for sub-{sub} ses={ses} task={task} run={run}"
            + (f" desc={desc}" if desc else "")
        )
    if len(files) > 1:
        paths = "\n".join(f"  {f}" for f in files)
        hint = "--ses/--task/--run" if desc else "--ses/--task/--run/--input-desc"
        raise ValueError(
            f"Multiple SNIRFs found; narrow down with {hint}:\n{paths}"
        )
    return files[0]


def bids_stem(snirf_path: Path) -> str:
    return snirf_path.name.replace("_nirs.snirf", "")


def read_raw_snirf(snirf_path: Path):
    import mne
    return mne.io.read_raw_snirf(str(snirf_path), preload=True, verbose=False)


def annotations_to_df(raw) -> pd.DataFrame:
    # first_time is nonzero only for a cropped recording, whose annotations still sit on
    # the original axis. BIDS wants onsets relative to the start of the data either way.
    a = raw.annotations
    return pd.DataFrame({
        "onset": a.onset - raw.first_time,
        "duration": a.duration,
        "trial_type": a.description,
    })


def deriv_nirs_dir(derivatives_dir: Path, deriv_name: str, sub: str, ses: str | None) -> Path:
    parts = [f"sub-{sub}"]
    if ses:
        parts.append(f"ses-{ses}")
    parts.append("nirs")
    return derivatives_dir / deriv_name / Path(*parts)


def ensure_dataset_description(deriv_root: Path, name: str, generated_by: str) -> None:
    from fnirs_pipe import __version__

    desc_path = deriv_root / "dataset_description.json"
    if desc_path.exists():
        return
    deriv_root.mkdir(parents=True, exist_ok=True)
    desc_path.write_text(json.dumps({
        "Name": name,
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": generated_by, "Version": __version__}],
    }, indent=2))


def copy_dataset_root(bids_dir: Path, deriv_root: Path) -> None:
    """Carry participants.tsv and README across so a derivative stands on its own as BIDS.

    Creates the destination: one caller reaches here before anything else has made it, and
    a dataset root with a participants.tsv would otherwise fail on the copy.
    """
    for name in ("participants.tsv", "participants.json", "README"):
        src = bids_dir / name
        if src.exists():
            deriv_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, deriv_root / name)


def copy_sidecars(snirf_path: Path, stem: str, dest_dir: Path, dest_stem: str | None = None) -> None:
    """Copy a run's sidecars, optionally renaming them onto dest_stem.

    A sidecar belongs to the file whose name it shares, so an output written under a
    different stem than its source needs its sidecars renamed to match or they describe
    a file that is not there. The montage pair is the exception: it describes the cap
    rather than the run, carries no task entity, and keeps its name.
    """
    for suffix in NIRS_SIDECAR_SUFFIXES:
        src = snirf_path.parent / f"{stem}{suffix}"
        if src.exists():
            shutil.copy2(src, dest_dir / f"{dest_stem or stem}{suffix}")

    subject_stem = "_".join(p for p in stem.split("_") if p.startswith(("sub-", "ses-")))
    for suffix in NIRS_MONTAGE_SUFFIXES:
        src = snirf_path.parent / f"{subject_stem}{suffix}"
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)
