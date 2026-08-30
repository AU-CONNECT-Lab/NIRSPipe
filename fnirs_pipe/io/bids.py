"""BIDS dataset querying and participants.tsv helpers."""

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from bids import BIDSLayout


def get_layout(bids_dir: Path, validate: bool = True) -> BIDSLayout:
    """Return a pybids BIDSLayout for the dataset."""
    return BIDSLayout(str(bids_dir), validate=validate)


def write_bids_from_snirf(
    input_file: Path,
    bids_dir: Path,
    subject: str,
    task: str,
    session: str | None = None,
    run: str | None = None,
    overwrite: bool = False,
    optode_frame: str = "unknown",
) -> None:
    """Convert one raw snirf file into a BIDS nirs entry.

    `optode_frame` names the space the SNIRF's optode coordinates live in. SNIRF does not
    record it, so MNE reads them as "unknown" and mne-bids then writes no `_optodes.tsv` or
    `_coordsystem.json`, both of which BIDS requires: it will not guess a frame on the
    author's behalf. Pass "head" when the positions were digitised against the nasion and
    the two preauricular points, "mri" when they are in a subject's MRI space. Leave it
    unknown rather than asserting a frame the positions were never measured in.
    """
    import mne
    import mne_bids

    raw = mne.io.read_raw_snirf(str(input_file), optode_frame=optode_frame, preload=False)
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task=task,
        session=session,
        run=run,
        root=bids_dir,
        datatype="nirs",
        suffix="nirs",
        extension=".snirf",
    )
    mne_bids.write_raw_bids(raw, bids_path=bids_path, overwrite=overwrite, format="auto")


def get_participant_age(
    layout: BIDSLayout,
    subject: str,
    fallback: float | None = None,
) -> float:
    """Resolve participant age from participants.tsv; use fallback if missing.

    Raises ValueError if age is not found and no fallback is provided.
    """
    from fnirs_pipe.io.tables import read_table

    tsv_path = Path(layout.root) / "participants.tsv"
    if tsv_path.exists():
        df = read_table(tsv_path)
        row = df[df["participant_id"] == f"sub-{subject}"]
        if not row.empty and "age" in df.columns:
            return float(row["age"].iloc[0])

    if fallback is not None:
        return float(fallback)

    raise ValueError(
        f"Age not found for sub-{subject} in participants.tsv and no --age fallback given."
    )


def get_nirs_files(
    layout: BIDSLayout,
    subject: str,
    session: str | None = None,
    task: str | None = None,
    filter_file: Path | None = None,
) -> list[Path]:
    """Return paths to all snirf files for a given subject/session/task.

    task: if None, all tasks are returned.
    filter_file: optional JSON with extra pybids query kwargs, e.g. {"run": "01"}.
    """
    kwargs: dict = {"subject": subject, "extension": ".snirf"}

    if session is not None:
        kwargs["session"] = session

    if task is not None:
        kwargs["task"] = task

    if filter_file is not None:
        extra = json.loads(filter_file.read_text())
        kwargs.update(extra)

    return [Path(f.path) for f in layout.get(**kwargs)]


def bids_label(subject: str, entities: dict) -> str:
    """Build a BIDS filename stem (sub-.._ses-.._task-.._run-..) from parsed entities."""
    parts = [f"sub-{subject}"]
    for key, prefix in (("session", "ses"), ("task", "task"), ("run", "run")):
        val = entities.get(key)
        if val:
            parts.append(f"{prefix}-{val}")
    return "_".join(parts)


def iter_run_files(
    layout: BIDSLayout,
    subjects: list[str],
    sessions: list[str | None],
    task: str | None,
) -> Iterator[tuple[Path, str]]:
    """Yield (snirf_path, bids_label) for each subject × session run matching task."""
    for sub in subjects:
        for ses in sessions:
            for f in get_nirs_files(layout, subject=sub, session=ses, task=task):
                entities = layout.parse_file_entities(str(f))
                yield f, bids_label(sub, entities)
