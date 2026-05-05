"""BIDS dataset querying and participants.tsv helpers."""

import json
from pathlib import Path

import pandas as pd
from bids import BIDSLayout


def get_layout(bids_dir: Path, validate: bool = True) -> BIDSLayout:
    """Return a pybids BIDSLayout for the dataset."""
    return BIDSLayout(str(bids_dir), validate=validate)


def get_participant_age(
    layout: BIDSLayout,
    subject: str,
    fallback: float | None = None,
) -> float:
    """Resolve participant age from participants.tsv; use fallback if missing.

    Raises ValueError if age is not found and no fallback is provided.
    """
    import pandas as pd

    tsv_path = Path(layout.root) / "participants.tsv"
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep="\t")
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
