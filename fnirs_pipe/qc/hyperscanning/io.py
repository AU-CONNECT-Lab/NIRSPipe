from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd

from fnirs_pipe.exceptions import GroupCSVError
from fnirs_pipe.io.derivatives import find_preproc_snirf
from fnirs_pipe.utils import load_toml


@dataclass
class GroupEntry:
    group_id: str
    subject_id: str
    task: str


def parse_group_csv(csv_path: Path) -> dict[tuple[str, str], list[GroupEntry]]:
    """Parse group CSV into {(group_id, task): [GroupEntry, ...]}."""
    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise GroupCSVError(f"Cannot read CSV {csv_path}: {exc}") from exc

    missing = {"group_id", "subject_id", "task"} - set(df.columns)
    if missing:
        raise GroupCSVError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=["group_id", "subject_id", "task"])
    if df.empty:
        raise GroupCSVError("CSV contains no valid rows after dropping NaN values")

    result: dict[tuple[str, str], list[GroupEntry]] = {}
    for _, row in df.iterrows():
        key = (str(row["group_id"]).strip(), str(row["task"]).strip())
        entry = GroupEntry(
            group_id=str(row["group_id"]).strip(),
            subject_id=str(row["subject_id"]).strip(),
            task=str(row["task"]).strip(),
        )
        result.setdefault(key, []).append(entry)

    for (gid, task), members in result.items():
        if len(members) < 2:
            raise GroupCSVError(
                f"Group '{gid}' task '{task}' has only {len(members)} subject — need at least 2"
            )

    return result


def load_group_iqm(output_dir: Path, group: list[GroupEntry]) -> dict[str, dict]:
    """Load per-subject IQM data from TOML scalars + channel metrics CSV.

    Returns {subject_id: iqm_dict}.
    sci_per_channel is populated from the channel CSV when available.
    """
    result: dict[str, dict] = {}
    for entry in group:
        nirs_dir = output_dir / entry.subject_id / "nirs"
        toml_path = nirs_dir / f"{entry.subject_id}_iqm.toml"
        csv_path = nirs_dir / f"{entry.subject_id}_channel_metrics.csv"

        iqm: dict = {}
        if toml_path.exists():
            iqm.update(load_toml(toml_path))

        if csv_path.exists():
            try:
                ch_df = pd.read_csv(csv_path)
                if {"name", "sci"}.issubset(ch_df.columns):
                    iqm["sci_per_channel"] = dict(
                        zip(ch_df["name"].astype(str), pd.to_numeric(ch_df["sci"], errors="coerce"))
                    )
                if "is_bad" in ch_df.columns:
                    iqm["bad_channels"] = ch_df.loc[
                        ch_df["is_bad"].astype(str).str.lower().isin({"true", "1"}), "name"
                    ].tolist()
            except Exception:
                pass

        result[entry.subject_id] = iqm

    return result


def load_group_haemo(output_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load desc-preproc haemo snirf for each subject.

    Returns {subject_id: raw_haemo}.
    Raises MissingDerivativesError if any snirf is absent.
    """
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        snirf_path = find_preproc_snirf(output_dir, entry.subject_id, entry.task)
        result[entry.subject_id] = mne.io.read_raw_snirf(
            str(snirf_path), preload=True, verbose=False
        )
    return result
