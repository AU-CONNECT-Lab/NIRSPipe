"""Hyperscanning pipeline utilities: group IO, alignment, and signal metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import coherence

from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
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


def load_group_haemo(output_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load desc-preproc haemo SNIRF for each subject.

    Returns {subject_id: raw_haemo}.
    Raises MissingDerivativesError if any SNIRF is absent.
    """
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        snirf_path = find_preproc_snirf(output_dir, entry.subject_id, entry.task)
        result[entry.subject_id] = mne.io.read_raw_snirf(
            str(snirf_path), preload=True, verbose=False
        )
    return result


def load_group_raw_bids(bids_dir: Path, group: list[GroupEntry]) -> dict[str, mne.io.Raw]:
    """Load raw CW-amplitude SNIRF from BIDS for each group member.

    Returns {subject_id: raw_intensity}.
    Raises MissingDerivativesError if no SNIRF is found for any member.
    """
    from fnirs_pipe.io.bids import get_layout, get_nirs_files

    layout = get_layout(bids_dir, validate=False)
    result: dict[str, mne.io.Raw] = {}
    for entry in group:
        sub_label = entry.subject_id.removeprefix("sub-")
        files = get_nirs_files(layout, subject=sub_label, task=entry.task)
        if not files:
            raise MissingDerivativesError(
                f"No SNIRF found in BIDS for {entry.subject_id} task-{entry.task}"
            )
        result[entry.subject_id] = mne.io.read_raw_snirf(
            str(files[0]), preload=True, verbose=False
        )
    return result


def _raw_to_haemo(raw: mne.io.Raw, dpf: float = 6.0) -> mne.io.Raw:
    raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
    return mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)


def compute_group_iqm_raw(
    group: list[GroupEntry],
    raws: dict[str, mne.io.Raw],
    sci_threshold: float,
    output_dir: Path,
) -> dict[str, dict]:
    """Compute raw-level IQM (SCI, bad channels) for each group member.

    Writes two TSVs to output_dir following BIDS/MRIQC conventions:
      group-{gid}_task-{task}_hyper-raw_iqm.tsv      — one row per subject (scalars)
      group-{gid}_task-{task}_hyper-raw_channels.tsv  — one row per subject × channel

    Returns {subject_id: iqm_dict} for use in the HTML report.
    """
    from fnirs_pipe.qc.quantitative_metrics import compute_raw_iqm

    gid  = group[0].group_id
    task = group[0].task
    output_dir.mkdir(parents=True, exist_ok=True)

    iqm_data: dict[str, dict] = {}
    scalar_rows: list[dict] = []
    channel_rows: list[dict] = []

    for entry in group:
        raw = raws[entry.subject_id]

        try:
            raw_od  = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
            sci_arr = mne.preprocessing.nirs.scalp_coupling_index(raw_od, verbose=False)
            sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
        except Exception:
            sci_scores = {ch: float("nan") for ch in raw.ch_names}

        bad_channels = [ch for ch, s in sci_scores.items() if s < sci_threshold]

        try:
            iqm = compute_raw_iqm(raw, sci_scores, bad_channels)
        except Exception:
            iqm = {}

        iqm["sci_per_channel"] = sci_scores
        iqm["bad_channels"]    = bad_channels
        iqm_data[entry.subject_id] = iqm

        scalar_rows.append({
            "group_id":               gid,
            "subject_id":             entry.subject_id,
            "task":                   task,
            "sci_mean":               iqm.get("sci_mean"),
            "n_bad_channels":         len(bad_channels),
            "channel_retention_rate": iqm.get("channel_retention_rate"),
        })

        for ch, sci_val in sci_scores.items():
            channel_rows.append({
                "group_id":   gid,
                "subject_id": entry.subject_id,
                "task":       task,
                "channel":    ch,
                "sci":        sci_val,
                "is_bad":     ch in bad_channels,
            })

    stem = f"group-{gid}_task-{task}_hyper-raw"
    pd.DataFrame(scalar_rows).to_csv(
        output_dir / f"{stem}_iqm.tsv", sep="\t", index=False
    )
    pd.DataFrame(channel_rows).to_csv(
        output_dir / f"{stem}_channels.tsv", sep="\t", index=False
    )

    return iqm_data


def align_recordings(
    raws: dict[str, mne.io.Raw],
    task: str,
) -> tuple[dict[str, mne.io.Raw], dict[str, float]]:
    """Align N recordings by the first shared annotation trigger.

    Crops each raw from its first occurrence of a common trigger description,
    then trims all to the same duration (shortest post-crop).

    Returns (aligned_raws, {subject_id: crop_offset_seconds}).
    Raises AlignmentError if no shared trigger exists across all subjects.
    """
    desc_sets: dict[str, set[str]] = {}
    for sub_id, raw in raws.items():
        desc_sets[sub_id] = {
            ann["description"]
            for ann in raw.annotations
            if not str(ann["description"]).upper().startswith("BAD")
        }

    if not any(desc_sets.values()):
        raise AlignmentError(
            "No non-BAD annotations found in any recording. "
            "Cannot align without shared trigger events."
        )

    common: set[str] = set.intersection(*desc_sets.values()) if desc_sets else set()
    if not common:
        all_descs = {sub: sorted(d) for sub, d in desc_sets.items()}
        raise AlignmentError(
            f"No shared trigger descriptions across all participants. "
            f"Per-subject descriptions: {all_descs}. "
            "Note: alignment requires a shared hardware trigger."
        )

    offsets: dict[str, float] = {}
    for sub_id, raw in raws.items():
        for ann in sorted(raw.annotations, key=lambda a: float(a["onset"])):
            if ann["description"] in common:
                offsets[sub_id] = float(ann["onset"])
                break
        if sub_id not in offsets:
            raise AlignmentError(f"Subject {sub_id}: no common trigger found (unexpected state)")

    aligned: dict[str, mne.io.Raw] = {}
    for sub_id, raw in raws.items():
        aligned[sub_id] = raw.copy().crop(tmin=offsets[sub_id])

    min_duration = min(r.times[-1] for r in aligned.values())
    for sub_id in list(aligned):
        aligned[sub_id].crop(tmax=min_duration)

    return aligned, offsets


def compute_pairwise_coherence(
    raws: dict[str, mne.io.Raw],
    fmin: float = 0.01,
    fmax: float = 0.10,
) -> pd.DataFrame:
    """Compute pairwise spectral coherence per HbO channel in [fmin, fmax] Hz.

    Returns DataFrame with columns: ch_name, sub1, sub2, coherence.
    Channels are matched by index; all subjects must share the same channel layout.
    """
    subject_ids = list(raws.keys())
    if len(subject_ids) < 2:
        raise ValueError("Need at least 2 subjects for pairwise coherence")

    ref_raw = raws[subject_ids[0]]
    sfreq = ref_raw.info["sfreq"]
    nperseg = min(512, max(64, ref_raw.n_times // 4))

    rows: list[dict] = []
    for sub1, sub2 in combinations(subject_ids, 2):
        raw1, raw2 = raws[sub1], raws[sub2]
        picks1 = mne.pick_types(raw1.info, fnirs="hbo")
        picks2 = mne.pick_types(raw2.info, fnirs="hbo")
        data1 = raw1.get_data(picks=picks1)
        data2 = raw2.get_data(picks=picks2)
        ch_names1 = [raw1.ch_names[p] for p in picks1]
        ch_names2 = [raw2.ch_names[p] for p in picks2]

        for i in range(min(len(picks1), len(picks2))):
            ch_label = ch_names1[i].rsplit(" ", 1)[0] if " " in ch_names1[i] else ch_names1[i]
            freqs, coh = coherence(data1[i], data2[i], fs=sfreq, nperseg=nperseg)
            mask = (freqs >= fmin) & (freqs <= fmax)
            mean_coh = float(np.mean(coh[mask])) if mask.any() else float("nan")
            rows.append({"ch_name": ch_label, "sub1": sub1, "sub2": sub2, "coherence": mean_coh})

    return pd.DataFrame(rows, columns=["ch_name", "sub1", "sub2", "coherence"])


def load_group_iqm(output_dir: Path, group: list[GroupEntry]) -> dict[str, dict]:
    """Load per-subject IQM scalars and channel metrics from derivatives.

    Returns {subject_id: iqm_dict}.
    """
    result: dict[str, dict] = {}
    for entry in group:
        nirs_dir = output_dir / entry.subject_id / "nirs"
        toml_path = nirs_dir / f"{entry.subject_id}_iqm.toml"
        csv_path  = nirs_dir / f"{entry.subject_id}_channel_metrics.csv"

        iqm: dict = {}
        if toml_path.exists():
            iqm.update(load_toml(toml_path))

        if csv_path.exists():
            try:
                ch_df = pd.read_csv(csv_path)
                if {"name", "sci"}.issubset(ch_df.columns):
                    iqm["sci_per_channel"] = dict(
                        zip(ch_df["name"].astype(str),
                            pd.to_numeric(ch_df["sci"], errors="coerce"))
                    )
                if "is_bad" in ch_df.columns:
                    iqm["bad_channels"] = ch_df.loc[
                        ch_df["is_bad"].astype(str).str.lower().isin({"true", "1"}), "name"
                    ].tolist()
            except Exception:
                pass

        result[entry.subject_id] = iqm

    return result
