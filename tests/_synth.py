"""Synthetic BIDS fNIRS datasets, generated at test time.

Nothing here is recorded data. Every property a test depends on is constructed
rather than hoped for: a dataset asked for a bad channel really has one, at a
known index, because that pair's two wavelengths are filled with uncorrelated
noise and therefore cannot correlate in the cardiac band. Real recordings only
sometimes contain the situation a test is about, which is why the PSD bad-channel
bug survived so long.

The SNIRF files are written through the pipeline's own ``write_snirf``, so the
round trip a test exercises is the real one.

    bids = make_bids_dataset(tmp_path)              # 2 subjects x 2 tasks
    bids, pairs = make_hyper_dataset(tmp_path)      # 1 dyad x 2 tasks + pairs.csv

Two requirements of ``write_snirf`` are easy to miss when building a Raw by hand:
``meas_date`` and ``subject_info`` must both be set, or it fails inside mne_nirs
with an error that names neither.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np

SFREQ = 10.0
# A 0.01 Hz high-pass, the lowest cutoff in use, builds a 3301-sample FIR at 10 Hz.
# Anything shorter than that filters with visible distortion and a RuntimeWarning.
DURATION = 400.0
WAVELENGTHS = (760.0, 850.0)

N_LONG_PAIRS = 4          # 3 cm source-detector separation
LONG_DISTANCE = 0.03
SHORT_DISTANCE = 0.008    # below MNE's 1 cm short-channel threshold

CARDIAC_FREQ = 1.2        # inside the 0.7-1.5 Hz band the SCI is computed over
BAD_PAIR = 2              # 0-based index of the long pair with no wavelength coupling

_TASKS_WITH_EVENTS = ("tapping", "hold", "nohold")


def _channel_layout(n_long: int, short: bool) -> tuple[list[str], list[np.ndarray], list[bool]]:
    """Names, MNE loc arrays, and an is-short flag per channel (two per pair)."""
    names: list[str] = []
    locs: list[np.ndarray] = []
    is_short: list[bool] = []

    pairs = [(i, LONG_DISTANCE) for i in range(n_long)]
    if short:
        pairs.append((n_long, SHORT_DISTANCE))

    for idx, distance in pairs:
        src = np.array([idx * 0.03, 0.0, 0.0])
        det = src + np.array([distance, 0.0, 0.0])
        for wavelength in WAVELENGTHS:
            loc = np.zeros(12)
            loc[0:3] = (src + det) / 2
            loc[3:6] = src
            loc[6:9] = det
            loc[9] = wavelength
            names.append(f"S{idx + 1}_D{idx + 1} {wavelength:.0f}")
            locs.append(loc)
            is_short.append(distance < 0.01)

    return names, locs, is_short


def _events(task: str, duration: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Block design for task runs, nothing at all for rest."""
    if task not in _TASKS_WITH_EVENTS:
        return np.array([]), np.array([]), []
    onsets = np.arange(20.0, duration - 20.0, 40.0)
    return onsets, np.full(onsets.shape, 10.0), [task] * len(onsets)


def synth_raw(
    subject: str,
    task: str,
    duration: float = DURATION,
    n_long_pairs: int = N_LONG_PAIRS,
    short_channels: bool = True,
    bad_pair: int | None = BAD_PAIR,
    motion_onset: float | None = 150.0,
    seed: int | None = None,
) -> mne.io.Raw:
    """One synthetic raw-intensity recording.

    Good pairs share a cardiac oscillation across their two wavelengths, so their
    scalp coupling index lands near 0.9. ``bad_pair`` gets independent noise in
    each wavelength instead, which drives its SCI to roughly 0.05, below any
    sensible threshold. Pass ``bad_pair=None`` for a clean recording.
    """
    # crc32, not hash(): str hashing is salted per process, so hash() would make the
    # data differ between runs and no golden baseline could ever be taken.
    if seed is None:
        seed = zlib.crc32(f"{subject}/{task}".encode())
    rng = np.random.default_rng(seed)
    n = int(SFREQ * duration)
    t = np.arange(n) / SFREQ

    names, locs, is_short = _channel_layout(n_long_pairs, short_channels)
    cardiac = np.sin(2 * np.pi * CARDIAC_FREQ * t)
    onsets, durations, labels = _events(task, duration)

    # Boxcar response at the events, so GLM has something to fit.
    response = np.zeros(n)
    for onset, dur in zip(onsets, durations):
        response[int(onset * SFREQ):int((onset + dur) * SFREQ)] = 1.0

    data = np.empty((len(names), n))
    for i, name in enumerate(names):
        pair = int(name.split("_")[0][1:]) - 1
        coupling = rng.normal(size=n) if pair == bad_pair else cardiac
        signal = 1.0 + 0.02 * coupling + 0.005 * rng.normal(size=n)
        if not is_short[i]:
            signal += 0.01 * response          # short channels carry no brain response
        signal += 0.004 * np.sin(2 * np.pi * 0.25 * t)      # respiration
        data[i] = 0.1 * signal

    if motion_onset is not None and motion_onset < duration:
        # A step, not a spike: this is the shape motion correction is meant to repair.
        data[:, int(motion_onset * SFREQ):] += 0.1 * 0.05

    info = mne.create_info(names, SFREQ, ["fnirs_cw_amplitude"] * len(names))
    for ch, loc in zip(info["chs"], locs):
        ch["loc"] = loc

    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.info["subject_info"] = {"first_name": "sub", "last_name": subject}
    raw.set_annotations(mne.Annotations(onsets, durations, labels))
    return raw


def _write_subject(bids_dir: Path, subject: str, task: str, raw: mne.io.Raw) -> Path:
    from fnirs_pipe.io.snirf import write_snirf

    nirs_dir = bids_dir / f"sub-{subject}" / "nirs"
    stem = f"sub-{subject}_task-{task}_nirs"
    path = nirs_dir / f"{stem}.snirf"
    write_snirf(raw, path)

    n_pairs = len(raw.ch_names) // len(WAVELENGTHS)
    (nirs_dir / f"{stem}.json").write_text(json.dumps({
        "SamplingFrequency": raw.info["sfreq"],
        "NIRSChannelCount": len(raw.ch_names),
        "NIRSSourceOptodeCount": n_pairs,
        "NIRSDetectorOptodeCount": n_pairs,
        "TaskName": task,
    }, indent=2))

    ann = raw.annotations
    if len(ann):
        rows = ["onset\tduration\ttrial_type"]
        rows += [f"{o:.3f}\t{d:.3f}\t{desc}"
                 for o, d, desc in zip(ann.onset, ann.duration, ann.description)]
        (nirs_dir / f"sub-{subject}_task-{task}_events.tsv").write_text("\n".join(rows) + "\n")

    return path


def _write_dataset_root(bids_dir: Path, subjects: list[str]) -> None:
    bids_dir.mkdir(parents=True, exist_ok=True)
    (bids_dir / "dataset_description.json").write_text(json.dumps({
        "Name": "synthetic fnirs test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
    }, indent=2))
    rows = ["participant_id"] + [f"sub-{s}" for s in subjects]
    (bids_dir / "participants.tsv").write_text("\n".join(rows) + "\n")


def make_bids_dataset(
    root: Path,
    subjects: tuple[str, ...] = ("01", "02"),
    tasks: tuple[str, ...] = ("tapping", "rest"),
    name: str = "bids",
    **raw_kwargs,
) -> Path:
    """Write a BIDS dataset under root/name and return its path.

    Defaults give two subjects and two tasks: ``tapping`` carries events, ``rest``
    carries none, which is the recordings-without-triggers case.
    """
    bids_dir = Path(root) / name
    _write_dataset_root(bids_dir, list(subjects))
    for subject in subjects:
        for task in tasks:
            _write_subject(bids_dir, subject, task,
                           synth_raw(subject, task, **raw_kwargs))
    return bids_dir


def make_hyper_dataset(
    root: Path,
    groups: dict[str, tuple[str, str]] | None = None,
    tasks: tuple[str, ...] = ("hold", "rest"),
    name: str = "bids_hyper",
    **raw_kwargs,
) -> tuple[Path, Path]:
    """Write a dyad dataset plus its pairs CSV. Returns (bids_dir, pairs_csv).

    ``hold`` carries triggers so alignment has something to align on; ``rest``
    carries none, which is the case that currently fails alignment outright.
    """
    groups = groups or {"1003": ("10031", "10032")}
    subjects = [s for members in groups.values() for s in members]

    bids_dir = Path(root) / name
    _write_dataset_root(bids_dir, subjects)
    for subject in subjects:
        for task in tasks:
            _write_subject(bids_dir, subject, task,
                           synth_raw(subject, task, **raw_kwargs))

    pairs_csv = Path(root) / f"{name}_pairs.csv"
    rows = ["group_id,subject_id,task"]
    for group_id, members in groups.items():
        for subject in members:
            rows += [f"{group_id},sub-{subject},{task}" for task in tasks]
    pairs_csv.write_text("\n".join(rows) + "\n")

    return bids_dir, pairs_csv
