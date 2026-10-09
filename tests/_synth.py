"""Synthetic BIDS fNIRS datasets, generated at test time.

Nothing here is recorded data. Every property a test depends on is constructed
rather than hoped for: a dataset asked for a bad channel really has one, at a
known index, because that pair's two wavelengths are filled with uncorrelated
noise and therefore cannot correlate in the cardiac band. Real recordings only
sometimes contain the situation a test is about.

The SNIRF files are written through the pipeline's own ``write_snirf``, so the
round trip a test exercises is the real one.

    bids = make_bids_dataset(tmp_path)              # 2 subjects x 2 tasks
    bids, pairs = make_hyper_dataset(tmp_path)      # 1 dyad x 2 tasks + pairs.csv
    bids = make_bids_dataset(tmp_path, tasks=("mixed",))   # blocks with trials inside them

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

# ---- Two annotation levels ----
# A condition window is built from one annotation, so a single-level design gives every
# window exactly one trial however many events it has. The per-condition trial panels need a
# long annotation naming the condition with shorter ones inside it, which is what this task
# is for and the only design in this file that has it.
_TASKS_BLOCKED = ("mixed",)

# (name, onset, duration, the boxcar height its trials evoke). The two differ so a reader can
# tell two condition pages apart, which is the thing a shared colour scale is there to show.
_BLOCKS = (("talk", 20.0, 150.0, 1.0), ("listen", 190.0, 150.0, 0.35))
_TRIALS_PER_BLOCK = 6
_TRIAL_DURATION = 8.0
_TRIAL_LEAD_IN = 10.0     # room for the epoch window's baseline at the block's own start
_TRIAL_SPACING = 25.0


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


def _events(
    task: str, duration: float,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Event table as (onsets, durations, labels, amplitudes).

    Block design for task runs, two levels for ``mixed``, nothing at all for rest.

    Amplitude is the boxcar height each event evokes. It is zero for a block marker, so a
    two-level design's response comes from the trials inside a block rather than from the
    block itself, and every trial in one block shares a height while the two blocks differ.
    """
    if task in _TASKS_BLOCKED:
        onsets: list[float] = []
        durations: list[float] = []
        labels: list[str] = []
        amps: list[float] = []
        for name, start, span, amp in _BLOCKS:
            onsets.append(start)
            durations.append(span)
            labels.append(name)
            amps.append(0.0)
            for k in range(_TRIALS_PER_BLOCK):
                onset = start + _TRIAL_LEAD_IN + k * _TRIAL_SPACING
                if onset + _TRIAL_DURATION > min(start + span, duration):
                    break
                onsets.append(onset)
                durations.append(_TRIAL_DURATION)
                # one description for every trial in the run: which condition a trial belongs
                # to is its window's answer, not its name's
                labels.append("trial")
                amps.append(amp)
        order = np.argsort(onsets)
        return (np.asarray(onsets)[order], np.asarray(durations)[order],
                [labels[i] for i in order], np.asarray(amps)[order])
    if task not in _TASKS_WITH_EVENTS:
        return np.array([]), np.array([]), [], np.array([])
    onsets = np.arange(20.0, duration - 20.0, 40.0)
    return (onsets, np.full(onsets.shape, 10.0), [task] * len(onsets),
            np.ones(onsets.shape))


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
    onsets, durations, labels, amps = _events(task, duration)

    # Boxcar response at the events, so GLM has something to fit. Added rather than assigned:
    # a two-level design has a block annotation spanning its own trials, and it contributes 0.
    response = np.zeros(n)
    for onset, dur, amp in zip(onsets, durations, amps):
        response[int(onset * SFREQ):int((onset + dur) * SFREQ)] += amp

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


def _write_subject(bids_dir: Path, subject: str, task: str, raw: mne.io.Raw,
                   session: "str | None" = None) -> Path:
    from fnirs_pipe.io.snirf import write_snirf

    nirs_dir = bids_dir / f"sub-{subject}" / (f"ses-{session}" if session else "") / "nirs"
    prefix = f"sub-{subject}" + (f"_ses-{session}" if session else "")
    stem = f"{prefix}_task-{task}_nirs"
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
        (nirs_dir / f"{prefix}_task-{task}_events.tsv").write_text("\n".join(rows) + "\n")

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
    carries none, which is the recordings-without-triggers case. ``mixed`` is the two-level
    design, and the only one whose per-condition pages carry a trial panel.
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
    sessions: "tuple[str, ...] | None" = None,
    member_sessions: "dict[str, str] | None" = None,
    occasion: "str | None" = None,
    **raw_kwargs,
) -> tuple[Path, Path]:
    """Write a dyad dataset plus its pairs CSV. Returns (bids_dir, pairs_csv).

    ``hold`` carries triggers so alignment has something to align on; ``rest``
    carries none, which is the case trigger alignment refuses outright. ``sessions``
    records every group once per session, each a recording of its own, and gives the
    pairs CSV a session column. ``member_sessions`` records each subject once under its
    own session label, as per-person visit numbering does, and ``occasion`` adds the
    column naming that sitting.
    """
    groups = groups or {"G01": ("11", "12")}
    subjects = [s for members in groups.values() for s in members]

    bids_dir = Path(root) / name
    _write_dataset_root(bids_dir, subjects)
    for subject in subjects:
        for task in tasks:
            if member_sessions:
                ses = member_sessions[subject]
                seed = zlib.crc32(f"{subject}/{task}/{ses}".encode())
                _write_subject(bids_dir, subject, task,
                               synth_raw(subject, task, seed=seed, **raw_kwargs), session=ses)
            elif sessions is None:
                _write_subject(bids_dir, subject, task,
                               synth_raw(subject, task, **raw_kwargs))
            for ses in sessions or ():
                seed = zlib.crc32(f"{subject}/{task}/{ses}".encode())
                _write_subject(bids_dir, subject, task,
                               synth_raw(subject, task, seed=seed, **raw_kwargs), session=ses)

    pairs_csv = Path(root) / f"{name}_pairs.csv"
    rows = ["group_id,subject_id,task" + (",session" if sessions or member_sessions else "")
            + (",occasion" if occasion else "")]
    for group_id, members in groups.items():
        for subject in members:
            if member_sessions:
                rows += [f"{group_id},sub-{subject},{task},{member_sessions[subject]}"
                         + (f",{occasion}" if occasion else "") for task in tasks]
            elif sessions is None:
                rows += [f"{group_id},sub-{subject},{task}" for task in tasks]
            rows += [f"{group_id},sub-{subject},{task},{ses}"
                     for ses in sessions or () for task in tasks]
    pairs_csv.write_text("\n".join(rows) + "\n")

    return bids_dir, pairs_csv
