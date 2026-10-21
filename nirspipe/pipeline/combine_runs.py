"""Runs of one task joined into one series, for ``--combine-runs``.

Each run is z-scored per channel before the runs are joined in run order, so that a run with
a larger variance does not outweigh the others in a correlation taken across the joined
series. The mean and SD come from the run's kept frames, the ones a correlation reads. A
channel takes part only in the runs where it is good; its frames in the others are NaN, so a
correlation over the joined series reads each pair over the frames both channels have.
"""

from __future__ import annotations

import mne
import numpy as np

from nirspipe.exceptions import StageError
from nirspipe.pipeline.glm_censored import kept_frames


def zscore_kept(data: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Each row centred and scaled by its mean and SD (ddof 0) over the kept frames; a flat row is zero.

    ::

      [[1, 2, 3, 100]], keep [True, True, True, False]  ->  [[-1.22, 0, 1.22, 120.02]]
    """
    data = np.asarray(data, float)
    mean = data[:, keep].mean(axis=1, keepdims=True)
    sd = data[:, keep].std(axis=1, keepdims=True)
    out = np.zeros_like(data)
    scaled = sd[:, 0] > 0
    out[scaled] = (data[scaled] - mean[scaled]) / sd[scaled]
    return out


def join_runs(raws: "list[mne.io.BaseRaw]", absent: "list[set[str]] | None" = None,
              ) -> "tuple[mne.io.RawArray, list[float]]":
    """The runs z-scored over their kept frames and joined in order, with each run's onset in seconds.

    A channel's frames are NaN in a run where it is ``absent`` (by default the run's bad
    channels). The first run's info carries over, its bad channels replaced by those bad in
    every run; each run's annotations keep their place within it. Refused when the runs differ
    in channels or rate.
    """
    first = raws[0]
    for raw in raws[1:]:
        if raw.ch_names != first.ch_names:
            raise StageError("the runs do not share one channel list, so they cannot be joined")
        if raw.info["sfreq"] != first.info["sfreq"]:
            raise StageError(f"the runs are sampled at {first.info['sfreq']:g} and "
                             f"{raw.info['sfreq']:g} Hz; join them after --resample-sfreq")
    if absent is None:
        absent = [set(raw.info["bads"]) for raw in raws]
    sfreq = float(first.info["sfreq"])
    info = first.info.copy()
    info["bads"] = [ch for ch in first.ch_names if all(ch in raw.info["bads"] for raw in raws)]
    onsets, onset = [], 0.0
    parts, onset_s, duration_s, description = [], [], [], []
    for raw, missing in zip(raws, absent):
        onsets.append(onset)
        part = zscore_kept(raw.get_data(), kept_frames(raw))
        part[[raw.ch_names.index(ch) for ch in missing]] = np.nan
        parts.append(part)
        ann = raw.annotations
        onset_s += list(ann.onset - raw.first_time + onset)
        duration_s += list(ann.duration)
        description += list(ann.description)
        onset += raw.n_times / sfreq
    joined = mne.io.RawArray(np.hstack(parts), info, verbose="error")
    joined.set_annotations(mne.Annotations(onset_s, duration_s, description,
                                           orig_time=joined.annotations.orig_time))
    return joined, onsets


def join_roi_signals(raws: "list[mne.io.BaseRaw]", labels: "list[str]",
                     roi_map: "dict[str, list[str]]", chromophore: str,
                     ) -> "tuple[dict[str, np.ndarray], dict[str, dict[str, list[str]]]]":
    """Each ROI averaged per run over that run's good members, z-scored over its kept frames and joined.

    Returns ``({roi: joined signal}, {run label: {roi: members}})``. An ROI is NaN in a run
    where none of its channels is good, and left out when that holds in every run.
    """
    from nirspipe.pipeline.restingstate import _roi_members

    members = {label: _roi_members(raw, roi_map, chromophore) for raw, label in zip(raws, labels)}
    signals = {}
    for roi in roi_map:
        if not any(roi in m for m in members.values()):
            continue
        parts = []
        for raw, label in zip(raws, labels):
            picks = members[label].get(roi)
            if picks is None:
                parts.append(np.full(raw.n_times, np.nan))
                continue
            mean = raw.get_data(picks=picks).mean(axis=0, keepdims=True)
            parts.append(zscore_kept(mean, kept_frames(raw))[0])
        signals[roi] = np.concatenate(parts)
    return signals, members
