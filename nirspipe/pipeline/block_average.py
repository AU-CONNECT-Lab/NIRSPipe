"""Block average: each condition's trials averaged per channel over one window, written as a table."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from nirspipe.io.derivatives import write_step_sidecar
from nirspipe.io.tables import write_tsv
from nirspipe.pipeline.glm import _glm_name
from nirspipe.qc.figures.common._utils import chunk_annotations
from nirspipe.qc.metrics._helpers import EPOCH_TMAX, EPOCH_TMIN, condition_epochs
from nirspipe.utils import is_marker
from nirspipe.utils.logging import get_logger

logger = get_logger("post.block_average")

def with_table_events(raw: mne.io.Raw, events: pd.DataFrame) -> mne.io.Raw:
    """A copy of ``raw`` whose markers are the rows of an events table, its ``BAD_`` spans kept."""
    rows = events[events["trial_type"].map(is_marker)]
    ann = raw.annotations
    bad = [i for i, d in enumerate(ann.description) if not is_marker(d)]
    out = raw.copy()
    # table onsets count from the first sample, annotation onsets from the measurement start
    out.set_annotations(mne.Annotations(
        np.concatenate([rows["onset"].to_numpy(float) + raw.first_time, ann.onset[bad]]),
        np.concatenate([rows["duration"].to_numpy(float), ann.duration[bad]]),
        [*rows["trial_type"].astype(str), *ann.description[bad]],
        orig_time=ann.orig_time))
    return out


def block_average_table(raw: mne.io.Raw, tmin: float, tmax: float) -> pd.DataFrame:
    """One row per condition, channel and time: the trials' mean and standard error, and their count."""
    frames = []
    for cond, epochs in condition_epochs(raw, tmin, tmax).items():
        data = epochs.get_data()
        n = data.shape[0]
        mean = data.mean(axis=0)
        se = data.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full_like(mean, np.nan)
        n_ch, n_t = mean.shape
        frames.append(pd.DataFrame({
            "condition": cond,
            "ch_name": np.repeat(epochs.ch_names, n_t),
            "chroma": np.repeat(epochs.get_channel_types(), n_t),
            "time": np.tile(epochs.times, n_ch),
            "mean": mean.ravel(),
            "se": se.ravel(),
            "n_trials": n,
        }))
    if not frames:
        return pd.DataFrame(columns=["condition", "ch_name", "chroma", "time", "mean", "se",
                                     "n_trials", "bad"])
    table = pd.concat(frames, ignore_index=True)
    table["bad"] = table["ch_name"].isin(raw.info["bads"])
    return table


def write_block_average(
    raw: mne.io.Raw, output_dir: Path, source_path: str,
    tmin: "float | None" = None, tmax: "float | None" = None,
    chunk_duration: "float | None" = None, events: "pd.DataFrame | None" = None,
    inputs: "list[Path]" = (),
) -> "Path | None":
    """``<run>_desc-blockavg_timeseries.tsv`` beside the GLM tables, or None when no trial fits the window.

    ``events`` is the ``--events-path`` table when the design was built from one, so the
    trials averaged are the ones the model was given.
    """
    tmin = EPOCH_TMIN if tmin is None else tmin
    tmax = EPOCH_TMAX if tmax is None else tmax
    if events is not None:
        raw = with_table_events(raw, events)
    raw = chunk_annotations(raw, chunk_duration)
    table = block_average_table(raw, tmin, tmax)
    if table.empty:
        logger.warning("no trial leaves a full %g to %g s window inside the recording and clear "
                       "of BAD_ spans, so no block average is written", tmin, tmax)
        return None
    path = Path(output_dir) / _glm_name(source_path, "timeseries", desc="blockavg")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(table, path)
    trials = table.groupby("condition", sort=False)["n_trials"].first()
    write_step_sidecar(path, "block_average", source_path, list(raw.info["bads"]), inputs=inputs,
                       epoch_tmin=tmin, epoch_tmax=tmax,
                       baseline=[tmin, 0.0] if tmin < 0 else None,
                       epoch_chunk_duration=chunk_duration, unit="M",
                       trials={str(c): int(n) for c, n in trials.items()})
    return path
