"""SNIRF cropping: single-segment and multi-segment to BIDS derivatives.

Crops a raw recording, or with ``input_desc`` a pipeline stage inside a derivatives tree.
The second is the order the pipeline wants: motion correction and the bandpass both read
whatever series they are handed, so they belong on the whole recording and the cut belongs
after them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from fnirs_pipe import __version__
from fnirs_pipe.io.auxiliary import write_aux_window
from fnirs_pipe.io.derivatives import data_state, write_sidecar_json
from fnirs_pipe.io.snirf import read_snirf, write_snirf
from fnirs_pipe.io.tables import read_table
from fnirs_pipe.utils.logging import get_logger
from fnirs_pipe.utils.snirf_prep import (
    annotations_to_df,
    bids_stem,
    copy_dataset_root,
    copy_sidecars,
    deriv_nirs_dir,
    ensure_dataset_description,
    find_snirf,
    read_raw_snirf,
)

logger = get_logger("pipeline.crop")

_DERIV_NAME = "cropped"


def _trigger_origin(raw, trigger_name: str | None) -> float:
    """Onset of the first annotation named `trigger_name`, the t=0 the window is measured from.

    Recordings of the same session start whenever each machine was told to, so a window given
    in absolute seconds names a different moment of the task in every subject. Measuring it
    from a shared hardware trigger puts them back on one clock.

    Example: a subject whose "start" trigger fires at 12.4 s, cropped with tmin=60, tmax=90,
    keeps 72.4 s to 102.4 s of its own recording.

    A missing trigger falls back to 0.0 with a warning rather than raising: the crop still
    produces a usable file, just on the recording's own clock, and refusing would take down
    a whole batch for one subject.
    """
    if not trigger_name:
        raise ValueError("--align trigger requires --trigger-name")
    onsets = [float(a["onset"]) for a in raw.annotations
              if str(a["description"]) == trigger_name]
    if not onsets:
        logger.warning("trigger '%s' not found; measuring the window from t=0 instead",
                       trigger_name)
        return 0.0
    return min(onsets)


def _window(seg) -> tuple[float, float]:
    """The span a cropped segment occupies on the original recording's clock."""
    return float(seg.first_time), float(seg.first_time + seg.times[-1])


def _write_segment(raw_seg, out_snirf: Path, snirf_path: Path,
                   windows: list[tuple[float, float]]) -> None:
    write_snirf(raw_seg, out_snirf)
    # write_snirf goes through the Raw, which has nowhere to hold an aux channel, so the
    # accelerometers have to be cut from the source and put back afterwards or a cropped
    # recording reaches postprocessing with no aux at all
    names = write_aux_window(snirf_path, out_snirf, windows)
    if names:
        logger.info("Carried %d aux channels into %s", len(names), out_snirf.name)
    events_path = out_snirf.parent / out_snirf.name.replace("_nirs.snirf", "_events.tsv")
    annotations_to_df(raw_seg).to_csv(events_path, sep="\t", index=False)


def _write_crop_sidecar(out_snirf: Path, raw_seg, source_path: Path,
                        windows: list[tuple[float, float]]) -> None:
    """Replace the copied sidecar with one describing the crop, for a derivative input.

    Two things would otherwise be lost or wrong. `copy_sidecars` brings the source stage's
    JSON across verbatim, so a short cut of a long file would claim that file's duration
    and name the bandpass as its own step. And SNIRF has no bad-channel field, so a segment
    whose sidecar does not list them reaches the next stage with none marked.

    The source's `parameters` are carried forward rather than replaced: `read_snirf` reads
    the bandpass off them, so dropping them would leave the segment not knowing its own band.
    """
    src = source_path.with_suffix(".json")
    try:
        parameters = (json.loads(src.read_text(encoding="utf-8")).get("parameters") or {})
    except (OSError, json.JSONDecodeError):
        parameters = {}
    write_sidecar_json(out_snirf, {
        "pipeline_version": __version__,
        "step": "crop",
        "Sources": [source_path.as_posix()],
        "parameters": {**parameters,
                       "crop_windows_s": [[round(a, 3), round(b, 3)] for a, b in windows]},
        "data": data_state(raw_seg),
        "bad_channels": list(raw_seg.info["bads"]),
    })


def _setup_deriv_dir(derivatives_dir: Path, sub: str, ses: str | None) -> Path:
    out_nirs_dir = deriv_nirs_dir(derivatives_dir, _DERIV_NAME, sub, ses)
    ensure_dataset_description(derivatives_dir / _DERIV_NAME, _DERIV_NAME, "fnirs-prep crop")
    out_nirs_dir.mkdir(parents=True, exist_ok=True)
    return out_nirs_dir


def _crop_raw(
    raw,
    snirf_path: Path,
    out_nirs_dir: Path,
    stem: str,
    *,
    tmin: float | None,
    tmax: float | None,
    segments_df: pd.DataFrame | None,
    combine: bool,
    t0: float = 0.0,
    derivative: bool = False,
) -> list[Path]:
    import mne

    duration = float(raw.times[-1])

    def _clip(a: float | None, b: float | None) -> tuple[float, float]:
        """Shift a window onto the recording and keep it inside it.

        A trigger origin pushes every window later, so one that fit on the recording's own
        clock can now run off the end. Clipping keeps the segment that does exist instead of
        letting mne refuse the whole crop. An open end is resolved here too: mne reads a
        `tmin` of None as a comparison against None, not as "from the start".
        """
        lo = t0 if a is None else min(t0 + a, duration)
        hi = duration if b is None else min(t0 + b, duration)
        if hi - lo <= 0:
            logger.warning("window [%s, %s] starts at or past the end of %s (%.1fs); "
                           "the segment will be empty",
                           a, b, snirf_path.name, duration)
        return lo, hi

    def _write(seg, name: str, windows: list[tuple[float, float]]) -> Path:
        out = out_nirs_dir / f"{name}_nirs.snirf"
        _write_segment(seg, out, snirf_path, windows)
        copy_sidecars(snirf_path, stem, out_nirs_dir, name)
        # after copy_sidecars, which writes the same filename: the copied one describes the
        # source stage over the whole recording, and this segment is neither
        if derivative:
            _write_crop_sidecar(out, seg, snirf_path, windows)
        return out

    if segments_df is not None:
        segs = [
            raw.copy().crop(*_clip(row.onset, row.onset + row.duration))
            for _, row in segments_df.iterrows()
        ]
        if combine:
            # taken before concatenating: mne.concatenate_raws appends into segs[0]
            windows = [_window(seg) for seg in segs]
            out = _write(mne.concatenate_raws(segs), stem, windows)
            logger.info("Written combined: %s", out)
            return [out]
        tasks = segments_df["task"] if "task" in segments_df.columns else None
        out_paths: list[Path] = []
        for i, seg in enumerate(segs, start=1):
            if tasks is None:
                name = f"{stem}_seg-{i:02d}"
            else:
                name = re.sub(r"task-[^_]+", f"task-{tasks.iloc[i - 1]}", stem)
            out = _write(seg, name, [_window(seg)])
            logger.info("Written segment %d: %s", i, out)
            out_paths.append(out)
        return out_paths

    lo, hi = _clip(tmin, tmax)
    seg = raw.copy().crop(tmin=lo, tmax=hi)
    out = _write(seg, stem, [_window(seg)])
    logger.info("Written: %s", out)
    return [out]


def crop_snirf_from_path(
    snirf_path: Path,
    derivatives_dir: Path,
    sub: str,
    ses: str | None = None,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    segments_df: pd.DataFrame | None = None,
    combine: bool = False,
    align: str = "none",
    trigger_name: str | None = None,
    derivative: bool = False,
) -> list[Path]:
    """Crop a SNIRF given its path directly (no BIDS layout lookup).

    Public entry point shared by the interface. ``derivative`` says the input is a pipeline
    stage rather than a recording, which changes how the output's sidecar is written.
    Returns list of written SNIRF paths.
    """
    stem = bids_stem(snirf_path)
    out_nirs_dir = _setup_deriv_dir(derivatives_dir, sub, ses)
    # read_snirf restores the bad-channel marks from the sidecar, which a derivative carries
    # and a raw recording does not
    raw = read_snirf(snirf_path) if derivative else read_raw_snirf(snirf_path)
    t0 = _trigger_origin(raw, trigger_name) if align == "trigger" else 0.0
    return _crop_raw(raw, snirf_path, out_nirs_dir, stem,
                     tmin=tmin, tmax=tmax, segments_df=segments_df, combine=combine, t0=t0,
                     derivative=derivative)


def crop_snirf(
    bids_dir: Path,
    derivatives_dir: Path,
    sub: str,
    ses: str | None = None,
    task: str | None = None,
    run: str | None = None,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    segments_path: Path | None = None,
    combine: bool = False,
    align: str = "none",
    trigger_name: str | None = None,
    validate: bool = False,
    input_desc: str | None = None,
) -> list[Path]:
    """Crop a SNIRF via BIDS layout lookup and write to derivatives/cropped/.

    Single segment: use tmin/tmax.
    Multi-segment: use segments_path (table with onset/duration columns).
    An optional `task` column names each segment, and its output takes that task entity
    instead of `_seg-NN`, which makes the segments separate tasks of a valid BIDS dataset
    rather than one task the pipeline cannot tell apart.
    combine=True concatenates multi-segment output into one file.

    align="trigger" measures tmin/tmax (and every segment onset) from the first annotation
    named `trigger_name` rather than from the recording start, so one window selects the same
    stretch of task in subjects whose recordings started at different moments.

    ``input_desc`` cuts a pipeline stage instead of a recording: `bids_dir` is then a
    derivatives tree and the file carrying that desc- entity is the input, e.g. "errts" for
    the residual. This is the order the pipeline wants, since motion correction and the
    bandpass both read whatever series they are handed. The desc- entity is kept on the
    output, so a cut of the residual is `..._task-baseline_desc-errts_nirs.snirf`.

    Returns list of written SNIRF paths.
    """
    derivative = input_desc is not None
    snirf_path = find_snirf(bids_dir, sub, ses, task, run,
                            validate=validate and not derivative, desc=input_desc)
    stem = bids_stem(snirf_path)
    out_nirs_dir = _setup_deriv_dir(derivatives_dir, sub, ses)
    copy_dataset_root(bids_dir, derivatives_dir / _DERIV_NAME)
    raw = read_snirf(snirf_path) if derivative else read_raw_snirf(snirf_path)

    segments_df = read_table(segments_path) if segments_path is not None else None
    t0 = _trigger_origin(raw, trigger_name) if align == "trigger" else 0.0
    return _crop_raw(raw, snirf_path, out_nirs_dir, stem,
                     tmin=tmin, tmax=tmax, segments_df=segments_df, combine=combine, t0=t0,
                     derivative=derivative)
