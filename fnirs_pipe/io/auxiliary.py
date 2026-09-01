"""Auxiliary (non-optical) channels carried in a SNIRF file, and getting them onto a
haemoglobin time axis without aliasing.

MNE's SNIRF reader does not touch the `aux` group at all, so a recording's accelerometers
and gyroscopes are simply absent from `read_raw_snirf` output. They are read here with h5py
instead, written once during preprocessing as a derivative table, and resampled onto
whatever time axis the postprocessing stage ends up with.

Two properties of the format shape everything below:

- The aux clock and the optical clock are independent, and neither the ratio between them
  nor the aux rate itself is guaranteed constant. Alignment therefore goes through the
  recorded timestamps rather than sample counts, which drift apart over a long recording.
- Aux is usually sampled well above the optical rate, so putting it on the optical time axis
  is decimation, and decimation without a low-pass folds the whole difference back into the
  band that is kept. `resample_to_grid` filters before it interpolates for that reason.

Anything that writes a SNIRF through MNE drops the aux group, since MNE's Raw cannot carry
one. `write_aux_window` puts it back after the write, which is what lets a cropped recording
reach postprocessing with its accelerometers intact.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from fnirs_pipe.utils.logging import get_logger

logger = get_logger("io.auxiliary")

# desc entity and suffix of the table preprocessing leaves behind
AUX_DESC = "aux"
AUX_SUFFIX = "timeseries"

# The column holding the recorded aux timestamps. It is what makes the table independent of
# any particular downstream rate, so it is written even though a nominal SamplingFrequency
# also goes in the sidecar.
TIME_COLUMN = "time"

# Six significant digits: below the resolution of the inertial and physiological sensors
# that populate an aux group, and appreciably smaller on disk than full repr precision.
_FLOAT_FORMAT = "%.6g"


# ---- reading ----

def _decode(value) -> str:
    """SNIRF metadata strings arrive as bytes, 0-d arrays or 1-element arrays.

    NIRx writes an aux channel's `name` as an HDF5 scalar; the reader in mne-nirs indexes it
    as `np.array(key)[0]` and raises IndexError on every file from that device.
    """
    array = np.asarray(value)
    item = array.item() if array.ndim == 0 else array.ravel()[0]
    return item.decode() if isinstance(item, bytes) else str(item)


def read_aux_snirf(path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Read every aux channel of a SNIRF file.

    Returns three dicts keyed by channel name: its timestamps, its values, and its unit.
    Each channel keeps its own time base because the format permits them to differ; callers
    that need one grid go through `resample_to_grid`.

    A `dataTimeSeries` of shape (T, 1) is squeezed to (T,). The spec allows it and the helper
    in mne-nirs does not, which is the second of the two reasons that helper fails here.
    """
    import h5py

    times: dict[str, np.ndarray] = {}
    values: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}

    with h5py.File(str(path), "r") as handle:
        if "nirs" not in handle:
            return times, values, units
        nirs = handle["nirs"]
        for key in sorted(k for k in nirs if k.startswith("aux")):
            group = nirs[key]
            if "dataTimeSeries" not in group or "time" not in group:
                logger.warning("aux group %s in %s has no data or no time, skipping", key, path)
                continue
            name = _decode(group["name"][()]) if "name" in group else key
            data = np.squeeze(np.asarray(group["dataTimeSeries"][()], dtype=float))
            if data.ndim != 1:
                logger.warning("aux channel %s is %s, not a single series, skipping", name, data.shape)
                continue
            stamps = np.asarray(group["time"][()], dtype=float).ravel()
            if len(stamps) != len(data):
                logger.warning("aux channel %s has %d timestamps for %d samples, skipping",
                               name, len(stamps), len(data))
                continue
            times[name], values[name] = stamps, data
            units[name] = _decode(group["dataUnit"][()]) if "dataUnit" in group else "n/a"

    return times, values, units


# ---- resampling ----

def resample_to_grid(t_src: np.ndarray, x: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Put a signal sampled at ``t_src`` onto the grid ``t_dst``, anti-aliased.

    A rate change downwards is decimation, and decimation without a low-pass folds every
    component above the target Nyquist back into the band that survives, landing inside the
    band a confound regression then works in. The low-pass runs first for that reason; when
    ``t_dst`` is the finer grid there is nothing to fold and only the interpolation happens.

    The filter treats ``t_src`` as uniform while the interpolation uses the timestamps as
    recorded. That split is deliberate: sampling jitter is normally far too small to affect
    a filter's response, and far too large to ignore when it accumulates over a long
    recording into a real offset at the end.

    Parameters
    ----------
    t_src : ndarray, shape (n_src,)
        Timestamps of ``x``, in seconds, ascending.
    x : ndarray, shape (n_src,)
        The signal to resample.
    t_dst : ndarray, shape (n_dst,)
        Target timestamps, in seconds. Values outside the span of ``t_src`` are held at the
        nearest end rather than extrapolated.

    Returns
    -------
    ndarray, shape (n_dst,)
        ``x`` on the target grid.

    Notes
    -----
    The anti-alias cutoff is 0.4 times the target rate, i.e. 80% of the target Nyquist,
    leaving a transition band that a 4th-order Butterworth run forwards and backwards
    (zero phase, effective order 8) has room to roll off in.
    """
    from scipy.signal import butter, sosfiltfilt

    if len(t_src) < 2 or len(t_dst) < 2:
        return np.interp(t_dst, t_src, x)

    fs_src = 1.0 / float(np.median(np.diff(t_src)))
    fs_dst = 1.0 / float(np.median(np.diff(t_dst)))

    filtered = x
    # 0.99 keeps a rate that is nominally the same from being filtered by rounding noise
    if fs_dst < fs_src * 0.99:
        cutoff = 0.4 * fs_dst
        # padlen default can exceed a very short series; filtfilt raises rather than coping
        if len(x) > 24:
            sos = butter(4, cutoff / (fs_src / 2.0), btype="low", output="sos")
            filtered = sosfiltfilt(sos, x)
        else:
            logger.warning("aux series of %d samples is too short to anti-alias", len(x))

    return np.interp(t_dst, t_src, filtered)


# ---- carrying aux through a crop ----

def write_aux_window(
    source_path: Path | str,
    dest_path: Path | str,
    windows: list[tuple[float, float]],
) -> list[str]:
    """Copy a recording's aux channels into another SNIRF, keeping only ``windows``.

    Cropping goes through MNE, and MNE's Raw has nowhere to hold an aux channel, so a
    cropped recording written by `write_snirf` has no aux group at all. This puts one back
    afterwards, reading the source with h5py and writing the retained samples straight into
    the file that was just written.

    Each window is rebased so the result starts at zero, matching what cropping does to the
    optical data and to the markers. Several windows are laid end to end, window ``i``
    starting at the summed length of those before it, which is how `mne.concatenate_raws`
    joins the segments the combined output is made of.

    Aux keeps the rate it was recorded at. Resampling is a postprocessing decision and
    `resample_to_grid` is where it belongs.

        source 0 to 600 s at 98.67 Hz, windows [(100, 400)]
            -> one channel of 29601 samples stamped 0 to 300 s

    Parameters
    ----------
    source_path : path
        The uncropped SNIRF, read for its aux group.
    dest_path : path
        A SNIRF already written by `write_snirf`. Its ``nirs`` group gains ``aux1``, ``aux2``
        and so on, numbered over the channels actually written.
    windows : list of (float, float)
        Spans to keep, in seconds on the source recording's clock, in output order.

    Returns
    -------
    list of str
        Names of the channels written. Empty when the source has no aux group, or when no
        channel had a sample inside any window.
    """
    import h5py

    times, values, units = read_aux_snirf(source_path)
    if not values:
        return []

    written: list[str] = []
    with h5py.File(str(dest_path), "a") as handle:
        nirs = handle.require_group("nirs")
        for name in values:
            stamps, data = times[name], values[name]
            kept_t: list[np.ndarray] = []
            kept_x: list[np.ndarray] = []
            offset = 0.0
            for start, stop in windows:
                inside = (stamps >= start) & (stamps <= stop)
                if inside.any():
                    kept_t.append(stamps[inside] - start + offset)
                    kept_x.append(data[inside])
                offset += stop - start
            if not kept_t:
                logger.warning("aux channel %s has no sample inside the cropped span", name)
                continue

            group = nirs.create_group(f"aux{len(written) + 1}")
            group.create_dataset("name", data=name.encode())
            group.create_dataset("dataTimeSeries", data=np.concatenate(kept_x))
            group.create_dataset("time", data=np.concatenate(kept_t))
            group.create_dataset("dataUnit", data=units[name].encode())
            written.append(name)

    return written


# ---- the derivative table ----

def aux_table_path(reference: Path) -> Path:
    """Where the aux table for a recording sits, given any snirf of the same run.

    ``sub-01/nirs/sub-01_task-hold_desc-preproc_nirs.snirf``
        -> ``sub-01/nirs/sub-01_task-hold_desc-aux_timeseries.tsv.gz``

    The desc entity names the stage and every stage of one run shares the rest of the name,
    so dropping desc and putting the aux one back is what makes any stage able to find it.
    """
    stem = reference.name.split(".")[0]
    tokens = [t for t in stem.split("_") if not t.startswith("desc-")][:-1]
    return reference.parent / ("_".join([*tokens, f"desc-{AUX_DESC}", AUX_SUFFIX]) + ".tsv.gz")


def find_aux_table(reference: Path) -> Path | None:
    path = aux_table_path(reference)
    return path if path.exists() else None


def write_aux_table(
    source_path: Path,
    out_path: Path,
) -> tuple[pd.DataFrame, dict] | None:
    """Extract a recording's aux channels into a gzipped TSV, at the rate they were recorded.

    Returns the table and the facts about it a sidecar should carry, or None when the file
    has no aux group. The table is deliberately not resampled: the rate it would be
    resampled to is a postprocessing decision, and baking it in here would make the file
    valid for exactly one downstream configuration.
    """
    times, values, units = read_aux_snirf(source_path)
    if not values:
        return None

    names = list(values)
    reference = times[names[0]]
    frame = {TIME_COLUMN: reference}
    interpolated: list[str] = []
    for name in names:
        if np.array_equal(times[name], reference):
            frame[name] = values[name]
        else:
            # a channel on its own time base is put on the first one's, anti-aliased when
            # that means going down in rate
            interpolated.append(name)
            frame[name] = resample_to_grid(times[name], values[name], reference)

    table = pd.DataFrame(frame)
    steps = np.diff(reference)
    nominal = 1.0 / float(np.median(steps)) if len(steps) else float("nan")
    # how far sample counting would drift from the recorded stamps by the end
    drift_s = float(np.abs(reference[0] + np.arange(len(reference)) / nominal - reference).max()) \
        if len(steps) else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as handle:
        table.to_csv(handle, sep="\t", index=False, float_format=_FLOAT_FORMAT)

    facts = {
        "Columns": list(table.columns),
        "Units": {name: units[name] for name in names},
        "SamplingFrequency": round(nominal, 4),
        "StartTime": round(float(reference[0]), 6),
        "n_samples": int(len(reference)),
        # BIDS assumes a physio table is on a uniform grid. These are not: the recorded
        # stamps are kept in a `time` column and this says how much that is worth.
        "timestamp_column": TIME_COLUMN,
        "constant_rate_drift_s": round(drift_s, 4),
    }
    if interpolated:
        facts["interpolated_onto_first_channel"] = interpolated
    return table, facts


def read_aux_table(path: Path) -> pd.DataFrame:
    """The table written by `write_aux_table`, time column included."""
    return pd.read_csv(path, sep="\t", compression="gzip")
