"""
snirf file read/write (wraps MNE-NIRS + h5py).

References:
https://mne.tools/stable/auto_tutorials/io/30_reading_fnirs_data.html
"""


import json
import re
from pathlib import Path
from typing import Any

import mne
from mne_nirs.io.snirf import write_raw_snirf

from fnirs_pipe.utils.lineage import stamp

_DESC_RE = re.compile(r"_desc-([A-Za-z0-9]+)[_.]")


def _zero_first_time(raw: mne.io.Raw) -> mne.io.Raw:
    """Rebuild a cropped recording on a time axis that starts at zero.

    A cropped Raw keeps its annotations on the original recording's axis and holds the
    offset in first_time, so a marker at second 0 of a segment taken from 543 s in still
    reads 543. SNIRF has nowhere to put that offset: its axis always starts at zero, and
    the marker would be written at 543 s of a 900 s segment, or past the end of a segment
    cropped from later still, where the reader silently drops it.
    """
    a = raw.annotations
    out = mne.io.RawArray(raw.get_data(), raw.info.copy(), verbose="error")
    out.set_annotations(mne.Annotations(a.onset - raw.first_time, a.duration, a.description))
    return out


def write_snirf(raw: mne.io.Raw, out_path: Path) -> None:
    # Recordings always carry both, but an object built in memory may not, and the
    # failure then lands inside mne_nirs with an error that names neither field.
    if raw.info["meas_date"] is None:
        raise ValueError("write_snirf needs info['meas_date']: SNIRF stores a measurement date")
    if not raw.info["subject_info"]:
        raise ValueError("write_snirf needs info['subject_info']: SNIRF derives its subject id from it")
    if raw.first_time:
        raw = _zero_first_time(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_raw_snirf(_patch_haemo_wavelengths(raw), str(out_path))


def _sidecar(path: Path) -> dict:
    """The JSON written beside a SNIRF, or {} when it is missing or unreadable."""
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _restore_bads(raw: mne.io.Raw, sidecar: dict) -> None:
    """SNIRF has no bad-channel field, so the sidecar carries the marks across the disk.

    Each sidecar lists the bads of its own file, so the names always match that file's
    channels; the filter only guards against a hand-edited sidecar.
    """
    bads = sidecar.get("bad_channels") or []
    if bads:
        raw.info["bads"] = [ch for ch in bads if ch in raw.ch_names]


# restored onto the lineage stamp, so the filter can be read off the Raw rather than the file
_FILTER_KEYS = ("high_pass", "low_pass", "filter_method", "filter_order")
# the drift basis empties a band too, so a consumer asking what the file's low edge is has
# to see both
_DRIFT_KEYS = ("drift_model", "drift_high_pass", "drift_order")


def read_snirf(path: Path | str, **kwargs: Any) -> mne.io.Raw:
    """Read a SNIRF and restore its pipeline stage from the desc- entity.

    The lineage stamp lives in info["temp"] and does not survive the SNIRF round
    trip, so the filename carries it instead: the writer derived desc- from the
    stamp, this reads the same string back. Files with no desc- entity are BIDS
    inputs and stamp as "raw". Bad-channel marks and the bandpass the file went
    through are restored from the sidecar for the same reason.
    """
    kwargs.setdefault("preload", True)
    path = Path(path)
    raw = mne.io.read_raw_snirf(str(path), **kwargs)
    sidecar = _sidecar(path)
    _restore_bads(raw, sidecar)
    params = sidecar.get("parameters") or {}
    desc = _DESC_RE.search(path.name)
    return stamp(raw, stage=desc.group(1) if desc else "raw", step="load", path=path.as_posix(),
                 **{k: params[k] for k in _FILTER_KEYS + _DRIFT_KEYS
                    if params.get(k) is not None})


def _patch_haemo_wavelengths(raw: mne.io.Raw) -> mne.io.Raw:
    # mne_nirs bug: list.index() on loc[9] fails for nan (nan != nan); sentinel floats
    # are harmless since the file still gets correct dataTypeLabel/dataType=99999.
    # TODO: switch to pysnirf2 once a release carries its NumPy 2.x fix
    ch_types = raw.get_channel_types()
    if "hbo" not in ch_types and "hbr" not in ch_types:
        return raw
    raw = raw.copy()
    for ch, ch_type in zip(raw.info["chs"], ch_types):
        if ch_type == "hbo":
            ch["loc"][9] = 1.0
        elif ch_type == "hbr":
            ch["loc"][9] = 2.0
    return raw


def has_short_channels(raw: mne.io.Raw, sep_bands=None) -> bool:
    """Return True if the recording contains short-distance reference channels."""
    from fnirs_pipe.qc.metrics._helpers import long_short_channels

    return bool(long_short_channels(raw, sep_bands)[1])


def long_channel_picks(
    raw: mne.io.Raw, ch_type: str = "hbo", exclude="bads", sep_bands=None,
) -> list[int]:
    """Picks for one chromophore with short-distance reference channels dropped.

    A 6-channel montage whose last pair is short ->
        long_channel_picks(raw, "hbo") == [0, 1]   (the third hbo pick is gone)

    Short channels sample scalp haemodynamics, so an inter-brain metric computed on them
    measures systemic physiology two people share by sitting in the same room rather than
    any brain coupling. Montages with no short channels lose nothing.

    "Long" is the package's one separation rule,
    :func:`~fnirs_pipe.qc.metrics._helpers.separation_bands`, and a channel outside both
    bands is in neither list. This dropped only the short channels until 0.30.0, so a
    separation past the long band was short-distance to the reports and usable to the dyad
    metrics; ``long_short_channels`` names such a channel in a warning.

    ``exclude`` is pick_types', so rejected channels are dropped by default, which is what a
    metric wants. ``exclude=[]`` keeps them, which is what the *axis* of a channel-by-channel
    matrix wants: an axis over the montage rather than over the survivors gives every subject
    and every dyad a matrix of one shape, so a group analysis can stack them however their
    rejections differ.
    """
    from fnirs_pipe.qc.metrics._helpers import long_short_channels

    long_names = set(long_short_channels(raw, sep_bands)[0])
    return [p for p in mne.pick_types(raw.info, fnirs=ch_type, exclude=exclude)
            if raw.ch_names[p] in long_names]
