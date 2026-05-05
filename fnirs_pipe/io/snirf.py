"""
snirf file read/write (wraps MNE-NIRS + h5py).

References:
https://mne.tools/stable/auto_tutorials/io/30_reading_fnirs_data.html
"""


from pathlib import Path

import mne
from mne_nirs.io.snirf import write_raw_snirf


def write_snirf(raw: mne.io.Raw, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_raw_snirf(_patch_haemo_wavelengths(raw), str(out_path))


def _patch_haemo_wavelengths(raw: mne.io.Raw) -> mne.io.Raw:
    # mne_nirs bug: list.index() on loc[9] fails for nan (nan != nan); sentinel floats
    # are harmless since the file still gets correct dataTypeLabel/dataType=99999.
    # TODO: switch to pysnirf2 once it supports NumPy 2.x
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


def has_short_channels(raw: mne.io.Raw) -> bool:
    """Return True if the recording contains short-distance reference channels."""
    from mne_nirs.channels import get_short_channels
    return len(get_short_channels(raw)) > 0
