"""A derivative must keep the optode numbers its channels came with.

mne-nirs indexes the optodes it finds 1, 2, 3, ... and MNE names channels by those indices,
so a recording that lost a whole source or detector came back with every later optode renamed.
The data and positions travelled with the wrong names, so nothing failed until a name was
looked up in a region map.
"""

import numpy as np
import pytest

from fnirs_pipe.io.snirf import read_snirf, write_snirf

from tests._synth import synth_raw


@pytest.fixture(scope="module")
def raw():
    return synth_raw("01", "tapping", duration=30.0)


def _round_trip(raw, tmp_path):
    path = tmp_path / "sub-01_task-tapping_desc-od_nirs.snirf"
    write_snirf(raw, path)
    return read_snirf(path)


def test_a_missing_optode_does_not_rename_the_later_ones(raw, tmp_path):
    gap = raw.copy().drop_channels(["S2_D2 760", "S2_D2 850"])
    back = _round_trip(gap, tmp_path)
    assert back.ch_names == gap.ch_names


def test_positions_stay_with_their_channels(raw, tmp_path):
    gap = raw.copy().drop_channels(["S2_D2 760", "S2_D2 850"])
    back = _round_trip(gap, tmp_path)
    for written, read in zip(gap.info["chs"], back.info["chs"]):
        np.testing.assert_allclose(read["loc"][3:9], written["loc"][3:9])


def test_a_montage_numbered_without_gaps_is_written_as_before(raw, tmp_path):
    import h5py
    path = tmp_path / "sub-01_task-tapping_desc-od_nirs.snirf"
    write_snirf(raw, path)
    with h5py.File(path) as f:
        assert f["nirs/probe/sourcePos3D"].shape[0] == 5
        assert not np.isnan(f["nirs/probe/sourcePos3D"][()]).any()
