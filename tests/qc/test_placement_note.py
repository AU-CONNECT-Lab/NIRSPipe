"""The run note for optodes no reading placed on fsaverage.

The placement itself is decided in `nirspipe.io.optode_space` and tested beside it; what is
pinned here is that the reports say so, from the same placement the brain views draw by.
"""

import numpy as np
import pytest

from nirspipe.io.optode_space import Placement, hold_placement
from nirspipe.io.snirf import read_snirf, write_snirf
from nirspipe.qc.common.channel_table import placement_note

from tests._synth import synth_raw

_REASON = ("no reading puts the optodes on the fsaverage scalp: as head frame 312.0 mm off, "
           "as fsaverage 298.4 mm off")


@pytest.fixture
def raw():
    return synth_raw("01", "rest")


def test_a_placed_run_gets_no_note(raw):
    assert placement_note(hold_placement(raw, Placement("landmarks", np.eye(4), 1.2, 9))) is None


def test_an_unplaced_run_says_why_and_that_the_metrics_survive(raw):
    note = placement_note(hold_placement(raw, Placement("none", note=_REASON)))
    assert _REASON in note
    assert "not anatomical" in note and "Nothing is drawn on the brain" in note
    assert "Separations, screening and metrics are unaffected" in note


def test_a_run_with_no_positions_is_left_to_the_separation_note(raw):
    for ch in raw.info["chs"]:
        ch["loc"][:9] = 0.0
    assert placement_note(hold_placement(raw, Placement("none", note=_REASON))) is None


def test_the_note_reads_the_placement_off_the_recording_file(raw, tmp_path):
    for ch in raw.info["chs"]:
        ch["loc"][:9] += np.tile([0.15, 0.10, 0.20], 3)
    path = tmp_path / "sub-01_task-rest_nirs.snirf"
    write_snirf(raw, path)
    note = placement_note(read_snirf(path))
    assert "could not be placed" in note and "no reading puts the optodes" in note
