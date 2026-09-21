"""Catching optode positions that were never registered to the head.

A SNIRF with no landmarks still reads as head coordinates, so the digitiser's own frame
reaches the figures unchallenged and the optodes are drawn wherever they happen to sit.
Nothing downstream fails, which is why it has to be measured rather than raised.
"""

import numpy as np
import mne

from fnirs_pipe.qc.common.channel_table import registration_note
from fnirs_pipe.qc.metrics import REGISTRATION_MAX_RATIO, registration_offset

# a head's worth of cardinal points, the shape a template fills in when a file carries none
NASION, LPA, RPA = [0.0, 0.085, -0.035], [-0.081, -0.029, -0.041], [0.084, -0.029, -0.041]


def _montage(offset=(0.0, 0.0, 0.0), fiducials=True):
    """Two source-detector pairs on the scalp, shifted bodily by ``offset`` metres."""
    names = ["S1_D1 760", "S1_D1 850", "S2_D2 760", "S2_D2 850"]
    info = mne.create_info(names, 10.0, ch_types=["fnirs_cw_amplitude"] * 4)
    for ch, wl in zip(info["chs"], [760.0, 850.0, 760.0, 850.0]):
        ch["loc"][9] = wl
    raw = mne.io.RawArray(np.zeros((4, 100)), info, verbose=False)
    o = np.asarray(offset, float)
    pos = {"S1": o + [-0.02, 0.06, 0.05], "D1": o + [0.01, 0.06, 0.05],
           "S2": o + [-0.02, -0.02, 0.07], "D2": o + [0.01, -0.02, 0.07]}
    fids = dict(nasion=NASION, lpa=LPA, rpa=RPA) if fiducials else {}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head", **fids),
                    verbose="error")
    return raw


def test_a_montage_on_the_head_is_not_flagged():
    assert registration_offset(_montage()) is None


def test_a_montage_in_the_digitiser_s_own_frame_is_flagged():
    # the cloud sits a third of a metre away, which is what an unregistered SNIRF looks like
    offset = registration_offset(_montage((0.15, 0.10, 0.25)))
    assert offset is not None
    reach, scalp = offset
    assert reach > REGISTRATION_MAX_RATIO * scalp
    # both numbers are in mm, since the note prints them
    assert 50 < scalp < 150 and reach > 200


def test_nothing_is_claimed_without_the_cardinal_points():
    """Those are the only thing the optodes are compared against, and a recording without
    them is already drawn without an anatomical claim."""
    assert registration_offset(_montage((0.15, 0.10, 0.25), fiducials=False)) is None


def test_positions_that_are_all_zero_are_not_a_registration_problem():
    """A montage with no positions at all is the separation notes' case, worded there as a
    missing registration; flagging it here too would say it twice."""
    raw = _montage()
    for ch in raw.info["chs"]:
        ch["loc"][:9] = 0.0
    assert registration_offset(raw) is None


def test_the_note_says_both_distances_and_that_the_metrics_survive():
    note = registration_note(registration_offset(_montage((0.15, 0.10, 0.25))))
    assert "396 mm" in note and "86 mm" in note
    # a reader's first question is whether the run is wasted, and it is not
    assert "Separations are measured between optodes" in note
    assert registration_note(None) is None
