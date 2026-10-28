"""Each optode's share of its long channels rejected: the count, and its fill on the flat map.

The montage below is laid out by hand so every share is read off it: S1 reaches D1 and D2 at
30 mm and the short detector D3 at 8 mm, S2 reaches D1 at 30 mm.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from nirspipe.io.optode_space import Placement, hold_placement
from nirspipe.qc.common.channel_table import optode_rejections
from nirspipe.qc.figures.subject import optode_layout

POSITIONS = {"S1": (0.0, 0.0, 0.0), "S2": (0.06, 0.0, 0.0),
             "D1": (0.03, 0.0, 0.0), "D2": (0.0, 0.03, 0.0), "D3": (0.008, 0.0, 0.0)}
PAIRS = ("S1_D1", "S2_D1", "S1_D2", "S1_D3")


def test_each_optode_counts_its_long_pairs_and_the_rejected_ones():
    status = {
        "S1_D1": {"separation": "long", "is_bad": True},
        "S1_D2": {"separation": "long", "is_bad": False},
        "S2_D1": {"separation": "long", "is_bad": False},
        # a short pair counts for neither end, rejected or not
        "S1_D9": {"separation": "short", "is_bad": True},
        # nor does one between the two ranges
        "S3_D3": {"separation": "unclassified", "is_bad": True},
        # a montage that never split writes n/a, which the tables read as long
        "S4_D4": {"separation": "n/a", "is_bad": True},
    }
    assert optode_rejections(status) == {
        "S1": (1, 2), "S2": (0, 1), "D1": (1, 2), "D2": (0, 1), "S4": (1, 1), "D4": (1, 1)}


def _montage():
    names = [f"{pair} {wl}" for pair in PAIRS for wl in (760, 850)]
    info = mne.create_info(names, 10.0, ch_types="fnirs_cw_amplitude")
    for ch, name in zip(info["chs"], names):
        pair, wl = name.split(" ")
        src, det = (np.array(POSITIONS[o]) for o in pair.split("_"))
        ch["loc"][:3], ch["loc"][3:6], ch["loc"][6:9] = (src + det) / 2, src, det
        ch["loc"][9] = float(wl)
    raw = mne.io.RawArray(np.ones((len(names), 50)), info, verbose=False)
    # drawn as laid out, so each optode sits at its own x, y
    return hold_placement(raw, Placement("none", note="laid out by hand"))


def _fills(bad, monkeypatch, sep_bands=None) -> "tuple[dict, set]":
    """Each optode's fill value as drawn, and the optodes drawn in the grey of no long channel."""
    drawn = []
    monkeypatch.setattr(optode_layout, "fig_png_b64", lambda fig: drawn.append(fig) or "")
    raw = _montage()
    optode_layout.optode_layout_static(raw, {n: 0.9 for n in raw.ch_names}, bad, 0.8,
                                       sep_bands=sep_bands)
    ax = drawn[0].axes[0]
    at = {tuple(np.round(t.get_position(), 9)): t.get_text() for t in ax.texts}
    values, grey = {}, set()
    for markers in ax.collections:
        names = [at[tuple(np.round(xy, 9))] for xy in markers.get_offsets()]
        shares = markers.get_array()
        if shares is None:
            grey |= set(names)
        else:
            values.update(zip(names, np.asarray(shares, float)))
    return values, grey


def test_an_optode_is_filled_by_the_share_of_its_long_channels_rejected(monkeypatch):
    values, grey = _fills(["S1_D1 760", "S1_D2 850"], monkeypatch)
    assert values == pytest.approx({"S1": 1.0, "S2": 0.0, "D1": 0.5, "D2": 1.0})
    # D3 is reached by the short pair alone
    assert grey == {"D3"}


def test_the_run_s_separations_decide_which_channels_count(monkeypatch):
    # 8 mm is long under these bands, so S1 has three long channels and D3 one
    values, grey = _fills(["S1_D1 760", "S1_D2 850"], monkeypatch,
                          sep_bands=(0.005, 0.007, None))
    assert values["S1"] == pytest.approx(2 / 3)
    assert values["D3"] == 0.0 and not grey
