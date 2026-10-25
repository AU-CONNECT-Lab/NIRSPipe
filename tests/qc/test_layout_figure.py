"""The montage figures: what each channel is drawn as.

The dashes follow the one long/short split the reports use, so a channel in the gap between
the two ranges is neither; the colours follow the wavelength.
"""

from __future__ import annotations

import numpy as np
import pytest

from nirspipe.qc.figures.common._utils import BAND_COLORS
from nirspipe.qc.figures.hyper.hyper_figures import _psd_band_shapes
from nirspipe.qc.figures.subject.brain_views import _lookup_sci
from nirspipe.qc.figures.subject.raw_figures import _ch_styles
from tests._synth import synth_raw


@pytest.fixture
def raw():
    return synth_raw("01", "hold", duration=20.0, motion_onset=None)


def test_channel_dashes_follow_the_long_short_split(raw):
    # move the first pair's detector into the gap the two ranges leave between them
    for ch in raw.info["chs"][:2]:
        ch["loc"][6:9] = ch["loc"][3:6] + np.array([0.012, 0.0, 0.0])
    dashes = {name: dash for name, (_, dash) in zip(raw.ch_names, _ch_styles(raw))}

    assert dashes[raw.ch_names[0]] == "dot"
    assert dashes[raw.ch_names[2]] == "solid"
    short = next(n for n, ch in zip(raw.ch_names, raw.info["chs"])
                 if np.linalg.norm(ch["loc"][6:9] - ch["loc"][3:6]) < 0.01)
    assert dashes[short] == "dash"


def test_a_pairs_shorter_wavelength_is_orange_and_its_longer_blue(raw):
    colours = {name: colour for name, (colour, _) in zip(raw.ch_names, _ch_styles(raw))}
    assert colours["S1_D1 760"] == "#e66101"
    assert colours["S1_D1 850"] == "#2166ac"


def test_an_sci_of_zero_is_kept_rather_than_passed_over():
    assert _lookup_sci({"S1_D1 hbo": 0.0, "S1_D1 hbr": 0.9}, "S1_D1 760") == 0.0


def test_the_dyad_psd_bands_take_the_shared_colours():
    shapes, _ = _psd_band_shapes(cardiac=(0.8, 1.6))
    assert {s["fillcolor"] for s in shapes} <= set(BAND_COLORS.values())
