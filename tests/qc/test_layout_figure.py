"""The montage figures: what each channel is drawn as.

The separation follows the one long/short split the reports use, so a channel in the gap
between the two ranges is neither; the hue follows the wavelength, lighter for a short channel.
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


def test_channel_separation_follows_the_long_short_split(raw):
    # move the first pair's detector into the gap the two ranges leave between them
    for ch in raw.info["chs"][:2]:
        ch["loc"][6:9] = ch["loc"][3:6] + np.array([0.012, 0.0, 0.0])
    styles = dict(zip(raw.ch_names, _ch_styles(raw)))

    assert styles[raw.ch_names[0]][1] == "unclassified"
    assert styles[raw.ch_names[2]][1] == "long"
    short = next(n for n, ch in zip(raw.ch_names, raw.info["chs"])
                 if np.linalg.norm(ch["loc"][6:9] - ch["loc"][3:6]) < 0.01)
    assert styles[short][1] == "short"
    # a short channel is drawn lighter than either long colour
    assert styles[short][0] not in {"#f28e2b", "#4e79a7"}


def test_a_pairs_shorter_wavelength_is_orange_and_its_longer_blue(raw):
    colours = {name: colour for name, (colour, _) in zip(raw.ch_names, _ch_styles(raw))}
    assert colours["S1_D1 760"] == "#f28e2b"
    assert colours["S1_D1 850"] == "#4e79a7"


def test_an_sci_of_zero_is_kept_rather_than_passed_over():
    assert _lookup_sci({"S1_D1 hbo": 0.0, "S1_D1 hbr": 0.9}, "S1_D1 760") == 0.0


def test_the_dyad_psd_bands_take_the_shared_colours():
    shapes, _ = _psd_band_shapes(cardiac=(0.8, 1.6))
    assert {s["fillcolor"] for s in shapes} <= set(BAND_COLORS.values())
