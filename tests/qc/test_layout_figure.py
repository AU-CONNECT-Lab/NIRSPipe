"""The montage figures: what each channel is drawn as.

The colours follow the one long/short split the reports use, so a channel in the gap between
the two ranges is neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from nirspipe.qc.figures.common._utils import (BAND_COLORS, LONG_COLOR, SHORT_COLOR,
                                                  UNCLASSIFIED_COLOR)
from nirspipe.qc.figures.hyper.hyper_figures import _psd_band_shapes
from nirspipe.qc.figures.subject.brain_views import _lookup_sci
from nirspipe.qc.figures.subject.raw_figures import (_ch_colors, _hex_to_rgba,
                                                        build_layout_figure)
from tests._synth import synth_raw


@pytest.fixture
def raw():
    return synth_raw("01", "hold", duration=20.0, motion_onset=None)


def test_a_bad_channel_stays_on_the_2d_map_and_is_drawn_red_line_and_dot(raw):
    bad = raw.ch_names[0]
    raw.info["bads"] = [bad]
    fig_2d = build_layout_figure(raw, {bad}, {}, sci_threshold=0.8)
    channels = next(t for t in fig_2d.data if t.name == "channels")

    assert bad in list(channels.customdata)
    assert channels.marker.color[list(channels.customdata).index(bad)] == "#e74c3c"
    # with no SCI to grade the rest, the rejected pair's line is the one red line
    reds = [t for t in fig_2d.data if t.mode == "lines" and t.line.color == "#e74c3c"]
    assert len(reds) == 1 and len(reds[0].x) == 3


def test_channel_colours_follow_the_long_short_split(raw):
    # move the first pair's detector into the gap the two ranges leave between them
    for ch in raw.info["chs"][:2]:
        ch["loc"][6:9] = ch["loc"][3:6] + np.array([0.012, 0.0, 0.0])
    colours = dict(zip(raw.ch_names, _ch_colors(raw)))

    assert colours[raw.ch_names[0]] == _hex_to_rgba(UNCLASSIFIED_COLOR, 0.78)
    assert colours[raw.ch_names[2]] == _hex_to_rgba(LONG_COLOR, 0.78)
    short = next(n for n, ch in zip(raw.ch_names, raw.info["chs"])
                 if np.linalg.norm(ch["loc"][6:9] - ch["loc"][3:6]) < 0.01)
    assert colours[short] == _hex_to_rgba(SHORT_COLOR, 0.78)


def test_an_sci_of_zero_is_kept_rather_than_passed_over():
    assert _lookup_sci({"S1_D1 hbo": 0.0, "S1_D1 hbr": 0.9}, "S1_D1 760") == 0.0


def test_the_dyad_psd_bands_take_the_shared_colours():
    shapes, _ = _psd_band_shapes(cardiac=(0.8, 1.6))
    assert {s["fillcolor"] for s in shapes} <= set(BAND_COLORS.values())
