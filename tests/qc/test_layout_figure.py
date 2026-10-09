"""The montage figures of the raw report: where the lines run, and what each channel is drawn as.

The 3-D lines are read off MNE's channel loc, whose first three values are the channel
midpoint and the next six the source and detector. The colours follow the one long/short
split the reports use, so a channel in the gap between the two ranges is neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from nirspipe.qc.figures.common._brain_utils import to_mni
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


@pytest.fixture
def no_brain_mesh(monkeypatch):
    """The fsaverage mesh is decoration here, and fetching it is a download."""
    import nilearn.datasets

    def _refuse(*args, **kwargs):
        raise OSError("mesh not needed for this test")
    monkeypatch.setattr(nilearn.datasets, "fetch_surf_fsaverage", _refuse)


def test_the_3d_lines_run_from_source_to_detector(raw, no_brain_mesh):
    _, fig_3d = build_layout_figure(raw, set(), {}, sci_threshold=0.8)
    lines = next(t for t in fig_3d.data if t.type == "scatter3d" and t.mode == "lines")
    first = np.array([[lines.x[0], lines.y[0], lines.z[0]],
                      [lines.x[1], lines.y[1], lines.z[1]]])

    loc = raw.info["chs"][0]["loc"]
    expected = to_mni(np.array([loc[3:6], loc[6:9]]), raw.info)
    np.testing.assert_allclose(first, expected, atol=1e-6)


def test_a_bad_channel_stays_on_the_2d_map_and_is_drawn_grey(raw, no_brain_mesh):
    bad = raw.ch_names[0]
    raw.info["bads"] = [bad]
    fig_2d, _ = build_layout_figure(raw, {bad}, {}, sci_threshold=0.8)
    channels = fig_2d.data[1]

    assert bad in list(channels.customdata)
    assert channels.marker.color[list(channels.customdata).index(bad)] == "#949e9f"


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
