"""The seed map's route from run_post to the report figure.

These pin the two halves of that route: run_post handing the frames back, and the figure
turning them into a flat map.

The figure is a Plotly figure, so what is asserted is the geometry that carries the meaning:
one row of panels per seed ROI, one column per chromophore, and None rather than an exception
when the montage has no optode positions.
"""

import base64

import numpy as np
import plotly.graph_objects as go
import pytest

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.pipeline.restingstate import compute_fc_seed
from fnirs_pipe.qc.figures.subject.rest_figures import _head_for, fc_seed_topo_figure

from tests._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
_FC_SEED = 6  # run_post: (result, glm_est, dm, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi)


@pytest.fixture(scope="module")
def haemo(tmp_path_factory):
    config = PrepConfig(subject="01", dpf=[6.0, 6.0], sci_threshold=0.8,
                        motion_correction="tddr", **_BANDS)
    result = run_prep(synth_raw("01", "rest"), config,
                      output_dir=tmp_path_factory.mktemp("seedfig_prep"),
                      source_entities={"task": "rest"})
    return result.raw_haemo


@pytest.fixture(scope="module")
def roi_map(haemo):
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    return {"left": hbo[:2], "right": hbo[2:4]}


@pytest.fixture(scope="module")
def seed_frames(haemo, roi_map):
    return {c: compute_fc_seed(haemo, roi_map, c) for c in ("hbo", "hbr")}


def _panel_titles(fig) -> list[str]:
    """The subplot titles, which name one panel each and so count the grid."""
    return [a.text for a in fig.layout.annotations if a.text]


# ---- run_post hands the frames back ----

def test_run_post_returns_a_seed_frame_per_chromophore(haemo, roi_map, tmp_path):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        drift_model="polynomial", drift_order=1, drift_high_pass=0.01,
                        roi_map=roi_map, **_BANDS)
    fc_seed = run_post(haemo.copy(), config, output_dir=tmp_path, mode="rest",
                       source_entities={"task": "rest"})[_FC_SEED]
    assert set(fc_seed) == {"hbo", "hbr"}
    for chromo, frame in fc_seed.items():
        assert list(frame.index) == list(roi_map)
        assert all(c.endswith(f" {chromo}") for c in frame.columns)


def test_run_post_returns_no_seed_frame_without_a_roi_map(haemo, tmp_path):
    config = PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                        drift_model="polynomial", drift_order=1, drift_high_pass=0.01,
                        **_BANDS)
    fc_seed = run_post(haemo.copy(), config, output_dir=tmp_path, mode="rest",
                       source_entities={"task": "rest"})[_FC_SEED]
    assert fc_seed == {}


# ---- the figure ----

def test_figure_is_a_plotly_figure(haemo, seed_frames):
    fig = fc_seed_topo_figure(haemo, seed_frames["hbo"], seed_frames["hbr"])
    assert isinstance(fig, go.Figure) and len(fig.data)


def test_figure_grows_a_row_per_seed(haemo):
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    one = compute_fc_seed(haemo, {"left": hbo[:2]}, "hbo")
    two = compute_fc_seed(haemo, {"left": hbo[:2], "right": hbo[2:4]}, "hbo")
    assert _panel_titles(fc_seed_topo_figure(haemo, one)) == ["left - HbO"]
    assert _panel_titles(fc_seed_topo_figure(haemo, two)) == ["left - HbO", "right - HbO"]
    assert (fc_seed_topo_figure(haemo, two).layout.height
            > fc_seed_topo_figure(haemo, one).layout.height)


def test_figure_grows_a_column_for_the_second_chromophore(haemo, seed_frames):
    hbo_only = _panel_titles(fc_seed_topo_figure(haemo, seed_frames["hbo"]))
    both = _panel_titles(fc_seed_topo_figure(haemo, seed_frames["hbo"], seed_frames["hbr"]))
    assert all(t.endswith("HbO") for t in hbo_only)
    assert len(both) == 2 * len(hbo_only)
    # one row is one seed either way, so the second chromophore is a column and not a row
    assert fc_seed_topo_figure(haemo, seed_frames["hbo"]).layout.height ==            fc_seed_topo_figure(haemo, seed_frames["hbo"], seed_frames["hbr"]).layout.height


def test_one_colour_bar_for_the_whole_grid(haemo, seed_frames):
    """Every panel is on the fixed +-1 scale, so a bar per panel would be the same bar."""
    fig = fc_seed_topo_figure(haemo, seed_frames["hbo"], seed_frames["hbr"])
    assert sum(1 for t in fig.data if getattr(t.marker, "showscale", False)) == 1


def test_a_seed_s_own_channels_are_drawn_without_a_colour(haemo, roi_map, seed_frames):
    """They sit inside the average, so their correlation says nothing; grey is the claim
    that none is being made, where a blue channel would claim no connection."""
    from fnirs_pipe.qc.figures.common.head_map import BLANK_COLOR

    fig = fc_seed_topo_figure(haemo, seed_frames["hbo"])
    assert any(t.marker.color == BLANK_COLOR for t in fig.data)


def test_figure_is_none_without_optode_positions(haemo, seed_frames):
    """A montage with no coordinates must skip the figure, not raise inside the projection."""
    flat = haemo.copy()
    for ch in flat.info["chs"]:
        ch["loc"] = np.zeros(12)
    assert fc_seed_topo_figure(flat, seed_frames["hbo"]) is None


def test_figure_is_none_with_no_frames(haemo):
    assert fc_seed_topo_figure(haemo, None, None) is None


def test_the_head_drops_a_pair_with_no_location(haemo):
    """The head draws pairs, so a pair goes only when neither of its channels is positioned:
    the optode coordinates are a property of the montage, not of a chromophore."""
    flat = haemo.copy()
    pair = flat.ch_names[0].split(" ")[0]
    for ch in flat.info["chs"]:
        if ch["ch_name"].startswith(pair):
            ch["loc"] = np.zeros(12)
    assert pair not in _head_for(flat, None)["long"]["names"]


def test_the_head_carries_the_long_channels_and_only_those(haemo):
    """A short channel measures extracerebral signal, so a correlation with it is not a
    connectivity claim and it is left off every map in this module."""
    from fnirs_pipe.qc.metrics import long_short_channels

    long_names, short_names = long_short_channels(haemo, None)
    drawn = set(_head_for(haemo, None)["long"]["names"])
    assert drawn == {c.split(" ")[0] for c in long_names}
    assert drawn.isdisjoint({c.split(" ")[0] for c in short_names})


# ---- the optode layout draws on the shared head outline ----

def test_optode_layout_still_renders_after_the_outline_moved(haemo):
    from fnirs_pipe.qc.figures import optode_layout_static
    b64 = optode_layout_static(haemo, {c: 0.9 for c in haemo.ch_names}, [], 0.8)
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"
