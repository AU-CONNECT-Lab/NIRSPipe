"""The seed map's route from run_post to the report figure.

`compute_fc_seed` and its TSVs were covered by test_rest_outputs; nothing carried the frames
any further, so the report could not draw them at all. These pin the two halves of that route:
run_post handing the frames back, and the figure turning them into a flat map.

The figure returns a base64 PNG, so its content cannot be asserted directly. What is asserted
instead is the geometry that carries the meaning: one row of panels per seed ROI, one column
per chromophore, and None rather than an exception when the montage has no optode positions.
"""

import base64
import io

import numpy as np
import pytest

from fnirs_pipe.pipeline.post_pipeline import PostConfig, run_post
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.pipeline.restingstate import compute_fc_seed
from fnirs_pipe.qc.figures.rest_figures import _channel_endpoints, fc_seed_topo_figure

from ._synth import synth_raw

_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
_FC_SEED = 7  # run_post: (result, glm_est, dm, alff_df, fc_df, fc_hbr_df, gcor_reg, fc_seed)


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


def _png_size(b64: str) -> tuple[int, int]:
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


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

def test_figure_returns_a_png(haemo, seed_frames):
    b64 = fc_seed_topo_figure(haemo, seed_frames["hbo"], seed_frames["hbr"])
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"


def test_figure_grows_a_row_per_seed(haemo, roi_map):
    hbo = [c for c in haemo.ch_names if c.endswith("hbo")]
    one = compute_fc_seed(haemo, {"left": hbo[:2]}, "hbo")
    two = compute_fc_seed(haemo, {"left": hbo[:2], "right": hbo[2:4]}, "hbo")
    assert _png_size(fc_seed_topo_figure(haemo, two))[1] > \
           _png_size(fc_seed_topo_figure(haemo, one))[1]


def test_figure_grows_a_column_for_the_second_chromophore(haemo, seed_frames):
    hbo_only = _png_size(fc_seed_topo_figure(haemo, seed_frames["hbo"]))[0]
    both = _png_size(fc_seed_topo_figure(haemo, seed_frames["hbo"], seed_frames["hbr"]))[0]
    assert both > hbo_only


def test_figure_is_none_without_optode_positions(haemo, seed_frames):
    """A montage with no coordinates must skip the figure, not raise inside matplotlib."""
    flat = haemo.copy()
    for ch in flat.info["chs"]:
        ch["loc"] = np.zeros(12)
    assert fc_seed_topo_figure(flat, seed_frames["hbo"]) is None


def test_figure_is_none_with_no_frames(haemo):
    assert fc_seed_topo_figure(haemo, None, None) is None


def test_endpoints_skip_channels_with_no_location(haemo):
    flat = haemo.copy()
    flat.info["chs"][0]["loc"] = np.zeros(12)
    ends = _channel_endpoints(flat)
    assert flat.ch_names[0] not in ends
    assert flat.ch_names[1] in ends


def test_endpoints_cover_every_positioned_channel(haemo):
    assert set(_channel_endpoints(haemo)) == set(haemo.ch_names)


# ---- the head outline moved to _utils, so the figure that had it inline must still draw ----

def test_optode_layout_still_renders_after_the_outline_moved(haemo):
    from fnirs_pipe.qc.figures import optode_layout_static
    b64 = optode_layout_static(haemo, {c: 0.9 for c in haemo.ch_names}, [])
    assert base64.b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"
