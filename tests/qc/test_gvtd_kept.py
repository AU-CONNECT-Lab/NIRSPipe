"""GVTD is taken over the kept channels: a rejected channel's artefact moves nothing."""

import pytest

from nirspipe.pipeline.prep_pipeline import intensity_to_od
from nirspipe.qc.metrics import _motion_metrics, gvtd_above_segments, gvtd_kept, with_bads
from tests._synth import synth_raw


def _od_with_jump(channel: str):
    od = intensity_to_od(synth_raw("01", "rest"))
    data = od._data
    i = od.ch_names.index(channel)
    # a large step halfway through one channel, the kind of artefact GVTD is there to catch
    data[i, data.shape[1] // 2:] += 5.0
    return od


def test_kept_drops_the_marked_channels_in_order():
    od = intensity_to_od(synth_raw("01", "rest"))
    a, b, c = od.ch_names[:3]
    assert gvtd_kept(with_bads(od, [b]), [a, b, c]) == [a, c]


def test_a_rejected_channel_is_the_same_as_an_absent_one():
    od = _od_with_jump("S1_D1 760")
    marked = _motion_metrics(with_bads(od, ["S1_D1 760", "S1_D1 850"]))
    absent = _motion_metrics(od.copy().drop_channels(["S1_D1 760", "S1_D1 850"]))
    for key in ("gvtd_filt_mean", "gvtd_filt_p95", "gvtd_thresh", "gvtd_pct_above_thresh"):
        assert marked[key] == pytest.approx(absent[key])


def test_the_artefact_flags_time_only_while_its_channel_is_kept():
    od = _od_with_jump("S1_D1 760")
    assert gvtd_above_segments(od)
    assert gvtd_above_segments(with_bads(od, ["S1_D1 760"])) == gvtd_above_segments(
        od.copy().drop_channels(["S1_D1 760"]))


def test_every_channel_rejected_leaves_no_gvtd():
    od = intensity_to_od(synth_raw("01", "rest"))
    out = _motion_metrics(with_bads(od, od.ch_names))
    assert out["gvtd_filt_mean"] is None and out["gvtd_pct_above_thresh"] is None
