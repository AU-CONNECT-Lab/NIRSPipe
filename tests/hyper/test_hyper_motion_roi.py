"""The dyad motion panel groups each member's long carpet rows by ROI, as the member's own carpet does."""

import numpy as np
import pytest

from nirspipe.qc.figures.hyper.hyper_figures import build_motion_panel, motion_series
from tests._synth import synth_raw

SIDS = ["sub-01", "sub-02"]
ROI_MAP = {"B": ["S1_D1 hbo"], "A": ["S3_D3 hbo"]}


@pytest.fixture(scope="module")
def raws():
    return {sid: synth_raw(sid.removeprefix("sub-"), "tapping", duration=120.0) for sid in SIDS}


@pytest.fixture(scope="module")
def motions(raws):
    return motion_series(raws, None, SIDS), motion_series(raws, None, SIDS, roi_map=ROI_MAP)


def test_each_member_s_long_rows_follow_the_map(motions):
    plain, grouped = motions
    for (sid, z, _, labels, _), (_, z0, _, labels0, _) in zip(grouped["carpets"]["before"],
                                                              plain["carpets"]["before"]):
        assert labels[:2] == ["S3_D3 760", "S3_D3 850"]
        assert labels[2:4] == ["S1_D1 760", "S1_D1 850"]
        assert sorted(labels) == sorted(labels0)
        # every row still carries its own channel
        rows0 = dict(zip(labels0, np.asarray(z0)))
        assert all(np.array_equal(row, rows0[c], equal_nan=True)
                   for c, row in zip(labels, np.asarray(z)))
        assert grouped["roi"][sid][:2] == [("A", 0, 1), ("B", 2, 3)]
    assert plain["roi"] == {sid: [] for sid in SIDS} and plain["roi_colours"] == {}


def test_the_panel_strips_every_member_s_carpet_and_names_the_rois(motions):
    plain, grouped = motions
    fig = build_motion_panel(grouped, "before")
    legend = {t.name: t.marker.color for t in fig.data
              if t.type == "scatter" and t.marker.symbol == "square"}
    assert list(legend)[:2] == ["A", "B"]
    carpets = [f"{t.yaxis} domain" for t in fig.data if t.type == "heatmap"]
    strip = [sh for sh in fig.layout.shapes if sh.type == "rect" and sh.x1 < -0.015]
    assert len(carpets) == 2 and sorted({sh.yref for sh in strip}) == sorted(carpets)
    for yref in carpets:
        top = max((sh for sh in strip if sh.yref == yref), key=lambda sh: sh.y1)
        assert top.fillcolor == legend["A"]

    bare = build_motion_panel(plain, "before")
    assert not [sh for sh in bare.layout.shapes if sh.type == "rect" and sh.x1 < -0.015]
    assert not [t for t in bare.data if t.type == "scatter" and t.marker.symbol == "square"]
