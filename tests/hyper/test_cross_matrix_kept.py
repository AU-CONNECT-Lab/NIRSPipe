"""A dyad's cross-brain matrices draw only the sites with a value; the tables keep them all."""

import numpy as np

from nirspipe.qc.figures.hyper.hyper_post_figures import build_isc_panel, build_isc_roi_matrix

SUBJECTS = ["sub-01", "sub-02"]


def _heat(fig):
    return next(t for t in fig.data if t.type == "heatmap")


def test_a_site_one_member_lost_leaves_that_member_s_axis_only():
    z = np.linspace(-0.5, 0.5, 16).reshape(4, 4)
    z[:, 2] = np.nan                       # the second member lost its third site
    heat = _heat(build_isc_panel(z, ["A", "B", "C", "D"], SUBJECTS))
    assert list(heat.y) == ["A", "B", "C", "D"]
    assert list(heat.x) == ["A", "B", "D"]
    assert np.isfinite(np.asarray(heat.z, dtype=float)).all()


def test_the_null_level_is_cut_with_the_matrix():
    z = np.full((3, 3), 0.4)
    z[1, :] = np.nan
    level = np.full((3, 3), 0.3)
    fig = build_isc_panel(z, ["A", "B", "C"], SUBJECTS, arc_level=level)
    # every kept cell clears its level, so the rule counts the 2 x 3 that are left
    assert "(6 of 6)" in fig.layout.annotations[1].text


def test_a_region_neither_panel_measured_leaves_the_roi_matrix():
    z = np.full((3, 3), 0.2)
    z[1, :] = np.nan
    fig = build_isc_roi_matrix({"HbO": (z, ["L", "M", "R"]), "HbR": (z.copy(), ["L", "M", "R"])},
                               SUBJECTS)
    for heat in (t for t in fig.data if t.type == "heatmap"):
        assert list(heat.y) == ["L", "R"] and list(heat.x) == ["L", "M", "R"]
    assert not any(s.fillcolor for s in fig.layout.shapes)
