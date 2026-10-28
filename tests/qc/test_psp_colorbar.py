"""A channel grid whose PSP row has no line carries that row's ramp as a colour bar."""

from nirspipe.qc.figures.subject.sci_psp_panel import (
    channel_quality_heatmap, condition_quality_heatmap)

NAMES = ["S1_D1 760", "S1_D1 850", "S2_D1 760", "S2_D1 850"]


def _args(psp):
    return {"ch_names": NAMES, "is_bad": [False] * 4,
            "sci_per_ch": dict.fromkeys(NAMES, 0.9), "cv_per_ch": dict.fromkeys(NAMES, 0.05),
            "snr_per_ch": dict.fromkeys(NAMES, 20.0), "psp_per_ch": dict(zip(NAMES, psp))}


def _bars(fig):
    return [t for t in fig.data if t.marker is not None and t.marker.showscale]


def test_the_bar_spans_the_psp_ramp_the_cells_are_drawn_on():
    fig = channel_quality_heatmap(sci_thresh=0.8, psp_thresh=None, **_args([0.0, 0.0, 0.5, 0.5]))
    (bar,) = _bars(fig)
    assert (bar.marker.cmin, bar.marker.cmax) == (0.0, 0.1)
    scale = bar.marker.colorscale
    assert (scale[0][1].upper(), scale[-1][1].upper()) == ("#F8786E", "#C5E0B3")
    assert bar.marker.colorbar.title.text == "PSP"
    assert list(bar.marker.colorbar.ticktext) == ["0", "0.05", "≥0.1"]
    cells = dict(zip(fig.data[0].text, fig.data[0].marker.color))
    psp_cells = {t.split(" · ")[0]: c for t, c in cells.items() if "PSP" in t}
    # the bar's two ends, as the cells at 0 and past 0.1 are drawn
    assert psp_cells["S1_D1 760"] == "rgb(248, 120, 110)"
    assert psp_cells["S2_D1 760"] == "rgb(197, 224, 179)"
    assert fig.layout.margin.r == 80


def test_no_bar_when_psp_has_a_line():
    fig = channel_quality_heatmap(sci_thresh=0.8, psp_thresh=0.1, **_args([0.0, 0.0, 0.5, 0.5]))
    assert not _bars(fig)
    assert fig.layout.margin.r == 20


def test_the_condition_grid_carries_the_same_bar():
    conditions = [("rest", _args([0.0] * 4)), ("task", _args([0.5] * 4))]
    fig = condition_quality_heatmap(conditions, sci_thresh=0.8, psp_thresh=None)
    (bar,) = _bars(fig)
    assert (bar.marker.cmin, bar.marker.cmax) == (0.0, 0.1)
    assert not _bars(condition_quality_heatmap(conditions, sci_thresh=0.8, psp_thresh=0.1))
