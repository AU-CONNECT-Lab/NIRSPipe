"""A phase arrow is drawn only on a cell the pipeline counts, never in the cone's corners."""

import numpy as np
import pytest

from nirspipe.qc.figures.hyper.hyper_post_figures import (
    _arrow_mask,
    build_wtc_map_interactive,
    wtc_condition_views,
)

FREQS = np.array([0.05, 0.1, 0.2])


def test_no_arrow_where_the_cone_leaves_no_frequency():
    # periods: at the edges 0, then 2 s, whose 0.5 Hz floor lies above the band's top
    coi = np.array([0.0, 2.0, 100.0, 2.0, 0.0])
    wtc = np.ones((len(FREQS), len(coi)))
    mask = _arrow_mask(wtc, None, FREQS, coi)
    # only the middle column's 0.01 Hz floor leaves the band measurable
    assert mask.tolist() == [[False, False, True, False, False]] * len(FREQS)


def test_a_column_is_counted_from_its_floor_up():
    coi = np.array([10.0])                     # a 0.1 Hz floor
    mask = _arrow_mask(np.ones((len(FREQS), 1)), None, FREQS, coi)
    assert mask[:, 0].tolist() == [False, True, True]


def _flat_map(coherence, n_times=200):
    wtc = np.full((len(FREQS), n_times), coherence)
    return {"wtc": wtc, "phase": np.zeros_like(wtc), "coi": np.full(n_times, 1e6)}


def _arrows(annotations):
    return [a for a in annotations if a["showarrow"]]


@pytest.mark.parametrize("arrow_min, drawn", [(0.5, True), (0.8, False)])
def test_the_live_panel_and_its_condition_views_draw_arrows_at_the_flag_s_coherence(arrow_min, drawn):
    data, times = _flat_map(0.6), np.arange(200.0)
    fig = build_wtc_map_interactive(data, FREQS, times, "S1_D1", [], {}, arrow_min=arrow_min)
    assert bool(_arrows(fig.layout.annotations)) is drawn
    views = wtc_condition_views(fig, [("task", 50.0, 150.0)], data, FREQS, times,
                                arrow_min=arrow_min)
    assert bool(_arrows(views["task"]["annotations"])) is drawn
