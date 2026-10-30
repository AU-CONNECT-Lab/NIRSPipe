"""A phase arrow is drawn only on a cell the pipeline counts, never in the cone's corners."""

import numpy as np

from nirspipe.qc.figures.hyper.hyper_post_figures import _arrow_mask

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
