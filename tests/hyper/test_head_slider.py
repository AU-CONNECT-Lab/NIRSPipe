"""The dyad's head slider: the coupling verdict, then the SCI and PSP behind it, on the run's lines."""

import numpy as np
import pytest

from nirspipe.qc.figures.common.head_map import head_geometry
from nirspipe.qc.figures.hyper.hyper_figures import build_head_slider
from nirspipe.qc.metrics.hyper import coupled_grid, member_series
from nirspipe.utils import pair_of
from tests._synth import synth_raw

SIDS = ["sub-01", "sub-02"]
CENTERS = np.array([5.0, 15.0, 25.0])
LINES = {"sci": 0.7, "psp": 0.15}
# the window where a movement holds one pair's SCI up and drives its PSP down
MOVED = 1


def _screened(order, moved):
    """Screening windows with every pair coupled except ``moved`` in window MOVED."""
    n_ch, n_win = len(order), len(CENTERS)
    sci, psp = np.full((n_ch, n_win), 0.9), np.full((n_ch, n_win), 0.5)
    mask = np.ones((n_ch, n_win), dtype=bool)
    for i, ch in enumerate(order):
        if pair_of(ch) == moved:
            sci[i, MOVED], psp[i, MOVED], mask[i, MOVED] = 0.95, 0.02, False
    return {"screen_windows": {"mask": mask, "centers": CENTERS, "sci": sci, "psp": psp,
                               "channel_order": order}}


def _slider(cutoffs):
    raw = synth_raw("01", "tapping")
    order = list(raw.ch_names)
    pairs = list(dict.fromkeys(pair_of(ch) for ch in order))
    sqm = {sid: _screened(order, pairs[0]) for sid in SIDS}
    grid = coupled_grid(sqm, SIDS, {})
    series = {sid: member_series(sqm, sid, grid) for sid in SIDS}
    geo = head_geometry(raw, pairs)
    fig = build_head_slider({sid: geo for sid in SIDS}, SIDS, grid, series, None, cutoffs,
                            step=1)
    return fig, geo, pairs[0]


@pytest.fixture(scope="module")
def slider():
    return _slider(LINES)


def _row_titles(fig) -> list[str]:
    layout = fig.layout.to_plotly_json()
    titled = [k for k in layout if k.startswith("yaxis") and layout[k].get("title")]
    return [layout[k]["title"]["text"] for k in sorted(titled, key=lambda k: int(k[5:] or 1))]


def _coloured(fig, row: int, col: int) -> int:
    """The trace carrying the colours of the head at (row, col)."""
    axis = fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")
    return next(i for i, tr in enumerate(fig.data)
                if tr.xaxis == axis and tr.marker.colorscale is not None)


def test_the_rows_read_coupled_then_sci_then_psp(slider):
    fig, _, _ = slider
    assert _row_titles(fig) == ["Coupled", "SCI", "PSP"]


def test_a_moved_window_is_not_coupled_while_its_sci_stays_over_the_line(slider):
    fig, geo, moved = slider
    per_pair = geo["long"]["per_pair"]
    first = geo["long"]["names"].index(moved)
    at = int(np.sum(per_pair[:first]))
    other = at + int(per_pair[first])           # the next pair's first marker, never moved
    frame = fig.frames[MOVED]
    for row, (in_moved, elsewhere) in {1: (0.0, 1.0), 2: (0.95, 0.9), 3: (0.02, 0.5)}.items():
        for col, sid in enumerate(SIDS, start=1):
            i = _coloured(fig, row, col)
            colour = np.asarray(frame.data[list(frame.traces).index(i)].marker.color)
            assert colour[at] == pytest.approx(in_moved, abs=1e-6), (row, sid)
            assert colour[other] == pytest.approx(elsewhere, abs=1e-6), (row, sid)
    words = frame.data[list(frame.traces).index(_coloured(fig, 1, 1))].customdata
    assert (words[at], words[other]) == ("no", "yes")


def test_sci_and_psp_are_scaled_with_the_run_s_lines_mid_way(slider):
    fig, _, _ = slider
    sci = fig.data[_coloured(fig, 2, 1)].marker
    psp = fig.data[_coloured(fig, 3, 1)].marker
    assert (sci.cmin, sci.cmax) == pytest.approx((0.4, 1.0))
    assert (psp.cmin, psp.cmax) == pytest.approx((0.0, 0.3))


def test_a_number_with_no_line_gets_no_row():
    fig, _, _ = _slider({"sci": 0.7})
    assert _row_titles(fig) == ["Coupled", "SCI"]
