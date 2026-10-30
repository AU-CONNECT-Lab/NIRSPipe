"""The dyad's head slider: the window verdict, then the SCI and PSP behind it, on the run's lines."""

import numpy as np
import pytest

from nirspipe.qc.figures.common.head_map import head_geometry
from nirspipe.qc.figures.common.head_map import build_head_slider
from nirspipe.qc.metrics.hyper import passing_grid, member_series
from nirspipe.utils import pair_of
from tests._synth import synth_raw

SIDS = ["sub-01", "sub-02"]
CENTERS = np.array([5.0, 15.0, 25.0])
LINES = {"sci": 0.7, "psp": 0.15}
# the window where a movement holds one pair's SCI up and drives its PSP down
MOVED = 1


def _screened(order, moved):
    """Screening windows with every pair passing except ``moved`` in window MOVED."""
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
    grid = passing_grid(sqm, SIDS)
    series = {sid: member_series(sqm, sid, grid) for sid in SIDS}
    geo = head_geometry(raw, pairs)
    fig = build_head_slider({sid: geo for sid in SIDS}, SIDS, grid, series, None, cutoffs,
                            step=1)
    return fig, geo, pairs[0]


@pytest.fixture(scope="module")
def slider():
    return _slider(LINES)


def _titles(fig) -> tuple[list[str], list[str]]:
    """Column titles left to right, row names top down."""
    layout = fig.layout.to_plotly_json()
    columns = [a["text"] for a in sorted(layout["annotations"], key=lambda a: a["x"]) if a["text"]]
    titled = [k for k in layout if k.startswith("yaxis") and layout[k].get("title")]
    rows = [layout[k]["title"]["text"] for k in sorted(titled, key=lambda k: int(k[5:] or 1))]
    return columns, rows


def _coloured(fig, row: int, col: int) -> int:
    """The trace carrying the colours of the head at (row, col): a member's row, a number's column."""
    axis = fig.get_subplot(row, col).xaxis.plotly_name.replace("axis", "")
    return next(i for i, tr in enumerate(fig.data)
                if tr.xaxis == axis and tr.marker.colorscale is not None)


def test_each_member_is_a_row_reading_passing_then_sci_then_psp(slider):
    fig, _, _ = slider
    assert _titles(fig) == (["Passing", "SCI", "PSP"], SIDS)


def test_a_moved_window_is_not_passing_while_its_sci_stays_over_the_line(slider):
    fig, geo, moved = slider
    per_pair = geo["long"]["per_pair"]
    first = geo["long"]["names"].index(moved)
    at = int(np.sum(per_pair[:first]))
    other = at + int(per_pair[first])           # the next pair's first marker, never moved
    frame = fig.frames[MOVED]
    for col, (in_moved, elsewhere) in {1: (0.0, 1.0), 2: (0.95, 0.9), 3: (0.02, 0.5)}.items():
        for row, sid in enumerate(SIDS, start=1):
            i = _coloured(fig, row, col)
            colour = np.asarray(frame.data[list(frame.traces).index(i)].marker.color)
            assert colour[at] == pytest.approx(in_moved, abs=1e-6), (col, sid)
            assert colour[other] == pytest.approx(elsewhere, abs=1e-6), (col, sid)
    words = frame.data[list(frame.traces).index(_coloured(fig, 1, 1))].customdata
    assert (words[at], words[other]) == ("no", "yes")


def test_sci_and_psp_are_scaled_with_the_run_s_lines_mid_way(slider):
    fig, _, _ = slider
    sci = fig.data[_coloured(fig, 1, 2)].marker
    psp = fig.data[_coloured(fig, 1, 3)].marker
    assert (sci.cmin, sci.cmax) == pytest.approx((0.4, 1.0))
    assert (psp.cmin, psp.cmax) == pytest.approx((0.0, 0.3))


def test_psp_with_no_line_runs_from_zero_to_the_cap_on_the_heads_scale():
    fig, _, _ = _slider({"sci": 0.7})
    assert _titles(fig)[0] == ["Passing", "SCI", "PSP"]
    psp = fig.data[_coloured(fig, 1, 3)].marker
    assert (psp.cmin, psp.cmax) == pytest.approx((0.0, 0.1))
    # the Passing and SCI heads' scale, low red to high green
    assert psp.colorscale[0][1] == "#F8786E" and psp.colorscale[-1][1] == "#C5E0B3"


def test_sci_with_no_line_gets_no_column():
    fig, _, _ = _slider({"psp": 0.15})
    assert _titles(fig)[0] == ["Passing", "PSP"]


def test_a_run_without_events_titles_each_frame_by_time_alone(slider):
    fig, _, _ = slider
    titles = [frame.layout.title.text for frame in fig.frames]
    assert titles[0] == "<span style='color:#8a949e'>t = 5 s</span>"
    assert not [t for t in titles if "between blocks" in t]
