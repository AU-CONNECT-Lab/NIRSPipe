"""The subject connectogram, and the circle it shares with the inter-brain one.

It used to be drawn through a private MNE entry point (`_plot_connectivity_circle`) into a
matplotlib PNG. What is pinned here is what the picture claims: which edges reach the
circle, which channels get a node, and that one circle's geometry serves both reports.
"""

import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.qc.figures.common.circle_map import DYAD_GAP, bezier, ring_angles
from fnirs_pipe.qc.figures.subject.connectogram import _keep_edges, fc_connectogram


def _frame(pairs, chromo="hbo", value=0.5):
    names = [f"{p} {chromo}" for p in pairs]
    m = np.full((len(names), len(names)), value, dtype=float)
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=names, columns=names)


# ---- the circle ----

def test_the_dyad_circle_puts_one_member_a_side():
    """Two members are two groups on the same ring, counterclockwise from the top, so the
    first runs down the left and the second comes back up the right: the two homologous ends
    of a montage then face each other across the split."""
    left, right = np.split(ring_angles([4, 4], gap=DYAD_GAP), 2)
    assert all(90 < a < 270 for a in left)
    assert all(a > 270 or a < 90 for a in (right % 360))


def test_the_hyper_circle_is_drawn_on_the_shared_one():
    """A second copy of the geometry is how the two circles would quietly fork."""
    import inspect

    from fnirs_pipe.qc.figures.hyper import hyper_post_figures as hp

    assert hp.ring_angles is ring_angles
    assert "ring_angles" in inspect.getsource(hp._circle_traces)


def test_a_group_costs_one_gap_and_the_nodes_share_what_is_left():
    a = ring_angles([2, 2], gap=10.0)
    b = ring_angles([4], gap=10.0)
    # four nodes either way, but two groups spend twice the blank circle, so their pitch
    # is tighter
    assert len(a) == len(b) == 4
    assert (a[1] - a[0]) < (b[1] - b[0])
    assert ring_angles([]).size == 0
    assert ring_angles([0, 3], gap=10.0).size == 3


def test_a_chord_bows_toward_the_centre():
    """How far it bows is how far apart its ends are, which is what lets a reader tell a
    neighbour pairing from one across the circle."""
    x, y = bezier((1.0, 0.0), (-1.0, 0.0))
    mid = (x[len(x) // 2], y[len(y) // 2])
    assert abs(mid[0]) < 0.1 and abs(mid[1]) < 0.1


# ---- which edges are drawn ----

def test_a_weak_edge_is_blanked_not_zeroed():
    """A zero is a value: it lands mid-scale and draws a near-white chord, which is the haze
    that used to sit under every real one."""
    m = np.array([[1.0, 0.1, 0.8], [0.1, 1.0, 0.2], [0.8, 0.2, 1.0]])
    out = _keep_edges(m, threshold=0.3, n_lines=None)
    assert np.isnan(out[0, 1]) and out[0, 2] == 0.8
    assert np.isnan(np.diag(out)).all()


def test_top_n_counts_edges_and_not_cells():
    """The matrix is symmetric, so every edge is in it twice."""
    m = np.array([[1.0, 0.9, 0.5, 0.2],
                  [0.9, 1.0, 0.4, 0.1],
                  [0.5, 0.4, 1.0, 0.3],
                  [0.2, 0.1, 0.3, 1.0]])
    out = _keep_edges(m, threshold=0.0, n_lines=2)
    assert np.isfinite(out).sum() == 4          # two edges, each cell twice
    assert out[0, 1] == 0.9 and out[0, 2] == 0.5


# ---- the figure ----

def test_one_circle_per_chromophore():
    fig = fc_connectogram(_frame(["S1_D1", "S1_D2"]), _frame(["S1_D1", "S1_D2"], "hbr"))
    assert [a.text for a in fig.layout.annotations if a.text in ("HbO", "HbR")] == \
           ["HbO", "HbR"]


def test_a_rejected_channel_gets_no_node():
    """It carries no edge, so a node kept for it would claim the ring has a channel the run
    did not measure."""
    frame = _frame(["S1_D1", "S1_D2", "S2_D1"])
    frame.loc["S1_D2 hbo", :] = np.nan
    frame.loc[:, "S1_D2 hbo"] = np.nan
    fig = fc_connectogram(frame)
    labelled = {a.text for a in fig.layout.annotations}
    assert "S1_D1" in labelled and "S1_D2" not in labelled


def test_every_chord_is_drawn_once():
    """Both triangles would lay each chord over itself and double the file for no second
    reading."""
    fig = fc_connectogram(_frame(["S1_D1", "S1_D2", "S2_D1"], value=0.8))
    chords = [t for t in fig.data if t.mode == "lines" and t.line.width == 1.6]
    assert len(chords) == 3          # three pairs among three channels


def test_the_nodes_carry_no_colour_of_their_own():
    """Position already separates the source blocks, so hue there would repeat the gaps and
    compete with the chords, which are what the figure is for."""
    from fnirs_pipe.qc.figures.common.circle_map import NODE_INK

    fig = fc_connectogram(_frame(["S1_D1", "S2_D1", "S3_D1"]))
    nodes = [t for t in fig.data if t.mode == "lines" and t.line.width == 9]
    assert nodes and {t.line.color for t in nodes} == {NODE_INK}


def test_the_title_says_the_rule_and_the_count():
    """The cut is a display choice, not a finding, and a reader has to be able to see it."""
    names = [f"{p} hbo" for p in ("S1_D1", "S1_D2", "S2_D1")]
    m = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.4], [0.7, 0.4, 1.0]])
    frame = pd.DataFrame(m, index=names, columns=names)
    assert "|r| >= 0.3" in fc_connectogram(frame).layout.title.text
    assert "3 drawn" in fc_connectogram(frame).layout.title.text
    assert "top 2" in fc_connectogram(frame, n_lines=2).layout.title.text
    assert "2 drawn" in fc_connectogram(frame, n_lines=2).layout.title.text


def test_one_colour_bar_for_the_whole_figure():
    """The chords are individual lines and none of them can show a scale."""
    fig = fc_connectogram(_frame(["S1_D1", "S1_D2"]), _frame(["S1_D1", "S1_D2"], "hbr"))
    assert sum(1 for t in fig.data if getattr(t.marker, "showscale", False)) == 1


def test_no_recognised_channels_is_an_error_not_an_empty_circle():
    with pytest.raises(ValueError):
        fc_connectogram(pd.DataFrame(np.eye(2), index=["a", "b"], columns=["a", "b"]))
