"""The circle the inter-brain connectogram is drawn on.

What is pinned here is the geometry: where a node sits on the ring, what a group costs, and
that the connectogram in ``hyper_post_figures`` is drawn on this circle rather than on a
second copy of it.
"""

import numpy as np

from fnirs_pipe.qc.figures.common.circle_map import DYAD_GAP, bezier, ring_angles


def test_the_dyad_circle_puts_one_member_a_side():
    """Two members are two groups on the same ring, counterclockwise from the top, so the
    first runs down the left and the second comes back up the right: the two homologous ends
    of a montage then face each other across the split."""
    left, right = np.split(ring_angles([4, 4], gap=DYAD_GAP), 2)
    assert all(90 < a < 270 for a in left)
    assert all(a > 270 or a < 90 for a in (right % 360))


def test_the_hyper_circle_is_drawn_on_the_shared_one():
    """A second copy of the geometry is how the circle would quietly fork off this module."""
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
