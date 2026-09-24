"""Cutting conditions into equal-length windows, and what the cut has to preserve.

Equal spans make every window the same estimation problem. These hold the two properties
the cut exists for, equal spans and a recoverable origin, plus the refusals that keep a
window of some other length out of the result.
"""

import pytest

from fnirs_pipe.qc.common.windows import split_windows


def test_a_long_condition_yields_whole_windows_and_a_short_one_yields_one():
    windows, sources = split_windows(
        [("baseline", 0.0, 300.0), ("game1", 500.0, 1400.0)], 300.0)
    assert [w[0] for w in windows] == [
        "baseline-w1", "game1-w1", "game1-w2", "game1-w3"]
    assert sources["game1-w2"] == ("game1", 300.0)
    assert sources["baseline-w1"] == ("baseline", 0.0)


def test_every_window_has_exactly_the_asked_length():
    windows, _ = split_windows([("game1", 521.306, 1421.279)], 299.9)
    for _, t0, t1 in windows:
        assert t1 - t0 == pytest.approx(299.9)


def test_the_windows_do_not_overlap_and_start_at_the_condition():
    windows, _ = split_windows([("game1", 500.0, 1400.0)], 300.0)
    assert windows[0][1] == 500.0
    for (_, _, prev_stop), (_, start, _) in zip(windows, windows[1:]):
        assert start == pytest.approx(prev_stop)


def test_the_remainder_is_dropped_rather_than_kept_short():
    """A short tail would reintroduce the length difference the split removes."""
    windows, _ = split_windows([("game1", 0.0, 1000.0)], 300.0)
    assert len(windows) == 3
    assert windows[-1][2] == pytest.approx(900.0)      # the last 100 s is not a window


def test_a_condition_too_short_for_one_window_yields_nothing():
    windows, sources = split_windows(
        [("blip", 0.0, 299.9), ("game1", 500.0, 1400.0)], 300.0)
    assert not any(label.startswith("blip") for label, _, _ in windows)
    assert not any(k.startswith("blip") for k in sources)


def test_the_boundary_is_exact_so_a_hair_short_is_one_window_fewer():
    """A trigger written a few ms early decides whether a block yields a window at all.

    Which is why the length is the caller's to choose: only the caller knows how much
    slack its own triggers have.
    """
    assert len(split_windows([("c", 0.0, 300.0)], 300.0)[0]) == 1
    assert split_windows([("c", 0.0, 299.999)], 300.0)[0] == []
    assert len(split_windows([("c", 0.0, 599.999)], 300.0)[0]) == 1


def test_a_repeated_description_keeps_its_own_numbering_as_the_origin():
    """`condition_windows` numbers a repeat `desc#2`; the origin has to stay that, not `desc`."""
    windows, sources = split_windows(
        [("game#1", 0.0, 600.0), ("game#2", 900.0, 1500.0)], 300.0)
    assert sources["game#1-w2"] == ("game#1", 300.0)
    assert sources["game#2-w1"] == ("game#2", 0.0)


def test_a_non_positive_length_is_refused():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive"):
            split_windows([("game1", 0.0, 900.0)], bad)


def test_an_empty_window_list_is_not_an_error():
    assert split_windows([], 300.0) == ([], {})
