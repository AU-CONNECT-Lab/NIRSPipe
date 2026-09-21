"""`--tstart`/`--tend`: the window is read out of the transform, never cut from the recording.

Both halves of that sentence are the kind that survives a refactor without being true. A
window that silently cropped the recordings would still produce a coherence, and one taken
out of the whole-run transform differs from one taken out of a cut only in the cone of
influence, which is not visible in a band mean. So the cost of getting it wrong is a number
that is too high by an amount that grows as the window shortens, and nothing raises.

`resolve_analysis_window` is pinned against the recordings it must not touch, and
`window_result` against the cone it must carry rather than recompute.
"""


import numpy as np
import pytest

from fnirs_pipe.exceptions import AlignmentError
from fnirs_pipe.pipeline.hyper import resolve_analysis_window
from fnirs_pipe.pipeline.hyper.synchrony import WTCResult, window_result


class _Untouchable:
    """A recording that raises if anything is read off it but its time axis.

    `crop_aligned_window` reaches for `.copy()`; the run path must not, so a test that only
    checked the returned tuple would pass for either one.
    """

    def __init__(self, duration: float):
        self.times = np.arange(0.0, duration + 0.1, 0.1)

    def __getattr__(self, name):
        raise AssertionError(f"the analysis window touched the recording: .{name}")


def _raws(*durations: float) -> dict:
    return {f"sub-{i:02d}": _Untouchable(d) for i, d in enumerate(durations, start=1)}


# ---- resolving the window ----

def test_no_window_asked_for_is_no_window():
    assert resolve_analysis_window(_raws(600.0, 600.0), None, None) is None


def test_both_ends_given_come_back_as_they_are():
    assert resolve_analysis_window(_raws(600.0, 600.0), 60.0, 300.0) == (60.0, 300.0)


def test_resolving_a_window_does_not_cut_the_recordings():
    """The whole point: `_Untouchable` raises on `.copy()`, which a crop would need."""
    assert resolve_analysis_window(_raws(600.0), 60.0, 300.0) == (60.0, 300.0)


def test_only_a_start_runs_to_the_end_of_the_shortest_member():
    assert resolve_analysis_window(_raws(600.0, 450.0), 60.0, None) == (60.0, 450.0)


def test_only_an_end_starts_at_zero():
    assert resolve_analysis_window(_raws(600.0), None, 300.0) == (0.0, 300.0)


def test_an_over_long_end_is_clipped_rather_than_refused():
    """Recordings differ in length, so "to 9999 s" is a request for "to the end"."""
    assert resolve_analysis_window(_raws(600.0, 450.0), 0.0, 9999.0) == (0.0, 450.0)


def test_a_start_past_the_end_has_no_data_to_describe():
    with pytest.raises(AlignmentError, match="at or past"):
        resolve_analysis_window(_raws(600.0), 700.0, None)


def test_a_start_at_the_shortest_member_raises_even_though_others_are_longer():
    """The window has to name the same moment in every member, so the shortest one rules."""
    with pytest.raises(AlignmentError, match="at or past"):
        resolve_analysis_window(_raws(600.0, 300.0), 400.0, 500.0)


def test_an_end_before_the_start_is_an_empty_window():
    with pytest.raises(AlignmentError, match="empty window"):
        resolve_analysis_window(_raws(600.0), 300.0, 60.0)


# ---- reading the window out of a finished transform ----

FREQS = np.array([0.2, 0.1, 0.05, 0.01])
TIMES = np.arange(0.0, 1200.0, 2.0)


@pytest.fixture
def result():
    rng = np.random.default_rng(0)
    shape = (len(FREQS), len(TIMES))
    # a cone that opens at both ends of the record, which is what a window must inherit
    edge = np.minimum(TIMES - TIMES[0], TIMES[-1] - TIMES)
    return WTCResult(
        pairs={("sub-01", "sub-02"): {"S1_D1": {
            "wtc": rng.random(shape), "phase": rng.random(shape) * 2 - 1,
            "coi": np.sqrt(2.0) * np.maximum(edge, 1e-9), "sig": np.full(len(FREQS), 0.7),
        }}},
        freqs=FREQS, times=TIMES,
    )


def _map(res):
    return res.pairs[("sub-01", "sub-02")]["S1_D1"]


def test_the_window_holds_only_the_times_asked_for(result):
    got = window_result(result, 300.0, 600.0)
    assert got.times.min() >= 300.0 and got.times.max() <= 600.0
    assert _map(got)["wtc"].shape[1] == len(got.times)


def test_the_frequency_axis_is_untouched(result):
    """A window is a column selection. A shorter record reaching fewer scales would be a
    different transform, which is exactly what this route exists to avoid."""
    assert np.array_equal(window_result(result, 300.0, 600.0).freqs, result.freqs)


def test_the_cone_is_the_whole_record_cone_carried_over_not_a_new_one(result):
    """The one that matters. A 300 s stretch cut and transformed alone has a cone reaching
    sqrt(2) * 150 s in from each of its own edges; read out of the whole record it carries
    the cone it already had, which in the middle of a 1200 s recording is wide open."""
    got = window_result(result, 300.0, 600.0)
    keep = (result.times >= 300.0) & (result.times <= 600.0)
    assert np.array_equal(_map(got)["coi"], _map(result)["coi"][keep])
    # and it does not narrow at the window's own edges, which a re-transform would
    assert _map(got)["coi"][0] > 0.5 * _map(got)["coi"].max()


def test_the_monte_carlo_level_is_carried_whole(result):
    """It is per frequency and constant over time, so a window of it is itself."""
    assert np.array_equal(_map(window_result(result, 300.0, 600.0))["sig"],
                          _map(result)["sig"])


def test_the_values_inside_the_window_are_the_ones_the_whole_run_had(result):
    got = window_result(result, 300.0, 600.0)
    keep = (result.times >= 300.0) & (result.times <= 600.0)
    assert np.array_equal(_map(got)["wtc"], _map(result)["wtc"][:, keep])
    assert np.array_equal(_map(got)["phase"], _map(result)["phase"][:, keep])


def test_a_window_holding_no_sample_says_what_the_result_spans(result):
    with pytest.raises(ValueError, match="0.0-1198.0 s"):
        window_result(result, 2000.0, 2500.0)
