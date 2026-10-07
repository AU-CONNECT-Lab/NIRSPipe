"""Channel screening by counting coupled windows.

The rule these cover: SCI and PSP are thresholded inside the same short window and
combined there, and a channel is kept when enough windows pass.
Two properties matter and neither is obvious from the code.

The pairing is temporal. Movement inflates SCI and drives PSP to near zero, so a window
with a high SCI and a low PSP is movement rather than coupling. Only a comparison made
inside one window can see that; two whole-run averages no longer know whether the good SCI
and the bad PSP happened at the same time.

The counting is what an average cannot do. A mean over the recording is carried over the
line by the stretch where the channel was fine, so a channel that is excellent for half a
long run and dead for the other half passes on the average and fails on the count.
"""

import numpy as np
import pytest

from fnirs_pipe.qc.metrics.windowed import (
    SCREEN_WINDOW_S,
    good_window_fraction,
)


@pytest.fixture
def stub_windows(monkeypatch):
    """Replace the two windowed measurements with fixed channel x window matrices.

    The grid comes back too, laid out as consecutive 10 s windows, because the scope is
    decided on window centres and a stub without times cannot be scoped.
    """
    from fnirs_pipe.qc.metrics import windowed

    def _install(sci, psp, window_s=SCREEN_WINDOW_S):
        sci, psp = np.asarray(sci, float), np.asarray(psp, float)
        grid = np.array([[i * window_s, (i + 1) * window_s]
                         for i in range(sci.shape[1])], float)
        monkeypatch.setattr(windowed, "compute_windowed_sci",
                            lambda *a, **k: (sci, grid))
        monkeypatch.setattr(windowed, "compute_windowed_psp",
                            lambda *a, **k: (psp, grid))
        return grid

    return _install


class _Raw:
    def __init__(self, names):
        self.ch_names = list(names)


def _frac(sci, psp, sci_cut=0.8, psp_cut=0.1, scope=None):
    names = [f"ch{i}" for i in range(len(sci))]
    return good_window_fraction(_Raw(names), 0.7, 1.5, sci_cut, psp_cut, scope=scope)


# ---- the count ----

def test_a_channel_coupled_throughout_scores_one(stub_windows):
    stub_windows(sci=[[0.9] * 4], psp=[[0.5] * 4])
    assert _frac([[0]], [[0]])["ch0"] == pytest.approx(1.0)


def test_a_channel_never_coupled_scores_zero(stub_windows):
    stub_windows(sci=[[0.2] * 4], psp=[[0.5] * 4])
    assert _frac([[0]], [[0]])["ch0"] == pytest.approx(0.0)


def test_half_a_run_dead_scores_a_half(stub_windows):
    """The case the whole-run average cannot express: the good half carries the mean over
    the line, and the count does not move."""
    stub_windows(sci=[[0.95, 0.95, 0.2, 0.2]], psp=[[0.5] * 4])
    frac = _frac([[0]], [[0]])["ch0"]
    assert frac == pytest.approx(0.5)
    # the average of the same windows still clears the line the count fails
    assert np.mean([0.95, 0.95, 0.2, 0.2]) > 0.5
    assert frac < 0.75


# ---- the pairing ----

def test_high_sci_with_dead_psp_is_not_a_coupled_window(stub_windows):
    """Movement fakes a high SCI, and PSP near zero is how that is caught. Both have to
    hold in the same window or the pair says nothing."""
    stub_windows(sci=[[0.99] * 4], psp=[[0.01] * 4])
    assert _frac([[0]], [[0]])["ch0"] == pytest.approx(0.0)


def test_the_pairing_is_inside_the_window_not_across_the_run(stub_windows):
    """A channel whose SCI is high exactly when its PSP is low is never coupled, though
    both averages pass. Judging the two whole-run means against their lines would keep it."""
    sci = [[0.99, 0.99, 0.10, 0.10]]
    psp = [[0.01, 0.01, 0.90, 0.90]]
    stub_windows(sci=sci, psp=psp)
    assert _frac([[0]], [[0]])["ch0"] == pytest.approx(0.0)
    assert np.mean(sci) > 0.5          # a whole-run SCI over its line
    assert np.mean(psp) > 0.1          # a whole-run PSP over its line


# ---- the lines ----

def test_raising_the_sci_line_can_only_lower_the_share(stub_windows):
    stub_windows(sci=[[0.85, 0.85, 0.95, 0.95]], psp=[[0.5] * 4])
    assert _frac([[0]], [[0]], sci_cut=0.8)["ch0"] == pytest.approx(1.0)
    assert _frac([[0]], [[0]], sci_cut=0.9)["ch0"] == pytest.approx(0.5)


def test_each_channel_is_counted_on_its_own(stub_windows):
    stub_windows(sci=[[0.9, 0.9], [0.9, 0.2]], psp=[[0.5, 0.5], [0.5, 0.5]])
    out = _frac([[0], [0]], [[0], [0]])
    assert out == {"ch0": pytest.approx(1.0), "ch1": pytest.approx(0.5)}


# ---- failing safely ----

def test_an_unmeasurable_metric_screens_nothing(monkeypatch):
    """Returning nothing keeps every channel. The alternative failure, an empty score
    treated as zero, would reject the whole montage for a reason unrelated to coupling."""
    from fnirs_pipe.qc.metrics import windowed

    def _boom(*a, **k):
        raise RuntimeError("cardiac band rejected by the filter")

    monkeypatch.setattr(windowed, "compute_windowed_sci", _boom)
    assert good_window_fraction(_Raw(["ch0"]), 0.7, 1.5, 0.8, 0.1) == {}


def test_grids_that_disagree_screen_nothing(stub_windows):
    """Both come off the same window length, so a shape mismatch means one of them did not
    do what it was asked; counting them against each other anyway would align window 3 of
    one against window 4 of the other."""
    stub_windows(sci=[[0.9] * 4], psp=[[0.5] * 3])
    assert _frac([[0]], [[0]]) == {}


# ---- the screening window is pinned ----

def test_the_screening_window_is_the_one_the_lines_were_set_at():
    """PSP is a power and moves with window length, so 0.1 selects a different set of
    channels at every length. The report's window is free; this one is not."""
    assert SCREEN_WINDOW_S == 10.0


# ---- the scope: which windows go in the denominator ----
#
# A run holds time no analysis reads, the lead-in before the first block and the gaps
# between them. Counting those holds a channel responsible for what it did while nobody was
# doing anything. Restricting the denominator is one channel set either way, which is what
# separates it from screening each condition on its own.

def test_no_scope_counts_the_whole_recording(stub_windows):
    stub_windows(sci=[[0.9, 0.9, 0.2, 0.2]], psp=[[0.5] * 4])
    assert _frac([[0]], [[0]])["ch0"] == pytest.approx(0.5)


def test_a_scope_drops_the_windows_outside_it(stub_windows):
    """The same channel, judged only on the stretch the analysis reads. Coupled in the first
    two windows and dead in the last two: over the run that is a half, over a scope holding
    only the first two it is all of them."""
    stub_windows(sci=[[0.9, 0.9, 0.2, 0.2]], psp=[[0.5] * 4])
    assert _frac([[0]], [[0]])["ch0"] == pytest.approx(0.5)
    assert _frac([[0]], [[0]], scope=[("block", 0.0, 20.0)])["ch0"] == pytest.approx(1.0)


def test_a_window_is_placed_by_its_centre(stub_windows):
    """Decided on one time rather than on overlap, so a window straddling a block edge
    counts for the side it mostly sits in and never for both."""
    from fnirs_pipe.qc.metrics.windowed import _in_scope, window_centers

    centers = window_centers([[0, 10], [10, 20], [20, 30]])
    keep = _in_scope(centers, [("a", 0.0, 12.0), ("b", 12.0, 30.0)])
    assert keep.tolist() == [True, True, True]
    assert _in_scope(centers, [("a", 0.0, 12.0)]).tolist() == [True, False, False]


def test_a_scope_that_keeps_nothing_screens_nothing(stub_windows):
    """Rejecting the whole montage because a scope missed every window would read as a
    verdict on the recording."""
    stub_windows(sci=[[0.9, 0.9]], psp=[[0.5, 0.5]])
    assert _frac([[0]], [[0]], scope=[("late", 900.0, 1000.0)]) == {}


# ---- resolving the scope from the annotations ----

def _annotated(onsets, durations, descs, dur=600.0):
    import mne

    info = mne.create_info(["S1_D1 hbo"], 10.0, ["hbo"])
    raw = mne.io.RawArray(np.zeros((1, int(10.0 * dur))), info, verbose="error")
    raw.set_annotations(mne.Annotations(onsets, durations, descs))
    return raw


def test_task_scope_takes_the_blocks_and_leaves_the_triggers():
    from fnirs_pipe.qc.metrics.windowed import task_scope_windows

    raw = _annotated([20.0, 400.0], [300.0, 0.0], ["rest", "trigger"])
    assert task_scope_windows(raw) == [("rest", 20.0, 320.0)]


def test_run_scope_is_the_whole_recording():
    from fnirs_pipe.qc.common.screen_scope import resolve_screen_scope

    assert resolve_screen_scope(_annotated([20.0], [300.0], ["rest"]), "run") is None


def test_task_scope_falls_back_when_only_triggers_are_annotated(caplog):
    """The case the fallback exists for: scoping to a handful of short triggers would count
    a minute of an hour and still read as a verdict on the run."""
    from fnirs_pipe.qc.common.screen_scope import resolve_screen_scope

    raw = _annotated([20.0, 100.0, 200.0], [10.0, 10.0, 10.0], ["t", "t", "t"])
    with caplog.at_level("WARNING"):
        assert resolve_screen_scope(raw, "task") is None
    assert "counting the whole recording" in caplog.text


def test_task_scope_returns_the_blocks_when_there_are_blocks():
    from fnirs_pipe.qc.common.screen_scope import resolve_screen_scope

    raw = _annotated([20.0, 400.0], [300.0, 100.0], ["rest", "talk"])
    assert resolve_screen_scope(raw, "task") == [("rest", 20.0, 320.0),
                                                 ("talk", 400.0, 500.0)]


def test_an_unknown_scope_is_refused():
    from fnirs_pipe.qc.common.screen_scope import resolve_screen_scope

    with pytest.raises(ValueError, match="screen scope"):
        resolve_screen_scope(_annotated([20.0], [300.0], ["rest"]), "poi")
