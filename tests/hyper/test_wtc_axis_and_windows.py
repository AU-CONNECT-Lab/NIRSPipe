"""The three WTC pieces that are wrong in ways nobody notices by looking at the figure.

A frequency axis, a set of task windows and a circular mean all produce something plausible
when they are wrong. A log axis left to plotly still draws ticks, a window cut at the wrong
boundary still yields a coherence, and an arithmetic mean of two opposite phases still points
somewhere. None of the three raises, so each is pinned here against the property that makes
it right rather than against a rendered figure.

The merge keys are here for the same reason: a filename that matched two kinds would
concatenate the whole-run table with the per-condition one, and the `condition` column would
come out half empty rather than as an error.
"""

import numpy as np
import pytest

from nirspipe.pipeline.hyper.roi import _mean_phase
from nirspipe.qc.figures.hyper.hyper_post_figures import _log_freq_ticks

# pycwt's default grid: 12 sub-octaves per octave, so neighbours differ by 2 ** (1 / 12)
WAVELET_GRID = np.sort(0.2 / 2 ** (np.arange(0, 70) / 12))


def _band(fmin: float, fmax: float) -> np.ndarray:
    band = WAVELET_GRID[(WAVELET_GRID >= fmin) & (WAVELET_GRID <= fmax)]
    assert band.size, f"the fixture grid does not reach {fmin}-{fmax} Hz"
    return band


# ---- the frequency axis ----

def test_a_wide_band_is_labelled_at_the_decades():
    """The default 0.004-0.2 Hz axis. Plotly's own choice here is "D1", which labels all nine
    digits of every decade: 0.008 and 0.009 land a fifth as far apart as 0.1 and 0.2.
    """
    major, minor = _log_freq_ticks(_band(0.004, 0.20))

    assert [round(f, 4) for f in major] == [0.01, 0.1]
    assert minor, "the intermediate digits still get a bare tick"
    assert not set(major) & set(minor)


def test_every_tick_lies_inside_the_data():
    """A tick outside the axis range is clipped away, which is how an axis ends up bare."""
    freqs = _band(0.004, 0.20)
    for f in [*_log_freq_ticks(freqs)[0], *_log_freq_ticks(freqs)[1]]:
        assert freqs.min() <= f <= freqs.max()


def test_a_band_holding_one_decade_falls_back_rather_than_carrying_one_label():
    """One label is not an axis. 0.02-0.09 Hz holds only 0.05 as a whole power of ten."""
    major, _ = _log_freq_ticks(_band(0.02, 0.09))
    assert len(major) >= 2


def test_a_band_holding_no_round_number_still_gets_labels():
    """The last resort. A narrow band has no decade and no 1-2-5 step inside it at all."""
    major, _ = _log_freq_ticks(_band(0.05, 0.07))
    assert len(major) >= 2
    assert all(0.05 <= f <= 0.07 for f in major)


def test_the_rule_does_not_change_with_the_band():
    """Two figures in one report must not tick by different rules.

    Plotly switches between "D1" and "D2" at a threshold that a slightly narrower band
    crosses, so the same report could label every digit on one panel and only 1-2-5 on the
    next. Every band wide enough gets decades here.
    """
    for fmin, fmax in ((0.004, 0.20), (0.005, 0.19), (0.004, 0.15)):
        major, _ = _log_freq_ticks(_band(fmin, fmax))
        assert [round(f, 4) for f in major] == [0.01, 0.1], f"{fmin}-{fmax}"


def test_a_degenerate_axis_asks_for_no_ticks():
    assert _log_freq_ticks(np.array([0.05])) == ([], [])
    assert _log_freq_ticks(np.array([0.0, 0.1])) == ([], [])


# ---- the circular mean behind the phase arrows ----

def test_opposite_phases_average_to_the_direction_they_share():
    """179 and -179 degrees are nearly the same direction; their arithmetic mean is 0, which
    is the one direction neither of them points in.
    """
    a = np.full((2, 3), np.deg2rad(179.0))
    b = np.full((2, 3), np.deg2rad(-179.0))

    out = np.rad2deg(_mean_phase([a, b]))
    assert np.allclose(np.abs(out), 180.0)


def test_one_member_is_returned_unchanged():
    a = np.deg2rad(np.array([[30.0, -120.0]]))
    np.testing.assert_allclose(_mean_phase([a]), a, atol=1e-6)


def test_no_members_means_no_phase():
    """Maps saved before phase existed load with none, and the ROI average must survive it."""
    assert _mean_phase([]) is None


def test_a_full_turn_added_to_one_member_changes_nothing():
    """The property that separates a circular mean from an arithmetic one. pycwt returns
    angles wrapped to (-pi, pi], but a member that arrives a turn out must not drag the mean
    a third of the way round.
    """
    a = np.deg2rad(np.array([[10.0, -40.0]]))
    b = np.deg2rad(np.array([[50.0, 20.0]]))

    np.testing.assert_allclose(_mean_phase([a, b]),
                               _mean_phase([a + 2 * np.pi, b]), atol=1e-6)


# ---- task windows ----

def _raw_with(descs, onsets, durations, end=900.0):
    import mne

    info = mne.create_info(["x"], 10.0, "misc")
    raw = mne.io.RawArray(np.zeros((1, int(end * 10))), info, verbose="error")
    raw.set_annotations(mne.Annotations(onsets, durations, descs))
    return raw


@pytest.fixture
def windows():
    from nirspipe.qc.common.windows import condition_windows

    return condition_windows


def test_an_annotation_with_a_duration_uses_it(windows):
    out = windows(_raw_with(["Video", "Talk"], [60, 400], [240, 300]), min_duration=100.0)
    assert out == [("Video", 60.0, 300.0), ("Talk", 400.0, 700.0)]


def test_a_zero_duration_trigger_runs_to_the_next_one(windows):
    """Many systems write onsets only. The last window runs to the end of the recording,
    which is as long as the recording happens to be rather than as long as the block was.
    """
    out = windows(_raw_with(["Base", "Video", "Talk"], [0, 300, 600], [0, 0, 0]),
                  min_duration=100.0)

    assert [w[0] for w in out] == ["Base", "Video", "Talk"]
    assert out[0][1:] == (0.0, 300.0)
    assert out[1][1:] == (300.0, 600.0)
    assert out[2][1] == 600.0 and out[2][2] > 890.0


def test_a_repeated_description_gets_one_numbered_window_each(windows):
    """Not one spliced record: a wavelet transform reads the join between two non-adjacent
    segments as a step, and that lands in the result as broadband coherence at the join.
    """
    out = windows(_raw_with(["Talk", "Rest", "Talk"], [0, 300, 600], [0, 0, 0]),
                  min_duration=100.0)
    assert [w[0] for w in out] == ["Talk#1", "Rest", "Talk#2"]


def test_a_window_too_short_for_the_lowest_frequency_is_dropped(windows):
    """An event-related design yields nothing here, and should. No crop can carry a
    frequency whose period is longer than the crop.
    """
    trials = _raw_with(["trial"] * 3, [10, 20, 30], [1, 1, 1])
    assert windows(trials, min_duration=250.0) == []
    assert len(windows(trials, min_duration=0.5)) == 3


def test_a_recording_with_no_triggers_yields_no_windows(windows):
    assert windows(_raw_with([], [], []), min_duration=1.0) == []


def test_a_window_is_clipped_to_the_recording(windows):
    """A duration running past the end is a trigger written optimistically, not a reason to
    ask mne to crop past the data.
    """
    out = windows(_raw_with(["Long"], [800], [500], end=900.0), min_duration=10.0)
    assert out[0][2] <= 900.0


def test_a_window_is_on_the_aligned_clock_not_the_original_one(windows):
    """Windows must not land late by the alignment offset.

    `align_recordings` crops every member from its first shared trigger, and a cropped Raw
    keeps its annotations on the original recording's axis while its data axis restarts at
    zero. Cropping to a window measured on the wrong one of those selects the wrong stretch.
    """
    raw = _raw_with(["baseline", "game1"], [22, 500], [300, 100], end=1000.0)
    aligned = raw.copy().crop(tmin=22.0)

    assert windows(aligned, min_duration=50.0) == [("baseline", 0.0, 300.0),
                                                   ("game1", 478.0, 578.0)]


# ---- the merged table names ----

def test_no_two_kinds_of_table_share_a_name(tmp_path):
    """A per-condition table matching the whole-run kind would concatenate its rows into that
    table, leaving the `condition` column half empty instead of erroring.
    """
    from nirspipe.pipeline.hyper.wtc_aggregate import merge_kinds
    from tests.hyper._names import KINDS, name

    nirs = tmp_path / "group-07" / "nirs"
    nirs.mkdir(parents=True)
    for kind in KINDS:
        (nirs / name("07", "rest", kind)).write_text("label\tcoherence\nS1_D1\t0.3\n")

    # every kind its own merge, the draws excepted: they are the same null in full
    kinds = merge_kinds(tmp_path)
    assert len(kinds) == len(KINDS) - sum("draws" in k for k in KINDS)
    assert all(len(paths) == 1 for paths in kinds.values())


def test_the_per_condition_tables_are_merged_apart_from_the_whole_run_ones():
    from tests.hyper._names import name

    assert name("07", "rest", "wtcbycond") != name("07", "rest", "wtc")
    assert name("07", "rest", "wtcbycond-roichan") != name("07", "rest", "wtc-roichan")
