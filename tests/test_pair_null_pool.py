"""Who may stand in for a member, and the cohort shape that makes the wider pool a lie.

Two cohort shapes reach this null and they want different pools. Many dyads recorded once
can draw a stand-in from anybody else; the same two people recorded over many days cannot,
because "another dyad" there is the same pair on another day and the wider pool would rank a
person against themselves. The pairs table says which shape it is, so the refusal is
measured rather than declared by the caller.
"""

import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.group_io import GroupEntry
from fnirs_pipe.pipeline.pair_null import condition_coverage, partner_pool


def _cohort(task="full", n_groups=3, repeated_people=False):
    """n_groups groups of two. Repeated people give every group the same two subjects."""
    groups = {}
    for i in range(1, n_groups + 1):
        gid = f"d{i:02d}"
        subs = (["sub-p1", "sub-p2"] if repeated_people
                else [f"sub-p1{gid}", f"sub-p2{gid}"])
        groups[(gid, task)] = [GroupEntry(gid, s, task) for s in subs]
    return groups


# ---- the position pool ----

def test_the_stand_ins_are_the_other_groups_member_at_the_same_index():
    got = partner_pool(_cohort(), "d01", "full")
    assert [e.subject_id for e in got] == ["sub-p2d02", "sub-p2d03"]


def test_the_target_group_is_never_its_own_stand_in():
    got = partner_pool(_cohort(), "d02", "full")
    assert "sub-p2d02" not in [e.subject_id for e in got]


def test_position_zero_replaces_the_other_member():
    got = partner_pool(_cohort(), "d01", "full", position=0)
    assert [e.subject_id for e in got] == ["sub-p1d02", "sub-p1d03"]


def test_another_task_is_never_a_stand_in():
    groups = {**_cohort(task="full"), **_cohort(task="rest")}
    got = partner_pool(groups, "d01", "full")
    assert all(e.task == "full" for e in got)


def test_the_same_two_people_over_many_days_still_get_a_position_pool():
    """The shape this null was asked for: p1 on one day against p2 on another."""
    got = partner_pool(_cohort(repeated_people=True), "d01", "full")
    # the same person, but a different recording of them, and never the target's own rows
    assert len(got) == 2
    assert [e.group_id for e in got] == ["d02", "d03"]


# ---- the any pool ----

def test_any_doubles_the_pool_when_nobody_repeats():
    got = partner_pool(_cohort(), "d01", "full", pool="any")
    assert sorted(e.subject_id for e in got) == [
        "sub-p1d02", "sub-p1d03", "sub-p2d02", "sub-p2d03"]


def test_any_is_refused_when_one_person_is_in_several_groups():
    with pytest.raises(StageError, match="pair somebody with themselves"):
        partner_pool(_cohort(repeated_people=True), "d01", "full", pool="any")


def test_the_refusal_names_the_subject_and_the_groups():
    with pytest.raises(StageError) as exc:
        partner_pool(_cohort(repeated_people=True), "d01", "full", pool="any")
    assert "sub-p1" in str(exc.value) and "d02" in str(exc.value)


def test_any_is_refused_when_a_group_lists_one_subject_twice():
    groups = _cohort()
    groups[("d02", "full")] = [GroupEntry("d02", "sub-x", "full")] * 2
    with pytest.raises(StageError, match="same subject twice"):
        partner_pool(groups, "d01", "full", pool="any")


def test_an_unknown_pool_is_refused():
    with pytest.raises(ValueError, match="pool must be one of"):
        partner_pool(_cohort(), "d01", "full", pool="everyone")


def test_a_group_absent_from_the_task_is_refused():
    with pytest.raises(StageError, match="no task"):
        partner_pool(_cohort(), "d09", "full")


# ---- condition coverage ----

class _FakeRaw:
    """Just enough of a Raw for the coverage helper: annotations already on the data axis."""

    def __init__(self, markers, duration=1000.0):
        import numpy as np
        self._markers = markers
        self.first_time = 0.0
        self.times = np.array([0.0, duration])


@pytest.fixture
def _no_offset(monkeypatch):
    monkeypatch.setattr("fnirs_pipe.qc.common.windows.markers_on_data_axis",
                        lambda raw: raw._markers)


def _m(desc, onset, duration):
    return {"description": desc, "onset": onset, "duration": duration}


def test_a_stand_in_on_the_same_timetable_covers_the_window_fully(_no_offset):
    raw = _FakeRaw([_m("game1", 100.0, 300.0)])
    assert condition_coverage(raw, [("game1", 100.0, 400.0)]) == {"game1": 1.0}


def test_a_stand_in_running_late_covers_part_of_it(_no_offset):
    raw = _FakeRaw([_m("game1", 150.0, 300.0)])
    assert condition_coverage(raw, [("game1", 100.0, 400.0)])["game1"] == pytest.approx(0.8333,
                                                                                       abs=1e-3)


def test_a_stand_in_doing_something_else_covers_none_of_it(_no_offset):
    raw = _FakeRaw([_m("game2", 100.0, 300.0)])
    assert condition_coverage(raw, [("game1", 100.0, 400.0)]) == {"game1": 0.0}


def test_a_numbered_repeat_matches_the_annotation_it_came_from(_no_offset):
    """condition_windows numbers a repeated description; the annotation keeps the bare one."""
    raw = _FakeRaw([_m("game1", 100.0, 300.0)])
    assert condition_coverage(raw, [("game1#2", 100.0, 400.0)]) == {"game1#2": 1.0}


def test_a_zero_duration_trigger_runs_to_the_end_of_the_recording(_no_offset):
    """Many systems write triggers with no duration, which is why the window rule exists."""
    raw = _FakeRaw([_m("game1", 100.0, 0.0)], duration=1000.0)
    assert condition_coverage(raw, [("game1", 100.0, 400.0)]) == {"game1": 1.0}
