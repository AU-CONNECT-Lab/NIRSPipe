"""Who may stand in for a member, and the cohort shape that makes the wider pool a lie.

Two cohort shapes reach this null and they want different pools. Many dyads recorded once
can draw a stand-in from anybody else; the same two people recorded over many days cannot,
because "another dyad" there is the same pair on another day and the wider pool would rank a
person against themselves. The pairs table says which shape it is, so the refusal is
measured rather than declared by the caller.
"""

import pytest

from fnirs_pipe.exceptions import StageError
from fnirs_pipe.pipeline.hyper.group_io import GroupEntry
from fnirs_pipe.pipeline.hyper.pair_null import partner_pool


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


# ---- the refusal cannot see every repeated cohort, so it says when it could not look ----

def test_any_warns_when_every_subject_id_is_unique(caplog):
    """The real cohort this null was built for defeats the refusal.

    BIDS puts one person under one `sub-` label and separates visits with `ses-`, and the
    pairs table takes a `session` column for exactly that. A table that instead bakes the
    visit into the subject id, `sub-p1d01` and `sub-p1d03` for one person, is
    indistinguishable from a table of strangers. Nothing on disk settles it, so the check
    has not passed there, it has had nothing to test, and it has to say so.
    """
    with caplog.at_level("WARNING"):
        partner_pool(_cohort(n_groups=4), "d01", "full", pool="any")
    assert "could not be checked" in caplog.text
    assert "session column" in caplog.text


def test_the_idiomatic_encoding_still_gets_a_refusal():
    """Same people, one stable subject id per person: the refusal has something to test."""
    with pytest.raises(StageError, match="pair somebody with themselves"):
        partner_pool(_cohort(repeated_people=True), "d01", "full", pool="any")


def test_no_warning_where_the_refusal_had_something_to_test(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(StageError):
            partner_pool(_cohort(repeated_people=True), "d01", "full", pool="any")
    assert "could not be checked" not in caplog.text
