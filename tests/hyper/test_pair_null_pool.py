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


def _cohort(task="main", n_groups=3, repeated_people=False):
    """n_groups groups of two. Repeated people give every group the same two subjects."""
    groups = {}
    for i in range(1, n_groups + 1):
        gid = f"G{i:02d}"
        subs = (["sub-01", "sub-02"] if repeated_people
                else [f"sub-01{gid}", f"sub-02{gid}"])
        groups[(gid, task)] = [GroupEntry(gid, s, task) for s in subs]
    return groups


# ---- the position pool ----

def test_the_stand_ins_are_the_other_groups_member_at_the_same_index():
    got = partner_pool(_cohort(), "G01", "main")
    assert [e.subject_id for e in got] == ["sub-02G02", "sub-02G03"]


def test_the_target_group_is_never_its_own_stand_in():
    got = partner_pool(_cohort(), "G02", "main")
    assert "sub-02G02" not in [e.subject_id for e in got]


def test_position_zero_replaces_the_other_member():
    got = partner_pool(_cohort(), "G01", "main", position=0)
    assert [e.subject_id for e in got] == ["sub-01G02", "sub-01G03"]


def test_another_task_is_never_a_stand_in():
    groups = {**_cohort(task="main"), **_cohort(task="rest")}
    got = partner_pool(groups, "G01", "main")
    assert all(e.task == "main" for e in got)


def test_the_same_two_people_over_many_days_still_get_a_position_pool():
    """One member on one day against the other member on another."""
    got = partner_pool(_cohort(repeated_people=True), "G01", "main")
    # the same person, but a different recording of them, and never the target's own rows
    assert len(got) == 2
    assert [e.group_id for e in got] == ["G02", "G03"]


# ---- the any pool ----

def test_any_doubles_the_pool_when_nobody_repeats():
    got = partner_pool(_cohort(), "G01", "main", pool="any")
    assert sorted(e.subject_id for e in got) == [
        "sub-01G02", "sub-01G03", "sub-02G02", "sub-02G03"]


def test_any_is_refused_when_one_person_is_in_several_groups():
    with pytest.raises(StageError, match="pair somebody with themselves"):
        partner_pool(_cohort(repeated_people=True), "G01", "main", pool="any")


def test_the_refusal_names_the_subject_and_the_groups():
    with pytest.raises(StageError) as exc:
        partner_pool(_cohort(repeated_people=True), "G01", "main", pool="any")
    assert "sub-01" in str(exc.value) and "G02" in str(exc.value)


def test_any_is_refused_when_a_group_lists_one_subject_twice():
    groups = _cohort()
    groups[("G02", "main")] = [GroupEntry("G02", "sub-x", "main")] * 2
    with pytest.raises(StageError, match="same subject twice"):
        partner_pool(groups, "G01", "main", pool="any")


def test_an_unknown_pool_is_refused():
    with pytest.raises(ValueError, match="pool must be one of"):
        partner_pool(_cohort(), "G01", "main", pool="everyone")


def test_a_group_absent_from_the_task_is_refused():
    with pytest.raises(StageError, match="no task"):
        partner_pool(_cohort(), "G09", "main")


# ---- the refusal cannot see every repeated cohort, so it says when it could not look ----

def test_any_warns_when_every_subject_id_is_unique(caplog):
    """A cohort that bakes the visit into the subject id defeats the refusal.

    BIDS puts one person under one `sub-` label and separates visits with `ses-`, and the
    pairs table takes a `session` column for exactly that. A table that instead bakes the
    visit into the subject id, `sub-01G01` and `sub-01G03` for one person, is
    indistinguishable from a table of strangers. Nothing on disk settles it, so the check
    has not passed there, it has had nothing to test, and it has to say so.
    """
    with caplog.at_level("WARNING"):
        partner_pool(_cohort(n_groups=4), "G01", "main", pool="any")
    assert "could not be checked" in caplog.text
    assert "session column" in caplog.text


def test_the_idiomatic_encoding_still_gets_a_refusal():
    """Same people, one stable subject id per person: the refusal has something to test."""
    with pytest.raises(StageError, match="pair somebody with themselves"):
        partner_pool(_cohort(repeated_people=True), "G01", "main", pool="any")


def test_no_warning_where_the_refusal_had_something_to_test(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(StageError):
            partner_pool(_cohort(repeated_people=True), "G01", "main", pool="any")
    assert "could not be checked" not in caplog.text
