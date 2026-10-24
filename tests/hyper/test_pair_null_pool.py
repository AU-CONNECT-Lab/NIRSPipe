"""Who may stand in for a member, and the cohort shape that makes the wider pool a lie.

Two cohort shapes reach this null and they want different pools. Many dyads recorded once
can draw a stand-in from anybody else; the same two people recorded over many days cannot,
because "another dyad" there is the same pair on another day and the wider pool would rank a
person against themselves. The pairs table says which shape it is, so the refusal is
measured rather than declared by the caller.

A group of more than two has every member after the first replaced, one stand-in per member
in each draw, all from one other occasion; the draws rotate a donor's members of one key
over the target's, so a stand-in fills a member at most once.
"""

import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.groups import GroupEntry
from nirspipe.pipeline.hyper.pair_null import pairing_stand_ins, stand_in_draws


def _cohort(task="main", n_groups=3, repeated_people=False):
    """n_groups groups of two. Repeated people give every group the same two subjects."""
    groups = {}
    for i in range(1, n_groups + 1):
        gid = f"G{i:02d}"
        subs = (["sub-01", "sub-02"] if repeated_people
                else [f"sub-01{gid}", f"sub-02{gid}"])
        groups[(gid, task, None)] = [GroupEntry(gid, s, task) for s in subs]
    return groups


def _ids(draws, member=1):
    return [d[member].subject_id for d in draws if member in d]


def _named(draws):
    return [{k: e.subject_id for k, e in d.items()} for d in draws]


# ---- the position pool ----

def test_the_stand_ins_are_the_other_groups_member_at_the_same_index():
    assert _ids(stand_in_draws(_cohort(), "G01", "main")) == ["sub-02G02", "sub-02G03"]


def test_the_target_group_is_never_its_own_stand_in():
    assert "sub-02G02" not in _ids(stand_in_draws(_cohort(), "G02", "main"))


def test_the_first_member_is_held_in_every_draw():
    assert all(set(d) == {1} for d in stand_in_draws(_cohort(), "G01", "main"))


def test_another_task_is_never_a_stand_in():
    groups = {**_cohort(task="main"), **_cohort(task="rest")}
    assert all(d[1].task == "main" for d in stand_in_draws(groups, "G01", "main"))


def test_the_same_two_people_over_many_days_still_get_a_position_pool():
    """One member on one day against the other member on another."""
    draws = stand_in_draws(_cohort(repeated_people=True), "G01", "main")
    # the same person, but a different recording of them, and never the target's own rows
    assert [d[1].group_id for d in draws] == ["G02", "G03"]


def _triads():
    """G01 and G02 three members each, G03 two."""
    groups = {}
    for gid, n in (("G01", 3), ("G02", 3), ("G03", 2)):
        groups[(gid, "main", None)] = [GroupEntry(gid, f"{gid}m{k}", "main") for k in range(n)]
    return groups


def test_a_triad_draws_each_later_member_from_the_same_index():
    assert _named(stand_in_draws(_triads(), "G01", "main")) == [
        {1: "G02m1", 2: "G02m2"}, {1: "G03m1"}]


def test_a_dyad_takes_a_triads_second_member_only():
    assert _named(stand_in_draws(_triads(), "G03", "main")) == [{1: "G01m1"}, {1: "G02m1"}]


# ---- the position pool by role ----

def _classes(teacher_first=True):
    """G01 a teacher and a student; G02 and G03 a teacher and two students, listed in any order."""
    def entry(gid, sub, role):
        return GroupEntry(gid, sub, "main", role=role)
    g01 = [entry("G01", "t1", "teacher"), entry("G01", "s1", "student")]
    return {
        ("G01", "main", None): g01 if teacher_first else g01[::-1],
        ("G02", "main", None): [entry("G02", "s2a", "student"), entry("G02", "t2", "teacher"),
                                entry("G02", "s2b", "student")],
        ("G03", "main", None): [entry("G03", "t3", "teacher"), entry("G03", "s3a", "student"),
                                entry("G03", "s3b", "student")],
    }


def test_with_roles_every_other_groups_member_in_the_role_stands_in():
    assert _ids(stand_in_draws(_classes(), "G01", "main")) == ["s2a", "s2b", "s3a", "s3b"]


def test_with_roles_the_index_no_longer_decides():
    """G01 lists its student first, so its second member is the teacher, and G02 lists a
    student first, so the index rule would have taken a student for it."""
    assert _ids(stand_in_draws(_classes(teacher_first=False), "G01", "main")) == ["t2", "t3"]


def test_two_students_rotate_so_each_fills_each_place_once():
    """A teacher and two students against a teacher and two students: two draws, and the
    one student G01 has fills the first student's place only, there being one."""
    assert _named(stand_in_draws(_classes(), "G03", "main")) == [
        {1: "s1"}, {1: "s2a", 2: "s2b"}, {1: "s2b", 2: "s2a"}]


def test_a_group_led_by_a_student_draws_a_teacher_and_a_student():
    """G02's held member is its first student, so its teacher and its other student are
    replaced, each from its own role, as many draws as the scarcer role allows."""
    assert _named(stand_in_draws(_classes(), "G02", "main")) == [
        {1: "t1", 2: "s1"}, {1: "t3", 2: "s3a"}]


# ---- the any pool ----

def test_any_doubles_the_pool_when_nobody_repeats():
    assert sorted(_ids(stand_in_draws(_cohort(), "G01", "main", pool="any"))) == [
        "sub-01G02", "sub-01G03", "sub-02G02", "sub-02G03"]


def test_any_rotates_a_donors_members_over_a_triads_places():
    assert _named(stand_in_draws(_triads(), "G01", "main", pool="any"))[3:] == [
        {1: "G03m0", 2: "G03m1"}, {1: "G03m1", 2: "G03m0"}]


def test_any_is_refused_when_one_person_is_in_several_groups():
    with pytest.raises(StageError, match="pair somebody with themselves"):
        stand_in_draws(_cohort(repeated_people=True), "G01", "main", pool="any")


def test_the_refusal_names_the_subject_and_the_groups():
    with pytest.raises(StageError) as exc:
        stand_in_draws(_cohort(repeated_people=True), "G01", "main", pool="any")
    assert "sub-01" in str(exc.value) and "G02" in str(exc.value)


def test_any_is_refused_when_a_group_lists_one_subject_twice():
    groups = _cohort()
    groups[("G02", "main", None)] = [GroupEntry("G02", "sub-x", "main")] * 2
    with pytest.raises(StageError, match="same subject twice"):
        stand_in_draws(groups, "G01", "main", pool="any")


def test_an_unknown_pool_is_refused():
    with pytest.raises(ValueError, match="pool must be one of"):
        stand_in_draws(_cohort(), "G01", "main", pool="everyone")


def test_a_group_absent_from_the_task_is_refused():
    with pytest.raises(StageError, match="no task"):
        stand_in_draws(_cohort(), "G09", "main")


# ---- each pairing's share of the draws ----

def test_a_pairing_takes_its_later_members_stand_in_under_the_draws_id():
    a, b, c, d, e = (GroupEntry("G", s, "main") for s in "abcde")
    got = pairing_stand_ins(["a", "b", "c"], [{1: d, 2: e}, {1: e}])
    assert got == {(0, 1): [("d+e", d), ("e", e)], (0, 2): [("d+e", e)],
                   (1, 2): [("d+e", e)]}


def test_a_dyads_draw_is_named_by_its_stand_ins_recording():
    d = GroupEntry("G", "sub-d", "main", session="2")
    assert pairing_stand_ins(["a", "b"], [{1: d}]) == {(0, 1): [("sub-d_ses-2", d)]}


def test_a_stand_in_who_is_the_held_member_on_another_day_is_left_out_of_that_pairing():
    """The same three people on two days: the second day's b may stand in for c against a,
    but against b it would be b against b."""
    day = [GroupEntry("G01", s, "main", session="2", role=r)
           for s, r in (("a", "lead"), ("b", "follow"), ("c", "follow"))]
    got = pairing_stand_ins(["a", "b", "c"], [{1: day[1], 2: day[2]}, {1: day[2], 2: day[1]}])
    assert [entry.subject_id for _, entry in got[(0, 2)]] == ["c", "b"]
    assert [entry.subject_id for _, entry in got[(1, 2)]] == ["c"]


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
        stand_in_draws(_cohort(n_groups=4), "G01", "main", pool="any")
    assert "could not be checked" in caplog.text
    assert "session column" in caplog.text


def test_the_idiomatic_encoding_still_gets_a_refusal():
    """Same people, one stable subject id per person: the refusal has something to test."""
    with pytest.raises(StageError, match="pair somebody with themselves"):
        stand_in_draws(_cohort(repeated_people=True), "G01", "main", pool="any")


def test_no_warning_where_the_refusal_had_something_to_test(caplog):
    with caplog.at_level("WARNING"):
        with pytest.raises(StageError):
            stand_in_draws(_cohort(repeated_people=True), "G01", "main", pool="any")
    assert "could not be checked" not in caplog.text
