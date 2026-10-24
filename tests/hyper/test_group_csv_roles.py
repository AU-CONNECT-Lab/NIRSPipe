"""The pairs table's role column: every member's part, or nobody's."""

import pytest

from nirspipe.exceptions import GroupCSVError
from nirspipe.io.groups import parse_group_csv


def _csv(tmp_path, text):
    path = tmp_path / "pairs.csv"
    path.write_text(text)
    return path


def test_each_member_carries_its_role_in_the_rows_order(tmp_path):
    groups = parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,role\n"
                                            "G1,sub-02,main,student\nG1,sub-01,main,teacher\n"
                                            "G1,sub-03,main,student\n"))
    members = groups[("G1", "main", None)]
    assert [(m.subject_id, m.role) for m in members] == [
        ("sub-02", "student"), ("sub-01", "teacher"), ("sub-03", "student")]


def test_without_the_column_nobody_has_a_role(tmp_path):
    groups = parse_group_csv(_csv(tmp_path, "group_id,subject_id,task\n"
                                            "G1,sub-01,main\nG1,sub-02,main\n"))
    assert all(m.role is None for m in groups[("G1", "main", None)])


def test_a_blank_role_is_refused_naming_the_row(tmp_path):
    with pytest.raises(GroupCSVError, match="sub-02 in group G1 has no role"):
        parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,role\n"
                                       "G1,sub-01,main,teacher\nG1,sub-02,main,\n"))


def test_a_role_that_is_not_one_word_is_refused(tmp_path):
    with pytest.raises(GroupCSVError, match="letters and digits only"):
        parse_group_csv(_csv(tmp_path, "group_id,subject_id,task,role\n"
                                       "G1,sub-01,main,team-lead\nG1,sub-02,main,member\n"))
