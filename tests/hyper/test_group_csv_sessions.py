"""A pairs CSV naming one group and task in two sessions must not become one group."""

import pytest

from fnirs_pipe.pipeline.hyper.group_io import GroupCSVError, parse_group_csv


@pytest.mark.xfail(strict=True, reason="B: groups are keyed by (group_id, task) alone, so two "
                                       "sessions merge and one member's recording replaces the other's")
def test_two_sessions_of_one_group_are_not_merged_into_one(tmp_path):
    csv = tmp_path / "pairs.csv"
    csv.write_text("group_id,subject_id,task,session\n"
                   "G1,sub-01,main,a\nG1,sub-02,main,a\n"
                   "G1,sub-01,main,b\nG1,sub-02,main,b\n")
    try:
        groups = parse_group_csv(csv)
    except GroupCSVError:
        return
    for members in groups.values():
        ids = [m.subject_id for m in members]
        assert len(ids) == len(set(ids)), ids
