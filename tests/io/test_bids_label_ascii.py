"""A BIDS label is written by the ASCII rule its reader matches, so a name reads back whole."""

import pytest

from nirspipe.cli.workflows import _bad_channel_rows
from nirspipe.io.derivatives import entity_of
from nirspipe.io.groups import GroupCSVError, parse_group_csv
from nirspipe.io.naming import bids_label, roi_map_name


def test_a_letter_outside_ascii_is_left_out_of_a_label():
    assert bids_label("frontal_régions") == "frontalrgions"


def test_a_written_label_reads_back_whole():
    label = roi_map_name("/studies/frontal_régions.json")
    assert entity_of(f"sub-01_seg-{label}_desc-fc_relmat.tsv", "seg") == label


def _groups_table(tmp_path, lines):
    path = tmp_path / "pairs.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_a_groups_table_refuses_an_occasion_a_file_name_could_not_carry(tmp_path):
    path = _groups_table(tmp_path, ["group_id,subject_id,task,session,occasion",
                                    "G1,sub-01,rest,01,visité", "G1,sub-02,rest,02,visité"])
    with pytest.raises(GroupCSVError, match="not a"):
        parse_group_csv(path)


def test_a_role_names_a_table_level_and_no_file_so_it_may_be_any_word(tmp_path):
    path = _groups_table(tmp_path, ["group_id,subject_id,task,role",
                                    "G1,sub-01,rest,mère", "G1,sub-02,rest,enfant"])
    (members,) = parse_group_csv(path).values()
    assert [m.role for m in members] == ["mère", "enfant"]


@pytest.mark.parametrize("role", ["the mother", "mère-1"])
def test_a_role_is_still_one_word(tmp_path, role):
    path = _groups_table(tmp_path, ["group_id,subject_id,task,role",
                                    f"G1,sub-01,rest,{role}", "G1,sub-02,rest,child"])
    with pytest.raises(GroupCSVError, match="not a label"):
        parse_group_csv(path)


def test_a_bad_channels_table_refuses_such_a_label(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("\n".join(["participant_id\ttask\tbad_channels",
                               "sub-01\treposé\tS1_D1"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a BIDS label"):
        _bad_channel_rows(path)
