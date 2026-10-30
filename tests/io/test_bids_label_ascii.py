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


# a table each reader accepts as it stands, but for the one label
_TABLES = {
    "occasion": ["group_id,subject_id,task,session,occasion",
                 "G1,sub-01,rest,01,{v}", "G1,sub-02,rest,02,{v}"],
    "role": ["group_id,subject_id,task,role", "G1,sub-01,rest,{v}", "G1,sub-02,rest,child"],
}


@pytest.mark.parametrize("column, value", [("occasion", "visité"), ("role", "mère")])
def test_a_groups_table_refuses_a_label_a_file_name_could_not_carry(tmp_path, column, value):
    path = tmp_path / "pairs.csv"
    path.write_text("\n".join(_TABLES[column]).format(v=value) + "\n", encoding="utf-8")
    with pytest.raises(GroupCSVError, match="not a"):
        parse_group_csv(path)


def test_a_bad_channels_table_refuses_such_a_label(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("\n".join(["participant_id\ttask\tbad_channels",
                               "sub-01\treposé\tS1_D1"]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a BIDS label"):
        _bad_channel_rows(path)
