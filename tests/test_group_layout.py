"""Where a group's outputs land on disk.

A subject owns a folder; a group did not, and wrote its tables and reports loose in the
derivatives root instead. A study of 25 dyads over 5 tasks put two thousand files there,
between the reader and the subject folders. These pin the folder, not the filenames, which
did not change.
"""

import pandas as pd

from fnirs_pipe.io.derivatives import group_data_dir, group_report_dir
from fnirs_pipe.pipeline.hyperscanning import GroupEntry, write_group_bads


def _group(task="hold"):
    return [GroupEntry("G1", "01", task), GroupEntry("G1", "02", task)]


# ---- the folders ----

def test_a_group_folder_mirrors_a_subject_folder(tmp_path):
    assert group_report_dir(tmp_path, "G1") == tmp_path / "group-G1"
    assert group_data_dir(tmp_path, "G1") == tmp_path / "group-G1" / "nirs"
    assert group_data_dir(tmp_path, "G1", "w1") == tmp_path / "group-G1" / "ses-w1" / "nirs"


def test_the_folder_is_created_not_merely_named(tmp_path):
    # every caller writes into what it gets back, so a path alone would push the mkdir
    # out to each of them
    assert group_data_dir(tmp_path, "G1").is_dir()


# ---- what lands there ----

def test_the_excluded_channels_land_in_the_group_folder(tmp_path):
    sqm = {"01": {"bad_channels": ["S1_D1 hbo"],
                  "bad_channel_sources": {"S1_D1 hbo": ["hold", "rest"]}},
           "02": {"bad_channels": []}}

    path = write_group_bads(tmp_path, _group(), sqm, "subject")

    assert path == tmp_path / "group-G1" / "nirs" / "group-G1_task-hold_hyper-bads.tsv"
    rows = pd.read_csv(path, sep="\t")
    assert rows["rejected_in"].tolist() == ["hold;rest"]


def test_the_derivatives_root_keeps_no_loose_group_files(tmp_path):
    write_group_bads(tmp_path, _group(), {"01": {"bad_channels": []}}, "run")

    loose = [p.name for p in tmp_path.iterdir() if p.is_file()]
    assert loose == []
