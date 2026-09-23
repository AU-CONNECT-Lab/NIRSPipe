"""Where a group's outputs land on disk.

A subject owns a folder; a group did not, and wrote its tables and reports loose in the
derivatives root instead. A study of any size puts thousands of files there, between the
reader and the subject folders. These pin the folder, not the filenames, which did not
change.
"""

import pandas as pd

from fnirs_pipe.io.derivatives import (
    group_data_dir,
    group_report_dir,
    subject_report_dir,
)
from fnirs_pipe.pipeline.hyper import GroupEntry, write_group_bads


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

    assert path == (tmp_path / "group-G1" / "nirs"
                    / "group-G1_task-hold_desc-bad_qc.tsv")
    rows = pd.read_csv(path, sep="\t")
    assert rows["rejected_in"].tolist() == ["hold;rest"]


def test_the_derivatives_root_keeps_no_loose_group_files(tmp_path):
    write_group_bads(tmp_path, _group(), {"01": {"bad_channels": []}}, "run")

    loose = [p.name for p in tmp_path.iterdir() if p.is_file()]
    assert loose == []


# ---- the subject side of the same rule ----

def test_a_subject_folder_is_named_the_same_way_from_either_form(tmp_path):
    assert subject_report_dir(tmp_path, "01") == tmp_path / "sub-01"
    assert subject_report_dir(tmp_path, "sub-01") == tmp_path / "sub-01"
    assert (tmp_path / "sub-01").is_dir()


def test_the_raw_qc_report_lands_in_the_subject_folder(tmp_path):
    """It used to sit loose in the root, one file per run beside the study's own."""
    from fnirs_pipe.qc.subject.prep_raw_report import build_prep_raw_report

    out = subject_report_dir(tmp_path, "01") / "sub-01_task-hold_desc-raw_nirs.html"
    build_prep_raw_report([], out, cardiac_l_freq=0.7, cardiac_h_freq=1.5, dpf=[6.0])

    assert out.exists()
    assert [q.name for q in tmp_path.iterdir() if q.is_file()] == []
