"""What `fnirs-prep align` leaves on disk: one aligned tree, every offset beside its file.

It also wrote `group-<id>_task-<task>_align-offsets.tsv` at the tree's root, a name with no
suffix and a data product nothing read. A shift a step applied to one file is recorded in
that file's sidecar instead, which is what the reference BIDS Apps do with a time shift, and
this tree is read as an input next, where a group table has no place.
"""

import json
import shutil

import pytest

from fnirs_pipe.cli.prep import cmd_align
from tests._synth import make_hyper_dataset


def _align(tmp_path):
    bids_dir, pairs_csv = make_hyper_dataset(tmp_path, tasks=("hold",))
    out = tmp_path / "out"
    cmd_align(bids_dir, out, pairs_csv, skip_bids_validation=True)
    return out / "aligned"


def test_each_member_sidecar_says_how_it_was_aligned(tmp_path):
    aligned = _align(tmp_path)

    for subject in ("10031", "10032"):
        sidecar = json.loads((aligned / f"sub-{subject}" / "nirs"
                              / f"sub-{subject}_task-hold_nirs.json").read_text())
        assert sidecar["align_group"] == "1003"
        assert sidecar["align_step"] == "align_recordings"
        assert isinstance(sidecar["align_offset_s"], float)
        assert sidecar["aligned_duration_s"] > 0
        # added to the copied BIDS sidecar, not written over it
        assert sidecar["TaskName"] == "hold"


def test_the_tree_holds_no_group_table(tmp_path):
    aligned = _align(tmp_path)
    # the dataset root it copied is the only table left at the top
    assert [p.name for p in aligned.glob("*.tsv")] == ["participants.tsv"]
    assert not list(aligned.rglob("*offsets*"))


def test_a_member_two_groups_share_is_refused_rather_than_overwritten(tmp_path):
    # one copy per recording, so the second group would re-cut the first group's file
    bids_dir, pairs_csv = make_hyper_dataset(
        tmp_path, groups={"A": ("10031", "10032"), "B": ("10031", "10033")}, tasks=("hold",))
    with pytest.raises(SystemExit):
        cmd_align(bids_dir, tmp_path / "out", pairs_csv, skip_bids_validation=True)

    nirs = tmp_path / "out" / "aligned" / "sub-10031" / "nirs"
    assert json.loads((nirs / "sub-10031_task-hold_nirs.json").read_text())["align_group"] == "A"


def _into_sessions(bids_dir, sessions=("a", "b")):
    """Each subject's one recording, copied into one folder per session under its name."""
    for sub_dir in bids_dir.glob("sub-*"):
        flat = sub_dir / "nirs"
        for ses in sessions:
            dest = sub_dir / f"ses-{ses}" / "nirs"
            dest.mkdir(parents=True)
            for f in flat.iterdir():
                name = f.name.replace(f"{sub_dir.name}_", f"{sub_dir.name}_ses-{ses}_", 1)
                shutil.copy2(f, dest / name)
        shutil.rmtree(flat)


def test_a_two_session_tree_aligns_the_session_the_csv_names(tmp_path):
    """The recording aligned, the file written and the sidecars copied are the same one.

    Loading honoured the CSV's session column, but the writer looked the file up again with
    no session, so on a two-session tree it could not tell the two apart.
    """
    bids_dir, pairs_csv = make_hyper_dataset(tmp_path, tasks=("hold",))
    _into_sessions(bids_dir)
    rows = pairs_csv.read_text().splitlines()
    pairs_csv.write_text("\n".join([rows[0] + ",session"] + [r + ",a" for r in rows[1:]]) + "\n")

    cmd_align(bids_dir, tmp_path / "out", pairs_csv, skip_bids_validation=True)

    for subject in ("10031", "10032"):
        sub = tmp_path / "out" / "aligned" / f"sub-{subject}"
        assert (sub / "ses-a" / "nirs" / f"sub-{subject}_ses-a_task-hold_nirs.snirf").exists()
        assert not (sub / "ses-b").exists() and not (sub / "nirs").exists()
