"""What `fnirs-prep align` leaves on disk: one aligned tree, every offset beside its file.

It also wrote `group-<id>_task-<task>_align-offsets.tsv` at the tree's root, a name with no
suffix and a data product nothing read. A shift a step applied to one file is recorded in
that file's sidecar instead, which is what the reference BIDS Apps do with a time shift, and
this tree is read as an input next, where a group table has no place.
"""

import json

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
