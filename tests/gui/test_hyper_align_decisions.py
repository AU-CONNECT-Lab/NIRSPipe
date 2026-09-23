"""The Hyper Preparation page files a member's channel decisions where the raw QC page does.

The page holds no session of its own, and it passed none, so on a tree with sessions it read
and wrote a path the raw QC page never touches. The session now comes off the file the
member was read from.
"""

from pathlib import Path

from fnirs_pipe.interface.callbacks.hyper_align_callbacks import _ha_decisions_path
from fnirs_pipe.io.derivatives import channel_decisions_path


def test_the_session_comes_from_the_file_the_member_was_read_from(tmp_path):
    source = Path("bids/sub-01/ses-a/nirs/sub-01_ses-a_task-rest_nirs.snirf")
    assert _ha_decisions_path(str(tmp_path), "sub-01", "rest", source) == \
        channel_decisions_path(tmp_path, "01", task="rest", session="a")


def test_a_tree_without_sessions_still_reads_the_plain_path(tmp_path):
    source = Path("bids/sub-01/nirs/sub-01_task-rest_nirs.snirf")
    assert _ha_decisions_path(str(tmp_path), "sub-01", "rest", source) == \
        channel_decisions_path(tmp_path, "01", task="rest")
