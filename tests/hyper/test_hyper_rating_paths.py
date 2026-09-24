"""The dyad rating server reads its group, session and task off the report it serves.

It matches the stem the raw dyad report carries: a stem it missed would send every served
page to task `unknown` and each member's channel decisions to a file the raw QC page never
reads.
"""

from fnirs_pipe.io.derivatives import channel_decisions_path
from fnirs_pipe.io.naming import report_name
from fnirs_pipe.qc.rating.app import HyperRatingApp


def test_the_decisions_file_is_the_one_the_raw_page_keeps(tmp_path):
    html = tmp_path / "group-G1" / report_name("group-G1_ses-01_task-rest", desc="raw")
    app = HyperRatingApp(html, tmp_path, ["01", "02"])

    assert (app.group_id, app.session, app.task) == ("G1", "01", "rest")
    assert app._decisions_path("01") == channel_decisions_path(
        tmp_path, "01", task="rest", session="01")


def test_a_tree_without_sessions_reads_none(tmp_path):
    html = tmp_path / "group-G1" / report_name("group-G1_task-rest", desc="raw")
    app = HyperRatingApp(html, tmp_path, ["01", "02"])

    assert (app.group_id, app.session, app.task) == ("G1", None, "rest")
