"""The two append-only rating logs sit under `logs/`, which `.bidsignore` already exempts."""

from fnirs_pipe.io.derivatives import write_bidsignore
from fnirs_pipe.io.naming import report_name
from fnirs_pipe.qc.rating.app import FNIRSRatingApp, RawRatingApp


def test_the_subject_rating_log_goes_under_logs(tmp_path):
    FNIRSRatingApp(tmp_path, ["01"])._append_jsonl("01", {"carpet": "good"}, {})
    assert (tmp_path / "logs" / "group_ratings.jsonl").exists()
    assert not list(tmp_path.glob("*.jsonl"))


def test_the_raw_rating_log_goes_under_logs(tmp_path):
    html = tmp_path / "sub-01" / report_name("sub-01_task-rest", desc="raw")
    RawRatingApp(html, tmp_path)._append_jsonl(html.stem, {"carpet": "good"}, {})
    assert (tmp_path / "logs" / "group_raw_ratings.jsonl").exists()
    assert not list(tmp_path.glob("*.jsonl"))


def test_logs_is_one_of_the_exemptions(tmp_path):
    write_bidsignore(tmp_path)
    assert "logs" in (tmp_path / ".bidsignore").read_text(encoding="utf-8").split()
