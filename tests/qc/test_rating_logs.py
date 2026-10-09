"""The two append-only rating logs sit under `logs/`, which `.bidsignore` already exempts."""

from nirspipe.io.derivatives import write_bidsignore
from nirspipe.io.naming import report_name
from nirspipe.qc.rating.app import FNIRSRatingApp, RawRatingApp


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


def test_a_rating_file_reads_back_what_was_written(tmp_path):
    from nirspipe.qc.rating.app import _read_rating_file, _write_rating_file

    path = tmp_path / "r.json"
    assert _read_rating_file(path) == {"ratings": {}, "notes": {}}
    _write_rating_file(path, "sub-01_task-rest", {"overall": "good"}, {"overall": "fine"})
    assert _read_rating_file(path) == {"ratings": {"overall": "good"}, "notes": {"overall": "fine"}}
