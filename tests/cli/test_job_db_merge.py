"""Merging the run logs moves each finished execution into the database once.

An execution is merged when its log says it ended, and its files are then moved to
``archived/`` beside where they were, so a second merge finds nothing new to insert. A
rebuild reads the archived logs too and writes a new database, never the old one.
"""

import sqlite3
from pathlib import Path

import pytest

from fnirs_pipe.utils import job_db


def _execution(db: Path, subject: str, *, finished: bool = True) -> int:
    eid = job_db.log_execution(db, command_line="fnirs-pipe ...", fnirs_pipe_version="test",
                               input_dir="in", output_dir="out", subjects=[subject])
    job_db.log_run_start(db, eid, subject, bids_task="rest")
    job_db.log_sqm(db, eid, subject, "raw", {"sci_mean": 0.9}, bids_task="rest")
    if finished:
        job_db.log_run_end(db, eid, subject, status="COMPLETED")
        job_db.update_execution(db, eid, "COMPLETED")
    return eid


def _counts(db: Path) -> dict:
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("pipeline_executions", "runs", "sqm")}
    finally:
        conn.close()


def _live_logs(db: Path) -> list[Path]:
    return [p for p in (db.parent / "json").rglob("*.jsonl") if "archived" not in p.parts]


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "logs" / "fnirs_pipe.db"


def test_merging_twice_inserts_nothing_the_second_time(db):
    _execution(db, "01")
    job_db.merge_jsonl(db)
    job_db.merge_jsonl(db)
    assert _counts(db) == {"pipeline_executions": 1, "runs": 1, "sqm": 1}


def test_a_merged_execution_s_logs_are_archived(db):
    _execution(db, "01")
    job_db.merge_jsonl(db)
    assert _live_logs(db) == []
    archived = list((db.parent / "json").rglob("archived/*.jsonl"))
    assert len(archived) == 3                 # execution, run, sqm


def test_an_execution_still_running_waits_for_the_next_merge(db, capsys):
    eid = _execution(db, "01", finished=False)
    job_db.merge_jsonl(db)
    assert _counts(db) == {"pipeline_executions": 0, "runs": 0, "sqm": 0}
    assert "1 execution" in capsys.readouterr().out

    job_db.log_run_end(db, eid, "01", status="COMPLETED")
    job_db.update_execution(db, eid, "COMPLETED")
    job_db.merge_jsonl(db)
    job_db.merge_jsonl(db)
    assert _counts(db) == {"pipeline_executions": 1, "runs": 1, "sqm": 1}


def test_a_log_that_cannot_be_archived_is_not_merged(db, monkeypatch):
    """Merged but left in place, it would be inserted again by the next merge."""
    _execution(db, "01")
    real_replace = Path.replace

    def refuse(self, target):
        raise PermissionError("file in use")

    monkeypatch.setattr(Path, "replace", refuse)
    job_db.merge_jsonl(db)
    assert _counts(db) == {"pipeline_executions": 0, "runs": 0, "sqm": 0}
    assert len(_live_logs(db)) == 3

    monkeypatch.setattr(Path, "replace", real_replace)
    job_db.merge_jsonl(db)
    assert _counts(db) == {"pipeline_executions": 1, "runs": 1, "sqm": 1}


def test_an_existing_database_is_backed_up_before_a_merge(db):
    _execution(db, "01")
    job_db.merge_jsonl(db)
    _execution(db, "02")
    job_db.merge_jsonl(db)
    backups = list((db.parent / "backup").glob("fnirs_pipe.backup_*.db"))
    assert len(backups) == 1
    assert _counts(backups[0]) == {"pipeline_executions": 1, "runs": 1, "sqm": 1}


def test_a_rebuild_reads_the_archive_into_a_new_database(db):
    _execution(db, "01")
    job_db.merge_jsonl(db)
    _execution(db, "02")
    job_db.merge_jsonl(db)
    before = db.read_bytes()

    rebuilt = job_db.rebuild_db(db)
    assert rebuilt != db and rebuilt.parent == db.parent
    assert _counts(rebuilt) == {"pipeline_executions": 2, "runs": 2, "sqm": 2}
    assert db.read_bytes() == before          # the original is never touched
    assert _live_logs(db) == []               # and nothing is moved back
