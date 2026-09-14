"""Three things that only go wrong when two runs are going at once.

Dyad runs write under their own `group-<id>/` and are already safe to run side by side. The
subject pipeline is not, for three reasons, and each fails in a way that looks like something
else: a reader sees a truncated `dataset_description.json` and reports bad BIDS, a second
merge reports "database is locked" partway through, and two runs started in the same
millisecond quietly file their records under one execution.

None of them is reproducible on demand, so these pin the mechanism rather than the race.
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.utils import job_db


# ---- dataset_description.json ----

def test_the_description_is_renamed_over_rather_than_truncated(tmp_path, monkeypatch):
    """`write_text` opens for truncation, so a reader between the open and the write sees
    nothing. The temporary carries the pid, or two writers would race for it as well."""
    seen = {}
    real_replace = job_db.os.replace

    def spy(src, dst):
        seen["src"], seen["dst"] = str(src), str(dst)
        return real_replace(src, dst)

    monkeypatch.setattr("fnirs_pipe.io.derivatives.os.replace", spy)
    write_dataset_description(tmp_path)

    assert seen["dst"].endswith("dataset_description.json")
    assert str(job_db.os.getpid()) in seen["src"], "the temporary does not name its process"
    assert seen["src"] != seen["dst"], "written straight over the target"
    assert json.loads((tmp_path / "dataset_description.json").read_text())["DatasetType"] == \
        "derivative"


def test_a_reader_never_sees_a_half_written_file(tmp_path):
    """The property that matters, and the one Windows allows.

    A rename cannot stop a reader opening the path at the instant it flips, so on Windows a
    reader can still be told the file is busy. It can no longer be handed a truncated one,
    which is the failure that reads as bad BIDS instead of as something to retry.
    """
    corrupt, busy = [], []

    def write(_):
        for _ in range(40):
            write_dataset_description(tmp_path)

    def read(_):
        path = tmp_path / "dataset_description.json"
        for _ in range(200):
            try:
                if path.exists():
                    assert json.loads(path.read_text())["DatasetType"] == "derivative"
            except OSError:
                busy.append(1)
            except Exception as exc:
                corrupt.append(repr(exc))

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda f: f(0), [write] * 4 + [read] * 2))

    assert not corrupt, f"a reader parsed a partial file: {corrupt[:3]}"
    assert not list(tmp_path.glob("*.tmp")), "a temporary was left behind"
    assert json.loads((tmp_path / "dataset_description.json").read_text())


@pytest.mark.parametrize("exc", [OSError("disk full"), PermissionError("held open")])
def test_a_write_that_cannot_finish_takes_its_temporary_with_it(tmp_path, monkeypatch, exc):
    """The retry loop exists for PermissionError and gives up after `_REPLACE_TRIES`. Either
    way the temporary has to go: it sits in the output root, where the next reader of the
    tree finds a file that is neither BIDS nor anything else."""
    def boom(src, dst):
        raise exc

    monkeypatch.setattr("fnirs_pipe.io.derivatives.os.replace", boom)
    monkeypatch.setattr("fnirs_pipe.io.derivatives._REPLACE_WAIT_S", 0.0)
    with pytest.raises(OSError):
        write_dataset_description(tmp_path)
    assert not list(tmp_path.glob("*.tmp")), "a failed write left its temporary behind"


def test_a_run_that_finds_it_already_written_leaves_it_alone(tmp_path, monkeypatch):
    """What keeps concurrent runs off the file at all: the content is fixed, so only the
    first run writes and the rename above is reached once rather than on every run."""
    write_dataset_description(tmp_path)
    calls = []
    monkeypatch.setattr("fnirs_pipe.io.derivatives.os.replace",
                        lambda src, dst: calls.append(1))
    write_dataset_description(tmp_path)
    assert not calls, "rewrote a description that was already correct"


# ---- the merge database ----

def test_the_merge_database_is_opened_in_wal_with_a_wait(tmp_path):
    """Default sqlite locks the whole file for a writer and gives up after five seconds."""
    db = tmp_path / "logs" / "fnirs_pipe.db"
    conn = job_db._get_conn(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000
    finally:
        conn.close()


def test_a_second_connection_waits_instead_of_failing(tmp_path):
    """The property WAL buys: a reader is not shut out while a writer holds the file."""
    db = tmp_path / "logs" / "fnirs_pipe.db"
    writer = job_db._get_conn(db)
    reader = job_db._get_conn(db)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO pipeline_executions (execution_id) VALUES (1)")
        # under the default rollback journal this raises "database is locked"
        reader.execute("SELECT count(*) FROM pipeline_executions").fetchone()
    except sqlite3.OperationalError as exc:  # pragma: no cover - the failure we guard against
        raise AssertionError(f"a concurrent reader was locked out: {exc}") from exc
    finally:
        writer.rollback()
        writer.close()
        reader.close()


# ---- the execution id ----

def test_two_runs_in_one_millisecond_get_different_ids(tmp_path, monkeypatch):
    """The id is what every later record joins on, so a shared one merges two runs into one."""
    db = tmp_path / "logs" / "fnirs_pipe.db"
    monkeypatch.setattr(job_db.time, "time", lambda: 1_800_000_000.123)

    ids = []
    for pid in (4242, 4243):
        monkeypatch.setattr(job_db.os, "getpid", lambda pid=pid: pid)
        ids.append(job_db.log_execution(
            db_path=db, command_line="fnirs-pipe", fnirs_pipe_version="0",
            input_dir="in", output_dir="out", subjects=["01"]))

    assert ids[0] != ids[1], "two processes in the same millisecond shared an execution"
    assert all(isinstance(i, int) for i in ids)
    assert len(list((db.parent / "json" / "_pipeline").glob("*.jsonl"))) == 2


def test_the_id_still_sorts_by_time(tmp_path, monkeypatch):
    """Runs are read back in order, so the pid must be below the clock, not above it."""
    db = tmp_path / "logs" / "fnirs_pipe.db"
    monkeypatch.setattr(job_db.os, "getpid", lambda: 99999)
    monkeypatch.setattr(job_db.time, "time", lambda: 1_800_000_000.000)
    early = job_db.log_execution(db_path=db, command_line="a", fnirs_pipe_version="0",
                                 input_dir="in", output_dir="out", subjects=["01"])
    monkeypatch.setattr(job_db.os, "getpid", lambda: 1)
    monkeypatch.setattr(job_db.time, "time", lambda: 1_800_000_000.001)
    late = job_db.log_execution(db_path=db, command_line="b", fnirs_pipe_version="0",
                                input_dir="in", output_dir="out", subjects=["01"])
    assert early < late, "a later run with a smaller pid sorted first"


def test_the_per_record_files_are_keyed_by_execution_too(tmp_path, monkeypatch):
    """`_sqm` and `_outputs` name themselves after the subject and the millisecond, unlike
    `_pipeline` and `_runs`. Two runs reaching one subject together would share the file."""
    db = tmp_path / "logs" / "fnirs_pipe.db"
    monkeypatch.setattr(job_db.time, "time", lambda: 1_800_000_000.123)

    for execution_id in (1, 2):
        job_db.log_sqm(db, execution_id, "01", "prep", {"sci": 0.9})
        job_db.log_output(db, execution_id, "01", command="fnirs-pipe")

    json_dir = db.parent / "json"
    assert len(list((json_dir / "_sqm").glob("*.jsonl"))) == 2
    assert len(list((json_dir / "_outputs").glob("*.jsonl"))) == 2
