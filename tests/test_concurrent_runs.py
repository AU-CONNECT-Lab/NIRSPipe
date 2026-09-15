"""What happens when several workers process one dataset into one output tree at once.

Two fixes were made for this and neither had been tested under load, which is the whole point
of this file: a 23-dyad run is roughly a day and a half, and a sharing violation or a locked
database at hour eighteen costs more than these tests do.

Real processes, not threads. The Windows failure these guard against is a file being opened
for writing by one process while another holds it, which threads inside one interpreter do not
reproduce.

Two things turned out **not** to need a guard, and the tests say so rather than leaving it to
be rediscovered: per-run JSONL is written to a path keyed by execution id, so parallel workers
never share a file, and an execution id carries the pid, so two started in the same
millisecond still differ.
"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.utils import job_db

WORKERS = 8


# ---- Workers. Module level and argument-only, so spawn can pickle them ----

def _write_description(output_dir: str) -> str:
    write_dataset_description(Path(output_dir))
    return "ok"


def _read_description_repeatedly(output_dir: str, times: int) -> list[str]:
    """Read the file while others rewrite it, and report anything that was not valid BIDS."""
    path = Path(output_dir) / "dataset_description.json"
    bad = []
    for _ in range(times):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            # the instant of the rename; a retry, not a corrupt tree
            continue
        try:
            if json.loads(text).get("Name") != "fnirs-pipe output":
                bad.append(text[:80])
        except json.JSONDecodeError:
            bad.append(text[:80])
    return bad


def _log_one_execution(db_path: str, subject: str) -> int:
    db = Path(db_path)
    execution_id = job_db.log_execution(
        db, command_line=f"fnirs-pipe ... {subject}", fnirs_pipe_version="test",
        input_dir="in", output_dir="out", subjects=[subject],
    )
    job_db.log_run_start(db, execution_id, subject, bids_task="rest")
    job_db.log_run_end(db, execution_id, subject, status="COMPLETED")
    return execution_id


def _merge(db_path: str) -> tuple[int, int]:
    return job_db.merge_jsonl(Path(db_path))


# ---- dataset_description.json ----

def test_concurrent_writers_leave_a_valid_description(tmp_path):
    out = str(tmp_path / "derivatives")
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_write_description, out) for _ in range(WORKERS * 3)]
        for f in as_completed(futures):
            assert f.result() == "ok"

    written = json.loads((Path(out) / "dataset_description.json").read_text(encoding="utf-8"))
    assert written["Name"] == "fnirs-pipe output"
    assert written["BIDSVersion"] == "1.8.0"


def test_a_reader_never_sees_a_half_written_description(tmp_path):
    """The defect this replaced: a plain write truncates, so a reader saw an empty file.

    A reader can still be told the path is busy at the instant of the rename, which Windows
    offers no way around; that is caught above and is a retry. What must never happen is
    reading a file that parses but says the wrong thing, or does not parse at all.
    """
    out = str(tmp_path / "derivatives")
    write_dataset_description(Path(out))

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        writers = [pool.submit(_write_description, out) for _ in range(WORKERS * 4)]
        readers = [pool.submit(_read_description_repeatedly, out, 200) for _ in range(3)]
        for f in writers:
            f.result()
        seen_bad = [b for f in readers for b in f.result()]

    assert not seen_bad, f"readers saw invalid content: {seen_bad[:3]}"


def test_no_temporary_files_are_left_behind(tmp_path):
    out = tmp_path / "derivatives"
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in [pool.submit(_write_description, str(out)) for _ in range(WORKERS * 2)]:
            f.result()
    leftovers = [p.name for p in out.iterdir() if p.name != "dataset_description.json"]
    assert not leftovers, f"temp files survived: {leftovers}"


# ---- job database ----

def test_parallel_workers_do_not_share_a_jsonl_file(tmp_path):
    """Why the run phase needs no lock: the path carries the execution id."""
    db = tmp_path / "logs" / "pipeline.db"
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_log_one_execution, str(db), f"{i:03d}") for i in range(WORKERS)]
        ids = [f.result() for f in futures]

    assert len(set(ids)) == len(ids), "two parallel workers were given one execution id"
    run_files = list((db.parent / "json" / "_runs").glob("*.jsonl"))
    assert len(run_files) == WORKERS, "workers shared a run file"
    for path in run_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            json.loads(line)  # a line interleaved by another writer would not parse


def test_every_parallel_worker_survives_the_merge(tmp_path):
    db = tmp_path / "logs" / "pipeline.db"
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_log_one_execution, str(db), f"{i:03d}") for i in range(WORKERS)]
        ids = {f.result() for f in futures}

    job_db.merge_jsonl(db)
    conn = sqlite3.connect(db)
    try:
        merged = {row[0] for row in conn.execute("SELECT execution_id FROM pipeline_executions")}
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        conn.close()
    assert merged == ids, "a worker's execution did not reach the database"
    assert runs == WORKERS


def test_concurrent_merges_do_not_report_a_locked_database(tmp_path):
    """What WAL and the busy timeout are for. Without them this is "database is locked"."""
    db = tmp_path / "logs" / "pipeline.db"
    for i in range(WORKERS):
        _log_one_execution(str(db), f"{i:03d}")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_merge, str(db)) for _ in range(WORKERS)]
        for f in as_completed(futures):
            files, _rows = f.result()   # an OperationalError would surface here
            assert files == WORKERS * 2  # one _pipeline and one _runs file per worker


@pytest.mark.xfail(reason="merge is not idempotent: runs, sqm and command_outputs have no "
                          "unique key, so every merge inserts again. Not a concurrency bug; "
                          "two merges in a row do it too.",
                   strict=True)
def test_merge_is_idempotent(tmp_path):
    """Several merges racing must not multiply the rows, or a study's counts are wrong."""
    db = tmp_path / "logs" / "pipeline.db"
    for i in range(WORKERS):
        _log_one_execution(str(db), f"{i:03d}")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for f in [pool.submit(_merge, str(db)) for _ in range(WORKERS)]:
            f.result()

    conn = sqlite3.connect(db)
    try:
        executions = conn.execute("SELECT COUNT(*) FROM pipeline_executions").fetchone()[0]
        runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        conn.close()
    assert executions == WORKERS
    assert runs == WORKERS


def test_wal_is_actually_on(tmp_path):
    # the fix is one PRAGMA; without this the tests above could pass by luck on a fast machine
    db = tmp_path / "logs" / "pipeline.db"
    _log_one_execution(str(db), "001")
    job_db.merge_jsonl(db)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
