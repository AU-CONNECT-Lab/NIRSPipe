"""JSONL-based job tracking with SQLite merge for fnirs-pipe.

Write flow: pipeline events → JSONL files under logs/json/
Merge flow: fnirs-db merge → reads all JSONLs → inserts into logs/fnirs_pipe.db
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _json_dir(db_path: Path) -> Path:
    return db_path.parent / "json"


# --- JSONL writers ---

def log_execution(
    db_path: Path,
    command_line: str,
    fnirs_pipe_version: str,
    input_dir: str,
    output_dir: str,
    subjects: list[str],
    *,
    work_dir: str | None = None,
    session_labels: list[str] | None = None,
    task_labels: list[str] | None = None,
    mode: str | None = None,
    dry_run: bool = False,
) -> int:
    """Log pipeline invocation start. Returns execution_id."""
    execution_id = int(time.time() * 1000)
    record = {
        "event": "execution_start",
        "execution_id": execution_id,
        "timestamp": datetime.now().isoformat(),
        "command_line": command_line,
        "fnirs_pipe_version": fnirs_pipe_version,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "work_dir": work_dir,
        "subjects": subjects,
        "session_labels": session_labels,
        "task_labels": task_labels,
        "mode": mode,
        "dry_run": dry_run,
        "status": "RUNNING",
    }
    path = _json_dir(db_path) / "_pipeline" / f"execution_{execution_id}.jsonl"
    _write_jsonl(path, record)
    return execution_id


def update_execution(
    db_path: Path,
    execution_id: int,
    status: str,
    error_msg: str | None = None,
) -> None:
    record = {
        "event": "execution_update",
        "execution_id": execution_id,
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "error_msg": error_msg,
    }
    path = _json_dir(db_path) / "_pipeline" / f"execution_{execution_id}.jsonl"
    _write_jsonl(path, record)


def log_run_start(
    db_path: Path,
    execution_id: int,
    subject: str,
    *,
    session: str | None = None,
    bids_task: str | None = None,
    sci_threshold: float | None = None,
    dpf: list[float] | None = None,
    motion_correction: str | None = None,
    mode: str | None = None,
    high_pass: float | None = None,
    low_pass: float | None = None,
    hrf_model: str | None = None,
) -> None:
    record = {
        "event": "run_start",
        "execution_id": execution_id,
        "subject": subject,
        "session": session,
        "bids_task": bids_task,
        "start_time": datetime.now().isoformat(),
        "status": "RUNNING",
        "sci_threshold": sci_threshold,
        "dpf": dpf,
        "motion_correction": motion_correction,
        "mode": mode,
        "high_pass": high_pass,
        "low_pass": low_pass,
        "hrf_model": hrf_model,
    }
    path = _json_dir(db_path) / "_runs" / f"sub-{subject}_{execution_id}.jsonl"
    _write_jsonl(path, record)


def log_run_end(
    db_path: Path,
    execution_id: int,
    subject: str,
    status: str,
    *,
    session: str | None = None,
    exit_code: int | None = None,
    error_msg: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    record = {
        "event": "run_end",
        "execution_id": execution_id,
        "subject": subject,
        "session": session,
        "end_time": datetime.now().isoformat(),
        "status": status,
        "exit_code": exit_code,
        "error_msg": error_msg,
        "duration_seconds": duration_seconds,
    }
    path = _json_dir(db_path) / "_runs" / f"sub-{subject}_{execution_id}.jsonl"
    _write_jsonl(path, record)


def log_sqm(
    db_path: Path,
    execution_id: int,
    subject: str,
    checkpoint: str,
    sqm: dict[str, Any],
    *,
    session: str | None = None,
    bids_task: str | None = None,
) -> None:
    """Log SQM scalars (dict values must be scalar, not nested dicts)."""
    record = {
        "event": "sqm",
        "execution_id": execution_id,
        "subject": subject,
        "session": session,
        "bids_task": bids_task,
        "checkpoint": checkpoint,
        "timestamp": datetime.now().isoformat(),
        **{k: v for k, v in sqm.items() if not isinstance(v, (dict, list))},
    }
    ts = int(time.time() * 1000)
    path = _json_dir(db_path) / "_sqm" / f"sub-{subject}_{checkpoint}_{ts}.jsonl"
    _write_jsonl(path, record)


def log_output(
    db_path: Path,
    execution_id: int,
    subject: str,
    *,
    session: str | None = None,
    command: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    exit_code: int | None = None,
    log_file_path: str | None = None,
) -> None:
    def _tail(s: str | None, n: int = 50) -> str | None:
        if not s:
            return s
        lines = s.splitlines()
        return "\n".join(lines[-n:]) if len(lines) > n else s

    record = {
        "event": "output",
        "execution_id": execution_id,
        "subject": subject,
        "session": session,
        "execution_time": datetime.now().isoformat(),
        "command": command,
        "stdout": _tail(stdout),
        "stderr": _tail(stderr),
        "exit_code": exit_code,
        "log_file_path": log_file_path,
    }
    ts = int(time.time() * 1000)
    path = _json_dir(db_path) / "_outputs" / f"sub-{subject}_{ts}.jsonl"
    _write_jsonl(path, record)


# --- SQLite schema ---

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_executions (
    id                   INTEGER PRIMARY KEY,
    execution_id         INTEGER UNIQUE,
    execution_time       TEXT,
    command_line         TEXT,
    fnirs_pipe_version   TEXT,
    input_dir            TEXT,
    output_dir           TEXT,
    work_dir             TEXT,
    subjects             TEXT,
    session_labels       TEXT,
    task_labels          TEXT,
    mode                 TEXT,
    dry_run              INTEGER,
    status               TEXT,
    error_msg            TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id                   INTEGER PRIMARY KEY,
    execution_id         INTEGER,
    subject              TEXT,
    session              TEXT,
    bids_task            TEXT,
    start_time           TEXT,
    end_time             TEXT,
    status               TEXT,
    exit_code            INTEGER,
    error_msg            TEXT,
    duration_seconds     REAL,
    sci_threshold        REAL,
    dpf                  TEXT,
    motion_correction    TEXT,
    mode                 TEXT,
    high_pass            REAL,
    low_pass             REAL,
    hrf_model            TEXT
);

CREATE TABLE IF NOT EXISTS sqm (
    id                          INTEGER PRIMARY KEY,
    execution_id                INTEGER,
    subject                     TEXT,
    session                     TEXT,
    bids_task                   TEXT,
    checkpoint                  TEXT,
    sci_mean                    REAL,
    channel_retention_rate      REAL,
    snr_mean                    REAL,
    cv_mean                     REAL,
    mean_amp_mean               REAL,
    ch_dist_mean                REAL,
    psp_mean                    REAL,
    qc_window_s                 REAL,
    cp_mean                     REAL,
    gvtd_mean                   REAL,
    gvtd_p95                    REAL,
    gvtd_filt_mean              REAL,
    gvtd_filt_p95               REAL,
    gvtd_vstd_mean              REAL,
    gvtd_vstd_p95               REAL,
    gvtd_thresh                 REAL,
    gvtd_num_above_thresh       INTEGER,
    gvtd_pct_above_thresh       REAL,
    spike_count                 INTEGER,
    hbo_hbr_corr_mean           REAL,
    gcor_hbo                    REAL,
    gcor_hbr                    REAL,
    cardiac_band_power_hbo      REAL,
    cardiac_band_power_hbr      REAL,
    cardiac_band_frac_hbo       REAL,
    cardiac_band_frac_hbr       REAL,
    resp_band_power_hbo         REAL,
    resp_band_power_hbr         REAL,
    resp_band_frac_hbo          REAL,
    resp_band_frac_hbr          REAL,
    lowfreq_drift_amplitude_hbo REAL,
    lowfreq_drift_amplitude_hbr REAL,
    pct_data_retained           REAL
);

CREATE TABLE IF NOT EXISTS command_outputs (
    id             INTEGER PRIMARY KEY,
    execution_id   INTEGER,
    subject        TEXT,
    session        TEXT,
    command        TEXT,
    stdout         TEXT,
    stderr         TEXT,
    exit_code      INTEGER,
    execution_time TEXT,
    log_file_path  TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_lookup  ON runs  (execution_id, subject);
CREATE INDEX IF NOT EXISTS idx_sqm_lookup   ON sqm   (execution_id, subject, checkpoint);
CREATE INDEX IF NOT EXISTS idx_out_lookup   ON command_outputs (execution_id, subject);
"""

_SQM_COLS = [
    "sci_win_mean", "sci_mean", "channel_retention_rate", "snr_mean", "cv_mean",
    "mean_amp_mean",
    "ch_dist_mean", "psp_mean", "qc_window_s", "cp_mean", "gvtd_mean", "gvtd_p95",
    "gvtd_filt_mean", "gvtd_filt_p95", "gvtd_vstd_mean", "gvtd_vstd_p95",
    "gvtd_thresh", "gvtd_num_above_thresh",
    "gvtd_pct_above_thresh", "spike_count",
    "hbo_hbr_corr_mean", "gcor_hbo", "gcor_hbr",
    "cardiac_band_power_hbo", "cardiac_band_power_hbr",
    "cardiac_band_frac_hbo", "cardiac_band_frac_hbr",
    "resp_band_power_hbo", "resp_band_power_hbr",
    "resp_band_frac_hbo", "resp_band_frac_hbr",
    "lowfreq_drift_amplitude_hbo", "lowfreq_drift_amplitude_hbr",
    "pct_data_retained",
    "cnr_hbo_mean", "cnr_hbr_mean", "cnr_n_epochs",
]


_SQM_INT_COLS = {"gvtd_num_above_thresh", "spike_count", "cnr_n_epochs"}


def _add_missing_sqm_columns(conn: sqlite3.Connection) -> None:
    """Bring an sqm table built by an older version up to the current column list.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table alone, so a database written
    before a metric was added has no column for it and every later insert fails on that
    name. Adding the columns is the whole migration: SQLite fills them with NULL.
    """
    have = {row[1] for row in conn.execute("PRAGMA table_info(sqm)")}
    for col in _SQM_COLS:
        if col not in have:
            kind = "INTEGER" if col in _SQM_INT_COLS else "REAL"
            conn.execute(f"ALTER TABLE sqm ADD COLUMN {col} {kind}")


def _get_conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    _add_missing_sqm_columns(conn)
    return conn


# --- Merge ---

def merge_jsonl(db_path: Path) -> tuple[int, int]:
    """Read all JSONL files and insert into SQLite. Returns (files_processed, rows_inserted)."""
    json_dir = _json_dir(db_path)
    if not json_dir.exists():
        return 0, 0

    conn = _get_conn(db_path)
    executions: dict[int, dict] = {}
    files = 0
    rows = 0

    for jsonl_file in sorted(json_dir.rglob("*.jsonl")):
        files += 1
        with jsonl_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows += _dispatch(conn, rec, executions)

    for ex in executions.values():
        _upsert_execution(conn, ex)
        rows += 1

    conn.commit()
    conn.close()
    return files, rows


def _dispatch(
    conn: sqlite3.Connection,
    rec: dict[str, Any],
    executions: dict[int, dict],
) -> int:
    event = rec.get("event")
    eid = rec.get("execution_id")

    if event == "execution_start":
        executions[eid] = {
            "execution_id": eid,
            "execution_time": rec.get("timestamp"),
            "command_line": rec.get("command_line"),
            "fnirs_pipe_version": rec.get("fnirs_pipe_version"),
            "input_dir": rec.get("input_dir"),
            "output_dir": rec.get("output_dir"),
            "work_dir": rec.get("work_dir"),
            "subjects": json.dumps(rec.get("subjects")),
            "session_labels": json.dumps(rec.get("session_labels")),
            "task_labels": json.dumps(rec.get("task_labels")),
            "mode": rec.get("mode"),
            "dry_run": int(rec.get("dry_run", False)),
            "status": rec.get("status", "RUNNING"),
            "error_msg": None,
        }
        return 0  # buffered; flushed after all files

    elif event == "execution_update":
        ex = executions.setdefault(eid, {"execution_id": eid})
        ex["status"] = rec.get("status", ex.get("status", "UNKNOWN"))
        ex["error_msg"] = rec.get("error_msg")
        return 0

    elif event == "run_start":
        conn.execute(
            """
            INSERT OR IGNORE INTO runs
              (execution_id, subject, session, bids_task, start_time, status,
               sci_threshold, dpf, motion_correction, mode, high_pass, low_pass, hrf_model)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                eid, rec.get("subject"), rec.get("session"), rec.get("bids_task"),
                rec.get("start_time"), rec.get("status", "RUNNING"),
                rec.get("sci_threshold"),
                json.dumps(rec["dpf"]) if rec.get("dpf") else None,
                rec.get("motion_correction"), rec.get("mode"),
                rec.get("high_pass"), rec.get("low_pass"), rec.get("hrf_model"),
            ),
        )
        return 1

    elif event == "run_end":
        conn.execute(
            """
            UPDATE runs
               SET end_time=?, status=?, exit_code=?, error_msg=?, duration_seconds=?
             WHERE execution_id=? AND subject=?
            """,
            (
                rec.get("end_time"), rec.get("status"),
                rec.get("exit_code"), rec.get("error_msg"),
                rec.get("duration_seconds"),
                eid, rec.get("subject"),
            ),
        )
        return 1

    elif event == "sqm":
        vals = [rec.get(col) for col in _SQM_COLS]
        placeholders = ", ".join(["?"] * (5 + len(_SQM_COLS)))
        conn.execute(
            f"INSERT INTO sqm (execution_id, subject, session, bids_task, checkpoint,"
            f" {', '.join(_SQM_COLS)}) VALUES ({placeholders})",
            [eid, rec.get("subject"), rec.get("session"),
             rec.get("bids_task"), rec.get("checkpoint")] + vals,
        )
        return 1

    elif event == "output":
        conn.execute(
            """
            INSERT INTO command_outputs
              (execution_id, subject, session, command, stdout, stderr,
               exit_code, execution_time, log_file_path)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                eid, rec.get("subject"), rec.get("session"),
                rec.get("command"), rec.get("stdout"), rec.get("stderr"),
                rec.get("exit_code"), rec.get("execution_time"),
                rec.get("log_file_path"),
            ),
        )
        return 1

    return 0


def _upsert_execution(conn: sqlite3.Connection, ex: dict[str, Any]) -> None:
    keys = [
        "execution_id", "execution_time", "command_line", "fnirs_pipe_version",
        "input_dir", "output_dir", "work_dir", "subjects", "session_labels",
        "task_labels", "mode", "dry_run", "status", "error_msg",
    ]
    conn.execute(
        f"INSERT OR REPLACE INTO pipeline_executions ({', '.join(keys)})"
        f" VALUES ({', '.join(['?'] * len(keys))})",
        [ex.get(k) for k in keys],
    )
