"""Run a CLI command in the background and let a page poll its output as it arrives.

The process is launched detached, one reader thread drains its output into a bounded buffer,
and the page reads that buffer on a timer, so a long command never freezes the page.

    run_id = start(["nirspipe", "..."])
    lines, returncode = poll(run_id)     # returncode is None while it is still running
    lines, nxt, returncode = read_from(run_id, 0)    # a stream asks from nxt next, so sends each line once

start_each runs several commands one after another under one id, for a page that converts
file by file.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque

# a long run prints more than a page can show; keep a scrollback rather than the whole history
_MAX_LINES = 2000

_runs: dict[str, "_Run"] = {}
_lock = threading.Lock()


class _Run:
    def __init__(self, steps: list[tuple[str | None, list[str]]]):
        self.steps = steps
        self.proc: subprocess.Popen | None = None
        self.returncode: int | None = None
        self.stopped = False
        self.lines: deque[str] = deque(maxlen=_MAX_LINES)
        self.dropped = 0
        self.lock = threading.Lock()
        # held across a launch, so Stop cannot fall between two commands and miss the next
        self.proc_lock = threading.Lock()

    def _append(self, line: str) -> None:
        with self.lock:
            if len(self.lines) == _MAX_LINES:
                self.dropped += 1
            self.lines.append(line)

    def _launch(self, header: str | None, argv: list[str]) -> None:
        if header:
            self._append(header)
        self.proc = _popen(argv)

    def _drain(self) -> None:
        n = len(self.steps)
        code, ok, attempted = 0, 0, 0
        for i, (header, argv) in enumerate(self.steps):
            # the first command was launched by start_each, so a missing executable raised there
            if i:
                with self.proc_lock:
                    if self.stopped:
                        break
                    try:
                        self._launch(header, argv)
                    except OSError as exc:
                        attempted += 1
                        code = 1
                        self._append(f"could not launch {argv[0]}: {exc}")
                        continue
            attempted += 1
            for line in self.proc.stdout:  # type: ignore[union-attr]
                self._append(line.rstrip("\n"))
            rc = self.proc.wait()  # type: ignore[union-attr]
            if rc:
                code = rc
                if n > 1:
                    # terminate() exits 1 on Windows, so the code alone cannot tell a stop
                    self._append("stopped" if self.stopped else f"failed (exit {rc})")
            else:
                ok += 1
        if n > 1:
            skipped = f", {n - attempted} not run" if attempted < n else ""
            self._append(f"{ok} of {n} succeeded{skipped}")
        # set last, so a poll that sees the code has already seen every line
        self.returncode = code


def _popen(argv: list[str]) -> subprocess.Popen:
    env = dict(os.environ)
    # without this a piped child buffers by block, so nothing reaches the page until it exits
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )


def start(argv: list[str]) -> str:
    """Launch argv and return the id to poll it with. Raises what Popen raises."""
    return start_each([(None, argv)])


def start_each(steps: list[tuple[str | None, list[str]]]) -> str:
    """Run (header, argv) steps one after another under one id; the exit code is the last failure's."""
    run = _Run(steps)
    run._launch(*steps[0])
    run_id = uuid.uuid4().hex[:12]
    with _lock:
        _runs[run_id] = run
    threading.Thread(target=run._drain, daemon=True, name=f"run-{run_id}").start()
    return run_id


def poll(run_id: str, tail: int = 200) -> tuple[list[str], int | None, int]:
    """Return (last ``tail`` lines, exit code or None while running, lines dropped)."""
    with _lock:
        run = _runs.get(run_id)
    if run is None:
        return [], None, 0
    # the code before the lines: the reader appends every line before it sets the code
    returncode = run.returncode
    with run.lock:
        lines = list(run.lines)[-tail:]
        dropped = run.dropped
    return lines, returncode, dropped


def read_from(run_id: str, start: int) -> tuple[list[str], int, int | None]:
    """Lines from the ``start``-th printed on, the index to ask for next, and the exit code or None.

    Raises KeyError for an id no run has. Lines already dropped from the scrollback are skipped.
    """
    with _lock:
        run = _runs[run_id]
    # the code before the lines: the reader appends every line before it sets the code
    returncode = run.returncode
    with run.lock:
        lines = list(run.lines)[max(start - run.dropped, 0):]
        total = run.dropped + len(run.lines)
    return lines, total, returncode


def sse(run_id: str):
    """The run's lines as server-sent events, then an ``end`` event with its exit code. Raises KeyError at once for an unknown id."""
    read_from(run_id, 0)

    def stream():
        seen = 0
        while True:
            lines, seen, code = read_from(run_id, seen)
            for line in lines:
                yield f"data: {json.dumps(line)}\n\n"
            if code is not None:
                end = {"code": code, "stopped": was_stopped(run_id)}
                yield f"event: end\ndata: {json.dumps(end)}\n\n"
                return
            time.sleep(0.2)

    return stream()


def stop(run_id: str) -> bool:
    """End the running command and skip any after it."""
    with _lock:
        run = _runs.get(run_id)
    if run is None or run.returncode is not None:
        return False
    with run.proc_lock:
        run.stopped = True
        proc = run.proc
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return True


def was_stopped(run_id: str) -> bool:
    """Whether Stop was pressed on this run, whatever exit code that produced."""
    with _lock:
        run = _runs.get(run_id)
    return run is not None and run.stopped


def forget(run_id: str) -> None:
    """Drop a finished run. A live one is left alone, so this cannot orphan a process."""
    with _lock:
        run = _runs.get(run_id)
        if run is not None and run.returncode is not None:
            del _runs[run_id]
