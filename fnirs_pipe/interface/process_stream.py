"""Run a CLI command in the background and let a page poll its output as it arrives.

The GUI's run buttons used to block on ``subprocess.run`` and show the tail once the command
had finished, which on a real dataset means half an hour of a frozen page. Here the process is
launched detached, one reader thread drains its output into a bounded buffer, and the page
reads that buffer on a timer.

    run_id = start(["fnirs-pipe", "..."])
    lines, returncode = poll(run_id)     # returncode is None while it is still running
"""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from collections import deque

# a long run prints more than a page can show; keep a scrollback rather than the whole history
_MAX_LINES = 2000

_runs: dict[str, "_Run"] = {}
_lock = threading.Lock()


class _Run:
    def __init__(self, proc: subprocess.Popen, argv: list[str]):
        self.proc = proc
        self.argv = argv
        self.lines: deque[str] = deque(maxlen=_MAX_LINES)
        self.dropped = 0
        self.lock = threading.Lock()

    def _drain(self) -> None:
        for line in self.proc.stdout:  # type: ignore[union-attr]
            with self.lock:
                if len(self.lines) == _MAX_LINES:
                    self.dropped += 1
                self.lines.append(line.rstrip("\n"))
        self.proc.wait()


def start(argv: list[str]) -> str:
    """Launch argv and return the id to poll it with. Raises what Popen raises."""
    env = dict(os.environ)
    # without this a piped child buffers by block, so nothing reaches the page until it exits
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    run = _Run(proc, argv)
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
    with run.lock:
        lines = list(run.lines)[-tail:]
        dropped = run.dropped
    return lines, run.proc.poll(), dropped


def stop(run_id: str) -> bool:
    with _lock:
        run = _runs.get(run_id)
    if run is None or run.proc.poll() is not None:
        return False
    run.proc.terminate()
    return True


def forget(run_id: str) -> None:
    """Drop a finished run. A live one is left alone, so this cannot orphan a process."""
    with _lock:
        run = _runs.get(run_id)
        if run is not None and run.proc.poll() is not None:
            del _runs[run_id]
