"""The background runner behind the GUI's run buttons.

The point of this module is that output arrives *while* the command runs, so the tests that
matter are the ones that read a line before the process has exited. The child scripts here
deliberately do not pass ``flush=True``: a piped child buffers by block unless something
unbuffers it, so a test that flushed by hand would pass even with the environment variable
gone, and the defect it exists to catch is exactly that variable going missing.
"""

from __future__ import annotations

import sys
import time

import pytest

from nirspipe.interface import process_stream


def _run(body: str) -> str:
    return process_stream.start([sys.executable, "-c", body])


def _wait_for(predicate, timeout=10.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return None


SLOW_PRINTER = (
    "import time\n"
    "for i in range(6):\n"
    "    print('line %d' % i)\n"
    "    time.sleep(0.3)\n"
)


def test_output_arrives_before_the_process_exits():
    run_id = _run(SLOW_PRINTER)
    got = _wait_for(lambda: process_stream.poll(run_id)[0] or None)
    assert got, "no output within the timeout: the child is buffering"

    lines, returncode, _ = process_stream.poll(run_id)
    assert returncode is None, "process already finished; this did not test streaming"
    assert lines[0] == "line 0"


def test_exit_code_and_full_output_once_finished():
    run_id = _run(SLOW_PRINTER)
    code = _wait_for(lambda: process_stream.poll(run_id)[1] is not None
                     and process_stream.poll(run_id)[1] is not None)
    assert code is not None
    lines, returncode, dropped = process_stream.poll(run_id)
    assert returncode == 0
    assert lines == [f"line {i}" for i in range(6)]
    assert dropped == 0


def test_stderr_is_interleaved_not_separate():
    run_id = _run("import sys\n"
                  "print('out')\n"
                  "print('err', file=sys.stderr)\n")
    _wait_for(lambda: process_stream.poll(run_id)[1] is not None)
    lines, _, _ = process_stream.poll(run_id)
    assert set(lines) == {"out", "err"}


def test_nonzero_exit_is_reported():
    run_id = _run("raise SystemExit(3)")
    _wait_for(lambda: process_stream.poll(run_id)[1] is not None)
    assert process_stream.poll(run_id)[1] == 3


def test_stop_ends_a_running_process():
    run_id = _run("import time\ntime.sleep(60)\n")
    assert _wait_for(lambda: process_stream.poll(run_id)[1] is None or True)
    assert process_stream.stop(run_id) is True
    assert _wait_for(lambda: process_stream.poll(run_id)[1] is not None) is not None
    assert process_stream.stop(run_id) is False, "a finished run cannot be stopped again"


def test_scrollback_is_bounded_and_says_what_it_dropped(monkeypatch):
    monkeypatch.setattr(process_stream, "_MAX_LINES", 10)
    run_id = _run("for i in range(50):\n    print(i)\n")
    _wait_for(lambda: process_stream.poll(run_id)[1] is not None)
    lines, _, dropped = process_stream.poll(run_id)
    assert len(lines) == 10
    assert lines[-1] == "49", "the tail is what is kept, not the head"
    assert dropped == 40


def test_polling_an_unknown_run_is_not_an_error():
    assert process_stream.poll("nope") == ([], None, 0)
    assert process_stream.stop("nope") is False
    process_stream.forget("nope")


def test_forget_leaves_a_live_run_alone():
    run_id = _run("import time\ntime.sleep(30)\n")
    process_stream.forget(run_id)
    assert process_stream.stop(run_id) is True, "forget() orphaned a running process"


def test_missing_executable_raises_rather_than_returning_an_id():
    with pytest.raises(FileNotFoundError):
        process_stream.start(["definitely-not-a-real-command-xyz"])
