"""What the two CLI-driving pages say while a command runs and when it ends.

The launch and the poll live in ``_cli_run`` for both pages, so the wording each page had
is pinned here: the Analysis page names the pipeline, the batch page does not.
"""

from __future__ import annotations

import pytest

from nirspipe.interface import process_stream
from nirspipe.interface.callbacks._cli_run import poll_run, start_run


def _header(panel) -> str:
    return panel.children[0].children


@pytest.mark.parametrize("returncode, noun, expected", [
    (0, "Pipeline", "Pipeline finished."), (0, "", "Finished."),
    (-9, "Pipeline", "Pipeline stopped (signal 9)."), (-9, "", "Stopped (signal 9)."),
    (2, "Pipeline", "Pipeline failed (exit 2)."), (2, "", "Failed (exit 2)."),
])
def test_the_outcome_line_keeps_each_page_s_wording(monkeypatch, returncode, noun, expected):
    monkeypatch.setattr(process_stream, "poll", lambda run_id: (["line"], returncode, 0))
    monkeypatch.setattr(process_stream, "forget", lambda run_id: None)
    panel, tick_off, stop_off, run_off = poll_run("r1", noun)
    assert _header(panel) == expected
    assert (tick_off, stop_off, run_off) == (True, True, False)


def test_a_running_command_keeps_the_tick_on(monkeypatch):
    monkeypatch.setattr(process_stream, "poll", lambda run_id: ([], None, 0))
    _, tick_off, stop_off, run_off = poll_run("r1")
    assert (tick_off, stop_off, run_off) == (False, False, True)


def test_no_command_yet_names_the_page_s_button():
    status, run_id, *_ = start_run(None, "Generate command", "`{exe}` not found on PATH.")
    assert "Click 'Generate command' first." in str(status.children) and run_id is None


def test_a_missing_executable_is_named(monkeypatch):
    def _missing(argv):
        raise FileNotFoundError
    monkeypatch.setattr(process_stream, "start", _missing)
    status, run_id, *_ = start_run({"argv": ["nirspipe", "x"]}, "Generate Command",
                                   "`{exe}` not found on PATH - make sure nirspipe is installed.")
    assert status.children == "`nirspipe` not found on PATH - make sure nirspipe is installed."
    assert run_id is None
