"""The page has to find the report the command just wrote, or say that it could not.

`find_report` returning nothing and a command that writes no report must look different on
the page, or a pattern that no longer matches its writer would be invisible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import dash_bootstrap_components as dbc
import pytest
from dash import html

from nirspipe.io.naming import derivative_path, report_name
from nirspipe.interface.callbacks._cli_run import (
    REPORT_PATTERNS, find_report, run_and_report,
)

# What each command actually writes. Built rather than spelled out, so this cannot agree
# with REPORT_PATTERNS while both disagree with the writers.
WRITTEN = {
    "prep-raw":     "sub-01/" + report_name("sub-01_task-rest", desc="raw"),
    "hyper-raw":    "group-G1/" + report_name("group-G1_task-rest", desc="raw"),
    "run":          "group-G1/" + report_name("group-G1_task-rest"),
    "index":        "group-G1/" + report_name("group-G1", desc="index"),
    "cohort":       derivative_path("", "report", ".html", desc="subjects").name,
    "cohort-hyper": derivative_path("", "report", ".html", desc="groups").name,
}


def test_every_command_in_the_table_has_a_known_output():
    assert set(REPORT_PATTERNS) == set(WRITTEN)


@pytest.mark.parametrize("command", sorted(WRITTEN))
def test_the_pattern_finds_what_the_writer_writes(tmp_path, command):
    written = tmp_path / WRITTEN[command]
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text("<html></html>", encoding="utf-8")

    found = find_report(REPORT_PATTERNS[command], tmp_path)
    assert found == written, (
        f"REPORT_PATTERNS[{command!r}] does not match {WRITTEN[command]}; "
        f"one of the two was renamed without the other"
    )


def test_nothing_is_found_in_an_empty_tree(tmp_path):
    for command, patterns in REPORT_PATTERNS.items():
        assert find_report(patterns, tmp_path) is None, command


@pytest.fixture
def _ran_fine(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, "done", ""))


def _second_panel(stored):
    return run_and_report(stored)[1]


def test_a_command_that_writes_no_report_says_only_that(tmp_path, _ran_fine):
    panel = _second_panel({"argv": ["nirspipe-hyper-merge"], "command": "merge",
                           "output_dir": str(tmp_path)})
    assert isinstance(panel, html.Small)
    assert "tables" in panel.children


def test_a_missing_report_is_a_warning_not_a_shrug(tmp_path, _ran_fine):
    panel = _second_panel({"argv": ["nirspipe-qc", "prep-raw"], "command": "prep-raw",
                           "output_dir": str(tmp_path)})
    assert isinstance(panel, dbc.Alert), "a command that should have written a page did not"
    assert panel.color == "warning"


def test_the_report_is_linked_when_it_is_there(tmp_path, _ran_fine):
    written = tmp_path / WRITTEN["prep-raw"]
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text("<html></html>", encoding="utf-8")

    panel = _second_panel({"argv": ["nirspipe-qc", "prep-raw"], "command": "prep-raw",
                           "output_dir": str(tmp_path)})
    assert not isinstance(panel, dbc.Alert)
