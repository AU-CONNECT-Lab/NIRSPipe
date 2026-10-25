"""The page has to find the report the command just wrote, or say that it could not.

`find_report` returning nothing and a command that writes no report must look different on
the page, or a pattern that no longer matches its writer would be invisible.
"""

from __future__ import annotations

import pytest

from nirspipe.io.naming import derivative_path, report_name
from nirspipe.interface.report_serve import REPORT_PATTERNS, find_report

# What each command actually writes. Built rather than spelled out, so this cannot agree
# with REPORT_PATTERNS while both disagree with the writers.
WRITTEN = {
    "participant":  "sub-01/" + report_name("sub-01", desc="index"),
    "raw-qc":       "sub-01/" + report_name("sub-01_task-rest", desc="raw"),
    "run":          "group-G1/" + report_name("group-G1_task-rest"),
    "index":        "group-G1/" + report_name("group-G1", desc="index"),
    "cohort":       derivative_path("", "report", ".html", desc="subjects").name,
    "groupnull":    derivative_path("", "report", ".html", task="rest", desc="groupnull").name,
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
