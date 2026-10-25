"""The command every CLI page prints: a flag beside its values, a path quoted for the chosen shell."""

from __future__ import annotations

import ctypes
import shlex
import sys

import pytest

from nirspipe.interface.commands import preview_text

_ARGVS = [
    ["nirspipe-qc", "C:\\data\\bids", "C:\\out\\", "participant", "--sci-threshold", "0.8",
     "--participant-label", "01", "02", "--skip-bids-validation"],
    ["nirspipe-prep", "edit-markers", "apply", "/data/my bids", "/out",
     "--rename", "1:alone", "2.0:collab", "--shift", "-2.5"],
    ["nirspipe-hyper", "S:\\derivatives", "S:\\a b\\", "group", "--pairs-csv", 'it\'s "x".csv'],
]


def _joined(text: str, cont: str) -> str:
    return text.replace(f" {cont}\n", " ")


@pytest.mark.parametrize("argv", _ARGVS)
def test_the_bash_preview_pastes_back_to_the_same_argv(argv):
    assert shlex.split(_joined(preview_text(argv, "bash"), "\\")) == argv


@pytest.mark.skipif(sys.platform != "win32", reason="the C runtime's own parser")
@pytest.mark.parametrize("argv", _ARGVS)
def test_the_cmd_preview_parses_back_to_the_same_argv(argv):
    parse = ctypes.windll.shell32.CommandLineToArgvW
    parse.restype = ctypes.POINTER(ctypes.c_wchar_p)
    count = ctypes.c_int()
    parsed = parse(_joined(preview_text(argv, "cmd"), "^"), ctypes.byref(count))
    assert [parsed[i] for i in range(count.value)] == argv


def test_a_flag_keeps_its_values_and_a_positional_takes_a_line():
    text = preview_text(["nirspipe-qc", "/bids", "/out", "group",
                         "--pairs-csv", "pairs.csv", "--participant-label", "101", "201",
                         "--skip-bids-validation"], "bash")
    assert text.splitlines() == [
        "nirspipe-qc \\",
        "  '/bids' \\",
        "  '/out' \\",
        "  group \\",
        "  --pairs-csv pairs.csv \\",
        "  --participant-label 101 201 \\",
        "  --skip-bids-validation",
    ]


@pytest.mark.parametrize("shell, cont, quoted", [
    ("bash", "\\", "'C:\\x y\\'"),
    ("powershell", "`", "'C:\\x y\\'"),
    ("cmd", "^", '"C:\\x y\\\\"'),
])
def test_each_shell_quotes_a_path_its_own_way(shell, cont, quoted):
    first, path = preview_text(["nirspipe-provenance", "C:\\x y\\"], shell).splitlines()
    assert first == f"nirspipe-provenance {cont}"
    assert path == f"  {quoted}"
