"""Every --flag a message, a help text, a docstring or a docs page names is one a command
defines, so a renamed or removed flag cannot live on in the text that tells users what to type."""

import argparse
import re
from pathlib import Path

from nirspipe.cli import gui as gui_cli
from nirspipe.interface.commands import COMMANDS, parser_for

ROOT = Path(__file__).resolve().parents[2]
# a whole flag: one followed by a hyphen is a family (--wtc-*) or a prefix (--no-), not a name
FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*(?![\w-])")
# other programs' flags, named where the package calls them or the docs show their command
FOREIGN = {"--json", "--dataset", "--include", "--target-dir"}


def _defined() -> set[str]:
    def walk(parser):
        yield from (s for a in parser._actions for s in a.option_strings)
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for sub in action.choices.values():
                    yield from walk(sub)

    flags = set(walk(gui_cli._build_parser()))
    for prog in COMMANDS:
        flags |= set(walk(parser_for(prog)))
    return flags


def _texts() -> list[Path]:
    package = ROOT / "nirspipe"
    return [*package.rglob("*.py"), *package.rglob("*.toml"),
            *(ROOT / "docs").glob("*.md"), ROOT / "README.md"]


def _mentions(text: str) -> set[str]:
    return set(FLAG.findall(text))


def test_a_flag_family_and_a_prefix_are_not_read_as_flags():
    assert _mentions("every --wtc-* option, the --no- form, and --dry-run.") == {"--dry-run"}
    assert _mentions("`--high-pass 0.01` (--low-pass) x--y a--b") == {"--high-pass", "--low-pass"}


def test_the_scan_reads_the_package_and_the_docs():
    names = {path.name for path in _texts()}
    assert {"run.py", "pair_null.py", "usage.md", "README.md"} <= names


def test_every_flag_named_in_the_text_exists():
    defined = _defined() | FOREIGN
    stale = sorted(
        f"{path.relative_to(ROOT).as_posix()}:{number}: {flag}"
        for path in _texts()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for flag in _mentions(line) - defined)
    assert not stale, "flags no command defines:\n  " + "\n  ".join(stale)
