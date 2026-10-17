"""The console commands the web app can run, their parsers, and an argv as a pasteable command."""

from __future__ import annotations

import argparse
import importlib
import re
import shutil
import sys
from pathlib import Path

# prog -> (module, function, key): the parser is built on first use, since some modules are slow to import
COMMANDS: dict[str, tuple[str, str, "str | None"]] = {
    "nirspipe-recon":           ("nirspipe.cli.recon", "_build_parser", None),
    "nirspipe-prep":            ("nirspipe.cli.prep", "_build_parser", None),
    "nirspipe-qc":              ("nirspipe.cli.qc", "_build_parser", None),
    "nirspipe":                 ("nirspipe.cli.run", "_build_parser", None),
    "nirspipe-hyper":           ("nirspipe.cli.hyper", "_parsers", "nirspipe-hyper"),
    "nirspipe-hyper-pairnull":  ("nirspipe.cli.hyper", "_parsers", "nirspipe-hyper-pairnull"),
    "nirspipe-hyper-groupnull": ("nirspipe.cli.hyper", "_parsers", "nirspipe-hyper-groupnull"),
    "nirspipe-hyper-band":      ("nirspipe.cli.hyper", "_parsers", "nirspipe-hyper-band"),
    "nirspipe-hyper-index":     ("nirspipe.cli.hyper", "_parsers", "nirspipe-hyper-index"),
    "nirspipe-hyper-merge":     ("nirspipe.cli.hyper", "_parsers", "nirspipe-hyper-merge"),
    "nirspipe-rate":            ("nirspipe.cli.rate", "_build_parser", None),
    "nirspipe-log":             ("nirspipe.cli.db", "_build_parser", None),
}

# subcommands that open a page in this app rather than run as a process
VIEWERS = {("nirspipe-rate", "rate"), ("nirspipe-rate", "raw"), ("nirspipe-rate", "hyper")}


def parser_for(prog: str) -> argparse.ArgumentParser:
    module, function, key = COMMANDS[prog]
    built = getattr(importlib.import_module(module), function)()
    return built[key] if key else built


def subcommands(parser: argparse.ArgumentParser) -> "dict[str, argparse.ArgumentParser]":
    """``{name: parser}`` for a command with subcommands, ``{}`` for one without."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def executable(prog: str) -> str:
    """The console script beside this interpreter, so the app needs no PATH set up; else the bare name."""
    return shutil.which(prog, path=str(Path(sys.executable).parent)) or prog


_CONTINUATION = {"bash": "\\", "cmd": "^", "powershell": "`"}
# a subcommand or an analysis level; a token that is not one ends the first line
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
# what every shell takes unquoted; anything else, a path included, is quoted
_PLAIN = re.compile(r"[A-Za-z0-9_.,:=+@%-]+")


def _quote(token: str, shell: str) -> str:
    if _PLAIN.fullmatch(token):
        return token
    if shell == "cmd":
        # the C runtime's rule: backslashes before a quote, the closing one included, double
        escaped = re.sub(r'(\\*)"', r'\1\1\\"', token)
        return '"' + re.sub(r"(\\+)$", r"\1\1", escaped) + '"'
    # single quotes are literal in both, so a trailing backslash or a $ survives
    if shell == "powershell":
        return "'" + token.replace("'", "''") + "'"
    return "'" + token.replace("'", "'\\''") + "'"


def preview_text(argv: list[str], shell: str) -> str:
    """The argv as a pasteable command: each positional, and each flag with its values, on a line."""
    first, rest = [argv[0]], list(argv[1:])
    while rest and _WORD.fullmatch(rest[0]):
        first.append(rest.pop(0))
    lines = [" ".join(first)]
    for token in rest:
        quoted = _quote(token, shell)
        # a value follows its flag; a positional, or a flag, starts a line
        if token.startswith("--") or not lines[-1].lstrip().startswith("--"):
            lines.append(f"  {quoted}")
        else:
            lines[-1] += f" {quoted}"
    cont = _CONTINUATION.get(shell, "\\")
    return f" {cont}\n".join(lines)
