"""A command's argparse parser as form fields, and a submitted form back into its argv."""

from __future__ import annotations

import argparse
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

# what --help itself leaves out of a form, and the subcommand switch, which has a page of its own
_SKIPPED = (argparse._HelpAction, argparse._VersionAction, argparse._SubParsersAction)


@dataclass
class Field:
    name: str                   # the long flag, or the dest of a positional
    kind: str                   # switch, boolean, choice, choices, list, number, text
    help: str
    required: bool
    choices: list[str]
    positional: bool
    action: argparse.Action

    @property
    def negative(self) -> str:
        """The ``--no-`` spelling of a boolean flag."""
        return next((s for s in self.action.option_strings if s.startswith("--no-")), "")

    @property
    def per_line(self) -> bool:
        """A list of paths, which may hold spaces, so one value per line."""
        return self.action.type is Path

    @property
    def default(self) -> str:
        """The command's default as text, or "" when it has none worth showing."""
        value = self.action.default
        if value in (None, False, argparse.SUPPRESS) or value == []:
            return ""
        return " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)


def _kind(action: argparse.Action) -> str:
    if isinstance(action, argparse._StoreTrueAction):
        return "switch"
    if isinstance(action, argparse.BooleanOptionalAction):
        return "boolean"
    many = action.nargs in ("+", "*") or isinstance(action, argparse._ExtendAction)
    if action.choices:
        return "choices" if many else "choice"
    if many:
        return "list"
    if action.type in (int, float):
        return "number"
    return "text"


def _name(action: argparse.Action) -> str:
    if not action.option_strings:
        return action.dest
    return next((s for s in action.option_strings if s.startswith("--")), action.option_strings[0])


def fields(parser: argparse.ArgumentParser) -> "list[tuple[str, list[Field]]]":
    """The parser's fields in its own groups and order, as ``[(group title, [Field, ...]), ...]``."""
    out = []
    for group in parser._action_groups:
        shown = [Field(name=_name(a), kind=_kind(a), help=a.help or "",
                       required=bool(a.required), choices=[str(c) for c in a.choices or []],
                       positional=not a.option_strings, action=a)
                 for a in group._group_actions
                 if not isinstance(a, _SKIPPED) and a.help != argparse.SUPPRESS]
        if shown:
            out.append((group.title, shown))
    return out


def _items(field: Field, values: list[str]) -> list[str]:
    """A list box's values: one path per line, since a path may hold spaces; anything else split on whitespace too."""
    if field.kind == "choices":
        return values
    if field.per_line:
        return [line.strip() for value in values for line in value.splitlines() if line.strip()]
    return [item for value in values for item in value.split()]


def _tokens(field: Field, values: list[str]) -> list[str]:
    flag = [] if field.positional else [field.name]
    if field.kind == "switch":
        return flag if values else []
    if field.kind == "boolean":
        if not values:
            return []
        return [field.name] if values[0] == "yes" else [field.negative]
    if field.kind in ("list", "choices"):
        items = _items(field, values)
        return flag + items if items else []
    return flag + values[:1] if values else []


def to_argv(prefix: list[str], parser: argparse.ArgumentParser, form) -> list[str]:
    """``prefix`` (the command and any subcommand) and the form's values as an argv.

    ``form`` is a mapping with ``getlist``, as Flask's ``request.form`` is. An empty field is
    left out, so the command's own default applies.
    """
    positionals, options = [], []
    for _title, group in fields(parser):
        for field in group:
            values = [v.strip() for v in form.getlist(field.name) if v.strip()]
            (positionals if field.positional else options).extend(_tokens(field, values))
    return [*prefix, *positionals, *options]


def form_values(parser: argparse.ArgumentParser, settings: dict) -> "dict[str, list[str]]":
    """Parsed settings, ``{dest: value}`` as a run record holds them, as the form's values."""
    values = {}
    for _title, group in fields(parser):
        for field in group:
            given = settings.get(field.action.dest)
            if given is None or (field.kind == "switch" and not given):
                continue
            if field.kind == "switch":
                values[field.name] = ["on"]
            elif field.kind == "boolean":
                values[field.name] = ["yes" if given else "no"]
            elif field.kind == "choices":
                values[field.name] = [str(v) for v in given]
            elif field.kind == "list":
                values[field.name] = [("\n" if field.per_line else " ").join(map(str, given))]
            else:
                values[field.name] = [str(given)]
    return values


def check(parser: argparse.ArgumentParser, args: list[str]) -> "str | None":
    """argparse's own complaint about ``args``, or None when the command line parses."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parser.parse_args(args)
    except SystemExit:
        lines = err.getvalue().strip().splitlines()
        return lines[-1] if lines else "the command line does not parse"
    return None
