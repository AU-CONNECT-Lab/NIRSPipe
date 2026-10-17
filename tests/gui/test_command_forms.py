"""Forms generated from the commands' own parsers cover every flag and round-trip through them."""

import argparse
from pathlib import Path

import pytest
from werkzeug.datastructures import MultiDict

from nirspipe.cli._shared import BidsLabel
from nirspipe.interface.commands import COMMANDS, parser_for, subcommands
from nirspipe.interface.forms import _SKIPPED, check, fields, required_first, to_argv


def _nested(prefix, parser, top):
    subs = subcommands(parser)
    if not subs:
        yield " ".join(prefix), prefix, parser, top
    for sub, child in subs.items():
        yield from _nested([*prefix, sub], child, top)


def _forms():
    for prog in COMMANDS:
        top = parser_for(prog)
        yield from _nested([prog], top, top)


FORMS = list(_forms())


def _sample(field):
    """(form value, what the parser should hand back), or None for a type with no safe sample."""
    action = field.action
    if field.kind == "switch":
        return "on", True
    if field.kind == "boolean":
        return "no", False
    if field.kind == "choice":
        return field.choices[-1], action.type(field.choices[-1]) if action.type else field.choices[-1]
    if field.kind == "choices":
        return field.choices[:2], field.choices[:2]
    if field.kind == "list":
        if field.per_line:
            return "C:/a b/c\nD:/e", [Path("C:/a b/c"), Path("D:/e")]
        if action.type is float:
            return "0.5 2", [0.5, 2.0]
        if action.type in (BidsLabel, None):
            return "sub-01 02", ["01", "02"] if action.type else ["sub-01", "02"]
        return None
    if field.kind == "number":
        return ("3", 3) if action.type is int else ("0.25", 0.25)
    if action.type is Path:
        return "C:/a b/c", Path("C:/a b/c")
    if action.type is BidsLabel:
        return "sub-07", "07"
    if action.type in (None, str):
        return "abc", "abc"
    return None


def _form(values: dict) -> MultiDict:
    form = MultiDict()
    for name, value in values.items():
        for item in value if isinstance(value, list) else [value]:
            form.add(name, item)
    return form


@pytest.mark.parametrize("label, prefix, parser, top", FORMS, ids=[f[0] for f in FORMS])
def test_every_flag_has_a_field(label, prefix, parser, top):
    shown = {field.action for _title, group in fields(parser) for field in group}
    expected = {a for a in parser._actions
                if not isinstance(a, _SKIPPED) and a.help != argparse.SUPPRESS}
    assert shown == expected


@pytest.mark.parametrize("label, prefix, parser, top", FORMS, ids=[f[0] for f in FORMS])
def test_each_field_round_trips_through_the_command_s_own_parser(label, prefix, parser, top):
    every = [f for _title, group in fields(parser) for f in group]
    required = {f.name: _sample(f) for f in every if f.required}
    if any(sample is None for sample in required.values()):
        pytest.skip("a required flag has no safe sample value")
    base = {name: sample[0] for name, sample in required.items()}
    for field in every:
        sample = _sample(field)
        if sample is None:
            continue
        form = _form({**base, field.name: sample[0]})
        argv = to_argv(prefix, parser, form)
        assert check(top, argv[1:]) is None, argv
        assert getattr(top.parse_args(argv[1:]), field.action.dest) == sample[1], field.name


def test_an_empty_field_leaves_the_command_s_default():
    parser = parser_for("nirspipe-qc")
    prep_raw = subcommands(parser)["prep-raw"]
    argv = to_argv(["nirspipe-qc", "prep-raw"], prep_raw,
                   _form({"bids_dir": "/bids", "output_dir": "/out", "--psp-threshold": ""}))
    assert "--psp-threshold" not in argv


def test_a_command_line_that_does_not_parse_is_named_by_argparse():
    parser = parser_for("nirspipe-qc")
    message = check(parser, ["prep-raw", "/bids", "/out", "--sci-threshold", "high"])
    assert message is not None and "--sci-threshold" in message


def test_required_fields_lead_and_none_is_lost():
    groups = fields(subcommands(parser_for("nirspipe-qc"))["prep-raw"])
    ordered = required_first(groups)
    assert ordered[0][0] == "Required"
    assert all(f.required for f in ordered[0][1])
    assert not any(f.required for _title, group in ordered[1:] for f in group)
    every = [f.name for _title, group in groups for f in group]
    assert sorted(f.name for _title, group in ordered for f in group) == sorted(every)
