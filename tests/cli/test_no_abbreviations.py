"""A flag is typed in full on every command, so a script keeps parsing when a flag sharing its
prefix is added later."""

import argparse

import pytest

from nirspipe.cli import gui as gui_cli
from nirspipe.cli import prep as prep_cli
from nirspipe.cli import run as run_cli
from nirspipe.interface.commands import COMMANDS, parser_for


def _walk(name, parser):
    """A parser and every subcommand parser under it, the group commands in between included."""
    yield name, parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub_name, sub in action.choices.items():
                yield from _walk(f"{name} {sub_name}", sub)


PARSERS = [found for prog in COMMANDS for found in _walk(prog, parser_for(prog))]
PARSERS += [("nirspipe-gui", gui_cli._build_parser())]


def test_the_walk_reaches_the_subcommands():
    names = {name for name, _parser in PARSERS}
    assert {"nirspipe", "nirspipe-prep crop", "nirspipe-prep edit-markers",
            "nirspipe-prep edit-markers apply", "nirspipe-rate export",
            "nirspipe-hyper-pairnull", "nirspipe-gui"} <= names


@pytest.mark.parametrize("name,parser", PARSERS, ids=[name for name, _parser in PARSERS])
def test_no_command_completes_a_prefix(name, parser):
    assert parser.allow_abbrev is False


def test_an_abbreviated_flag_stops_the_pipeline_command(capsys):
    with pytest.raises(SystemExit) as exit_:
        run_cli.main(["bids", "out", "participant", "--pop", "adult", "--sci-threshold", "0.8"])
    assert exit_.value.code == 2
    assert "unrecognized arguments: --pop adult" in capsys.readouterr().err


def test_an_abbreviated_flag_stops_a_subcommand(capsys):
    with pytest.raises(SystemExit) as exit_:
        prep_cli.main(["crop", "bids", "out", "--participant-label", "01", "--tmi", "0"])
    assert exit_.value.code == 2
    assert "unrecognized arguments: --tmi 0" in capsys.readouterr().err
