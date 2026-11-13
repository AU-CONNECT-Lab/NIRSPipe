"""Every command's parser, and every subcommand parser under it."""

import argparse

from nirspipe.cli import gui as gui_cli
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
