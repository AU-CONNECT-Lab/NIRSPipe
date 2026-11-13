"""A flag is typed in full on every command, so a script keeps parsing when a flag sharing its
prefix is added later."""

import pytest

from nirspipe.cli import prep as prep_cli
from nirspipe.cli import run as run_cli

from tests.cli._parsers import PARSERS


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
