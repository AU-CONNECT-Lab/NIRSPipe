"""A --config file is keyed by flag name and a run record by dest, so every dest is its flag's name."""

import argparse

import pytest

from tests.cli._parsers import PARSERS

# spellings a BIDS App runner passes, as fMRIPrep and MRIQC take them
RUNNER_ALIASES = {"--n_cpus", "--n-cpus", "--skip_bids_validator"}


def _options(parser):
    return [a for a in parser._actions if a.option_strings
            and not isinstance(a, (argparse._HelpAction, argparse._VersionAction))]


def _allowed_alias(primary, alias, action):
    if isinstance(action, argparse.BooleanOptionalAction) and alias.startswith("--no-"):
        alias = "--" + alias.removeprefix("--no-")
        if alias == primary:
            return True
    return alias == "--" + primary[2:].replace("-", "_") or alias in RUNNER_ALIASES


@pytest.mark.parametrize("name,parser", PARSERS, ids=[name for name, _parser in PARSERS])
def test_every_dest_is_its_flag_name(name, parser):
    wrong = {a.option_strings[0]: a.dest for a in _options(parser)
             if a.dest != a.option_strings[0][2:].replace("-", "_")}
    assert not wrong


@pytest.mark.parametrize("name,parser", PARSERS, ids=[name for name, _parser in PARSERS])
def test_a_second_spelling_is_the_underscore_one_or_a_runner_s(name, parser):
    other = {a.option_strings[0]: alias for a in _options(parser) for alias in a.option_strings[1:]
             if not _allowed_alias(a.option_strings[0], alias, a)}
    assert not other


def test_the_rule_catches_a_renamed_dest_and_a_second_name():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-cross", dest="wtc_channel_cross")
    parser.add_argument("--by-condition", "--wtc-by-condition", action=argparse.BooleanOptionalAction)
    (cross, by) = _options(parser)
    assert cross.dest != cross.option_strings[0][2:].replace("-", "_")
    assert not _allowed_alias("--by-condition", "--wtc-by-condition", by)
    assert _allowed_alias("--by-condition", "--no-by-condition", by)
