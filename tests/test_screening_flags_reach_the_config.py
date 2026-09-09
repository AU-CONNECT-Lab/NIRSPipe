"""The three screening flags have to arrive in PrepConfig, not merely parse.

`fnirs-pipe` builds its config out of the parsed args with a run of conditional
``**({...} if args.get(k) is not None else {})`` splats, and the same pattern appears in the
call that logs the run to the job database. A splat written into the wrong one of those two
calls parses fine, records fine, and never reaches the config: the flag becomes a silent
no-op when its default is None, and a `TypeError` on every run when it is not.

Both happened. `--psp-threshold` sat in the job-database call from the start and was a
no-op whenever it was passed; `--min-good-frac` was added beside it and inherited that; then
`--screen-scope`, whose default is a string rather than None, turned the latent version into
an unconditional crash on 2026-09-09.

`test_cli_dispatch_surface.py` cannot see this. That one compares a command's parsed dests
against its own signature, and both calls here are internal.
"""

import pytest

from fnirs_pipe.qc.metrics import GOOD_FRAC_PASS, PSP_PASS


def _config(argv_extra):
    """The PrepConfig `fnirs-pipe participant` would build for these flags."""
    from fnirs_pipe.cli import run as run_cli
    from fnirs_pipe.cli.workflows import _make_prep_config

    argv = ["bids", "out", "participant",
            "--participant-label", "01",
            "--dpf", "6.0",
            "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
            "--resp-l-freq", "0.2", "--resp-h-freq", "0.5",
            *argv_extra]
    args = vars(run_cli._build_parser().parse_args(argv))
    return _make_prep_config("01", None, args)


def test_the_defaults_arrive():
    config = _config([])
    assert config.psp_threshold == PSP_PASS
    assert config.min_good_frac == GOOD_FRAC_PASS
    assert config.screen_scope == "run"


@pytest.mark.parametrize("flag,value,field,expected", [
    ("--psp-threshold", "0.25", "psp_threshold", 0.25),
    ("--min-good-frac", "0.5", "min_good_frac", 0.5),
    ("--screen-scope", "task", "screen_scope", "task"),
])
def test_each_screening_flag_reaches_the_config(flag, value, field, expected):
    assert getattr(_config([flag, value]), field) == expected


def test_all_three_at_once():
    """Together, since the bug was one splat landing in the wrong call among several."""
    config = _config(["--psp-threshold", "0.2", "--min-good-frac", "0.6",
                      "--screen-scope", "task"])
    assert (config.psp_threshold, config.min_good_frac, config.screen_scope) == (
        0.2, 0.6, "task")
