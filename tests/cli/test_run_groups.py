"""A group that fails for a reason written for the user reads as that reason, not a crash."""

import logging

import pytest

from fnirs_pipe.cli.hyper import _run_groups
from fnirs_pipe.exceptions import StageError


def test_a_stage_error_is_reported_as_its_message(capsys, caplog):
    def process(gid, task, members):
        raise StageError("desc-od holds optical density, not haemoglobin concentration")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        _run_groups({("G01", "main"): []}, process)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "[error] desc-od holds optical density" in err
    assert "unexpected" not in err
    assert not [r for r in caplog.records if r.exc_info]


def test_the_other_groups_still_run(capsys):
    seen = []

    def process(gid, task, members):
        seen.append(gid)
        if gid == "G01":
            raise StageError("no real WTC table to rank against")

    with pytest.raises(SystemExit):
        _run_groups({("G01", "main"): [], ("G02", "main"): []}, process)
    assert seen == ["G01", "G02"]
    assert "1 succeeded, 1 failed" in capsys.readouterr().out
