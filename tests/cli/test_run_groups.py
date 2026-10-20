"""A group that fails for a reason written for the user reads as that reason, not a crash."""

import ast
import logging
import os
import re

import pytest

from nirspipe.cli.hyper import _run_groups
from nirspipe.exceptions import AlignmentError, StageError


# ---- Workers for --nprocs. Module level, so a spawned process can unpickle them ----

def _report_pid_and_blas(gid, task, ses, members):
    import numpy  # noqa: F401  loads the BLAS threadpoolctl reports on
    from threadpoolctl import threadpool_info

    logging.getLogger("nirspipe.test").info("working on %s", gid)
    if gid == "G02":
        raise StageError("planted failure")
    threads = sorted({p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"})
    return f"pid={os.getpid()} blas={threads}"


def _drawn(out: str) -> dict[str, tuple[int, list[int]]]:
    """``{group: (pid, BLAS thread counts)}`` from the report lines of a run."""
    found = {}
    for gid, pid, blas in re.findall(r"(G\d+)/main \(0 subjects\)[^\n]*pid=(\d+) blas=(\[[^\]]*\])",
                                     out):
        found[gid] = (int(pid), ast.literal_eval(blas))
    return found


def test_a_stage_error_is_reported_as_its_message(capsys, caplog):
    def process(gid, task, ses, members):
        raise StageError("desc-od holds optical density, not haemoglobin concentration")

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        _run_groups({("G01", "main", None): []}, process)

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "[error] desc-od holds optical density" in err
    assert "unexpected" not in err
    assert not [r for r in caplog.records if r.exc_info]


def test_the_other_groups_still_run(capsys):
    seen = []

    def process(gid, task, ses, members):
        seen.append(gid)
        if gid == "G01":
            raise StageError("no real WTC table to rank against")

    with pytest.raises(SystemExit):
        _run_groups({("G01", "main", None): [], ("G02", "main", None): []}, process)
    assert seen == ["G01", "G02"]
    assert "1 succeeded, 1 failed" in capsys.readouterr().out


def test_a_failed_alignment_names_no_align_and_what_it_assumes(capsys):
    def process(gid, task, ses, members):
        raise AlignmentError("No non-BAD annotations found in any recording.")

    with pytest.raises(SystemExit):
        _run_groups({("G01", "rest", None): []}, process)
    err = capsys.readouterr().err
    assert "No non-BAD annotations" in err
    assert "--no-align" in err and "simultaneous starts" in err


# ---- --nprocs ----

GROUPS = {("G01", "main", None): [], ("G02", "main", None): [], ("G03", "main", None): []}


def test_with_nprocs_groups_run_in_processes_of_their_own_and_share_the_cores(capsys):
    with pytest.raises(SystemExit):
        _run_groups(GROUPS, _report_pid_and_blas, nprocs=4)
    captured = capsys.readouterr()
    drawn = _drawn(captured.out)
    # three groups on four cores: three at once, one BLAS thread each
    assert sorted(drawn) == ["G01", "G03"]
    assert all(pid != os.getpid() and blas == [1] for pid, blas in drawn.values())
    assert "3 at once" in captured.out
    assert "2 succeeded, 1 failed" in captured.out
    assert "G02/main (0 subjects): [error] planted failure" in captured.err


def test_without_nprocs_groups_run_here_on_every_core(capsys):
    with pytest.raises(SystemExit):
        _run_groups(GROUPS, _report_pid_and_blas)
    drawn = _drawn(capsys.readouterr().out.replace("\n     report ->", " report ->"))
    assert {pid for pid, _ in drawn.values()} == {os.getpid()}


def test_one_core_holds_blas_to_one_thread_in_this_process(capsys):
    with pytest.raises(SystemExit):
        _run_groups(GROUPS, _report_pid_and_blas, nprocs=1)
    drawn = _drawn(capsys.readouterr().out.replace("\n     report ->", " report ->"))
    assert {pid for pid, _ in drawn.values()} == {os.getpid()}
    assert all(blas == [1] for _, blas in drawn.values())


def test_each_group_logs_to_its_own_file_on_either_path(tmp_path):
    for nprocs in (None, 3):
        logs = tmp_path / str(nprocs)
        with pytest.raises(SystemExit):
            _run_groups(GROUPS, _report_pid_and_blas, nprocs=nprocs,
                        log_file=lambda key: logs / f"{key[0]}.log")
        for gid in ("G01", "G02", "G03"):
            text = (logs / f"{gid}.log").read_text(encoding="utf-8")
            assert f"working on {gid}" in text
            assert all(f"working on {other}" not in text
                       for other in ("G01", "G02", "G03") if other != gid)
