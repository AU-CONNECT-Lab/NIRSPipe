"""``--nprocs`` is the cores a command may use in all, split between subjects at once and BLAS threads."""

from __future__ import annotations

import threading

import numpy  # noqa: F401  loads the BLAS the limits act on
import pytest
from threadpoolctl import threadpool_info

from nirspipe.cli import _shared
from nirspipe.cli import prep as prep_cli
from nirspipe.cli import run as run_cli

_BANDS = ["--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.2", "--resp-h-freq", "0.5"]
_MIN = ["bids", "out", "participant", "--dpf", "6.0", "--sci-threshold", "0.8", *_BANDS]


def _blas_threads() -> int:
    blas = [i["num_threads"] for i in threadpool_info() if i["user_api"] == "blas"]
    if not blas:
        pytest.skip("no BLAS library loaded")
    return blas[0]


@pytest.mark.parametrize("nprocs, n_subjects, expected", [
    (None, 5, (1, None)),
    (8, 3, (3, 2)),
    (7, 2, (2, 3)),
    (4, 10, (4, 1)),
    (1, 4, (1, 1)),
    (6, 1, (1, 6)),
])
def test_split_cpus(nprocs, n_subjects, expected):
    assert _shared.split_cpus(nprocs, n_subjects) == expected


@pytest.mark.parametrize("flag", ["--nprocs", "--n_cpus", "--n-cpus"])
def test_every_spelling_parses(flag):
    assert run_cli._build_parser().parse_args(_MIN + [flag, "3"]).nprocs == 3


def test_prep_takes_it():
    args = prep_cli._build_parser().parse_args(
        ["crop", "/b", "/d", "--participant-label", "01", "--tmin", "0", "--nprocs", "2"])
    assert args.nprocs == 2


@pytest.mark.parametrize("value", ["0", "-2"])
def test_below_one_is_refused(value):
    with pytest.raises(SystemExit):
        run_cli._build_parser().parse_args(_MIN + ["--nprocs", value])


def test_subjects_run_at_once_with_their_share_of_blas():
    # neither thread passes the barrier unless the other is running at the same time
    barrier = threading.Barrier(2, timeout=30)

    def fn(subject):
        barrier.wait()
        return subject, _blas_threads()

    assert _shared.map_subjects(fn, ["01", "02"], nprocs=6) == [("01", 3), ("02", 3)]


def test_one_subject_is_held_to_nprocs():
    assert _shared.map_subjects(lambda s: _blas_threads(), ["01"], nprocs=2) == [2]


def test_unset_leaves_blas_alone():
    before = _blas_threads()
    assert _shared.map_subjects(lambda s: _blas_threads(), ["01", "02"], nprocs=None) == [before] * 2
    assert _blas_threads() == before


def test_order_follows_the_subjects():
    assert _shared.map_subjects(str.upper, ["c", "a", "b"], nprocs=3) == ["C", "A", "B"]
