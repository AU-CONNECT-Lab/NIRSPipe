"""Flags and guards every command shares; the BIDS App's option blocks are in ``_app_options``."""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path


class BidsLabel(str):
    """A BIDS entity label with its ``sub-``/``ses-``/``task-`` prefix taken off.

    Accepts ``--participant-label sub-01`` as well as the bare label, and strips it here so
    every command treats the two alike.

    Used as an argparse ``type``, so it strips once, at parse time::

        BidsLabel("sub-01") == "01"
    """

    _PREFIXES = ("sub-", "ses-", "task-", "run-", "group-")

    def __new__(cls, value: str):
        text = str(value).strip()
        for prefix in cls._PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return super().__new__(cls, text)


def refuse_output_in_input(bids_dir: Path, output_dir: Path, tool: str,
                           writes: str | None = None) -> None:
    """Exit non-zero when the output directory, or its ``writes`` folder, is the input."""
    bids_dir = Path(bids_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == bids_dir:
        print(f"Error: the output directory is the input BIDS directory; choose another, "
              f"e.g. {bids_dir / 'derivatives' / tool}.", file=sys.stderr)
        raise SystemExit(1)
    if writes and output_dir / writes == bids_dir:
        print(f"Error: this command writes into {output_dir / writes}, which is the input "
              f"directory; choose another output directory.", file=sys.stderr)
        raise SystemExit(1)


def refuse_output_is_source(derivatives_dir: Path, output_dir: Path) -> None:
    """Exit non-zero when the nirspipe-hyper tree would be the nirspipe tree it reads."""
    if Path(output_dir).resolve() == Path(derivatives_dir).resolve():
        print("Error: the source and the output directory are the same. Each tree carries "
              "its own dataset_description.json, so one path cannot be both.",
              file=sys.stderr)
        raise SystemExit(1)


def add_skip_bids_validation(container) -> None:
    """``--skip-bids-validation`` for a command that reads a BIDS dataset, with a ``--no-`` form."""
    container.add_argument(
        "--skip-bids-validation", "--skip_bids_validation",
        "--skip_bids_validator", dest="skip_bids_validation",
        action=argparse.BooleanOptionalAction, default=False,
        help="Do not check the input with bids-validator.")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def add_nprocs(container, unit: str = "subject", note: str = "") -> None:
    """``--nprocs``, the cores a command may use in all, spelled as the BIDS Apps interface spells it."""
    container.add_argument(
        "--nprocs", "--n_cpus", "--n-cpus", dest="nprocs", type=_positive_int, default=None,
        help=f"CPU cores to use in all. Up to this many {unit}s run at once and spare cores go "
             f"to each {unit}'s numerics. Omit to run one {unit} at a time on every core."
             + (f" {note}" if note else ""))


def seed_or_drawn(seed: int | None, flag: str) -> int:
    """The seed given, or one drawn now and printed, so every run's draws can be repeated."""
    if seed is not None:
        return seed
    seed = secrets.randbelow(2**31)
    print(f"[info] no {flag} given; drew {seed}. Pass {flag} {seed} to repeat this run's "
          "random draws.", file=sys.stderr)
    return seed


def split_cpus(nprocs: int | None, n_subjects: int) -> tuple[int, int | None]:
    """(subjects at once, BLAS threads each) within ``nprocs`` cores; None leaves BLAS unlimited.

    Example: ``split_cpus(8, 3) == (3, 2)``, ``split_cpus(4, 10) == (4, 1)``.
    """
    if nprocs is None:
        return 1, None
    jobs = max(1, min(nprocs, n_subjects))
    return jobs, max(1, nprocs // jobs)


def map_subjects(fn, subjects: list[str], nprocs: int | None) -> list:
    """``fn(subject)`` for every subject on threads within ``nprocs`` cores, in subject order."""
    from contextlib import nullcontext

    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    jobs, blas = split_cpus(nprocs, len(subjects))
    # threads, not processes, so a caller's shared lists and logs stay shared
    with threadpool_limits(limits=blas) if blas else nullcontext():
        if jobs == 1:
            return [fn(subject) for subject in subjects]
        return Parallel(n_jobs=jobs, prefer="threads")(delayed(fn)(s) for s in subjects)


def coerce_setting(action: argparse.Action, key: str, value, path: Path):
    """A TOML value through the flag's own type and choices, as argparse would have."""
    def fail(why: str):
        print(f"Error: {path}: {key} = {value!r}: {why}", file=sys.stderr)
        raise SystemExit(1)

    if action.nargs == 0:
        if not isinstance(value, bool):
            fail("expected true or false")
        return value
    many = action.nargs in ("+", "*")
    items = value if isinstance(value, list) else [value]
    if isinstance(value, list) and not many:
        fail("expected one value, not a list")
    try:
        items = [action.type(str(item)) if action.type else item for item in items]
    except (argparse.ArgumentTypeError, TypeError, ValueError) as exc:
        fail(str(exc))
    if action.choices is not None and any(item not in action.choices for item in items):
        fail(f"expected one of {', '.join(map(str, action.choices))}")
    return items if many else items[0]


def typed_flags(parser: argparse.ArgumentParser, argv: "list[str] | None") -> set[str]:
    """The dests of the flags actually typed, which argparse's own defaults would otherwise hide."""
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return set(vars(parser.parse_args(argv)))
