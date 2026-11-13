"""Flags and guards every command shares; the BIDS App's option blocks are in ``_app_options``."""

from __future__ import annotations

import argparse
import math
import secrets
import sys
from pathlib import Path
from typing import NoReturn


VERBOSE_HELP = ("Print debug messages on the console too, each step's timing and paths among them.")


def fail(message: str) -> NoReturn:
    """Print one refusal to stderr and exit non-zero."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def refuse_inverted(low, high, low_flag: str, high_flag: str) -> None:
    """Exit non-zero when both edges are numbers and the lower is not below the upper."""
    if (isinstance(low, (int, float)) and isinstance(high, (int, float))
            and not low < high):
        fail(f"{low_flag} {low:g} has to be below {high_flag} {high:g}.")


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
    """``--skip-bids-validation`` for a command that reads a BIDS dataset."""
    container.add_argument(
        "--skip-bids-validation", "--skip_bids_validation", "--skip_bids_validator",
        action="store_true", help="Do not check the input with bids-validator.")


def _bounded(low: float, high: float, said: str, above: bool = False):
    """An argparse type for a finite number from ``low`` to ``high``, or above ``low`` alone."""
    def number(text: str) -> float:
        value = float(text)
        inside = (low < value if above else low <= value) and value <= high
        if not (math.isfinite(value) and inside):
            raise argparse.ArgumentTypeError(f"must be {said}, got {text}")
        return value
    return number


# what a flag's value can be by definition, checked as it is parsed, from a --config file too
positive = _bounded(0.0, math.inf, "greater than 0", above=True)
non_negative = _bounded(0.0, math.inf, "0 or more")
share = _bounded(0.0, 1.0, "between 0 and 1")
correlation = _bounded(-1.0, 1.0, "between -1 and 1")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, got {value}")
    return value


def seconds_or_auto(text: str) -> "float | str":
    """A length in seconds, or ``auto`` where the command works one out."""
    return "auto" if text.strip().lower() == "auto" else non_negative(text)


# the types above that take a plain number, which a form shows as a number field
NUMBER_TYPES = (positive, non_negative, share, correlation, positive_int, non_negative_int)


def add_nprocs(container, unit: str = "subject", note: str = "") -> None:
    """``--nprocs``, the cores a command may use in all, spelled as the BIDS Apps interface spells it."""
    container.add_argument(
        "--nprocs", "--n_cpus", "--n-cpus", dest="nprocs", type=positive_int, default=None,
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
    # a table is the NAME=VALUE form typed out, as --stim-dur Tapping=20 Control=15
    if many and isinstance(value, dict):
        value = [f"{name}={item}" for name, item in value.items()]
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


def read_input_copies(raw: dict, keys: "dict[str, tuple[str, ...]]", actions: dict,
                      path: Path, out: dict) -> dict[str, str]:
    """Point each setting a record's [inputs] copied at its copy, in ``out``; returns each one's path as first given."""
    given: dict[str, str] = {}
    for key, entry in (raw.get("inputs") or {}).items():
        section = next((s for s, allowed in keys.items() if key in allowed), None)
        if section is None or not isinstance(entry, dict) or "copy" not in entry:
            continue
        copy = Path(path).parent / entry["copy"]
        # an --events-path template's copies are listed one per run under sha256
        names = entry.get("sha256")
        files = [copy.parent / n for n in names] if isinstance(names, dict) else [copy]
        if not all(f.is_file() for f in files):
            print(f"Warning: {path}: the copy of {key} ({entry['copy']}) is not beside it, so "
                  "the path its settings name is read instead.", file=sys.stderr)
            continue
        out[section][key] = coerce_setting(actions[key], key, copy.as_posix(), path)
        given[key] = entry.get("given")
    return given


def typed_flags(parser: argparse.ArgumentParser, argv: "list[str] | None") -> set[str]:
    """The dests of the flags actually typed, which argparse's own defaults would otherwise hide."""
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return set(vars(parser.parse_args(argv)))
