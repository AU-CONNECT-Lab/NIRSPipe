"""Flags and guards every command shares; the BIDS App's option blocks are in ``_app_options``."""

from __future__ import annotations

import argparse
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


def add_n_jobs(container) -> None:
    """``--n-jobs``, parallel subjects; --nprocs / --n_cpus are what the BIDS Apps interface calls it."""
    container.add_argument("--n-jobs", "--n_jobs", "--nprocs", "--n_cpus", dest="n_jobs",
                           type=int, default=1, help="Parallel subject jobs.")
