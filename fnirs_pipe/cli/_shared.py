"""Option blocks more than one ``fnirs-*`` command declares.

Each flag is declared here exactly once. ``add_*`` puts one on a parser or on an argument
group, for a command that files it under a heading of its own; the plain functions return a
parent parser to hand to ``add_parser(parents=[...])``.

They live here rather than in each CLI because two commands spelling one parameter two ways
is how the dyad coherence band ended up as ``--fmin`` in ``fnirs-qc`` and ``--wtc-fmin`` in
``fnirs-hyper``, and how the SCI line ended up written out as ``0.8`` in three places
instead of read from :data:`~fnirs_pipe.qc.metrics.SCI_PASS`.

Only genuinely shared parameters belong here. A flag that means something different to two
commands stays in each of them: the dyad coherence window is 30 s with a 5 s step and the
per-channel QC window is 10 s and does not overlap, so one ``--window-length`` covering both
would name two things.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_sci_threshold(container, default: "float | None" = None, note: str = "") -> None:
    """``--sci-threshold``: the coupling line channel screening rejects on.

    ``default`` is None for a command that requires the flag instead of defaulting it.
    """
    from fnirs_pipe.qc.metrics import SCI_PASS

    container.add_argument(
        "--sci-threshold", type=float, default=default,
        help="Scalp coupling index below which a channel is rejected"
             + (f" (default {SCI_PASS})." if default is not None else ", e.g. 0.8.")
             + (f" {note}" if note else ""))


def add_psp_threshold(container, note: str = "") -> None:
    """``--psp-threshold``: the other line, from the same criteria table."""
    from fnirs_pipe.qc.metrics import PSP_PASS

    container.add_argument(
        "--psp-threshold", type=float, default=None,
        help=f"Peak spectral power below which a channel is rejected (default {PSP_PASS}). "
             f"Screening is a union, so a channel failing either line goes. PSP catches the "
             f"movement that fakes a high SCI, so raising it prunes more than "
             f"--sci-threshold alone does."
             + (f" {note}" if note else ""))


def screening(sci_default: "float | None" = None, note: str = "") -> argparse.ArgumentParser:
    """Both screening lines as a parent parser.

    One block for both, because screening is a union over the criteria table and a command
    that can move one line has no reason not to move the other.
    """
    p = argparse.ArgumentParser(add_help=False)
    add_sci_threshold(p, sci_default, note)
    add_psp_threshold(p, note)
    return p


def pairs_selection() -> argparse.ArgumentParser:
    """``--pairs-csv`` / ``--group-id`` / ``--task-label``: which dyads to process."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pairs-csv", type=Path, required=True,
                   help="CSV with columns: group_id, subject_id, task. Each unique "
                        "(group_id, task) pair is processed as one session.")
    p.add_argument("--group-id", default=None,
                   help="Process only this group_id. Omit to process all groups.")
    p.add_argument("--task-label", nargs="+", action="extend",
                   help="Task label(s) to include, filtering the pairs table.")
    return p


def alignment_window() -> argparse.ArgumentParser:
    """``--normalize`` / ``--no-align`` / ``--tstart`` / ``--tend``: what the metrics see."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                   help="Z-score each channel per subject after alignment.")
    p.add_argument("--no-align", action="store_true",
                   help="Skip trigger-based alignment; trim all recordings to the shortest "
                        "duration.")
    p.add_argument("--tstart", type=float, default=None,
                   help="Keep only from this time (s) on the aligned clock, where 0 is the "
                        "shared trigger. Omit to start at the alignment point.")
    p.add_argument("--tend", type=float, default=None,
                   help="Keep only up to this time (s) on the aligned clock. Omit to run to "
                        "the end; a value past the end is clipped. The window narrows the "
                        "synchrony metrics only: the per-subject quality record describes "
                        "the whole recording either way.")
    return p
