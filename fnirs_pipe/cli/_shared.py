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


def add_separation_bands(container, note: str = "") -> None:
    """``--short-max-dist`` / ``--long-min-dist`` / ``--long-max-dist``, in mm.

    Three flags rather than one taking three values, because the upper bound defaults to
    off and a single flag has no natural way to spell that. They are validated together in
    :func:`separation_bands_from_args`, so moving one and leaving the others cannot produce
    overlapping bands.
    """
    from fnirs_pipe.qc.metrics._helpers import LONG_MIN_DIST, SHORT_MAX_DIST

    container.add_argument(
        "--short-max-dist", type=float, default=None, metavar="MM",
        help=f"Separation at or below which a channel is short-distance, in mm "
             f"(default {SHORT_MAX_DIST * 1e3:.0f}). Short channels see scalp only and are "
             f"measured, and regressed, separately from the long ones."
             + (f" {note}" if note else ""))
    container.add_argument(
        "--long-min-dist", type=float, default=None, metavar="MM",
        help=f"Separation at or above which a channel is long, in mm (default "
             f"{LONG_MIN_DIST * 1e3:.0f}). The gap above --short-max-dist is deliberate: a "
             f"channel in it is too far to be scalp-only and too near to reach cortex, and "
             f"screening cannot catch that because such a channel scores well."
             + (f" {note}" if note else ""))
    container.add_argument(
        "--long-max-dist", type=float, default=None, metavar="MM",
        help="Separation above which a channel is too far to be long, in mm. Off by "
             "default, so any separation past --long-min-dist counts as long. Set it on a "
             "montage carrying pairs too far apart to trust, which SCI and PSP catch only "
             "most of the time."
             + (f" {note}" if note else ""))


def separation_bands_from_args(args) -> dict:
    """The three flags as PrepConfig / PostConfig fields, in metres, validated together.

    args with --long-max-dist 55 -> {"short_max_dist": None, "long_min_dist": None,
                                     "long_max_dist": 0.055}

    Values arrive in mm because that is how a montage is described, and are stored in
    metres because that is what MNE reports. A flag left off stays None so the config
    keeps the package default; validation therefore runs on the resolved bands rather
    than on what was typed.
    """
    from fnirs_pipe.qc.metrics._helpers import separation_bands, validate_bands

    def _mm(name):
        value = getattr(args, name, None) if not isinstance(args, dict) else args.get(name)
        return None if value is None else float(value) / 1e3

    fields = {name: _mm(name)
              for name in ("short_max_dist", "long_min_dist", "long_max_dist")}
    # a class rather than the dict, since separation_bands reads attributes
    validate_bands(separation_bands(type("Args", (), fields)))
    return fields


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
