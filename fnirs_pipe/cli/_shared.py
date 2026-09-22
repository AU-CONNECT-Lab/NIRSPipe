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


class BidsLabel(str):
    """A BIDS entity label with its ``sub-``/``ses-``/``task-`` prefix taken off.

    The BIDS Apps interface asks for bare labels, but the folder on disk is what a user
    reads and types, so ``--participant-label sub-01`` arrives often enough to be worth
    accepting. Stripping it here rather than in each command is what keeps one CLI tolerant
    and the next one silently finding no files.

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


# One tuple so the flag names, the config keys and the PrepConfig / PostConfig fields cannot
# drift apart. They are spelled the same in all three places on purpose.
SEPARATION_BAND_KEYS = ("short_max_dist", "long_min_dist", "long_max_dist")


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

    fields = {name: _mm(name) for name in SEPARATION_BAND_KEYS}
    # a class rather than the dict, since separation_bands reads attributes
    validate_bands(separation_bands(type("Args", (), fields)))
    return fields


def resolved_separation_bands(args):
    """The same three flags as the resolved ``Bands`` the metrics take, defaults filled in.

    For a caller that measures channels rather than building a config: the quality record
    has to be split on the bands the run was processed with, and reaching for
    :func:`separation_bands` directly would give it the package defaults.
    """
    from fnirs_pipe.qc.metrics._helpers import separation_bands

    return separation_bands(type("Args", (), separation_bands_from_args(args)))


def add_sci_threshold(container, default: "float | None" = None, note: str = "") -> None:
    """``--sci-threshold``: the coupling line channel screening rejects on.

    ``default`` is None for a command that requires the flag instead of defaulting it.
    """
    from fnirs_pipe.qc.metrics import SCI_PASS

    container.add_argument(
        "--sci-threshold", type=float, default=default,
        help="Scalp coupling index a window must reach"
             + (f" (default {SCI_PASS})." if default is not None else ", e.g. 0.8.")
             + (f" {note}" if note else ""))


def add_psp_threshold(container, note: str = "") -> None:
    """``--psp-threshold``: the other line, from the same criteria table."""
    from fnirs_pipe.qc.metrics import PSP_PASS

    container.add_argument(
        "--psp-threshold", type=float, default=None,
        help=f"Peak spectral power a window must reach (default {PSP_PASS}). PSP catches "
             f"the movement that fakes a high SCI, so a window counts as coupled only when "
             f"it clears this line and --sci-threshold together."
             + (f" {note}" if note else ""))


def add_min_good_frac(container, note: str = "") -> None:
    """``--min-good-frac``: how much of the recording a channel has to be coupled for."""
    from fnirs_pipe.qc.metrics import GOOD_FRAC_PASS

    container.add_argument(
        "--min-good-frac", type=float, default=None,
        help=f"Share of windows a channel must be coupled in to be kept, 0 to 1 (default "
             f"{GOOD_FRAC_PASS}). A window counts when SCI and PSP both clear their lines "
             f"in it. This is the criterion that rejects; the two lines above set what a "
             f"coupled window is. Counting windows rather than averaging them is what stops "
             f"a channel that was fine for the first half of a long recording and dead for "
             f"the second half from passing."
             + (f" {note}" if note else ""))


def add_screen_scope(container, note: str = "") -> None:
    """``--screen-scope``: which part of the recording the coupled windows are counted over."""
    container.add_argument(
        "--screen-scope", choices=["run", "task"], default="run",
        help="Which windows count toward --min-good-frac. 'run' (default) counts the whole "
             "recording. 'task' counts only the annotated task blocks, so the lead-in "
             "before the first block and the gaps between them stop being held against a "
             "channel that is coupled throughout every block. 'task' falls back to 'run' "
             "when no annotation is long enough to hold two screening windows, which is "
             "what a recording carrying only short triggers looks like."
             + (f" {note}" if note else ""))


def screening(sci_default: "float | None" = None, note: str = "") -> argparse.ArgumentParser:
    """The screening lines as a parent parser.

    One block for all three, because they are one decision: SCI and PSP say what a coupled
    window is, and the third says how many of them a channel needs. A command that can move
    one has no reason not to move the others.
    """
    p = argparse.ArgumentParser(add_help=False)
    add_sci_threshold(p, sci_default, note)
    add_psp_threshold(p, note)
    add_min_good_frac(p, note)
    add_screen_scope(p, note)
    return p


def pairs_selection() -> argparse.ArgumentParser:
    """``--pairs-csv`` / ``--group-id`` / ``--task-label``: which dyads to process."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pairs-csv", type=Path, required=True,
                   help="CSV with columns: group_id, subject_id, task. Each unique "
                        "(group_id, task) pair is processed as one session.")
    p.add_argument("--group-id", default=None, type=BidsLabel,
                   help="Process only this group_id. Omit to process all groups.")
    p.add_argument("--task-label", "--task_label", nargs="+", action="extend", type=BidsLabel,
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
                   help="Report on this time (s) onward, on the aligned clock, where 0 is "
                        "the shared trigger. Omit to start at the alignment point.")
    p.add_argument("--tend", type=float, default=None,
                   help="Report up to this time (s) on the aligned clock. Omit to run to "
                        "the end; a value past the end is clipped. The window narrows the "
                        "synchrony metrics only: the per-subject quality record describes "
                        "the whole recording either way. For `fnirs-hyper run` this selects "
                        "rather than cuts: the wavelet transform is computed over the whole "
                        "recording and the window read out of it, so the window carries the "
                        "recording's cone of influence rather than two edges of its own, "
                        "and a condition falling outside it is dropped from the run. This "
                        "cut the recording until now, so numbers from before are not "
                        "reproducible with it.")
    return p
