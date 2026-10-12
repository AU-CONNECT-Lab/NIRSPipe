"""Option blocks more than one ``fnirs-*`` command declares.

Each flag is declared here exactly once. ``add_*`` puts one on a parser or on an argument
group, for a command that files it under a heading of its own; the plain functions return a
parent parser to hand to ``add_parser(parents=[...])``.

Only genuinely shared parameters belong here. A flag that means something different to two
commands stays in each of them, as the dyad coherence window and the per-channel QC window
do.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nirspipe.utils import bare_roi_map, roi_overlaps
from nirspipe.utils.logging import get_logger


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


def load_roi_mapping(path: "Path | None") -> "dict[str, list[str]] | None":
    """The ``--roi-mapping`` JSON as S-D pairs, or exit non-zero if it cannot be read.

    A file that fails to load must not just drop the ROI output. Entries lose any chromophore
    suffix, which every path ignores anyway: the chromophore comes from the pass, not the map.
    """
    if not path:
        return None
    try:
        roi_map = bare_roi_map(json.loads(Path(path).read_text()))
    except Exception as exc:
        print(f"Error: failed to load ROI mapping {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    shared = roi_overlaps(roi_map)
    if shared:
        get_logger("cli").warning(
            "ROI mapping %s lists %d channel(s) in more than one ROI; each counts in all of "
            "them: %s", path, len(shared),
            ", ".join(f"{ch} ({', '.join(rois)})" for ch, rois in shared.items()))
    return roi_map


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


def add_separation_bands(container, note: str = "") -> None:
    """``--short-max-dist`` / ``--long-min-dist`` / ``--long-max-dist``, in mm.

    They are validated together in :func:`separation_bands_from_args`, so moving one and
    leaving the others cannot produce overlapping bands.
    """
    from nirspipe.qc.metrics._helpers import LONG_MIN_DIST, SHORT_MAX_DIST

    container.add_argument(
        "--short-max-dist", type=float, default=None, metavar="MM",
        help=f"Separation at or below which a channel is short-distance, in mm "
             f"(default {SHORT_MAX_DIST * 1e3:.0f}). Short channels see scalp only and are "
             f"measured, and regressed, separately from the long ones."
             + (f" {note}" if note else ""))
    container.add_argument(
        "--long-min-dist", type=float, default=None, metavar="MM",
        help=f"Separation at or above which a channel is long, in mm (default "
             f"{LONG_MIN_DIST * 1e3:.0f}). A channel in the gap above --short-max-dist "
             f"counts as neither short nor long."
             + (f" {note}" if note else ""))
    container.add_argument(
        "--long-max-dist", type=float, default=None, metavar="MM",
        help="Separation above which a channel is too far to be long, in mm. Off by "
             "default, so any separation past --long-min-dist counts as long."
             + (f" {note}" if note else ""))


# One tuple so the flag names, the config keys and the PrepConfig / PostConfig fields cannot
# drift apart. They are spelled the same in all three places on purpose.
SEPARATION_BAND_KEYS = ("short_max_dist", "long_min_dist", "long_max_dist")


def separation_bands_from_args(args) -> dict:
    """The three flags as PrepConfig / PostConfig fields, in metres, validated together.

    args with --long-max-dist 55 -> {"short_max_dist": None, "long_min_dist": None,
                                     "long_max_dist": 0.055}

    Values arrive in mm and are stored in metres, as MNE reports them. A flag left off stays None so the config
    keeps the package default; validation therefore runs on the resolved bands rather
    than on what was typed.
    """
    from nirspipe.qc.metrics._helpers import separation_bands, validate_bands

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
    from nirspipe.qc.metrics._helpers import separation_bands

    return separation_bands(type("Args", (), separation_bands_from_args(args)))


def add_sci_threshold(container, required: bool = False, note: str = "") -> None:
    """``--sci-threshold``: the coupling line channel screening rejects on. It has no default
    anywhere; ``required=False`` is for a parser that enforces it per analysis level."""
    container.add_argument(
        "--sci-threshold", type=float, required=required,
        help="Scalp coupling index a window must reach, e.g. 0.8. Required: no default."
             + (f" {note}" if note else ""))


def add_psp_threshold(container, note: str = "") -> None:
    """``--psp-threshold``: the other line, from the same criteria table."""
    from nirspipe.qc.metrics import PSP_PASS

    container.add_argument(
        "--psp-threshold", type=float, default=None,
        help=f"Peak spectral power a window must reach (default {PSP_PASS}). A window "
             f"counts as coupled only when it clears this line and --sci-threshold together."
             + (f" {note}" if note else ""))


def add_min_good_frac(container, note: str = "") -> None:
    """``--min-good-frac``: how much of the recording a channel has to be coupled for."""
    from nirspipe.qc.metrics import GOOD_FRAC_PASS

    container.add_argument(
        "--min-good-frac", type=float, default=None,
        help=f"Share of windows a channel must be coupled in to be kept, 0 to 1 (default "
             f"{GOOD_FRAC_PASS}). A window counts when SCI and PSP both clear their lines "
             f"in it. This is the criterion that rejects; the two lines above set what a "
             f"coupled window is."
             + (f" {note}" if note else ""))


def add_screen_scope(container, note: str = "") -> None:
    """``--screen-scope``: which part of the recording the coupled windows are counted over."""
    container.add_argument(
        "--screen-scope", choices=["run", "task"], default="run",
        help="Which windows count toward --min-good-frac: 'run' (default) the whole "
             "recording, 'task' only the annotated task blocks. 'task' falls back to 'run' "
             "when no annotation holds two screening windows."
             + (f" {note}" if note else ""))


def add_motion_settings(container) -> None:
    """``--motion-std-thresh`` and ``--wavelet-iqr``: the two settings some correction methods read."""
    from nirspipe.pipeline.motion import IQR_FACTOR_DEFAULTS, STD_THRESH_DEFAULTS

    def defaults(table: dict[str, float]) -> str:
        return ", ".join(f"{method} {value:g}" for method, value in table.items())

    container.add_argument(
        "--motion-std-thresh", type=float, default=None,
        help="Motion detection threshold of the spline methods, in standard deviations of "
             "each channel's first difference: a source-detector pair's samples are marked "
             "where either wavelength changes by more than this within the detection window. "
             f"Read by {', '.join(STD_THRESH_DEFAULTS)}; default per method "
             f"({defaults(STD_THRESH_DEFAULTS)}). Refused with any other method.")
    container.add_argument(
        "--wavelet-iqr", type=float, default=None,
        help="Wavelet outlier fence, in interquartile ranges beyond the quartiles: detail "
             "coefficients past it are zeroed, so a smaller value removes more of the signal. "
             f"Read by {', '.join(IQR_FACTOR_DEFAULTS)}; default per method "
             f"({defaults(IQR_FACTOR_DEFAULTS)}). Refused with any other method.")


def check_motion_settings(method: str | None, std_thresh: float | None,
                          iqr_factor: float | None) -> None:
    """Exit non-zero when a motion setting is one the chosen method does not read, or not positive."""
    from nirspipe.pipeline.motion import motion_settings

    try:
        motion_settings(method, std_thresh, iqr_factor)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)


def screening(note: str = "") -> argparse.ArgumentParser:
    """The screening lines as a parent parser.

    One block for all three, because they are one decision: SCI and PSP say what a coupled
    window is, and the third says how many of them a channel needs.
    """
    p = argparse.ArgumentParser(add_help=False)
    add_sci_threshold(p, required=True, note=note)
    add_psp_threshold(p, note)
    add_min_good_frac(p, note)
    add_screen_scope(p, note)
    return p


def pairs_selection() -> argparse.ArgumentParser:
    """``--pairs-csv`` / ``--group-id`` / ``--participant-label`` / ``--task-label``: which dyads to process."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--pairs-csv", type=Path, required=True,
                   help="CSV with columns: group_id, subject_id, task, and optionally "
                        "session, run and occasion. Each unique (group_id, task, session) is "
                        "processed as one group; an occasion column joins members whose "
                        "session labels differ and names the group's outputs ses-<occasion>.")
    p.add_argument("--group-id", default=None, type=BidsLabel,
                   help="Process only this group_id. Omit to process all groups.")
    p.add_argument("--participant-label", "--participant_label", nargs="+", action="extend",
                   type=BidsLabel,
                   help="Process only the groups with at least one of these subjects as a "
                        "member. The whole group still runs, partner included.")
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
                   help="Report up to this time (s) on the aligned clock; omit to run to the "
                        "end. Narrows the synchrony metrics only, not the per-subject quality "
                        "record. In `nirspipe-hyper` the window is read out of the "
                        "whole-recording transform, and conditions outside it are dropped.")
    return p
