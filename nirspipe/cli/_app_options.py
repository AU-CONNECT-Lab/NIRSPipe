"""Option blocks the BIDS App's commands share: nirspipe, nirspipe-qc, the nirspipe-hyper family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nirspipe.cli._shared import BidsLabel
from nirspipe.io.head_template import CHOICES, FSAVERAGE, resolve_template, space_of, template_dir
from nirspipe.pipeline.population import CHOICES as POPULATIONS, age_span, population_values
from nirspipe.utils import bare_roi_map, roi_overlaps
from nirspipe.utils.logging import get_logger


def load_roi_mapping(path: "Path | None") -> "dict[str, list[str]] | None":
    """The ``--roi-mapping`` JSON as S-D pairs, or exit non-zero if it cannot be read.

    A file that fails to load must not just drop the ROI output. Entries lose any chromophore
    suffix, which every path ignores anyway: the chromophore comes from the pass, not the map.
    """
    if not path:
        return None
    try:
        roi_map = bare_roi_map(json.loads(Path(path).read_text(encoding="utf-8-sig")))
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


def dpf_value(text: str) -> "float | str":
    """One ``--dpf`` token: a number, or the word ``age``."""
    if text.strip().lower() == "age":
        return "age"
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is neither a number nor 'age'") from None


def add_dpf(container, required: bool = False) -> None:
    """``--dpf``: the factors, or ``age`` for one per wavelength from participants.tsv."""
    # extend: accepts space-separated (--dpf 6 6) and repeated (--dpf 6 --dpf 6) forms
    container.add_argument(
        "--dpf", nargs="+", type=dpf_value, action="extend", required=required,
        help="Differential pathlength factor: one value, one per wavelength, or age for one "
             "per wavelength from each participant's age in participants.tsv (Scholkmann & "
             "Wolf 2013).")


def add_template(container) -> None:
    """``--template``: the head the optodes are placed on; fsaverage, or by age under ``--dpf age``."""
    container.add_argument(
        "--template", choices=CHOICES, default=None,
        help="Head the optodes are placed on and drawn over: fsaverage (adult) or an MNE infant "
             "template by age, 2wk to 2yr. Default fsaverage; with --dpf age, the template "
             "nearest each participant's age, and ages over 2 and under 18 need this option.")


def add_population(container, bands: str) -> None:
    """``--population``: an age group's values for ``bands``, the DPF and the template, each overridden by its own flag."""
    groups = ", ".join(f"{name} ({age_span(name)})" for name in POPULATIONS)
    container.add_argument(
        "--population", choices=POPULATIONS, default=None,
        help=f"Age group whose values fill {bands}, --dpf and --template where they are not "
             f"given: {groups}. The DPF is the age equation at one age for the whole batch; "
             f"infant places each participant on the template nearest their age.")


def fill_population(args, bands: "tuple[str, ...]") -> "list[str]":
    """Set on ``args`` what ``--population`` fills where nothing else did, returning those dests.

    ``--population adult --cardiac-h-freq 3`` with ``bands`` the two cardiac edges gives
    ``["cardiac_l_freq", "dpf"]`` and leaves the 3 in place.
    """
    if args.population is None:
        return []
    values = population_values(args.population)
    filled = [key for key in bands if getattr(args, key) is None]
    for key in filled:
        setattr(args, key, values[key])
    if args.dpf is None:
        # the age equation at the population's age, never a participant's
        args.dpf = ["age"]
        filled.append("dpf")
    if values["template_by_age"] and args.template is None:
        filled.append("template")
    return filled


def population_age(population: "str | None", from_population) -> "float | None":
    """The age the DPF is computed at where ``--population`` set it, else None."""
    return (population_values(population)["dpf_age"]
            if population and "dpf" in (from_population or ()) else None)


def participant_ages(dpf: list, from_population, bids_dir: Path, subjects: "list[str]") -> dict:
    """Each subject's age where ``--dpf age`` or an age-picked template reads it; exits on one without."""
    from nirspipe.io.bids import get_participant_age
    from nirspipe.pipeline.dpf import dpf_from_age

    from_population = from_population or ()
    try:
        by_dpf = dpf_from_age(dpf or []) and "dpf" not in from_population
        by_template = "template" in from_population
        if not (by_dpf or by_template):
            return {}
        fixes = []
        if by_dpf:
            fixes.append("pass --dpf as numbers")
        if by_template:
            fixes.append("pass --template (--population infant places each participant on "
                         "the template nearest their age)")
        return {s: get_participant_age(bids_dir, s, fix=" or ".join(fixes)) for s in subjects}
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def templates_for(template: "str | None", subjects: "list[str]", ages: dict) -> dict:
    """Each subject's template, every infant one fetched before the first subject runs.

    ``ages`` holds a subject's age only under ``--dpf age``. Exits non-zero on an age no
    template fits, or a template that cannot be fetched.
    """
    chosen = {}
    for subject in subjects:
        try:
            chosen[subject] = resolve_template(template, ages.get(subject))
        except ValueError as exc:
            print(f"Error: sub-{subject}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
    for name in sorted(set(chosen.values()) - {FSAVERAGE}):
        try:
            template_dir(space_of(name))
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
    return chosen


def add_sci_threshold(container, required: bool = False, note: str = "") -> None:
    """``--sci-threshold``: the coupling line channel screening rejects on. It has no default
    anywhere; ``required=False`` is for a parser that enforces it per analysis level."""
    container.add_argument(
        "--sci-threshold", type=float, required=required,
        help="Scalp coupling index a window must reach, e.g. 0.8. Required: no default."
             + (f" {note}" if note else ""))


def add_psp_threshold(container, note: str = "") -> None:
    """``--psp-threshold``: an optional second line; without it SCI alone decides a window."""
    container.add_argument(
        "--psp-threshold", type=float, default=None,
        help="Peak spectral power (QT-NIRS scale) a window must also reach. No default: SCI "
             "alone decides and PSP is only shown. 0.1 is a line for fitting the headgear; "
             "0.075 has been used after recording."
             + (f" {note}" if note else ""))


def add_min_good_frac(container, note: str = "") -> None:
    """``--min-good-frac``: how much of the recording a channel has to pass in."""
    from nirspipe.qc.metrics import GOOD_FRAC_PASS

    container.add_argument(
        "--min-good-frac", type=float, default=None,
        help=f"Share of windows a channel must pass in to be kept, 0 to 1 (default "
             f"{GOOD_FRAC_PASS}). A window passes when it clears --sci-threshold, and "
             f"--psp-threshold if given. This is the criterion that rejects."
             + (f" {note}" if note else ""))


def add_screen_scope(container, note: str = "") -> None:
    """``--screen-scope``: which part of the recording the passing windows are counted over."""
    container.add_argument(
        "--screen-scope", choices=["run", "task"], default="run",
        help="Which windows count toward --min-good-frac: 'run' (default) the whole "
             "recording, 'task' only the annotated task blocks. 'task' falls back to 'run' "
             "when no annotation holds two screening windows."
             + (f" {note}" if note else ""))


MOTION_CHOICES = ["tddr", "wavelet", "tpca", "spline", "spline_sg", "wavelet_tddr",
                  "spline_wavelet", "none"]


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


def add_reviewer_marks(container, bad_spans_note: str = "") -> None:
    """``--bad-channels``, ``--keep-spans``, ``--keep-spans-from``, ``--bad-spans``: what a reviewer marked."""
    container.add_argument("--bad-channels",
                           help="Source-detector labels to mark as bad, e.g. 'S1_D1,S2_D3', for "
                                "every subject; or a table with a bad_channels column, rows "
                                "optionally limited by participant_id, session, task, acq and "
                                "run as in --keep-spans. Either wavelength marks the pair. "
                                "Marked, not removed, and added to the SCI-detected bad channels. "
                                "A table with a rater column holds one rater's decisions.")
    container.add_argument("--keep-spans", type=Path, metavar="TABLE",
                           help="Table of the stretches to use (onset and duration in s on each "
                                "recording's own time axis), rows optionally limited by "
                                "participant_id, session, task, acq and run. The rest is marked "
                                "BAD_unselected and left out of screening and every estimate; a "
                                "recording no row matches is used whole. See --keep-spans-from.")
    container.add_argument("--keep-spans-from", metavar="MARKER",
                           help="Measure each --keep-spans onset from the first annotation named "
                                "MARKER instead of the recording's start, so one row serves "
                                "recordings that did not start time-locked; an onset may be "
                                "negative. --bad-spans stays on the recording's own axis.")
    container.add_argument("--bad-spans", type=Path, metavar="TABLE",
                           help="Table of corrupted stretches (motion, a loose optode) in the "
                                "--keep-spans format. Each is marked BAD_manual: left out of "
                                "screening and of every estimate. One rater's, as for "
                                "--bad-channels." + bad_spans_note)


def check_keep_spans_from(keep_spans: "Path | None", keep_spans_from: "str | None") -> None:
    if keep_spans_from is not None and keep_spans is None:
        print("Error: --keep-spans-from needs --keep-spans.", file=sys.stderr)
        raise SystemExit(1)


def add_bids_filter_file(container) -> None:
    container.add_argument("--bids-filter-file", "--bids_filter_file", type=Path,
                           help="JSON file with extra pybids query filters.")


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


def add_alignment_window(container) -> None:
    """``--normalize`` / ``--no-align`` / ``--tstart`` / ``--tend``: what the metrics see."""
    container.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                           help="Z-score each channel per subject after alignment.")
    container.add_argument("--no-align", action="store_true",
                           help="Skip trigger-based alignment; trim all recordings to the "
                                "shortest duration.")
    container.add_argument("--tstart", type=float, default=None,
                           help="Report on this time (s) onward, on the aligned clock, where 0 "
                                "is the shared trigger. Omit to start at the alignment point.")
    container.add_argument("--tend", type=float, default=None,
                           help="Report up to this time (s) on the aligned clock; omit to run to "
                                "the end. Narrows the synchrony metrics only, not the "
                                "per-subject quality record. In `nirspipe-hyper` the window is "
                                "read out of the whole-recording transform, and conditions "
                                "outside it are dropped.")


def alignment_window() -> argparse.ArgumentParser:
    """:func:`add_alignment_window` as a parent parser."""
    p = argparse.ArgumentParser(add_help=False)
    add_alignment_window(p)
    return p
