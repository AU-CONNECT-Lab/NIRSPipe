"""fnirs-prep CLI (argparse): data preparation utilities (marker editing, crop, etc.)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fnirs_pipe import __version__

from fnirs_pipe.cli import _shared
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("cli.prep")


def _run_parallel(fn, subjects: list[str], n_jobs: int, **kwargs) -> list[tuple]:
    """Run fn(sub, **kwargs) for each subject, return list of (sub, ok, msg)."""
    from joblib import Parallel, delayed

    def _one(sub):
        try:
            result = fn(sub, **kwargs)
            return sub, True, str(result)
        except Exception as exc:
            return sub, False, str(exc)

    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one)(sub) for sub in subjects
    )


def _report(results: list[tuple]) -> None:
    n_fail = 0
    for sub, ok, msg in results:
        if ok:
            print(f"  sub-{sub}: {msg}")
        else:
            print(f"  sub-{sub}: [error] {msg}", file=sys.stderr)
            n_fail += 1
    if n_fail:
        raise SystemExit(1)


def cmd_crop(
    bids_dir: Path, output_dir: Path, participant_label: list[str],
    session_label: str | None, task_label: str | None, run_label: str | None,
    tmin: float | None, tmax: float | None, segments_path: Path | None,
    combine: bool, align: str, trigger_name: str | None, input_desc: str | None,
    n_jobs: int, skip_bids_validation: bool, margin_s: str | None = None,
    band_fmin: float | None = None,
) -> None:
    """Crop SNIRFs and write to derivatives/cropped/."""
    # cropping a pipeline output back into its own tree is the intended use of --input-desc
    if input_desc is None:
        _shared.refuse_output_in_input(bids_dir, output_dir, "fnirs-prep")
    if not skip_bids_validation:
        from fnirs_pipe.io.bids import validate_bids
        validate_bids(bids_dir)
    if segments_path is not None and (tmin is not None or tmax is not None):
        print("[error] --segments-path and --tmin/--tmax are mutually exclusive.", file=sys.stderr)
        raise SystemExit(1)
    if segments_path is None and tmin is None and tmax is None:
        print("[error] Specify --segments-path or at least one of --tmin / --tmax.", file=sys.stderr)
        raise SystemExit(1)
    if combine and segments_path is None:
        print("[error] --combine requires --segments-path.", file=sys.stderr)
        raise SystemExit(1)
    if align == "trigger" and not trigger_name:
        print("[error] --align trigger requires --trigger-name.", file=sys.stderr)
        raise SystemExit(1)

    from fnirs_pipe.pipeline.crop import _segment_stems, crop_snirf

    # every subject reads the same table, so a table that cannot name its segments is one
    # error here rather than the same error once per subject
    if segments_path is not None and not combine:
        from fnirs_pipe.io.tables import read_table

        try:
            _segment_stems(read_table(segments_path), stem="")
        except ValueError as exc:
            print(f"[error] {segments_path}: {exc}", file=sys.stderr)
            raise SystemExit(1)

    # "auto" is the width the lowest analysed frequency needs, so it can only be resolved
    # once that frequency is known; a crop made for a band nobody has chosen yet keeps none
    margin = 0.0
    if margin_s not in (None, "", "0"):
        if str(margin_s).lower() == "auto":
            if not band_fmin:
                print("[error] --margin auto requires --band-fmin (the lowest frequency the "
                      "analysis will average over).", file=sys.stderr)
                raise SystemExit(1)
            from fnirs_pipe.pipeline.hyper.wtc import cone_margin_s
            margin = cone_margin_s(band_fmin)
        else:
            margin = float(margin_s)
        print(f"[info] keeping a {margin:.1f}s margin on each side of every segment")

    def _crop_one(sub):
        return crop_snirf(
            bids_dir, output_dir, sub,
            ses=session_label, task=task_label, run=run_label,
            tmin=tmin, tmax=tmax,
            segments_path=segments_path,
            combine=combine,
            align=align, trigger_name=trigger_name,
            input_desc=input_desc,
            validate=True,
            margin_s=margin,
        )

    _report(_run_parallel(_crop_one, participant_label, n_jobs))


def cmd_align(
    bids_dir: Path, output_dir: Path, group_csv: Path, skip_bids_validation: bool,
) -> None:
    """Align multi-subject recordings by shared trigger and write SNIRF files."""
    _shared.refuse_output_in_input(bids_dir, output_dir, "fnirs-prep")
    if not skip_bids_validation:
        from fnirs_pipe.io.bids import validate_bids
        validate_bids(bids_dir)
    from fnirs_pipe.exceptions import AlignmentError
    from fnirs_pipe.io.derivatives import LINK_RAW, entity_of, write_dataset_description
    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.pipeline.hyper import (
        align_recordings, member_snirfs, parse_group_csv, write_aligned_member,
    )
    from fnirs_pipe.utils.snirf_prep import (
        deriv_nirs_dir, copy_dataset_root,
    )

    _DERIV_NAME = "aligned"

    try:
        groups = parse_group_csv(group_csv)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    n_fail = 0
    for (group_id, task, ses), group in groups.items():
        print(f"Group {group_id}{f' ses-{ses}' if ses else ''} task-{task} ({len(group)} subjects)")

        try:
            paths = member_snirfs(bids_dir, group, validate=True)
            raws = {sid: read_snirf(p, verbose=False) for sid, p in paths.items()}
        except Exception as exc:
            print(f"  [error] loading: {exc}", file=sys.stderr)
            n_fail += 1
            continue

        try:
            aligned_raws, offsets = align_recordings(raws, task)
        except AlignmentError as exc:
            print(f"  [warning] {exc}", file=sys.stderr)
            n_fail += 1
            continue

        write_dataset_description(output_dir / _DERIV_NAME, name=_DERIV_NAME,
                                  generated_by="fnirs-prep align", source=bids_dir,
                                  link=LINK_RAW)
        copy_dataset_root(bids_dir, output_dir / _DERIV_NAME)

        for entry in group:
            path = paths[entry.subject_id]
            out_dir = deriv_nirs_dir(output_dir, _DERIV_NAME,
                                     entry.subject_id.removeprefix("sub-"),
                                     entity_of(path, "ses"))
            try:
                out_snirf = write_aligned_member(aligned_raws[entry.subject_id], path,
                                                 out_dir, group_id)
            except AlignmentError as exc:
                print(f"  {entry.subject_id}: [error] {exc}", file=sys.stderr)
                n_fail += 1
                continue
            print(f"  {entry.subject_id}: offset={offsets[entry.subject_id]:.3f}s -> {out_snirf.name}")

    if n_fail:
        raise SystemExit(1)


def cmd_markers_export(
    bids_dir: Path, out_dir: Path, participant_label: list[str],
    session_label: str | None, task_label: str | None, run_label: str | None,
    n_jobs: int, skip_bids_validation: bool,
) -> None:
    """Export events.tsv(s) to out_dir for manual editing."""
    _shared.refuse_output_in_input(bids_dir, out_dir, "fnirs-prep")
    if not skip_bids_validation:
        from fnirs_pipe.io.bids import validate_bids
        validate_bids(bids_dir)
    from fnirs_pipe.pipeline.edit_markers import export_markers

    def _export_one(sub):
        return export_markers(
            bids_dir, sub, out_dir,
            ses=session_label, task=task_label, run=run_label,
            validate=True,
        )

    _report(_run_parallel(_export_one, participant_label, n_jobs))


def cmd_markers_apply(
    bids_dir: Path, output_dir: Path, participant_label: list[str],
    session_label: str | None, task_label: str | None, run_label: str | None,
    tsv: Path | None, shift: float | None, set_duration: float | None,
    rename: list[str] | None, n_jobs: int, skip_bids_validation: bool,
) -> None:
    """Apply marker edits to runs and write to derivatives/marker_edited/."""
    _shared.refuse_output_in_input(bids_dir, output_dir, "fnirs-prep")
    if not skip_bids_validation:
        from fnirs_pipe.io.bids import validate_bids
        validate_bids(bids_dir)
    ops = [x for x in (tsv, shift, set_duration, rename) if x is not None]
    if not ops:
        print("[error] Specify one of: --tsv, --shift, --set-duration, --rename", file=sys.stderr)
        raise SystemExit(1)
    if len(ops) > 1:
        print("[error] Only one of --tsv, --shift, --set-duration, --rename can be used at a time.", file=sys.stderr)
        raise SystemExit(1)
    from fnirs_pipe.pipeline.edit_markers import apply_markers

    def _apply_one(sub):
        return apply_markers(
            bids_dir, output_dir, sub,
            ses=session_label, task=task_label, run=run_label,
            tsv=tsv, shift=shift, set_duration=set_duration, rename=rename,
            validate=True,
        )

    _report(_run_parallel(_apply_one, participant_label, n_jobs))


def _add_selection(sp) -> None:
    sp.add_argument("--participant-label", "--participant_label", nargs="+",
                    action="extend", required=True, type=_shared.BidsLabel,
                    help="Subject ID(s) to process.")
    sp.add_argument("--session-label", "--session_label", default=None, type=_shared.BidsLabel,
                    help="Session label.")
    sp.add_argument("--task-label", "--task_label", default=None, type=_shared.BidsLabel,
                    help="Task label.")
    sp.add_argument("--run-label", "--run_label", default=None, type=_shared.BidsLabel,
                    help="Run label.")
    _shared.add_n_jobs(sp)
    _shared.add_skip_bids_validation(sp)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-prep",
        description="fNIRS data preparation: cropping, alignment and marker editing.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-prep {__version__}")
    sub = p.add_subparsers(required=True)

    crop = sub.add_parser("crop", help="Crop SNIRFs to a time window or segments.")
    crop.add_argument("bids_dir",        type=Path, help="BIDS dataset root, or a "
                                                         "derivatives tree with --input-desc.")
    crop.add_argument("output_dir", type=Path,
                  help="Where the derived tree goes. A `cropped/` subtree is created under it, so point this at a derivatives root rather than at one dataset.")
    _add_selection(crop)
    crop.add_argument("--tmin", type=float, default=None, help="Start time in seconds (single segment).")
    crop.add_argument("--tmax", type=float, default=None, help="End time in seconds (single segment).")
    crop.add_argument("--segments-path", type=Path, default=None,
                      help="TSV with onset/duration columns defining segments to keep. Each "
                           "segment written to its own file is named by a `task` column, "
                           "which more than one segment needs unless --combine is given.")
    crop.add_argument("--align", choices=["none", "trigger"], default="none",
                      help="t=0 for --tmin/--tmax and segment onsets: 'none' = recording start, "
                           "'trigger' = first annotation named --trigger-name.")
    crop.add_argument("--trigger-name", default=None,
                      help="Annotation description marking t=0 when --align trigger. "
                           "A recording without it falls back to the recording start.")
    crop.add_argument("--combine", action=argparse.BooleanOptionalAction, default=False,
                      help="Concatenate multi-segment output into one file.")
    crop.add_argument("--margin", dest="margin_s", default=None, metavar="SEC|auto",
                      help="Keep this many extra seconds on each side of every segment, and "
                           "record the span that was asked for in the sidecar. 'auto' takes "
                           "the width from --band-fmin. Off by default.")
    crop.add_argument("--band-fmin", type=float, default=None, metavar="HZ",
                      help="Lowest frequency the later analysis will average over, used only "
                           "to resolve --margin auto. Give the same value as "
                           "fnirs-hyper --wtc-band-fmin.")
    crop.add_argument("--input-desc", default=None, metavar="DESC",
                      help="Cut a processed stage instead of a recording, e.g. 'errts' or "
                           "'filtered'. bids_dir is then a derivatives tree. The desc- "
                           "entity is kept on the output.")
    crop.set_defaults(func=cmd_crop)

    align = sub.add_parser("align", help="Align multi-subject recordings by shared trigger.")
    align.add_argument("bids_dir",        type=Path, help="BIDS dataset root.")
    align.add_argument("output_dir", type=Path,
                   help="Where the derived tree goes. An `aligned/` subtree is created under it.")
    align.add_argument("--group-csv", type=Path, required=True,
                       help="CSV with group_id, subject_id, task columns.")
    _shared.add_skip_bids_validation(align)
    align.set_defaults(func=cmd_align)

    markers = sub.add_parser("edit-markers", help="Edit markers in SNIRF files.")
    msub = markers.add_subparsers(required=True)

    exp = msub.add_parser("export", help="Export events.tsv(s) for manual editing.")
    exp.add_argument("bids_dir", type=Path, help="BIDS dataset root.")
    exp.add_argument("out_dir",  type=Path, help="Directory to write the exported events TSV(s).")
    _add_selection(exp)
    exp.set_defaults(func=cmd_markers_export)

    app = msub.add_parser("apply", help="Apply marker edits (tsv/shift/set-duration/rename).")
    app.add_argument("bids_dir",        type=Path, help="BIDS dataset root.")
    app.add_argument("output_dir", type=Path,
                 help="Where the derived tree goes. A `marker_edited/` subtree is created under it.")
    _add_selection(app)
    app.add_argument("--tsv", type=Path, default=None,
                     help="Edited events TSV to apply (same file applied to all subjects).")
    app.add_argument("--shift", type=float, default=None,
                     help="Shift all onsets by this many seconds (negative = earlier). Clipped to 0.")
    app.add_argument("--set-duration", type=float, default=None,
                     help="Set all marker durations to this value (seconds).")
    app.add_argument("--rename", nargs="+", action="extend", default=None,
                     help="Rename marker: 'old:new'. Repeatable.")
    app.set_defaults(func=cmd_markers_apply)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
