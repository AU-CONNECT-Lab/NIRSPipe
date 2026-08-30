"""fnirs-prep CLI (argparse) — data preparation utilities (marker editing, crop, etc.)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    bids_dir: Path, derivatives_dir: Path, participant_label: list[str],
    ses: str | None, task: str | None, run: str | None,
    tmin: float | None, tmax: float | None, segments_path: Path | None,
    combine: bool, n_jobs: int, skip_bids_validation: bool,
) -> None:
    """Crop raw SNIRFs and write to derivatives/cropped/."""
    if segments_path is not None and (tmin is not None or tmax is not None):
        print("[error] --segments-path and --tmin/--tmax are mutually exclusive.", file=sys.stderr)
        raise SystemExit(1)
    if segments_path is None and tmin is None and tmax is None:
        print("[error] Specify --segments-path or at least one of --tmin / --tmax.", file=sys.stderr)
        raise SystemExit(1)
    if combine and segments_path is None:
        print("[error] --combine requires --segments-path.", file=sys.stderr)
        raise SystemExit(1)

    from fnirs_pipe.pipeline.crop import crop_snirf

    def _crop_one(sub):
        return crop_snirf(
            bids_dir, derivatives_dir, sub,
            ses=ses, task=task, run=run,
            tmin=tmin, tmax=tmax,
            segments_path=segments_path,
            combine=combine,
            validate=not skip_bids_validation,
        )

    _report(_run_parallel(_crop_one, participant_label, n_jobs))


def cmd_align(
    bids_dir: Path, derivatives_dir: Path, group_csv: Path, skip_bids_validation: bool,
) -> None:
    """Align multi-subject recordings by shared trigger and write SNIRF files."""
    import mne
    import pandas as pd

    from fnirs_pipe.exceptions import AlignmentError
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings, load_group_raw_bids, parse_group_csv,
    )
    from fnirs_pipe.io.snirf import write_snirf
    from fnirs_pipe.utils.snirf_prep import (
        annotations_to_df, bids_stem, copy_sidecars, deriv_nirs_dir,
        copy_dataset_root, ensure_dataset_description, find_snirf,
    )

    _DERIV_NAME = "aligned"

    try:
        groups = parse_group_csv(group_csv)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    n_fail = 0
    for (group_id, task), group in groups.items():
        print(f"Group {group_id} task-{task} ({len(group)} subjects)")

        try:
            raws = load_group_raw_bids(bids_dir, group)
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

        ensure_dataset_description(
            derivatives_dir / _DERIV_NAME, _DERIV_NAME, "fnirs-prep align"
        )
        copy_dataset_root(bids_dir, derivatives_dir / _DERIV_NAME)

        offset_rows: list[dict] = []
        for entry in group:
            sub_label = entry.subject_id.removeprefix("sub-")
            try:
                snirf_path = find_snirf(
                    bids_dir, sub_label, None, task, None,
                    validate=not skip_bids_validation,
                )
            except Exception as exc:
                print(f"  {entry.subject_id}: [error] {exc}", file=sys.stderr)
                n_fail += 1
                continue

            stem     = bids_stem(snirf_path)
            out_dir  = deriv_nirs_dir(derivatives_dir, _DERIV_NAME, sub_label, None)
            out_dir.mkdir(parents=True, exist_ok=True)
            copy_sidecars(snirf_path, stem, out_dir)

            out_snirf   = out_dir / f"{stem}_nirs.snirf"
            raw_aligned = aligned_raws[entry.subject_id]
            write_snirf(raw_aligned, out_snirf)
            annotations_to_df(raw_aligned).to_csv(
                out_dir / f"{stem}_events.tsv", sep="\t", index=False
            )

            print(f"  {entry.subject_id}: offset={offsets[entry.subject_id]:.3f}s -> {out_snirf.name}")
            offset_rows.append({
                "subject_id": entry.subject_id,
                "offset_s":   round(offsets[entry.subject_id], 3),
                "duration_s": round(raw_aligned.times[-1], 1),
            })

        offsets_path = (
            derivatives_dir / _DERIV_NAME
            / f"group-{group_id}_task-{task}_align-offsets.tsv"
        )
        pd.DataFrame(offset_rows).to_csv(offsets_path, sep="\t", index=False)
        print(f"  Offsets: {offsets_path.name}")

    if n_fail:
        raise SystemExit(1)


def cmd_markers_export(
    bids_dir: Path, out_dir: Path, participant_label: list[str],
    ses: str | None, task: str | None, run: str | None,
    n_jobs: int, skip_bids_validation: bool,
) -> None:
    """Export events.tsv(s) to out_dir for manual editing."""
    from fnirs_pipe.pipeline.edit_markers import export_markers

    def _export_one(sub):
        return export_markers(
            bids_dir, sub, out_dir,
            ses=ses, task=task, run=run,
            validate=not skip_bids_validation,
        )

    _report(_run_parallel(_export_one, participant_label, n_jobs))


def cmd_markers_apply(
    bids_dir: Path, derivatives_dir: Path, participant_label: list[str],
    ses: str | None, task: str | None, run: str | None,
    tsv: Path | None, shift: float | None, set_duration: float | None,
    rename: list[str] | None, n_jobs: int, skip_bids_validation: bool,
) -> None:
    """Apply marker edits to runs and write to derivatives/marker_edited/."""
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
            bids_dir, derivatives_dir, sub,
            ses=ses, task=task, run=run,
            tsv=tsv, shift=shift, set_duration=set_duration, rename=rename,
            validate=not skip_bids_validation,
        )

    _report(_run_parallel(_apply_one, participant_label, n_jobs))


def _add_selection(sp) -> None:
    sp.add_argument("--participant-label", nargs="+", action="extend", required=True,
                    help="Subject ID(s) to process.")
    sp.add_argument("--ses",  default=None, help="Session label.")
    sp.add_argument("--task", default=None, help="Task label.")
    sp.add_argument("--run",  default=None, help="Run label.")
    sp.add_argument("--n-jobs", type=int, default=1, help="Parallel subject jobs.")
    sp.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-prep",
        description="fNIRS data preparation: cropping, alignment and marker editing.",
    )
    sub = p.add_subparsers(required=True)

    crop = sub.add_parser("crop", help="Crop raw SNIRFs to a time window or segments.")
    crop.add_argument("bids_dir",        type=Path, help="BIDS dataset root.")
    crop.add_argument("derivatives_dir", type=Path, help="Derivatives output directory.")
    _add_selection(crop)
    crop.add_argument("--tmin", type=float, default=None, help="Start time in seconds (single segment).")
    crop.add_argument("--tmax", type=float, default=None, help="End time in seconds (single segment).")
    crop.add_argument("--segments-path", type=Path, default=None,
                      help="TSV with onset/duration columns defining segments to keep.")
    crop.add_argument("--combine", action=argparse.BooleanOptionalAction, default=False,
                      help="Concatenate multi-segment output into one file.")
    crop.set_defaults(func=cmd_crop)

    align = sub.add_parser("align", help="Align multi-subject recordings by shared trigger.")
    align.add_argument("bids_dir",        type=Path, help="BIDS dataset root.")
    align.add_argument("derivatives_dir", type=Path, help="Derivatives output directory.")
    align.add_argument("--group-csv", type=Path, required=True,
                       help="CSV with group_id, subject_id, task columns.")
    align.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
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
    app.add_argument("derivatives_dir", type=Path, help="Derivatives output directory.")
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
