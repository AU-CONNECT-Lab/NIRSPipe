"""fnirs-qc CLI (argparse) — quality control for fNIRS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fnirs_pipe.utils.logging import get_logger, setup_logging

setup_logging()

logger = get_logger("cli.qc")


def cmd_prep_raw(
    bids_dir: Path, output_dir: Path, participant_label: str,
    session_label: list[str] | None, task_label: list[str] | None,
    sci_threshold: float, skip_bids_validation: bool,
) -> None:
    """Generate static raw QC report for a single participant."""
    from collections import defaultdict

    from fnirs_pipe.io.bids import get_layout, get_nirs_files
    from fnirs_pipe.qc.prep_raw_report import build_prep_raw_report

    layout = get_layout(bids_dir, validate=not skip_bids_validation)
    sessions = session_label or [None]
    tasks    = task_label    or [None]

    all_runs: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for session in sessions:
        for task in tasks:
            files = get_nirs_files(layout, subject=participant_label, session=session, task=task)
            for f in files:
                entities = layout.parse_file_entities(str(f))
                actual_ses = entities.get("session")
                actual_task = entities.get("task")
                actual_run = entities.get("run")

                parts = [f"sub-{participant_label}"]
                if actual_ses:  parts.append(f"ses-{actual_ses}")
                if actual_task: parts.append(f"task-{actual_task}")
                if actual_run:  parts.append(f"run-{actual_run}")
                label = "_".join(parts)

                snirf_p = Path(f)
                events_p = snirf_p.parent / (snirf_p.name.replace("_nirs.snirf", "_events.tsv"))
                run_dict = {
                    "label":       label,
                    "subject_id":  participant_label,
                    "snirf_path":  str(f),
                    "events_path": str(events_p) if events_p.exists() else None,
                    "session":     actual_ses,
                    "task":        actual_task,
                }
                all_runs.append(run_dict)
                groups[(actual_ses, actual_task)].append(run_dict)

    if not all_runs:
        print(f"Error: no SNIRF files found for sub-{participant_label}.", file=sys.stderr)
        raise SystemExit(1)

    for (ses, task), group_runs in groups.items():
        name_parts = [f"sub-{participant_label}"]
        if ses:  name_parts.append(f"ses-{ses}")
        if task: name_parts.append(f"task-{task}")
        html_path = output_dir / ("_".join(name_parts) + "_desc-raw_nirs.html")
        print(f"Generating raw QC report: {html_path.name} ...")
        try:
            build_prep_raw_report(group_runs, html_path, sci_threshold=sci_threshold)
            print(f"  -> {html_path}")
        except Exception as exc:
            logger.exception("Raw report generation failed for %s", html_path.name)
            print(f"  [error] {exc}", file=sys.stderr)

    print(f"Done. {len(all_runs)} run(s) processed.")


def cmd_hyper_raw(
    bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
    sci_threshold: float, coherence_fmin: float, coherence_fmax: float,
    normalize: bool, no_align: bool,
    session_label: list[str] | None, task_label: list[str] | None,
    skip_bids_validation: bool,
) -> None:
    """Generate hyperscanning raw QC report from BIDS raw data."""
    from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
    from fnirs_pipe.pipeline.hyperscanning import (
        _raw_to_haemo,
        align_recordings,
        compute_group_sqm_raw,
        compute_pairwise_coherence,
        load_group_raw_bids,
        normalize_raws,
        parse_group_csv,
        trim_to_shortest,
    )
    from fnirs_pipe.qc.hyper_report import build_hyper_report

    try:
        groups = parse_group_csv(pairs_csv)
    except GroupCSVError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    if group_id is not None:
        groups = {k: v for k, v in groups.items() if k[0] == group_id}
        if not groups:
            print(f"[error] group_id '{group_id}' not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    if task_label is not None:
        groups = {k: v for k, v in groups.items() if k[1] in task_label}
        if not groups:
            print(f"[error] task_label {task_label} not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    ses = session_label[0] if session_label else None

    n_total = len(groups)
    print(f"Processing {n_total} group session(s)...")

    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        label = f"{gid}/{task}"
        print(f"  -> {label} ({len(members)} subjects)")
        try:
            raws_cw = load_group_raw_bids(bids_dir, members)
            sqm_data = compute_group_sqm_raw(members, raws_cw, sci_threshold, output_dir)
            raws_haemo = {sid: _raw_to_haemo(r) for sid, r in raws_cw.items()}
            if no_align:
                aligned_raws, offsets = trim_to_shortest(raws_haemo)
            else:
                aligned_raws, offsets = align_recordings(raws_haemo, task)
            if normalize:
                aligned_raws = normalize_raws(aligned_raws)
            coherence_df = compute_pairwise_coherence(
                aligned_raws, fmin=coherence_fmin, fmax=coherence_fmax
            )
            report_path = build_hyper_report(
                group_id=gid,
                task=task,
                group=members,
                sqm_data=sqm_data,
                aligned_raws=aligned_raws,
                offsets=offsets,
                raw_raws=raws_haemo,
                coherence_df=coherence_df,
                output_dir=output_dir,
                session=ses,
                sci_threshold=sci_threshold,
                coherence_fmin=coherence_fmin,
                coherence_fmax=coherence_fmax,
            )
            print(f"     report -> {report_path}")
            n_ok += 1
        except MissingDerivativesError as exc:
            print(f"     [skip] {exc}", file=sys.stderr)
            n_fail += 1
        except AlignmentError as exc:
            print(f"     [skip] alignment failed: {exc}", file=sys.stderr)
            n_fail += 1
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            print(f"     [error] unexpected error: {exc}", file=sys.stderr)
            n_fail += 1

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise SystemExit(1)


def cmd_group_raw(output_dir: Path) -> None:
    """Aggregate per-subject prep-raw SQMs into group_nirs.tsv + group_nirs.html."""
    from fnirs_pipe.qc.group_writer import build_group_raw_report

    path = build_group_raw_report(output_dir)
    print(f"report -> {path}")


def cmd_group_hyper_raw(output_dir: Path) -> None:
    """Aggregate per-group hyper-raw SQMs into group_hyper_nirs.tsv + group_hyper_nirs.html."""
    from fnirs_pipe.qc.group_writer import build_group_hyper_raw_report

    path = build_group_hyper_raw_report(output_dir)
    print(f"report -> {path}")


def cmd_window_raw(
    bids_dir: Path, output_dir: Path, task_label: str, tstart: float, tend: float,
    participant_label: list[str] | None, session_label: list[str] | None,
    align: str, trigger_name: str | None, name: str | None,
    sci_threshold: float, skip_bids_validation: bool,
) -> None:
    """Crop each subject's raw to [tstart, tend] + aggregate SQM into a windowed group report."""
    from fnirs_pipe.qc.window_writer import build_window_raw_report

    path = build_window_raw_report(
        bids_dir=bids_dir, output_dir=output_dir,
        task=task_label, tstart=tstart, tend=tend,
        participant_label=participant_label, session_label=session_label,
        align=align, trigger_name=trigger_name, name=name,
        sci_threshold=sci_threshold, skip_bids_validation=skip_bids_validation,
    )
    print(f"report -> {path}")


def cmd_hyper_post(
    bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
    roi_mapping: Path | None, wtc_fmin: float, wtc_fmax: float, isc_threshold: float,
    normalize: bool, no_align: bool,
    session_label: list[str] | None, task_label: list[str] | None,
    skip_bids_validation: bool,
) -> None:
    """Generate hyperscanning post-processing QC report (WTC, ISC, connectivity)."""
    import json

    from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings,
        load_group_haemo,
        load_group_sqm,
        normalize_raws,
        parse_group_csv,
        trim_to_shortest,
    )
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    try:
        groups = parse_group_csv(pairs_csv)
    except GroupCSVError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    if group_id is not None:
        groups = {k: v for k, v in groups.items() if k[0] == group_id}
        if not groups:
            print(f"[error] group_id '{group_id}' not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    if task_label is not None:
        groups = {k: v for k, v in groups.items() if k[1] in task_label}
        if not groups:
            print(f"[error] task_label {task_label} not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    roi_map: dict[str, list[str]] | None = None
    if roi_mapping is not None:
        try:
            roi_map = json.loads(roi_mapping.read_text())
        except Exception as exc:
            print(f"[error] failed to load ROI mapping: {exc}", file=sys.stderr)
            raise SystemExit(1)

    n_total = len(groups)
    print(f"Processing {n_total} group session(s)...")

    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        label = f"{gid}/{task}"
        print(f"  -> {label} ({len(members)} subjects)")
        try:
            raws = load_group_haemo(output_dir, members)
            if no_align:
                aligned_raws, offsets = trim_to_shortest(raws)
            else:
                aligned_raws, offsets = align_recordings(raws, task)
            if normalize:
                aligned_raws = normalize_raws(aligned_raws)
            bad_channels = {
                sid: sqm.get("bad_channels", [])
                for sid, sqm in load_group_sqm(output_dir, members).items()
            }
            report_path = build_hyper_post_report(
                group_id=gid,
                task=task,
                group=members,
                aligned_raws=aligned_raws,
                offsets=offsets,
                output_dir=output_dir,
                roi_map=roi_map,
                bad_channels=bad_channels,
                wtc_fmin=wtc_fmin,
                wtc_fmax=wtc_fmax,
                isc_threshold=isc_threshold,
            )
            print(f"     report -> {report_path}")
            n_ok += 1
        except MissingDerivativesError as exc:
            print(f"     [skip] {exc}", file=sys.stderr)
            n_fail += 1
        except AlignmentError as exc:
            print(f"     [skip] alignment failed: {exc}", file=sys.stderr)
            n_fail += 1
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            print(f"     [error] unexpected error: {exc}", file=sys.stderr)
            n_fail += 1

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise SystemExit(1)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-qc",
        description="fNIRS quality control: individual, hyperscanning and group-level reports.",
    )
    sub = p.add_subparsers(required=True)

    pr = sub.add_parser("prep-raw", help="Static raw QC report for a single participant.")
    pr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    pr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    pr.add_argument("participant_label", help="Subject ID to inspect, e.g. '01'")
    pr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    pr.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    pr.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI pass/fail threshold for bad channel detection.")
    pr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    pr.set_defaults(func=cmd_prep_raw)

    hr = sub.add_parser("hyper-raw", help="Hyperscanning raw QC report from BIDS raw data.")
    hr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    hr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    hr.add_argument("--pairs-csv", type=Path, required=True,
                    help="CSV with columns: group_id, subject_id, task. "
                         "Each unique (group_id, task) pair is processed as one session.")
    hr.add_argument("--group-id", default=None,
                    help="Process only this group_id. Omit to process all groups.")
    hr.add_argument("--sci-threshold", type=float, default=0.80,
                    help="SCI pass/fail threshold for channel quality comparison.")
    hr.add_argument("--fmin", dest="coherence_fmin", type=float, default=0.01,
                    help="Lower bound (Hz) for coherence frequency band.")
    hr.add_argument("--fmax", dest="coherence_fmax", type=float, default=0.10,
                    help="Upper bound (Hz) for coherence frequency band.")
    hr.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                    help="Z-score each channel per subject after alignment.")
    hr.add_argument("--no-align", action="store_true",
                    help="Skip trigger-based alignment; trim all recordings to the shortest duration.")
    hr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    hr.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    hr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    hr.set_defaults(func=cmd_hyper_raw)

    gr = sub.add_parser("group-raw", help="Aggregate per-subject prep-raw SQMs.")
    gr.add_argument("output_dir", type=Path,
                    help="fnirs-pipe derivatives directory (contains sub-*/nirs/ SQM JSONs)")
    gr.set_defaults(func=cmd_group_raw)

    ghr = sub.add_parser("group-hyper-raw", help="Aggregate per-group hyper-raw SQMs.")
    ghr.add_argument("output_dir", type=Path,
                     help="fnirs-pipe derivatives directory (contains group-*/nirs/ SQM JSONs)")
    ghr.set_defaults(func=cmd_group_hyper_raw)

    wr = sub.add_parser("window-raw", help="Windowed group raw QC report over [tstart, tend].")
    wr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    wr.add_argument("output_dir", type=Path, help="QC output directory (where group_nirs.html lives)")
    wr.add_argument("--task-label", required=True, help="BIDS task label (one task at a time).")
    wr.add_argument("--tstart", type=float, required=True, help="Window start time (s)")
    wr.add_argument("--tend",   type=float, required=True, help="Window end time (s)")
    wr.add_argument("--participant-label", nargs="+", action="extend",
                    help="Subject(s) to include (default: all).")
    wr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    wr.add_argument("--align", default="none",
                    help="t=0 origin: 'none' = recording start, 'trigger' = first matching annotation.")
    wr.add_argument("--trigger-name", default=None,
                    help="Annotation description used when --align trigger.")
    wr.add_argument("--name", default=None, help="Output suffix (default: window-{tstart}-{tend}).")
    wr.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI threshold for bad-channel detection.")
    wr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    wr.set_defaults(func=cmd_window_raw)

    hp = sub.add_parser("hyper-post", help="Hyperscanning post QC report (WTC, ISC, connectivity).")
    hp.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    hp.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    hp.add_argument("--pairs-csv", type=Path, required=True,
                    help="CSV with columns: group_id, subject_id, task. "
                         "Each unique (group_id, task) pair is processed as one session.")
    hp.add_argument("--group-id", default=None,
                    help="Process only this group_id. Omit to process all groups.")
    hp.add_argument("--roi-mapping", type=Path, default=None,
                    help="JSON file mapping ROI labels to lists of channel names. For ROI-level WTC. Optional.")
    hp.add_argument("--wtc-fmin", type=float, default=0.004, help="Lower bound (Hz) for WTC frequency axis.")
    hp.add_argument("--wtc-fmax", type=float, default=0.20,  help="Upper bound (Hz) for WTC frequency axis.")
    hp.add_argument("--isc-threshold", type=float, default=0.3,
                    help="Minimum mean ISC to draw an arc in the connectivity circle.")
    hp.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                    help="Z-score each channel per subject after alignment.")
    hp.add_argument("--no-align", action="store_true",
                    help="Skip trigger-based alignment; trim all recordings to the shortest duration.")
    hp.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    hp.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    hp.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    hp.set_defaults(func=cmd_hyper_post)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
