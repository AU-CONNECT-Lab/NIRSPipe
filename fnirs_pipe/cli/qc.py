"""fnirs-qc CLI — quality control for fNIRS data."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe.utils.logging import get_logger, setup_logging

setup_logging()

app = typer.Typer(
    name="fnirs-qc",
    help="fNIRS quality control: interactive viewer and group-level reports.",
    pretty_exceptions_show_locals=False,
)

logger = get_logger("cli.qc")


@app.command()
def prep_raw(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root")],
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory")],
    participant_label: Annotated[str, typer.Argument(help="Subject ID to inspect, e.g. '01'")],
    session_label: Annotated[Optional[list[str]], typer.Option("--session-label", help="Session label(s) to include.")] = None,
    task_label: Annotated[Optional[list[str]], typer.Option("--task-label", help="Task label(s) to include.")] = None,
    sci_threshold: Annotated[float, typer.Option("--sci-threshold", help="SCI pass/fail threshold for bad channel detection.")] = 0.8,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
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
        typer.echo(f"Error: no SNIRF files found for sub-{participant_label}.", err=True)
        raise typer.Exit(1)

    # Generate one HTML per (session, task) group before launching viewer
    for (ses, task), group_runs in groups.items():
        name_parts = [f"sub-{participant_label}"]
        if ses:  name_parts.append(f"ses-{ses}")
        if task: name_parts.append(f"task-{task}")
        html_path = output_dir / ("_".join(name_parts) + "_desc-raw_nirs.html")
        typer.echo(f"Generating raw QC report: {html_path.name} ...")
        try:
            build_prep_raw_report(group_runs, html_path, sci_threshold=sci_threshold)
            typer.echo(f"  -> {html_path}")
        except Exception as exc:
            logger.exception("Raw report generation failed for %s", html_path.name)
            typer.echo(f"  [error] {exc}", err=True)

    typer.echo(f"Done. {len(all_runs)} run(s) processed.")


@app.command()
def hyper_raw(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root")],
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory")],
    pairs_csv: Annotated[Path, typer.Option("--pairs-csv", help=(
        "CSV with columns: group_id, subject_id, task. "
        "Each unique (group_id, task) pair is processed as one session."
    ))],
    group_id: Annotated[Optional[str], typer.Option("--group-id", help=(
        "Process only this group_id. Omit to process all groups."
    ))] = None,
    sci_threshold: Annotated[float, typer.Option("--sci-threshold", help=(
        "SCI pass/fail threshold for channel quality comparison."
    ))] = 0.80,
    coherence_fmin: Annotated[float, typer.Option("--fmin", help=(
        "Lower bound (Hz) for coherence frequency band."
    ))] = 0.01,
    coherence_fmax: Annotated[float, typer.Option("--fmax", help=(
        "Upper bound (Hz) for coherence frequency band."
    ))] = 0.10,
    normalize: Annotated[bool, typer.Option("--normalize/--no-normalize", help=(
        "Z-score each channel per subject after alignment. "
        "Useful when subjects have very different signal amplitudes."
    ))] = False,
    no_align: Annotated[bool, typer.Option("--no-align", help=(
        "Skip trigger-based alignment; trim all recordings to the shortest duration. "
        "Use for resting-state data without shared triggers."
    ))] = False,
    session_label: Annotated[Optional[list[str]], typer.Option("--session-label", help="Session label(s) to include.")] = None,
    task_label: Annotated[Optional[list[str]], typer.Option("--task-label", help="Task label(s) to include.")] = None,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Generate hyperscanning raw QC report from BIDS raw data."""
    from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
    from fnirs_pipe.pipeline.hyperscanning import (
        _raw_to_haemo,
        align_recordings,
        compute_group_iqm_raw,
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
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)

    if group_id is not None:
        groups = {k: v for k, v in groups.items() if k[0] == group_id}
        if not groups:
            typer.echo(f"[error] group_id '{group_id}' not found in CSV", err=True)
            raise typer.Exit(1)

    if task_label is not None:
        groups = {k: v for k, v in groups.items() if k[1] in task_label}
        if not groups:
            typer.echo(f"[error] task_label {task_label} not found in CSV", err=True)
            raise typer.Exit(1)

    ses = session_label[0] if session_label else None

    n_total = len(groups)
    typer.echo(f"Processing {n_total} group session(s)...")

    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        label = f"{gid}/{task}"
        typer.echo(f"  -> {label} ({len(members)} subjects)")
        try:
            raws_cw = load_group_raw_bids(bids_dir, members)
            iqm_data = compute_group_iqm_raw(members, raws_cw, sci_threshold, output_dir)
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
                iqm_data=iqm_data,
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
            typer.echo(f"     report -> {report_path}")
            n_ok += 1
        except MissingDerivativesError as exc:
            typer.echo(f"     [skip] {exc}", err=True)
            n_fail += 1
        except AlignmentError as exc:
            typer.echo(f"     [skip] alignment failed: {exc}", err=True)
            n_fail += 1
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            typer.echo(f"     [error] unexpected error: {exc}", err=True)
            n_fail += 1

    typer.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise typer.Exit(1)


@app.command()
def group_raw(
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory (contains sub-*/nirs/ IQM JSONs)")],
) -> None:
    """Aggregate per-subject prep-raw IQMs into group_nirs.tsv + group_nirs.html."""
    from fnirs_pipe.qc.group_writer import build_group_raw_report

    path = build_group_raw_report(output_dir)
    typer.echo(f"report -> {path}")


@app.command()
def group_hyper_raw(
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory (contains group-*/nirs/ IQM JSONs)")],
) -> None:
    """Aggregate per-group hyper-raw IQMs into group_hyper_nirs.tsv + group_hyper_nirs.html."""
    from fnirs_pipe.qc.group_writer import build_group_hyper_raw_report

    path = build_group_hyper_raw_report(output_dir)
    typer.echo(f"report -> {path}")


@app.command()
def window_raw(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root")],
    output_dir: Annotated[Path, typer.Argument(help="QC output directory (where group_nirs.html lives)")],
    task_label: Annotated[str, typer.Option("--task-label", help="BIDS task label (one task at a time).")],
    tstart: Annotated[float, typer.Option("--tstart", help="Window start time (s)")],
    tend: Annotated[float, typer.Option("--tend",   help="Window end time (s)")],
    participant_label: Annotated[Optional[list[str]], typer.Option("--participant-label", help="Subject(s) to include (default: all).")] = None,
    session_label: Annotated[Optional[list[str]], typer.Option("--session-label", help="Session label(s) to include.")] = None,
    align: Annotated[str, typer.Option("--align", help="t=0 origin: 'none' = recording start, 'trigger' = first matching annotation.")] = "none",
    trigger_name: Annotated[Optional[str], typer.Option("--trigger-name", help="Annotation description used when --align trigger.")] = None,
    name: Annotated[Optional[str], typer.Option("--name", help="Output suffix (default: window-{tstart}-{tend}).")] = None,
    sci_threshold: Annotated[float, typer.Option("--sci-threshold", help="SCI threshold for bad-channel detection.")] = 0.8,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Crop each subject's raw to [tstart, tend] + aggregate IQM into a windowed group report."""
    from fnirs_pipe.qc.window_writer import build_window_raw_report

    path = build_window_raw_report(
        bids_dir=bids_dir, output_dir=output_dir,
        task=task_label, tstart=tstart, tend=tend,
        participant_label=participant_label, session_label=session_label,
        align=align, trigger_name=trigger_name, name=name,
        sci_threshold=sci_threshold, skip_bids_validation=skip_bids_validation,
    )
    typer.echo(f"report -> {path}")


@app.command()
def hyper_post(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root")],
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory")],
    pairs_csv: Annotated[Path, typer.Option("--pairs-csv", help=(
        "CSV with columns: group_id, subject_id, task. "
        "Each unique (group_id, task) pair is processed as one session."
    ))],
    group_id: Annotated[Optional[str], typer.Option("--group-id", help=(
        "Process only this group_id. Omit to process all groups."
    ))] = None,
    roi_mapping: Annotated[Optional[Path], typer.Option("--roi-mapping", help=(
        "JSON file mapping ROI labels to lists of channel names. "
        "Used for ROI-level WTC. Optional."
    ))] = None,
    wtc_fmin: Annotated[float, typer.Option("--wtc-fmin", help=(
        "Lower bound (Hz) for WTC frequency axis."
    ))] = 0.004,
    wtc_fmax: Annotated[float, typer.Option("--wtc-fmax", help=(
        "Upper bound (Hz) for WTC frequency axis."
    ))] = 0.20,
    isc_threshold: Annotated[float, typer.Option("--isc-threshold", help=(
        "Minimum mean ISC to draw an arc in the connectivity circle."
    ))] = 0.3,
    normalize: Annotated[bool, typer.Option("--normalize/--no-normalize", help=(
        "Z-score each channel per subject after alignment. "
        "Useful when subjects have very different signal amplitudes."
    ))] = False,
    no_align: Annotated[bool, typer.Option("--no-align", help=(
        "Skip trigger-based alignment; trim all recordings to the shortest duration. "
        "Use for resting-state data without shared triggers."
    ))] = False,
    session_label: Annotated[Optional[list[str]], typer.Option("--session-label", help="Session label(s) to include.")] = None,
    task_label: Annotated[Optional[list[str]], typer.Option("--task-label", help="Task label(s) to include.")] = None,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Generate hyperscanning post-processing QC report (WTC, ISC, connectivity)."""
    import json

    from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings,
        load_group_haemo,
        normalize_raws,
        parse_group_csv,
        trim_to_shortest,
    )
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    try:
        groups = parse_group_csv(pairs_csv)
    except GroupCSVError as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)

    if group_id is not None:
        groups = {k: v for k, v in groups.items() if k[0] == group_id}
        if not groups:
            typer.echo(f"[error] group_id '{group_id}' not found in CSV", err=True)
            raise typer.Exit(1)

    if task_label is not None:
        groups = {k: v for k, v in groups.items() if k[1] in task_label}
        if not groups:
            typer.echo(f"[error] task_label {task_label} not found in CSV", err=True)
            raise typer.Exit(1)

    # TODO: optionally auto-generate this roi.json from the montage via
    # mne_nirs.io.fold_channel_specificity (needs fOLD Excel DB + MNE_NIRS_FOLD_PATH).
    roi_map: dict[str, list[str]] | None = None
    if roi_mapping is not None:
        try:
            roi_map = json.loads(roi_mapping.read_text())
        except Exception as exc:
            typer.echo(f"[error] failed to load ROI mapping: {exc}", err=True)
            raise typer.Exit(1)

    n_total = len(groups)
    typer.echo(f"Processing {n_total} group session(s)...")

    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        label = f"{gid}/{task}"
        typer.echo(f"  -> {label} ({len(members)} subjects)")
        try:
            raws = load_group_haemo(output_dir, members)
            if no_align:
                aligned_raws, offsets = trim_to_shortest(raws)
            else:
                aligned_raws, offsets = align_recordings(raws, task)
            if normalize:
                aligned_raws = normalize_raws(aligned_raws)
            report_path = build_hyper_post_report(
                group_id=gid,
                task=task,
                group=members,
                aligned_raws=aligned_raws,
                offsets=offsets,
                output_dir=output_dir,
                roi_map=roi_map,
                wtc_fmin=wtc_fmin,
                wtc_fmax=wtc_fmax,
                isc_threshold=isc_threshold,
            )
            typer.echo(f"     report -> {report_path}")
            n_ok += 1
        except MissingDerivativesError as exc:
            typer.echo(f"     [skip] {exc}", err=True)
            n_fail += 1
        except AlignmentError as exc:
            typer.echo(f"     [skip] alignment failed: {exc}", err=True)
            n_fail += 1
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            typer.echo(f"     [error] unexpected error: {exc}", err=True)
            n_fail += 1

    typer.echo(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise typer.Exit(1)
