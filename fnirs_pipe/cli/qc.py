"""fnirs-qc CLI — quality control for fNIRS data."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe.utils.logging import get_logger

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
        html_path = output_dir / ("_".join(name_parts) + "_raw.html")
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
    session_label: Annotated[Optional[list[str]], typer.Option("--session-label", help="Session label(s) to include.")] = None,
    task_label: Annotated[Optional[list[str]], typer.Option("--task-label", help="Task label(s) to include.")] = None,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Generate hyperscanning raw QC report (BIDS derivatives)."""
    from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings,
        compute_pairwise_coherence,
        load_group_haemo,
        load_group_iqm,
        parse_group_csv,
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

    n_total = len(groups)
    typer.echo(f"Processing {n_total} group session(s)...")

    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        label = f"{gid}/{task}"
        typer.echo(f"  -> {label} ({len(members)} subjects)")
        try:
            iqm_data = load_group_iqm(output_dir, members)
            raws = load_group_haemo(output_dir, members)
            aligned_raws, offsets = align_recordings(raws, task)
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
                coherence_df=coherence_df,
                output_dir=output_dir,
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
