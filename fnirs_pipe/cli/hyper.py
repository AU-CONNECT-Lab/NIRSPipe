"""fnirs-hyper CLI — hyperscanning group QC."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe.utils.logging import get_logger

app = typer.Typer(
    name="fnirs-hyper",
    help="Hyperscanning QC: alignment, channel quality, and inter-brain coherence.",
    pretty_exceptions_show_locals=False,
)

logger = get_logger("cli.hyper")


@app.command()
def run(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root")],
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory")],
    pairs_csv: Annotated[Path, typer.Option("--pairs-csv", help=(
        "CSV with columns: group_id, subject_id, task. "
        "Each unique (group_id, task) pair is processed as one session."
    ))],
    group_id: Annotated[Optional[str], typer.Option("--group-id", help=(
        "Process only this group_id (all tasks). Omit to process all groups."
    ))] = None,
    sci_threshold: Annotated[float, typer.Option("--sci-threshold", help=(
        "SCI pass/fail threshold for channel usability heatmap."
    ))] = 0.80,
    coherence_fmin: Annotated[float, typer.Option("--fmin", help=(
        "Lower bound (Hz) for coherence frequency band."
    ))] = 0.01,
    coherence_fmax: Annotated[float, typer.Option("--fmax", help=(
        "Upper bound (Hz) for coherence frequency band."
    ))] = 0.10,
) -> None:
    from fnirs_pipe.exceptions import AlignmentError, GroupCSVError, MissingDerivativesError
    from fnirs_pipe.qc.hyperscanning.align import align_recordings
    from fnirs_pipe.qc.hyperscanning.io import load_group_haemo, load_group_iqm, parse_group_csv
    from fnirs_pipe.qc.hyperscanning.metrics import compute_pairwise_coherence
    from fnirs_pipe.qc.hyperscanning.report import build_hyper_report

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
        typer.echo(f"  → {label} ({len(members)} subjects)")
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
            typer.echo(f"     report → {report_path}")
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
