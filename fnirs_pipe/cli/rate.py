"""fnirs-rate CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="fnirs-rate",
    help="Interactive QC review for fNIRS data: rating, individual viewer, hyperscanning viewer.",
    pretty_exceptions_show_locals=False,
)


def _discover_subjects(output_dir: Path) -> list[str]:
    return sorted(
        d.name[4:] for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )


@app.command()
def rate(
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe output directory.")],
    participant_label: Annotated[
        Optional[list[str]],
        typer.Option("--participant-label", help="Subject ID(s) to open. Default: all found."),
    ] = None,
    port: Annotated[int, typer.Option("--port", help="Local server port.")] = 8765,
) -> None:
    """Launch QC rating interface for fnirs-pipe reports."""
    from fnirs_pipe.qc.rating.app import FNIRSRatingApp

    subjects = participant_label or _discover_subjects(output_dir)
    if not subjects:
        typer.echo("No subjects found in output directory.", err=True)
        raise typer.Exit(1)

    FNIRSRatingApp(output_dir, subjects).run(port=port)


# @app.command(name="raw-indiv")
# def raw_indiv(
#     bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root.")],
#     output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory.")],
#     participant_label: Annotated[str, typer.Argument(help="Subject ID to inspect, e.g. '01'.")],
#     session_label: Annotated[Optional[list[str]], typer.Option("--session-label", help="Session label(s) to include.")] = None,
#     task_label: Annotated[Optional[list[str]], typer.Option("--task-label", help="Task label(s) to include.")] = None,
#     port: Annotated[int, typer.Option("--port", help="Local server port.")] = 5050,
#     skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
# ) -> None:
#     """Launch interactive Flask viewer for a single participant's raw fNIRS data."""
#     from fnirs_pipe.io.bids import get_layout, get_nirs_files
#     from fnirs_pipe.qc.app import launch

#     layout = get_layout(bids_dir, validate=not skip_bids_validation)
#     sessions = session_label or [None]
#     tasks    = task_label    or [None]

#     all_runs: list[dict] = []
#     for session in sessions:
#         for task in tasks:
#             files = get_nirs_files(layout, subject=participant_label, session=session, task=task)
#             for f in files:
#                 entities = layout.parse_file_entities(str(f))
#                 actual_ses  = entities.get("session")
#                 actual_task = entities.get("task")
#                 actual_run  = entities.get("run")

#                 parts = [f"sub-{participant_label}"]
#                 if actual_ses:  parts.append(f"ses-{actual_ses}")
#                 if actual_task: parts.append(f"task-{actual_task}")
#                 if actual_run:  parts.append(f"run-{actual_run}")
#                 label = "_".join(parts)

#                 snirf_p = Path(f)
#                 events_p = snirf_p.parent / (snirf_p.name.replace("_nirs.snirf", "_events.tsv"))
#                 all_runs.append({
#                     "label":       label,
#                     "snirf_path":  str(f),
#                     "events_path": str(events_p) if events_p.exists() else None,
#                     "session":     actual_ses,
#                     "task":        actual_task,
#                 })

#     if not all_runs:
#         typer.echo(f"Error: no SNIRF files found for sub-{participant_label}.", err=True)
#         raise typer.Exit(1)

#     typer.echo(f"Launching viewer for sub-{participant_label} ({len(all_runs)} run(s)) ...")
#     out_dir = output_dir / f"sub-{participant_label}" / "nirs"
#     launch(all_runs, out_dir, port=port)


# @app.command(name="raw-hyper")
# def raw_hyper(
#     bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root.")],
#     output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe derivatives directory.")],
#     pairs_csv: Annotated[Path, typer.Option("--pairs-csv", help="CSV with columns: group_id, subject_id, task.")],
#     group_id: Annotated[Optional[str], typer.Option("--group-id", help="Process only this group_id.")] = None,
#     port: Annotated[int, typer.Option("--port", help="Local server port.")] = 5051,
#     skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
# ) -> None:
#     """Launch interactive Flask viewer for hyperscanning raw fNIRS data (not yet implemented)."""
#     raise NotImplementedError(
#         f"fnirs-rate raw-hyper is not yet implemented "
#         f"(bids={bids_dir}, output={output_dir}, pairs={pairs_csv}, "
#         f"group={group_id}, port={port}, skip_val={skip_bids_validation})."
#     )
