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


@app.command()
def raw(
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe output directory.")],
    participant_label: Annotated[str, typer.Argument(help="Subject ID, e.g. '01'.")],
    session_label: Annotated[Optional[str], typer.Option("--session-label", help="Session label.")] = None,
    task_label: Annotated[Optional[str], typer.Option("--task-label", help="Task label.")] = None,
    sci_threshold: Annotated[float, typer.Option("--sci-threshold", help="SCI threshold for pre-highlighting bad channels.")] = 0.8,
    port: Annotated[int, typer.Option("--port", help="Local server port.")] = 5052,
) -> None:
    """Launch interactive raw QC viewer with section ratings and channel decisions."""
    from fnirs_pipe.qc.rating.app import RawRatingApp

    name_parts = [f"sub-{participant_label}"]
    if session_label:
        name_parts.append(f"ses-{session_label}")
    if task_label:
        name_parts.append(f"task-{task_label}")
    html_path = output_dir / ("_".join(name_parts) + "_raw.html")

    if not html_path.exists():
        typer.echo(f"Error: raw report not found: {html_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Launching raw viewer: {html_path.name} ...")
    RawRatingApp(html_path, output_dir, sci_threshold).run(port=port)


@app.command()
def hyper(
    output_dir: Annotated[Path, typer.Argument(help="fnirs-pipe output directory.")],
    group_id: Annotated[str, typer.Argument(help="Group ID, e.g. 'A'.")],
    task_label: Annotated[str, typer.Argument(help="Task label, e.g. 'tapping'.")],
    sci_threshold: Annotated[float, typer.Option("--sci-threshold", help="SCI threshold.")] = 0.8,
    port: Annotated[int, typer.Option("--port", help="Local server port.")] = 5053,
) -> None:
    """Launch interactive hyperscanning QC viewer with section ratings and channel decisions."""
    from fnirs_pipe.qc.rating.app import HyperRatingApp

    html_path = output_dir / f"group-{group_id}_task-{task_label}_desc-hyperraw_nirs.html"
    if not html_path.exists():
        typer.echo(f"Error: hyper report not found: {html_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Launching hyper viewer: {html_path.name} ...")
    HyperRatingApp(html_path, output_dir, sci_threshold).run(port=port)
