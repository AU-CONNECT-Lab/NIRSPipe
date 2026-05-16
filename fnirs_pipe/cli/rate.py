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
