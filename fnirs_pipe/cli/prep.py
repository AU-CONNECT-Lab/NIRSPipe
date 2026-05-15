"""fnirs-prep CLI — data preparation utilities (marker editing, crop, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe.utils.logging import get_logger

app = typer.Typer(
    name="fnirs-prep",
    help="fNIRS data preparation: marker editing and related utilities.",
    pretty_exceptions_show_locals=False,
)

markers_app = typer.Typer(help="Edit markers in SNIRF files.")
app.add_typer(markers_app, name="edit-markers")

logger = get_logger("cli.prep")


@markers_app.command("export")
def markers_export(
    bids_dir: Annotated[Path, typer.Argument(help="BIDS dataset root.")],
    out_dir:  Annotated[Path, typer.Argument(help="Directory to write the exported events TSV.")],
    sub:  Annotated[str,           typer.Option("--sub",  help="Subject ID, e.g. '01'.")],
    ses:  Annotated[Optional[str], typer.Option("--ses",  help="Session label.")] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task label.")] = None,
    run:  Annotated[Optional[str], typer.Option("--run",  help="Run label.")] = None,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Export a run's events.tsv to out_dir for manual editing."""
    from fnirs_pipe.pipeline.edit_markers import export_markers

    try:
        out_path = export_markers(
            bids_dir, sub, out_dir,
            ses=ses, task=task, run=run,
            validate=not skip_bids_validation,
        )
        typer.echo(f"Exported: {out_path}")
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)


@markers_app.command("apply")
def markers_apply(
    bids_dir:        Annotated[Path, typer.Argument(help="BIDS dataset root.")],
    derivatives_dir: Annotated[Path, typer.Argument(help="Derivatives output directory.")],
    sub:  Annotated[str,           typer.Option("--sub",  help="Subject ID, e.g. '01'.")],
    ses:  Annotated[Optional[str], typer.Option("--ses",  help="Session label.")] = None,
    task: Annotated[Optional[str], typer.Option("--task", help="Task label.")] = None,
    run:  Annotated[Optional[str], typer.Option("--run",  help="Run label.")] = None,
    tsv:          Annotated[Optional[Path],      typer.Option("--tsv",          help="Edited events TSV to apply.")] = None,
    shift:        Annotated[Optional[float],     typer.Option("--shift",        help="Shift all onsets by this many seconds (negative = earlier). Clipped to 0.")] = None,
    set_duration: Annotated[Optional[float],     typer.Option("--set-duration", help="Set all marker durations to this value (seconds).")] = None,
    rename:       Annotated[Optional[list[str]], typer.Option("--rename",       help="Rename marker: 'old:new'. Repeatable.")] = None,
    skip_bids_validation: Annotated[bool, typer.Option("--skip-bids-validation/--no-skip-bids-validation")] = False,
) -> None:
    """Apply marker edits to a run and write to derivatives/marker_edited/.

    Exactly one operation per call: --tsv, --shift, --set-duration, or --rename.
    """
    ops = [x for x in (tsv, shift, set_duration, rename) if x is not None]
    if not ops:
        typer.echo("[error] Specify one of: --tsv, --shift, --set-duration, --rename", err=True)
        raise typer.Exit(1)
    if len(ops) > 1:
        typer.echo("[error] Only one of --tsv, --shift, --set-duration, --rename can be used at a time.", err=True)
        raise typer.Exit(1)

    from fnirs_pipe.pipeline.edit_markers import apply_markers

    try:
        out_snirf = apply_markers(
            bids_dir, derivatives_dir, sub,
            ses=ses, task=task, run=run,
            tsv=tsv, shift=shift, set_duration=set_duration, rename=rename,
            validate=not skip_bids_validation,
        )
        typer.echo(f"Written: {out_snirf}")
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"[error] {exc}", err=True)
        raise typer.Exit(1)
