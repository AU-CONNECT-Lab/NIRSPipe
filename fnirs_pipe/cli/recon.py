"""fnirs-recon: convert raw snirf files to BIDS format."""

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="fnirs-recon",
    help="Convert a raw snirf file to BIDS format (one subject at a time).",
    pretty_exceptions_show_locals=False,
)


@app.command()
def main(
    input_file: Annotated[Path, typer.Argument(help="Input snirf file.")],
    bids_dir:   Annotated[Path, typer.Argument(help="Output BIDS dataset directory.")],

    # required BIDS metadata
    subject: Annotated[str, typer.Option("--subject", help="Subject label, e.g. 01.")],
    task:    Annotated[str, typer.Option("--task",    help="Task label, e.g. tapping.")],

    # optional BIDS metadata
    session:   Annotated[Optional[str], typer.Option("--session", help="Session label. Omit if dataset has no session layer.")] = None,
    run:       Annotated[Optional[str], typer.Option("--run",     help="Run index, e.g. 01.")] = None,
    overwrite: Annotated[bool,          typer.Option("--overwrite/--no-overwrite")] = False,
):
    """Convert a raw snirf file to BIDS format.

    Run once per subject. For multiple subjects, loop over this command in a shell script.

    Example:
      fnirs-recon sub01.snirf /data/bids --subject 01 --task tapping
    """
    import mne
    import mne_bids

    if not input_file.exists():
        typer.echo(f"ERROR: input file not found: {input_file}", err=True)
        raise typer.Exit(1)

    raw = mne.io.read_raw_snirf(str(input_file), preload=False)

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        task=task,
        session=session,
        run=run,
        root=bids_dir,
        datatype="nirs",
        suffix="nirs",
        extension=".snirf",
    )

    mne_bids.write_raw_bids(raw, bids_path=bids_path, overwrite=overwrite, format="auto")
    typer.echo(f"BIDS output written to: {bids_dir}")
