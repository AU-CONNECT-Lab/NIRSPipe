"""fnirs-gui CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="fnirs-gui",
    help="Dash-based GUI for interactive fNIRS data inspection and rating.",
    pretty_exceptions_show_locals=False,
)


@app.command()
def launch(
    port: Annotated[int, typer.Option("--port", help="Local server port.")] = 8050,
) -> None:
    """Launch the fnirs-gui Dash interface."""
    from fnirs_pipe.interface.app import launch as _launch
    _launch(port=port)
