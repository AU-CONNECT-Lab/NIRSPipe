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
