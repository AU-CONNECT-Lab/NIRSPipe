"""fnirs-log CLI — database management for fnirs-pipe."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from fnirs_pipe.utils import job_db as _db

app = typer.Typer(
    name="fnirs-log",
    help="Merge JSONL run logs into the fnirs-pipe SQLite database.",
    pretty_exceptions_show_locals=False,
)


def _default_db(output_dir: Path) -> Path:
    return output_dir / "logs" / "fnirs_pipe.db"


@app.command("merge")
def merge(
    output_dir: Annotated[Path, typer.Argument(help="Pipeline output directory")],
    db_path: Annotated[Optional[Path], typer.Option(help="Override default DB path")] = None,
) -> None:
    """Merge all JSONL files under logs/json/ into the SQLite database."""
    db = db_path or _default_db(output_dir)
    files, rows = _db.merge_jsonl(db)
    typer.echo(f"Merged {files} files, {rows} rows → {db}")
