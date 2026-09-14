"""fnirs-log CLI (argparse) — database management for fnirs-pipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from fnirs_pipe import __version__

from fnirs_pipe.utils import job_db as _db


def _default_db(output_dir: Path) -> Path:
    return output_dir / "logs" / "fnirs_pipe.db"


def cmd_merge(output_dir: Path, db_path: Path | None) -> None:
    """Merge all JSONL files under logs/json/ into the SQLite database."""
    db = db_path or _default_db(output_dir)
    files, rows = _db.merge_jsonl(db)
    print(f"Merged {files} files, {rows} rows -> {db}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-log",
        description="Merge JSONL run logs into the fnirs-pipe SQLite database.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-log {__version__}")
    sub = p.add_subparsers(required=True)

    m = sub.add_parser("merge", help="Merge JSONL logs under logs/json/ into the SQLite database.")
    m.add_argument("output_dir", type=Path, help="Pipeline output directory")
    m.add_argument("--db-path", type=Path, default=None, help="Override default DB path")
    m.set_defaults(func=cmd_merge)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
