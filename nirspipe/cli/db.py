"""nirspipe-log CLI (argparse): database management for nirspipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from nirspipe import __version__

from nirspipe.utils import job_db as _db


def _default_db(output_dir: Path) -> Path:
    return output_dir / "logs" / "nirspipe.db"


def cmd_merge(output_dir: Path, db_path: Path | None) -> None:
    """Merge every finished execution's JSONL logs into the SQLite database, then archive them."""
    db = db_path or _default_db(output_dir)
    files, rows = _db.merge_jsonl(db)
    print(f"Merged {files} files, {rows} rows -> {db}")


def cmd_rebuild(output_dir: Path, db_path: Path | None) -> None:
    """Build a new database from every JSONL log, archived ones included."""
    db = db_path or _default_db(output_dir)
    print(f"Rebuilt from the logs -> {_db.rebuild_db(db)} ({db} left as it was)")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-log",
        description="Merge JSONL run logs into the nirspipe SQLite database.",
    )
    p.add_argument("--version", action="version", version=f"nirspipe-log {__version__}")
    sub = p.add_subparsers(required=True)

    m = sub.add_parser("merge", help="Merge every finished execution's JSONL logs under logs/json/ "
                                     "into the SQLite database and move them to archived/, so a "
                                     "second merge adds nothing. The database is backed up to "
                                     "logs/backup/ first.")
    m.add_argument("output_dir", type=Path, help="Pipeline output directory")
    m.add_argument("--db-path", type=Path, default=None, help="Override default DB path")
    m.set_defaults(func=cmd_merge)

    r = sub.add_parser("rebuild", help="Build a new, timestamped database from every JSONL log, "
                                       "archived ones included. The existing database is not "
                                       "touched and no log is moved.")
    r.add_argument("output_dir", type=Path, help="Pipeline output directory")
    r.add_argument("--db-path", type=Path, default=None, help="Override default DB path")
    r.set_defaults(func=cmd_rebuild)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
