"""fnirs-rate CLI entry point (argparse)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fnirs_pipe import __version__

from fnirs_pipe.cli import _shared


def _discover_subjects(output_dir: Path) -> list[str]:
    return sorted(
        d.name[4:] for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )


def cmd_rate(output_dir: Path, participant_label: list[str] | None, port: int) -> None:
    """Launch QC rating interface for fnirs-pipe reports."""
    from fnirs_pipe.qc.rating.app import FNIRSRatingApp

    subjects = participant_label or _discover_subjects(output_dir)
    if not subjects:
        print("No subjects found in output directory.", file=sys.stderr)
        raise SystemExit(1)

    FNIRSRatingApp(output_dir, subjects).run(port=port)


def cmd_raw(
    output_dir: Path, participant_label: str,
    session_label: str | None, task_label: str | None,
    sci_threshold: float, port: int,
) -> None:
    """Launch interactive raw QC viewer with section ratings and channel decisions."""
    from fnirs_pipe.qc.rating.app import RawRatingApp

    name_parts = [f"sub-{participant_label}"]
    if session_label:
        name_parts.append(f"ses-{session_label}")
    if task_label:
        name_parts.append(f"task-{task_label}")
    html_path = (output_dir / f"sub-{participant_label}"
                 / ("_".join(name_parts) + "_desc-raw_nirs.html"))

    if not html_path.exists():
        print(f"Error: raw report not found: {html_path}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Launching raw viewer: {html_path.name} ...")
    RawRatingApp(html_path, output_dir, sci_threshold).run(port=port)


def cmd_hyper(
    output_dir: Path, group_id: str, task_label: str, pairs_csv: Path,
    session_label: str | None, sci_threshold: float, port: int,
) -> None:
    """Launch interactive hyperscanning QC viewer with section ratings and channel decisions."""
    from fnirs_pipe.pipeline.hyperscanning import parse_group_csv
    from fnirs_pipe.qc.rating.app import HyperRatingApp

    name_parts = [f"group-{group_id}"]
    if session_label:
        name_parts.append(f"ses-{session_label}")
    name_parts.append(f"task-{task_label}")
    fname = "_".join(name_parts) + "_desc-hyperraw_nirs.html"
    html_path = output_dir / f"group-{group_id}" / fname
    if not html_path.exists() and (output_dir / fname).exists():
        html_path = output_dir / fname      # a tree written before the group folder existed
    if not html_path.exists():
        print(f"Error: hyper report not found: {html_path}", file=sys.stderr)
        raise SystemExit(1)

    groups = parse_group_csv(pairs_csv)
    members = groups.get((group_id, task_label))
    if not members:
        print(f"Error: group_id '{group_id}' + task '{task_label}' not found in {pairs_csv}", file=sys.stderr)
        raise SystemExit(1)
    subject_ids = [e.subject_id for e in members]

    print(f"Launching hyper viewer: {html_path.name} ...")
    HyperRatingApp(html_path, output_dir, subject_ids, sci_threshold).run(port=port)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-rate",
        description="Interactive QC review for fNIRS data: rating, individual viewer, hyperscanning viewer.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-rate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("rate", help="Launch QC rating interface.")
    pr.add_argument("output_dir", type=Path, help="fnirs-pipe output directory.")
    pr.add_argument("--participant-label", nargs="+", action="extend",
                    type=_shared.BidsLabel,
                    help="Subject ID(s) to open. Default: all found.")
    pr.add_argument("--port", type=int, default=8765, help="Local server port.")
    pr.set_defaults(func=cmd_rate)

    pw = sub.add_parser("raw", help="Launch interactive raw QC viewer.")
    pw.add_argument("output_dir", type=Path, help="fnirs-pipe output directory.")
    # one subject, not a list: the viewer opens one recording at a time
    pw.add_argument("--participant-label", required=True, type=_shared.BidsLabel,
                    help="Subject ID to open, e.g. '01'.")
    pw.add_argument("--session-label", default=None, type=_shared.BidsLabel,
                    help="Session label.")
    pw.add_argument("--task-label", default=None, type=_shared.BidsLabel,
                    help="Task label.")
    pw.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI threshold for pre-highlighting bad channels.")
    pw.add_argument("--port", type=int, default=5052, help="Local server port.")
    pw.set_defaults(func=cmd_raw)

    ph = sub.add_parser("hyper", help="Launch interactive hyperscanning QC viewer.")
    ph.add_argument("output_dir", type=Path, help="fnirs-pipe output directory.")
    ph.add_argument("--group-id", required=True, type=_shared.BidsLabel,
                    help="Group ID to open, e.g. 'A'.")
    ph.add_argument("--task-label", required=True, type=_shared.BidsLabel,
                    help="Task label, e.g. 'tapping'.")
    ph.add_argument("--pairs-csv", type=Path, required=True,
                    help="CSV with columns: group_id, subject_id, task (same as fnirs-qc hyper-raw). "
                         "Used to look up subject IDs in this group.")
    ph.add_argument("--session-label", default=None, type=_shared.BidsLabel,
                    help="Session label.")
    ph.add_argument("--sci-threshold", type=float, default=0.8, help="SCI threshold.")
    ph.add_argument("--port", type=int, default=5053, help="Local server port.")
    ph.set_defaults(func=cmd_hyper)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k not in ("func", "command")}
    args.func(**kw)
