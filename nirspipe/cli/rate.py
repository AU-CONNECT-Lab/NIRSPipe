"""nirspipe-rate CLI entry point (argparse)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nirspipe import __version__
from nirspipe.exceptions import StageError

from nirspipe.io.derivatives import entity_of, group_label, subject_labels
from nirspipe.io.naming import is_bids_label, report_name
from nirspipe.cli import _shared


def rater_name(value: str) -> str:
    """A rater's name as it goes into file names: letters and digits only."""
    text = str(value).strip()
    if not is_bids_label(text):
        raise argparse.ArgumentTypeError(
            f"{value!r}: a rater's name goes into file names, so letters and digits only, "
            "e.g. 'QY' or 'rater2'")
    return text


def _say_rater(rater: str | None) -> None:
    if rater:
        print(f"Rating as {rater}: ratings, channel decisions and marked stretches go to "
              f"files named rater-{rater}.")


def _serve(opened: list, port: int | None, default_port: int, log_name: str) -> None:
    from nirspipe.interface.server import WebApp

    web = WebApp()
    paths = [path for viewer, label, entries in opened
             for path in web.mount(viewer.app, label, entries)]
    # several pages open on the home page that lists them, not a browser tab each
    web.serve(port, default_port, log_name, paths if len(opened) == 1 else [""])


def open_rate(output_dir: Path, participant_label: list[str] | None,
              rater: str | None = None) -> list:
    """[(viewer, label, entry pages)] for the run reports; exits when there is nothing to rate."""
    from nirspipe.qc.rating.app import FNIRSRatingApp

    subjects = participant_label or subject_labels(output_dir)
    if not subjects:
        print("No subjects found in output directory.", file=sys.stderr)
        raise SystemExit(1)
    return [(FNIRSRatingApp(output_dir, subjects, rater), "Run reports",
             tuple(f"sub-{subject}/" for subject in subjects))]


def cmd_rate(output_dir: Path, participant_label: list[str] | None, port: int | None,
             rater: str | None = None) -> None:
    """Launch QC rating interface for nirspipe reports."""
    from nirspipe.qc.rating.app import RATE_PORT

    opened = open_rate(output_dir, participant_label, rater)
    _say_rater(rater)
    _serve(opened, port, RATE_PORT, "nirspipe-rate")


def open_raw(
    output_dir: Path, participant_label: str,
    session_label: str | None, task_label: str | None,
    sci_threshold: float, run_label: str | None = None, rater: str | None = None,
) -> list:
    """[(viewer, label, entry pages)], one per raw report the selection matches; exits on none."""
    from nirspipe.qc.rating.app import RawRatingApp

    wanted = {"ses": session_label, "task": task_label, "run": run_label}
    sub_dir = output_dir / f"sub-{participant_label}"
    pages = [page for page in sorted(sub_dir.glob(f"sub-{participant_label}_*_desc-raw_report.html"))
             if "_cond-" not in page.name
             and all(value is None or entity_of(page.stem, key) == value
                     for key, value in wanted.items())]
    if not pages:
        print(f"Error: no raw report in {sub_dir} matches the selection", file=sys.stderr)
        raise SystemExit(1)
    return [(RawRatingApp(page, output_dir, sci_threshold, rater), page.stem, ("",))
            for page in pages]


def cmd_raw(
    output_dir: Path, participant_label: str,
    session_label: str | None, task_label: str | None,
    sci_threshold: float, port: int | None, run_label: str | None = None,
    rater: str | None = None,
) -> None:
    """Launch interactive raw QC viewer with section ratings and channel decisions."""
    from nirspipe.qc.rating.app import RAW_PORT

    opened = open_raw(output_dir, participant_label, session_label, task_label, sci_threshold,
                      run_label, rater)
    for viewer, _label, _entries in opened:
        print(f"Launching raw viewer: {viewer.html_path.name} ...")
    _say_rater(rater)
    _serve(opened, port, RAW_PORT, "nirspipe-rate raw")


def open_hyper(
    output_dir: Path, group_id: str, task_label: str, pairs_csv: Path,
    session_label: str | None, sci_threshold: float, rater: str | None = None,
) -> tuple:
    """(viewer, label, entry pages) for one dyad's raw report; exits when it or the group is missing."""
    from nirspipe.io.groups import parse_group_csv
    from nirspipe.qc.rating.app import HyperRatingApp

    fname = report_name(group_label(group_id, task_label, session_label), desc="raw")
    html_path = output_dir / f"group-{group_id}" / fname
    if not html_path.exists():
        print(f"Error: hyper report not found: {html_path}", file=sys.stderr)
        raise SystemExit(1)

    groups = parse_group_csv(pairs_csv)
    # a pairs table without sessions keys every group under None, whatever the page's name
    members = (groups.get((group_id, task_label, session_label))
               or groups.get((group_id, task_label, None)))
    if not members:
        print(f"Error: group_id '{group_id}' + task '{task_label}' not found in {pairs_csv}", file=sys.stderr)
        raise SystemExit(1)
    subject_ids = [e.subject_id for e in members]
    return [(HyperRatingApp(html_path, output_dir, subject_ids, sci_threshold, rater=rater),
             html_path.stem, ("",))]


def cmd_hyper(
    output_dir: Path, group_id: str, task_label: str, pairs_csv: Path,
    session_label: str | None, sci_threshold: float, port: int | None,
    rater: str | None = None,
) -> None:
    """Launch interactive hyperscanning QC viewer with section ratings and channel decisions."""
    from nirspipe.qc.rating.app import HYPER_PORT

    opened = open_hyper(output_dir, group_id, task_label, pairs_csv, session_label,
                        sci_threshold, rater)
    print(f"Launching hyper viewer: {opened[0][0].html_path.name} ...")
    _say_rater(rater)
    _serve(opened, port, HYPER_PORT, "nirspipe-rate hyper")


def cmd_export(trees: list[Path], out_dir: Path | None, rater: str | None = None) -> None:
    """Write the rated channels as a --bad-channels table, and the FINAL verdicts as a list."""
    from nirspipe.qc.rating.export import export_ratings, export_spans, raters

    found = raters(trees)
    if rater is not None and rater not in found:
        print(f"[warning] no files of rater {rater} under {', '.join(map(str, trees))}; "
              f"raters found: {', '.join(r or '(none)' for r in found) or 'none'}.",
              file=sys.stderr)
    try:
        channels, verdicts, conflicts = export_ratings(trees, out_dir or trees[0], rater)
    except StageError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Channel decisions: {channels}")
    print(f"  pass it to nirspipe as --bad-channels {channels}")
    if rater is None and len(found) > 1:
        named = ", ".join(r or "(saved without --rater)" for r in found)
        print(f"[info] the tables hold several raters' decisions ({named}). nirspipe applies "
              "one set: export one rater's with --rater NAME, or settle them into one table "
              "and blank its rater column.", file=sys.stderr)
    spans = export_spans(trees, out_dir or trees[0], rater)
    if spans:
        print(f"Marked bad stretches: {spans}")
        print(f"  pass it to nirspipe as --bad-spans {spans}")
    print(f"FINAL verdicts (for reading; nothing reads them): {verdicts}")
    if conflicts:
        print("[warning] rated both bad and good for one recording; nirspipe will refuse "
              "these until they are settled in the viewer:", file=sys.stderr)
        for line in conflicts:
            print(f"  {line}", file=sys.stderr)


def _add_rater(parser: argparse.ArgumentParser, kept: str) -> None:
    parser.add_argument("--rater", type=rater_name, default=None,
                        help=f"Your name or initials, letters and digits. Your {kept} go to "
                             "files of your own, so raters never overwrite each other. "
                             "Default: the shared files.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-rate",
        description="Interactive QC review for fNIRS data: rating, individual viewer, hyperscanning viewer.",
    )
    p.add_argument("--version", action="version", version=f"nirspipe-rate {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("rate", help="Launch QC rating interface.")
    pr.add_argument("output_dir", type=Path, help="nirspipe output directory.")
    pr.add_argument("--participant-label", "--participant_label", nargs="+", action="extend",
                    type=_shared.BidsLabel,
                    help="Subject ID(s) to open. Default: all found.")
    pr.add_argument("--port", type=int, default=None,
                    help="Local server port. Default: 8765, or the next free port above it.")
    _add_rater(pr, "ratings")
    pr.set_defaults(func=cmd_rate)

    pw = sub.add_parser("raw", help="Launch interactive raw QC viewer: rate sections, decide channels, mark bad stretches.")
    pw.add_argument("output_dir", type=Path, help="nirspipe output directory.")
    # one subject, not a list: the viewer serves that subject's raw reports
    pw.add_argument("--participant-label", "--participant_label", required=True, type=_shared.BidsLabel,
                    help="Subject ID to open, e.g. '01'. Every raw report of the subject the "
                         "other labels match opens, each recording on its own page.")
    pw.add_argument("--session-label", "--session_label", default=None, type=_shared.BidsLabel,
                    help="Session label.")
    pw.add_argument("--task-label", "--task_label", default=None, type=_shared.BidsLabel,
                    help="Task label.")
    pw.add_argument("--run-label", "--run_label", default=None, type=_shared.BidsLabel,
                    help="Run label.")
    pw.add_argument("--sci-threshold", type=float, default=None,
                    help="A second SCI line: outline each SCI (10 s) cell below it. The red "
                         "cells stay the run's own line. Off when not given.")
    pw.add_argument("--port", type=int, default=None,
                    help="Local server port. Default: 5052, or the next free port above it.")
    _add_rater(pw, "ratings, channel decisions and marked stretches")
    pw.set_defaults(func=cmd_raw)

    ph = sub.add_parser("hyper", help="Launch interactive hyperscanning QC viewer.")
    ph.add_argument("output_dir", type=Path,
                    help="The tree holding the group-<id>/ raw report, as nirspipe-qc's group "
                         "level wrote it.")
    ph.add_argument("--group-id", required=True, type=_shared.BidsLabel,
                    help="Group ID to open, e.g. 'A'.")
    ph.add_argument("--task-label", "--task_label", required=True, type=_shared.BidsLabel,
                    help="Task label, e.g. 'tapping'.")
    ph.add_argument("--pairs-csv", type=Path, required=True,
                    help="CSV with columns: group_id, subject_id, task (same as nirspipe-qc's group level). "
                         "Used to look up subject IDs in this group.")
    ph.add_argument("--session-label", "--session_label", default=None, type=_shared.BidsLabel,
                    help="Session label.")
    ph.add_argument("--sci-threshold", type=float, default=None,
                    help="A second SCI line: outline each member's SCI (10 s) cell below it. "
                         "The red cells stay each member's own line. Off when not given.")
    ph.add_argument("--port", type=int, default=None,
                    help="Local server port. Default: 5053, or the next free port above it.")
    _add_rater(ph, "ratings and channel decisions")
    ph.set_defaults(func=cmd_hyper)

    pe = sub.add_parser("export", help="Export channel decisions for nirspipe --bad-channels.")
    pe.add_argument("trees", type=Path, nargs="+",
                    help="The raw QC trees holding the decision files.")
    pe.add_argument("--out-dir", "--out_dir", type=Path, default=None,
                    help="Folder for the two tables. Default: the first tree.")
    pe.add_argument("--rater", type=rater_name, default=None,
                    help="Export only this rater's files. Default: every rater's, named in a "
                         "rater column; nirspipe takes one rater's table.")
    pe.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k not in ("func", "command")}
    args.func(**kw)
