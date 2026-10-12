"""Run a built argv and show what it made. Shared by the pages that drive a CLI."""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html, no_update

from nirspipe.interface import process_stream
from nirspipe.interface.report_serve import report_url
from nirspipe.utils.logging import get_logger

logger = get_logger("interface.cli_run")


def log_panel(header: str, lines, dropped: int, color: str):
    body = "\n".join(lines) if lines else "(no output yet)"
    if dropped:
        body = f"... {dropped} earlier lines dropped ...\n{body}"
    return dbc.Alert([
        html.Div(header, className="fw-bold"),
        html.Pre(body, className="fp-run-tail"),
    ], color=color, className="mb-0")


def start_run(cmd_data, generate_label: str, not_found: str):
    """Launch a page's built argv: (status, run_id, tick off, stop off, run off) for its outputs.

    ``generate_label`` is the page's button for building the command, ``not_found`` what to
    say when the executable is missing, with ``{exe}`` for its name.
    """
    if not cmd_data or not cmd_data.get("argv"):
        return (dbc.Alert(f"Click '{generate_label}' first.", color="warning",
                          className="mb-0"), None, True, True, False)
    argv = cmd_data["argv"]
    try:
        run_id = process_stream.start(argv)
    except FileNotFoundError:
        return (dbc.Alert(not_found.format(exe=argv[0]), color="danger", className="mb-0"),
                None, True, True, False)
    except Exception as exc:
        return (dbc.Alert(f"Failed to launch: {exc}", color="danger", className="mb-0"),
                None, True, True, False)
    logger.info("started %s as run %s", argv[0], run_id)
    return log_panel("Running...", [], 0, "info"), run_id, False, False, True


def _tick(run_id, noun: str):
    """The status panel for one poll, and the exit code once the command has ended."""
    lines, returncode, dropped = process_stream.poll(run_id)
    if returncode is None:
        return log_panel("Running...", lines, dropped, "info"), None
    if returncode == 0:
        outcome, color = "finished.", "success"
    elif process_stream.was_stopped(run_id):
        outcome, color = "stopped.", "warning"
    elif returncode < 0:
        outcome, color = f"stopped (signal {-returncode}).", "warning"
    else:
        outcome, color = f"failed (exit {returncode}).", "danger"
    header = f"{noun} {outcome}" if noun else outcome[0].upper() + outcome[1:]
    process_stream.forget(run_id)
    return log_panel(header, lines, dropped, color), returncode


def poll_run(run_id, noun: str = ""):
    """One tick of a running command: (status, tick off, stop off, run off).

    ``noun`` leads the outcome line, "Pipeline finished." against a bare "Finished.".
    """
    if not run_id:
        return no_update, True, True, False
    status, returncode = _tick(run_id, noun)
    ended = returncode is not None
    return status, ended, ended, not ended


def poll_run_and_report(run_id, stored: dict | None):
    """poll_run plus the report preview, filled once the command has succeeded."""
    if not run_id:
        return no_update, no_update, True, True, False
    status, returncode = _tick(run_id, "")
    if returncode is None:
        return status, no_update, False, False, True
    return status, report_preview(stored) if returncode == 0 else None, True, True, False


def preview_text(argv: list[str], shell: str) -> str:
    cont = {"bash": "\\", "cmd": "^", "powershell": "`"}.get(shell, "\\")
    return f" {cont}\n".join([" ".join(argv[:2])] + [f"  {a}" for a in argv[2:]])


# The landing page each command writes, newest match wins. A command absent from this table
# writes tables and no report; a command present here that produced nothing is a failure,
# not a quiet "not found".
REPORT_PATTERNS: dict[str, list[str]] = {
    "prep-raw":     ["sub-*/sub-*_desc-raw_report.html"],
    "hyper-raw":    ["group-*/group-*_desc-raw_report.html"],
    # the dyad's landing page is its index, which `nirspipe-hyper` writes too; the task page is the
    # fallback for a tree whose index failed. A bare `_task-*_report.html` glob would also
    # match the per-condition and per-pairing pages, and those are not where to land
    "run":          ["group-*/group-*_desc-index_report.html",
                     "group-*/group-*_task-*_report.html"],
    "index":        ["group-*/group-*_desc-index_report.html"],
    "cohort":       ["desc-subjects_report.html"],
    "cohort-hyper": ["desc-groups_report.html"],
}


def find_report(patterns: list[str], output_dir: Path) -> Path | None:
    """The newest HTML matching what this command is known to write."""
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(output_dir.glob(pattern))
        if candidates:
            break
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def report_preview(stored: dict | None):
    """What a successful command wrote, framed, or why there is nothing to frame."""
    if not stored:
        return None
    patterns = REPORT_PATTERNS.get(stored["command"])
    if patterns is None:
        return html.Small("This command writes tables, not a report.", className="text-muted")

    report = find_report(patterns, Path(stored["output_dir"]))
    if report is None:
        # the command is known to write a page and did not, so say so instead of reading
        # as "nothing to show here": a pattern left behind by a rename looks identical
        return dbc.Alert([
            html.Div("The run finished but its report is missing.", className="fw-bold"),
            html.Small(f"Looked for {', '.join(patterns)} under {stored['output_dir']}."),
        ], color="warning", className="mb-0")

    return html.Div([
        html.Small(str(report), className="text-muted d-block mb-2"),
        html.Iframe(src=report_url(report), className="fp-report-frame"),
    ])
