"""Run a built argv and show what it made. Shared by the two pages that drive a CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import html

from fnirs_pipe.interface.report_serve import report_url


def log_panel(header: str, lines, dropped: int, color: str):
    body = "\n".join(lines) if lines else "(no output yet)"
    if dropped:
        body = f"... {dropped} earlier lines dropped ...\n{body}"
    return dbc.Alert([
        html.Div(header, className="fw-bold"),
        html.Pre(body, className="fp-run-tail"),
    ], color=color, className="mb-0")


def preview_text(argv: list[str], shell: str) -> str:
    cont = {"bash": "\\", "cmd": "^", "powershell": "`"}.get(shell, "\\")
    return f" {cont}\n".join([" ".join(argv[:2])] + [f"  {a}" for a in argv[2:]])


# The landing page each command writes, newest match wins. A command absent from this table
# writes tables and no report; a command present here that produced nothing is a failure,
# not a quiet "not found". Keeping every pattern in one place is what makes a renamed
# report show up as one broken page rather than none.
REPORT_PATTERNS: dict[str, list[str]] = {
    "prep-raw":     ["sub-*/sub-*_desc-raw_report.html"],
    "hyper-raw":    ["group-*/group-*_desc-raw_report.html"],
    # the dyad's landing page is its index, which `fnirs-hyper` writes too; the task page is the
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


def run_and_report(stored: dict | None):
    if not stored or not stored.get("argv"):
        return dbc.Alert("Click 'Generate command' first.", color="warning",
                         className="mb-0"), None

    argv = stored["argv"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return dbc.Alert(f"`{argv[0]}` not found on PATH. Make sure fnirs-pipe is installed.",
                         color="danger", className="mb-0"), None
    except Exception as exc:
        return dbc.Alert(f"Failed to launch: {exc}", color="danger", className="mb-0"), None

    tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-30:])
    ok = proc.returncode == 0
    alert = dbc.Alert([
        html.Div("Finished." if ok else f"Failed (exit {proc.returncode}).",
                 className="fw-bold"),
        html.Pre(tail, className="fp-run-tail"),
    ], color="success" if ok else "danger", className="mb-0")

    if not ok:
        return alert, None

    patterns = REPORT_PATTERNS.get(stored["command"])
    if patterns is None:
        return alert, html.Small("This command writes tables, not a report.",
                                 className="text-muted")

    report = find_report(patterns, Path(stored["output_dir"]))
    if report is None:
        # the command is known to write a page and did not, so say so instead of reading
        # as "nothing to show here": a pattern left behind by a rename looks identical
        return alert, dbc.Alert([
            html.Div("The run finished but its report is missing.", className="fw-bold"),
            html.Small(f"Looked for {', '.join(patterns)} under {stored['output_dir']}."),
        ], color="warning", className="mb-0")

    return alert, html.Div([
        html.Small(str(report), className="text-muted d-block mb-2"),
        html.Iframe(src=report_url(report), className="fp-report-frame"),
    ])
