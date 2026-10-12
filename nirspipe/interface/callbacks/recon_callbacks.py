"""Callbacks for the Recon page: detect raw snirf files, preview, run nirspipe-recon per file."""

from __future__ import annotations

import re
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

from nirspipe.interface import process_stream
from nirspipe.interface.callbacks._cli_run import log_panel, poll_run
from nirspipe.interface.cli_args import build_recon_args

_CONT = {"bash": "\\", "cmd": "^", "powershell": "`"}


def _sanitize_label(text: str) -> str:
    """Strip to BIDS-legal alphanumeric characters."""
    return re.sub(r"[^0-9a-zA-Z]", "", text)


@callback(
    Output("rc-detect-result", "children"),
    Output("rc-table",         "rowData"),
    Input("rc-detect-btn",     "n_clicks"),
    State("rc-input-dir",      "value"),
    prevent_initial_call=True,
)
def detect_files(n_clicks, input_dir):
    if not input_dir:
        return "Set the input folder first.", []
    root = Path(input_dir)
    if not root.exists():
        return f"Folder not found: {input_dir}", []

    files = sorted(root.rglob("*.snirf"))
    if not files:
        return "No .snirf files found.", []

    rows = [
        {
            "path":    str(f),
            "file":    f.name,
            "subject": _sanitize_label(f.stem),
            "task":    "",
            "session": "",
            "run":     "",
        }
        for f in files
    ]
    return f"Detected {len(rows)} file(s).", rows


def _row_preview(argv: list[str], cont: str) -> str:
    # the two paths quoted, so a folder with spaces pastes as it stands; a flag keeps its value
    lines = [f'{argv[0]} "{argv[1]}"', f'  "{argv[2]}"']
    for token in argv[3:]:
        if token.startswith("--"):
            lines.append(f"  {token}")
        else:
            lines[-1] += f" {token}"
    return f" {cont}\n".join(lines)


@callback(
    Output("rc-command-preview", "children"),
    Input("rc-table",         "virtualRowData"),
    Input("rc-shell-select",  "value"),
    Input("rc-bids-dir",      "value"),
    Input("rc-overwrite",     "value"),
    Input("rc-optode-frame",  "value"),
)
def preview_commands(rows, shell, bids_dir, overwrite, optode_frame):
    if not rows:
        return "Detect files to preview commands."
    cont = _CONT.get(shell, "\\")
    bids_dir = bids_dir or "<output-bids-dir>"
    argvs = [build_recon_args(r, bids_dir, overwrite, optode_frame) for r in rows]
    cmds = [_row_preview(argv, cont) for argv in argvs if argv is not None]
    if not cmds:
        return "Fill in subject and task for at least one file."
    return "\n\n".join(cmds)


@callback(
    Output("rc-log",       "children"),
    Output("rc-skipped",   "children"),
    Output("rc-run-store", "data"),
    Output("rc-run-tick",  "disabled"),
    Output("rc-stop-btn",  "disabled"),
    Output("rc-run-btn",   "disabled"),
    Input("rc-run-btn",    "n_clicks"),
    State("rc-table",      "virtualRowData"),
    State("rc-bids-dir",   "value"),
    State("rc-overwrite",  "value"),
    State("rc-optode-frame", "value"),
    prevent_initial_call=True,
)
def run_all(n_clicks, rows, bids_dir, overwrite, optode_frame):
    idle = (None, None, True, True, False)
    if not bids_dir:
        return dbc.Alert("Set the output BIDS directory.", color="warning"), *idle
    if not rows:
        return dbc.Alert("Detect files first.", color="warning"), *idle

    steps, skipped = [], []
    for row in rows:
        argv = build_recon_args(row, bids_dir, overwrite, optode_frame)
        if argv is None:
            skipped.append(row["file"])
        else:
            steps.append((row, argv))
    if not steps:
        return dbc.Alert("Fill in subject and task for at least one file.",
                         color="warning"), *idle

    n = len(steps)
    headed = [(f"[{i}/{n}] {row['file']} -> sub-{row['subject']}", argv)
              for i, (row, argv) in enumerate(steps, 1)]
    try:
        run_id = process_stream.start_each(headed)
    except FileNotFoundError:
        return dbc.Alert("`nirspipe-recon` not found on PATH. Make sure nirspipe is installed.",
                         color="danger"), *idle
    note = (html.Small(f"Skipped, no subject or task: {', '.join(skipped)}",
                       className="text-warning d-block mt-2") if skipped else None)
    return log_panel("Running...", [], 0, "info"), note, run_id, False, False, True


@callback(
    Output("rc-log",      "children", allow_duplicate=True),
    Output("rc-run-tick", "disabled", allow_duplicate=True),
    Output("rc-stop-btn", "disabled", allow_duplicate=True),
    Output("rc-run-btn",  "disabled", allow_duplicate=True),
    Input("rc-run-tick",  "n_intervals"),
    State("rc-run-store", "data"),
    prevent_initial_call=True,
)
def stream_run_output(_n, run_id):
    return poll_run(run_id, "Recon")


@callback(
    Output("rc-stop-btn", "disabled", allow_duplicate=True),
    Input("rc-stop-btn",  "n_clicks"),
    State("rc-run-store", "data"),
    prevent_initial_call=True,
)
def stop_run(_n, run_id):
    # the tick reports the outcome; this only asks the current file's command to end
    if run_id:
        process_stream.stop(run_id)
    return True
