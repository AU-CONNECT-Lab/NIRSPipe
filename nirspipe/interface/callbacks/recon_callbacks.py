"""Callbacks for the Recon page: detect raw snirf files, preview, run nirspipe-recon per file."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

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
    Output("rc-log",     "children"),
    Input("rc-run-btn",  "n_clicks"),
    State("rc-table",    "virtualRowData"),
    State("rc-bids-dir", "value"),
    State("rc-overwrite","value"),
    State("rc-optode-frame", "value"),
    prevent_initial_call=True,
)
def run_all(n_clicks, rows, bids_dir, overwrite, optode_frame):
    if not bids_dir:
        return dbc.Alert("Set the output BIDS directory.", color="warning")
    if not rows:
        return dbc.Alert("Detect files first.", color="warning")

    lines, n_ok, n_fail = [], 0, 0
    for row in rows:
        argv = build_recon_args(row, bids_dir, overwrite, optode_frame)
        if argv is None:
            n_fail += 1
            lines.append(html.Div(f"✗ {row['file']}: subject/task missing", className="text-danger"))
            continue
        try:
            proc = subprocess.run(argv, capture_output=True, text=True)
        except FileNotFoundError:
            return dbc.Alert("`nirspipe-recon` not found on PATH. Make sure nirspipe is installed.",
                             color="danger")
        if proc.returncode == 0:
            n_ok += 1
            lines.append(html.Div(f"✓ {row['file']} → sub-{row['subject']}", className="text-success"))
        else:
            n_fail += 1
            # the CLI's last line is its error message; the rest is the traceback above it
            why = ((proc.stderr or proc.stdout or "").strip().splitlines() or [f"exit {proc.returncode}"])[-1]
            lines.append(html.Div(f"✗ {row['file']}: {why}", className="text-danger"))

    color = "success" if n_fail == 0 else "warning"
    header = dbc.Alert(f"Done: {n_ok} written, {n_fail} failed.", color=color, className="py-2")
    return html.Div([header, *lines])
