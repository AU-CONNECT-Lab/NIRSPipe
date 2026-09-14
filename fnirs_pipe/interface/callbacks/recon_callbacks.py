"""Callbacks for the Recon page: detect raw snirf files, preview, batch convert."""

from __future__ import annotations

import re
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

_CONT = {"bash": "\\", "cmd": "^", "powershell": "`"}


def _sanitize_label(text: str) -> str:
    """Strip to BIDS-legal alphanumeric characters."""
    return re.sub(r"[^0-9a-zA-Z]", "", text)


@callback(
    Output("rc-detect-result", "children"),
    Output("rc-table",         "data"),
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


def _row_command(row: dict, bids_dir: str, overwrite: bool, cont: str) -> str | None:
    subject, task = row.get("subject"), row.get("task")
    if not subject or not task:
        return None
    argv = [
        "fnirs-recon",
        f'"{row["path"]}"',
        f'"{bids_dir}"',
        "--participant-label", subject,
        "--task-label", task,
    ]
    if row.get("session"):
        argv += ["--session-label", row["session"]]
    if row.get("run"):
        argv += ["--run-label", row["run"]]
    if overwrite:
        argv.append("--overwrite")
    lines = [argv[0]] + [f"  {a}" for a in argv[1:]]
    return f" {cont}\n".join(lines)


@callback(
    Output("rc-command-preview", "children"),
    Input("rc-table",         "data"),
    Input("rc-shell-select",  "value"),
    Input("rc-bids-dir",      "value"),
    Input("rc-overwrite",     "value"),
)
def preview_commands(rows, shell, bids_dir, overwrite):
    if not rows:
        return "Detect files to preview commands."
    cont = _CONT.get(shell, "\\")
    bids_dir = bids_dir or "<output-bids-dir>"
    cmds = [_row_command(r, bids_dir, overwrite, cont) for r in rows]
    cmds = [c for c in cmds if c is not None]
    if not cmds:
        return "Fill in subject and task for at least one file."
    return "\n\n".join(cmds)


@callback(
    Output("rc-log",     "children"),
    Input("rc-run-btn",  "n_clicks"),
    State("rc-table",    "data"),
    State("rc-bids-dir", "value"),
    State("rc-overwrite","value"),
    prevent_initial_call=True,
)
def run_all(n_clicks, rows, bids_dir, overwrite):
    if not bids_dir:
        return dbc.Alert("Set the output BIDS directory.", color="warning")
    if not rows:
        return dbc.Alert("Detect files first.", color="warning")

    from fnirs_pipe.io.bids import write_bids_from_snirf

    out = Path(bids_dir)
    lines, n_ok, n_fail = [], 0, 0
    for row in rows:
        subject, task = row.get("subject"), row.get("task")
        if not subject or not task:
            n_fail += 1
            lines.append(html.Div(f"✗ {row['file']}: subject/task missing", className="text-danger"))
            continue
        try:
            write_bids_from_snirf(
                Path(row["path"]), out, subject=subject, task=task,
                session=row.get("session") or None, run=row.get("run") or None,
                overwrite=bool(overwrite),
            )
            n_ok += 1
            lines.append(html.Div(f"✓ {row['file']} → sub-{subject}", className="text-success"))
        except Exception as e:
            n_fail += 1
            lines.append(html.Div(f"✗ {row['file']}: {e}", className="text-danger"))

    color = "success" if n_fail == 0 else "warning"
    header = dbc.Alert(f"Done: {n_ok} written, {n_fail} failed.", color=color, className="py-2")
    return html.Div([header, *lines])
