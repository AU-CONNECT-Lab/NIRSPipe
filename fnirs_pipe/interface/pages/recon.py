"""Page: Recon, which batch-converts raw snirf files into a BIDS dataset."""

from __future__ import annotations

import platform

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import html

from fnirs_pipe.interface.components import (
    PATH, action_field, actions, card, field, params, split,
)
from fnirs_pipe.interface.grid import AUTO_HEIGHT, COL_DEF

dash.register_page(__name__, path="/recon", name="Recon")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_COLS = [
    {"headerName": "File",    "field": "file",    "editable": False},
    {"headerName": "subject", "field": "subject", "editable": True},
    {"headerName": "task",    "field": "task",    "editable": True},
    {"headerName": "session", "field": "session", "editable": True},
    {"headerName": "run",     "field": "run",     "editable": True},
]


def _files():
    return card("Files",
        params(
            field("Input folder (raw snirf)",
                  dbc.Input(id="rc-input-dir", type="text", placeholder="/path/to/raw"),
                  span=PATH),
            field("Output BIDS directory",
                  dbc.Input(id="rc-bids-dir", type="text", placeholder="/path/to/bids"),
                  span=PATH),
            action_field(
                dbc.Button("Detect", id="rc-detect-btn", color="primary"),
                dbc.Checkbox(id="rc-overwrite", label="Overwrite existing",
                             value=False, className="ms-2"),
            ),
        ),
        html.Div(id="rc-detect-result", className="mt-2 mb-2 small"),
        dag.AgGrid(
            id="rc-table",
            columnDefs=_COLS,
            rowData=[],
            className="fp-grid fp-grid-mono",
            columnSize="responsiveSizeToFit",
            defaultColDef=COL_DEF,
            dashGridOptions=AUTO_HEIGHT,
            style={"height": None},
        ),
        html.Small("subject/task are required; session/run optional. "
                   "subject must be alphanumeric only.",
                   className="text-muted"),
    )


def _run_panel():
    return card("Run",
        actions(dbc.Button("Run all", id="rc-run-btn", color="success")),
        html.Div(id="rc-log", className="mt-2 small"),
    )


def _command_panel():
    return card("Command",
        dbc.Select(
            id="rc-shell-select",
            options=[{"label": "bash / zsh (macOS, Linux)", "value": "bash"},
                     {"label": "cmd (Windows, Anaconda Prompt)", "value": "cmd"},
                     {"label": "PowerShell (Windows)", "value": "powershell"}],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mb-2",
        ),
        html.Pre(id="rc-command-preview", className="fp-command"),
    )


layout = dbc.Container([
    html.H3("Recon: raw snirf → BIDS"),
    html.Hr(),

    split(
        main=[_files()],
        aside=[_run_panel(), _command_panel()],
    ),
], fluid=True)
