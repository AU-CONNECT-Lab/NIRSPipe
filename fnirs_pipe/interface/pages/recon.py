"""Page: Recon — batch-convert raw snirf files into a BIDS dataset."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

dash.register_page(__name__, path="/recon", name="Recon")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_COLS = [
    {"name": "File",    "id": "file",    "editable": False},
    {"name": "subject", "id": "subject", "editable": True},
    {"name": "task",    "id": "task",    "editable": True},
    {"name": "session", "id": "session", "editable": True},
    {"name": "run",     "id": "run",     "editable": True},
]


def _card(title, *children):
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(list(children))],
        className="mb-3",
    )


layout = dbc.Container([
    dbc.Row([dbc.Col([html.H3("Recon — raw snirf → BIDS"), html.Hr()])]),

    _card("Data Source",
        dbc.Row([
            dbc.Col([
                dbc.Label("Input Folder (raw snirf)"),
                dbc.Input(id="rc-input-dir", type="text", placeholder="/path/to/raw"),
            ], width=5),
            dbc.Col([
                dbc.Label("Output BIDS Directory"),
                dbc.Input(id="rc-bids-dir", type="text", placeholder="/path/to/bids"),
            ], width=5),
        ], className="g-3"),
    ),

    _card("Files",
        dbc.Row([
            dbc.Col(dbc.Button("Detect", id="rc-detect-btn",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(dbc.Checkbox(id="rc-overwrite", label="Overwrite existing",
                                 value=False), width="auto"),
        ], className="g-2 mb-2 align-items-center"),
        html.Div(id="rc-detect-result", className="mb-2 small"),
        dash_table.DataTable(
            id="rc-table",
            columns=_COLS,
            data=[],
            editable=True,
            style_cell={"fontFamily": "monospace", "fontSize": "13px",
                        "textAlign": "left", "padding": "4px 8px"},
            style_header={"fontWeight": "bold"},
        ),
        html.Small("subject/task are required; session/run optional. "
                   "subject must be alphanumeric only.",
                   className="text-muted"),
    ),

    _card("Command Preview",
        dbc.Row([
            dbc.Col(dbc.Label("Shell (line-continuation style)"), width="auto"),
            dbc.Col(dcc.Dropdown(
                id="rc-shell-select",
                options=[
                    {"label": "bash / zsh (macOS, Linux)", "value": "bash"},
                    {"label": "cmd (Windows, Anaconda Prompt)", "value": "cmd"},
                    {"label": "PowerShell (Windows)", "value": "powershell"},
                ],
                value=_DEFAULT_SHELL,
                clearable=False,
            ), width=4),
        ], className="mb-2 align-items-center"),
        html.Pre(
            id="rc-command-preview",
            style={
                "background":   "#f8f9fa",
                "padding":      "1rem",
                "borderRadius": "4px",
                "fontSize":     "13px",
                "fontFamily":   "monospace",
                "whiteSpace":   "pre-wrap",
                "wordBreak":    "break-all",
                "minHeight":    "60px",
            },
        ),
    ),

    _card("Run",
        dbc.Button("Run All", id="rc-run-btn", color="success", size="sm"),
        html.Div(id="rc-log", className="mt-2 small"),
    ),
], fluid=True)
