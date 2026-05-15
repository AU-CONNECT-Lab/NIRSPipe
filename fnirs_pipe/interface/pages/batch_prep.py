"""Page: Batch data preparation — marker editing and crop across multiple subjects."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

dash.register_page(__name__, path="/batch-prep", name="Batch Prep")

_RENAME_COLS = [
    {"name": "From", "id": "from_name", "editable": True},
    {"name": "To",   "id": "to_name",   "editable": True},
]

_SEG_COLS = [
    {"name": "Onset (s)",    "id": "onset",    "editable": True, "type": "numeric"},
    {"name": "Duration (s)", "id": "duration", "editable": True, "type": "numeric"},
]


def _card(title, *children):
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(list(children))],
        className="mb-3",
    )


layout = dbc.Container([
    dbc.Row([dbc.Col([html.H3("Batch Preparation"), html.Hr()])]),

    # ── Data Source ───────────────────────────────────────────────────────────
    _card("Data Source",
        dbc.Row([
            dbc.Col([
                dbc.Label("BIDS Directory"),
                dbc.Input(id="bp-bids-dir", type="text",
                          placeholder="/path/to/bids"),
            ], width=5),
            dbc.Col([
                dbc.Label("Derivatives Directory"),
                dbc.Input(id="bp-deriv-dir", type="text",
                          placeholder="/path/to/derivatives"),
            ], width=5),
        ], className="g-3"),
    ),

    # ── Subjects ──────────────────────────────────────────────────────────────
    _card("Subjects",
        dbc.Row([
            dbc.Col(dbc.Button("Detect", id="bp-detect-btn",
                               color="primary", size="sm"), width="auto"),
            dbc.Col(dbc.Button("Select All", id="bp-select-all-btn",
                               color="outline-secondary", size="sm"), width="auto"),
            dbc.Col(dbc.Button("Clear", id="bp-clear-btn",
                               color="outline-secondary", size="sm"), width="auto"),
        ], className="g-2 mb-2"),
        html.Div(id="bp-subjects-result", className="mb-2"),
        html.Div(id="bp-subjects-container"),
    ),

    # ── Run filter ────────────────────────────────────────────────────────────
    _card("Run Filter (optional)",
        dbc.Row([
            dbc.Col([
                dbc.Label("Session"),
                dbc.Input(id="bp-ses", type="text",
                          placeholder="e.g. 01", size="sm"),
            ], width=2),
            dbc.Col([
                dbc.Label("Task"),
                dbc.Input(id="bp-task", type="text",
                          placeholder="e.g. tapping", size="sm"),
            ], width=2),
            dbc.Col([
                dbc.Label("Run"),
                dbc.Input(id="bp-run", type="text",
                          placeholder="e.g. 01", size="sm"),
            ], width=2),
        ], className="g-3"),
    ),

    # ── Operation ─────────────────────────────────────────────────────────────
    _card("Operation",
        dbc.RadioItems(
            id="bp-operation",
            options=[
                {"label": "Edit Markers", "value": "markers"},
                {"label": "Crop",         "value": "crop"},
            ],
            value="markers", inline=True, className="mb-3",
        ),

        # Markers panel ───────────────────────────────────────────────────────
        html.Div(id="bp-markers-panel", children=[
            dbc.RadioItems(
                id="bp-marker-op",
                options=[
                    {"label": "Shift onsets",   "value": "shift"},
                    {"label": "Set duration",   "value": "set_duration"},
                    {"label": "Rename markers", "value": "rename"},
                ],
                value="shift", inline=True, className="mb-3 small",
            ),

            html.Div(id="bp-shift-panel", children=[
                dbc.Row([
                    dbc.Col(dbc.Label("Shift (s)", className="small mb-0"),
                            width="auto", className="d-flex align-items-center"),
                    dbc.Col(dbc.Input(id="bp-shift-val", type="number", size="sm",
                                      style={"width": "100px"}), width="auto"),
                    dbc.Col(html.Small("negative = earlier; clipped to 0",
                                       className="text-muted"),
                            width="auto", className="d-flex align-items-center"),
                ], className="g-2 align-items-center"),
            ]),

            html.Div(id="bp-duration-panel", style={"display": "none"}, children=[
                dbc.Row([
                    dbc.Col(dbc.Label("Duration (s)", className="small mb-0"),
                            width="auto", className="d-flex align-items-center"),
                    dbc.Col(dbc.Input(id="bp-duration-val", type="number", size="sm",
                                      style={"width": "100px"}), width="auto"),
                ], className="g-2 align-items-center"),
            ]),

            html.Div(id="bp-rename-panel", style={"display": "none"}, children=[
                dash_table.DataTable(
                    id="bp-rename-table",
                    columns=_RENAME_COLS,
                    data=[{"from_name": "", "to_name": ""}],
                    editable=True,
                    row_deletable=True,
                    style_table={"overflowX": "auto", "maxHeight": "200px",
                                 "overflowY": "auto"},
                    style_header={"fontWeight": "600", "fontSize": "0.78rem"},
                    style_cell={"fontSize": "0.78rem", "padding": "3px 6px"},
                ),
                dbc.Button("+ Add pair", id="bp-rename-add-btn", size="sm",
                           color="outline-secondary", className="mt-2"),
            ]),
        ]),

        # Crop panel ──────────────────────────────────────────────────────────
        html.Div(id="bp-crop-panel", style={"display": "none"}, children=[
            dbc.RadioItems(
                id="bp-crop-mode",
                options=[
                    {"label": "Single segment",  "value": "single"},
                    {"label": "Multi-segment", "value": "multi"},
                ],
                value="single", inline=True, className="mb-3 small",
            ),

            html.Div(id="bp-crop-single", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("tmin (s)", className="small mb-0"),
                        dbc.Input(id="bp-crop-tmin", type="number", size="sm",
                                  style={"width": "100px"}),
                    ], width="auto"),
                    dbc.Col([
                        dbc.Label("tmax (s)", className="small mb-0"),
                        dbc.Input(id="bp-crop-tmax", type="number", size="sm",
                                  style={"width": "100px"}),
                    ], width="auto"),
                ], className="g-3"),
            ]),

            html.Div(id="bp-crop-multi", style={"display": "none"}, children=[
                dash_table.DataTable(
                    id="bp-crop-seg-table",
                    columns=_SEG_COLS,
                    data=[{"onset": 0.0, "duration": 30.0}],
                    editable=True,
                    row_deletable=True,
                    style_table={"overflowX": "auto", "maxHeight": "200px",
                                 "overflowY": "auto"},
                    style_header={"fontWeight": "600", "fontSize": "0.78rem"},
                    style_cell={"fontSize": "0.78rem", "padding": "3px 6px"},
                ),
                dbc.Row([
                    dbc.Col(dbc.Button("+ Segment", id="bp-seg-add-btn", size="sm",
                                       color="outline-secondary"), width="auto"),
                    dbc.Col(dbc.Checklist(
                        id="bp-crop-combine",
                        options=[{"label": "Combine into one file", "value": "combine"}],
                        value=[], inline=True, className="small",
                    ), width="auto"),
                ], className="g-2 mt-2 align-items-center"),
            ]),
        ]),
    ),

    # ── Run ───────────────────────────────────────────────────────────────────
    _card("Run",
        dbc.Row([
            dbc.Col([
                dbc.Label("Parallel jobs"),
                dbc.Input(id="bp-n-jobs", type="number", value=1, min=1, step=1,
                          size="sm", style={"width": "80px"}),
            ], width="auto"),
            dbc.Col(
                dbc.Button("Run Batch", id="bp-run-btn", color="success"),
                width="auto", className="d-flex align-items-end",
            ),
        ], className="g-3 align-items-end"),
        html.Div(id="bp-log", className="mt-3"),
    ),

], fluid=True)
