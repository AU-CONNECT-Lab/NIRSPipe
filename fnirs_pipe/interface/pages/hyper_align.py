"""Page: Hyperscanning align — align multi-subject recordings by shared trigger."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

dash.register_page(__name__, path="/hyper-align", name="Hyper Align")

_OFFSET_COLS = [
    {"name": "Group",        "id": "group_id"},
    {"name": "Subject",      "id": "subject_id"},
    {"name": "Task",         "id": "task"},
    {"name": "Offset (s)",   "id": "offset_s"},
    {"name": "Duration (s)", "id": "duration_s"},
]


def _card(title, *children):
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(list(children))],
        className="mb-3",
    )


layout = dbc.Container([
    dbc.Row([dbc.Col([html.H3("Hyperscanning Align"), html.Hr()])]),

    _card("Data Source",
        dbc.Row([
            dbc.Col([
                dbc.Label("BIDS Directory"),
                dbc.Input(id="ha-bids-dir", type="text",
                          placeholder="/path/to/bids"),
            ], width=4),
            dbc.Col([
                dbc.Label("Derivatives Directory"),
                dbc.Input(id="ha-deriv-dir", type="text",
                          placeholder="/path/to/derivatives"),
            ], width=4),
            dbc.Col([
                dbc.Label("Group CSV"),
                dbc.Input(id="ha-group-csv", type="text",
                          placeholder="/path/to/groups.csv"),
            ], width=4),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(
                dbc.Button("Load & Align", id="ha-load-btn", color="primary",
                           className="mt-2"),
                width="auto",
            ),
        ]),
        html.Div(id="ha-load-status", className="mt-2 small"),
    ),

    html.Div(id="ha-results-panel", style={"display": "none"}, children=[

        _card("Alignment Offsets",
            dash_table.DataTable(
                id="ha-offset-table",
                columns=_OFFSET_COLS,
                style_table={"overflowX": "auto"},
                style_header={"fontWeight": "600", "fontSize": "0.82rem"},
                style_cell={"fontSize": "0.82rem", "padding": "4px 8px"},
            ),
        ),

        _card("Trigger Timeline",
            html.Small("aligned time axis — each row is one subject",
                       className="text-muted d-block mb-1"),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(id="ha-group-select",
                                 placeholder="Select group"),
                    width=4,
                ),
            ], className="mb-2"),
            dcc.Graph(id="ha-trigger-timeline"),
        ),

        _card("Signal Overlay",
            html.Small("HbO per subject (µmol/L) · first channel shown by default",
                       className="text-muted d-block mb-1"),
            dcc.Graph(id="ha-signal-overlay"),
        ),

        _card("Channel Decisions",
            html.Small("per-subject good/bad decisions · click chip to cycle · auto-saves",
                       className="text-muted d-block mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Cardiac Band (Hz)", className="small"),
                    dbc.InputGroup([
                        dbc.Input(id="ha-cardiac-l", type="number", step=0.1, placeholder="lo (adult ~0.7)"),
                        dbc.InputGroupText("–"),
                        dbc.Input(id="ha-cardiac-h", type="number", step=0.1, placeholder="hi (adult ~1.5)"),
                    ]),
                ], width=4),
            ], className="mb-2"),
            html.Div(id="ha-decisions-table",
                     children=html.Small("Select a group to rate channels.",
                                         className="text-muted")),
            html.Div(id="ha-decisions-status", className="mt-1 small text-muted"),
        ),

        _card("Export",
            dbc.Button("Export Aligned SNIRFs", id="ha-export-btn",
                       color="success"),
            html.Div(id="ha-export-status", className="mt-2 small"),
        ),
    ]),

], fluid=True)
