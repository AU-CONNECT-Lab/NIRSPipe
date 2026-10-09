"""Page: Batch data preparation: marker editing and crop across multiple subjects."""

from __future__ import annotations

import platform

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspipe.interface.components import (
    FULL, PATH, action_field, actions, card, field, params, section, split,
)
from nirspipe.interface.grid import AUTO_HEIGHT, COL_DEF, DEL_COL

dash.register_page(__name__, path="/batch-prep", name="Batch Prep")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_RENAME_COLS = [
    DEL_COL,
    {"headerName": "From", "field": "from_name", "editable": True},
    {"headerName": "To",   "field": "to_name",   "editable": True},
]

_NUM = {"editable": True, "cellDataType": "number", "cellEditor": "agNumberCellEditor"}

_SEG_COLS = [
    DEL_COL,
    {"headerName": "Onset (s)",    "field": "onset",    **_NUM},
    {"headerName": "Duration (s)", "field": "duration", **_NUM},
    {"headerName": "Task", "field": "task", "editable": True},
]

_HIDDEN = {"display": "none"}


def _scope():
    return card("Scope",
        section("Data source",
            params(
                field("BIDS directory",
                      dbc.Input(id="bp-bids-dir", type="text", placeholder="/path/to/bids"),
                      span=PATH),
                field("Derivatives directory",
                      dbc.Input(id="bp-deriv-dir", type="text",
                                placeholder="/path/to/derivatives"),
                      span=PATH),
            ),
        ),
        # both sections are hidden whole when the operation does not select subjects
        html.Div(id="bp-subjects-card", children=section("Subjects",
            actions(
                dbc.Button("Detect", id="bp-detect-btn", color="primary", size="sm"),
                dbc.Button("Select all", id="bp-select-all-btn",
                           color="outline-secondary", size="sm"),
                dbc.Button("Clear", id="bp-clear-btn",
                           color="outline-secondary", size="sm"),
            ),
            html.Div(id="bp-subjects-result", className="mt-2 mb-2"),
            html.Div(id="bp-subjects-container"),
        )),
        html.Div(id="bp-run-filter-card", children=section("Run filter (optional)",
            params(
                field("Session", dbc.Input(id="bp-ses", type="text", placeholder="e.g. 01")),
                field("Task", dbc.Input(id="bp-task", type="text", placeholder="e.g. tapping")),
                field("Run", dbc.Input(id="bp-run", type="text", placeholder="e.g. 01")),
            ),
        )),
    )


def _operation():
    return card("Operation",
        dbc.RadioItems(
            id="bp-operation",
            options=[{"label": "Edit markers", "value": "markers"},
                     {"label": "Crop", "value": "crop"},
                     {"label": "Hyperscanning align", "value": "hyper_align"}],
            value="markers", inline=True, className="mb-3",
        ),

        html.Div(id="bp-hyper-panel", style=_HIDDEN, children=[
            params(
                field("Group CSV",
                      dbc.Input(id="bp-group-csv", type="text",
                                placeholder="/path/to/groups.csv"),
                      span=PATH,
                      hint="Columns: group_id, subject_id, task."
                           " Each (group_id, task) pair is aligned independently."),
            ),
        ]),

        html.Div(id="bp-markers-panel", children=[
            dbc.RadioItems(
                id="bp-marker-op",
                options=[{"label": "Shift onsets", "value": "shift"},
                         {"label": "Set duration", "value": "set_duration"},
                         {"label": "Rename markers", "value": "rename"}],
                value="shift", inline=True, className="mb-3 small",
            ),

            html.Div(id="bp-shift-panel", children=[
                params(
                    field("Shift (s)",
                          dbc.Input(id="bp-shift-val", type="number"),
                          hint="negative = earlier; clipped to 0"),
                ),
            ]),

            html.Div(id="bp-duration-panel", style=_HIDDEN, children=[
                params(
                    field("Duration (s)", dbc.Input(id="bp-duration-val", type="number")),
                ),
            ]),

            html.Div(id="bp-rename-panel", style=_HIDDEN, children=[
                dag.AgGrid(
                    id="bp-rename-table",
                    columnDefs=_RENAME_COLS,
                    rowData=[{"from_name": "", "to_name": ""}],
                    className="fp-grid fp-grid-sm fp-grid-capped",
                    columnSize="responsiveSizeToFit",
                    defaultColDef=COL_DEF,
                    dashGridOptions=AUTO_HEIGHT,
                    style={"height": None},
                ),
                dbc.Button("+ Add pair", id="bp-rename-add-btn", size="sm",
                           color="outline-secondary", className="mt-2"),
            ]),
        ]),

        html.Div(id="bp-crop-panel", style=_HIDDEN, children=[
            dbc.RadioItems(
                id="bp-crop-mode",
                options=[{"label": "Single segment", "value": "single"},
                         {"label": "Multi-segment", "value": "multi"}],
                value="single", inline=True, className="mb-3 small",
            ),

            html.Div(id="bp-crop-single", children=[
                params(
                    field("tmin (s)", dbc.Input(id="bp-crop-tmin", type="number")),
                    field("tmax (s)", dbc.Input(id="bp-crop-tmax", type="number")),
                ),
            ]),

            html.Div(id="bp-crop-multi", style=_HIDDEN, children=[
                dag.AgGrid(
                    id="bp-crop-seg-table",
                    columnDefs=_SEG_COLS,
                    rowData=[{"onset": 0.0, "duration": 30.0}],
                    className="fp-grid fp-grid-sm fp-grid-capped",
                    columnSize="responsiveSizeToFit",
                    defaultColDef=COL_DEF,
                    dashGridOptions=AUTO_HEIGHT,
                    style={"height": None},
                ),
                actions(
                    dbc.Button("+ Segment", id="bp-seg-add-btn", size="sm",
                               color="outline-secondary"),
                    dbc.Checklist(
                        id="bp-crop-combine",
                        options=[{"label": "Combine into one file", "value": "combine"}],
                        value=[], inline=True, className="small",
                    ),
                    className="mt-2",
                ),
            ]),
        ]),
    )


def _run_panel():
    return card("Run",
        params(
            field("Parallel jobs",
                  dbc.Input(id="bp-n-jobs", type="number", value=1, min=1, step=1)),
            action_field(
                dbc.Button("Run batch", id="bp-run-btn", color="success"),
                dbc.Button("Stop", id="bp-stop-btn", color="danger",
                           outline=True, disabled=True),
                dbc.Button("Generate command", id="bp-generate-btn",
                           color="secondary", outline=True),
                span=FULL,
            ),
        ),
        html.Div(id="bp-log", className="mt-3"),
    )


def _command_panel():
    return card("Command",
        dbc.Select(
            id="bp-shell-select",
            options=[{"label": s, "value": s} for s in ("bash", "cmd", "powershell")],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mb-2",
        ),
        html.Pre(id="bp-command-preview", className="fp-command"),
        subtitle="what this form is, as nirspipe-prep",
    )


layout = dbc.Container([
    dcc.Store(id="bp-command-store"),
    dcc.Store(id="bp-run-store", storage_type="memory"),
    # the run is polled rather than waited on; disabled until there is something to poll
    dcc.Interval(id="bp-run-tick", interval=1000, disabled=True),

    html.H3("Batch Preparation"),
    html.Hr(),

    split(
        main=[_scope(), _operation()],
        aside=[_run_panel(), _command_panel()],
    ),
], fluid=True)
