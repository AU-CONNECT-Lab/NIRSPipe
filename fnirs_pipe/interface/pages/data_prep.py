"""Page 1: individual and hyperscanning data preparation and modification."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

dash.register_page(__name__, path="/", name="Data Preparation")

_MARKER_COLS = [
    {"name": "Onset (s)",    "id": "onset",      "editable": True, "type": "numeric"},
    {"name": "Duration (s)", "id": "duration",   "editable": True, "type": "numeric"},
    {"name": "Trial Type",   "id": "trial_type", "editable": True},
]

_IQM_COLS = [
    {"name": "Metric", "id": "metric"},
    {"name": "Value",  "id": "value"},
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


def _crop_card():
    return _card("Crop",
        dcc.Graph(id="dp-trigger-timeline", style={"height": "110px"}),
        dbc.Row([
            dbc.Col(
                dbc.RadioItems(
                    id="dp-crop-mode",
                    options=[
                        {"label": "Single segment",  "value": "single"},
                        {"label": "Multi-segment", "value": "multi"},
                    ],
                    value="single", inline=True, className="small",
                ),
                width="auto", className="d-flex align-items-center",
            ),
            dbc.Col(
                html.Div(id="dp-crop-single-panel",
                         className="d-flex gap-2 align-items-center flex-wrap",
                         children=[
                    dbc.Label("tmin (s)", className="small mb-0"),
                    dbc.Input(id="dp-crop-tmin", type="number", size="sm",
                              placeholder="0.0", style={"width": "80px"}),
                    dbc.Label("tmax (s)", className="small mb-0"),
                    dbc.Input(id="dp-crop-tmax", type="number", size="sm",
                              placeholder="end", style={"width": "80px"}),
                    dbc.Button("Use zoom", id="dp-crop-use-zoom",
                               size="sm", color="outline-secondary"),
                ]),
                width="auto",
            ),
            dbc.Col(
                html.Div(id="dp-crop-multi-panel", style={"display": "none"},
                         className="d-flex gap-2 align-items-center flex-wrap",
                         children=[
                    dbc.Button("+ Segment", id="dp-crop-add-seg-btn",
                               size="sm", color="outline-secondary"),
                    dbc.Checklist(
                        id="dp-crop-combine",
                        options=[{"label": "Combine into one file", "value": "combine"}],
                        value=[], inline=True, className="small",
                    ),
                ]),
                width="auto",
            ),
            dbc.Col(
                dbc.Button("Apply", id="dp-crop-apply-btn",
                           size="sm", color="outline-success"),
                width="auto", className="ms-auto d-flex align-items-center",
            ),
        ], className="g-2 mt-2 align-items-center flex-wrap"),
        html.Div(id="dp-crop-seg-wrap", style={"display": "none"}, className="mt-2",
                 children=[
            dash_table.DataTable(
                id="dp-crop-seg-table",
                columns=_SEG_COLS,
                editable=True,
                row_deletable=True,
                style_table={"overflowX": "auto", "maxHeight": "150px", "overflowY": "auto"},
                style_header={"fontWeight": "600", "fontSize": "0.78rem"},
                style_cell={"fontSize": "0.78rem", "padding": "3px 6px"},
            ),
        ]),
        html.Div(id="dp-crop-status", className="mt-1 small"),
    )


def _marker_editor():
    return _card("Markers",
        # Step offset row
        dbc.Row([
            dbc.Col(dbc.Label("Step (s)", className="small mb-0"), width="auto",
                    className="d-flex align-items-center pe-0"),
            dbc.Col(dbc.Input(id="dp-step-input", type="number", value=1.0,
                              min=0.001, step=0.1, size="sm",
                              style={"width": "64px"}), width="auto"),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("−All", id="dp-offset-all-minus", size="sm",
                           color="outline-secondary"),
                dbc.Button("+All", id="dp-offset-all-plus",  size="sm",
                           color="outline-secondary"),
                dbc.Button("−Sel", id="dp-offset-sel-minus", size="sm",
                           color="outline-danger", disabled=True),
                dbc.Button("+Sel", id="dp-offset-sel-plus",  size="sm",
                           color="outline-secondary", disabled=True),
            ]), width="auto"),
        ], className="g-1 mb-2 align-items-center flex-wrap"),
        # Batch rename row
        dbc.Row([
            dbc.Col(dbc.Label("Rename", className="small mb-0"), width="auto",
                    className="d-flex align-items-center pe-0"),
            dbc.Col(dbc.Input(id="dp-rename-from", type="text",
                              placeholder="old", size="sm",
                              style={"width": "72px"}), width="auto"),
            dbc.Col(html.Span("→", className="small text-muted"), width="auto",
                    className="d-flex align-items-center px-1"),
            dbc.Col(dbc.Input(id="dp-rename-to", type="text",
                              placeholder="new", size="sm",
                              style={"width": "72px"}), width="auto"),
            dbc.Col(dbc.Button("Apply", id="dp-batch-rename-btn",
                               size="sm", color="outline-secondary"), width="auto"),
        ], className="g-1 mb-2 align-items-center flex-wrap"),
        # Table
        dash_table.DataTable(
            id="dp-marker-table",
            columns=_MARKER_COLS,
            editable=True,
            row_deletable=True,
            row_selectable="single",
            style_table={"overflowX": "auto", "maxHeight": "260px",
                         "overflowY": "auto"},
            style_header={"fontWeight": "600", "fontSize": "0.78rem"},
            style_cell={"fontSize": "0.78rem", "padding": "3px 6px"},
        ),
        # Toolbar
        dbc.ButtonGroup([
            dbc.Button("+ Add",       id="dp-add-marker-btn",
                       color="outline-secondary", size="sm"),
            dbc.Button("↓ TSV",       id="dp-export-tsv-btn",
                       color="outline-primary",   size="sm"),
            dbc.Button("Save",        id="dp-save-markers-btn",
                       color="outline-success",   size="sm"),
        ], className="mt-2"),
        html.Div(id="dp-save-status", className="mt-2 small"),
    )


layout = dbc.Container([
    dcc.Download(id="dp-tsv-download"),
    dcc.Interval(id="dp-mount-tick", interval=150, max_intervals=1),

    dbc.Row([dbc.Col([html.H3("Data Preparation"), html.Hr()])]),

    # ── Data Source ──────────────────────────────────────────────────────────
    _card("Data Source",
        dbc.Row([
            dbc.Col([
                dbc.Label("BIDS Directory"),
                dbc.Input(id="dp-bids-dir", type="text",
                          placeholder="/path/to/bids"),
            ], width=4),
            dbc.Col([
                dbc.Label("Output Directory"),
                dbc.Input(id="dp-output-dir", type="text",
                          placeholder="/path/to/output"),
            ], width=4),
            dbc.Col([
                dbc.Label("Cache Directory"),
                dbc.Input(id="dp-cache-dir", type="text",
                          placeholder="/path/to/cache  (optional)"),
            ], width=4),
        ], className="g-3"),
    ),

    # ── Subject Selection ────────────────────────────────────────────────────
    _card("Subject Selection",
        dbc.Row([
            dbc.Col([
                dbc.Label("Enter Subject ID"),
                dbc.Input(id="dp-manual-subject", type="text",
                          placeholder="e.g. 10031", debounce=True),
            ], width=5),
            dbc.Col([
                dbc.Label(""),
                dbc.ButtonGroup([
                    dbc.Button("Detect", id="dp-detect-btn", color="primary"),
                    dbc.Button("Clear",  id="dp-clear-btn",  color="secondary"),
                ]),
            ], width="auto", className="d-flex align-items-end"),
        ], className="g-3 mb-3"),
        html.Div(id="dp-subjects-result",    className="mb-2"),
        html.Div(id="dp-subjects-container"),
    ),

    # ── Run ──────────────────────────────────────────────────────────────────
    _card("Run",
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(id="dp-run-dropdown", options=[],
                             placeholder="Select run"),
            ], width=5),
            dbc.Col([
                dbc.Label("SCI Threshold"),
                dbc.Input(id="dp-sci-thresh", type="number", value=0.8,
                          min=0.0, max=1.0, step=0.01),
            ], width=2),
            dbc.Col([
                dbc.Label(" "),
                dbc.Button("Load Run", id="dp-load-btn", color="success",
                           className="d-block w-100"),
            ], width="auto", className="d-flex align-items-end"),
        ], className="g-3 align-items-end"),
        html.Div(id="dp-load-status", className="mt-2 small"),
    ),

    dbc.Tabs([

        # ════════════════════════════════════════════════════════════════════
        dbc.Tab(label="Viewer", tab_id="tab-viewer", children=[
            html.Div(className="mt-3", children=[

                # Topo (full width, top of viewer)
                _card("Signal Topo",
                    html.Small("raw HbO / HbR per channel · click trace to select",
                               className="text-muted d-block mb-1"),
                    dcc.Graph(id="dp-evoked-topo"),
                ),

                # Crop (trigger timeline + crop controls)
                _crop_card(),

                # Raw Signal | Marker Editor
                dbc.Row([
                    dbc.Col(
                        _card("Raw Signal",
                            html.Small(
                                "blue = short channel · orange = long · "
                                "click trace = select channel",
                                className="text-muted d-block mb-1",
                            ),
                            dcc.Graph(id="dp-ts-figure"),
                        ),
                        width=9,
                    ),
                    dbc.Col(_marker_editor(), width=3),
                ], className="mb-3", align="start"),

                # Channel selector
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id="dp-channel-selector", options=[],
                            placeholder="Select channel pair · or click a trace above",
                        ),
                        width=5,
                    ),
                ], className="mb-2"),

                # Channel Detail
                dbc.Card([
                    dbc.CardHeader(html.Span([
                        "Channel Detail",
                        html.Small(
                            " · click a channel trace to view HbO / HbR",
                            id="dp-detail-title",
                            className="text-muted fw-normal",
                        ),
                    ])),
                    dbc.CardBody(dcc.Graph(id="dp-channel-detail")),
                ], className="mb-3"),

                # PSD
                _card("PSD",
                    html.Small("mean across channels",
                               id="dp-psd-subtitle", className="text-muted d-block mb-1"),
                    dcc.Graph(id="dp-channel-psd"),
                ),

                # Epoch Preview
                _card("Epoch Preview",
                    html.Small("select a channel to view",
                               id="dp-epoch-subtitle", className="text-muted d-block mb-1"),
                    dcc.Graph(id="dp-channel-epoch"),
                ),

                # Optode 2D | 3D
                dbc.Row([
                    dbc.Col(
                        _card("Optode Layout (2D)",
                              html.Small("colour = SCI · click = select channel",
                                         className="text-muted d-block mb-1"),
                              dcc.Graph(id="dp-layout-2d")),
                        width=4,
                    ),
                    dbc.Col(
                        _card("3D (fsaverage)", dcc.Graph(id="dp-layout-3d")),
                        width=8,
                    ),
                ], className="mb-3"),

            ]),
        ]),

        # ════════════════════════════════════════════════════════════════════
        dbc.Tab(label="QC Metrics", tab_id="tab-qc", children=[
            html.Div(className="mt-3", children=[

                _card("SCI / PSP",
                      dcc.Graph(id="dp-sci-psp-figure",
                                style={"minHeight": "400px"})),

                _card("Image Quality Metrics",
                    dash_table.DataTable(
                        id="dp-iqm-table",
                        columns=_IQM_COLS,
                        style_table={"overflowX": "auto"},
                    ),
                ),

                _card("Channel Quality Summary",
                      html.Small("status / SCI / CV / PSP / SNR per channel",
                                 className="text-muted d-block mb-1"),
                      dcc.Graph(id="dp-ch-summary-figure",
                                style={"minHeight": "400px"})),

            ]),
        ]),

    ], id="dp-tabs", active_tab="tab-viewer"),

], fluid=True)
