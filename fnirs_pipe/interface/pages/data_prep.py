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


def _card(title, *children):
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(list(children))],
        className="mb-3",
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

    # ── Row 1: Raw Signal (wide) | Marker Editor (narrow) ────────────────────
    # Mirrors HTML row-main: grid-template-columns 3fr 1fr
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
        dbc.Col(
            _marker_editor(),
            width=3,
        ),
    ], className="mb-3", align="start"),

    # ── Channel selector row (above Channel Detail, like HTML click-to-select) ─
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="dp-channel-selector", options=[],
                placeholder="Select channel pair · or click a trace above",
            ),
            width=5,
        ),
    ], className="mb-2"),

    # ── Row 2: Channel Detail (full width, title updates on selection) ────────
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

    # ── Row 3: PSD (full width, mean → per-channel on selection) ─────────────
    _card("PSD",
        html.Small("mean across channels",
                   id="dp-psd-subtitle", className="text-muted d-block mb-1"),
        dcc.Graph(id="dp-channel-psd"),
    ),

    # ── Row 4: Epoch preview (full width) ─────────────────────────────────────
    _card("Epoch Preview",
        html.Small("select a channel to view",
                   id="dp-epoch-subtitle", className="text-muted d-block mb-1"),
        dcc.Graph(id="dp-channel-epoch"),
    ),

    # ── Row 5: Layout 2D (narrow) | 3D (wide) ────────────────────────────────
    # Mirrors HTML row-layout: grid-template-columns 1fr 2fr
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

    # ── Row 6: SCI / PSP ─────────────────────────────────────────────────────
    _card("SCI / PSP", dcc.Graph(id="dp-sci-psp-figure")),

    # ── Row 7: IQM ───────────────────────────────────────────────────────────
    _card("Image Quality Metrics",
        dash_table.DataTable(
            id="dp-iqm-table",
            columns=_IQM_COLS,
            style_table={"overflowX": "auto"},
        ),
    ),

    # ── Row 8: Channel Quality Heatmap ────────────────────────────────────────
    _card("Channel Quality Summary",
          html.Small("status / SCI / CV / PSP / SNR per channel",
                     className="text-muted d-block mb-1"),
          dcc.Graph(id="dp-ch-summary-figure")),

], fluid=True)
