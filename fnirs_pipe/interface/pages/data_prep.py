"""Page 1: individual and hyperscanning data preparation and modification."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html


dash.register_page(__name__, path="/", name="Data Preparation")

# The rows come from the metric registry already labelled and formatted, so this only
# names the columns. `cls` carries the registry's verdict and drives the row colouring in
# _SQM_STYLE rather than being printed.
_SQM_COLS = [
    {"name": "Metric", "id": "label"},
    {"name": "Value",  "id": "value"},
]

_SQM_STYLE = [
    {"if": {"filter_query": "{cls} = 'qm-ok'",   "column_id": "value"}, "color": "#27ae60"},
    {"if": {"filter_query": "{cls} = 'qm-warn'", "column_id": "value"}, "color": "#e67e22"},
    {"if": {"filter_query": "{cls} = 'qm-bad'",  "column_id": "value"}, "color": "#c0392b"},
]

_SEG_COLS = [
    {"name": "Onset (s)",    "id": "onset",    "editable": True, "type": "numeric"},
    {"name": "Duration (s)", "id": "duration", "editable": True, "type": "numeric"},
]

_HIDDEN = {"display": "none"}


def _card(title, *children, extra_class="", body_class="", body_style=None):
    cls = ("mb-3 " + extra_class).strip() if extra_class else "mb-3"
    return dbc.Card(
        [dbc.CardHeader(title),
         dbc.CardBody(list(children), className=body_class or None, style=body_style)],
        className=cls,
    )


def _crop_card(extra_class=""):
    return _card("Crop",
        html.Div(id="dp-trigger-timeline-wrap", style=_HIDDEN, children=[
            dcc.Graph(id="dp-trigger-timeline", style={"minHeight": "300px"}),
        ]),
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
        extra_class=extra_class,
        body_class="p-2",
    )


def _marker_editor(extra_class=""):
    return _card("Markers",
        dbc.Row([
            dbc.Col(dbc.Label("Step (s)", className="small mb-0"), width="auto",
                    className="d-flex align-items-center pe-0"),
            dbc.Col(dbc.Input(id="dp-step-input", type="number", value=1.0,
                              min=0.001, step=0.1, size="sm",
                              style={"width": "64px"}), width="auto"),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("− All", id="dp-offset-all-minus", size="sm",
                           color="outline-secondary"),
                dbc.Button("+ All", id="dp-offset-all-plus",  size="sm",
                           color="outline-secondary"),
                dbc.Button("− Sel", id="dp-offset-sel-minus", size="sm",
                           color="outline-danger", disabled=True),
                dbc.Button("+ Sel", id="dp-offset-sel-plus",  size="sm",
                           color="outline-secondary", disabled=True),
            ]), width="auto"),
        ], className="g-1 mb-2 align-items-center flex-wrap"),
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
            dbc.Col(dbc.Button("Apply All", id="dp-batch-rename-btn",
                               size="sm", color="outline-secondary"), width="auto"),
        ], className="g-1 mb-2 align-items-center flex-wrap"),
        dbc.Row([
            dbc.Col(width=1),
            dbc.Col(html.Small("Onset (s)",   className="text-muted fw-semibold"), width=4),
            dbc.Col(html.Small("Dur (s)",     className="text-muted fw-semibold"), width=3),
            dbc.Col(html.Small("Description", className="text-muted fw-semibold")),
        ], className="g-1 px-2 mb-1"),
        html.Div(id="dp-marker-rows-container",
                 style={"maxHeight": "300px", "overflowY": "auto"}),
        dbc.Row([
            dbc.Col(
                dbc.ButtonGroup([
                    dbc.Button("+ Add",        id="dp-add-marker-btn",
                               color="outline-secondary", size="sm"),
                    dbc.Button("↓ Export TSV", id="dp-export-tsv-btn",
                               color="primary",           size="sm"),
                ]),
                width="auto",
            ),
            dbc.Col(
                dbc.Button("Save", id="dp-save-markers-btn",
                           color="outline-success", size="sm"),
                width="auto", className="ms-auto",
            ),
        ], className="g-1 mt-2 align-items-center"),
        html.Div(id="dp-save-status", className="mt-2 small"),
        extra_class=extra_class,
    )


layout = dbc.Container([
    dcc.Download(id="dp-tsv-download"),
    dcc.Store(id="dp-marker-store",    storage_type="memory"),
    dcc.Store(id="dp-decisions-store", storage_type="memory"),
    dcc.Interval(id="dp-mount-tick", interval=150, max_intervals=1),

    dbc.Row([dbc.Col([html.H3("Data Preparation"), html.Hr()])]),

    # ── Data Source ──────────────────────────────────────────────────────────
    _card("Data Source",
        dbc.Row([
            dbc.Col([
                dbc.Label("BIDS Directory"),
                dbc.Input(id="dp-bids-dir", type="text",
                          placeholder="/path/to/bids"),
            ], width=6),
            dbc.Col([
                dbc.Label(["Output Directory ", html.Span("*", className="text-danger")]),
                dbc.Input(id="dp-output-dir", type="text",
                          placeholder="/path/to/output  (required)"),
            ], width=6),
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
            dbc.Col([
                dbc.Label("Run"),
                dcc.Dropdown(id="dp-run-dropdown", options=[],
                             placeholder="Select run"),
            ]),
        ], className="g-3 mb-3"),
        html.Div(id="dp-subjects-result",    className="mb-2"),
        html.Div(id="dp-subjects-container"),
    ),

    # ── Parameters ───────────────────────────────────────────────────────────
    _card("Parameters",
        dbc.Row([
            dbc.Col([
                dbc.Label("SCI Threshold"),
                dbc.Input(id="dp-sci-thresh", type="number", value=0.8,
                          min=0.0, max=1.0, step=0.01),
            ], width=3),
            dbc.Col([
                dbc.Label("Cardiac Band (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="dp-cardiac-l", type="number", step=0.1, placeholder="lo (adult ~0.7)"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="dp-cardiac-h", type="number", step=0.1, placeholder="hi (adult ~1.5)"),
                ]),
            ], width=6),
            dbc.Col([
                dbc.Label("DPF"),
                dbc.Input(id="dp-dpf", type="number", step=0.1, placeholder="e.g. 6.0"),
            ], width=3),
        ], className="g-3 align-items-end"),
        dbc.Row([
            dbc.Col([
                dbc.Label("SCI / PSP Window (s)"),
                dbc.Input(id="dp-window-s", type="number", value=10.0, min=1.0, step=1.0),
            ], width=3),
            dbc.Col([
                dbc.Label("Epoch Window (s)"),
                dbc.InputGroup([
                    dbc.Input(id="dp-epoch-tmin", type="number", step=0.5, value=-5.0),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="dp-epoch-tmax", type="number", step=0.5, value=25.0),
                ]),
            ], width=5),
            dbc.Col(
                dbc.Checklist(
                    id="dp-epoch-qc",
                    options=[{"label": "Per-trial QC", "value": "on"}],
                    value=[], switch=True, className="small",
                ),
                width=3, className="d-flex align-items-end pb-2",
            ),
        ], className="g-3 mt-1 align-items-end"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Separations (mm)"),
                dbc.InputGroup([
                    dbc.InputGroupText("short ≤"),
                    dbc.Input(id="dp-short-max-dist", type="number", step=0.5,
                              min=0.1, placeholder="10"),
                    dbc.InputGroupText("long ≥"),
                    dbc.Input(id="dp-long-min-dist", type="number", step=0.5,
                              min=0.1, placeholder="15"),
                    dbc.InputGroupText("to"),
                    dbc.Input(id="dp-long-max-dist", type="number", step=0.5,
                              min=0.1, placeholder="no limit"),
                ]),
            ], width=7),
        ], className="g-3 mt-1 align-items-end"),
        html.Div(id="dp-load-status", className="mt-2 small"),
    ),

    dbc.Tabs([

        # ════════════════════════════════════════════════════════════════════
        dbc.Tab(label="Viewer", tab_id="tab-viewer", children=[
            html.Div(className="mt-3", children=[

                # Signal Topo + Optode layouts side by side
                dbc.Row([
                    dbc.Col(
                        _card("Signal Topo",
                            html.Small("raw HbO / HbR per channel · click to select",
                                       className="text-muted d-block mb-1"),
                            html.Div(id="dp-evoked-topo-wrap", style=_HIDDEN, children=[
                                # plotly's 20px axis drag handles blanket cells this small,
                                # so a neighbour's handle takes the click
                                dcc.Graph(id="dp-evoked-topo", responsive=True,
                                          config={"showAxisDragHandles": False},
                                          style={"aspectRatio": "1 / 1"}),
                            ]),
                        ),
                        width=6,
                    ),
                    dbc.Col([
                        _card("Optode Layout (2D)",
                              html.Small("colour = SCI · click = select channel",
                                         className="text-muted d-block mb-1"),
                              html.Div(id="dp-layout-2d-wrap", style=_HIDDEN, children=[
                                  dcc.Graph(id="dp-layout-2d", responsive=True,
                                            style={"height": "370px"}),
                              ]),
                              body_class="p-2"),
                        _card("3D (fsaverage)",
                              html.Div(id="dp-layout-3d-wrap", style=_HIDDEN, children=[
                                  dcc.Graph(id="dp-layout-3d", responsive=True,
                                            style={"height": "320px"}),
                              ])),
                    ], width=6),
                ], className="mb-3", align="start"),

                # Channel Detail (full width)
                dbc.Card([
                    dbc.CardHeader(html.Span([
                        "Channel Detail",
                        html.Small(
                            " · click a channel trace to view HbO / HbR",
                            id="dp-detail-title",
                            className="text-muted fw-normal",
                        ),
                    ])),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="dp-channel-selector", options=[],
                            placeholder="Select channel pair · or click a trace",
                            className="mb-2",
                        ),
                        html.Div(id="dp-channel-detail-wrap", style=_HIDDEN, children=[
                            dcc.Graph(id="dp-channel-detail"),
                        ]),
                        # this channel's own PSD, kept in the same card as its timeseries so
                        # it cannot be mistaken for the all-channel panel below
                        html.Div(id="dp-channel-psd-wrap", style=_HIDDEN, children=[
                            html.Small("PSD · HbO / HbR concentration, after Beer-Lambert",
                                       className="text-muted d-block mb-1"),
                            dcc.Graph(id="dp-channel-psd"),
                        ]),
                    ]),
                ], className="mb-3"),

                # PSD (below Channel Detail)
                _card("PSD",
                    html.Small("mean across channels · optical density, before Beer-Lambert",
                               className="text-muted d-block mb-1"),
                    html.Div(id="dp-psd-mean-wrap", style=_HIDDEN, children=[
                        dcc.Graph(id="dp-psd-mean"),
                    ]),
                ),

                # Crop | Markers
                dbc.Row([
                    dbc.Col(_crop_card(), width=9),
                    dbc.Col(_marker_editor(), width=3),
                ], className="mb-3", align="start"),

                # Epoch Preview
                _card("Epoch Preview",
                    html.Small("select a channel to view",
                               className="text-muted d-block mb-1"),
                    html.Div(id="dp-channel-epoch-wrap", style=_HIDDEN, children=[
                        dcc.Graph(id="dp-channel-epoch"),
                    ]),
                ),

            ]),
        ]),

        # ════════════════════════════════════════════════════════════════════
        dbc.Tab(label="QC Metrics", tab_id="tab-qc", children=[
            html.Div(className="mt-3", children=[

                dbc.Card([
                    dbc.CardHeader(html.Span([
                        "Raw Signal",
                        html.Small(
                            " · blue = short channel · orange = long"
                            " · click trace = select channel · double-click = reset",
                            className="text-muted fw-normal",
                        ),
                    ])),
                    dbc.CardBody([
                        html.Div(id="dp-ts-figure-wrap", style=_HIDDEN, children=[
                            dcc.Graph(id="dp-ts-figure"),
                        ]),
                    ]),
                ], className="mb-3"),

                _card("Carpet + GVTD",
                      html.Div(id="dp-carpet-gvtd-wrap", style=_HIDDEN, children=[
                          dcc.Graph(id="dp-carpet-gvtd", responsive=True,
                                    style={"minHeight": "600px"}),
                      ])),

                _card("SCI / PSP",
                      html.Div(id="dp-sci-psp-figure-wrap", style=_HIDDEN, children=[
                          dcc.Graph(id="dp-sci-psp-figure", style={"minHeight": "400px"}),
                      ])),

                _card("Image Quality Metrics",
                    html.Small(id="dp-sqm-scope",
                               className="text-muted d-block mb-1"),
                    dash_table.DataTable(
                        id="dp-sqm-table",
                        columns=_SQM_COLS,
                        tooltip_data=[],
                        tooltip_duration=None,
                        style_table={"overflowX": "auto"},
                        style_data_conditional=_SQM_STYLE,
                        style_cell={"fontSize": "0.82rem", "textAlign": "left"},
                    ),
                    html.Div(id="dp-sqm-split", className="mt-2"),
                ),

                _card("Per-trial QC",
                      html.Small("one column per trial, scored over the epoch window"
                                 " · needs Per-trial QC switched on",
                                 className="text-muted d-block mb-1"),
                      html.Div(id="dp-trial-qc-wrap", style=_HIDDEN, children=[
                          dcc.Graph(id="dp-trial-qc", style={"minHeight": "400px"}),
                      ])),

                _card("Channel Quality Summary",
                      html.Small("status / SCI / CV / PSP / SNR per channel · status "
                                 "screens on the coupled-window share; SCI / PSP / CV / "
                                 "SNR are reported only",
                                 className="text-muted d-block mb-1"),
                      html.Div(id="dp-ch-summary-figure-wrap", style=_HIDDEN, children=[
                          dcc.Graph(id="dp-ch-summary-figure", style={"minHeight": "400px"}),
                      ])),

                _card("Per-channel Metrics",
                      html.Small("one row per source-detector pair · click chip to cycle: "
                                 "— → good → bad · auto-saves",
                                 className="text-muted d-block mb-2"),
                      html.Div(id="dp-decisions-table",
                               children=html.Small("Load a run to rate channels.",
                                                   className="text-muted")),
                      html.Div(id="dp-decisions-status", className="mt-1 small text-muted")),

            ]),
        ]),

    ], id="dp-tabs", active_tab="tab-viewer"),

], fluid=True)
