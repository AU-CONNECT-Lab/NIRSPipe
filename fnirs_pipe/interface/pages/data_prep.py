"""Page 1: individual and hyperscanning data preparation and modification."""

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from fnirs_pipe.interface.components import (
    PATH, action_field, actions, band, card, field, params, section, switches,
)
from fnirs_pipe.interface.grid import AUTO_HEIGHT, COL_DEF, DEL_COL


dash.register_page(__name__, path="/", name="Data Preparation")

# The rows come from the metric registry already labelled and formatted, so this only names
# the columns. `cls` carries the registry's verdict and `tip` its explanation; neither is
# printed, one reaches the Value cell as a colour and the other the Metric cell as a tooltip.
_SQM_COLS = [
    {"headerName": "Metric", "field": "label", "tooltipField": "tip"},
    {"headerName": "Value",  "field": "value"},
]

# the colours themselves are in assets/interface.css, keyed on these class names
_SQM_ROW_CLASS = {
    "qm-ok":   "data.cls == 'qm-ok'",
    "qm-warn": "data.cls == 'qm-warn'",
    "qm-bad":  "data.cls == 'qm-bad'",
}

# the DataTable's tooltip_duration=None never timed out; this is the nearest AG Grid has
_SQM_GRID = {**AUTO_HEIGHT, "tooltipShowDelay": 300, "tooltipHideDelay": 60000}

_NUM = {"editable": True, "cellDataType": "number", "cellEditor": "agNumberCellEditor"}

_SEG_COLS = [
    DEL_COL,
    {"headerName": "Onset (s)",    "field": "onset",    **_NUM},
    {"headerName": "Duration (s)", "field": "duration", **_NUM},
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
            dag.AgGrid(
                id="dp-crop-seg-table",
                columnDefs=_SEG_COLS,
                rowData=[],
                className="fp-grid fp-grid-sm fp-grid-capped-sm",
                columnSize="responsiveSizeToFit",
                defaultColDef=COL_DEF,
                dashGridOptions=AUTO_HEIGHT,
                style={"height": None},
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
        html.Div(id="dp-drift-hint", className="mt-2"),
        extra_class=extra_class,
    )


layout = dbc.Container([
    dcc.Download(id="dp-tsv-download"),
    dcc.Store(id="dp-marker-store",    storage_type="memory"),
    dcc.Store(id="dp-decisions-store", storage_type="memory"),
    dcc.Interval(id="dp-mount-tick", interval=150, max_intervals=1),

    dbc.Row([dbc.Col([html.H3("Data Preparation"), html.Hr()])]),

    # ── Setup ────────────────────────────────────────────────────────────────
    card("Setup",
        section("Data source",
            params(
                field("BIDS directory",
                      dbc.Input(id="dp-bids-dir", type="text", placeholder="/path/to/bids"),
                      span=PATH),
                field(["Output directory ", html.Span("*", className="text-danger")],
                      dbc.Input(id="dp-output-dir", type="text",
                                placeholder="/path/to/output  (required)"),
                      span=PATH),
            ),
        ),
        section("Subject",
            params(
                field("Subject ID",
                      dbc.Input(id="dp-manual-subject", type="text",
                                placeholder="e.g. 10031", debounce=True)),
                field("Run",
                      dcc.Dropdown(id="dp-run-dropdown", options=[],
                                   placeholder="Select run"),
                      span=PATH),
                action_field(
                    dbc.Button("Detect", id="dp-detect-btn", color="primary"),
                    dbc.Button("Clear",  id="dp-clear-btn",  color="secondary", outline=True),
                ),
            ),
            html.Div(id="dp-subjects-result",    className="mt-2"),
            html.Div(id="dp-subjects-container"),
        ),
        section("Parameters",
            params(
                field("SCI threshold",
                      dbc.Input(id="dp-sci-thresh", type="number", value=0.8,
                                min=0.0, max=1.0, step=0.01)),
                field("SCI / PSP window (s)",
                      dbc.Input(id="dp-window-s", type="number", value=10.0,
                                min=1.0, step=1.0)),
                field("DPF",
                      dbc.Input(id="dp-dpf", type="number", step=0.1, placeholder="e.g. 6.0")),
                band("Cardiac band (Hz)",
                     dbc.Input(id="dp-cardiac-l", type="number", step=0.1,
                               placeholder="lo (adult ~0.7)"),
                     "–",
                     dbc.Input(id="dp-cardiac-h", type="number", step=0.1,
                               placeholder="hi (adult ~1.5)")),
                band("Epoch window (s)",
                     dbc.Input(id="dp-epoch-tmin", type="number", step=0.5, value=-5.0),
                     "–",
                     dbc.Input(id="dp-epoch-tmax", type="number", step=0.5, value=25.0)),
                switches(dbc.Checklist(
                    id="dp-epoch-qc",
                    options=[{"label": "Per-trial QC", "value": "on"}],
                    value=[], switch=True, className="small",
                )),
                band("Separations (mm)",
                     "short ≤",
                     dbc.Input(id="dp-short-max-dist", type="number", step=0.5,
                               min=0.1, placeholder="10"),
                     "long ≥",
                     dbc.Input(id="dp-long-min-dist", type="number", step=0.5,
                               min=0.1, placeholder="15"),
                     "to",
                     dbc.Input(id="dp-long-max-dist", type="number", step=0.5,
                               min=0.1, placeholder="no limit"),
                     span=PATH),
            ),
            html.Div(id="dp-load-status", className="mt-2 small"),
         key="dp-params", open=False),
        section("Report",
            actions(
                dbc.Button("Write QC report", id="dp-report-btn",
                           color="secondary", outline=True),
            ),
            html.Small("Runs fnirs-qc prep-raw on the loaded run with the parameters above,"
                       " and writes the report and its quality record into the output"
                       " directory.", className="fp-hint d-block mt-1"),
            html.Div(id="dp-report-status", className="mt-2"),
            html.Div(id="dp-report-preview", className="mt-2"),
         key="dp-report", open=False),
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
                    dag.AgGrid(
                        id="dp-sqm-table",
                        columnDefs=_SQM_COLS,
                        rowData=[],
                        className="fp-grid",
                        columnSize="responsiveSizeToFit",
                        defaultColDef=COL_DEF,
                        rowClassRules=_SQM_ROW_CLASS,
                        dashGridOptions=_SQM_GRID,
                        style={"height": None},
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
