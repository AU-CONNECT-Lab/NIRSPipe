"""Page: one recording at a time, to look at and to edit its markers and its time span."""

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspipe.interface.components import (
    PATH, action_field, band, card, field, params, section, switches,
)
from nirspipe.interface.grid import AUTO_HEIGHT, COL_DEF, DEL_COL


dash.register_page(__name__, path="/", name="Data Preparation")

_NUM = {"editable": True, "cellDataType": "number", "cellEditor": "agNumberCellEditor"}

_SEG_COLS = [
    DEL_COL,
    {"headerName": "Onset (s)",    "field": "onset",    **_NUM},
    {"headerName": "Duration (s)", "field": "duration", **_NUM},
    {"headerName": "Task", "field": "task", "editable": True},
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
                    dbc.Input(id="dp-crop-tmin", type="number", step="any", size="sm",
                              placeholder="0.0", style={"width": "80px"}),
                    dbc.Label("tmax (s)", className="small mb-0"),
                    dbc.Input(id="dp-crop-tmax", type="number", step="any", size="sm",
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
                              min=0.001, step="any", size="sm",
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
            dbc.Col(html.Small("Onset (s)",   className="text-muted fw-semibold"), width=3),
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
                                placeholder="e.g. 01", debounce=True)),
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
                      dbc.Input(id="dp-sci-thresh", type="number", placeholder="e.g. 0.8",
                                min=0.0, max=1.0, step="any")),
                field("SCI / PSP window (s)",
                      dbc.Input(id="dp-window-s", type="number", value=10.0,
                                min=1.0, step="any")),
                field("PSP threshold",
                      dbc.Input(id="dp-psp-thresh", type="number", value=0.1,
                                min=0.0, max=1.0, step="any")),
                field("Coupled windows",
                      dbc.Input(id="dp-min-good-frac", type="number", value=0.75,
                                min=0.0, max=1.0, step="any"),
                      hint="Share a channel needs to be kept. This is what rejects."),
                field("Screening scope",
                      dcc.Dropdown(id="dp-screen-scope",
                                   options=[{"label": "Whole run", "value": "run"},
                                            {"label": "Task blocks only", "value": "task"}],
                                   value="run", clearable=False)),
                field("DPF",
                      dbc.Input(id="dp-dpf", type="number", step="any", placeholder="e.g. 6.0")),
                band("Cardiac band (Hz)",
                     dbc.Input(id="dp-cardiac-l", type="number", step="any",
                               placeholder="lo (adult ~0.7)"),
                     "–",
                     dbc.Input(id="dp-cardiac-h", type="number", step="any",
                               placeholder="hi (adult ~2.0)")),
                band("Epoch window (s)",
                     dbc.Input(id="dp-epoch-tmin", type="number", step="any", value=-5.0),
                     "–",
                     dbc.Input(id="dp-epoch-tmax", type="number", step="any", value=25.0)),
                band("Separations (mm)",
                     "short ≤",
                     dbc.Input(id="dp-short-max-dist", type="number", step="any",
                               min=0.1, placeholder="10"),
                     "long ≥",
                     dbc.Input(id="dp-long-min-dist", type="number", step="any",
                               min=0.1, placeholder="15"),
                     "to",
                     dbc.Input(id="dp-long-max-dist", type="number", step="any",
                               min=0.1, placeholder="no limit"),
                     span=PATH),
            ),
            switches(dbc.Checklist(
                id="dp-ignore-status",
                options=[{"label": "Ignore channels.tsv status", "value": "on"}],
                value=[], switch=True, className="small",
            )),
            html.Div(id="dp-load-status", className="mt-2 small"),
         key="dp-params", open=False),
    ),

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
                        # the figure's own width to height, its legend strip included
                        dcc.Graph(id="dp-evoked-topo", responsive=True,
                                  config={"showAxisDragHandles": False},
                                  style={"aspectRatio": "700 / 760"}),
                    ]),
                ),
                width=6,
            ),
            dbc.Col(
                _card("Optode Layout (2D)",
                      html.Small("colour = SCI · click = select channel",
                                 className="text-muted d-block mb-1"),
                      html.Div(id="dp-layout-2d-wrap", style=_HIDDEN, children=[
                          # as tall as Signal Topo beside it
                          dcc.Graph(id="dp-layout-2d", responsive=True,
                                    style={"aspectRatio": "700 / 760"}),
                      ])),
                width=6,
            ),
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
            dbc.Col(_crop_card(), width=8),
            dbc.Col(_marker_editor(), width=4),
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

], fluid=True)
