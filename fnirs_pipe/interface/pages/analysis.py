"""Page 2: pipeline analysis configuration and execution."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import dcc, html

dash.register_page(__name__, path="/analysis", name="Analysis")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_DAG_COLOURS = {
    "prep": "#E67E22",
    "post": "#2980B9",
    "glm":  "#27AE60",
    "qc":   "#8E44AD",
}

_DAG_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "content":          "data(label)",
            "text-valign":      "center",
            "text-halign":      "center",
            "text-wrap":        "wrap",
            "shape":            "round-rectangle",
            "width":            "140px",
            "height":           "44px",
            "font-size":        "12px",
            "font-family":      "sans-serif",
            "background-opacity": 0,
            "border-width":     2,
            "color":            "#374151",
            "font-weight":      "bold",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style":        "bezier",
            "width":              1.5,
            "target-arrow-shape": "triangle",
            "arrow-scale":        0.9,
        },
    },
] + [
    {
        "selector": f".{cls}",
        "style": {
            "border-color":       colour,
            "line-color":         colour,
            "target-arrow-color": colour,
        },
    }
    for cls, colour in _DAG_COLOURS.items()
]


def _dag_legend():
    labels = [
        ("prep", "Preprocessing"),
        ("post", "Postprocessing"),
        ("glm",  "GLM"),
    ]
    return html.Div([
        html.Span(
            label,
            style={
                "display":      "inline-block",
                "marginRight":  "12px",
                "padding":      "2px 8px",
                "borderRadius": "4px",
                "border":       f"2px solid {_DAG_COLOURS[cls]}",
                "color":        "#374151",
                "fontSize":     "12px",
                "fontWeight":   "bold",
            },
        )
        for cls, label in labels
    ], className="mb-2")


def _dag_card():
    return _card(
        "Pipeline DAG",
        _dag_legend(),
        cyto.Cytoscape(
            id="an-dag",
            elements=[],
            stylesheet=_DAG_STYLESHEET,
            style={"width": "100%", "height": "260px", "background": "#FAFAFA"},
            layout={
                "name":          "dagre",
                "rankDir":       "LR",
                "nodeSep":       35,
                "rankSep":       80,
                "spacingFactor": 1.0,
            },
            userZoomingEnabled=True,
            userPanningEnabled=True,
            autoungrabify=True,
        ),
    )


def _card(title, *children):
    return dbc.Card(
        [dbc.CardHeader(title), dbc.CardBody(list(children))],
        className="mb-3",
    )


layout = dbc.Container([
    dcc.Store(id="an-subjects-store", storage_type="memory"),
    dcc.Store(id="an-command-store",  storage_type="memory"),
    dbc.Row([dbc.Col([html.H3("Analysis"), html.Hr()])]),

    # ── Data source ───────────────────────────────────────────────────────────
    _card("Data Source",
        dbc.Row([
            dbc.Col([
                dbc.Label("BIDS Directory"),
                dbc.Input(id="an-bids-dir", type="text", placeholder="/path/to/bids"),
            ], width=5),
            dbc.Col([
                dbc.Label("Output Directory"),
                dbc.Input(id="an-output-dir", type="text", placeholder="/path/to/output"),
            ], width=5),
            dbc.Col([
                dbc.Label(" "),
                dbc.Button("Detect Subjects", id="an-detect-btn",
                           color="primary", className="d-block w-100"),
            ], width=2),
        ], className="align-items-end g-3"),
    ),

    # ── Subject selection ─────────────────────────────────────────────────────
    _card("Subject Selection",
        html.Div(id="an-subjects-result", className="mb-2"),
        html.Div(id="an-subjects-container"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Session label(s)"),
                dbc.Input(id="an-session-label", type="text",
                          placeholder="e.g. 01  (optional, space-separated)"),
            ], width=3),
            dbc.Col([
                dbc.Label("Task label(s)"),
                dbc.Input(id="an-task-label", type="text",
                          placeholder="e.g. rest  (optional)"),
            ], width=3),
        ], className="mt-3 g-3"),
    ),

    # ── Preprocessing ─────────────────────────────────────────────────────────
    _card("Preprocessing",
        dbc.Row([
            dbc.Col([
                dbc.Label("DPF"),
                dbc.Input(id="an-dpf", type="number", placeholder="e.g. 6.0"),
            ], width=2),
            dbc.Col([
                dbc.Label("SCI Threshold"),
                dbc.Input(id="an-sci-thresh", type="number",
                          value=0.8, min=0.0, max=1.0, step=0.01),
            ], width=2),
            dbc.Col([
                dbc.Label("Motion Correction"),
                dcc.Dropdown(
                    id="an-motion-correction",
                    options=[
                        {"label": "TDDR",    "value": "tddr"},
                        {"label": "Wavelet", "value": "wavelet"},
                        {"label": "Spline",  "value": "spline"},
                        {"label": "None",    "value": "none"},
                    ],
                    value="tddr",
                    clearable=False,
                ),
            ], width=3),
            dbc.Col([
                dbc.Label("Cardiac Band (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="an-cardiac-l", type="number", step=0.1, placeholder="lo (adult ~0.7)"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="an-cardiac-h", type="number", step=0.1, placeholder="hi (adult ~1.5)"),
                ]),
            ], width=3),
            dbc.Col([
                dbc.Label("Respiration Band (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="an-resp-l", type="number", step=0.1, placeholder="lo (adult ~0.1)"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="an-resp-h", type="number", step=0.1, placeholder="hi (adult ~0.5)"),
                ]),
            ], width=3),
        ], className="g-3"),
    ),

    # ── Postprocessing ────────────────────────────────────────────────────────
    _card("Postprocessing",
        dbc.Row([
            dbc.Col([
                dbc.Label("Mode"),
                dbc.RadioItems(
                    id="an-post-mode",
                    options=[
                        {"label": "None",    "value": "none"},
                        {"label": "Denoise", "value": "denoise"},
                        {"label": "GLM",     "value": "glm"},
                        {"label": "Rest",    "value": "rest"},
                    ],
                    value="none",
                    inline=True,
                ),
            ]),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("High-pass (Hz)"),
                dbc.Input(id="an-high-pass", type="number", placeholder="e.g. 0.01"),
            ], width=2),
            dbc.Col([
                dbc.Label("Low-pass (Hz)"),
                dbc.Input(id="an-low-pass", type="number", placeholder="e.g. 0.5"),
            ], width=2),
            dbc.Col([
                dbc.Label("Resample (Hz)"),
                dbc.Input(id="an-resample", type="number", placeholder="e.g. 2.0"),
            ], width=2),
            dbc.Col([
                dbc.Label("n_jobs"),
                dbc.Input(id="an-n-jobs", type="number", value=1, min=1, step=1),
            ], width=2),
        ], className="g-3"),

        # GLM-only options
        html.Div(id="an-glm-section", children=[
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Label("HRF Model"),
                    dcc.Dropdown(
                        id="an-hrf-model",
                        options=[
                            {"label": "SPM",                 "value": "spm"},
                            {"label": "SPM + derivative",    "value": "spm + derivative"},
                            {"label": "Glover",              "value": "glover"},
                            {"label": "Glover + derivative", "value": "glover + derivative"},
                            {"label": "FIR",                 "value": "fir"},
                        ],
                        value="spm",
                        clearable=False,
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Label("Noise Model"),
                    dcc.Dropdown(
                        id="an-noise-model",
                        options=[{"label": v, "value": v}
                                 for v in ["ols", "ar1", "ar2", "ar3", "ar4", "ar5"]],
                        value="ar1",
                        clearable=False,
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label("Short Channel"),
                    dcc.Dropdown(
                        id="an-short-channel",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Mean", "value": "mean"},
                            {"label": "PCA",  "value": "pca"},
                        ],
                        value="none",
                        clearable=False,
                    ),
                ], width=2),
                dbc.Col([
                    dbc.Label("Stim Duration (s)"),
                    dbc.Input(id="an-stim-dur", type="number", placeholder="optional"),
                ], width=2),
            ], className="g-3"),
        ]),
    ),

    # ── Execution ─────────────────────────────────────────────────────────────
    _card("Execution",
        dbc.Row([
            dbc.Col([
                dbc.Checklist(
                    id="an-flags",
                    options=[
                        {"label": "Dry run",              "value": "dry_run"},
                        {"label": "Skip BIDS validation", "value": "skip_bids_validation"},
                        {"label": "No report",            "value": "no_report"},
                        {"label": "Combine runs",         "value": "combine_runs"},
                    ],
                    value=["dry_run"],
                    inline=True,
                ),
            ], width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("Generate Command", id="an-generate-btn", color="info"),
                    dbc.Button("Run Pipeline",     id="an-run-btn",      color="success"),
                ]),
            ], width=4, className="d-flex align-items-center justify-content-end"),
        ], className="align-items-center"),
        html.Div(id="an-run-status", className="mt-2 small"),
    ),

    # ── Command preview ───────────────────────────────────────────────────────
    _card("Command Preview",
        dbc.Row([
            dbc.Col(dbc.Label("Shell (line-continuation style)"), width="auto"),
            dbc.Col(dcc.Dropdown(
                id="an-shell-select",
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
            id="an-command-preview",
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

    _dag_card(),
], fluid=True)
