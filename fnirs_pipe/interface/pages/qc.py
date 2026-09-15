"""Page: cohort-level quality control — `fnirs-qc`, which aggregates records already on disk."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from fnirs_pipe.interface.components import PATH, actions, card, field, params, split

dash.register_page(__name__, path="/qc", name="Cohort Reports")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_COMMANDS = [
    {"label": "Cohort QC (individual subjects)", "value": "cohort"},
    {"label": "Cohort QC (hyperscanning groups)", "value": "cohort-hyper"},
    {"label": "Provenance graphs", "value": "provenance"},
]


def _scope():
    return card("Scope",
        params(
            field("Command",
                  dcc.Dropdown(id="qc-command", options=_COMMANDS,
                               value="cohort", clearable=False),
                  span=PATH),
            field("Derivatives directory",
                  dbc.Input(id="qc-output-dir", type="text",
                            placeholder="path to derivatives"),
                  span=PATH),
        ),
    )


def _run_panel():
    return card("Run",
        actions(
            dbc.Button("Run", id="qc-run-btn", color="success"),
            dbc.Button("Generate command", id="qc-generate-btn",
                       color="secondary", outline=True),
        ),
        dbc.Select(
            id="qc-shell-select",
            options=[{"label": s, "value": s} for s in ("bash", "cmd", "powershell")],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mt-3 mb-2",
        ),
        html.Pre(id="qc-command-preview", className="fp-command"),
        html.Div(id="qc-run-status", className="mt-3"),
    )


layout = dbc.Container([
    dcc.Store(id="qc-command-store"),

    html.H3("Cohort Reports"),
    html.P("Cohort aggregates over the quality records every run already wrote. Each command "
           "takes one derivatives directory and nothing else.", className="text-muted"),
    html.Hr(),

    split(
        main=[_scope(), card("Report preview", html.Div(id="qc-report-preview"))],
        aside=[_run_panel()],
    ),
], fluid=True)
