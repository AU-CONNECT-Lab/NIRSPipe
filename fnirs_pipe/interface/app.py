"""Dash application factory and entry point for fnirs-gui."""

from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import dcc, html

cyto.load_extra_layouts()


def _sidebar() -> dbc.Nav:
    return dbc.Nav(
        [
            html.Div(
                [
                    html.H5("fnirs-pipe", className="text-white mb-0"),
                    html.Small("Interface", className="text-white-50"),
                ],
                className="px-3 pt-3 pb-2",
            ),
            html.Hr(className="border-secondary my-0"),
            dbc.NavLink("Data Preparation", href="/",         active="exact", className="text-white"),
            dbc.NavLink("Analysis",          href="/analysis", active="exact", className="text-white"),
        ],
        vertical=True,
        pills=True,
        style={
            "position":   "fixed",
            "top":        0,
            "left":       0,
            "bottom":     0,
            "width":      "200px",
            "background": "#2c3e50",
        },
    )


def launch(port: int = 8050) -> None:
    pages_folder = os.path.join(os.path.dirname(__file__), "pages")

    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder=pages_folder,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )

    import fnirs_pipe.interface.callbacks.data_prep_callbacks
    import fnirs_pipe.interface.callbacks.analysis_callbacks

    app.layout = html.Div([
        dcc.Store(id="app-bids-dir",   storage_type="session"),
        dcc.Store(id="app-output-dir", storage_type="session"),
        dcc.Store(id="app-cache-dir",  storage_type="local"),
        dcc.Store(id="dp-run-store",   storage_type="memory"),
        _sidebar(),
        html.Div(
            dash.page_container,
            style={"marginLeft": "200px", "padding": "2rem"},
        ),
    ])

    app.run(port=port, debug=False)
