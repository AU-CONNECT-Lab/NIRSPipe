"""Dash application factory and entry point for fnirs-gui."""

from __future__ import annotations

import os

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import Input, Output, State, callback, dcc, html

from fnirs_pipe.interface.theme import SIDEBAR_BG

cyto.load_extra_layouts()


_SIDEBAR_EXPANDED = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "200px", "background": SIDEBAR_BG,
    "transition": "width 0.2s", "overflow": "hidden", "zIndex": 1000,
}
_SIDEBAR_COLLAPSED = {
    **_SIDEBAR_EXPANDED,
    "width": "44px",
}
_CONTENT_EXPANDED  = {"marginLeft": "200px", "padding": "2rem", "transition": "margin-left 0.2s"}
_CONTENT_COLLAPSED = {"marginLeft": "44px",  "padding": "2rem", "transition": "margin-left 0.2s"}


def _sidebar() -> html.Div:
    return html.Div(
        id="app-sidebar",
        style=_SIDEBAR_EXPANDED,
        children=[
            html.Div(
                className="d-flex align-items-center px-2 pt-3 pb-2",
                children=[
                    html.Div(
                        id="app-sidebar-title",
                        className="me-auto",
                        children=[
                            html.H5("fnirs-pipe", className="text-white mb-0"),
                            html.Small("Interface", className="text-white-50"),
                        ],
                    ),
                    dbc.Button(
                        "☰", id="app-sidebar-toggle",
                        color="link", size="sm",
                        className="text-white p-0",
                        style={"fontSize": "1.1rem", "lineHeight": 1},
                    ),
                ],
            ),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.12)", "opacity": 1,
                           "margin": 0}),
            dbc.Collapse(
                id="app-sidebar-nav",
                is_open=True,
                children=dbc.Nav([
                    dbc.NavLink("Data Preparation", href="/",            active="exact", className="text-white"),
                    dbc.NavLink("Recon",            href="/recon",      active="exact", className="text-white"),
                    dbc.NavLink("Batch Prep",       href="/batch-prep", active="exact", className="text-white"),
                    dbc.NavLink("Hyper Align",      href="/hyper-align",active="exact", className="text-white"),
                    dbc.NavLink("Analysis",         href="/analysis",   active="exact", className="text-white"),
                    dbc.NavLink("QC Reports",       href="/qc",         active="exact", className="text-white"),
                ], vertical=True, pills=True),
            ),
        ],
    )


@callback(
    Output("app-sidebar",       "style"),
    Output("app-page-content",  "style"),
    Output("app-sidebar-nav",   "is_open"),
    Output("app-sidebar-title", "style"),
    Input("app-sidebar-toggle", "n_clicks"),
    State("app-sidebar-nav",    "is_open"),
    prevent_initial_call=True,
)
def _toggle_sidebar(n_clicks, is_open):
    if is_open:
        return _SIDEBAR_COLLAPSED, _CONTENT_COLLAPSED, False, {"display": "none"}
    return _SIDEBAR_EXPANDED, _CONTENT_EXPANDED, True, {}


def launch(port: int = 8050) -> None:
    pages_folder = os.path.join(os.path.dirname(__file__), "pages")

    app = dash.Dash(
        __name__,
        use_pages=True,
        pages_folder=pages_folder,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    import fnirs_pipe.interface.callbacks.data_prep_callbacks
    import fnirs_pipe.interface.callbacks.recon_callbacks
    import fnirs_pipe.interface.callbacks.batch_prep_callbacks
    import fnirs_pipe.interface.callbacks.hyper_align_callbacks
    import fnirs_pipe.interface.callbacks.analysis_callbacks
    import fnirs_pipe.interface.callbacks.qc_callbacks  # noqa: F401  (side effect: registers callbacks)

    # lets the QC page show a generated report in an iframe; group reports are iframe shells
    # whose panels are sibling files, so they have to be served rather than inlined
    from fnirs_pipe.interface.report_serve import attach as _attach_reports
    _attach_reports(app.server)

    app.layout = html.Div([
        dcc.Store(id="app-bids-dir",   storage_type="session"),
        dcc.Store(id="app-output-dir", storage_type="session"),
        dcc.Store(id="dp-run-store",   storage_type="memory"),
        _sidebar(),
        html.Div(
            id="app-page-content",
            children=dash.page_container,
            style=_CONTENT_EXPANDED,
        ),
    ])

    app.run(port=port, debug=False)
