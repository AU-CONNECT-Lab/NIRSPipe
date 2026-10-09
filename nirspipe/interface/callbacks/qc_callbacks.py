"""Callbacks for the QC Reports page: build an `nirspipe-qc` argv, run it, show the report."""

from __future__ import annotations

from dash import Input, Output, State, callback

from nirspipe.interface.callbacks._cli_run import preview_text, run_and_report
from nirspipe.interface.cli_args import build_qc_args, missing
from nirspipe.utils.logging import get_logger

logger = get_logger("interface.qc_callbacks")


@callback(
    Output("qc-command-preview", "children"),
    Output("qc-command-store", "data"),
    Input("qc-generate-btn", "n_clicks"),
    State("qc-command", "value"),
    State("qc-output-dir", "value"),
    State("qc-shell-select", "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, command, output_dir, shell):
    opts = {"output_dir": output_dir}

    problem = missing(command, opts)
    if problem:
        return f"Error: {problem}", {}

    argv = build_qc_args(command, opts)
    return (preview_text(argv, shell),
            {"argv": argv, "command": command, "output_dir": output_dir})


@callback(
    Output("qc-run-status", "children"),
    Output("qc-report-preview", "children"),
    Input("qc-run-btn", "n_clicks"),
    State("qc-command-store", "data"),
    prevent_initial_call=True,
)
def run_command(n_clicks, stored):
    return run_and_report(stored)
