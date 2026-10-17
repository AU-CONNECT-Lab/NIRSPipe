"""Callbacks for the QC Reports page: build an `nirspipe-qc` argv, run it, show the report."""

from __future__ import annotations

from dash import Input, Output, State, callback

from nirspipe.interface import process_stream
from nirspipe.interface.callbacks._cli_run import poll_run_and_report, start_run
from nirspipe.interface.commands import preview_text
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
    Output("qc-run-store", "data"),
    Output("qc-run-tick", "disabled"),
    Output("qc-stop-btn", "disabled"),
    Output("qc-run-btn", "disabled"),
    Input("qc-run-btn", "n_clicks"),
    State("qc-command-store", "data"),
    prevent_initial_call=True,
)
def run_command(n_clicks, stored):
    status, run_id, *buttons = start_run(stored, "Generate command",
                                         "`{exe}` not found on PATH. Make sure nirspipe is installed.")
    # the previous run's report would otherwise sit under this run's log
    return status, None, run_id, *buttons


@callback(
    Output("qc-run-status", "children", allow_duplicate=True),
    Output("qc-report-preview", "children", allow_duplicate=True),
    Output("qc-run-tick", "disabled", allow_duplicate=True),
    Output("qc-stop-btn", "disabled", allow_duplicate=True),
    Output("qc-run-btn", "disabled", allow_duplicate=True),
    Input("qc-run-tick", "n_intervals"),
    State("qc-run-store", "data"),
    State("qc-command-store", "data"),
    prevent_initial_call=True,
)
def stream_run_output(_n, run_id, stored):
    return poll_run_and_report(run_id, stored)


@callback(
    Output("qc-stop-btn", "disabled", allow_duplicate=True),
    Input("qc-stop-btn", "n_clicks"),
    State("qc-run-store", "data"),
    prevent_initial_call=True,
)
def stop_command(_n, run_id):
    # the tick reports the outcome; this only asks the process to end
    if run_id:
        process_stream.stop(run_id)
    return True
