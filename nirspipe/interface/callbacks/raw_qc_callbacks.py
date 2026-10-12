"""Callbacks for the Raw QC page: build an `nirspipe-qc prep-raw/hyper-raw` argv, run it, show the report."""

from __future__ import annotations

import subprocess

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

from nirspipe.interface import process_stream
from nirspipe.interface.callbacks._cli_run import (
    poll_run_and_report, preview_text, start_run,
)
from nirspipe.interface.callbacks._sections import rng, separations, summary, value
from nirspipe.interface.cli_args import build_raw_qc_args, missing_raw_qc
from nirspipe.pipeline.motion import IQR_FACTOR_DEFAULTS, STD_THRESH_DEFAULTS

_SHOWN = {"display": ""}
_GONE  = {"display": "none"}

_STATES = [
    State("rq-bids-dir", "value"), State("rq-output-dir", "value"),
    State("rq-subjects", "value"), State("rq-ses", "value"), State("rq-task", "value"),
    State("rq-roi-mapping", "value"), State("rq-flags", "value"),
    State("rq-sci-thresh", "value"), State("rq-dpf", "value"),
    State("rq-cardiac-l", "value"), State("rq-cardiac-h", "value"),
    State("rq-psp-thresh", "value"), State("rq-min-good-frac", "value"),
    State("rq-screen-scope", "value"),
    State("rq-short-max-dist", "value"), State("rq-long-min-dist", "value"),
    State("rq-long-max-dist", "value"),
    State("rq-window-length", "value"),
    State("rq-epoch-tmin", "value"), State("rq-epoch-tmax", "value"),
    State("rq-motion-correction", "value"),
    State("rq-motion-std-thresh", "value"), State("rq-wavelet-iqr", "value"),
    State("rq-subject-flags", "value"),
    State("rq-pairs-csv", "value"), State("rq-group-id", "value"),
    State("rq-derivatives-dir", "value"),
    State("rq-tstart", "value"), State("rq-tend", "value"),
    State("rq-dyad-flags", "value"),
]

_KEYS = ["bids_dir", "output_dir", "subjects", "ses", "task", "roi_mapping", "flags",
         "sci_threshold", "dpf", "cardiac_l", "cardiac_h",
         "psp_threshold", "min_good_frac", "screen_scope",
         "short_max_dist", "long_min_dist", "long_max_dist",
         "window_length", "epoch_tmin", "epoch_tmax",
         "motion_correction", "motion_std_thresh", "wavelet_iqr", "subject_flags",
         "pairs_csv", "group_id", "derivatives_dir", "tstart", "tend", "dyad_flags"]


def _opts(values) -> dict:
    opts = dict(zip(_KEYS, values))
    # each checklist's values are the option keys the argv builder reads
    for checklist in ("flags", "subject_flags", "dyad_flags"):
        for key in opts.pop(checklist) or []:
            opts[key] = True
    return opts


def _one(value: "str | None") -> "str | None":
    """The label when the box holds exactly one, which is all a viewer can open."""
    labels = (value or "").split()
    return labels[0] if len(labels) == 1 else None


def rate_commands(command: str, opts: dict) -> list[str]:
    """The nirspipe-rate lines that open what this command writes."""
    out = opts["output_dir"]
    ses, task = _one(opts.get("ses")), _one(opts.get("task"))
    if command == "prep-raw":
        tail = (f" --session-label {ses}" if ses else "") + (f" --task-label {task}" if task else "")
        return [f"nirspipe-rate raw {out} --participant-label {sub}{tail}"
                for sub in (opts.get("subjects") or "").split()]
    line = (f"nirspipe-rate hyper {out} --group-id {opts.get('group_id') or '<group>'}"
            f" --task-label {task or '<task>'} --pairs-csv {opts['pairs_csv']}")
    if ses:
        line += f" --session-label {ses}"
    if opts.get("derivatives_dir"):
        line += f" --derivatives-dir {opts['derivatives_dir']}"
    return [line]


def export_args(command: str, opts: dict) -> list[str]:
    """The export that turns those ratings into Analysis's --bad-channels table."""
    # the dyad viewer files each member's decisions in the nirspipe tree when given one
    trees = [opts["output_dir"]]
    if command == "hyper-raw" and opts.get("derivatives_dir"):
        trees.append(opts["derivatives_dir"])
    return ["nirspipe-rate", "export", *trees]


def export_command(command: str, opts: dict) -> str:
    return " ".join(export_args(command, opts))


@callback(
    Output("rq-subject-section", "style"),
    Output("rq-dyad-section", "style"),
    Input("rq-command", "value"),
)
def toggle_sections(command):
    return (_SHOWN, _GONE) if command == "prep-raw" else (_GONE, _SHOWN)


# ── The motion settings: shown for the methods that read them, defaults as placeholders ──

@callback(
    Output("rq-motion-std-thresh-wrap", "style"),
    Output("rq-wavelet-iqr-wrap",       "style"),
    Output("rq-motion-std-thresh",      "placeholder"),
    Output("rq-wavelet-iqr",            "placeholder"),
    Input("rq-motion-correction", "value"),
)
def motion_setting_fields(method):
    std, iqr = STD_THRESH_DEFAULTS.get(method), IQR_FACTOR_DEFAULTS.get(method)
    return (_SHOWN if std is not None else _GONE, _SHOWN if iqr is not None else _GONE,
            f"{std:g} ({method} default)" if std is not None else "",
            f"{iqr:g} ({method} default)" if iqr is not None else "")


@callback(
    Output("rq-required-summary", "children"),
    Output("rq-screen-summary", "children"),
    Input("rq-sci-thresh", "value"), Input("rq-dpf", "value"),
    Input("rq-cardiac-l", "value"), Input("rq-cardiac-h", "value"),
    Input("rq-psp-thresh", "value"), Input("rq-min-good-frac", "value"),
    Input("rq-screen-scope", "value"),
    Input("rq-short-max-dist", "value"), Input("rq-long-min-dist", "value"),
)
def section_summaries(sci, dpf, card_l, card_h, psp, good_frac, scope, short_max, long_min):
    return (summary(value("SCI", sci), value("DPF", dpf), rng("cardiac", card_l, card_h, " Hz")),
            summary(value("PSP", psp), value("coupled", good_frac), scope,
                    separations(short_max, long_min)))


@callback(
    Output("rq-command-preview", "children"),
    Output("rq-command-store", "data"),
    Output("rq-rate-hint", "children"),
    Input("rq-generate-btn", "n_clicks"),
    State("rq-command", "value"),
    *_STATES,
    State("rq-shell-select", "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, command, *values):
    *state_values, shell = values
    opts = _opts(state_values)

    problem = missing_raw_qc(command, opts)
    if problem:
        return f"Error: {problem}", {}, None

    argv = build_raw_qc_args(command, opts)
    hint = html.Div([
        html.Small("Then rate the reports with:", className="text-muted d-block mb-1"),
        html.Pre("\n".join(rate_commands(command, opts)), className="fp-command"),
        html.Small("and export the channel decisions for Analysis with:",
                   className="text-muted d-block mb-1"),
        html.Pre(export_command(command, opts), className="fp-command"),
    ])
    return (preview_text(argv, shell),
            {"argv": argv, "command": command, "output_dir": opts["output_dir"]},
            hint)


@callback(
    Output("rq-run-status", "children"),
    Output("rq-report-preview", "children"),
    Output("rq-run-store", "data"),
    Output("rq-run-tick", "disabled"),
    Output("rq-stop-btn", "disabled"),
    Output("rq-run-btn", "disabled"),
    Input("rq-run-btn", "n_clicks"),
    State("rq-command-store", "data"),
    prevent_initial_call=True,
)
def run_command(n_clicks, stored):
    status, run_id, *buttons = start_run(stored, "Generate command",
                                         "`{exe}` not found on PATH. Make sure nirspipe is installed.")
    # the previous run's report would otherwise sit under this run's log
    return status, None, run_id, *buttons


@callback(
    Output("rq-run-status", "children", allow_duplicate=True),
    Output("rq-report-preview", "children", allow_duplicate=True),
    Output("rq-run-tick", "disabled", allow_duplicate=True),
    Output("rq-stop-btn", "disabled", allow_duplicate=True),
    Output("rq-run-btn", "disabled", allow_duplicate=True),
    Input("rq-run-tick", "n_intervals"),
    State("rq-run-store", "data"),
    State("rq-command-store", "data"),
    prevent_initial_call=True,
)
def stream_run_output(_n, run_id, stored):
    return poll_run_and_report(run_id, stored)


@callback(
    Output("rq-stop-btn", "disabled", allow_duplicate=True),
    Input("rq-stop-btn", "n_clicks"),
    State("rq-run-store", "data"),
    prevent_initial_call=True,
)
def stop_command(_n, run_id):
    # the tick reports the outcome; this only asks the process to end
    if run_id:
        process_stream.stop(run_id)
    return True


@callback(
    Output("rq-export-status", "children"),
    Input("rq-export-btn", "n_clicks"),
    State("rq-command", "value"),
    State("rq-output-dir", "value"),
    State("rq-derivatives-dir", "value"),
    prevent_initial_call=True,
)
def export_decisions(n_clicks, command, output_dir, derivatives_dir):
    if not output_dir:
        return dbc.Alert("Set the output directory.", color="warning", className="mb-0")
    argv = export_args(command, {"output_dir": output_dir, "derivatives_dir": derivatives_dir})
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return dbc.Alert("`nirspipe-rate` not found on PATH. Make sure nirspipe is installed.",
                         color="danger", className="mb-0")
    ok = proc.returncode == 0
    # the CLI prints the table's path and the --bad-channels line to give Analysis
    return dbc.Alert([
        html.Div("Exported." if ok else f"Failed (exit {proc.returncode}).", className="fw-bold"),
        html.Code(" ".join(argv), className="d-block small mb-1"),
        html.Pre(((proc.stdout or "") + (proc.stderr or "")).strip(), className="fp-run-tail mb-0"),
    ], color="success" if ok else "danger", className="mb-0")
