"""Callbacks for the analysis page."""

from __future__ import annotations

import subprocess
import sys

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update


# ── Directory sync: store → page (one-way; page 1 writes, page 2 reads) ──────

@callback(
    Output("an-bids-dir",   "value"),
    Output("an-output-dir", "value"),
    Input("app-bids-dir",   "data"),
    Input("app-output-dir", "data"),
)
def _prefill_an_from_store(bids_dir, output_dir):
    return bids_dir or no_update, output_dir or no_update


@callback(
    Output("an-subjects-result",    "children"),
    Output("an-subjects-container", "children"),
    Output("an-subjects-store",     "data"),
    Input("an-detect-btn", "n_clicks"),
    State("an-bids-dir",   "value"),
    prevent_initial_call=True,
)
def detect_subjects(n_clicks, bids_dir):
    if not bids_dir:
        return dbc.Alert("BIDS directory is empty.", color="warning", className="mb-0"), "", []
    try:
        from fnirs_pipe.io.bids import get_layout
        layout = get_layout(bids_dir, validate=False)
        subjects = sorted(layout.get_subjects())
    except Exception as exc:
        return dbc.Alert(f"Failed to read BIDS dir: {exc}", color="danger", className="mb-0"), "", []

    if not subjects:
        return dbc.Alert("No subjects found.", color="warning", className="mb-0"), "", []

    alert = dbc.Alert(f"Found {len(subjects)} subject(s).", color="success", className="mb-0")
    checklist = dbc.Checklist(
        id="an-subjects-checklist",
        options=[{"label": s, "value": s} for s in subjects],
        value=subjects,
        inline=True,
        className="mt-2",
    )
    return alert, checklist, subjects


@callback(
    Output("an-subjects-store", "data", allow_duplicate=True),
    Input("an-subjects-checklist", "value"),
    prevent_initial_call=True,
)
def _sync_subjects_store(selected):
    return selected or []


@callback(
    Output("an-glm-section", "style"),
    Input("an-post-mode", "value"),
)
def toggle_glm_section(mode):
    return {"display": "block"} if mode == "glm" else {"display": "none"}


@callback(
    Output("an-dag", "elements"),
    Input("an-post-mode", "value"),
)
def update_dag(post_mode):

    def node(nid, label, cls):
        return {"data": {"id": nid, "label": label}, "classes": cls}

    def edge(src, tgt, cls):
        return {"data": {"source": src, "target": tgt}, "classes": cls}

    elements = [
        node("load",   "Load SNIRF",         "prep"),
        node("od",     "Optical Density",     "prep"),
        node("sci",    "SCI / Bad Channels",  "prep"),
        node("bll",    "Beer-Lambert Law",    "prep"),
        node("motion", "Motion Correction",   "prep"),
        edge("load",   "od",     "prep"),
        edge("od",     "sci",    "prep"),
        edge("sci",    "bll",    "prep"),
        edge("bll",    "motion", "prep"),
    ]

    last = "motion"

    if post_mode and post_mode != "none":
        elements += [
            node("filter", "Bandpass Filter", "post"),
            edge(last, "filter", "post"),
        ]
        last = "filter"

        elements += [
            node("resample", "Resample", "post"),
            edge(last, "resample", "post"),
        ]
        last = "resample"

        if post_mode == "glm":
            elements += [
                node("glm", "GLM", "glm"),
                edge(last, "glm", "glm"),
            ]
            last = "glm"

    return elements


def _build_cli_args(opts: dict) -> list[str]:
    """Translate widget values into a fnirs-pipe argv list."""
    args: list[str] = [
        "fnirs-pipe",
        opts["bids_dir"],
        opts["output_dir"],
        "participant",
    ]
    # list-typed options need the flag repeated per value (Typer/Click)
    for sub in opts.get("subjects") or []:
        args += ["--participant-label", sub]
    for ses in (opts.get("session_label") or "").split():
        args += ["--session-label", ses]
    for task in (opts.get("task_label") or "").split():
        args += ["--task-label", task]
    if opts.get("dpf") is not None:
        args += ["--dpf", str(opts["dpf"])]
    if opts.get("sci_thresh") is not None:
        args += ["--sci-threshold", str(opts["sci_thresh"])]
    if opts.get("motion_correction"):
        args += ["--motion-correction", opts["motion_correction"]]
    if opts.get("cardiac_l") is not None:
        args += ["--cardiac-l-freq", str(opts["cardiac_l"])]
    if opts.get("cardiac_h") is not None:
        args += ["--cardiac-h-freq", str(opts["cardiac_h"])]
    if opts.get("resp_l") is not None:
        args += ["--resp-l-freq", str(opts["resp_l"])]
    if opts.get("resp_h") is not None:
        args += ["--resp-h-freq", str(opts["resp_h"])]

    mode = opts.get("post_mode")
    if mode and mode != "none":
        args += ["--mode", mode]
        if opts.get("high_pass") is not None:
            args += ["--high-pass", str(opts["high_pass"])]
        if opts.get("low_pass") is not None:
            args += ["--low-pass", str(opts["low_pass"])]
        if opts.get("resample") is not None:
            args += ["--resample-sfreq", str(opts["resample"])]
        if mode == "glm":
            if opts.get("hrf_model"):     args += ["--hrf-model",     opts["hrf_model"]]
            if opts.get("noise_model"):   args += ["--noise-model",   opts["noise_model"]]
            if opts.get("short_channel"): args += ["--short-channel", opts["short_channel"]]

    if opts.get("n_jobs"):
        args += ["--n-jobs", str(opts["n_jobs"])]

    flags = opts.get("flags") or []
    if "dry_run"              in flags: args.append("--dry-run")
    if "skip_bids_validation" in flags: args.append("--skip-bids-validation")
    if "no_report"            in flags: args.append("--no-report")
    if "combine_runs"         in flags: args.append("--combine-runs")
    return args


@callback(
    Output("an-command-preview", "children"),
    Output("an-command-store",   "data"),
    Input("an-generate-btn",     "n_clicks"),
    State("an-bids-dir",          "value"),
    State("an-output-dir",        "value"),
    State("an-subjects-store",    "data"),
    State("an-dpf",               "value"),
    State("an-sci-thresh",        "value"),
    State("an-motion-correction", "value"),
    State("an-cardiac-l",         "value"),
    State("an-cardiac-h",         "value"),
    State("an-resp-l",            "value"),
    State("an-resp-h",            "value"),
    State("an-post-mode",         "value"),
    State("an-high-pass",         "value"),
    State("an-low-pass",          "value"),
    State("an-resample",          "value"),
    State("an-n-jobs",            "value"),
    State("an-hrf-model",         "value"),
    State("an-noise-model",       "value"),
    State("an-short-channel",     "value"),
    State("an-flags",             "value"),
    State("an-session-label",     "value"),
    State("an-task-label",        "value"),
    State("an-shell-select",      "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, bids_dir, output_dir, subjects, dpf, sci_thresh,
                     motion_correction, cardiac_l, cardiac_h, resp_l, resp_h,
                     post_mode, high_pass, low_pass, resample, n_jobs,
                     hrf_model, noise_model, short_channel, flags,
                     session_label, task_label, shell):
    if not bids_dir or not output_dir:
        return "Error: BIDS and output directories are required.", {}
    if not subjects:
        return "Error: detect and select subjects first.", {}
    if dpf is None or sci_thresh is None:
        return "Error: DPF and SCI threshold are required.", {}
    if cardiac_l is None or cardiac_h is None:
        return "Error: cardiac band lower/upper frequency is required (population-dependent).", {}
    if resp_l is None or resp_h is None:
        return "Error: respiration band lower/upper frequency is required (population-dependent).", {}

    opts = dict(
        bids_dir=bids_dir, output_dir=output_dir, subjects=subjects,
        session_label=session_label, task_label=task_label,
        dpf=dpf, sci_thresh=sci_thresh,
        motion_correction=motion_correction, cardiac_l=cardiac_l, cardiac_h=cardiac_h,
        resp_l=resp_l, resp_h=resp_h,
        post_mode=post_mode, high_pass=high_pass, low_pass=low_pass,
        resample=resample, n_jobs=n_jobs,
        hrf_model=hrf_model, noise_model=noise_model, short_channel=short_channel,
        flags=flags,
    )
    argv = _build_cli_args(opts)
    # Pretty-print: shell-specific line-continuation per arg group for readability.
    cont = {"bash": "\\", "cmd": "^", "powershell": "`"}.get(shell, "\\")
    lines = [argv[0]] + [f"  {a}" for a in argv[1:]]
    preview = f" {cont}\n".join(lines)
    return preview, {"argv": argv}


@callback(
    Output("an-run-status", "children"),
    Input("an-run-btn",     "n_clicks"),
    State("an-command-store", "data"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, cmd_data):
    if not cmd_data or not cmd_data.get("argv"):
        return dbc.Alert("Click 'Generate Command' first.", color="warning", className="mb-0")

    argv = cmd_data["argv"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return dbc.Alert(
            f"`{argv[0]}` not found on PATH — make sure fnirs-pipe is installed.",
            color="danger", className="mb-0",
        )
    except Exception as exc:
        return dbc.Alert(f"Failed to launch: {exc}", color="danger", className="mb-0")

    tail = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(tail.splitlines()[-30:])  # last 30 lines

    if proc.returncode == 0:
        color, header = "success", "Pipeline finished."
    else:
        color, header = "danger", f"Pipeline failed (exit {proc.returncode})."

    return dbc.Alert([
        html.Div(header, className="fw-bold"),
        html.Pre(tail, style={
            "background": "rgba(0,0,0,0.04)", "padding": "0.5rem",
            "borderRadius": "4px", "maxHeight": "300px", "overflow": "auto",
            "fontSize": "12px", "margin": "0.5rem 0 0",
        }),
    ], color=color, className="mb-0")
