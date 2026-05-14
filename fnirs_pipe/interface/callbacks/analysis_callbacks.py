"""Callbacks for the analysis page."""

from __future__ import annotations

from dash import Input, Output, State, callback, no_update


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
    Input("an-detect-btn", "n_clicks"),
    State("an-bids-dir",   "value"),
    prevent_initial_call=True,
)
def detect_subjects(n_clicks, bids_dir):
    return no_update, no_update


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


@callback(
    Output("an-command-preview", "children"),
    Input("an-generate-btn",     "n_clicks"),
    State("an-bids-dir",          "value"),
    State("an-output-dir",        "value"),
    State("an-dpf",               "value"),
    State("an-sci-thresh",        "value"),
    State("an-motion-correction", "value"),
    State("an-post-mode",         "value"),
    State("an-high-pass",         "value"),
    State("an-low-pass",          "value"),
    State("an-resample",          "value"),
    State("an-n-jobs",            "value"),
    State("an-flags",             "value"),
    State("an-session-label",     "value"),
    State("an-task-label",        "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, bids_dir, output_dir, dpf, sci_thresh,
                     motion_correction, post_mode, high_pass, low_pass,
                     resample, n_jobs, flags, session_label, task_label):
    return no_update


@callback(
    Output("an-run-status", "children"),
    Input("an-run-btn",              "n_clicks"),
    State("an-command-preview",      "children"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks, command):
    return no_update
