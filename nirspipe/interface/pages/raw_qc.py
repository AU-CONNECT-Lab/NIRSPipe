"""Page: raw quality reports (`nirspipe-qc prep-raw` / `hyper-raw`), written before any analysis."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspipe.interface.components import (
    PATH, actions, band, card, field, params, section, split, switches,
)

dash.register_page(__name__, path="/raw-qc", name="Raw QC")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_COMMANDS = [
    {"label": "Subjects (nirspipe-qc prep-raw)", "value": "prep-raw"},
    {"label": "Dyads (nirspipe-qc hyper-raw)", "value": "hyper-raw"},
]

_MOTION = [{"label": "None", "value": "none"},
           {"label": "TDDR", "value": "tddr"},
           {"label": "Wavelet", "value": "wavelet"},
           {"label": "Targeted PCA", "value": "tpca"},
           {"label": "Spline", "value": "spline"},
           {"label": "Spline-SG", "value": "spline_sg"},
           {"label": "Wavelet + TDDR", "value": "wavelet_tddr"},
           {"label": "Spline + Wavelet", "value": "spline_wavelet"}]


def _scope():
    return card("Scope",
        params(
            field("Command",
                  dcc.Dropdown(id="rq-command", options=_COMMANDS,
                               value="prep-raw", clearable=False),
                  span=PATH),
            field("BIDS directory",
                  dbc.Input(id="rq-bids-dir", type="text", placeholder="/path/to/bids"),
                  span=PATH),
            field("Output directory",
                  dbc.Input(id="rq-output-dir", type="text",
                            placeholder="where the reports go"),
                  span=PATH),
            field("Subjects",
                  dbc.Input(id="rq-subjects", type="text",
                            placeholder="space-separated, e.g. 01 02"),
                  hint="Required for subjects; for dyads, only groups with one of them."),
            field("Session labels",
                  dbc.Input(id="rq-ses", type="text", placeholder="all sessions")),
            field("Task labels",
                  dbc.Input(id="rq-task", type="text", placeholder="all tasks")),
            field("ROI mapping",
                  dbc.Input(id="rq-roi-mapping", type="text", placeholder="roi.json (optional)"),
                  span=PATH),
        ),
        switches(dbc.Checklist(
            id="rq-flags",
            options=[{"label": "Ignore channels.tsv status", "value": "ignore_channel_status"},
                     {"label": "Skip BIDS validation", "value": "skip_bids_validation"}],
            value=[], inline=True, switch=True,
        )),
    )


def _screening():
    return card("Screening",
        section("Required",
            params(
                field("SCI threshold",
                      dbc.Input(id="rq-sci-thresh", type="number", placeholder="e.g. 0.8",
                                min=0.0, max=1.0, step="any")),
                field("DPF", dbc.Input(id="rq-dpf", type="number", step="any",
                                       placeholder="e.g. 6.0")),
                band("Cardiac band (Hz)",
                     dbc.Input(id="rq-cardiac-l", type="number", step="any",
                               placeholder="lo (adult ~0.7)"),
                     "–",
                     dbc.Input(id="rq-cardiac-h", type="number", step="any",
                               placeholder="hi (adult ~2.0)")),
            ),
         key="rq-required", open=True),
        section("Channel screening",
            params(
                field("PSP threshold",
                      dbc.Input(id="rq-psp-thresh", type="number", value=0.1,
                                min=0.0, max=1.0, step="any")),
                field("Coupled windows",
                      dbc.Input(id="rq-min-good-frac", type="number", value=0.75,
                                min=0.0, max=1.0, step="any"),
                      hint="Share a channel needs to be kept. This is what rejects."),
                field("Screening scope",
                      dcc.Dropdown(id="rq-screen-scope",
                                   options=[{"label": "Whole run", "value": "run"},
                                            {"label": "Task blocks only", "value": "task"}],
                                   value="run", clearable=False)),
                band("Separations (mm)",
                     "≤", dbc.Input(id="rq-short-max-dist", type="number", step="any",
                                    min=0.1, placeholder="10"),
                     "≥", dbc.Input(id="rq-long-min-dist", type="number", step="any",
                                    min=0.1, placeholder="15"),
                     "to", dbc.Input(id="rq-long-max-dist", type="number", step="any",
                                     min=0.1, placeholder="none"),
                     span=PATH),
            ),
         key="rq-screen", open=False),
    )


def _subject_section():
    return html.Div(id="rq-subject-section", children=card(
        "Subject reports",
        params(
            field("SCI / PSP window (s)",
                  dbc.Input(id="rq-window-length", type="number", step="any", min=1,
                            placeholder="10")),
            band("Trial window (s)",
                 dbc.Input(id="rq-epoch-tmin", type="number", step="any", placeholder="start"),
                 "–",
                 dbc.Input(id="rq-epoch-tmax", type="number", step="any", placeholder="end"),
                 hint="Both or neither; empty uses each event's own duration."),
            field("Motion correction preview",
                  dcc.Dropdown(id="rq-motion-correction", options=_MOTION,
                               value="none", clearable=False),
                  hint="Reported beside the recording; nothing is written back."),
            # the placeholders name the chosen method's defaults; empty sends nothing
            field("Detection threshold (SD)",
                  dbc.Input(id="rq-motion-std-thresh", type="number", step="any", min=0.5),
                  id="rq-motion-std-thresh-wrap"),
            field("Wavelet IQR",
                  dbc.Input(id="rq-wavelet-iqr", type="number", step="any", min=0.1),
                  id="rq-wavelet-iqr-wrap"),
        ),
        switches(dbc.Checklist(
            id="rq-subject-flags",
            options=[{"label": "Per-trial QC", "value": "epoch_qc"},
                     {"label": "One report per condition", "value": "by_condition"}],
            value=[], inline=True, switch=True,
        )),
    ))


def _dyad_section():
    return html.Div(id="rq-dyad-section", style={"display": "none"}, children=card(
        "Dyad reports",
        params(
            field("Pairs CSV",
                  dbc.Input(id="rq-pairs-csv", type="text", placeholder="path to pairs.csv"),
                  span=PATH),
            field("Group ID",
                  dbc.Input(id="rq-group-id", type="text", placeholder="all groups")),
            field("nirspipe derivatives",
                  dbc.Input(id="rq-derivatives-dir", type="text",
                            placeholder="optional: draws the after-correction side"),
                  span=PATH),
            band("Window (s)",
                 dbc.Input(id="rq-tstart", type="number", step="any", placeholder="start"),
                 "–",
                 dbc.Input(id="rq-tend", type="number", step="any", placeholder="end"),
                 hint="On the aligned clock; narrows the synchrony metrics only."),
        ),
        switches(dbc.Checklist(
            id="rq-dyad-flags",
            options=[{"label": "Z-score each channel", "value": "normalize"},
                     {"label": "No trigger alignment", "value": "no_align"}],
            value=[], inline=True, switch=True,
        )),
    ))


def _run_panel():
    return card("Run",
        actions(
            dbc.Button("Run", id="rq-run-btn", color="success"),
            dbc.Button("Stop", id="rq-stop-btn", color="danger",
                       outline=True, disabled=True),
            dbc.Button("Generate command", id="rq-generate-btn",
                       color="secondary", outline=True),
        ),
        dbc.Select(
            id="rq-shell-select",
            options=[{"label": s, "value": s} for s in ("bash", "cmd", "powershell")],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mt-3 mb-2",
        ),
        html.Pre(id="rq-command-preview", className="fp-command"),
        html.Div(id="rq-run-status", className="mt-3"),
        html.Div(id="rq-rate-hint", className="mt-3"),
        html.Hr(),
        html.Small("Once the reports are rated, write the decisions as Analysis's "
                   "Bad channels table.", className="text-muted d-block mb-2"),
        dbc.Button("Export channel decisions", id="rq-export-btn", color="secondary",
                   outline=True, size="sm"),
        html.Div(id="rq-export-status", className="mt-2"),
    )


layout = dbc.Container([
    dcc.Store(id="rq-command-store"),
    dcc.Store(id="rq-run-store", storage_type="memory"),
    # the run is polled rather than waited on; disabled until there is something to poll
    dcc.Interval(id="rq-run-tick", interval=1000, disabled=True),

    html.H3("Raw QC"),
    html.P("Quality reports on the recordings as delivered, before any analysis. Rate them "
           "and decide channels with nirspipe-rate.", className="text-muted"),
    html.Hr(),

    split(
        main=[
            _scope(),
            _screening(),
            _subject_section(),
            _dyad_section(),
            card("Report preview", html.Div(id="rq-report-preview")),
        ],
        aside=[_run_panel()],
    ),
], fluid=True)
