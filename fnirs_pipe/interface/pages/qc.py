"""Page 6: quality control reports — hyperscanning, group aggregates and windowed drill-down.

The whole `fnirs-qc` aggregate layer used to be command line only, which left the Hyper Align
page preparing dyads that nothing on screen could then analyse. Like the Analysis page, this
one builds an argv and runs it; it does not reimplement anything.
"""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from fnirs_pipe.qc.provenance import _DOMAIN

dash.register_page(__name__, path="/qc", name="QC Reports")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

# --desc takes a free-form string, but load_group_haemo refuses anything pre-Beer-Lambert,
# so the offer is the haemoglobin half of the domain map rather than a list retyped here
_DESC_CHOICES = [desc for desc, domain in _DOMAIN.items() if domain == "haemo"]

# one output_dir and nothing else, so they share a form
_AGGREGATE_COMMANDS = [
    {"label": "Group QC (individual subjects)", "value": "group-raw"},
    {"label": "Group QC (hyperscanning dyads)", "value": "group-hyper-raw"},
    {"label": "Merge WTC tables across dyads", "value": "group-hyper-wtc"},
    {"label": "Provenance graphs", "value": "provenance"},
]


def _card(title, *children, subtitle=None):
    head = [html.H5(title, className="mb-0")]
    if subtitle:
        head.append(html.Small(subtitle, className="text-muted"))
    return dbc.Card(dbc.CardBody([*head, html.Hr(), *children]), className="mb-3")


def _dirs():
    return _card(
        "Directories",
        dbc.Row([
            dbc.Col([
                dbc.Label("BIDS Directory"),
                dbc.Input(id="qc-bids-dir", type="text", placeholder="path to BIDS root"),
                dbc.FormText("Not needed by the aggregate commands."),
            ], width=6),
            dbc.Col([
                dbc.Label("Derivatives / Output Directory"),
                dbc.Input(id="qc-output-dir", type="text", placeholder="path to derivatives"),
            ], width=6),
        ], className="g-3"),
    )


def _hyper_post():
    return html.Div(id="qc-hyper-post-section", children=_card(
        "Hyperscanning post-analysis",
        dbc.Row([
            dbc.Col([
                dbc.Label("Pairs CSV"),
                dbc.Input(id="qc-pairs-csv", type="text", placeholder="path to pairs.csv"),
                dbc.FormText("Required."),
            ], width=4),
            dbc.Col([
                dbc.Label("Group ID"),
                dbc.Input(id="qc-group-id", type="text", placeholder="all groups"),
            ], width=2),
            dbc.Col([
                dbc.Label("Source Stage"),
                dcc.Dropdown(id="qc-desc",
                             options=[{"label": c, "value": c} for c in _DESC_CHOICES],
                             value="preproc", clearable=False),
                dbc.FormText("errts reads the confound-regression residual."),
            ], width=3),
            dbc.Col([
                dbc.Label("ROI Mapping"),
                dbc.Input(id="qc-roi-mapping", type="text", placeholder="roi.json (optional)"),
                dbc.FormText("Required by the ROI options below."),
            ], width=3),
        ], className="g-3"),
        html.Hr(),
        dbc.Row([
            dbc.Col([
                dbc.Label("WTC Frequency Range (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="qc-wtc-fmin", type="number", value=0.004, step=0.001),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="qc-wtc-fmax", type="number", value=0.2, step=0.01),
                ]),
                dbc.FormText("The axis the coherence is computed on."),
            ], width=4),
            dbc.Col([
                dbc.Label("Band Averaged Over (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="qc-wtc-band-fmin", type="number", step=0.001,
                              placeholder="whole axis"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="qc-wtc-band-fmax", type="number", step=0.01,
                              placeholder="whole axis"),
                ]),
                dbc.FormText("Narrow this to the frequencies the task lives in."),
            ], width=4),
            dbc.Col([
                dbc.Label("Surrogates"),
                dbc.Input(id="qc-wtc-mc-count", type="number", value=300, min=1, step=50),
                dbc.FormText("Runtime scales with this. Preview low, settle high."),
            ], width=2),
            dbc.Col([
                dbc.Label("Seed"),
                dbc.Input(id="qc-wtc-seed", type="number", placeholder="none"),
                dbc.FormText("Makes the surrogates reproducible."),
            ], width=2),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Session Label"),
                dbc.Input(id="qc-hyper-session", type="text", placeholder="all sessions"),
            ], width=3),
            dbc.Col([
                dbc.Label("Task Label"),
                dbc.Input(id="qc-hyper-task", type="text", placeholder="all tasks"),
                dbc.FormText("Filters the pairs CSV, which already names a task per row."),
            ], width=3),
            dbc.Col([
                dbc.Label("ISC Threshold"),
                dbc.Input(id="qc-isc-threshold", type="number", step=0.05,
                          placeholder="default"),
            ], width=2),
            dbc.Col([
                dbc.Label("Pseudo-dyad Iterations"),
                dbc.Input(id="qc-wtc-pseudo", type="number", min=0, step=10,
                          placeholder="0, off"),
                dbc.FormText("The null a coherence is read against. One full WTC run each; "
                             "published work uses 100."),
            ], width=2),
            dbc.Col([
                dbc.Label("Min Channels per ROI"),
                dbc.Input(id="qc-wtc-roi-min-channels", type="number", min=1, step=1,
                          placeholder="2"),
                dbc.FormText("Below this an ROI cell is dropped rather than resting on one "
                             "optode."),
            ], width=2),
        ], className="g-3 mt-1"),
        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="qc-hyper-flags",
                options=[
                    {"label": "Significance testing (slow)", "value": "wtc_significance"},
                    {"label": "Mask cone of influence",      "value": "wtc_mask_coi"},
                    {"label": "Cross channels between brains (slow)",
                                                             "value": "wtc_channel_cross"},
                    {"label": "Bad channels: union over runs", "value": "bads_subject"},
                    {"label": "Save WTC maps (large)",         "value": "wtc_save_maps"},
                    {"label": "Skip alignment",              "value": "no_align"},
                    {"label": "Normalize recordings",        "value": "normalize"},
                ],
                value=[], inline=True, switch=True,
            )),
        ], className="g-3 mt-1"),
    ))


def _wtc_band():
    return html.Div(id="qc-wtc-band-section", children=_card(
        "Re-average saved WTC maps",
        dbc.Row([
            dbc.Col([
                dbc.Label("New Band (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="qc-band-fmin", type="number", step=0.001,
                              placeholder="lo"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="qc-band-fmax", type="number", step=0.01,
                              placeholder="hi"),
                ]),
                dbc.FormText("Both required. Must lie inside the axis the maps were computed on."),
            ], width=4),
            dbc.Col([
                dbc.Label("Suffix"),
                dbc.Input(id="qc-band-suffix", type="text", placeholder="e.g. band0p05-0p2"),
                dbc.FormText("Names the new tables so they sit beside the originals. "
                             "Defaults to the band."),
            ], width=4),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="qc-band-flags",
                options=[{"label": "Mask cone of influence", "value": "band_mask_coi"}],
                value=[], inline=True, switch=True,
            )),
        ], className="g-3 mt-1"),
        subtitle="Reads the npz written by hyper-post with Save WTC maps on. "
                 "Writes tables, not a report.",
    ))


def _window_raw():
    return html.Div(id="qc-window-section", children=_card(
        "Windowed drill-down",
        dbc.Row([
            dbc.Col([
                dbc.Label("Task Label"),
                dbc.Input(id="qc-task-label", type="text", placeholder="e.g. rest"),
                dbc.FormText("Required, one task at a time."),
            ], width=3),
            dbc.Col([
                dbc.Label("Window (s)"),
                dbc.InputGroup([
                    dbc.Input(id="qc-tstart", type="number", placeholder="start"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="qc-tend", type="number", placeholder="end"),
                ]),
            ], width=3),
            dbc.Col([
                dbc.Label("Name"),
                dbc.Input(id="qc-window-name", type="text", placeholder="suffix (optional)"),
                dbc.FormText("Names the output, so windows do not overwrite each other."),
            ], width=3),
            dbc.Col([
                dbc.Label("Align"),
                dcc.Dropdown(id="qc-align",
                             options=[{"label": "none", "value": "none"},
                                      {"label": "trigger", "value": "trigger"}],
                             value="none", clearable=False),
            ], width=3),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Cardiac Band (Hz)"),
                dbc.InputGroup([
                    dbc.Input(id="qc-cardiac-l", type="number", step=0.1,
                              placeholder="lo (adult ~0.7)"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="qc-cardiac-h", type="number", step=0.1,
                              placeholder="hi (adult ~1.5)"),
                ]),
                dbc.FormText("Required, population-dependent."),
            ], width=4),
            dbc.Col([
                dbc.Label("SCI Threshold"),
                dbc.Input(id="qc-sci-thresh", type="number", value=0.8, step=0.05),
            ], width=2),
            dbc.Col([
                dbc.Label("Window Length (s)"),
                dbc.Input(id="qc-window-length", type="number", value=10.0, step=1.0),
            ], width=2),
        ], className="g-3 mt-1"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Participant Label"),
                dbc.Input(id="qc-participant-label", type="text", placeholder="all subjects"),
                dbc.FormText("Space-separated for several."),
            ], width=3),
            dbc.Col([
                dbc.Label("Session Label"),
                dbc.Input(id="qc-window-session", type="text", placeholder="all sessions"),
            ], width=3),
            dbc.Col([
                dbc.Label("Trigger Name"),
                dbc.Input(id="qc-trigger-name", type="text", placeholder="align by trigger"),
                dbc.FormText("Only used when Align is trigger."),
            ], width=3),
        ], className="g-3 mt-1"),
    ))


layout = dbc.Container([
    dcc.Store(id="qc-command-store"),

    html.H3("QC Reports", className="mt-4"),
    html.P("Hyperscanning analysis, group aggregation and windowed drill-down. "
           "Every button here builds a fnirs-qc command and runs it.",
           className="text-muted"),

    _card(
        "Report",
        dbc.Row([
            dbc.Col([
                dbc.Label("Command"),
                dcc.Dropdown(
                    id="qc-command",
                    options=[
                        {"label": "Hyperscanning post-analysis (hyper-post)",
                         "value": "hyper-post"},
                        {"label": "Windowed group QC (window-raw)", "value": "window-raw"},
                        {"label": "Re-average saved WTC maps (wtc-band)",
                         "value": "wtc-band"},
                        *_AGGREGATE_COMMANDS,
                    ],
                    value="hyper-post",
                    clearable=False,
                ),
            ], width=6),
            dbc.Col([
                dbc.Label("Shell"),
                dcc.Dropdown(
                    id="qc-shell-select",
                    options=[{"label": s, "value": s} for s in ("bash", "cmd", "powershell")],
                    value=_DEFAULT_SHELL, clearable=False,
                ),
            ], width=3),
        ], className="g-3"),
    ),

    _dirs(),
    _hyper_post(),
    _wtc_band(),
    _window_raw(),

    _card(
        "Run",
        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="qc-run-flags",
                options=[{"label": "Skip BIDS validation", "value": "skip_bids_validation"}],
                value=[], switch=True,
            )),
        ], className="g-2 mb-3"),
        dbc.Row([
            dbc.Col(dbc.Button("Generate Command", id="qc-generate-btn",
                               color="secondary"), width="auto"),
            dbc.Col(dbc.Button("Run", id="qc-run-btn", color="primary"), width="auto"),
        ], className="g-2"),
        html.Pre(id="qc-command-preview",
                 style={"background": "rgba(0,0,0,0.04)", "padding": "0.75rem",
                        "borderRadius": "4px", "marginTop": "1rem", "fontSize": "12px",
                        "whiteSpace": "pre-wrap", "minHeight": "3rem"}),
        html.Div(id="qc-run-status", className="mt-3"),
    ),

    _card(
        "Report preview",
        html.Div(id="qc-report-preview"),
        subtitle="The HTML the run produced, served from disk so its panels resolve.",
    ),
], fluid=True)
