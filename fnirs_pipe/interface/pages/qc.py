"""Page 6: quality control reports — hyperscanning analysis and group aggregates.

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
    {"label": "Merge WTC tables across dyads", "value": "merge"},
    {"label": "Rebuild the dyad landing pages", "value": "index"},
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
                dbc.Label("Derivatives / Output Directory"),
                dbc.Input(id="qc-output-dir", type="text", placeholder="path to derivatives"),
                dbc.FormText("Every command on this page reads the derivatives tree only."),
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
                dbc.Label("Task Label"),
                dbc.Input(id="qc-hyper-task", type="text", placeholder="all tasks"),
                dbc.FormText("Filters the pairs CSV, which already names a task per row."),
            ], width=4),
            dbc.Col([
                dbc.Label("ISC Threshold"),
                dbc.Input(id="qc-isc-threshold", type="number", step=0.05,
                          placeholder="default"),
            ], width=2),
            dbc.Col([
                dbc.Label("Pseudo-dyad Iterations"),
                dbc.Input(id="qc-wtc-pseudo", type="number", min=0, step=10,
                          placeholder="0, off"),
                dbc.FormText("The null a coherence is read against, computed in the same "
                             "run. One full WTC run each; published work uses 100. Leave "
                             "empty and no null is written."),
            ], width=2),
            dbc.Col([
                dbc.Label("Min Channels per ROI"),
                dbc.Input(id="qc-wtc-roi-min-channels", type="number", min=1, step=1,
                          placeholder="2"),
                dbc.FormText("Below this an ROI cell is dropped rather than resting on one "
                             "optode."),
            ], width=2),
            dbc.Col([
                dbc.Label("Chromophore"),
                dcc.Dropdown(id="qc-wtc-chroma",
                             options=[{"label": "HbO and HbR", "value": "both"},
                                      {"label": "HbO only", "value": "hbo"},
                                      {"label": "HbR only", "value": "hbr"}],
                             placeholder="both"),
                dbc.FormText("Two parallel passes, never mixed and never averaged, so both "
                             "takes twice the time. A coupling in HbO with nothing in HbR "
                             "is a caution flag."),
            ], width=2),
        ], className="g-3 mt-1"),
        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="qc-hyper-flags",
                options=[
                    {"label": "Significance testing (slow)", "value": "wtc_significance"},
                    {"label": "Average the whole band (no COI mask)",
                                                             "value": "wtc_no_mask_coi"},
                    {"label": "Cross channels between brains (slow)",
                                                             "value": "wtc_channel_cross"},
                    {"label": "Cross channels for the null too (very slow)",
                                                             "value": "wtc_pseudo_cross"},
                    {"label": "Whole recording only (no per-condition results)",
                                                             "value": "no_by_condition"},
                    {"label": "Bad channels: union over runs", "value": "bads_subject"},
                    {"label": "Save WTC maps (large)",         "value": "wtc_save_maps"},
                    {"label": "Skip alignment",              "value": "no_align"},
                    {"label": "Normalize recordings",        "value": "normalize"},
                    {"label": "Check only (compute nothing)", "value": "check_only"},
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
                options=[{"label": "Average the whole band (no COI mask)",
                          "value": "band_no_mask_coi"}],
                value=[], inline=True, switch=True,
            )),
        ], className="g-3 mt-1"),
        subtitle="Reads the npz the hyper level writes with Save WTC maps on. "
                 "Writes tables, not a report.",
    ))


def _analysis_window():
    return html.Div(id="qc-window-section", children=_card(
        "Analysis window",
        dbc.Row([
            dbc.Col([
                dbc.Label("Window (s)"),
                dbc.InputGroup([
                    dbc.Input(id="qc-tstart", type="number", placeholder="start"),
                    dbc.InputGroupText("–"),
                    dbc.Input(id="qc-tend", type="number", placeholder="end"),
                ]),
                dbc.FormText("Measured from the shared trigger, after alignment. "
                             "Leave empty to use the whole recording."),
            ], width=4),
        ], className="g-3"),
    ))


layout = dbc.Container([
    dcc.Store(id="qc-command-store"),

    html.H3("QC Reports", className="mt-4"),
    html.P("Hyperscanning analysis and group aggregation. Every button here builds a "
           "command and runs it: the group QC and provenance entries call fnirs-qc, the "
           "rest call fnirs-hyper. Neither reads BIDS, so only a derivatives directory "
           "is needed.",
           className="text-muted"),

    _card(
        "Report",
        dbc.Row([
            dbc.Col([
                dbc.Label("Command"),
                dcc.Dropdown(
                    id="qc-command",
                    options=[
                        {"label": "Hyperscanning analysis (fnirs-hyper run)",
                         "value": "run"},
                        {"label": "Re-average saved WTC maps (fnirs-hyper band)",
                         "value": "band"},
                        *_AGGREGATE_COMMANDS,
                    ],
                    value="run",
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
    _analysis_window(),

    _card(
        "Run",
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
