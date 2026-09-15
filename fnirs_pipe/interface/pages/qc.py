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

from fnirs_pipe.interface.components import actions, band, card, field, params, section, split, switches
from fnirs_pipe.qc.common.provenance import _DOMAIN

dash.register_page(__name__, path="/qc", name="QC Reports")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

# --desc takes a free-form string, but load_group_haemo refuses anything pre-Beer-Lambert,
# so the offer is the haemoglobin half of the domain map rather than a list retyped here
_DESC_CHOICES = [desc for desc, domain in _DOMAIN.items() if domain == "haemo"]

# one output_dir and nothing else, so they share a form
_AGGREGATE_COMMANDS = [
    {"label": "Cohort QC (individual subjects)", "value": "cohort"},
    {"label": "Cohort QC (hyperscanning groups)", "value": "cohort-hyper"},
    {"label": "Merge WTC tables across dyads", "value": "merge"},
    {"label": "Rebuild the dyad landing pages", "value": "index"},
    {"label": "Provenance graphs", "value": "provenance"},
]


def _report():
    return card("Report",
        params(
            field("Command",
                  dcc.Dropdown(
                      id="qc-command",
                      options=[{"label": "Hyperscanning analysis (fnirs-hyper run)",
                                "value": "run"},
                               {"label": "Re-average saved WTC maps (fnirs-hyper band)",
                                "value": "band"},
                               *_AGGREGATE_COMMANDS],
                      value="run", clearable=False),
                  span=2),
            field("Derivatives / output directory",
                  dbc.Input(id="qc-output-dir", type="text",
                            placeholder="path to derivatives"),
                  span=2),
        ),
    )


def _hyper_post():
    return html.Div(id="qc-hyper-post-section", children=card(
        "Hyperscanning post-analysis",
        section("Inputs",
            params(
                field("Pairs CSV",
                      dbc.Input(id="qc-pairs-csv", type="text", placeholder="path to pairs.csv"),
                      span=2),
                field("Group ID",
                      dbc.Input(id="qc-group-id", type="text", placeholder="all groups")),
                field("Task label",
                      dbc.Input(id="qc-hyper-task", type="text", placeholder="all tasks")),
                field("Source stage",
                      dcc.Dropdown(id="qc-desc",
                                   options=[{"label": c, "value": c} for c in _DESC_CHOICES],
                                   value="preproc", clearable=False)),
                field("ROI mapping",
                      dbc.Input(id="qc-roi-mapping", type="text",
                                placeholder="roi.json (optional)"),
                      span=2),
            ),
        ),
        section("Coherence",
            params(
                band("Frequency range (Hz)",
                     dbc.Input(id="qc-wtc-fmin", type="number", value=0.004, step=0.001),
                     "–",
                     dbc.Input(id="qc-wtc-fmax", type="number", value=0.2, step=0.01)),
                band("Band averaged over (Hz)",
                     dbc.Input(id="qc-wtc-band-fmin", type="number", step=0.001,
                               placeholder="whole axis"),
                     "–",
                     dbc.Input(id="qc-wtc-band-fmax", type="number", step=0.01,
                               placeholder="whole axis")),
                field("Chromophore",
                      dcc.Dropdown(id="qc-wtc-chroma",
                                   options=[{"label": "HbO and HbR", "value": "both"},
                                            {"label": "HbO only", "value": "hbo"},
                                            {"label": "HbR only", "value": "hbr"}],
                                   placeholder="both")),
                field("Min channels per ROI",
                      dbc.Input(id="qc-wtc-roi-min-channels", type="number", min=1, step=1,
                                placeholder="2")),
            ),
        ),
        section("Coherence null",
            params(
                field("Surrogates",
                      dbc.Input(id="qc-wtc-mc-count", type="number", value=300,
                                min=1, step=50)),
                field("Seed",
                      dbc.Input(id="qc-wtc-seed", type="number", placeholder="none")),
                field("Pseudo-dyad iterations",
                      dbc.Input(id="qc-wtc-pseudo", type="number", min=0, step=10,
                                placeholder="0, off"),
                      span=2),
            ),
        ),
        section("Correlation",
            params(
                field("ISC threshold",
                      dbc.Input(id="qc-isc-threshold", type="number", step=0.05,
                                placeholder="default")),
                field("Whitening order",
                      dbc.Input(id="qc-isc-whiten", type="number", min=0, step=1,
                                placeholder="0, off")),
                field("Max lag (s)",
                      dbc.Input(id="qc-isc-max-lag", type="number", min=0, step=0.5,
                                placeholder="0, none")),
                field("Null iterations",
                      dbc.Input(id="qc-isc-pseudo", type="number", min=0, step=10,
                                placeholder="0, off")),
            ),
        ),
        section("Options",
            dbc.Checklist(
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
                    {"label": "Save WTC maps (large)", "value": "wtc_save_maps"},
                    {"label": "Skip alignment", "value": "no_align"},
                    {"label": "Normalize recordings", "value": "normalize"},
                    {"label": "Check only (compute nothing)", "value": "check_only"},
                ],
                value=[], inline=True, switch=True,
            ),
        ),
    ))


def _wtc_band():
    return html.Div(id="qc-wtc-band-section", children=card(
        "Re-average saved WTC maps",
        params(
            band("New band (Hz)",
                 dbc.Input(id="qc-band-fmin", type="number", step=0.001, placeholder="lo"),
                 "–",
                 dbc.Input(id="qc-band-fmax", type="number", step=0.01, placeholder="hi")),
            field("Suffix",
                  dbc.Input(id="qc-band-suffix", type="text",
                            placeholder="e.g. band0p05-0p2"),
                  span=2),
            switches(dbc.Checklist(
                id="qc-band-flags",
                options=[{"label": "Average the whole band (no COI mask)",
                          "value": "band_no_mask_coi"}],
                value=[], inline=True, switch=True,
            ), span=2),
        ),
        subtitle="Reads the npz the hyper level writes with Save WTC maps on."
                 " Writes tables, not a report.",
    ))


def _analysis_window():
    return html.Div(id="qc-window-section", children=card(
        "Analysis window",
        params(
            band("Window (s)",
                 dbc.Input(id="qc-tstart", type="number", placeholder="start"),
                 "–",
                 dbc.Input(id="qc-tend", type="number", placeholder="end")),
        ),
    ))


def _run_panel():
    return card("Run",
        actions(
            dbc.Button("Run", id="qc-run-btn", color="success"),
            dbc.Button("Generate command", id="qc-generate-btn",
                       color="secondary", outline=True),
        ),
        dbc.Select(
            id="qc-shell-select",
            options=[{"label": s, "value": s} for s in ("bash", "cmd", "powershell")],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mt-3 mb-2",
        ),
        html.Pre(id="qc-command-preview", className="fp-command"),
        html.Div(id="qc-run-status", className="mt-3"),
    )


layout = dbc.Container([
    dcc.Store(id="qc-command-store"),

    html.H3("QC Reports"),
    html.P("Hyperscanning analysis and group aggregation. Every button here builds a "
           "command and runs it: the group QC and provenance entries call fnirs-qc, the "
           "rest call fnirs-hyper. Neither reads BIDS, so only a derivatives directory "
           "is needed.",
           className="text-muted"),
    html.Hr(),

    split(
        main=[
            _report(),
            _hyper_post(),
            _wtc_band(),
            _analysis_window(),
            card("Report preview",
                 html.Div(id="qc-report-preview"),
                 subtitle="The HTML the run produced, served from disk so its panels resolve."),
        ],
        aside=[_run_panel()],
    ),
], fluid=True)
