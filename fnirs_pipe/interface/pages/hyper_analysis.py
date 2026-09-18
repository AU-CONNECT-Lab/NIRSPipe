"""Page: the hyperscanning pipeline — `fnirs-hyper`, which runs on what `fnirs-pipe` wrote."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from fnirs_pipe.interface.components import (
    PATH, actions, band, card, field, params, section, split, switches,
)
from fnirs_pipe.qc.common.provenance import _DOMAIN

dash.register_page(__name__, path="/hyper-analysis", name="Hyper Analysis")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

# --desc takes a free-form string, but load_group_haemo refuses anything pre-Beer-Lambert,
# so the offer is the haemoglobin half of the domain map rather than a list retyped here
_DESC_CHOICES = [desc for desc, domain in _DOMAIN.items() if domain == "haemo"]

_COMMANDS = [
    {"label": "Analyse dyads (run)", "value": "run"},
    {"label": "Re-average saved WTC maps (band)", "value": "band"},
    {"label": "Merge WTC tables across dyads (merge)", "value": "merge"},
    {"label": "Rebuild the dyad landing pages (index)", "value": "index"},
]


def _scope():
    return card("Scope",
        params(
            field("Command",
                  dcc.Dropdown(id="hy-command", options=_COMMANDS,
                               value="run", clearable=False),
                  span=PATH),
            field("Derivatives directory",
                  dbc.Input(id="hy-output-dir", type="text",
                            placeholder="path to derivatives"),
                  span=PATH),
        ),
    )


def _run_section():
    return html.Div(id="hy-run-section", children=card(
        "Dyad analysis",
        section("Inputs",
            params(
                field("Pairs CSV",
                      dbc.Input(id="hy-pairs-csv", type="text",
                                placeholder="path to pairs.csv"),
                      span=PATH),
                field("Group ID",
                      dbc.Input(id="hy-group-id", type="text", placeholder="all groups")),
                field("Task label",
                      dbc.Input(id="hy-task", type="text", placeholder="all tasks")),
                field("Source stage",
                      dcc.Dropdown(id="hy-desc",
                                   options=[{"label": c, "value": c} for c in _DESC_CHOICES],
                                   value="preproc", clearable=False)),
                field("ROI mapping",
                      dbc.Input(id="hy-roi-mapping", type="text",
                                placeholder="roi.json (optional)"),
                      span=PATH),
            ),
        ),
        section("Coherence",
            params(
                band("Frequency range (Hz)",
                     dbc.Input(id="hy-wtc-fmin", type="number", value=0.004, step=0.001),
                     "–",
                     dbc.Input(id="hy-wtc-fmax", type="number", value=0.2, step=0.01)),
                band("Band averaged over (Hz)",
                     dbc.Input(id="hy-wtc-band-fmin", type="number", step=0.001,
                               placeholder="whole axis"),
                     "–",
                     dbc.Input(id="hy-wtc-band-fmax", type="number", step=0.01,
                               placeholder="whole axis")),
                field("Chromophore",
                      dcc.Dropdown(id="hy-wtc-chroma",
                                   options=[{"label": "HbO and HbR", "value": "both"},
                                            {"label": "HbO only", "value": "hbo"},
                                            {"label": "HbR only", "value": "hbr"}],
                                   placeholder="both")),
                field("Min channels per ROI",
                      dbc.Input(id="hy-wtc-roi-min-channels", type="number", min=1, step=1,
                                placeholder="2")),
            ),
        ),
        section("Coherence null",
            params(
                field("Surrogates",
                      dbc.Input(id="hy-wtc-mc-count", type="number", value=300,
                                min=1, step=50)),
                field("Seed",
                      dbc.Input(id="hy-wtc-seed", type="number", placeholder="none")),
                field("Phase-null iterations",
                      dbc.Input(id="hy-wtc-phasenull", type="number", min=0, step=10,
                                placeholder="0, off"),
                      span=PATH),
            ),
        ),
        section("Correlation",
            params(
                field("ISC threshold",
                      dbc.Input(id="hy-isc-threshold", type="number", step=0.05,
                                placeholder="default")),
                field("Whitening order",
                      dbc.Input(id="hy-isc-whiten", type="number", min=0, step=1,
                                placeholder="0, off")),
                field("Max lag (s)",
                      dbc.Input(id="hy-isc-max-lag", type="number", min=0, step=0.5,
                                placeholder="0, none")),
                field("Null iterations",
                      dbc.Input(id="hy-isc-phasenull", type="number", min=0, step=10,
                                placeholder="0, off")),
            ),
        ),
        section("Options",
            switches(dbc.Checklist(
                id="hy-flags",
                options=[
                    {"label": "Significance testing (slow)", "value": "wtc_significance"},
                    {"label": "Average the whole band (no COI mask)",
                     "value": "wtc_no_mask_coi"},
                    {"label": "Cross channels between brains (slow)",
                     "value": "wtc_channel_cross"},
                    {"label": "Cross channels for the null too (very slow)",
                     "value": "wtc_phase_null_cross"},
                    {"label": "Whole recording only (no per-condition results)",
                     "value": "no_by_condition"},
                    {"label": "Bad channels: union over runs", "value": "bads_subject"},
                    {"label": "Save WTC maps (large)", "value": "wtc_save_maps"},
                    {"label": "Skip alignment", "value": "no_align"},
                    {"label": "Normalize recordings", "value": "normalize"},
                    {"label": "Check only (compute nothing)", "value": "check_only"},
                ],
                value=[], switch=True,
            ), columns=True),
        ),
    ))


def _pair_null_section():
    return html.Div(id="hy-pairnull-section", children=card(
        "Re-paired null",
        params(
            field("Stand-in pool",
                  dbc.Select(id="hy-pair-pool",
                             options=[{"label": "Same position in the group", "value": "position"},
                                      {"label": "Any member of another group", "value": "any"}],
                             value="position"),
                  span=PATH),
            field("Draw limit",
                  dbc.Input(id="hy-pair-max", type="number", min=1, step=1,
                            placeholder="every eligible one")),
            switches(dbc.Checklist(
                id="hy-pair-flags",
                options=[{"label": "Every channel pair, not homologous only",
                          "value": "wtc_pair_cross"}],
                value=[], inline=True, switch=True,
            )),
        ),
        subtitle="Pairs one member with people from the other groups who did the same task. "
                 "Run it after a run: the band, the mask and the window come off the tables "
                 "that run wrote, not off this form. The number of draws is the number of "
                 "other groups, which is what limits how finely it can rank.",
    ))


def _band_section():
    return html.Div(id="hy-band-section", children=card(
        "Re-average saved WTC maps",
        params(
            band("New band (Hz)",
                 dbc.Input(id="hy-band-fmin", type="number", step=0.001, placeholder="lo"),
                 "–",
                 dbc.Input(id="hy-band-fmax", type="number", step=0.01, placeholder="hi")),
            field("Suffix",
                  dbc.Input(id="hy-band-suffix", type="text",
                            placeholder="e.g. band0p05-0p2"),
                  span=PATH),
            switches(dbc.Checklist(
                id="hy-band-flags",
                options=[{"label": "Average the whole band (no COI mask)",
                          "value": "band_no_mask_coi"}],
                value=[], inline=True, switch=True,
            )),
        ),
        subtitle="Reads the npz a run with Save WTC maps on wrote. Writes tables, not a report.",
    ))


def _window_section():
    return html.Div(id="hy-window-section", children=card(
        "Analysis window",
        params(
            band("Window (s)",
                 dbc.Input(id="hy-tstart", type="number", placeholder="start"),
                 "–",
                 dbc.Input(id="hy-tend", type="number", placeholder="end")),
        ),
    ))


def _run_panel():
    return card("Run",
        actions(
            dbc.Button("Run", id="hy-run-btn", color="success"),
            dbc.Button("Generate command", id="hy-generate-btn",
                       color="secondary", outline=True),
        ),
        dbc.Select(
            id="hy-shell-select",
            options=[{"label": s, "value": s} for s in ("bash", "cmd", "powershell")],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mt-3 mb-2",
        ),
        html.Pre(id="hy-command-preview", className="fp-command"),
        html.Div(id="hy-run-status", className="mt-3"),
    )


layout = dbc.Container([
    dcc.Store(id="hy-command-store"),

    html.H3("Hyper Analysis"),
    html.P("Wavelet coherence and inter-subject correlation over dyads, run by fnirs-hyper "
           "on what the individual pipeline already wrote. Reads a derivatives tree, never "
           "BIDS.", className="text-muted"),
    html.Hr(),

    split(
        main=[
            _scope(),
            _run_section(),
            _pair_null_section(),
            _band_section(),
            _window_section(),
            card("Report preview", html.Div(id="hy-report-preview")),
        ],
        aside=[_run_panel()],
    ),
], fluid=True)
