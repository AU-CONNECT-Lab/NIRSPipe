"""Page: the hyperscanning pipeline (`nirspipe-hyper`), which runs on what `nirspipe` wrote."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from nirspipe.interface.components import (
    PATH, actions, band, card, field, params, section, split, switches,
)
from nirspipe.qc.common.provenance import _DOMAIN
from nirspipe.utils import ROI_MIN_CHANNELS

dash.register_page(__name__, path="/hyper-analysis", name="Hyper Analysis")

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

# --desc takes a free-form string, but load_group_haemo refuses anything pre-Beer-Lambert,
# so the offer is the haemoglobin half of the domain map rather than a list retyped here
_DESC_CHOICES = [desc for desc, domain in _DOMAIN.items() if domain == "haemo"]

_COMMANDS = [
    {"label": "Analyse dyads (nirspipe-hyper)", "value": "run"},
    {"label": "Re-paired null (nirspipe-hyper-pairnull)", "value": "pair-null"},
    {"label": "Read the draws above the cell (nirspipe-hyper-groupnull)", "value": "group-null"},
    {"label": "Re-average saved WTC maps (nirspipe-hyper-band)", "value": "band"},
    {"label": "Merge WTC tables across dyads (nirspipe-hyper-merge)", "value": "merge"},
    {"label": "Rebuild the dyad landing pages (nirspipe-hyper-index)", "value": "index"},
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
                # only these two commands open a member's recording, and this section is
                # shown for exactly those two
                field("Source derivatives",
                      dbc.Input(id="hy-derivatives-dir", type="text",
                                placeholder="tree nirspipe wrote, holding sub-*/nirs/"),
                      span=PATH),
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
                     dbc.Input(id="hy-wtc-fmin", type="number", value=0.004, step="any"),
                     "–",
                     dbc.Input(id="hy-wtc-fmax", type="number", value=0.2, step="any")),
                band("Band (Hz)",
                     dbc.Input(id="hy-shared-band-fmin", type="number", step="any",
                               placeholder="lo"),
                     "–",
                     dbc.Input(id="hy-shared-band-fmax", type="number", step="any",
                               placeholder="hi"),
                     hint="Both metrics: the coherence averages over it, the correlation filters to it."),
                # each pair overrides the shared band for one metric, as on the command line
                band("WTC band (Hz)",
                     dbc.Input(id="hy-wtc-band-fmin", type="number", step="any",
                               placeholder="lo"),
                     "–",
                     dbc.Input(id="hy-wtc-band-fmax", type="number", step="any",
                               placeholder="hi"),
                     hint="Coherence only. Empty takes the band, else the whole axis."),
                band("ISC band (Hz)",
                     dbc.Input(id="hy-isc-fmin", type="number", step="any",
                               placeholder="lo"),
                     "–",
                     dbc.Input(id="hy-isc-fmax", type="number", step="any",
                               placeholder="hi"),
                     hint="Correlation only. Empty takes the band, else the whole passband."),
                field("Chromophore",
                      dcc.Dropdown(id="hy-wtc-chroma",
                                   options=[{"label": "HbO and HbR", "value": "both"},
                                            {"label": "HbO only", "value": "hbo"},
                                            {"label": "HbR only", "value": "hbr"}],
                                   placeholder="both")),
                field("Min channels per ROI",
                      dbc.Input(id="hy-wtc-roi-min-channels", type="number", min=1, step=1,
                                placeholder=str(ROI_MIN_CHANNELS))),
                # left empty the condition is the unit, which is what the CLI defaults to
                field("Equal windows (s)",
                      dbc.Input(id="hy-wtc-window-s", type="number", min=1, step="any",
                                placeholder="condition length")),
                field("Prewhitening order (s)",
                      dbc.Input(id="hy-wtc-whiten", type="number", min=0, step="any",
                                placeholder="0, off")),
            ),
        ),
        section("Coherence null",
            params(
                field("Surrogates",
                      dbc.Input(id="hy-wtc-mc-count", type="number", value=300,
                                min=1, step=1)),
                field("Seed",
                      dbc.Input(id="hy-wtc-seed", type="number", step=1, placeholder="none")),
                field("Phase-null iterations",
                      dbc.Input(id="hy-wtc-phasenull", type="number", min=0, step=1,
                                placeholder="0, off"),
                      span=PATH),
            ),
        ),
        section("Correlation",
            params(
                field("ISC threshold",
                      dbc.Input(id="hy-isc-threshold", type="number", step="any",
                                placeholder="default")),
                field("Prewhitening order (s)",
                      dbc.Input(id="hy-isc-whiten", type="number", min=0, step="any",
                                placeholder="0, off")),
                field("Max lag (s)",
                      dbc.Input(id="hy-isc-max-lag", type="number", min=0, step="any",
                                placeholder="0, none")),
                field("Null iterations",
                      dbc.Input(id="hy-isc-phasenull", type="number", min=0, step=1,
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
                value=["wtc_channel_cross", "wtc_phase_null_cross"], switch=True,
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
                options=[{"label": "Homologous pairings only, even over a crossed table",
                          "value": "wtc_pair_homologous"}],
                value=[], inline=True, switch=True,
            )),
        ),
        subtitle="Pairs each member with people from other groups on the same task. Runs "
                 "after nirspipe-hyper and reuses its settings; one draw per other group.",
    ))


def _group_null_section():
    return html.Div(id="hy-groupnull-section", children=card(
        "Null above the cell",
        params(
            field("Task",
                  dbc.Input(id="hy-gn-task", type="text", placeholder="task label (required)")),
            field("Chromophores",
                  dbc.Checklist(id="hy-gn-chroma", inline=True, value=[],
                                options=[{"label": "HbO", "value": "hbo"},
                                         {"label": "HbR", "value": "hbr"}]),
                  hint="None ticked reads every chromophore the draws hold."),
            field("Draws to read",
                  dbc.Checklist(id="hy-gn-null", inline=True, value=[],
                                options=[{"label": "Re-paired", "value": "repaired"},
                                         {"label": "Phase-scrambled", "value": "phase"}]),
                  hint="None ticked reads every null whose draws are on disk."),
            field("ROI mapping",
                  dbc.Input(id="hy-gn-roi-mapping", type="text",
                            placeholder="path to roi.json, optional"),
                  span=PATH),
            field("Resamples",
                  dbc.Input(id="hy-gn-resample", type="number", min=1, step=1,
                            value=20000)),
            field("Seed",
                  dbc.Input(id="hy-gn-seed", type="number", step=1, placeholder="none")),
            field("P correction",
                  dbc.Select(id="hy-gn-p-correction",
                             options=[{"label": "None (raw p only)", "value": "none"},
                                      {"label": "FDR, Benjamini–Hochberg", "value": "fdr_bh"},
                                      {"label": "FDR, Benjamini–Yekutieli", "value": "fdr_by"},
                                      {"label": "Holm", "value": "holm"},
                                      {"label": "Bonferroni", "value": "bonferroni"}],
                             value="none")),
            field("ISC test",
                  dbc.Select(id="hy-gn-isc-test",
                             options=[{"label": "Signed, two-sided", "value": "signed"},
                                      {"label": "Magnitude |z|, above null",
                                       "value": "magnitude"}],
                             value="signed")),
        ),
        subtitle="Tests the pairing as a whole, per occasion and over the cohort, from the "
                 "draws nirspipe-hyper-pairnull wrote.",
    ))


def _band_section():
    return html.Div(id="hy-band-section", children=card(
        "Re-average saved WTC maps",
        params(
            band("New band (Hz)",
                 dbc.Input(id="hy-band-fmin", type="number", step="any", placeholder="lo"),
                 "–",
                 dbc.Input(id="hy-band-fmax", type="number", step="any", placeholder="hi")),
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
                 dbc.Input(id="hy-tstart", type="number", step="any", placeholder="start"),
                 "–",
                 dbc.Input(id="hy-tend", type="number", step="any", placeholder="end")),
        ),
    ))


def _run_panel():
    return card("Run",
        actions(
            dbc.Button("Run", id="hy-run-btn", color="success"),
            dbc.Button("Stop", id="hy-stop-btn", color="danger",
                       outline=True, disabled=True),
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
    dcc.Store(id="hy-run-store", storage_type="memory"),
    # the run is polled rather than waited on; disabled until there is something to poll
    dcc.Interval(id="hy-run-tick", interval=1000, disabled=True),

    html.H3("Hyper Analysis"),
    html.P("Wavelet coherence and inter-subject correlation over dyads, from the individual "
           "pipeline's derivatives.", className="text-muted"),
    html.Hr(),

    split(
        main=[
            _scope(),
            _run_section(),
            _pair_null_section(),
            _group_null_section(),
            _band_section(),
            _window_section(),
            card("Report preview", html.Div(id="hy-report-preview")),
        ],
        aside=[_run_panel()],
    ),
], fluid=True)
