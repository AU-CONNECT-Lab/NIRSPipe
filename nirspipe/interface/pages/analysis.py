"""Page 2: pipeline analysis configuration and execution."""

from __future__ import annotations

import platform

import dash
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash import dcc, html

from nirspipe.interface.components import (
    PATH,
    action_field,
    actions,
    band,
    card,
    field,
    params,
    section,
    split,
    switches,
)

from nirspipe.cli.run import (
    _DRIFT_CHOICES,
    _HRF_CHOICES,
    _NOISE_CHOICES,
    _SHORT_CHANNEL_CHOICES,
    CUTOFF_PATTERN,
    NOISE_MODEL_PATTERN,
)

from nirspipe.pipeline.denoise import (
    DEFAULT_FILTER_METHOD,
    DEFAULT_FILTER_ORDER,
    FILTER_METHODS,
)

dash.register_page(__name__, path="/analysis", name="Analysis")


def _opts(choices: list[str]) -> list[dict]:
    return [{"label": c, "value": c} for c in choices]

_DEFAULT_SHELL = "cmd" if platform.system() == "Windows" else "bash"

_DAG_COLOURS = {
    "prep": "#E67E22",
    "post": "#2980B9",
    "glm":  "#27AE60",
    "qc":   "#8E44AD",
}

_DAG_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "content":          "data(label)",
            "text-valign":      "center",
            "text-halign":      "center",
            "text-wrap":        "wrap",
            "shape":            "round-rectangle",
            "width":            "140px",
            "height":           "44px",
            "font-size":        "12px",
            "font-family":      "sans-serif",
            "background-opacity": 0,
            "border-width":     2,
            "color":            "#374151",
            "font-weight":      "bold",
        },
    },
    {
        "selector": "edge",
        "style": {
            "curve-style":        "bezier",
            "width":              1.5,
            "target-arrow-shape": "triangle",
            "arrow-scale":        0.9,
        },
    },
] + [
    {
        "selector": f".{cls}",
        "style": {
            "border-color":       colour,
            "line-color":         colour,
            "target-arrow-color": colour,
        },
    }
    for cls, colour in _DAG_COLOURS.items()
]


def _dag_legend():
    labels = [
        ("prep", "Preprocessing"),
        ("post", "Postprocessing"),
        ("glm",  "GLM"),
    ]
    return html.Div([
        html.Span(
            label,
            style={
                "display":      "inline-block",
                "marginRight":  "12px",
                "padding":      "2px 8px",
                "borderRadius": "4px",
                "border":       f"2px solid {_DAG_COLOURS[cls]}",
                "color":        "#374151",
                "fontSize":     "12px",
                "fontWeight":   "bold",
            },
        )
        for cls, label in labels
    ], className="mb-2")


def _dag_card():
    return card(
        "Pipeline DAG",
        html.Div([
            _dag_legend(),
            dbc.Button("Download PNG", id="an-dag-download", color="secondary",
                       outline=True, size="sm"),
        ], className="d-flex justify-content-between align-items-start"),
        cyto.Cytoscape(
            id="an-dag",
            elements=[],
            stylesheet=_DAG_STYLESHEET,
            style={"width": "100%", "height": "260px", "background": "#FAFAFA"},
            layout={
                "name":          "dagre",
                "rankDir":       "LR",
                "nodeSep":       35,
                "rankSep":       80,
                "spacingFactor": 1.0,
            },
            # a fixed, fitted view: dragging the canvas only lost the graph off its edge
            userZoomingEnabled=False,
            userPanningEnabled=False,
            autoungrabify=True,
        ),
    )


# ---- Form column ----

def _scope():
    return card("Scope",
        params(
            field("BIDS directory",
                  dbc.Input(id="an-bids-dir", type="text", placeholder="/path/to/bids"),
                  span=PATH),
            field("Output directory",
                  dbc.Input(id="an-output-dir", type="text", placeholder="/path/to/output"),
                  span=PATH),
            field("Session label(s)",
                  dbc.Input(id="an-session-label", type="text", placeholder="01  (optional)")),
            field("Task label(s)",
                  dbc.Input(id="an-task-label", type="text", placeholder="rest  (optional)")),
            action_field(dbc.Button("Detect subjects", id="an-detect-btn",
                                    color="primary")),
        ),
        html.Div(id="an-subjects-result", className="mt-2"),
        html.Div(id="an-subjects-container"),
        params(
            field("Run record",
                  dbc.Input(id="an-config-path", type="text",
                            placeholder="<output>/sub-01/logs/sub-01.toml"),
                  span=PATH,
                  hint="Fills the form with that run's settings; the subjects stay yours to pick."),
            action_field(dbc.Button("Load settings", id="an-config-load-btn",
                                    color="secondary", outline=True)),
        ),
        html.Div(id="an-config-status", className="mt-2"),
    )


def _preprocessing():
    return card("Preprocessing",
        section("Optical density",
            params(
                field("DPF", dbc.Input(id="an-dpf", type="number", placeholder="6.0")),
                field("Motion correction",
                      dcc.Dropdown(id="an-motion-correction",
                                   options=[{"label": "TDDR", "value": "tddr"},
                                            {"label": "Wavelet", "value": "wavelet"},
                                            {"label": "Targeted PCA", "value": "tpca"},
                                            {"label": "Spline", "value": "spline"},
                                            {"label": "Spline-SG", "value": "spline_sg"},
                                            {"label": "Wavelet + TDDR", "value": "wavelet_tddr"},
                                            {"label": "Spline + Wavelet",
                                             "value": "spline_wavelet"},
                                            {"label": "None", "value": "none"}],
                                   value="tddr", clearable=False)),
                # the placeholders name the chosen method's defaults; empty sends nothing
                field("Detection threshold (SD)",
                      dbc.Input(id="an-motion-std-thresh", type="number", step=0.5, min=0.5),
                      hint="Spline detection; fewer marks when higher.",
                      id="an-motion-std-thresh-wrap"),
                field("Wavelet IQR",
                      dbc.Input(id="an-wavelet-iqr", type="number", step=0.1, min=0.1),
                      hint="Coefficients past it are zeroed; more removed when lower.",
                      id="an-wavelet-iqr-wrap"),
                band("Separations (mm)",
                     "≤", dbc.Input(id="an-short-max-dist", type="number", step=0.5,
                                         min=0.1, placeholder="10"),
                     "≥", dbc.Input(id="an-long-min-dist", type="number", step=0.5,
                                         min=0.1, placeholder="15"),
                     "to", dbc.Input(id="an-long-max-dist", type="number", step=0.5,
                                     min=0.1, placeholder="none"),
                     span=PATH),
            ),
         key="an-od", open=True),
        section("Channel screening",
            params(
                field("SCI threshold",
                      dbc.Input(id="an-sci-thresh", type="number", placeholder="e.g. 0.8",
                                min=0.0, max=1.0, step=0.01)),
                field("PSP threshold",
                      dbc.Input(id="an-psp-thresh", type="number", value=0.1,
                                min=0.0, max=1.0, step=0.01)),
                field("Coupled windows",
                      dbc.Input(id="an-min-good-frac", type="number", value=0.75,
                                min=0.0, max=1.0, step=0.05),
                      hint="Share a channel needs to be kept. This is what rejects."),
                field("Screening scope",
                      dcc.Dropdown(id="an-screen-scope",
                                   options=[{"label": "Whole run", "value": "run"},
                                            {"label": "Task blocks only", "value": "task"}],
                                   value="run", clearable=False)),
                field("SCI / PSP window (s)",
                      dbc.Input(id="an-window-length", type="number", step=1,
                                min=1, placeholder="10")),
            ),
         key="an-screen", open=False),
        section("Channel and time tables",
            params(
                field("Bad channels",
                      dbc.Input(id="an-bad-channels", type="text",
                                placeholder="S1_D1,S2_D3 or a table path"),
                      span=PATH,
                      hint="nirspipe-rate export writes this table, keeps included."),
                field("Keep spans",
                      dbc.Input(id="an-keep-spans", type="text",
                                placeholder="table of stretches to use (optional)"),
                      span=PATH),
                field("Keep spans from marker",
                      dbc.Input(id="an-keep-spans-from", type="text", placeholder="marker"),
                      hint="Onsets in the keep-spans table count from this marker."),
                field("Bad spans",
                      dbc.Input(id="an-bad-spans", type="text",
                                placeholder="table of corrupted stretches (optional)"),
                      span=PATH),
            ),
         key="an-tables", open=False),
        section("Frequency bands",
            params(
                band("Cardiac (Hz)",
                     dbc.Input(id="an-cardiac-l", type="number", step=0.1,
                               placeholder="lo (adult ~0.7)"),
                     "–",
                     dbc.Input(id="an-cardiac-h", type="number", step=0.1,
                               placeholder="hi (adult ~2.0)")),
                band("Respiration (Hz)",
                     dbc.Input(id="an-resp-l", type="number", step=0.1,
                               placeholder="lo (adult ~0.1)"),
                     "–",
                     dbc.Input(id="an-resp-h", type="number", step=0.1,
                               placeholder="hi (adult ~0.5)")),
            ),
         key="an-bands", open=False),
        section("Epochs",
            params(
                field("Epoch tmin (s)",
                      dbc.Input(id="an-epoch-tmin", type="number", placeholder="-5")),
                field("Epoch tmax (s)",
                      dbc.Input(id="an-epoch-tmax", type="number", placeholder="25")),
                field("Trial chunk (s)",
                      dbc.Input(id="an-epoch-chunk", type="number", step=1, min=1,
                                placeholder="off"),
                      hint="Cut each block into trials this long before epoching."),
                switches(dbc.Checklist(id="an-by-condition",
                                       options=[{"label": "A QC page per condition",
                                                 "value": "by_condition"}],
                                       value=[], switch=True),
                         hint="Sliced out of the run's own windows; nothing is measured again."),
            ),
         key="an-epochs", open=False),
        section("GVTD censoring",
            params(
                field("Censoring",
                      dcc.Dropdown(id="an-gvtd-censor",
                                   options=[{"label": "Off", "value": "off"},
                                            {"label": "Long channels", "value": "long"},
                                            {"label": "Short channels", "value": "short"},
                                            {"label": "All channels", "value": "all"}],
                                   value="off", clearable=False),
                      hint="Annotates BAD_gvtd. Nothing is cut."),
                field("Threshold (SD)",
                      dbc.Input(id="an-gvtd-n-std", type="number", step=0.5, min=0.5,
                                placeholder="10"),
                      id="an-gvtd-n-std-wrap"),
                field("Fill",
                      dcc.Dropdown(id="an-censor-fill",
                                   options=[{"label": "Linear", "value": "linear"},
                                            {"label": "Cubic spline", "value": "spline"},
                                            {"label": "Lomb-Scargle", "value": "lomb"}],
                                   value="linear", clearable=False),
                      hint="Fills the spans before the bandpass; needs a post mode.",
                      id="an-censor-fill-wrap"),
                field("Min kept time (s)",
                      dbc.Input(id="an-min-time", type="number", step=1, min=0,
                                placeholder="0 (off)"),
                      hint="FC and ALFF are skipped when less time lies outside the spans.",
                      id="an-min-time-wrap"),
                html.Div(switches(dbc.Checklist(id="an-export-mean-filled",
                                                options=[{"label": "Also write a mean-filled copy "
                                                                   "(--export-mean-filled)",
                                                          "value": "export"}],
                                                value=[], switch=True),
                                  hint="BAD_ frames at each channel's kept mean, for tools that "
                                       "ignore annotations; not for ALFF or a GLM. Needs a "
                                       "post mode."),
                         id="an-export-mean-filled-wrap"),
            ),
         key="an-gvtd", open=False),
    )


def _postprocessing():
    return card("Postprocessing",
        html.Div(
            dbc.RadioItems(
                id="an-post-mode",
                options=[{"label": "None", "value": "none"},
                         {"label": "Denoise", "value": "denoise"},
                         {"label": "GLM", "value": "glm"},
                         {"label": "Rest", "value": "rest"}],
                value="none", inline=True,
            ),
            className="mb-3",
        ),
        params(
            # text rather than number, so `none` can switch off a cutoff the mode fills in
            field("High-pass (Hz)",
                  dbc.Input(id="an-high-pass", type="text", inputMode="decimal", debounce=True,
                            placeholder="off", pattern=CUTOFF_PATTERN)),
            field("Low-pass (Hz)",
                  dbc.Input(id="an-low-pass", type="text", inputMode="decimal", debounce=True,
                            placeholder="off", pattern=CUTOFF_PATTERN)),
            field("Filter",
                  dbc.Select(id="an-filter-method",
                             options=[{"label": m, "value": m} for m in FILTER_METHODS],
                             value=DEFAULT_FILTER_METHOD)),
            field("Filter order",
                  dbc.Input(id="an-filter-order", type="number",
                            value=DEFAULT_FILTER_ORDER, min=1, step=1)),
            field("Resample (Hz)",
                  dbc.Input(id="an-resample", type="number", placeholder="2.0")),
            field("n_jobs",
                  dbc.Input(id="an-n-jobs", type="number", value=1, min=1, step=1)),
        ),
        html.Div(id="an-band-note", className="mt-2"),

        # every mode honours these, so not GLM-only; an empty field takes the mode's default
        html.Div(id="an-confound-section", children=[
            section("Confound regression",
                params(
                    field("Drift model",
                          dcc.Dropdown(id="an-drift-model", options=_opts(_DRIFT_CHOICES),
                                       value=None, placeholder="mode default")),
                    field("Drift high-pass (Hz)",
                          dbc.Input(id="an-drift-high-pass", type="number",
                                    placeholder="from your design"),
                          id="an-drift-hp-wrap"),
                    field("Drift order",
                          dbc.Input(id="an-drift-order", type="number",
                                    value=1, min=0, step=1),
                          id="an-drift-order-wrap"),
                    field("Short channel",
                          dcc.Dropdown(id="an-short-channel",
                                       options=_opts(_SHORT_CHANNEL_CHOICES),
                                       value="none", clearable=False)),
                    field("ROI mapping",
                          dbc.Input(id="an-roi-mapping", type="text",
                                    placeholder="path to roi.json (optional)"),
                          span=PATH,
                          hint="Needed for the ROI and seed connectivity products."),
                    switches(dbc.Checklist(id="an-fc",
                                           options=[{"label": "Write FC products (--fc)",
                                                     "value": "fc"}],
                                           value=[], switch=True),
                             hint="Denoise and GLM. Rest writes them anyway."),
                ),
             key="an-confound", open=False),
        ]),

        html.Div(id="an-glm-section", children=[
            section("GLM",
                params(
                    field("HRF model",
                          dcc.Dropdown(id="an-hrf-model", options=_opts(_HRF_CHOICES),
                                       value=None, placeholder="mode default")),
                    # free text with a suggestion list, not a dropdown: the CLI takes any
                    # `arN` and a closed list here would be the narrower surface. `pattern`
                    # is the CLI's own rule, so the browser refuses what the CLI would
                    field("Noise model",
                          html.Div([
                              dbc.Input(id="an-noise-model", debounce=True,
                                        placeholder="mode default",
                                        list="an-noise-model-suggest",
                                        pattern=NOISE_MODEL_PATTERN),
                              html.Datalist(id="an-noise-model-suggest",
                                            children=[html.Option(value=c)
                                                      for c in _NOISE_CHOICES]),
                          ])),
                    field("Stim duration (s)",
                          dbc.Input(id="an-stim-dur", type="number", placeholder="optional")),
                    field("Events table",
                          dbc.Input(id="an-events-path", type="text",
                                    placeholder="*_events.tsv, instead of stim duration"),
                          span=PATH),
                    field("Contrast file",
                          dbc.Input(id="an-contrast-file", type="text",
                                    placeholder="contrasts.toml (optional)"),
                          span=PATH),
                ),
             key="an-glm", open=True),
        ]),
    )


# ---- Panel that stays put while the form scrolls ----

def _run_panel():
    return card("Run",
        dbc.Checklist(
            id="an-flags",
            options=[{"label": "Dry run", "value": "dry_run"},
                     {"label": "Skip BIDS validation", "value": "skip_bids_validation"},
                     {"label": "No report", "value": "no_report"},
                     {"label": "Combine runs", "value": "combine_runs"}],
            value=["dry_run"],
            switch=True,
            className="mb-3",
        ),
        actions(
            dbc.Button("Run pipeline", id="an-run-btn", color="success"),
            dbc.Button("Stop", id="an-stop-btn", color="danger", outline=True, disabled=True),
            dbc.Button("Generate command", id="an-generate-btn",
                       color="secondary", outline=True),
        ),
        html.Div(id="an-run-status", className="mt-2 small"),
    )


def _command_panel():
    return card("Command",
        dbc.Select(
            id="an-shell-select",
            options=[{"label": "bash / zsh (macOS, Linux)", "value": "bash"},
                     {"label": "cmd (Windows, Anaconda Prompt)", "value": "cmd"},
                     {"label": "PowerShell (Windows)", "value": "powershell"}],
            value=_DEFAULT_SHELL,
            size="sm",
            className="mb-2",
        ),
        html.Pre(id="an-command-preview", className="fp-command"),
    )


layout = dbc.Container([
    dcc.Store(id="an-subjects-store", storage_type="memory"),
    dcc.Store(id="an-command-store",  storage_type="memory"),
    dcc.Store(id="an-run-store",      storage_type="memory"),
    # the run is polled rather than waited on; disabled until there is something to poll
    dcc.Interval(id="an-run-tick", interval=1000, disabled=True),

    html.H3("Analysis"),
    html.Hr(),

    split(
        main=[_scope(), _preprocessing(), _postprocessing(), _dag_card()],
        aside=[_run_panel(), _command_panel()],
    ),
], fluid=True)


