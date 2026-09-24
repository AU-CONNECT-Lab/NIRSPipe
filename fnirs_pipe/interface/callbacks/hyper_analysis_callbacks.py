"""Callbacks for the Hyper Analysis page: build an `fnirs-hyper` argv, run it, show the report."""

from __future__ import annotations

from dash import Input, Output, State, callback

from fnirs_pipe.interface.callbacks._cli_run import preview_text, run_and_report
from fnirs_pipe.interface.cli_args import build_qc_args, missing
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("interface.hyper_analysis_callbacks")

# which form sections each command needs; anything not listed here is hidden
_SECTIONS = {
    "run":  {"hy-run-section", "hy-window-section"},
    # the pool and the limit are its own; the rest of the selection is the run form's
    "pair-null": {"hy-run-section", "hy-pairnull-section"},
    # reads finished tables, so it shares nothing with the run form
    "group-null": {"hy-groupnull-section"},
    "band": {"hy-band-section"},
}

_ALL_SECTIONS = ("hy-run-section", "hy-pairnull-section", "hy-groupnull-section",
                 "hy-band-section", "hy-window-section")

# report each command writes, relative to output_dir, best match first. The hyper level names
# its file after the group, so it is found by glob rather than named here.
_STATES = [
    State("hy-output-dir", "value"),
    State("hy-derivatives-dir", "value"),
    State("hy-pairs-csv", "value"), State("hy-group-id", "value"),
    State("hy-desc", "value"), State("hy-roi-mapping", "value"),
    State("hy-wtc-fmin", "value"), State("hy-wtc-fmax", "value"),
    State("hy-wtc-band-fmin", "value"), State("hy-wtc-band-fmax", "value"),
    State("hy-wtc-mc-count", "value"), State("hy-wtc-seed", "value"),
    State("hy-isc-threshold", "value"),
    State("hy-wtc-phasenull", "value"), State("hy-wtc-roi-min-channels", "value"),
    State("hy-wtc-window-s", "value"), State("hy-wtc-whiten", "value"),
    State("hy-isc-whiten", "value"), State("hy-isc-max-lag", "value"),
    State("hy-isc-fmin", "value"), State("hy-isc-fmax", "value"),
    State("hy-isc-phasenull", "value"),
    State("hy-wtc-chroma", "value"),
    State("hy-task", "value"),
    State("hy-flags", "value"),
    State("hy-pair-pool", "value"), State("hy-pair-max", "value"),
    State("hy-pair-flags", "value"),
    State("hy-gn-task", "value"), State("hy-gn-chroma", "value"),
    State("hy-gn-null", "value"), State("hy-gn-roi-mapping", "value"),
    State("hy-gn-resample", "value"), State("hy-gn-seed", "value"),
    State("hy-band-fmin", "value"), State("hy-band-fmax", "value"),
    State("hy-band-suffix", "value"), State("hy-band-flags", "value"),
    State("hy-tstart", "value"), State("hy-tend", "value"),
]

_KEYS = ["output_dir", "derivatives_dir", "pairs_csv", "group_id", "desc", "roi_mapping",
         "wtc_fmin", "wtc_fmax", "wtc_band_fmin", "wtc_band_fmax", "wtc_mc_count",
         "wtc_seed", "isc_threshold", "wtc_phase_null", "wtc_roi_min_channels",
         "wtc_window_s", "wtc_whiten",
         "isc_whiten", "isc_max_lag", "isc_fmin", "isc_fmax", "isc_phase_null",
         "wtc_chroma", "hyper_task", "hyper_flags",
         "wtc_pair_pool", "wtc_pair_max", "pair_flags",
         "gn_task", "gn_chroma", "gn_null", "gn_roi_mapping",
         "gn_resample", "gn_seed",
         "band_fmin", "band_fmax", "band_suffix", "band_flags",
         "tstart", "tend"]


@callback(
    *[Output(sec, "style") for sec in _ALL_SECTIONS],
    Input("hy-command", "value"),
)
def toggle_sections(command):
    needed = _SECTIONS.get(command, set())
    return tuple({"display": "block"} if sec in needed else {"display": "none"}
                 for sec in _ALL_SECTIONS)


@callback(
    Output("hy-command-preview", "children"),
    Output("hy-command-store", "data"),
    Input("hy-generate-btn", "n_clicks"),
    State("hy-command", "value"),
    *_STATES,
    State("hy-shell-select", "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, command, *values):
    *state_values, shell = values
    opts = dict(zip(_KEYS, state_values))

    problem = missing(command, opts)
    if problem:
        return f"Error: {problem}", {}

    argv = build_qc_args(command, opts)
    return (preview_text(argv, shell),
            {"argv": argv, "command": command, "output_dir": opts["output_dir"]})


@callback(
    Output("hy-run-status", "children"),
    Output("hy-report-preview", "children"),
    Input("hy-run-btn", "n_clicks"),
    State("hy-command-store", "data"),
    prevent_initial_call=True,
)
def run_command(n_clicks, stored):
    return run_and_report(stored)
