"""Callbacks for the QC Reports page: build a fnirs-qc argv, run it, show what it made."""

from __future__ import annotations

import subprocess
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, html

from fnirs_pipe.interface.report_serve import report_url
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("interface.qc_callbacks")

# commands whose whole argument list is one output_dir
_AGGREGATE = ("group-raw", "group-hyper-raw", "group-hyper-wtc", "provenance")

# which form sections each command needs; anything not listed here is hidden
_SECTIONS = {
    "hyper-post":  {"qc-hyper-post-section"},
    "window-raw":  {"qc-window-section"},
}

# report each command writes, relative to output_dir, best match first. hyper-post names its
# file after the group, so it is found by glob rather than named here.
_REPORTS = {
    "group-raw":       ["group_nirs.html"],
    "group-hyper-raw": ["group_hyper_nirs.html"],
    "window-raw":      ["group_nirs_*.html", "group_nirs.html"],
    "hyper-post":      ["group-*_hyper*.html", "group-*/*.html"],
}


def _num(flag: str, value) -> list[str]:
    return [flag, str(value)] if value is not None else []


def _text(flag: str, value) -> list[str]:
    return [flag, str(value)] if value else []


def _split(flag: str, value) -> list[str]:
    """A space-separated box feeds an nargs="+" flag, which repeats rather than joins."""
    return [x for item in (value or "").split() for x in (flag, item)]


def build_qc_args(command: str, opts: dict) -> list[str]:
    """Translate widget values into a fnirs-qc argv list.

    Kept beside the page rather than derived from the parser: argparse can say a flag exists
    but not which widget should fill it. `tests/test_gui_cli_surface.py` is what stops the
    two drifting apart.
    """
    args = ["fnirs-qc", command]

    if command in _AGGREGATE:
        return args + [opts["output_dir"]]

    args += [opts["bids_dir"], opts["output_dir"]]

    if command == "hyper-post":
        args += _text("--pairs-csv", opts.get("pairs_csv"))
        args += _text("--group-id", opts.get("group_id"))
        args += _text("--desc", opts.get("desc"))
        args += _text("--roi-mapping", opts.get("roi_mapping"))
        args += _num("--wtc-fmin", opts.get("wtc_fmin"))
        args += _num("--wtc-fmax", opts.get("wtc_fmax"))
        args += _num("--wtc-band-fmin", opts.get("wtc_band_fmin"))
        args += _num("--wtc-band-fmax", opts.get("wtc_band_fmax"))
        args += _num("--wtc-mc-count", opts.get("wtc_mc_count"))
        args += _num("--wtc-seed", opts.get("wtc_seed"))
        args += _num("--isc-threshold", opts.get("isc_threshold"))
        args += _split("--session-label", opts.get("hyper_session"))
        args += _split("--task-label", opts.get("hyper_task"))
        flags = opts.get("hyper_flags") or []
        if "wtc_significance" in flags:
            args.append("--wtc-significance")
        if "wtc_roi_cross" in flags:
            args.append("--wtc-roi-cross")
        if "wtc_channel_cross" in flags:
            args.append("--wtc-channel-cross")
        if "bads_subject" in flags:
            args += ["--bads-scope", "subject"]
        if "no_align" in flags:
            args.append("--no-align")
        if "normalize" in flags:
            args.append("--normalize")

    elif command == "window-raw":
        args += _text("--task-label", opts.get("task_label"))
        args += _num("--tstart", opts.get("tstart"))
        args += _num("--tend", opts.get("tend"))
        args += _text("--name", opts.get("window_name"))
        args += _text("--align", opts.get("align"))
        args += _num("--cardiac-l-freq", opts.get("cardiac_l"))
        args += _num("--cardiac-h-freq", opts.get("cardiac_h"))
        args += _num("--sci-threshold", opts.get("sci_thresh"))
        args += _num("--window-length", opts.get("window_length"))
        args += _text("--trigger-name", opts.get("trigger_name"))
        args += _split("--participant-label", opts.get("participant_label"))
        args += _split("--session-label", opts.get("window_session"))

    if "skip_bids_validation" in (opts.get("run_flags") or []):
        args.append("--skip-bids-validation")

    return args


def _missing(command: str, opts: dict) -> str | None:
    """The CLI would fail on these anyway; saying so here costs no subprocess."""
    if not opts.get("output_dir"):
        return "Output directory is required."
    if command in _AGGREGATE:
        return None
    if not opts.get("bids_dir"):
        return "BIDS directory is required."
    if command == "hyper-post" and not opts.get("pairs_csv"):
        return "hyper-post needs a pairs CSV."
    if command == "window-raw":
        if not opts.get("task_label"):
            return "window-raw needs a task label."
        if opts.get("tstart") is None or opts.get("tend") is None:
            return "window-raw needs both a window start and end."
        if opts.get("cardiac_l") is None or opts.get("cardiac_h") is None:
            return "Cardiac band lower/upper frequency is required (population-dependent)."
    return None


@callback(
    Output("qc-hyper-post-section", "style"),
    Output("qc-window-section", "style"),
    Input("qc-command", "value"),
)
def toggle_sections(command):
    needed = _SECTIONS.get(command, set())
    return tuple({"display": "block"} if section in needed else {"display": "none"}
                 for section in ("qc-hyper-post-section", "qc-window-section"))


_STATES = [
    State("qc-bids-dir", "value"), State("qc-output-dir", "value"),
    State("qc-pairs-csv", "value"), State("qc-group-id", "value"),
    State("qc-desc", "value"), State("qc-roi-mapping", "value"),
    State("qc-wtc-fmin", "value"), State("qc-wtc-fmax", "value"),
    State("qc-wtc-band-fmin", "value"), State("qc-wtc-band-fmax", "value"),
    State("qc-wtc-mc-count", "value"), State("qc-wtc-seed", "value"),
    State("qc-isc-threshold", "value"),
    State("qc-hyper-session", "value"), State("qc-hyper-task", "value"),
    State("qc-hyper-flags", "value"),
    State("qc-task-label", "value"), State("qc-tstart", "value"), State("qc-tend", "value"),
    State("qc-window-name", "value"), State("qc-align", "value"),
    State("qc-cardiac-l", "value"), State("qc-cardiac-h", "value"),
    State("qc-sci-thresh", "value"), State("qc-window-length", "value"),
    State("qc-trigger-name", "value"), State("qc-participant-label", "value"),
    State("qc-window-session", "value"), State("qc-run-flags", "value"),
]


def _opts(values) -> dict:
    keys = ["bids_dir", "output_dir", "pairs_csv", "group_id", "desc", "roi_mapping",
            "wtc_fmin", "wtc_fmax", "wtc_band_fmin", "wtc_band_fmax", "wtc_mc_count",
            "wtc_seed", "isc_threshold", "hyper_session", "hyper_task", "hyper_flags",
            "task_label", "tstart", "tend", "window_name",
            "align", "cardiac_l", "cardiac_h", "sci_thresh", "window_length",
            "trigger_name", "participant_label", "window_session", "run_flags"]
    return dict(zip(keys, values))


@callback(
    Output("qc-command-preview", "children"),
    Output("qc-command-store", "data"),
    Input("qc-generate-btn", "n_clicks"),
    State("qc-command", "value"),
    *_STATES,
    State("qc-shell-select", "value"),
    prevent_initial_call=True,
)
def generate_command(n_clicks, command, *values):
    *state_values, shell = values
    opts = _opts(state_values)

    problem = _missing(command, opts)
    if problem:
        return f"Error: {problem}", {}

    argv = build_qc_args(command, opts)
    cont = {"bash": "\\", "cmd": "^", "powershell": "`"}.get(shell, "\\")
    preview = f" {cont}\n".join([" ".join(argv[:2])] + [f"  {a}" for a in argv[2:]])
    return preview, {"argv": argv, "command": command, "output_dir": opts["output_dir"]}


def _find_report(command: str, output_dir: Path) -> Path | None:
    """The newest HTML matching what this command is known to write."""
    candidates: list[Path] = []
    for pattern in _REPORTS.get(command, []):
        candidates.extend(output_dir.glob(pattern))
        if candidates:
            break
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


@callback(
    Output("qc-run-status", "children"),
    Output("qc-report-preview", "children"),
    Input("qc-run-btn", "n_clicks"),
    State("qc-command-store", "data"),
    prevent_initial_call=True,
)
def run_command(n_clicks, stored):
    if not stored or not stored.get("argv"):
        return dbc.Alert("Click 'Generate Command' first.", color="warning",
                         className="mb-0"), None

    argv = stored["argv"]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return dbc.Alert(f"`{argv[0]}` not found on PATH — make sure fnirs-pipe is installed.",
                         color="danger", className="mb-0"), None
    except Exception as exc:
        return dbc.Alert(f"Failed to launch: {exc}", color="danger", className="mb-0"), None

    tail = "\n".join(((proc.stdout or "") + (proc.stderr or "")).splitlines()[-30:])
    ok = proc.returncode == 0
    alert = dbc.Alert([
        html.Div("Finished." if ok else f"Failed (exit {proc.returncode}).",
                 className="fw-bold"),
        html.Pre(tail, style={
            "background": "rgba(0,0,0,0.04)", "padding": "0.5rem", "borderRadius": "4px",
            "maxHeight": "300px", "overflow": "auto", "fontSize": "12px",
            "margin": "0.5rem 0 0"}),
    ], color="success" if ok else "danger", className="mb-0")

    if not ok:
        return alert, None

    report = _find_report(stored["command"], Path(stored["output_dir"]))
    if report is None:
        return alert, html.Small(
            "This command writes tables rather than a report, or the report was not found.",
            className="text-muted")

    return alert, html.Div([
        html.Small(str(report), className="text-muted d-block mb-2"),
        html.Iframe(src=report_url(report),
                    style={"width": "100%", "height": "700px", "border": "1px solid #dee2e6",
                           "borderRadius": "4px"}),
    ])
