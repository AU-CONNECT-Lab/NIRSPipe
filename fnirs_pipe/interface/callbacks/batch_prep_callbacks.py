"""Callbacks for the batch data preparation page."""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, no_update

from fnirs_pipe.interface import process_stream
from fnirs_pipe.interface.callbacks._cli_run import log_panel, preview_text
from fnirs_pipe.interface.cli_args import build_prep_args, missing_prep
from fnirs_pipe.interface.grid import rows_minus_clicked
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("interface.batch_prep_callbacks")


# ── Directory sync: page → shared store ──────────────────────────────────────

@callback(
    Output("app-bids-dir",   "data", allow_duplicate=True),
    Output("app-output-dir", "data", allow_duplicate=True),
    Input("bp-bids-dir",  "value"),
    Input("bp-deriv-dir", "value"),
    prevent_initial_call=True,
)
def _bp_dirs_to_store(bids_dir, deriv_dir):
    return bids_dir, deriv_dir


@callback(
    Output("bp-bids-dir",  "value"),
    Output("bp-deriv-dir", "value"),
    Input("app-bids-dir",   "data"),
    Input("app-output-dir", "data"),
    prevent_initial_call="initial_duplicate",
)
def _restore_dirs(bids_dir, output_dir):
    return bids_dir or no_update, output_dir or no_update


# ── Subject detection ─────────────────────────────────────────────────────────

@callback(
    Output("bp-subjects-result",    "children"),
    Output("bp-subjects-container", "children"),
    Input("bp-detect-btn",   "n_clicks"),
    Input("bp-select-all-btn", "n_clicks"),
    Input("bp-clear-btn",    "n_clicks"),
    State("bp-bids-dir",     "value"),
    State("bp-subjects-container", "children"),
    prevent_initial_call=True,
)
def manage_subjects(detect_clicks, select_all_clicks, clear_clicks, bids_dir, current_container):
    trigger = ctx.triggered_id

    if trigger == "bp-clear-btn":
        return "", []

    if trigger == "bp-select-all-btn":
        if not current_container:
            return no_update, no_update
        checklist = current_container
        if isinstance(checklist, dict):
            options = checklist.get("props", {}).get("options", [])
            all_vals = [o["value"] for o in options]
            patched_checklist = dbc.Checklist(
                id="bp-subject-checklist",
                options=options,
                value=all_vals,
                className="mt-1",
                inline=True,
            )
            return no_update, patched_checklist
        return no_update, no_update

    if trigger == "bp-detect-btn":
        if not bids_dir or not Path(bids_dir).is_dir():
            return dbc.Alert("Invalid BIDS directory.", color="warning"), []
        subjects = sorted(
            d.name[4:] for d in Path(bids_dir).iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        if not subjects:
            return dbc.Alert("No subjects found.", color="warning"), []
        checklist = dbc.Checklist(
            id="bp-subject-checklist",
            options=[{"label": f"sub-{s}", "value": s} for s in subjects],
            value=subjects,
            className="mt-1",
            inline=True,
        )
        result = dbc.Alert(
            f"Found {len(subjects)} subject(s), all selected",
            color="success", className="mb-0 py-2",
        )
        return result, checklist

    return no_update, no_update


# ── Toggle operation panel ────────────────────────────────────────────────────

@callback(
    Output("bp-subjects-card",  "style"),
    Output("bp-run-filter-card","style"),
    Output("bp-markers-panel",  "style"),
    Output("bp-crop-panel",     "style"),
    Output("bp-hyper-panel",    "style"),
    Input("bp-operation", "value"),
)
def toggle_operation(op):
    show, hide = {}, {"display": "none"}
    is_hyper = op == "hyper_align"
    return (
        hide if is_hyper else show,
        hide if is_hyper else show,
        show if op == "markers"   else hide,
        show if op == "crop"      else hide,
        show if is_hyper          else hide,
    )


# ── Toggle marker sub-operation ───────────────────────────────────────────────

@callback(
    Output("bp-shift-panel",    "style"),
    Output("bp-duration-panel", "style"),
    Output("bp-rename-panel",   "style"),
    Input("bp-marker-op", "value"),
)
def toggle_marker_op(op):
    show, hide = {}, {"display": "none"}
    return (
        show if op == "shift"        else hide,
        show if op == "set_duration" else hide,
        show if op == "rename"       else hide,
    )


# ── Toggle crop mode ──────────────────────────────────────────────────────────

@callback(
    Output("bp-crop-single", "style"),
    Output("bp-crop-multi",  "style"),
    Input("bp-crop-mode", "value"),
)
def toggle_crop_mode(mode):
    show, hide = {}, {"display": "none"}
    return (show, hide) if mode == "single" else (hide, show)


# ── Add rename pair ───────────────────────────────────────────────────────────

@callback(
    Output("bp-rename-table", "rowData", allow_duplicate=True),
    Input("bp-rename-add-btn", "n_clicks"),
    State("bp-rename-table",   "virtualRowData"),
    prevent_initial_call=True,
)
def add_rename_pair(n_clicks, rows):
    rows = list(rows or [])
    rows.append({"from_name": "", "to_name": ""})
    return rows


# ── Delete rename pair ────────────────────────────────────────────────────────

@callback(
    Output("bp-rename-table", "rowData", allow_duplicate=True),
    Input("bp-rename-table",  "cellClicked"),
    State("bp-rename-table",  "virtualRowData"),
    prevent_initial_call=True,
)
def delete_rename_pair(cell, rows):
    kept = rows_minus_clicked(cell, rows)
    return no_update if kept is None else kept          # [] means the last row was deleted


# ── Add crop segment ──────────────────────────────────────────────────────────

@callback(
    Output("bp-crop-seg-table", "rowData", allow_duplicate=True),
    Input("bp-seg-add-btn",     "n_clicks"),
    State("bp-crop-seg-table",  "virtualRowData"),
    prevent_initial_call=True,
)
def add_crop_segment(n_clicks, rows):
    rows = list(rows or [])
    rows.append({"onset": 0.0, "duration": 30.0})
    return rows


# ── Delete crop segment ───────────────────────────────────────────────────────

@callback(
    Output("bp-crop-seg-table", "rowData", allow_duplicate=True),
    Input("bp-crop-seg-table",  "cellClicked"),
    State("bp-crop-seg-table",  "virtualRowData"),
    prevent_initial_call=True,
)
def delete_crop_segment(cell, rows):
    kept = rows_minus_clicked(cell, rows)
    return no_update if kept is None else kept



# ── Generate the equivalent fnirs-prep command ───────────────────────────────

# the segments table has to reach the CLI as a file, and the derivatives tree is where the
# record of what was cut belongs
_SEGMENTS_TSV = "batch-crop-segments.tsv"


def _rename_pairs(rows) -> list[str]:
    return [f"{r.get('from_name')}:{r.get('to_name')}"
            for r in (rows or [])
            if r.get("from_name") and r.get("to_name")]


def _write_segments(deriv_dir: str, rows) -> str:
    import pandas as pd

    path = Path(deriv_dir) / _SEGMENTS_TSV
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["onset", "duration", "task"]).to_csv(path, sep="\t", index=False)
    return str(path)


@callback(
    Output("bp-command-preview", "children"),
    Output("bp-command-store",   "data"),
    Input("bp-generate-btn", "n_clicks"),
    State("bp-subjects-container", "children"),
    State("bp-bids-dir",    "value"),
    State("bp-deriv-dir",   "value"),
    State("bp-ses",         "value"),
    State("bp-task",        "value"),
    State("bp-run",         "value"),
    State("bp-operation",   "value"),
    State("bp-marker-op",   "value"),
    State("bp-shift-val",   "value"),
    State("bp-duration-val", "value"),
    State("bp-rename-table", "virtualRowData"),
    State("bp-crop-mode",   "value"),
    State("bp-crop-tmin",   "value"),
    State("bp-crop-tmax",   "value"),
    State("bp-crop-seg-table", "virtualRowData"),
    State("bp-crop-combine", "value"),
    State("bp-group-csv",   "value"),
    State("bp-n-jobs",      "value"),
    State("bp-shell-select", "value"),
    prevent_initial_call=True,
)
def generate_command(
    n_clicks, subjects_container,
    bids_dir, deriv_dir, ses, task, run,
    operation, marker_op, shift_val, duration_val, rename_rows,
    crop_mode, crop_tmin, crop_tmax, seg_rows, combine_val,
    group_csv, n_jobs, shell,
):
    subjects = []
    if isinstance(subjects_container, dict):
        subjects = subjects_container.get("props", {}).get("value", []) or []

    opts = {
        "bids_dir": bids_dir, "deriv_dir": deriv_dir, "subjects": subjects,
        "ses": ses, "task": task, "run": run, "n_jobs": n_jobs,
        "marker_op": marker_op, "shift": shift_val, "set_duration": duration_val,
        "rename": _rename_pairs(rename_rows),
        "crop_mode": crop_mode, "crop_tmin": crop_tmin, "crop_tmax": crop_tmax,
        "segments": seg_rows, "combine": bool(combine_val),
        "group_csv": group_csv,
    }

    problem = missing_prep(operation, opts)
    if problem:
        return f"Error: {problem}", {}

    if operation == "crop" and crop_mode == "multi":
        opts["segments_path"] = _write_segments(deriv_dir, seg_rows)

    argv = build_prep_args(operation, opts)
    return preview_text(argv, shell), {"argv": argv}


# ── Run the generated command ────────────────────────────────────────────────

@callback(
    Output("bp-log",       "children"),
    Output("bp-run-store", "data"),
    Output("bp-run-tick",  "disabled"),
    Output("bp-stop-btn",  "disabled"),
    Output("bp-run-btn",   "disabled"),
    Input("bp-run-btn",    "n_clicks"),
    State("bp-command-store", "data"),
    prevent_initial_call=True,
)
def run_batch(n_clicks, cmd_data):
    if not cmd_data or not cmd_data.get("argv"):
        return (dbc.Alert("Click 'Generate command' first.", color="warning",
                          className="mb-0"), None, True, True, False)

    argv = cmd_data["argv"]
    try:
        run_id = process_stream.start(argv)
    except FileNotFoundError:
        return (dbc.Alert(f"`{argv[0]}` not found on PATH.", color="danger",
                          className="mb-0"), None, True, True, False)
    except Exception as exc:
        return (dbc.Alert(f"Failed to launch: {exc}", color="danger", className="mb-0"),
                None, True, True, False)

    logger.info("started %s as run %s", argv[0], run_id)
    return log_panel("Running...", [], 0, "info"), run_id, False, False, True


@callback(
    Output("bp-log",      "children", allow_duplicate=True),
    Output("bp-run-tick", "disabled", allow_duplicate=True),
    Output("bp-stop-btn", "disabled", allow_duplicate=True),
    Output("bp-run-btn",  "disabled", allow_duplicate=True),
    Input("bp-run-tick",  "n_intervals"),
    State("bp-run-store", "data"),
    prevent_initial_call=True,
)
def stream_run_output(_n, run_id):
    if not run_id:
        return no_update, True, True, False

    lines, returncode, dropped = process_stream.poll(run_id)
    if returncode is None:
        return log_panel("Running...", lines, dropped, "info"), False, False, True

    if returncode == 0:
        header, color = "Finished.", "success"
    elif returncode < 0:
        header, color = f"Stopped (signal {-returncode}).", "warning"
    else:
        header, color = f"Failed (exit {returncode}).", "danger"
    process_stream.forget(run_id)
    return log_panel(header, lines, dropped, color), True, True, False


@callback(
    Output("bp-stop-btn", "disabled", allow_duplicate=True),
    Input("bp-stop-btn",  "n_clicks"),
    State("bp-run-store", "data"),
    prevent_initial_call=True,
)
def stop_batch(_n, run_id):
    # the tick reports the outcome; this only asks the process to end
    if run_id:
        process_stream.stop(run_id)
    return True
