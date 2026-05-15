"""Callbacks for the batch data preparation page."""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, ctx, html, no_update


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
            import dash
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
            f"Found {len(subjects)} subject(s) — all selected",
            color="success", className="mb-0 py-2",
        )
        return result, checklist

    return no_update, no_update


# ── Toggle operation panel ────────────────────────────────────────────────────

@callback(
    Output("bp-markers-panel", "style"),
    Output("bp-crop-panel",    "style"),
    Input("bp-operation", "value"),
)
def toggle_operation(op):
    show, hide = {}, {"display": "none"}
    return (show, hide) if op == "markers" else (hide, show)


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
    Output("bp-rename-table", "data", allow_duplicate=True),
    Input("bp-rename-add-btn", "n_clicks"),
    State("bp-rename-table",   "data"),
    prevent_initial_call=True,
)
def add_rename_pair(n_clicks, rows):
    rows = rows or []
    rows.append({"from_name": "", "to_name": ""})
    return rows


# ── Add crop segment ──────────────────────────────────────────────────────────

@callback(
    Output("bp-crop-seg-table", "data", allow_duplicate=True),
    Input("bp-seg-add-btn",     "n_clicks"),
    State("bp-crop-seg-table",  "data"),
    prevent_initial_call=True,
)
def add_crop_segment(n_clicks, rows):
    rows = rows or []
    rows.append({"onset": 0.0, "duration": 30.0})
    return rows


# ── Run batch ─────────────────────────────────────────────────────────────────

@callback(
    Output("bp-log",     "children"),
    Input("bp-run-btn",  "n_clicks"),
    State("bp-subjects-container", "children"),
    State("bp-bids-dir",    "value"),
    State("bp-deriv-dir",   "value"),
    State("bp-ses",         "value"),
    State("bp-task",        "value"),
    State("bp-run",         "value"),
    State("bp-operation",   "value"),
    State("bp-marker-op",   "value"),
    State("bp-shift-val",   "value"),
    State("bp-duration-val","value"),
    State("bp-rename-table","data"),
    State("bp-crop-mode",   "value"),
    State("bp-crop-tmin",   "value"),
    State("bp-crop-tmax",   "value"),
    State("bp-crop-seg-table", "data"),
    State("bp-crop-combine","value"),
    State("bp-n-jobs",      "value"),
    prevent_initial_call=True,
)
def run_batch(
    n_clicks, subjects_container,
    bids_dir, deriv_dir, ses, task, run,
    operation, marker_op, shift_val, duration_val, rename_rows,
    crop_mode, crop_tmin, crop_tmax, seg_rows, combine_val,
    n_jobs,
):
    if not bids_dir or not deriv_dir:
        return dbc.Alert("Set BIDS and derivatives directories.", color="warning")

    # Resolve selected subjects
    subjects: list[str] = []
    if isinstance(subjects_container, dict):
        subjects = subjects_container.get("props", {}).get("value", [])
    if not subjects:
        return dbc.Alert("No subjects selected.", color="warning")

    bids_path  = Path(bids_dir)
    deriv_path = Path(deriv_dir)
    ses_  = ses  or None
    task_ = task or None
    run_  = run  or None
    n_jobs_ = int(n_jobs or 1)

    from joblib import Parallel, delayed

    results: list[tuple[str, bool, str]] = []

    if operation == "markers":
        from fnirs_pipe.pipeline.edit_markers import apply_markers

        rename: list[str] | None = None
        if marker_op == "rename":
            rename = [
                f"{r['from_name']}:{r['to_name']}"
                for r in (rename_rows or [])
                if r.get("from_name") and r.get("to_name") is not None
            ]
            if not rename:
                return dbc.Alert("Add at least one rename pair.", color="warning")

        shift_    = float(shift_val)    if marker_op == "shift"        and shift_val    is not None else None
        duration_ = float(duration_val) if marker_op == "set_duration" and duration_val is not None else None

        if marker_op == "shift" and shift_ is None:
            return dbc.Alert("Enter a shift value.", color="warning")
        if marker_op == "set_duration" and duration_ is None:
            return dbc.Alert("Enter a duration value.", color="warning")

        def _apply_one(sub):
            try:
                out = apply_markers(
                    bids_path, deriv_path, sub,
                    ses=ses_, task=task_, run=run_,
                    shift=shift_, set_duration=duration_, rename=rename,
                )
                return sub, True, str(out)
            except Exception as exc:
                return sub, False, str(exc)

        results = Parallel(n_jobs=n_jobs_, prefer="threads")(
            delayed(_apply_one)(sub) for sub in subjects
        )

    else:  # crop
        from fnirs_pipe.pipeline.crop import crop_snirf

        combine = bool(combine_val)
        segments_df = None

        if crop_mode == "single":
            if crop_tmin is None and crop_tmax is None:
                return dbc.Alert("Set tmin or tmax.", color="warning")
        else:
            import pandas as pd
            if not seg_rows:
                return dbc.Alert("Add at least one segment.", color="warning")
            segments_df = pd.DataFrame(seg_rows, columns=["onset", "duration"])
            segments_df["onset"]    = pd.to_numeric(segments_df["onset"],    errors="coerce").fillna(0.0)
            segments_df["duration"] = pd.to_numeric(segments_df["duration"], errors="coerce").fillna(0.0)

        def _crop_one(sub):
            try:
                import tempfile, pandas as pd
                if segments_df is not None:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".tsv", delete=False
                    ) as f:
                        tmp = Path(f.name)
                        segments_df.to_csv(f, sep="\t", index=False)
                    try:
                        out = crop_snirf(
                            bids_path, deriv_path, sub,
                            ses=ses_, task=task_, run=run_,
                            segments_path=tmp, combine=combine,
                        )
                    finally:
                        tmp.unlink(missing_ok=True)
                else:
                    out = crop_snirf(
                        bids_path, deriv_path, sub,
                        ses=ses_, task=task_, run=run_,
                        tmin=float(crop_tmin) if crop_tmin is not None else None,
                        tmax=float(crop_tmax) if crop_tmax is not None else None,
                    )
                return sub, True, ", ".join(str(p) for p in out)
            except Exception as exc:
                return sub, False, str(exc)

        results = Parallel(n_jobs=n_jobs_, prefer="threads")(
            delayed(_crop_one)(sub) for sub in subjects
        )

    return _build_log(results)


def _build_log(results: list[tuple[str, bool, str]]):
    rows = []
    for sub, ok, msg in results:
        badge = dbc.Badge("✓", color="success") if ok else dbc.Badge("✗", color="danger")
        rows.append(html.Tr([
            html.Td(badge, style={"width": "40px"}),
            html.Td(f"sub-{sub}", style={"width": "100px", "fontWeight": "600"}),
            html.Td(html.Small(msg, className="text-muted" if ok else "text-danger")),
        ]))

    n_ok   = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    summary_color = "success" if n_fail == 0 else ("warning" if n_ok > 0 else "danger")
    summary = dbc.Alert(
        f"{n_ok}/{len(results)} succeeded" + (f", {n_fail} failed" if n_fail else ""),
        color=summary_color, className="mb-2 py-2",
    )

    return [
        summary,
        dbc.Table(html.Tbody(rows), bordered=False, size="sm",
                  className="mb-0", style={"fontSize": "0.82rem"}),
    ]
