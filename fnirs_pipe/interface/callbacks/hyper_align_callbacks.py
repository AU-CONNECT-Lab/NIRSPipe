"""Callbacks for the hyperscanning align page."""

from __future__ import annotations

import json
from pathlib import Path
import hashlib

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, ctx, html, no_update

from fnirs_pipe.interface.callbacks._cli_run import run_and_report
from fnirs_pipe.io.derivatives import channel_decisions_path, entity_of
from fnirs_pipe.interface.cli_args import build_raw_qc_args, missing_raw_qc
from fnirs_pipe.interface.theme import style_figure
from fnirs_pipe.exceptions import AlignmentError

# aligned_raws not JSON-serializable, so keep in process memory
_ALIGNED_CACHE: dict[str, dict] = {}


def _cache_key(bids_dir: str, group_csv: str) -> str:
    return hashlib.md5(f"{bids_dir}|{group_csv}".encode()).hexdigest()[:16]


def _to_haemo(raws: dict) -> dict:
    import mne
    result = {}
    for sid, raw in raws.items():
        try:
            raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
            result[sid] = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
        except Exception:
            result[sid] = raw
    return result


# ── Dir sync ──────────────────────────────────────────────────────────────────

@callback(
    Output("app-bids-dir",   "data", allow_duplicate=True),
    Output("app-output-dir", "data", allow_duplicate=True),
    Input("ha-bids-dir",  "value"),
    Input("ha-deriv-dir", "value"),
    prevent_initial_call=True,
)
def _ha_dirs_to_store(bids_dir, deriv_dir):
    return bids_dir, deriv_dir


@callback(
    Output("ha-bids-dir",  "value"),
    Output("ha-deriv-dir", "value"),
    Input("app-bids-dir",   "data"),
    Input("app-output-dir", "data"),
    prevent_initial_call="initial_duplicate",
)
def _restore_dirs(bids_dir, output_dir):
    return bids_dir or no_update, output_dir or no_update


# ── Load & Align ──────────────────────────────────────────────────────────────

@callback(
    Output("ha-load-status",   "children"),
    Output("ha-results-panel", "style"),
    Output("ha-offset-table",  "rowData"),
    Output("ha-group-select",  "options"),
    Output("ha-group-select",  "value"),
    Input("ha-load-btn",       "n_clicks"),
    State("ha-bids-dir",       "value"),
    State("ha-deriv-dir",      "value"),
    State("ha-group-csv",      "value"),
    prevent_initial_call=True,
)
def load_and_align(n_clicks, bids_dir, deriv_dir, group_csv):
    _HIDE = {"display": "none"}
    _SHOW = {}

    if not bids_dir or not group_csv:
        return (
            dbc.Alert("Set BIDS directory and group CSV.", color="warning"),
            _HIDE, no_update, no_update, no_update,
        )

    from fnirs_pipe.io.snirf import read_snirf
    from fnirs_pipe.pipeline.hyper import align_recordings, member_snirfs, parse_group_csv

    try:
        groups = parse_group_csv(Path(group_csv))
    except Exception as exc:
        return (
            dbc.Alert(f"Group CSV error: {exc}", color="danger"),
            _HIDE, no_update, no_update, no_update,
        )

    key = _cache_key(bids_dir, group_csv)
    all_aligned: dict[tuple, dict] = {}
    errors: list[str] = []

    for (group_id, task), group in groups.items():
        try:
            paths = member_snirfs(Path(bids_dir), group)
            raws = {sid: read_snirf(p, verbose=False) for sid, p in paths.items()}
            aligned_raws, offsets = align_recordings(raws, task)
            all_aligned[(group_id, task)] = {
                "aligned_raws": aligned_raws,
                "offsets":      offsets,
                "subject_ids":  [e.subject_id for e in group],
                # the file each member was read from, which export and decisions follow
                "paths":        paths,
            }
        except AlignmentError as exc:
            errors.append(f"Group {group_id}/{task}: {exc}")
        except Exception as exc:
            errors.append(f"Group {group_id}/{task}: {exc}")

    if not all_aligned:
        msg = " | ".join(errors) if errors else "No groups could be aligned."
        return (
            dbc.Alert(msg, color="danger"),
            _HIDE, no_update, no_update, no_update,
        )

    _ALIGNED_CACHE[key] = {
        "groups":    all_aligned,
        "bids_dir":  bids_dir,
        "deriv_dir": deriv_dir,
        "group_csv": group_csv,
    }

    offset_rows = []
    for (group_id, task), info in all_aligned.items():
        for sid in info["subject_ids"]:
            raw_a = info["aligned_raws"].get(sid)
            offset_rows.append({
                "group_id":   group_id,
                "subject_id": sid,
                "task":       task,
                "offset_s":   round(info["offsets"].get(sid, 0.0), 3),
                "duration_s": round(raw_a.times[-1], 1) if raw_a is not None else None,
            })

    group_options = [
        {"label": f"Group {gid} / task-{task}", "value": f"{gid}|{task}"}
        for (gid, task) in all_aligned
    ]
    first_key = next(iter(all_aligned))
    first_val = f"{first_key[0]}|{first_key[1]}"

    parts = [f"Aligned {len(offset_rows)} subject(s) across {len(all_aligned)} group(s)."]
    if errors:
        parts.append(f"{len(errors)} group(s) failed.")
    status = dbc.Alert(
        " ".join(parts),
        color="success" if not errors else "warning",
        className="mb-0 py-2",
    )

    return status, _SHOW, offset_rows, group_options, first_val


# ── Group selector → redraw figures ──────────────────────────────────────────

@callback(
    Output("ha-trigger-timeline", "figure"),
    Output("ha-signal-overlay",   "figure"),
    Input("ha-group-select",      "value"),
    State("ha-bids-dir",          "value"),
    State("ha-group-csv",         "value"),
    prevent_initial_call=True,
)
def update_group_figures(group_val, bids_dir, group_csv):
    if not group_val or not bids_dir or not group_csv:
        return no_update, no_update

    key   = _cache_key(bids_dir, group_csv)
    cache = _ALIGNED_CACHE.get(key)
    if not cache:
        return no_update, no_update

    parts = group_val.split("|", 1)
    if len(parts) != 2:
        return no_update, no_update
    group_id, task = parts[0], parts[1]

    info = cache["groups"].get((group_id, task))
    if not info:
        return no_update, no_update

    from fnirs_pipe.qc.figures.hyper.hyper_figures import (
        _cond_colors, _extract_markers,
        build_signal_overlay, build_trigger_timeline,
    )

    aligned_raws = info["aligned_raws"]
    subject_ids  = info["subject_ids"]
    first_raw    = aligned_raws.get(subject_ids[0])
    markers_list = _extract_markers(first_raw) if first_raw else []
    all_descs    = list(dict.fromkeys(m["description"] for m in markers_list))
    cond_colors_ = _cond_colors(all_descs)

    def _safe(fn, *args):
        try:
            fig = fn(*args)
            return style_figure(fig.to_dict()) if fig is not None else no_update
        except Exception as exc:
            print(f"[DEBUG hyper_align] {fn.__name__} failed: {exc}")
            return no_update

    aligned_haemo = _to_haemo(aligned_raws)
    return (
        _safe(build_trigger_timeline, aligned_raws, subject_ids),
        _safe(build_signal_overlay, aligned_haemo, subject_ids,
              markers_list, cond_colors_),
    )


# ── Export ────────────────────────────────────────────────────────────────────

@callback(
    Output("ha-export-status", "children"),
    Input("ha-export-btn",     "n_clicks"),
    State("ha-bids-dir",       "value"),
    State("ha-deriv-dir",      "value"),
    State("ha-group-csv",      "value"),
    prevent_initial_call=True,
)
def export_snirfs(n_clicks, bids_dir, deriv_dir, group_csv):
    if not bids_dir or not deriv_dir or not group_csv:
        return dbc.Alert("Set all directories.", color="warning",
                         className="mb-0 py-2")

    key   = _cache_key(bids_dir, group_csv)
    cache = _ALIGNED_CACHE.get(key)
    if not cache or not cache.get("groups"):
        return dbc.Alert("Load and align first.", color="warning",
                         className="mb-0 py-2")

    from fnirs_pipe.pipeline.hyper import write_aligned_member
    from fnirs_pipe.utils.snirf_prep import deriv_nirs_dir, ensure_dataset_description

    _DERIV_NAME = "aligned"
    deriv_path  = Path(deriv_dir)
    written: list[str] = []
    errors:  list[str] = []

    for (group_id, task), info in cache["groups"].items():
        ensure_dataset_description(
            deriv_path / _DERIV_NAME, _DERIV_NAME, "fnirs-gui hyper-align"
        )
        for sid in info["subject_ids"]:
            path = info["paths"][sid]
            out_dir = deriv_nirs_dir(deriv_path, _DERIV_NAME, sid.removeprefix("sub-"),
                                     entity_of(path, "ses"))
            try:
                out_snirf = write_aligned_member(info["aligned_raws"][sid], path,
                                                 out_dir, group_id)
                written.append(out_snirf.name)
            except Exception as exc:
                errors.append(f"{sid}: {exc}")

    parts = [f"Written {len(written)} file(s)."]
    if errors:
        parts.append(f"{len(errors)} error(s): {'; '.join(errors[:3])}")
    return dbc.Alert(
        " ".join(parts),
        color="success" if not errors else "warning",
        className="mb-0 py-2",
    )


# Channel Decisions

_HA_CD_STATES = ["unrated", "good", "bad"]
_HA_CD_COLOR  = {"unrated": "secondary", "good": "success", "bad": "danger"}
_HA_CD_LABEL  = {"unrated": "—", "good": "good", "bad": "bad"}
_HA_RUN_KEY   = "_hyper"


def _ha_cd_btn(sid: str, pair: str, state: str) -> dbc.Button:
    return dbc.Button(
        _HA_CD_LABEL[state],
        id={"type": "ha-cd-btn", "index": f"{sid}|{pair}"},
        color=_HA_CD_COLOR[state],
        size="sm",
        n_clicks=0,
        style={"minWidth": "54px", "fontSize": "0.75rem", "padding": "1px 6px"},
    )


def _ha_decisions_path(deriv_dir: str, sid: str, task: str, source: Path) -> Path:
    # the session of the file this member was read from, which is the one the raw QC
    # page files its decisions under
    return channel_decisions_path(Path(deriv_dir), sid, task=task,
                                  session=entity_of(source, "ses"))


def _read_ha_decisions(deriv_dir: str, subject_ids: list, task: str, paths: dict) -> dict:
    decisions: dict = {}
    for sid in subject_ids:
        path = _ha_decisions_path(deriv_dir, sid, task, paths[sid])
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                decisions[sid] = data.get(_HA_RUN_KEY, {})
            except Exception:
                decisions[sid] = {}
        else:
            decisions[sid] = {}
    return decisions


def _write_ha_decisions(deriv_dir: str, task: str, decisions: dict, paths: dict) -> None:
    for sid, ch_map in decisions.items():
        path = _ha_decisions_path(deriv_dir, sid, task, paths[sid])
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                pass
        existing[_HA_RUN_KEY] = ch_map
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _compute_sci_from_cw(raws: dict, subject_ids: list, cardiac_l_freq, cardiac_h_freq) -> dict:
    import mne

    if cardiac_l_freq is None or cardiac_h_freq is None:
        return {sid: {} for sid in subject_ids}

    sci_by_sid: dict = {}
    for sid in subject_ids:
        raw = raws.get(sid)
        if raw is None:
            sci_by_sid[sid] = {}
            continue
        try:
            raw_od  = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
            sci_arr = mne.preprocessing.nirs.scalp_coupling_index(
                raw_od, l_freq=cardiac_l_freq, h_freq=cardiac_h_freq, verbose=False)
            pair_sci: dict = {}
            for i, ch in enumerate(raw.ch_names):
                pair = ch.rsplit(" ", 1)[0] if " " in ch else ch
                pair_sci.setdefault(pair, []).append(float(sci_arr[i]))
            sci_by_sid[sid] = {p: round(sum(v) / len(v), 3) for p, v in pair_sci.items()}
        except Exception:
            sci_by_sid[sid] = {}
    return sci_by_sid


def _build_ha_decisions_table(
    subject_ids: list,
    ch_pairs: list,
    sci_by_sid: dict,
    decisions: dict,
    sci_thresh: float = 0.8,
) -> html.Div:
    header_cells = [html.Th("Channel", style={"fontSize": "0.78rem"})]
    for sid in subject_ids:
        header_cells.append(
            html.Th(sid, colSpan=2,
                    style={"fontSize": "0.78rem", "textAlign": "center"})
        )
    sub_header = [html.Th("")]
    for _ in subject_ids:
        sub_header += [
            html.Th("SCI",      style={"fontSize": "0.72rem", "color": "#888"}),
            html.Th("Decision", style={"fontSize": "0.72rem", "color": "#888"}),
        ]

    rows = []
    for pair in ch_pairs:
        cells = [html.Td(pair, style={"fontSize": "0.78rem", "whiteSpace": "nowrap"})]
        for sid in subject_ids:
            sci_val = sci_by_sid.get(sid, {}).get(pair)
            sci_style: dict = {"fontSize": "0.75rem", "textAlign": "center"}
            if sci_val is not None and sci_val < sci_thresh:
                sci_style["color"] = "#dc3545"
            sci_text = f"{sci_val:.2f}" if sci_val is not None else "—"
            state_key = f"{pair} hbo"
            state = decisions.get(sid, {}).get(state_key, "unrated")
            if state not in _HA_CD_STATES:
                state = "unrated"
            cells += [
                html.Td(sci_text, style=sci_style),
                html.Td(_ha_cd_btn(sid, pair, state), style={"textAlign": "center"}),
            ]
        rows.append(html.Tr(cells))

    table = html.Table(
        [
            html.Thead([html.Tr(header_cells), html.Tr(sub_header)]),
            html.Tbody(rows),
        ],
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.78rem"},
    )
    return html.Div(table, style={"overflowX": "auto", "maxHeight": "420px",
                                  "overflowY": "auto"})


def _ha_ch_pairs_from_haemo(aligned_raws: dict, subject_ids: list) -> list:
    import mne
    aligned_haemo = _to_haemo(aligned_raws)
    ch_pairs: list = []
    ref = next((aligned_haemo[s] for s in subject_ids if s in aligned_haemo), None)
    if ref:
        for pick in mne.pick_types(ref.info, fnirs="hbo"):
            ch = ref.ch_names[pick]
            pair = ch.rsplit(" ", 1)[0] if " " in ch else ch
            if pair not in ch_pairs:
                ch_pairs.append(pair)
    return ch_pairs


@callback(
    Output("ha-decisions-table",  "children"),
    Output("ha-decisions-status", "children"),
    Input("ha-group-select",      "value"),
    State("ha-bids-dir",          "value"),
    State("ha-group-csv",         "value"),
    State("ha-deriv-dir",         "value"),
    State("ha-cardiac-l",         "value"),
    State("ha-cardiac-h",         "value"),
    prevent_initial_call=True,
)
def load_ha_decisions(group_val, bids_dir, group_csv, deriv_dir, cardiac_l, cardiac_h):
    if not group_val or not bids_dir or not group_csv:
        return no_update, no_update

    key   = _cache_key(bids_dir, group_csv)
    cache = _ALIGNED_CACHE.get(key)
    if not cache:
        return no_update, no_update

    parts = group_val.split("|", 1)
    if len(parts) != 2:
        return no_update, no_update
    group_id, task = parts[0], parts[1]

    info = cache["groups"].get((group_id, task))
    if not info:
        return no_update, no_update

    subject_ids  = info["subject_ids"]
    aligned_raws = info["aligned_raws"]

    ch_pairs   = _ha_ch_pairs_from_haemo(aligned_raws, subject_ids)
    sci_by_sid = _compute_sci_from_cw(aligned_raws, subject_ids, cardiac_l, cardiac_h)
    decisions  = (
        _read_ha_decisions(deriv_dir, subject_ids, task, info["paths"])
        if deriv_dir else {s: {} for s in subject_ids}
    )

    table  = _build_ha_decisions_table(subject_ids, ch_pairs, sci_by_sid, decisions)
    status = f"{len(ch_pairs)} channel pair(s) · {len(subject_ids)} subject(s)"
    return table, status


@callback(
    Output("ha-decisions-table",  "children", allow_duplicate=True),
    Output("ha-decisions-status", "children", allow_duplicate=True),
    Input({"type": "ha-cd-btn", "index": ALL}, "n_clicks"),
    State("ha-group-select", "value"),
    State("ha-bids-dir",     "value"),
    State("ha-group-csv",    "value"),
    State("ha-deriv-dir",    "value"),
    State("ha-cardiac-l",    "value"),
    State("ha-cardiac-h",    "value"),
    prevent_initial_call=True,
)
def click_ha_cd(n_clicks_list, group_val, bids_dir, group_csv, deriv_dir, cardiac_l, cardiac_h):
    if not ctx.triggered_id or not isinstance(ctx.triggered_id, dict):
        return no_update, no_update
    if not any(n for n in n_clicks_list if n):
        return no_update, no_update

    index = ctx.triggered_id["index"]
    iparts = index.split("|", 1)
    if len(iparts) != 2:
        return no_update, no_update
    sid, pair = iparts[0], iparts[1]

    if not group_val or not bids_dir or not group_csv or not deriv_dir:
        return no_update, no_update

    key   = _cache_key(bids_dir, group_csv)
    cache = _ALIGNED_CACHE.get(key)
    if not cache:
        return no_update, no_update

    gparts = group_val.split("|", 1)
    if len(gparts) != 2:
        return no_update, no_update
    group_id, task = gparts[0], gparts[1]

    info = cache["groups"].get((group_id, task))
    if not info:
        return no_update, no_update

    subject_ids  = info["subject_ids"]
    aligned_raws = info["aligned_raws"]

    decisions  = _read_ha_decisions(deriv_dir, subject_ids, task, info["paths"])
    state_key  = f"{pair} hbo"
    cur_state  = decisions.get(sid, {}).get(state_key, "unrated")
    if cur_state not in _HA_CD_STATES:
        cur_state = "unrated"
    next_state = _HA_CD_STATES[(_HA_CD_STATES.index(cur_state) + 1) % len(_HA_CD_STATES)]

    decisions.setdefault(sid, {})[state_key] = next_state
    _write_ha_decisions(deriv_dir, task, decisions, info["paths"])

    ch_pairs   = _ha_ch_pairs_from_haemo(aligned_raws, subject_ids)
    sci_by_sid = _compute_sci_from_cw(aligned_raws, subject_ids, cardiac_l, cardiac_h)
    table  = _build_ha_decisions_table(subject_ids, ch_pairs, sci_by_sid, decisions)
    status = f"Saved · {len(ch_pairs)} channel pair(s)"
    return table, status


# ── Static QC report for the dyads in the CSV ────────────────────────────────

@callback(
    Output("ha-report-status",  "children"),
    Output("ha-report-preview", "children"),
    Input("ha-report-btn",  "n_clicks"),
    State("ha-bids-dir",    "value"),
    State("ha-deriv-dir",   "value"),
    State("ha-group-csv",   "value"),
    State("ha-group-select", "value"),
    State("ha-dpf",         "value"),
    State("ha-cardiac-l",   "value"),
    State("ha-cardiac-h",   "value"),
    State("ha-sci-thresh",  "value"),
    prevent_initial_call=True,
)
def write_hyper_raw_report(n_clicks, bids_dir, deriv_dir, group_csv, group_val,
                           dpf, cardiac_l, cardiac_h, sci_thresh):
    # the viewer's group picker doubles as the report's scope; empty means every group
    group_id = None
    if group_val:
        group_id = group_val.split("|")[0] if "|" in str(group_val) else str(group_val)

    opts = {
        "bids_dir": bids_dir, "output_dir": deriv_dir,
        "pairs_csv": group_csv, "group_id": group_id,
        "dpf": dpf, "cardiac_l": cardiac_l, "cardiac_h": cardiac_h,
        "sci_threshold": sci_thresh,
    }

    problem = missing_raw_qc("hyper-raw", opts)
    if problem:
        return dbc.Alert(problem, color="warning", className="mb-0 py-2"), None

    argv = build_raw_qc_args("hyper-raw", opts)
    return run_and_report({"argv": argv, "command": "hyper-raw", "output_dir": deriv_dir})
