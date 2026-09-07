"""Callbacks for the data preparation page."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, Patch, State, callback, ctx, dcc, html, no_update

from fnirs_pipe.interface.theme import style_figure

# Server-side cache: cache_key -> _process_run result dict (large figures stay here)
_RESULT_CACHE: dict[str, dict] = {}
# In-memory only: cache_key -> raw_haemo MNE object (not pickled)
_HAEMO_CACHE: dict[str, object] = {}


def _parse_bids_entities(filename: str) -> dict:
    return dict(re.findall(r'([a-zA-Z]+)-([^_\.]+)', filename))


def _snirf_options(subject: str, bids_dir: str) -> list[dict]:
    nirs_dir = Path(bids_dir) / f"sub-{subject}" / "nirs"
    if not nirs_dir.is_dir():
        return []
    options = []
    for f in sorted(nirs_dir.glob("*_nirs.snirf")):
        e = _parse_bids_entities(f.name)
        parts = [f"sub-{e.get('sub', subject)}"]
        if e.get("ses"):  parts.append(f"ses-{e['ses']}")
        if e.get("task"): parts.append(f"task-{e['task']}")
        if e.get("run"):  parts.append(f"run-{e['run']}")
        options.append({"label": "_".join(parts), "value": str(f)})
    return options


# bump whenever a cached figure's builder changes, or the disk cache keeps serving the old one
_CACHE_VERSION = 3


def _make_cache_key(snirf_path: str, sci_thresh: float,
                    cardiac_l: float, cardiac_h: float, dpf: float) -> str:
    import hashlib
    payload = f"v{_CACHE_VERSION}|{snirf_path}|{sci_thresh}|{cardiac_l}|{cardiac_h}|{dpf}"
    return hashlib.md5(payload.encode()).hexdigest()[:16]


# ── Directory sync: page → shared Store ──────────────────────────────────────

@callback(
    Output("app-bids-dir",   "data", allow_duplicate=True),
    Output("app-output-dir", "data", allow_duplicate=True),
    Input("dp-bids-dir",   "value"),
    Input("dp-output-dir", "value"),
    prevent_initial_call=True,
)
def _dp_dirs_to_store(bids_dir, output_dir):
    return bids_dir, output_dir




# ── Subject detection ─────────────────────────────────────────────────────────

@callback(
    Output("dp-subjects-result",    "children"),
    Output("dp-subjects-container", "children"),
    Output("dp-manual-subject",     "value"),
    Input("dp-detect-btn",     "n_clicks"),
    Input("dp-manual-subject", "value"),
    Input("dp-clear-btn",      "n_clicks"),
    State("dp-bids-dir",       "value"),
)
def detect_subjects(detect_clicks, manual_input, clear_clicks, bids_dir):
    trigger = ctx.triggered_id

    if trigger == "dp-clear-btn":
        return "", "", ""

    if trigger == "dp-manual-subject" and manual_input:
        subject = manual_input.strip().removeprefix("sub-")
        if not subject:
            return "", "", manual_input
        radio = dbc.RadioItems(
            id="dp-subject-radio",
            options=[{"label": f"sub-{subject}", "value": subject}],
            value=subject,
            className="mt-1",
        )
        result = dbc.Alert(f"Manually entered: sub-{subject}",
                           color="info", className="mb-0 py-2")
        return result, radio, manual_input

    if trigger == "dp-detect-btn":
        if not bids_dir or not Path(bids_dir).is_dir():
            return dbc.Alert("Invalid BIDS directory.", color="warning"), "", ""
        subjects = sorted(
            d.name[4:] for d in Path(bids_dir).iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        if not subjects:
            return dbc.Alert("No subjects found.", color="warning"), "", ""
        radio = dbc.RadioItems(
            id="dp-subject-radio",
            options=[{"label": f"sub-{s}", "value": s} for s in subjects],
            value=subjects[0],
            className="mt-1",
        )
        result = dbc.Alert([
            f"Found {len(subjects)} subjects",
            html.Br(),
            html.Small(f"Subjects: {', '.join(subjects)}", className="text-muted"),
        ], color="success", className="mb-0 py-2")
        return result, radio, ""

    return "", "", ""


@callback(
    Output("dp-run-dropdown", "options"),
    Input("dp-subject-radio", "value"),
    State("dp-bids-dir",      "value"),
    prevent_initial_call=True,
)
def populate_runs(subject, bids_dir):
    if not subject or not bids_dir:
        return []
    return _snirf_options(subject, bids_dir)


# ── Load run → write everything into the shared store ─────────────────────────
# (No direct figure outputs — restore_from_store handles all graphs)

@callback(
    Output("dp-run-store",   "data"),
    Output("dp-load-status", "children"),
    Input("dp-run-dropdown", "value"),
    State("dp-sci-thresh",   "value"),
    State("dp-cardiac-l",    "value"),
    State("dp-cardiac-h",    "value"),
    State("dp-dpf",          "value"),
    State("app-output-dir",  "data"),
    prevent_initial_call=True,
)
def load_run(run_path, sci_thresh, cardiac_l, cardiac_h, dpf, output_dir):
    import pickle

    if not run_path:
        return no_update, no_update
    if not output_dir or not Path(output_dir).is_dir():
        return no_update, dbc.Alert("Set Output Directory first.", color="warning")
    if cardiac_l is None or cardiac_h is None or dpf is None:
        return no_update, dbc.Alert(
            "Set Cardiac Band (lo/hi) and DPF before loading a run.", color="warning")

    sci_threshold = float(sci_thresh if sci_thresh is not None else 0.8)
    cardiac_l, cardiac_h, dpf = float(cardiac_l), float(cardiac_h), float(dpf)
    snirf_path    = run_path
    cache_key     = _make_cache_key(snirf_path, sci_threshold, cardiac_l, cardiac_h, dpf)
    disk_path     = Path(output_dir) / ".fnirs_cache" / f"{cache_key}.pkl"

    if cache_key in _RESULT_CACHE:
        result = _RESULT_CACHE[cache_key]
        source = "memory"
    elif disk_path.exists():
        try:
            with open(disk_path, "rb") as f:
                result = pickle.load(f)
            _RESULT_CACHE[cache_key] = result
            source = "disk"
        except Exception as exc:
            print(f"[DEBUG load_run] disk cache load failed: {exc}, recomputing")
            result = None
            source = "compute"
    else:
        result = None
        source = "compute"

    if result is None:
        try:
            from fnirs_pipe.qc.prep_raw_report import _process_run
            run_label = Path(snirf_path).stem
            run_dir   = Path(output_dir) / ".fnirs_cache"
            result = _process_run(
                {"snirf_path": snirf_path, "label": run_label},
                sci_threshold,
                run_dir,
                cardiac_l,
                cardiac_h,
                [dpf],
            )
            raw_haemo = result.pop("_raw_haemo", None)
            if raw_haemo is not None:
                _HAEMO_CACHE[cache_key] = raw_haemo

            if raw_haemo is not None:
                from fnirs_pipe.qc.figures import build_channel_figure
                from fnirs_pipe.qc.prep_raw_report import (
                    _EPOCH_TMAX, _EPOCH_TMIN, _MAX_TS_PTS,
                )
                channels = result.setdefault("channels", {})
                markers = result.get("ts", {}).get("markers", [])
                for pair in result.get("channel_pairs", []):
                    try:
                        d_fig, p_fig, e_fig = build_channel_figure(
                            raw_haemo, markers, pair,
                            _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX,
                        )
                        channels[pair] = {
                            "detail_figure": d_fig.to_dict() if d_fig else None,
                            "psd_figure":    p_fig.to_dict() if p_fig else None,
                            "epoch_figure":  e_fig.to_dict() if e_fig else None,
                        }
                    except Exception as exc:
                        print(f"[DEBUG load_run] channel {pair!r} failed: {exc}")
                        channels[pair] = {}

            _RESULT_CACHE[cache_key] = result
            if disk_path:
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(disk_path, "wb") as f:
                        pickle.dump(result, f)
                except Exception as exc:
                    print(f"[DEBUG load_run] disk cache save failed: {exc}")
        except Exception as exc:
            return no_update, dbc.Alert(f"Failed to load: {exc}", color="danger")

    print(f"[DEBUG load_run] source={source}, pairs={len(result.get('channel_pairs', []))}")

    store = {
        "cache_key":  cache_key,
        "snirf_path": snirf_path,
    }
    status = dbc.Alert(f"Loaded ({source}): {Path(snirf_path).name}",
                       color="success", className="mb-0 py-2")
    return store, status


# ── Restore all graphs from store (fires on load AND on page re-mount) ────────

_SHOW = {}

@callback(
    Output("dp-ts-figure",              "figure"),
    Output("dp-layout-2d",              "figure"),
    Output("dp-layout-3d",              "figure"),
    Output("dp-sci-psp-figure",         "figure"),
    Output("dp-channel-psd",            "figure",  allow_duplicate=True),
    Output("dp-ch-summary-figure",      "figure"),
    Output("dp-marker-store",           "data",    allow_duplicate=True),
    Output("dp-channel-selector",       "options"),
    Output("dp-sqm-table",              "data"),
    Output("dp-channel-selector",       "value",   allow_duplicate=True),
    Output("dp-evoked-topo",            "figure"),
    Output("dp-trigger-timeline",       "figure"),
    Output("dp-carpet-gvtd",            "figure"),
    Output("dp-ts-figure-wrap",         "style"),
    Output("dp-layout-2d-wrap",         "style"),
    Output("dp-layout-3d-wrap",         "style"),
    Output("dp-sci-psp-figure-wrap",    "style"),
    Output("dp-ch-summary-figure-wrap", "style"),
    Output("dp-evoked-topo-wrap",       "style"),
    Output("dp-trigger-timeline-wrap",  "style"),
    Output("dp-carpet-gvtd-wrap",       "style"),
    Input("dp-run-store",          "data"),
    Input("dp-mount-tick",         "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def restore_from_store(store, _tick):
    if not store:
        return (no_update,) * 21
    cached = _RESULT_CACHE.get(store.get("cache_key"), {})
    if not cached:
        return (no_update,) * 21

    def _fig(nested, *keys):
        d = nested
        for k in keys:
            if not isinstance(d, dict):
                return no_update
            d = d.get(k)
        return style_figure(d) if d else no_update

    ts_fig       = _fig(cached, "ts",               "figure")
    layout_2d    = _fig(cached, "layout",           "layout_2d_figure")
    layout_3d    = _fig(cached, "layout",           "layout_3d_figure")
    sci_psp_out  = _fig(cached, "sci_psp",          "figure")
    psd_out      = _fig(cached, "psd",              "figure")
    ch_sum_out   = _fig(cached, "ch_summary",       "figure")
    evoked_topo  = _fig(cached, "evoked_topo",      "figure")
    trigger_tl   = _fig(cached, "trigger_timeline", "figure")

    markers = cached.get("ts", {}).get("markers", [])
    marker_rows = [
        {
            "onset":      m["onset"],
            "duration":   m["duration"],
            "trial_type": m.get("description", ""),
            "color":      m.get("color", "#888"),
        }
        for m in markers
    ] or no_update

    ch_pairs = cached.get("channel_pairs", sorted(cached.get("channels", {}).keys()))
    ch_options = [{"label": p, "value": p} for p in ch_pairs] or no_update

    sqm_scalars = cached.get("sqm", {}).get("scalars", {})
    sqm_rows = [
        {"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in sqm_scalars.items()
    ] or no_update

    carpet_src = cached.get("carpet_gvtd", {}).get("figure")
    carpet_src = style_figure(carpet_src) if carpet_src else no_update

    def _wrap(val):
        return _SHOW if val is not no_update else no_update

    return (
        ts_fig,
        layout_2d,
        layout_3d,
        sci_psp_out,
        psd_out,
        ch_sum_out,
        marker_rows,
        ch_options,
        sqm_rows,
        no_update,
        evoked_topo,
        trigger_tl,
        carpet_src,
        _wrap(ts_fig),
        _wrap(layout_2d),
        _wrap(layout_3d),
        _wrap(sci_psp_out),
        _wrap(ch_sum_out),
        _wrap(evoked_topo),
        _wrap(trigger_tl),
        _wrap(carpet_src),
    )


# ── Trace click → channel selector only (highlight handled by selector callback) ─

@callback(
    Output("dp-channel-selector", "value"),
    Input("dp-ts-figure",         "clickData"),
    State("dp-run-store",         "data"),
    prevent_initial_call=True,
)
def on_ts_click(click_data, store):
    if not click_data or not store:
        return no_update
    cached      = _RESULT_CACHE.get(store.get("cache_key"), {})
    ts_fig      = cached.get("ts", {}).get("figure", {})
    trace_names = [t.get("name", "") for t in ts_fig.get("data", [])]
    valid_pairs = set(cached.get("channel_pairs", []))

    curve_num = click_data["points"][0]["curveNumber"]
    if curve_num >= len(trace_names):
        return no_update
    trace_name = trace_names[curve_num]
    pair = trace_name.rsplit(" ", 1)[0] if " " in trace_name else trace_name
    print(f"[DEBUG on_ts_click] pair={pair!r}, valid={pair in valid_pairs}")
    return pair if pair in valid_pairs else no_update


# ── Selector → highlight Raw Signal traces ────────────────────────────────────

@callback(
    Output("dp-ts-figure", "figure", allow_duplicate=True),
    Input("dp-channel-selector", "value"),
    State("dp-run-store",        "data"),
    prevent_initial_call=True,
)
def highlight_ts_from_selector(channel_pair, store):
    if not store:
        return no_update
    cached      = _RESULT_CACHE.get(store.get("cache_key"), {})
    ts_fig      = cached.get("ts", {}).get("figure", {})
    trace_names = [t.get("name", "") for t in ts_fig.get("data", [])]
    if not trace_names:
        return no_update

    def _pair(name):
        return name.rsplit(" ", 1)[0] if " " in name else name

    patched = Patch()
    for i, name in enumerate(trace_names):
        patched["data"][i]["opacity"] = 1.0 if (
            not channel_pair or _pair(name) == channel_pair
        ) else 0.05
    return patched


# ── Selector → highlight 3D optode ───────────────────────────────────────────

@callback(
    Output("dp-layout-3d", "figure", allow_duplicate=True),
    Input("dp-channel-selector", "value"),
    State("dp-run-store",        "data"),
    prevent_initial_call=True,
)
def highlight_optode_3d(channel_pair, store):
    if not store:
        return no_update
    cached   = _RESULT_CACHE.get(store.get("cache_key"), {})
    fig_dict = cached.get("layout", {}).get("layout_3d_figure")
    if not isinstance(fig_dict, dict):
        return no_update
    try:
        traces  = fig_dict.get("data", [])
        ch_tr   = next((t for t in traces if t.get("name") == "Channels"), None)
        hl_idx  = next((i for i, t in enumerate(traces) if t.get("name") == "_hl"), None)
        if ch_tr is None or hl_idx is None:
            return no_update

        customdata = ch_tr.get("customdata", [])
        xs, ys, zs = ch_tr.get("x", []), ch_tr.get("y", []), ch_tr.get("z", [])
        target = f"{channel_pair} hbo" if channel_pair else None

        if target and target in customdata:
            idx = customdata.index(target)
            hx, hy, hz = [xs[idx]], [ys[idx]], [zs[idx]]
        else:
            hx, hy, hz = [], [], []

        patched = Patch()
        patched["data"][hl_idx]["x"] = hx
        patched["data"][hl_idx]["y"] = hy
        patched["data"][hl_idx]["z"] = hz
        return patched
    except Exception as exc:
        print(f"[DEBUG highlight_optode_3d] {exc}")
        return no_update


def _placeholder_fig(msg: str, height: int = 220) -> dict:
    return {
        "data": [],
        "layout": {
            "annotations": [{
                "text": msg, "x": 0.5, "y": 0.5,
                "xref": "paper", "yref": "paper",
                "showarrow": False, "font": {"size": 12, "color": "#aaa"},
            }],
            "height": height,
            "plot_bgcolor": "white", "paper_bgcolor": "white",
            "xaxis": {"visible": False}, "yaxis": {"visible": False},
            "margin": {"l": 30, "r": 30, "t": 30, "b": 30},
        },
    }


# ── Per-channel detail (reads pre-computed figures from store) ────────────────

@callback(
    Output("dp-channel-detail",      "figure"),
    Output("dp-channel-psd",         "figure",  allow_duplicate=True),
    Output("dp-channel-epoch",       "figure"),
    Output("dp-channel-detail-wrap", "style"),
    Output("dp-channel-psd-wrap",    "style"),
    Output("dp-channel-epoch-wrap",  "style"),
    Input("dp-channel-selector", "value"),
    State("dp-run-store",        "data"),
    prevent_initial_call=True,
)
def update_channel_detail(channel_pair, store):
    if not store or not channel_pair:
        return no_update, no_update, no_update, no_update, no_update, no_update

    cache_key = store.get("cache_key")
    cached    = _RESULT_CACHE.get(cache_key, {})
    channels  = cached.setdefault("channels", {})

    if channel_pair not in channels:
        # lazy compute
        raw_haemo = _HAEMO_CACHE.get(cache_key)
        if raw_haemo is None:
            # loaded from disk cache — recompute haemo from SNIRF
            try:
                import mne
                snirf_path = store.get("snirf_path", "")
                raw = mne.io.read_raw_snirf(snirf_path, preload=True, verbose=False)
                raw_od = mne.preprocessing.nirs.optical_density(raw, verbose=False)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
                _HAEMO_CACHE[cache_key] = raw_haemo
            except Exception as exc:
                print(f"[DEBUG update_channel_detail] recompute haemo failed: {exc}")
                return (
                    _placeholder_fig("Failed to load channel data", 160),
                    no_update, no_update, _SHOW, no_update, no_update,
                )
        try:
            from fnirs_pipe.qc.figures import build_channel_figure
            from fnirs_pipe.qc.prep_raw_report import _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX
            markers = cached.get("ts", {}).get("markers", [])
            detail_fig, psd_fig, epoch_fig = build_channel_figure(
                raw_haemo, markers, channel_pair, _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX,
            )
            channels[channel_pair] = {
                "detail_figure": detail_fig.to_dict() if detail_fig else None,
                "psd_figure":    psd_fig.to_dict()    if psd_fig    else None,
                "epoch_figure":  epoch_fig.to_dict()  if epoch_fig  else None,
            }
            print(f"[DEBUG update_channel_detail] computed {channel_pair!r}")
        except Exception as exc:
            print(f"[DEBUG update_channel_detail] build failed: {exc}")
            channels[channel_pair] = {}

    ch_data = channels.get(channel_pair, {})
    return (
        style_figure(ch_data.get("detail_figure") or _placeholder_fig("No channel data", 160)),
        style_figure(ch_data.get("psd_figure")    or _placeholder_fig("No PSD available", 220)),
        style_figure(ch_data.get("epoch_figure")
                     or _placeholder_fig("No markers — epoch preview not available", 220)),
        _SHOW,
        _SHOW,
        _SHOW,
    )


# ── Marker rows: render ───────────────────────────────────────────────────────

def _make_marker_row(i: int, row: dict):
    color = row.get("color", "#888")
    return dbc.Row([
        dbc.Col(
            dcc.Checklist(
                id={"type": "mk-sel", "index": i},
                options=[{"label": "", "value": "on"}],
                value=[],
                inputStyle={"cursor": "pointer"},
            ),
            width="auto", className="d-flex align-items-center ps-1",
        ),
        dbc.Col(
            html.Span("●", style={"color": color, "fontSize": "10px"}),
            width="auto", className="d-flex align-items-center pe-0",
        ),
        dbc.Col(
            dbc.Input(
                id={"type": "mk-onset", "index": i},
                type="number", value=row.get("onset", 0.0),
                debounce=True, size="sm", style={"width": "80px"},
            ),
            width=4,
        ),
        dbc.Col(
            dbc.Input(
                id={"type": "mk-dur", "index": i},
                type="number", value=row.get("duration", 1.0),
                debounce=True, size="sm", style={"width": "65px"},
            ),
            width=3,
        ),
        dbc.Col(
            dbc.Input(
                id={"type": "mk-type", "index": i},
                type="text", value=row.get("trial_type", ""),
                debounce=True, size="sm",
            ),
        ),
        dbc.Col(
            dbc.Button(
                "×", id={"type": "mk-del", "index": i},
                color="outline-danger", size="sm",
                style={"padding": "0 6px", "lineHeight": "1.4"},
            ),
            width="auto",
        ),
    ], className="g-1 mb-1 align-items-center flex-nowrap px-1")


@callback(
    Output("dp-marker-rows-container", "children"),
    Input("dp-marker-store", "data"),
)
def render_marker_rows(rows):
    rows = rows or []
    if not rows:
        return html.Div("No markers", className="text-muted small px-2 py-1")
    return [_make_marker_row(i, r) for i, r in enumerate(rows)]


@callback(
    Output("dp-marker-store", "data", allow_duplicate=True),
    Input({"type": "mk-del", "index": ALL}, "n_clicks"),
    State("dp-marker-store", "data"),
    prevent_initial_call=True,
)
def delete_marker_row(del_clicks, rows):
    if not any(c for c in (del_clicks or [])):
        return no_update
    trigger = ctx.triggered_id
    if not isinstance(trigger, dict):
        return no_update
    idx = trigger["index"]
    rows = rows or []
    return [r for i, r in enumerate(rows) if i != idx]


@callback(
    Output("dp-marker-store", "data", allow_duplicate=True),
    Input({"type": "mk-onset", "index": ALL}, "value"),
    Input({"type": "mk-dur",   "index": ALL}, "value"),
    Input({"type": "mk-type",  "index": ALL}, "value"),
    State("dp-marker-store", "data"),
    prevent_initial_call=True,
)
def sync_marker_fields(onsets, durs, types, rows):
    current = rows or []
    n = len(onsets)
    if n == 0:
        return no_update
    updated = []
    changed = False
    for i in range(n):
        base = current[i] if i < len(current) else {}
        new_onset    = float(onsets[i]) if onsets[i] is not None else base.get("onset", 0.0)
        new_dur      = float(durs[i])   if durs[i]   is not None else base.get("duration", 1.0)
        new_type     = types[i]         if types[i]  is not None else base.get("trial_type", "")
        if (new_onset != base.get("onset") or new_dur != base.get("duration")
                or new_type != base.get("trial_type", "")):
            changed = True
        updated.append({
            "color":      base.get("color", "#888"),
            "onset":      new_onset,
            "duration":   new_dur,
            "trial_type": new_type,
        })
    if not changed and len(updated) == len(current):
        return no_update
    return updated


# ── Marker table editing ──────────────────────────────────────────────────────

@callback(
    Output("dp-marker-store", "data", allow_duplicate=True),
    Input("dp-add-marker-btn", "n_clicks"),
    State("dp-marker-store",   "data"),
    prevent_initial_call=True,
)
def add_marker_row(n_clicks, rows):
    rows = list(rows or [])
    rows.append({"onset": 0.0, "duration": 1.0, "trial_type": "", "color": "#888"})
    return rows


@callback(
    Output("dp-save-status", "children"),
    Input("dp-save-markers-btn", "n_clicks"),
    State("dp-marker-store",     "data"),
    State("dp-run-store",        "data"),
    State("app-output-dir",      "data"),
    prevent_initial_call=True,
)
def save_markers(n_clicks, rows, store, output_dir):
    if not store or not store.get("snirf_path"):
        return dbc.Alert("No run loaded.", color="warning", className="mb-0 py-2")
    if not output_dir:
        return dbc.Alert("Output directory not set.", color="warning", className="mb-0 py-2")

    import pandas as pd
    from fnirs_pipe.pipeline.edit_markers import apply_markers_from_df

    snirf_path = Path(store["snirf_path"])
    entities = _parse_bids_entities(snirf_path.name)
    sub = entities.get("sub", "")
    ses = entities.get("ses")

    rows = rows or []
    df = pd.DataFrame(rows, columns=["onset", "duration", "trial_type"])
    df["onset"]    = pd.to_numeric(df["onset"],    errors="coerce").fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    try:
        out_snirf = apply_markers_from_df(
            snirf_path, Path(output_dir), sub, ses, df
        )
        return dbc.Alert(f"Saved to derivatives: {out_snirf.name}",
                         color="success", className="mb-0 py-2")
    except Exception as exc:
        return dbc.Alert(f"Save failed: {exc}", color="danger", className="mb-0 py-2")


# ── Enable/disable − Sel / + Sel (uses row checkboxes) ───────────────────────

@callback(
    Output("dp-offset-sel-minus", "disabled"),
    Output("dp-offset-sel-plus",  "disabled"),
    Input({"type": "mk-sel", "index": ALL}, "value"),
)
def toggle_sel_buttons(sel_values):
    any_selected = any(bool(v) for v in (sel_values or []))
    return not any_selected, not any_selected


# ── Marker offset (all / selected) ────────────────────────────────────────────

@callback(
    Output("dp-marker-store", "data", allow_duplicate=True),
    Input("dp-offset-all-minus", "n_clicks"),
    Input("dp-offset-all-plus",  "n_clicks"),
    State("dp-step-input",       "value"),
    State("dp-marker-store",     "data"),
    prevent_initial_call=True,
)
def offset_all_markers(minus_clicks, plus_clicks, step, rows):
    if not rows:
        return no_update
    delta = -(step or 1.0) if ctx.triggered_id == "dp-offset-all-minus" else (step or 1.0)
    rows = [r.copy() for r in rows]
    for row in rows:
        row["onset"] = max(0.0, round(float(row.get("onset", 0)) + delta, 4))
    return rows


@callback(
    Output("dp-marker-store", "data", allow_duplicate=True),
    Input("dp-offset-sel-minus",            "n_clicks"),
    Input("dp-offset-sel-plus",             "n_clicks"),
    State("dp-step-input",                  "value"),
    State("dp-marker-store",                "data"),
    State({"type": "mk-sel", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def offset_selected_markers(minus_clicks, plus_clicks, step, rows, sel_values):
    if not rows:
        return no_update
    delta = -(step or 1.0) if ctx.triggered_id == "dp-offset-sel-minus" else (step or 1.0)
    rows = [r.copy() for r in rows]
    for i, sel in enumerate(sel_values or []):
        if sel and i < len(rows):
            rows[i]["onset"] = max(0.0, round(float(rows[i].get("onset", 0)) + delta, 4))
    return rows


# ── Batch rename ──────────────────────────────────────────────────────────────

@callback(
    Output("dp-marker-store",  "data", allow_duplicate=True),
    Output("dp-rename-from",   "value"),
    Output("dp-rename-to",     "value"),
    Input("dp-batch-rename-btn", "n_clicks"),
    State("dp-rename-from",      "value"),
    State("dp-rename-to",        "value"),
    State("dp-marker-store",     "data"),
    prevent_initial_call=True,
)
def batch_rename_markers(n_clicks, from_val, to_val, rows):
    if not rows or not from_val:
        return no_update, no_update, no_update
    to_val = to_val or ""
    rows = [r.copy() for r in rows]
    for row in rows:
        if row.get("trial_type") == from_val:
            row["trial_type"] = to_val
    return rows, "", ""


# ── Export TSV ────────────────────────────────────────────────────────────────

@callback(
    Output("dp-tsv-download",  "data"),
    Input("dp-export-tsv-btn", "n_clicks"),
    State("dp-marker-store",   "data"),
    prevent_initial_call=True,
)
def export_tsv(n_clicks, rows):
    if not rows:
        return no_update
    buf = io.StringIO()
    buf.write("onset\tduration\tdescription\n")
    for r in rows:
        buf.write(
            f"{float(r.get('onset', 0)):.4f}\t"
            f"{float(r.get('duration', 1)):.4f}\t"
            f"{r.get('trial_type', '')}\n"
        )
    return dcc.send_string(buf.getvalue(), "events.tsv")


# ── Signal Topo: highlight selected channel box ───────────────────────────────

@callback(
    Output("dp-evoked-topo", "figure", allow_duplicate=True),
    Input("dp-channel-selector", "value"),
    State("dp-run-store", "data"),
    prevent_initial_call=True,
)
def highlight_topo_channel(channel_pair, store):
    if not store:
        return no_update
    cached = _RESULT_CACHE.get(store.get("cache_key"), {})
    # by shape name, not by index: a positional lookup highlighted the wrong cell
    shapes = cached.get("evoked_topo", {}).get("figure", {}).get("layout", {}).get("shapes", [])
    cells = [(i, sh.get("name")) for i, sh in enumerate(shapes) if sh.get("name")]
    if not cells:
        return no_update
    patched = Patch()
    for i, pair in cells:
        if pair == channel_pair:
            patched["layout"]["shapes"][i]["line"]["color"] = "#f39c12"
            patched["layout"]["shapes"][i]["line"]["width"] = 2.5
            patched["layout"]["shapes"][i]["fillcolor"] = "rgba(243,156,18,0.12)"
        else:
            patched["layout"]["shapes"][i]["line"]["color"] = "#ccc"
            patched["layout"]["shapes"][i]["line"]["width"] = 0.8
            patched["layout"]["shapes"][i]["fillcolor"] = "rgba(255,255,255,0.80)"
    return patched


# ── Channel Detail title (updates when channel is selected) ──────────────────

@callback(
    Output("dp-detail-title", "children"),
    Input("dp-channel-selector", "value"),
)
def update_detail_title(channel_pair):
    if not channel_pair:
        return " · click a channel trace to view HbO / HbR"
    return f" · {channel_pair}"


# ── 2D optode highlight when channel is selected (mirrors HTML selectChannel) ─

@callback(
    Output("dp-layout-2d", "figure", allow_duplicate=True),
    Input("dp-channel-selector", "value"),
    State("dp-run-store", "data"),
    prevent_initial_call=True,
)
def highlight_optode_2d(channel_pair, store):
    if not store:
        return no_update
    cached   = _RESULT_CACHE.get(store.get("cache_key"), {})
    fig_dict = cached.get("layout", {}).get("layout_2d_figure")
    if not isinstance(fig_dict, dict) or len(fig_dict.get("data", [])) < 2:
        return no_update
    try:
        customdata = fig_dict["data"][1].get("customdata", [])
        if not customdata:
            return no_update

        def _pair(ch):
            return ch.rsplit(" ", 1)[0] if " " in ch else ch

        sel       = channel_pair or ""
        sizes     = [15 if _pair(ch) == sel else 10  for ch in customdata]
        opacities = [1.0 if _pair(ch) == sel else 0.55 for ch in customdata]
        lw        = [2.0 if _pair(ch) == sel else 0.8  for ch in customdata]

        patched = Patch()
        patched["data"][1]["marker"]["size"]          = sizes
        patched["data"][1]["marker"]["opacity"]       = opacities
        patched["data"][1]["marker"]["line"]["width"] = lw
        return patched
    except Exception as exc:
        print(f"[DEBUG highlight_optode_2d] {exc}")
        return no_update


# ── Optode layout click → channel selector ────────────────────────────────────

@callback(
    Output("dp-channel-selector", "value", allow_duplicate=True),
    Input("dp-layout-2d",  "clickData"),
    State("dp-run-store",  "data"),
    prevent_initial_call=True,
)
def on_layout_2d_click(click_data, store):
    if not click_data or not store:
        return no_update
    cached      = _RESULT_CACHE.get(store.get("cache_key"), {})
    valid_pairs = set(cached.get("channel_pairs", []))
    points      = click_data.get("points", [])
    if not points:
        return no_update
    ch_name = points[0].get("customdata", "")
    pair    = ch_name.rsplit(" ", 1)[0] if " " in ch_name else ch_name
    print(f"[DEBUG on_layout_2d_click] ch_name={ch_name!r}, pair={pair!r}, valid={pair in valid_pairs}")
    return pair if pair in valid_pairs else no_update


# ── Evoked topo click → channel selector ─────────────────────────────────────

@callback(
    Output("dp-channel-selector", "value", allow_duplicate=True),
    Input("dp-evoked-topo", "clickData"),
    State("dp-run-store",   "data"),
    prevent_initial_call=True,
)
def on_topo_click(click_data, store):
    if not click_data or not store:
        return no_update
    cached = _RESULT_CACHE.get(store.get("cache_key"), {})
    valid_pairs = set(cached.get("channel_pairs", []))
    points = click_data.get("points", [])
    if not points:
        return no_update

    # Use curveNumber to index into the figure's stored trace list and read
    # its customdata — avoids the "nearest point across subplots" misfire.
    curve_num = points[0].get("curveNumber", 0)
    fig_dict = cached.get("evoked_topo", {}).get("figure", {})
    traces = fig_dict.get("data", [])
    if curve_num < len(traces):
        cd = traces[curve_num].get("customdata", [])
        pair = cd[0] if isinstance(cd, list) and cd else str(cd) if cd else ""
        if pair in valid_pairs:
            print(f"[DEBUG on_topo_click] curve={curve_num}, pair={pair!r}")
            return pair

    # Fallback: customdata on the point itself
    pair = str(points[0].get("customdata", ""))
    return pair if pair in valid_pairs else no_update


# ── Crop: show/hide mode panels ───────────────────────────────────────────────

@callback(
    Output("dp-crop-single-panel", "style"),
    Output("dp-crop-multi-panel",  "style"),
    Output("dp-crop-seg-wrap",     "style"),
    Input("dp-crop-mode", "value"),
)
def toggle_crop_mode(mode):
    show, hide = {}, {"display": "none"}
    if mode == "single":
        return show, hide, hide
    return hide, show, show


# ── Crop: add segment row ─────────────────────────────────────────────────────

@callback(
    Output("dp-crop-seg-table", "data", allow_duplicate=True),
    Input("dp-crop-add-seg-btn", "n_clicks"),
    State("dp-crop-seg-table",   "data"),
    prevent_initial_call=True,
)
def add_crop_segment(n_clicks, rows):
    rows = rows or []
    rows.append({"onset": 0.0, "duration": 30.0})
    return rows


# ── Crop: fill tmin/tmax from zoom range ──────────────────────────────────────

@callback(
    Output("dp-crop-tmin", "value"),
    Output("dp-crop-tmax", "value"),
    Input("dp-crop-use-zoom",    "n_clicks"),
    State("dp-trigger-timeline", "relayoutData"),
    prevent_initial_call=True,
)
def use_zoom_range(n_clicks, relayout):
    if not relayout:
        return no_update, no_update
    tmin = relayout.get("xaxis.range[0]")
    tmax = relayout.get("xaxis.range[1]")
    return tmin, tmax


# ── Crop: highlight crop region on timeline ───────────────────────────────────

@callback(
    Output("dp-trigger-timeline", "figure", allow_duplicate=True),
    Input("dp-crop-tmin",         "value"),
    Input("dp-crop-tmax",         "value"),
    Input("dp-crop-seg-table",    "data"),
    Input("dp-crop-mode",         "value"),
    prevent_initial_call=True,
)
def update_crop_highlight(tmin, tmax, segments, mode):
    patched = Patch()
    shapes = []
    fill = "rgba(52, 152, 219, 0.15)"
    if mode == "single":
        if tmin is not None or tmax is not None:
            shapes.append({
                "type": "rect", "xref": "x", "yref": "paper",
                "x0": float(tmin) if tmin is not None else 0,
                "x1": float(tmax) if tmax is not None else 1e9,
                "y0": 0, "y1": 1,
                "fillcolor": fill, "line": {"width": 0},
            })
    elif mode == "multi" and segments:
        for row in segments:
            onset    = row.get("onset")
            duration = row.get("duration")
            if onset is not None and duration is not None:
                shapes.append({
                    "type": "rect", "xref": "x", "yref": "paper",
                    "x0": float(onset), "x1": float(onset) + float(duration),
                    "y0": 0, "y1": 1,
                    "fillcolor": fill, "line": {"width": 0},
                })
    patched["layout"]["shapes"] = shapes
    return patched


# ── Crop: apply ───────────────────────────────────────────────────────────────

@callback(
    Output("dp-crop-status",    "children"),
    Input("dp-crop-apply-btn",  "n_clicks"),
    State("dp-crop-mode",       "value"),
    State("dp-crop-tmin",       "value"),
    State("dp-crop-tmax",       "value"),
    State("dp-crop-seg-table",  "data"),
    State("dp-crop-combine",    "value"),
    State("dp-run-store",       "data"),
    State("app-output-dir",     "data"),
    prevent_initial_call=True,
)
def apply_crop(n_clicks, mode, tmin, tmax, seg_rows, combine_val, store, output_dir):
    if not store or not store.get("snirf_path"):
        return dbc.Alert("No run loaded.", color="warning", className="mb-0 py-2")
    if not output_dir:
        return dbc.Alert("Output directory not set.", color="warning", className="mb-0 py-2")

    import pandas as pd
    from fnirs_pipe.pipeline.crop import crop_snirf_from_path

    snirf_path = Path(store["snirf_path"])
    entities   = _parse_bids_entities(snirf_path.name)
    sub        = entities.get("sub", "")
    ses        = entities.get("ses")
    combine    = bool(combine_val)

    try:
        if mode == "single":
            if tmin is None and tmax is None:
                return dbc.Alert("Set tmin or tmax.", color="warning", className="mb-0 py-2")
            out_paths = crop_snirf_from_path(
                snirf_path, Path(output_dir), sub, ses,
                tmin=float(tmin) if tmin is not None else None,
                tmax=float(tmax) if tmax is not None else None,
            )
        else:
            if not seg_rows:
                return dbc.Alert("Add at least one segment.", color="warning", className="mb-0 py-2")
            df = pd.DataFrame(seg_rows, columns=["onset", "duration"])
            df["onset"]    = pd.to_numeric(df["onset"],    errors="coerce").fillna(0.0)
            df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
            out_paths = crop_snirf_from_path(
                snirf_path, Path(output_dir), sub, ses,
                segments_df=df, combine=combine,
            )
        names = ", ".join(p.name for p in out_paths)
        return dbc.Alert(f"Written: {names}", color="success", className="mb-0 py-2")
    except Exception as exc:
        return dbc.Alert(f"Crop failed: {exc}", color="danger", className="mb-0 py-2")


# ── Channel Decisions ─────────────────────────────────────────────────────────

_CD_STATES = ["unrated", "good", "bad"]
_CD_COLOR  = {"unrated": "outline-secondary", "good": "success", "bad": "danger"}
_CD_LABEL  = {"unrated": "—", "good": "good", "bad": "bad"}


def _decisions_file_info(snirf_path: str, output_dir: str) -> tuple[Path, str]:
    """Return (decisions_json_path, run_label) for the given snirf file."""
    entities  = _parse_bids_entities(Path(snirf_path).stem)
    sub       = entities.get("sub", "unknown")
    ses       = entities.get("ses")
    task      = entities.get("task")
    run       = entities.get("run")
    base_parts = [f"sub-{sub}"]
    if ses:  base_parts.append(f"ses-{ses}")
    if task: base_parts.append(f"task-{task}")
    html_stem  = "_".join(base_parts) + "_raw"
    run_parts  = list(base_parts)
    if run:  run_parts.append(f"run-{run}")
    run_label  = "_".join(run_parts)
    return Path(output_dir) / f"{html_stem}_channel_decisions.json", run_label


def _read_decisions(dec_path: Path, run_label: str) -> dict:
    if not dec_path.exists():
        return {}
    try:
        data = json.loads(dec_path.read_text(encoding="utf-8"))
        return data.get(run_label, {})
    except Exception:
        return {}


def _write_decisions(dec_path: Path, run_label: str, run_decisions: dict) -> None:
    existing: dict = {}
    if dec_path.exists():
        try:
            existing = json.loads(dec_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing[run_label] = run_decisions
    dec_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _cd_btn(pair: str, state: str) -> dbc.Button:
    s = state if state in _CD_STATES else "unrated"
    return dbc.Button(
        _CD_LABEL[s],
        id={"type": "dp-cd-btn", "index": pair},
        color=_CD_COLOR[s],
        size="sm",
        n_clicks=0,
        style={"minWidth": "3.5rem", "fontSize": "0.78rem"},
    )


def _build_decisions_table(pairs: list[str], sci_map: dict,
                            run_decisions: dict, sci_thresh: float) -> html.Table:
    _th = lambda t: html.Th(t, style={"padding": "4px 8px", "fontWeight": "600",
                                       "borderBottom": "2px solid #dde3ea",
                                       "background": "#f8f9fa", "fontSize": "0.8rem"})
    rows = []
    for pair in pairs:
        hbo   = f"{pair} hbo"
        sci   = sci_map.get(hbo, sci_map.get(f"{pair} hbr", None))
        below = sci is not None and sci < sci_thresh
        state = run_decisions.get(hbo, "unrated")
        rows.append(html.Tr([
            html.Td(pair, style={"fontWeight": "500", "padding": "3px 8px",
                                  "background": "#fff8f8" if below else "",
                                  "borderBottom": "1px solid #f0f0f0"}),
            html.Td(f"{sci:.3f}" if sci is not None else "—",
                    style={"color": "#c0392b" if below else "#6c757d",
                           "fontVariantNumeric": "tabular-nums",
                           "padding": "3px 8px",
                           "borderBottom": "1px solid #f0f0f0"}),
            html.Td(_cd_btn(pair, state),
                    style={"padding": "2px 6px", "borderBottom": "1px solid #f0f0f0"}),
        ]))
    return html.Table(
        [html.Thead(html.Tr([_th("Channel"), _th("SCI"), _th("Decision")])),
         html.Tbody(rows)],
        style={"borderCollapse": "collapse", "width": "100%", "fontSize": "0.82rem"},
    )


@callback(
    Output("dp-decisions-store", "data"),
    Output("dp-decisions-table", "children"),
    Input("dp-run-store", "data"),
    State("dp-sci-thresh",    "value"),
    State("app-output-dir",   "data"),
    prevent_initial_call=True,
)
def load_decisions(store, sci_thresh, output_dir):
    if not store:
        return no_update, no_update
    cached    = _RESULT_CACHE.get(store.get("cache_key"), {})
    pairs     = cached.get("channel_pairs", [])
    sci_map   = (cached.get("sqm") or {}).get("per_channel", {}).get("sci_per_channel", {})
    sci_thresh = float(sci_thresh or 0.8)
    if not pairs:
        return {}, html.Small("No channels found.", className="text-muted")

    run_decisions = {}
    state: dict = {"snirf_path": store["snirf_path"], "sci_threshold": sci_thresh}
    if output_dir:
        dec_path, run_label = _decisions_file_info(store["snirf_path"], output_dir)
        run_decisions       = _read_decisions(dec_path, run_label)
        state.update({"output_dir": output_dir, "run_label": run_label,
                      "dec_path": str(dec_path), "run_decisions": run_decisions})

    table = html.Div(
        _build_decisions_table(pairs, sci_map, run_decisions, sci_thresh),
    )
    return state, table


@callback(
    Output("dp-decisions-store", "data",    allow_duplicate=True),
    Output("dp-decisions-table", "children",allow_duplicate=True),
    Output("dp-decisions-status","children"),
    Input({"type": "dp-cd-btn", "index": ALL}, "n_clicks"),
    State("dp-decisions-store", "data"),
    State("dp-run-store",       "data"),
    State("dp-sci-thresh",      "value"),
    prevent_initial_call=True,
)
def click_cd(_, dec_state, run_store, sci_thresh):
    if not dec_state or not ctx.triggered_id:
        return no_update, no_update, no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "dp-cd-btn":
        return no_update, no_update, no_update

    pair          = triggered["index"]
    run_decisions = dict(dec_state.get("run_decisions", {}))
    hbo, hbr      = f"{pair} hbo", f"{pair} hbr"
    cur           = run_decisions.get(hbo, "unrated")
    new           = _CD_STATES[(_CD_STATES.index(cur) + 1) % len(_CD_STATES)]
    run_decisions[hbo] = new
    run_decisions[hbr] = new

    dec_state = {**dec_state, "run_decisions": run_decisions}

    if dec_state.get("dec_path"):
        try:
            _write_decisions(Path(dec_state["dec_path"]),
                             dec_state["run_label"], run_decisions)
            msg = "Saved."
        except Exception as exc:
            msg = f"Save failed: {exc}"
    else:
        msg = "No output dir set — decisions not saved."

    cached     = _RESULT_CACHE.get((run_store or {}).get("cache_key"), {})
    pairs      = cached.get("channel_pairs", [])
    sci_map    = (cached.get("sqm") or {}).get("per_channel", {}).get("sci_per_channel", {})
    sci_thresh = float(sci_thresh or 0.8)
    table = html.Div(
        _build_decisions_table(pairs, sci_map, run_decisions, sci_thresh),
    )
    return dec_state, table, msg
