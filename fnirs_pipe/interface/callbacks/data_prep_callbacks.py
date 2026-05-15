"""Callbacks for the data preparation page."""

from __future__ import annotations

import io
import re
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, Patch, State, callback, ctx, dcc, html, no_update

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


def _make_cache_key(snirf_path: str, sci_thresh: float) -> str:
    import hashlib
    return hashlib.md5(f"{snirf_path}|{sci_thresh}".encode()).hexdigest()[:16]


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


@callback(
    Output("app-cache-dir", "data", allow_duplicate=True),
    Input("dp-cache-dir", "value"),
    prevent_initial_call=True,
)
def _cache_dir_to_store(cache_dir):
    return cache_dir


@callback(
    Output("dp-cache-dir", "value"),
    Input("app-cache-dir", "data"),
    prevent_initial_call="initial_duplicate",
)
def _restore_cache_dir(cache_dir):
    return cache_dir or no_update


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
    Input("dp-load-btn",  "n_clicks"),
    State("dp-run-dropdown", "value"),
    State("dp-sci-thresh",   "value"),
    State("app-cache-dir",   "data"),
    prevent_initial_call=True,
)
def load_run(n_clicks, run_path, sci_thresh, cache_dir):
    import pickle

    if not run_path:
        return no_update, dbc.Alert("Select a run first.", color="warning")

    sci_threshold = float(sci_thresh if sci_thresh is not None else 0.8)
    snirf_path    = run_path
    cache_key     = _make_cache_key(snirf_path, sci_threshold)

    disk_path = None
    if cache_dir and Path(cache_dir).is_dir():
        disk_path = Path(cache_dir) / f"{cache_key}.pkl"

    if cache_key in _RESULT_CACHE:
        result = _RESULT_CACHE[cache_key]
        source = "memory"
    elif disk_path and disk_path.exists():
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
            result = _process_run(
                {"snirf_path": snirf_path},
                sci_threshold,
            )
            # extract raw_haemo before caching (not picklable)
            raw_haemo = result.pop("_raw_haemo", None)
            if raw_haemo is not None:
                _HAEMO_CACHE[cache_key] = raw_haemo
            _RESULT_CACHE[cache_key] = result
            if disk_path:
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(disk_path, "wb") as f:
                        pickle.dump(result, f)
                    print(f"[DEBUG load_run] saved to disk: {disk_path}")
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

@callback(
    Output("dp-ts-figure",         "figure"),
    Output("dp-layout-2d",         "figure"),
    Output("dp-layout-3d",         "figure"),
    Output("dp-sci-psp-figure",    "figure"),
    Output("dp-channel-psd",       "figure",  allow_duplicate=True),
    Output("dp-ch-summary-figure", "figure"),
    Output("dp-marker-table",      "data",    allow_duplicate=True),
    Output("dp-channel-selector",  "options"),
    Output("dp-iqm-table",         "data"),
    Output("dp-channel-selector",  "value",   allow_duplicate=True),
    Output("dp-evoked-topo",       "figure"),
    Output("dp-trigger-timeline",  "figure"),
    Input("dp-run-store",          "data"),
    Input("dp-mount-tick",         "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def restore_from_store(store, _tick):
    if not store:
        return (no_update,) * 12
    cached = _RESULT_CACHE.get(store.get("cache_key"), {})
    if not cached:
        return (no_update,) * 12

    def _fig(nested, *keys):
        d = nested
        for k in keys:
            if not isinstance(d, dict):
                return no_update
            d = d.get(k)
        return d or no_update

    sci_psp_out = _fig(cached, "sci_psp",    "figure")
    ch_sum_out  = _fig(cached, "ch_summary", "figure")
    print(f"[DEBUG restore] sci_psp populated: {sci_psp_out is not no_update}")
    print(f"[DEBUG restore] ch_summary populated: {ch_sum_out is not no_update}")

    markers = cached.get("ts", {}).get("markers", [])
    marker_rows = [
        {"onset": m["onset"], "duration": m["duration"], "trial_type": m.get("description", "")}
        for m in markers
    ] or no_update

    ch_pairs = cached.get("channel_pairs", sorted(cached.get("channels", {}).keys()))
    ch_options = [{"label": p, "value": p} for p in ch_pairs] or no_update
    first_pair = ch_pairs[0] if ch_pairs else no_update

    iqm_scalars = cached.get("iqm", {}).get("scalars", {})
    iqm_rows = [
        {"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
        for k, v in iqm_scalars.items()
    ] or no_update

    return (
        _fig(cached, "ts",               "figure"),
        _fig(cached, "layout",           "layout_2d_figure"),
        _fig(cached, "layout",           "layout_3d_figure"),
        sci_psp_out,
        _fig(cached, "psd",              "figure"),
        ch_sum_out,
        marker_rows,
        ch_options,
        iqm_rows,
        first_pair,
        _fig(cached, "evoked_topo",      "figure"),
        _fig(cached, "trigger_timeline", "figure"),
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
    Output("dp-channel-detail", "figure"),
    Output("dp-channel-psd",    "figure",  allow_duplicate=True),
    Output("dp-channel-epoch",  "figure"),
    Input("dp-channel-selector", "value"),
    State("dp-run-store",        "data"),
    prevent_initial_call=True,
)
def update_channel_detail(channel_pair, store):
    if not store or not channel_pair:
        return no_update, no_update, no_update

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
                    no_update, no_update,
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
        ch_data.get("detail_figure") or _placeholder_fig("No channel data", 160),
        ch_data.get("psd_figure")    or _placeholder_fig("No PSD available", 220),
        ch_data.get("epoch_figure")  or _placeholder_fig("No markers — epoch preview not available", 220),
    )


# ── Marker table editing ──────────────────────────────────────────────────────

@callback(
    Output("dp-marker-table", "data", allow_duplicate=True),
    Input("dp-add-marker-btn", "n_clicks"),
    State("dp-marker-table",   "data"),
    prevent_initial_call=True,
)
def add_marker_row(n_clicks, rows):
    rows = rows or []
    rows.append({"onset": 0.0, "duration": 1.0, "trial_type": ""})
    return rows


@callback(
    Output("dp-save-status", "children"),
    Input("dp-save-markers-btn", "n_clicks"),
    State("dp-marker-table",     "data"),
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


# ── Enable/disable − Sel / + Sel ─────────────────────────────────────────────

@callback(
    Output("dp-offset-sel-minus", "disabled"),
    Output("dp-offset-sel-plus",  "disabled"),
    Input("dp-marker-table",      "selected_rows"),
)
def toggle_sel_buttons(selected):
    disabled = not bool(selected)
    return disabled, disabled


# ── Marker offset (all / selected) ────────────────────────────────────────────

@callback(
    Output("dp-marker-table", "data", allow_duplicate=True),
    Input("dp-offset-all-minus", "n_clicks"),
    Input("dp-offset-all-plus",  "n_clicks"),
    State("dp-step-input",       "value"),
    State("dp-marker-table",     "data"),
    prevent_initial_call=True,
)
def offset_all_markers(minus_clicks, plus_clicks, step, rows):
    if not rows:
        return no_update
    delta = -(step or 1.0) if ctx.triggered_id == "dp-offset-all-minus" else (step or 1.0)
    for row in rows:
        row["onset"] = max(0.0, round(float(row.get("onset", 0)) + delta, 4))
    return rows


@callback(
    Output("dp-marker-table", "data", allow_duplicate=True),
    Input("dp-offset-sel-minus", "n_clicks"),
    Input("dp-offset-sel-plus",  "n_clicks"),
    State("dp-step-input",       "value"),
    State("dp-marker-table",     "data"),
    State("dp-marker-table",     "selected_rows"),
    prevent_initial_call=True,
)
def offset_selected_marker(minus_clicks, plus_clicks, step, rows, selected):
    if not rows or not selected:
        return no_update
    delta = -(step or 1.0) if ctx.triggered_id == "dp-offset-sel-minus" else (step or 1.0)
    for i in selected:
        if i < len(rows):
            rows[i]["onset"] = max(0.0, round(float(rows[i].get("onset", 0)) + delta, 4))
    return rows


# ── Batch rename ──────────────────────────────────────────────────────────────

@callback(
    Output("dp-marker-table",    "data", allow_duplicate=True),
    Output("dp-rename-from",     "value"),
    Output("dp-rename-to",       "value"),
    Input("dp-batch-rename-btn", "n_clicks"),
    State("dp-rename-from",      "value"),
    State("dp-rename-to",        "value"),
    State("dp-marker-table",     "data"),
    prevent_initial_call=True,
)
def batch_rename_markers(n_clicks, from_val, to_val, rows):
    if not rows or not from_val:
        return no_update, no_update, no_update
    to_val = to_val or ""
    for row in rows:
        if row.get("trial_type") == from_val:
            row["trial_type"] = to_val
    return rows, "", ""


# ── Export TSV ────────────────────────────────────────────────────────────────

@callback(
    Output("dp-tsv-download",  "data"),
    Input("dp-export-tsv-btn", "n_clicks"),
    State("dp-marker-table",   "data"),
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
    cached      = _RESULT_CACHE.get(store.get("cache_key"), {})
    valid_pairs = set(cached.get("channel_pairs", []))
    points      = click_data.get("points", [])
    if not points:
        return no_update
    pair = points[0].get("customdata", "")
    print(f"[DEBUG on_topo_click] pair={pair!r}, valid={pair in valid_pairs}")
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
