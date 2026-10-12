"""Callbacks for the data preparation page."""

from __future__ import annotations

import io
import re
from pathlib import Path
import functools
import hashlib
import pickle
import subprocess
import tempfile

import dash_bootstrap_components as dbc
import numpy as np
from dash import ALL, Input, Output, Patch, State, callback, ctx, dcc, html, no_update

from nirspipe.utils import is_marker, pair_of
from nirspipe.interface.callbacks._sections import rng, summary, value
from nirspipe.interface.grid import rows_minus_clicked
from nirspipe.interface.theme import style_figure
from nirspipe.io.derivatives import subject_labels
from nirspipe.utils.logging import get_logger

logger = get_logger("interface.data_prep_callbacks")

# Server-side cache: cache_key -> _process_run result dict (large figures stay here)
_RESULT_CACHE: dict[str, dict] = {}
# In-memory only: cache_key -> raw_haemo MNE object (not pickled)
_HAEMO_CACHE: dict[str, object] = {}


def _parse_bids_entities(filename: str) -> dict:
    return dict(re.findall(r'([a-zA-Z]+)-([^_\.]+)', filename))


def _snirf_options(subject: str, bids_dir: str) -> list[dict]:
    sub_dir = Path(bids_dir) / f"sub-{subject}"
    options = []
    for f in sorted([*sub_dir.glob("nirs/*_nirs.snirf"), *sub_dir.glob("ses-*/nirs/*_nirs.snirf")]):
        e = _parse_bids_entities(f.name)
        parts = [f"sub-{e.get('sub', subject)}"]
        if e.get("ses"):  parts.append(f"ses-{e['ses']}")
        if e.get("task"): parts.append(f"task-{e['task']}")
        if e.get("run"):  parts.append(f"run-{e['run']}")
        options.append({"label": "_".join(parts), "value": str(f)})
    return options


# bump whenever a cached figure's builder changes, or the disk cache keeps serving the old one
_CACHE_VERSION = 9


def _make_cache_key(snirf_path: str, sci_thresh: float, cardiac_l: float, cardiac_h: float,
                    dpf: float, window_s: float, epoch_tmin: float, epoch_tmax: float,
                    sep_bands=None, screening=None) -> str:
    payload = (f"v{_CACHE_VERSION}|{snirf_path}|{sci_thresh}|{cardiac_l}|{cardiac_h}|{dpf}"
               f"|{window_s}|{epoch_tmin}|{epoch_tmax}"
               f"|{sep_bands}|{screening}")
    return hashlib.md5(payload.encode()).hexdigest()[:16]


@functools.lru_cache(maxsize=2)
def _layout(bids_dir: str):
    """One pybids index per dataset, which is what the channels.tsv lookup walks."""
    from nirspipe.io.bids import get_layout
    return get_layout(Path(bids_dir), validate=False)


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
        subjects = subject_labels(Path(bids_dir))
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
# (No direct figure outputs: restore_from_store handles all graphs)

@callback(
    Output("dp-run-store",   "data"),
    Output("dp-load-status", "children"),
    Input("dp-run-dropdown", "value"),
    State("dp-sci-thresh",   "value"),
    State("dp-cardiac-l",    "value"),
    State("dp-cardiac-h",    "value"),
    State("dp-dpf",          "value"),
    State("dp-window-s",     "value"),
    State("dp-psp-thresh",   "value"),
    State("dp-min-good-frac", "value"),
    State("dp-screen-scope", "value"),
    State("dp-ignore-status", "value"),
    State("dp-epoch-tmin",   "value"),
    State("dp-epoch-tmax",   "value"),
    State("dp-short-max-dist", "value"),
    State("dp-long-min-dist",  "value"),
    State("dp-long-max-dist",  "value"),
    State("app-bids-dir",    "data"),
    State("app-output-dir",  "data"),
    prevent_initial_call=True,
)
def load_run(run_path, sci_thresh, cardiac_l, cardiac_h, dpf,
             window_s, psp_thresh, min_good_frac, screen_scope, ignore_status,
             epoch_tmin, epoch_tmax,
             short_max_dist, long_min_dist, long_max_dist, bids_dir, output_dir):
    if not run_path:
        return no_update, no_update
    if not output_dir or not Path(output_dir).is_dir():
        return no_update, dbc.Alert("Set Output Directory first.", color="warning")
    if not bids_dir:
        return no_update, dbc.Alert("Set BIDS Directory first.", color="warning")
    if sci_thresh is None or cardiac_l is None or cardiac_h is None or dpf is None:
        return no_update, dbc.Alert(
            "Set SCI Threshold, Cardiac Band (lo/hi) and DPF before loading a run.",
            color="warning")

    sci_threshold = float(sci_thresh)
    cardiac_l, cardiac_h, dpf = float(cardiac_l), float(cardiac_h), float(dpf)
    from nirspipe.qc.subject.prep_raw_report import _EPOCH_TMAX, _EPOCH_TMIN
    window_s   = float(window_s if window_s is not None else 10.0)
    ep_tmin    = float(epoch_tmin if epoch_tmin is not None else _EPOCH_TMIN)
    ep_tmax    = float(epoch_tmax if epoch_tmax is not None else _EPOCH_TMAX)
    from nirspipe.cli._shared import separation_bands_from_args
    from nirspipe.qc.metrics._helpers import separation_bands
    try:
        sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
            "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
            "long_max_dist": long_max_dist,
        })))
    except ValueError as exc:
        return no_update, dbc.Alert(str(exc), color="warning")
    snirf_path = run_path
    # the recording's channels.tsv rejections, joined as nirspipe-qc prep-raw joins them
    tsv_bads: list[str] = []
    if not ignore_status and bids_dir:
        from nirspipe.io.bids import channel_status_bads
        try:
            tsv_bads = channel_status_bads(_layout(str(bids_dir)), Path(snirf_path))
        except Exception as exc:
            logger.warning("channels.tsv status unreadable, screening alone decides: %s", exc)
    screening = (psp_thresh, min_good_frac, screen_scope or "run", tuple(tsv_bads))
    cache_key  = _make_cache_key(snirf_path, sci_threshold, cardiac_l, cardiac_h, dpf,
                                 window_s, ep_tmin, ep_tmax, sep_bands, screening)
    disk_path  = Path(output_dir) / ".fnirs_cache" / f"{cache_key}.pkl"

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
            logger.warning("disk cache load failed, recomputing: %s", exc)
            result = None
            source = "compute"
    else:
        result = None
        source = "compute"

    if result is None:
        try:
            from nirspipe.qc.subject.prep_raw_report import _process_run
            run_label = Path(snirf_path).stem
            from nirspipe.io.derivatives import LINK_RAW, write_dataset_description
            # laid out as prep-raw lays out its tree: the quality record filed under sub- names
            # its sources through the description's link, one root per BIDS dataset read
            cache_root = (Path(output_dir) / ".fnirs_cache"
                          / hashlib.md5(str(bids_dir).encode()).hexdigest()[:8])
            cache_root.mkdir(parents=True, exist_ok=True)
            write_dataset_description(cache_root, source=Path(bids_dir), link=LINK_RAW)
            subject   = _parse_bids_entities(Path(snirf_path).name).get("sub", "")
            run_dir   = cache_root / f"sub-{subject}"
            # the second half is what the per-condition report pages need: windowed
            # matrices and paths, none of it serialisable and none of it this page's
            result, _ = _process_run(
                {"snirf_path": snirf_path, "label": run_label, "channels_tsv_bads": tsv_bads},
                sci_threshold,
                run_dir,
                cardiac_l,
                cardiac_h,
                [dpf],
                window_s=window_s,
                epoch_tmin=ep_tmin,
                epoch_tmax=ep_tmax,
                psp_threshold=psp_thresh,
                sep_bands=sep_bands,
                min_good_frac=min_good_frac,
                screen_scope=screen_scope or "run",
            )
            result["epoch_window"] = [ep_tmin, ep_tmax]
            # the per-channel figures are built on demand in update_channel_detail; the DPF
            # travels with the result so that lazy path converts the same way this one did
            result["dpf"] = dpf

            _RESULT_CACHE[cache_key] = result
            if disk_path:
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(disk_path, "wb") as f:
                        pickle.dump(result, f)
                except Exception as exc:
                    logger.warning("disk cache save failed: %s", exc)
        except Exception as exc:
            return no_update, dbc.Alert(f"Failed to load: {exc}", color="danger")

    logger.debug("load_run: source=%s, pairs=%d", source, len(result.get("channel_pairs", [])))

    store = {
        "cache_key":  cache_key,
        "snirf_path": snirf_path,
        "dpf":        dpf,
    }
    status = dbc.Alert(f"Loaded ({source}): {Path(snirf_path).name}",
                       color="success", className="mb-0 py-2")
    return store, status


# ── Restore all graphs from store (fires on load AND on page re-mount) ────────

_SHOW = {}

@callback(
    Output("dp-layout-2d",              "figure"),
    Output("dp-layout-3d",              "figure"),
    Output("dp-psd-mean",               "figure"),
    Output("dp-marker-store",           "data",    allow_duplicate=True),
    Output("dp-channel-selector",       "options"),
    Output("dp-channel-selector",       "value",   allow_duplicate=True),
    Output("dp-evoked-topo",            "figure"),
    Output("dp-trigger-timeline",       "figure"),
    Output("dp-layout-2d-wrap",         "style"),
    Output("dp-layout-3d-wrap",         "style"),
    Output("dp-psd-mean-wrap",          "style"),
    Output("dp-evoked-topo-wrap",       "style"),
    Output("dp-trigger-timeline-wrap",  "style"),
    Input("dp-run-store",          "data"),
    Input("dp-mount-tick",         "n_intervals"),
    prevent_initial_call="initial_duplicate",
)
def restore_from_store(store, _tick):
    if not store:
        return (no_update,) * len(ctx.outputs_list)
    cached = _RESULT_CACHE.get(store.get("cache_key"), {})
    if not cached:
        return (no_update,) * len(ctx.outputs_list)

    def _fig(nested, *keys):
        d = nested
        for k in keys:
            if not isinstance(d, dict):
                return no_update
            d = d.get(k)
        return style_figure(d) if d else no_update

    layout_2d    = _fig(cached, "layout",           "layout_2d_figure")
    layout_3d    = _fig(cached, "layout",           "layout_3d_figure")
    psd_out      = _fig(cached, "psd",              "figure")
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

    ch_pairs = cached.get("channel_pairs", [])
    ch_labels = cached.get("channel_labels", {})
    ch_options = [{"label": ch_labels.get(p, p), "value": p} for p in ch_pairs] or no_update

    def _wrap(val):
        return _SHOW if val is not no_update else no_update

    return (
        layout_2d,
        layout_3d,
        psd_out,
        marker_rows,
        ch_options,
        no_update,
        evoked_topo,
        trigger_tl,
        _wrap(layout_2d),
        _wrap(layout_3d),
        _wrap(psd_out),
        _wrap(evoked_topo),
        _wrap(trigger_tl),
    )


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

        # this figure is built on the CW object, so its channels are 'S1_D1 760', not ' hbo'
        idx = next((i for i, ch in enumerate(customdata)
                    if pair_of(ch) == channel_pair), None) if channel_pair else None
        if idx is None:
            hx, hy, hz = [], [], []
        else:
            hx, hy, hz = [xs[idx]], [ys[idx]], [zs[idx]]

        patched = Patch()
        patched["data"][hl_idx]["x"] = hx
        patched["data"][hl_idx]["y"] = hy
        patched["data"][hl_idx]["z"] = hz
        return patched
    except Exception as exc:
        logger.warning("3D optode highlight failed", exc_info=exc)
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
    Output("dp-channel-psd",         "figure"),
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
            # loaded from disk cache, so recompute haemo from SNIRF
            try:
                import mne
                snirf_path = store.get("snirf_path", "")
                raw = mne.io.read_raw_snirf(snirf_path, preload=True, verbose=False)
                raw_od = mne.preprocessing.nirs.optical_density(raw, verbose=False)
                ppf = float(cached.get("dpf") or store.get("dpf") or 6.0)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=ppf)
                _HAEMO_CACHE[cache_key] = raw_haemo
            except Exception as exc:
                logger.warning("channel detail: haemoglobin recompute failed", exc_info=exc)
                return (
                    _placeholder_fig("Failed to load channel data", 160),
                    no_update, no_update, _SHOW, no_update, no_update,
                )
        try:
            from nirspipe.qc.common.figure_io import extract_markers
            from nirspipe.qc.figures import build_channel_figure
            from nirspipe.qc.subject.prep_raw_report import _EPOCH_TMAX, _EPOCH_TMIN, _MAX_TS_PTS
            # off the raw being drawn, not the payload, whose list is on the data axis and
            # late by first_time when cropped; `_section_channel_detail` does the same
            markers = extract_markers(raw_haemo)
            tmin, tmax = cached.get("epoch_window", [_EPOCH_TMIN, _EPOCH_TMAX])
            cardiac = cached.get("cardiac")
            detail_fig, psd_fig, epoch_fig = build_channel_figure(
                raw_haemo, markers, channel_pair, _MAX_TS_PTS, tmin, tmax,
                cardiac=tuple(cardiac) if cardiac else None,
            )
            channels[channel_pair] = {
                "detail_figure": detail_fig.to_dict() if detail_fig else None,
                "psd_figure":    psd_fig.to_dict()    if psd_fig    else None,
                "epoch_figure":  epoch_fig.to_dict()  if epoch_fig  else None,
            }
            logger.debug("channel detail: computed %r", channel_pair)
        except Exception as exc:
            logger.warning("channel detail: %r build failed", channel_pair, exc_info=exc)
            channels[channel_pair] = {}

    ch_data = channels.get(channel_pair, {})
    return (
        style_figure(ch_data.get("detail_figure") or _placeholder_fig("No channel data", 160)),
        style_figure(ch_data.get("psd_figure")    or _placeholder_fig("No PSD available", 220)),
        style_figure(ch_data.get("epoch_figure")
                     or _placeholder_fig("No markers: epoch preview not available", 220)),
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


def _selection(snirf_path: Path) -> list[str]:
    """nirspipe-prep's selection flags that pick out exactly the loaded run."""
    entities = _parse_bids_entities(snirf_path.name)
    args = ["--participant-label", entities.get("sub", "")]
    for flag, key in (("--session-label", "ses"), ("--task-label", "task"),
                      ("--run-label", "run")):
        if entities.get(key):
            args += [flag, entities[key]]
    # the run was read from this tree a moment ago; validating it again costs the save seconds
    return args + ["--skip-bids-validation"]


def _run_prep(argv: list[str]) -> dbc.Alert:
    """Run one nirspipe-prep command to the end and show what it printed."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return dbc.Alert(f"`{argv[0]}` not found on PATH. Make sure nirspipe is installed.",
                         color="danger", className="mb-0 py-2")
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    ok = proc.returncode == 0
    return dbc.Alert([
        html.Div("Done." if ok else f"Failed (exit {proc.returncode}).", className="fw-bold"),
        html.Code(" ".join(argv), className="d-block small mb-1"),
        html.Pre("\n".join(output.splitlines()[-10:]), className="fp-run-tail mb-0"),
    ], color="success" if ok else "danger", className="mb-0 py-2")


@callback(
    Output("dp-save-status", "children"),
    Input("dp-save-markers-btn", "n_clicks"),
    State("dp-marker-store",     "data"),
    State("dp-run-store",        "data"),
    State("app-bids-dir",        "data"),
    State("app-output-dir",      "data"),
    prevent_initial_call=True,
)
def save_markers(n_clicks, rows, store, bids_dir, output_dir):
    if not store or not store.get("snirf_path"):
        return dbc.Alert("No run loaded.", color="warning", className="mb-0 py-2")
    if not bids_dir or not output_dir:
        return dbc.Alert("Set the BIDS and output directories.", color="warning",
                         className="mb-0 py-2")

    import pandas as pd
    from nirspipe.io.tables import write_tsv

    df = pd.DataFrame(rows or [], columns=["onset", "duration", "trial_type"])
    df["onset"]    = pd.to_numeric(df["onset"],    errors="coerce").fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)

    with tempfile.TemporaryDirectory() as tmp:
        events = write_tsv(df, Path(tmp) / "events.tsv")
        return _run_prep(["nirspipe-prep", "edit-markers", "apply", str(bids_dir),
                          str(output_dir), *_selection(Path(store["snirf_path"])),
                          "--tsv", str(events)])


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
    # by shape name, not by index: a positional lookup would highlight the wrong cell
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

        sel       = channel_pair or ""
        sizes     = [15 if pair_of(ch) == sel else 10  for ch in customdata]
        opacities = [1.0 if pair_of(ch) == sel else 0.55 for ch in customdata]
        lw        = [2.0 if pair_of(ch) == sel else 0.8  for ch in customdata]

        patched = Patch()
        patched["data"][1]["marker"]["size"]          = sizes
        patched["data"][1]["marker"]["opacity"]       = opacities
        patched["data"][1]["marker"]["line"]["width"] = lw
        return patched
    except Exception as exc:
        logger.warning("2D optode highlight failed", exc_info=exc)
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
    pair    = pair_of(ch_name)
    logger.debug("on_layout_2d_click: ch_name=%r, pair=%r, valid=%s", ch_name, pair,
                 pair in valid_pairs)
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
    # its customdata, which avoids the "nearest point across subplots" misfire.
    curve_num = points[0].get("curveNumber", 0)
    fig_dict = cached.get("evoked_topo", {}).get("figure", {})
    traces = fig_dict.get("data", [])
    if curve_num < len(traces):
        cd = traces[curve_num].get("customdata", [])
        pair = cd[0] if isinstance(cd, list) and cd else str(cd) if cd else ""
        if pair in valid_pairs:
            logger.debug("on_topo_click: curve=%s, pair=%r", curve_num, pair)
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
    Output("dp-crop-seg-table", "rowData", allow_duplicate=True),
    Input("dp-crop-add-seg-btn", "n_clicks"),
    State("dp-crop-seg-table",   "virtualRowData"),
    prevent_initial_call=True,
)
def add_crop_segment(n_clicks, rows):
    rows = list(rows or [])
    rows.append({"onset": 0.0, "duration": 30.0})
    return rows


# ── Crop: delete segment row ──────────────────────────────────────────────────

@callback(
    Output("dp-crop-seg-table", "rowData", allow_duplicate=True),
    Input("dp-crop-seg-table",  "cellClicked"),
    State("dp-crop-seg-table",  "virtualRowData"),
    prevent_initial_call=True,
)
def delete_crop_segment(cell, rows):
    kept = rows_minus_clicked(cell, rows)
    return no_update if kept is None else kept


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
    Input("dp-crop-seg-table",    "virtualRowData"),
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
    State("dp-crop-seg-table",  "virtualRowData"),
    State("dp-crop-combine",    "value"),
    State("dp-run-store",       "data"),
    State("app-bids-dir",       "data"),
    State("app-output-dir",     "data"),
    prevent_initial_call=True,
)
def apply_crop(n_clicks, mode, tmin, tmax, seg_rows, combine_val, store, bids_dir, output_dir):
    if not store or not store.get("snirf_path"):
        return dbc.Alert("No run loaded.", color="warning", className="mb-0 py-2")
    if not bids_dir or not output_dir:
        return dbc.Alert("Set the BIDS and output directories.", color="warning",
                         className="mb-0 py-2")

    argv = ["nirspipe-prep", "crop", str(bids_dir), str(output_dir),
            *_selection(Path(store["snirf_path"]))]
    if mode == "single":
        if tmin is None and tmax is None:
            return dbc.Alert("Set tmin or tmax.", color="warning", className="mb-0 py-2")
        if tmin is not None:
            argv += ["--tmin", str(tmin)]
        if tmax is not None:
            argv += ["--tmax", str(tmax)]
        return _run_prep(argv)

    if not seg_rows:
        return dbc.Alert("Add at least one segment.", color="warning", className="mb-0 py-2")

    import pandas as pd
    from nirspipe.io.tables import write_tsv

    df = pd.DataFrame(seg_rows, columns=["onset", "duration", "task"])
    df["onset"]    = pd.to_numeric(df["onset"],    errors="coerce").fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
    if combine_val:
        argv.append("--combine")
    with tempfile.TemporaryDirectory() as tmp:
        segments = write_tsv(df, Path(tmp) / "segments.tsv")
        return _run_prep([*argv, "--segments-path", str(segments)])


@callback(
    Output("dp-params-summary", "children"),
    Input("dp-sci-thresh", "value"), Input("dp-window-s", "value"),
    Input("dp-psp-thresh", "value"), Input("dp-min-good-frac", "value"),
    Input("dp-screen-scope", "value"),
    Input("dp-dpf", "value"),
    Input("dp-cardiac-l", "value"), Input("dp-cardiac-h", "value"),
    Input("dp-epoch-tmin", "value"), Input("dp-epoch-tmax", "value"),
    Input("dp-short-max-dist", "value"), Input("dp-long-min-dist", "value"),
    Input("dp-ignore-status", "value"),
)
def params_summary(sci, window, psp, good_frac, scope, dpf, card_l, card_h, tmin, tmax,
                   short_max, long_min, ignore_status):
    return summary(value("SCI", sci), value("window", window, " s"), value("PSP", psp),
                   value("coupled", good_frac), scope, value("DPF", dpf),
                   rng("cardiac", card_l, card_h, " Hz"),
                   rng("epoch", tmin, tmax, " s"),
                   rng("separations", short_max, long_min, " mm"),
                   "channels.tsv ignored" if ignore_status else None)


# ── The drift cutoff this run's own markers allow ────────────────────────────

@callback(
    Output("dp-drift-hint", "children"),
    Input("dp-marker-store", "data"),
)
def drift_cutoff_hint(rows):
    """What to put in Analysis's drift high-pass, read off the markers on screen.

    Analysis is a batch page and cannot compute this: the interval differs per subject.
    Here one run is loaded, so the number is knowable.
    """
    intervals = {}
    for row in rows or []:
        name = str(row.get("trial_type") or "").strip()
        if not name or not is_marker(name):
            continue
        intervals.setdefault(name, []).append(float(row.get("onset") or 0.0))

    gaps = {name: max(np.diff(sorted(on))) for name, on in intervals.items() if len(on) >= 2}
    if not gaps:
        return None

    slowest_name = max(gaps, key=gaps.get)
    slowest = gaps[slowest_name]
    # every condition, since the one that sets the bound is rarely the one being modelled
    per_condition = " · ".join(f"{name} {gap:.0f} s"
                               for name, gap in sorted(gaps.items(), key=lambda kv: -kv[1]))
    return html.Small([
        "Drift high-pass for a GLM on this run: ",
        html.B(f"{1.0 / (2.0 * slowest):.4g} Hz"),
        f" or lower, set by {slowest_name}. Slowest repeat per condition: {per_condition}.",
    ], className="fp-hint d-block")
