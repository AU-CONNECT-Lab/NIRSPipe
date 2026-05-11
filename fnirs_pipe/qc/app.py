"""Flask-based interactive raw fNIRS QC viewer.

Launch via:  fnirs-pipe <bids> <out> participant --participant-label 01 --qc-raw
Or directly: from fnirs_pipe.qc.app import launch; launch(bids_dir, subject, runs, out_dir)

Server state (module-level, single-process):
  _raw           raw intensity MNE object for current run
  _raw_haemo     Beer-Lambert converted (HbO/HbR), computed on load
  _markers       list[dict]  {onset, duration, description}
  _bad_channels  set[str]    channel names marked bad by user
  _sci_scores    dict[str, float]
  _iqm           dict        from compute_raw_iqm
  _runs          list[dict]  [{label, snirf_path, events_path}]
  _run_idx       int         index into _runs
  _out_dir       Path        where to save edited files
"""

from __future__ import annotations

import json
import threading
import webbrowser
from pathlib import Path
from typing import Any

import mne
from flask import Flask, jsonify, render_template, request

from fnirs_pipe.qc.figures import (
    build_channel_figure,
    build_layout_figure,
    build_psd_mean_figure,
    build_ts_figure,
    condition_colors,
)
from fnirs_pipe.qc.quantitative_metrics import compute_raw_iqm
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.app")

app = Flask(__name__, template_folder="templates")

# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

_raw: mne.io.Raw | None = None
_raw_haemo: mne.io.Raw | None = None
_markers: list[dict] = []
_bad_channels: set[str] = set()
_sci_scores: dict[str, float] = {}
_iqm: dict[str, Any] = {}
_runs: list[dict] = []
_run_idx: int = 0
_out_dir: Path = Path(".")

_MAX_TS_PTS = 4000
_SHORT_THRESH = 0.015   # metres
_SCI_THRESHOLD = 0.75
_EPOCH_TMIN = -5.0
_EPOCH_TMAX = 25.0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_run(run_idx: int) -> None:
    global _raw, _raw_haemo, _markers, _sci_scores, _iqm, _run_idx, _bad_channels

    _run_idx = run_idx
    run = _runs[run_idx]
    logger.info("loading run %s …", run["label"])

    raw = mne.io.read_raw_snirf(str(run["snirf_path"]), preload=True, verbose=False)

    # compute SCI — requires OD data
    try:
        raw_od_sci = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        sci_arr = mne.preprocessing.nirs.scalp_coupling_index(raw_od_sci, verbose=False)
        _sci_scores = {ch: float(sci_arr[i]) for i, ch in enumerate(raw.ch_names)}
    except Exception as exc:
        logger.warning("SCI failed: %s", exc)
        _sci_scores = {ch: 1.0 for ch in raw.ch_names}

    bad = [ch for ch, s in _sci_scores.items() if s < _SCI_THRESHOLD]

    try:
        _iqm = compute_raw_iqm(raw, _sci_scores, bad)
    except Exception as exc:
        logger.warning("IQM failed: %s", exc)
        _iqm = {}

    # Beer-Lambert for HbO/HbR detail + epoch preview
    try:
        raw_od = mne.preprocessing.nirs.optical_density(raw.copy(), verbose=False)
        _raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6.0)
    except Exception as exc:
        logger.warning("Beer-Lambert failed: %s", exc)
        _raw_haemo = None

    _raw = raw
    _bad_channels = set()   # reset per run

    # load events
    events_path = run.get("events_path")
    if events_path and Path(events_path).exists():
        import pandas as pd
        df = pd.read_csv(events_path, sep="\t")
        # BIDS uses "trial_type"; fall back to "description" then literal "stim"
        cols = df.columns.tolist()
        desc_col = next((c for c in ("trial_type", "description") if c in cols), None)
        _markers = [
            {
                "onset":       float(row["onset"]),
                "duration":    float(row.get("duration", 1.0)),
                "description": str(row[desc_col]) if desc_col else "stim",
            }
            for _, row in df.iterrows()
        ]
    else:
        # fall back to annotations embedded in the SNIRF
        anns = raw.annotations
        _markers = [
            {"onset": float(a["onset"]), "duration": float(a["duration"]),
             "description": str(a["description"])}
            for a in anns
        ]


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    run_labels = [r["label"] for r in _runs]
    return render_template("raw_viewer.html", run_labels=run_labels, run_idx=_run_idx)


@app.route("/api/ts_figure")
def api_ts_figure():
    if _raw is None:
        return jsonify({"error": "no data loaded"}), 503
    fig, marker_data, cond_colors_, band_shapes, t_start, t_end = build_ts_figure(
        _raw, _markers, _bad_channels, _MAX_TS_PTS, _SHORT_THRESH,
    )
    return jsonify({
        "figure":      fig.to_dict(),
        "markers":     marker_data,
        "cond_colors": cond_colors_,
        "band_shapes": band_shapes,
        "t_start":     t_start,
        "t_end":       t_end,
    })


@app.route("/api/channel_figure")
def api_channel_figure():
    """Per-channel HbO+HbR detail, PSD, and epoch figures."""
    ch_pair = request.args.get("pair", "")
    if _raw_haemo is None or not ch_pair:
        return jsonify({"error": "no haemo data or no channel specified"}), 400
    try:
        detail_fig, psd_fig, epoch_fig = build_channel_figure(
            _raw_haemo, _markers, ch_pair, _MAX_TS_PTS, _EPOCH_TMIN, _EPOCH_TMAX,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({
        "detail_figure": detail_fig.to_dict(),
        "psd_figure":    psd_fig.to_dict() if psd_fig else None,
        "epoch_figure":  epoch_fig.to_dict() if epoch_fig else None,
    })


@app.route("/api/layout_figure")
def api_layout_figure():
    """2D top-down + 3D fsaverage optode layout figures."""
    if _raw is None:
        return jsonify({"error": "no data"}), 503
    fig_2d, fig_3d = build_layout_figure(_raw, _bad_channels, _sci_scores, _SHORT_THRESH)
    return jsonify({
        "layout_2d_figure": fig_2d.to_dict() if fig_2d else None,
        "layout_3d_figure": fig_3d.to_dict() if fig_3d else None,
    })


@app.route("/api/psd_mean_figure")
def api_psd_mean_figure():
    """Mean PSD figure across all channels."""
    if _raw is None:
        return jsonify({}), 503
    fig = build_psd_mean_figure(_raw)
    if fig is None:
        return jsonify({}), 500
    return jsonify({"figure": fig.to_dict()})


@app.route("/api/iqm")
def api_iqm():
    scalars = {k: v for k, v in _iqm.items() if not isinstance(v, (dict, list))}
    per_ch = {
        k: v for k, v in _iqm.items()
        if isinstance(v, dict) and k.endswith("_per_channel")
    }
    return jsonify({"scalars": scalars, "per_channel": per_ch})


@app.route("/api/markers", methods=["GET"])
def api_markers_get():
    return jsonify({"markers": _markers, "cond_colors": condition_colors(_markers)})


@app.route("/api/markers/add", methods=["POST"])
def api_markers_add():
    body = request.get_json()
    _markers.append({
        "onset":       round(float(body["onset"]), 4),
        "duration":    float(body.get("duration", 1.0)),
        "description": str(body.get("description", "stim")),
    })
    _markers.sort(key=lambda m: m["onset"])
    return jsonify({"markers": _markers})


@app.route("/api/markers/update", methods=["POST"])
def api_markers_update():
    body = request.get_json()
    idx = int(body["idx"])
    if 0 <= idx < len(_markers):
        if "onset"       in body: _markers[idx]["onset"]       = round(float(body["onset"]), 4)
        if "duration"    in body: _markers[idx]["duration"]    = float(body["duration"])
        if "description" in body: _markers[idx]["description"] = str(body["description"])
        _markers.sort(key=lambda m: m["onset"])
    return jsonify({"markers": _markers})


@app.route("/api/markers/delete", methods=["POST"])
def api_markers_delete():
    idx = int(request.get_json()["idx"])
    if 0 <= idx < len(_markers):
        _markers.pop(idx)
    return jsonify({"markers": _markers})


@app.route("/api/markers/replace_description", methods=["POST"])
def api_markers_replace_description():
    body = request.get_json()
    old, new = str(body["old"]), str(body["new"])
    for m in _markers:
        if m["description"] == old:
            m["description"] = new
    return jsonify({"markers": _markers, "cond_colors": condition_colors(_markers)})


@app.route("/api/markers/offset", methods=["POST"])
def api_markers_offset():
    delta = float(request.get_json()["delta"])
    for m in _markers:
        m["onset"] = round(m["onset"] + delta, 4)
    return jsonify({"markers": _markers})


@app.route("/api/bad_channels/toggle", methods=["POST"])
def api_bad_toggle():
    ch = request.get_json()["ch"]
    if ch in _bad_channels:
        _bad_channels.discard(ch)
    else:
        _bad_channels.add(ch)
    return jsonify({"bad_channels": sorted(_bad_channels)})


@app.route("/api/run/<int:idx>", methods=["POST"])
def api_switch_run(idx: int):
    if 0 <= idx < len(_runs):
        _load_run(idx)
    return jsonify({"run_idx": _run_idx, "label": _runs[_run_idx]["label"]})


@app.route("/api/save", methods=["POST"])
def api_save():
    import pandas as pd

    run = _runs[_run_idx]
    stem = Path(run["snirf_path"]).stem
    out = _out_dir
    out.mkdir(parents=True, exist_ok=True)

    saved = []

    # events TSV
    if _markers:
        tsv_path = out / f"{stem}_events_edited.tsv"
        df = pd.DataFrame(_markers)
        df.to_csv(tsv_path, sep="\t", index=False)
        saved.append(str(tsv_path))

    # TODO(maybe): export a new snirf with edited annotations + bad channels dropped
    #   via fnirs_pipe.io.snirf.write_snirf; raw.set_annotations(...) + raw.drop_channels(bad)

    # bad channels JSON
    if _bad_channels:
        json_path = out / f"{stem}_bad_channels.json"
        json_path.write_text(json.dumps(sorted(_bad_channels), indent=2), encoding="utf-8")
        saved.append(str(json_path))

    logger.info("saved: %s", saved)
    return jsonify({"saved": saved})


# ---------------------------------------------------------------------------
# Public launch entry point
# ---------------------------------------------------------------------------

def launch(
    runs: list[dict],
    out_dir: Path,
    host: str = "127.0.0.1",
    port: int = 5050,
    debug: bool = False,
) -> None:
    """Start the Flask viewer.

    Parameters
    ----------
    runs:
        List of dicts with keys: ``label``, ``snirf_path``, ``events_path`` (optional).
    out_dir:
        Directory where edited files are saved.
    """
    global _runs, _out_dir
    _runs = runs
    _out_dir = Path(out_dir)

    _load_run(0)

    url = f"http://{host}:{port}"
    logger.info("raw viewer → %s", url)
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=port, debug=debug, use_reloader=False)
