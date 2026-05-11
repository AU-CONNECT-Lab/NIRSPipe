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
import numpy as np
from flask import Flask, jsonify, render_template, request

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

def _ch_colors(raw: mne.io.Raw) -> list[str]:
    picks = list(range(len(raw.ch_names)))
    try:
        dists = mne.preprocessing.nirs.source_detector_distances(raw.info, picks=picks)
        return ["#78b1f2" if d <= _SHORT_THRESH else "rgba(243,125,125,0.78)" for d in dists]
    except Exception:
        return ["rgba(243,125,125,0.78)"] * len(raw.ch_names)


def _decimate(arr: np.ndarray, times: np.ndarray, max_pts: int):
    if len(times) <= max_pts:
        return arr, times
    step = max(1, len(times) // max_pts)
    return arr[:, ::step], times[::step]


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


def _condition_colors(markers: list[dict]) -> dict[str, str]:
    palette = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    ]
    descs = list(dict.fromkeys(m["description"] for m in markers))
    return {d: palette[i % len(palette)] for i, d in enumerate(descs)}


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    run_labels = [r["label"] for r in _runs]
    return render_template("raw_viewer.html", run_labels=run_labels, run_idx=_run_idx)


@app.route("/api/timeseries")
def api_timeseries():
    if _raw is None:
        return jsonify({"error": "no data loaded"}), 503

    picks = mne.pick_types(_raw.info, meg=False, fnirs=True)
    if len(picks) == 0:
        picks = list(range(len(_raw.ch_names)))

    data, times = _raw.get_data(picks=picks, return_times=True)
    data, times = _decimate(data, times, _MAX_TS_PTS)

    all_colors = _ch_colors(_raw)
    cond_colors = _condition_colors(_markers)

    traces = []
    for i, pick in enumerate(picks):
        ch = _raw.ch_names[pick]
        trace_arr = data[i]
        std = float(np.std(trace_arr))
        normed = ((trace_arr - trace_arr.mean()) / std).tolist() if std > 0 else (trace_arr - trace_arr.mean()).tolist()
        normed_shifted = [v + i * 3 for v in normed]
        color = "#e74c3c" if ch in _bad_channels else (all_colors[pick] if pick < len(all_colors) else "#aaa")
        traces.append({
            "ch": ch,
            "y": normed_shifted,
            "color": color,
            "offset": i * 3,
            "is_bad": ch in _bad_channels,
        })

    # marker shapes
    marker_shapes = []
    for m in _markers:
        color = cond_colors.get(m["description"], "#f39c12")
        marker_shapes.append({
            "onset":       m["onset"],
            "duration":    m["duration"],
            "description": m["description"],
            "color":       color,
        })

    return jsonify({
        "times":   times.tolist(),
        "traces":  traces,
        "markers": marker_shapes,
        "ch_names": [_raw.ch_names[p] for p in picks],
        "n_channels": len(picks),
        "cond_colors": cond_colors,
    })


@app.route("/api/channel")
def api_channel():
    """Per-channel HbO+HbR detail, PSD, and epoch preview."""
    ch_pair = request.args.get("pair", "")
    if _raw_haemo is None or not ch_pair:
        return jsonify({"error": "no haemo data or no channel specified"}), 400

    hbo_name = f"{ch_pair} hbo"
    hbr_name = f"{ch_pair} hbr"
    haemo_names = _raw_haemo.ch_names

    if hbo_name not in haemo_names or hbr_name not in haemo_names:
        return jsonify({"error": f"channel pair {ch_pair!r} not found in haemo data"}), 404

    hbo_pick = haemo_names.index(hbo_name)
    hbr_pick = haemo_names.index(hbr_name)
    haemo_data, times = _raw_haemo.get_data(picks=[hbo_pick, hbr_pick], return_times=True)
    haemo_data, times_d = _decimate(haemo_data, times, _MAX_TS_PTS)

    detail = {
        "times": times_d.tolist(),
        "hbo":   (haemo_data[0] * 1e6).tolist(),   # µmol/L
        "hbr":   (haemo_data[1] * 1e6).tolist(),
        "ch_pair": ch_pair,
    }

    # per-channel PSD
    try:
        from scipy.signal import welch
        raw_haemo_arr = _raw_haemo.get_data(picks=[hbo_pick, hbr_pick])
        sfreq = _raw_haemo.info["sfreq"]
        nperseg = min(512, max(64, raw_haemo_arr.shape[1] // 4))
        freqs, psd_hbo = welch(raw_haemo_arr[0], fs=sfreq, nperseg=nperseg)
        _, psd_hbr     = welch(raw_haemo_arr[1], fs=sfreq, nperseg=nperseg)
        fmax = min(2.0, sfreq / 2)
        mask = freqs <= fmax
        psd_result = {
            "freqs": freqs[mask].tolist(),
            "hbo":   psd_hbo[mask].tolist(),
            "hbr":   psd_hbr[mask].tolist(),
        }
    except Exception as exc:
        logger.warning("PSD failed for %s: %s", ch_pair, exc)
        psd_result = {}

    # epoch preview
    epoch_result = {}
    if _markers:
        try:
            import pandas as pd
            anns = mne.Annotations(
                onset=[m["onset"] for m in _markers],
                duration=[m["duration"] for m in _markers],
                description=[m["description"] for m in _markers],
            )
            raw_copy = _raw_haemo.copy().set_annotations(anns)
            events_mne, event_id = mne.events_from_annotations(raw_copy, verbose=False)
            if len(events_mne) > 0:
                epochs = mne.Epochs(
                    raw_copy, events_mne, event_id,
                    tmin=_EPOCH_TMIN, tmax=_EPOCH_TMAX,
                    picks=[hbo_pick, hbr_pick],
                    baseline=(_EPOCH_TMIN, 0),
                    preload=True, verbose=False,
                )
                epoch_times = epochs.times.tolist()
                by_cond: dict[str, dict] = {}
                for cond, eid in event_id.items():
                    try:
                        ep = epochs[cond].get_data()   # (n_epochs, 2, n_times)
                        by_cond[cond] = {
                            "hbo": (ep[:, 0, :].mean(axis=0) * 1e6).tolist(),
                            "hbr": (ep[:, 1, :].mean(axis=0) * 1e6).tolist(),
                            "n":   int(ep.shape[0]),
                        }
                    except Exception:
                        pass
                epoch_result = {"times": epoch_times, "conditions": by_cond}
        except Exception as exc:
            logger.warning("Epoch preview failed for %s: %s", ch_pair, exc)

    return jsonify({"detail": detail, "psd": psd_result, "epoch": epoch_result})


@app.route("/api/layout")
def api_layout():
    """2D top-down + 3D fsaverage optode positions."""
    if _raw is None:
        return jsonify({"error": "no data"}), 503

    chs = _raw.info["chs"]
    ch_names = _raw.ch_names
    colors = _ch_colors(_raw)
    sci = _sci_scores

    ch_locs = np.array([ch["loc"][:3] for ch in chs])
    has_positions = np.any(ch_locs != 0)

    layout_2d = None
    if has_positions:
        x = (ch_locs[:, 0] * 1000).tolist()
        y = (ch_locs[:, 1] * 1000).tolist()
        seen: set = set()
        lines_x, lines_y = [], []
        for ch in chs:
            src = tuple(round(v, 6) for v in ch["loc"][3:6])
            det = tuple(round(v, 6) for v in ch["loc"][:3])
            key = src + det
            if key in seen or not (any(src) or any(det)):
                continue
            seen.add(key)
            lines_x += [src[0] * 1000, det[0] * 1000, None]
            lines_y += [src[1] * 1000, det[1] * 1000, None]
        layout_2d = {
            "x": x, "y": y,
            "lines_x": lines_x, "lines_y": lines_y,
            "ch_names": ch_names,
            "colors": [("#e74c3c" if n in _bad_channels else colors[i]) for i, n in enumerate(ch_names)],
            "sci": [sci.get(n, 1.0) for n in ch_names],
        }

    # 3D: raw head coordinates; brain_viewer.py handles MNI transform in JS
    layout_3d = None
    if has_positions:
        import os
        coords_mm = ch_locs * 1000
        try:
            fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
            trans = mne.read_trans(os.path.join(fs_dir, "bem", "fsaverage-trans.fif"))
            coords_mm = mne.transforms.apply_trans(trans, ch_locs) * 1000
        except Exception:
            pass

        seen3: set = set()
        lx, ly, lz = [], [], []
        for ch in chs:
            src, det = ch["loc"][:3], ch["loc"][3:6]
            key = tuple(round(float(v), 5) for v in np.concatenate([src, det]))
            if key in seen3 or not (np.any(src) or np.any(det)):
                continue
            seen3.add(key)
            try:
                src_m = mne.transforms.apply_trans(trans, src.reshape(1, 3))[0] * 1000
                det_m = mne.transforms.apply_trans(trans, det.reshape(1, 3))[0] * 1000
            except Exception:
                src_m, det_m = src * 1000, det * 1000
            lx += [float(src_m[0]), float(det_m[0]), None]
            ly += [float(src_m[1]), float(det_m[1]), None]
            lz += [float(src_m[2]), float(det_m[2]), None]

        layout_3d = {
            "x": coords_mm[:, 0].tolist(),
            "y": coords_mm[:, 1].tolist(),
            "z": coords_mm[:, 2].tolist(),
            "lines_x": lx, "lines_y": ly, "lines_z": lz,
            "ch_names": ch_names,
            "colors": [("#e74c3c" if n in _bad_channels else colors[i]) for i, n in enumerate(ch_names)],
        }

    return jsonify({"layout_2d": layout_2d, "layout_3d": layout_3d})


@app.route("/api/brain_mesh")
def api_brain_mesh():
    """fsaverage5 pial mesh vertices + faces."""
    try:
        from nilearn import datasets, surface as surf
        fsavg5 = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        meshes = []
        for key in ("pial_left", "pial_right"):
            verts, faces = surf.load_surf_mesh(fsavg5[key])
            meshes.append({
                "x": verts[:, 0].tolist(), "y": verts[:, 1].tolist(), "z": verts[:, 2].tolist(),
                "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist(),
            })
        return jsonify({"meshes": meshes})
    except Exception as exc:
        logger.warning("brain mesh failed: %s", exc)
        return jsonify({"meshes": []})


@app.route("/api/psd_mean")
def api_psd_mean():
    """Mean PSD across all channels (used before any channel is selected)."""
    if _raw is None:
        return jsonify({}), 503
    try:
        from scipy.signal import welch
        picks = mne.pick_types(_raw.info, meg=False, fnirs=True)
        if len(picks) == 0:
            picks = list(range(len(_raw.ch_names)))
        raw_od = mne.preprocessing.nirs.optical_density(_raw.copy(), verbose=False)
        data = raw_od.get_data(picks=picks)
        sfreq = raw_od.info["sfreq"]
        nperseg = min(512, max(64, data.shape[1] // 4))
        freqs, psds = welch(data, fs=sfreq, nperseg=nperseg)
        fmax = min(2.0, sfreq / 2)
        mask = freqs <= fmax
        return jsonify({
            "freqs":    freqs[mask].tolist(),
            "mean":     psds[:, mask].mean(axis=0).tolist(),
            "all":      psds[:, mask].tolist(),
        })
    except Exception as exc:
        logger.warning("mean PSD failed: %s", exc)
        return jsonify({}), 500


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
    return jsonify({"markers": _markers, "cond_colors": _condition_colors(_markers)})


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
    return jsonify({"markers": _markers, "cond_colors": _condition_colors(_markers)})


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
