"""Callbacks for the hyperscanning align page."""

from __future__ import annotations

from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, no_update

# aligned_raws not JSON-serializable — keep in process memory
_ALIGNED_CACHE: dict[str, dict] = {}


def _cache_key(bids_dir: str, group_csv: str) -> str:
    import hashlib
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
    Output("ha-offset-table",  "data"),
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

    from fnirs_pipe.exceptions import AlignmentError
    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings, load_group_raw_bids, parse_group_csv,
    )

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
            raws = load_group_raw_bids(Path(bids_dir), group)
            aligned_raws, offsets = align_recordings(raws, task)
            all_aligned[(group_id, task)] = {
                "aligned_raws": aligned_raws,
                "offsets":      offsets,
                "subject_ids":  [e.subject_id for e in group],
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

    from fnirs_pipe.qc.figures.hyper_figures import (
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
            return fig.to_dict() if fig is not None else no_update
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

    import mne
    import pandas as pd

    from fnirs_pipe.utils.snirf_prep import (
        annotations_to_df, bids_stem, copy_sidecars,
        deriv_nirs_dir, ensure_dataset_description, find_snirf,
    )

    _DERIV_NAME = "aligned"
    deriv_path  = Path(deriv_dir)
    bids_path   = Path(bids_dir)
    written: list[str] = []
    errors:  list[str] = []

    for (group_id, task), info in cache["groups"].items():
        ensure_dataset_description(
            deriv_path / _DERIV_NAME, _DERIV_NAME, "fnirs-gui hyper-align"
        )
        offset_rows: list[dict] = []
        for sid in info["subject_ids"]:
            sub_label = sid.removeprefix("sub-")
            try:
                snirf_path = find_snirf(bids_path, sub_label, None, task, None)
            except Exception as exc:
                errors.append(f"{sid}: {exc}")
                continue

            stem    = bids_stem(snirf_path)
            out_dir = deriv_nirs_dir(deriv_path, _DERIV_NAME, sub_label, None)
            out_dir.mkdir(parents=True, exist_ok=True)
            copy_sidecars(snirf_path, stem, out_dir)

            out_snirf   = out_dir / f"{stem}_nirs.snirf"
            raw_aligned = info["aligned_raws"][sid]
            try:
                mne.export.export_raw(
                    str(out_snirf), raw_aligned, fmt="snirf",
                    overwrite=True, verbose=False,
                )
                annotations_to_df(raw_aligned).to_csv(
                    out_dir / f"{stem}_events.tsv", sep="\t", index=False,
                )
                written.append(out_snirf.name)
            except Exception as exc:
                errors.append(f"{sid}: {exc}")
                continue

            offset_rows.append({
                "subject_id": sid,
                "offset_s":   round(info["offsets"].get(sid, 0.0), 3),
                "duration_s": round(raw_aligned.times[-1], 1),
            })

        offsets_path = (
            deriv_path / _DERIV_NAME
            / f"group-{group_id}_task-{task}_align-offsets.tsv"
        )
        pd.DataFrame(offset_rows).to_csv(offsets_path, sep="\t", index=False)

    parts = [f"Written {len(written)} file(s)."]
    if errors:
        parts.append(f"{len(errors)} error(s): {'; '.join(errors[:3])}")
    return dbc.Alert(
        " ".join(parts),
        color="success" if not errors else "warning",
        className="mb-0 py-2",
    )
