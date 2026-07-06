"""Per-trial (epoch) QC: crop each trial's time window, recompute raw SQM scalars,
render a per-recording trial x metric report. Reuses window_writer's crop + SQM core
and group_writer's TSV/HTML rendering. Trial windows come from SNIRF events (or a CSV);
each trial is one row, so the heatmap/table become trial x metric. No good/bad flag is
applied — the SQM values are reported as-is."""

from __future__ import annotations

from pathlib import Path

import mne

from fnirs_pipe.io.bids import get_layout, iter_run_files
from fnirs_pipe.qc.group_writer import _render_group, rows_to_dataframe
from fnirs_pipe.qc.window_writer import _crop_to_window, _sqm_for_cropped
from fnirs_pipe.utils.logging import get_logger

logger = get_logger("qc.epoch_writer")


def _read_trials(raw: mne.io.Raw, events_csv: Path | None) -> list[tuple[float, float, str]]:
    """Return (onset_s, duration_s, condition) per trial — from CSV if given, else SNIRF events."""
    if events_csv is not None:
        import pandas as pd
        df = pd.read_csv(events_csv, sep=None, engine="python")
        cols = {c.lower(): c for c in df.columns}
        if "onset" not in cols:
            raise ValueError("events CSV must have an 'onset' column")
        dur_c = cols.get("duration")
        cond_c = cols.get("trial_type") or cols.get("condition")
        return [
            (float(r[cols["onset"]]),
             float(r[dur_c]) if dur_c else 0.0,
             str(r[cond_c]) if cond_c else "")
            for _, r in df.iterrows()
        ]
    return [
        (float(a["onset"]), float(a["duration"]), str(a["description"]))
        for a in raw.annotations
    ]


def _epoch_qc_one(
    snirf_path: Path, bids_name: str, mode: str, tmin: float | None, tmax: float | None,
    events_csv: Path | None, sci_threshold: float,
    cardiac_l_freq: float, cardiac_h_freq: float, output_dir: Path,
) -> Path | None:
    raw = mne.io.read_raw_snirf(str(snirf_path), preload=True, verbose=False)
    trials = _read_trials(raw, events_csv)
    if not trials:
        logger.warning("no events for %s; skipping", snirf_path.name)
        return None

    rows: list[dict] = []
    for i, (onset, dur, cond) in enumerate(trials, start=1):
        if mode == "duration":
            if dur <= 0:
                logger.warning("%s trial %d has no duration; skipping", bids_name, i)
                continue
            t0, t1 = onset, onset + dur
        else:  # epoch: fixed window relative to onset
            t0, t1 = onset + tmin, onset + tmax
        cropped = _crop_to_window(raw, t0, t1, align="none")
        if cropped is None:
            continue
        sqm = _sqm_for_cropped(cropped, sci_threshold, cardiac_l_freq, cardiac_h_freq, windowed=False)
        label = f"trial-{i:03d}_{onset:.0f}s" + (f"_{cond}" if cond else "")
        rows.append({"bids_name": label, **sqm})

    if not rows:
        logger.warning("no valid trial windows for %s", bids_name)
        return None

    out_stem = f"{bids_name}_epochqc"
    title = f"Epoch QC ({mode}) — {bids_name} ({len(rows)} trials)"
    df = rows_to_dataframe(rows)
    return _render_group(output_dir, out_stem, title, df, rows)


def build_epoch_qc_report(
    bids_dir: Path,
    output_dir: Path,
    task: str,
    mode: str,
    tmin: float | None,
    tmax: float | None,
    events_csv: Path | None = None,
    participant_label: list[str] | None = None,
    session_label: list[str] | None = None,
    sci_threshold: float = 0.8,
    *,
    cardiac_l_freq: float,
    cardiac_h_freq: float,
    skip_bids_validation: bool = False,
) -> list[Path]:
    """Per-recording trial x metric QC report ({bids_name}_epochqc.{tsv,html})."""
    layout = get_layout(bids_dir, validate=not skip_bids_validation)
    subjects = participant_label or sorted(layout.get_subjects())
    sessions = session_label or [None]

    out_paths: list[Path] = []
    for snirf_path, bids_name in iter_run_files(layout, subjects, sessions, task):
        try:
            p = _epoch_qc_one(
                snirf_path, bids_name, mode, tmin, tmax, events_csv,
                sci_threshold, cardiac_l_freq, cardiac_h_freq, output_dir,
            )
            if p is not None:
                out_paths.append(p)
        except Exception:
            logger.exception("epoch QC failed for %s", bids_name)
    return out_paths
