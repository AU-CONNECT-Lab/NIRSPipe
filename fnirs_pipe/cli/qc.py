"""fnirs-qc CLI (argparse) — quality control for fNIRS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fnirs_pipe.utils.logging import get_logger, setup_logging

setup_logging()

logger = get_logger("cli.qc")


def _select_groups(pairs_csv: Path, group_id: str | None, task_label: list[str] | None) -> dict:
    """Parse the group CSV and filter by group_id / task_label. Exits non-zero on empty selection."""
    from fnirs_pipe.exceptions import GroupCSVError
    from fnirs_pipe.pipeline.hyperscanning import parse_group_csv

    try:
        groups = parse_group_csv(pairs_csv)
    except GroupCSVError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)

    if group_id is not None:
        groups = {k: v for k, v in groups.items() if k[0] == group_id}
        if not groups:
            print(f"[error] group_id '{group_id}' not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    if task_label is not None:
        groups = {k: v for k, v in groups.items() if k[1] in task_label}
        if not groups:
            print(f"[error] task_label {task_label} not found in CSV", file=sys.stderr)
            raise SystemExit(1)

    return groups


def _run_groups(groups: dict, process) -> None:
    """Run process(gid, task, members) -> report_path per group, tally ok/fail, exit non-zero on failure."""
    from fnirs_pipe.exceptions import AlignmentError, MissingDerivativesError

    print(f"Processing {len(groups)} group session(s)...")
    n_ok = n_fail = 0
    for (gid, task), members in groups.items():
        print(f"  -> {gid}/{task} ({len(members)} subjects)")
        try:
            report_path = process(gid, task, members)
            print(f"     report -> {report_path}")
            n_ok += 1
        except MissingDerivativesError as exc:
            print(f"     [skip] {exc}", file=sys.stderr)
            n_fail += 1
        except AlignmentError as exc:
            print(f"     [skip] alignment failed: {exc}", file=sys.stderr)
            n_fail += 1
        except Exception as exc:
            logger.exception("group %s task %s failed", gid, task)
            print(f"     [error] unexpected error: {exc}", file=sys.stderr)
            n_fail += 1

    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    if n_fail > 0:
        raise SystemExit(1)


def cmd_prep_raw(
    bids_dir: Path, output_dir: Path, participant_label: str,
    session_label: list[str] | None, task_label: list[str] | None,
    dpf: list[float], sci_threshold: float, cardiac_l_freq: float, cardiac_h_freq: float,
    window_length: float,
    skip_bids_validation: bool,
) -> None:
    """Generate static raw QC report for a single participant."""
    from collections import defaultdict

    from fnirs_pipe.io.bids import bids_label, get_layout, get_nirs_files
    from fnirs_pipe.qc.prep_raw_report import build_prep_raw_report

    layout = get_layout(bids_dir, validate=not skip_bids_validation)
    sessions = session_label or [None]
    tasks    = task_label    or [None]

    all_runs: list[dict] = []
    groups: dict[tuple, list[dict]] = defaultdict(list)

    for session in sessions:
        for task in tasks:
            files = get_nirs_files(layout, subject=participant_label, session=session, task=task)
            for f in files:
                entities = layout.parse_file_entities(str(f))
                actual_ses = entities.get("session")
                actual_task = entities.get("task")
                label = bids_label(participant_label, entities)

                snirf_p = Path(f)
                events_p = snirf_p.parent / (snirf_p.name.replace("_nirs.snirf", "_events.tsv"))
                run_dict = {
                    "label":       label,
                    "subject_id":  participant_label,
                    "snirf_path":  str(f),
                    "events_path": str(events_p) if events_p.exists() else None,
                    "session":     actual_ses,
                    "task":        actual_task,
                }
                all_runs.append(run_dict)
                groups[(actual_ses, actual_task)].append(run_dict)

    if not all_runs:
        print(f"Error: no SNIRF files found for sub-{participant_label}.", file=sys.stderr)
        raise SystemExit(1)

    for (ses, task), group_runs in groups.items():
        name_parts = [f"sub-{participant_label}"]
        if ses:  name_parts.append(f"ses-{ses}")
        if task: name_parts.append(f"task-{task}")
        html_path = output_dir / ("_".join(name_parts) + "_desc-raw_nirs.html")
        print(f"Generating raw QC report: {html_path.name} ...")
        try:
            build_prep_raw_report(group_runs, html_path, dpf=dpf, sci_threshold=sci_threshold,
                                  cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
                                  window_s=window_length)
            print(f"  -> {html_path}")
        except Exception as exc:
            logger.exception("Raw report generation failed for %s", html_path.name)
            print(f"  [error] {exc}", file=sys.stderr)

    print(f"Done. {len(all_runs)} run(s) processed.")


def cmd_hyper_raw(
    bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
    dpf: list[float], sci_threshold: float, cardiac_l_freq: float, cardiac_h_freq: float,
    coherence_fmin: float, coherence_fmax: float,
    normalize: bool, no_align: bool,
    session_label: list[str] | None, task_label: list[str] | None,
    skip_bids_validation: bool,
) -> None:
    """Generate hyperscanning raw QC report from BIDS raw data."""
    from fnirs_pipe.pipeline.hyperscanning import (
        _raw_to_haemo,
        align_recordings,
        compute_group_sqm_raw,
        compute_pairwise_coherence,
        load_group_raw_bids,
        normalize_raws,
        trim_to_shortest,
    )
    from fnirs_pipe.qc.hyper_report import build_hyper_report

    groups = _select_groups(pairs_csv, group_id, task_label)
    ses = session_label[0] if session_label else None

    def _process(gid, task, members):
        raws_cw = load_group_raw_bids(bids_dir, members)
        sqm_data = compute_group_sqm_raw(members, raws_cw, sci_threshold, output_dir,
                                         cardiac_l_freq, cardiac_h_freq)
        raws_haemo = {sid: _raw_to_haemo(r, dpf) for sid, r in raws_cw.items()}
        if no_align:
            aligned_raws, offsets = trim_to_shortest(raws_haemo)
        else:
            aligned_raws, offsets = align_recordings(raws_haemo, task)
        if normalize:
            aligned_raws = normalize_raws(aligned_raws)
        coherence_df = compute_pairwise_coherence(
            aligned_raws, fmin=coherence_fmin, fmax=coherence_fmax
        )
        return build_hyper_report(
            group_id=gid,
            task=task,
            group=members,
            sqm_data=sqm_data,
            aligned_raws=aligned_raws,
            offsets=offsets,
            raw_raws=raws_haemo,
            coherence_df=coherence_df,
            output_dir=output_dir,
            session=ses,
            sci_threshold=sci_threshold,
            cardiac_l_freq=cardiac_l_freq,
            cardiac_h_freq=cardiac_h_freq,
            coherence_fmin=coherence_fmin,
            coherence_fmax=coherence_fmax,
        )

    _run_groups(groups, _process)


def cmd_group_raw(output_dir: Path) -> None:
    """Aggregate per-subject prep-raw SQMs into group_nirs.tsv + group_nirs.html."""
    from fnirs_pipe.qc.group_writer import build_group_raw_report

    path = build_group_raw_report(output_dir)
    print(f"report -> {path}")


def cmd_group_hyper_raw(output_dir: Path) -> None:
    """Aggregate per-group hyper-raw SQMs into group_hyper_nirs.tsv + group_hyper_nirs.html."""
    from fnirs_pipe.qc.group_writer import build_group_hyper_raw_report

    path = build_group_hyper_raw_report(output_dir)
    print(f"report -> {path}")


def cmd_group_hyper_wtc(output_dir: Path) -> None:
    """Merge every per-dyad WTC band-mean table into one long table per kind."""
    from fnirs_pipe.qc.wtc_aggregate import write_aggregate_wtc

    wrote = False
    for kind in ("wtc", "wtc-roi"):
        path = write_aggregate_wtc(output_dir, kind=kind)
        if path is not None:
            print(f"{kind} -> {path}")
            wrote = True
    if not wrote:
        print(f"no hyper-wtc tables under {output_dir}; run `fnirs-qc hyper-post` first")


def cmd_provenance(output_dir: Path) -> None:
    """Render the file provenance graph for every subject / group in a derivatives tree.

    Reads the JSON sidecars already on disk, so it works on any past run.
    """
    from fnirs_pipe.qc.provenance import write_provenance

    # the root is always searched too: hyper-raw writes its group TSVs there, not under nirs/
    targets = [
        *sorted(output_dir.glob("sub-*/nirs")),
        *sorted(output_dir.glob("group-*/nirs")),
        output_dir,
    ]

    from fnirs_pipe.qc.sqm_record import scan_runs

    total = 0
    for nirs_dir in targets:
        dest = nirs_dir.parent if nirs_dir.name == "nirs" else nirs_dir
        # a subject holds one graph per run; anything else (a group tree, the root) has no
        # run entity to split on and keeps its single graph. Same destinations and stem the
        # run itself uses, so re-rendering refreshes the images an already-written QC
        # report points at
        runs = list(scan_runs(nirs_dir)) if dest.name.startswith("sub-") else []
        jobs = ([(dest / "figures" / label, label) for label in runs]
                or [(dest / "figures", None)])
        for out_dir, label in jobs:
            written = write_provenance(nirs_dir, out_dir, stem="provenance",
                                       title=label or dest.name, label=label)
            for path in written:
                print(f"{path}")
            total += bool(written)

    if not total:
        print("no provenance sidecars found — run the pipeline first")


def cmd_window_raw(
    bids_dir: Path, output_dir: Path, task_label: str, tstart: float, tend: float,
    participant_label: list[str] | None, session_label: list[str] | None,
    align: str, trigger_name: str | None, name: str | None,
    sci_threshold: float, cardiac_l_freq: float, cardiac_h_freq: float,
    window_length: float,
    skip_bids_validation: bool,
) -> None:
    """Crop each subject's raw to [tstart, tend] + aggregate SQM into a windowed group report."""
    from fnirs_pipe.qc.window_writer import build_window_raw_report

    path = build_window_raw_report(
        bids_dir=bids_dir, output_dir=output_dir,
        task=task_label, tstart=tstart, tend=tend,
        participant_label=participant_label, session_label=session_label,
        align=align, trigger_name=trigger_name, name=name,
        sci_threshold=sci_threshold,
        cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
        window_s=window_length,
        skip_bids_validation=skip_bids_validation,
    )
    print(f"report -> {path}")


def cmd_epoch(
    bids_dir: Path, output_dir: Path, task_label: str, mode: str,
    tmin: float | None, tmax: float | None, events_csv: Path | None,
    participant_label: list[str] | None, session_label: list[str] | None,
    sci_threshold: float, cardiac_l_freq: float, cardiac_h_freq: float,
    skip_bids_validation: bool,
) -> None:
    """Per-trial (epoch) QC: one window per event, recompute SQM, render trial x metric report."""
    from fnirs_pipe.qc.epoch_writer import build_epoch_qc_report

    if mode == "epoch" and (tmin is None or tmax is None):
        print("Error: --mode epoch requires --tmin and --tmax.", file=sys.stderr)
        raise SystemExit(1)

    paths = build_epoch_qc_report(
        bids_dir=bids_dir, output_dir=output_dir, task=task_label,
        mode=mode, tmin=tmin, tmax=tmax, events_csv=events_csv,
        participant_label=participant_label, session_label=session_label,
        sci_threshold=sci_threshold,
        cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
        skip_bids_validation=skip_bids_validation,
    )
    for p in paths:
        print(f"report -> {p}")
    if not paths:
        print("No epoch QC reports produced (no events / no matching files).", file=sys.stderr)


def cmd_hyper_post(
    bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
    desc: str,
    roi_mapping: Path | None, wtc_fmin: float, wtc_fmax: float,
    wtc_band_fmin: float | None, wtc_band_fmax: float | None, wtc_significance: bool,
    wtc_seed: int | None, wtc_mc_count: int, wtc_roi_cross: bool,
    wtc_channel_cross: bool, bads_scope: str,
    isc_threshold: float, normalize: bool, no_align: bool,
    session_label: list[str] | None, task_label: list[str] | None,
    skip_bids_validation: bool,
) -> None:
    """Generate hyperscanning post-processing QC report (WTC, ISC, connectivity)."""
    import json

    from fnirs_pipe.pipeline.hyperscanning import (
        align_recordings,
        load_group_haemo,
        load_group_sqm,
        normalize_raws,
        trim_to_shortest,
        write_group_bads,
    )
    from fnirs_pipe.qc.hyper_report import build_hyper_post_report

    if wtc_roi_cross and roi_mapping is None:
        print("[error] --wtc-roi-cross needs --roi-mapping: it crosses ROIs, and without a "
              "mapping there are none.", file=sys.stderr)
        raise SystemExit(1)

    groups = _select_groups(pairs_csv, group_id, task_label)

    roi_map: dict[str, list[str]] | None = None
    if roi_mapping is not None:
        try:
            roi_map = json.loads(roi_mapping.read_text())
        except Exception as exc:
            print(f"[error] failed to load ROI mapping: {exc}", file=sys.stderr)
            raise SystemExit(1)

    def _process(gid, task, members):
        raws = load_group_haemo(output_dir, members, desc=desc)
        if no_align:
            aligned_raws, offsets = trim_to_shortest(raws)
        else:
            aligned_raws, offsets = align_recordings(raws, task)
        if normalize:
            aligned_raws = normalize_raws(aligned_raws)
        group_sqm = load_group_sqm(output_dir, members, bads_scope=bads_scope)
        bad_channels = {sid: sqm.get("bad_channels", []) for sid, sqm in group_sqm.items()}
        write_group_bads(output_dir, members, group_sqm, bads_scope)
        return build_hyper_post_report(
            group_id=gid,
            task=task,
            group=members,
            aligned_raws=aligned_raws,
            offsets=offsets,
            output_dir=output_dir,
            roi_map=roi_map,
            bad_channels=bad_channels,
            wtc_fmin=wtc_fmin,
            wtc_fmax=wtc_fmax,
            wtc_band_fmin=wtc_band_fmin,
            wtc_band_fmax=wtc_band_fmax,
            wtc_significance=wtc_significance,
            wtc_seed=wtc_seed,
            wtc_mc_count=wtc_mc_count,
            wtc_roi_cross=wtc_roi_cross,
            wtc_channel_cross=wtc_channel_cross,
            isc_threshold=isc_threshold,
        )

    _run_groups(groups, _process)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-qc",
        description="fNIRS quality control: individual, hyperscanning and group-level reports.",
    )
    sub = p.add_subparsers(required=True)

    pr = sub.add_parser("prep-raw", help="Static raw QC report for a single participant.")
    pr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    pr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    pr.add_argument("participant_label", help="Subject ID to inspect, e.g. '01'")
    pr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    pr.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    pr.add_argument("--dpf", nargs="+", type=float, action="extend", required=True,
                    help="Differential pathlength factor. One value or one per wavelength.")
    pr.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI pass/fail threshold for bad channel detection.")
    pr.add_argument("--cardiac-l-freq", type=float, required=True,
                    help="Lower bound of cardiac band in Hz (required; population-dependent).")
    pr.add_argument("--cardiac-h-freq", type=float, required=True,
                    help="Upper bound of cardiac band in Hz (required; population-dependent).")
    pr.add_argument("--window-length", type=float, default=10.0,
                    help="Sliding-window length (s) for windowed SCI/PSP/GVTD series.")
    pr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    pr.set_defaults(func=cmd_prep_raw)

    hr = sub.add_parser("hyper-raw", help="Hyperscanning raw QC report from BIDS raw data.")
    hr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    hr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    hr.add_argument("--pairs-csv", type=Path, required=True,
                    help="CSV with columns: group_id, subject_id, task. "
                         "Each unique (group_id, task) pair is processed as one session.")
    hr.add_argument("--group-id", default=None,
                    help="Process only this group_id. Omit to process all groups.")
    hr.add_argument("--dpf", nargs="+", type=float, action="extend", required=True,
                    help="Differential pathlength factor. One value or one per wavelength.")
    hr.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI pass/fail threshold for channel quality comparison.")
    hr.add_argument("--cardiac-l-freq", type=float, required=True,
                    help="Lower bound of cardiac band in Hz (required; population-dependent).")
    hr.add_argument("--cardiac-h-freq", type=float, required=True,
                    help="Upper bound of cardiac band in Hz (required; population-dependent).")
    hr.add_argument("--fmin", dest="coherence_fmin", type=float, default=0.01,
                    help="Lower bound (Hz) for coherence frequency band.")
    hr.add_argument("--fmax", dest="coherence_fmax", type=float, default=0.10,
                    help="Upper bound (Hz) for coherence frequency band.")
    hr.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                    help="Z-score each channel per subject after alignment.")
    hr.add_argument("--no-align", action="store_true",
                    help="Skip trigger-based alignment; trim all recordings to the shortest duration.")
    hr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    hr.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    hr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    hr.set_defaults(func=cmd_hyper_raw)

    gr = sub.add_parser("group-raw", help="Aggregate per-subject prep-raw SQMs.")
    gr.add_argument("output_dir", type=Path,
                    help="fnirs-pipe derivatives directory (contains sub-*/nirs/ SQM JSONs)")
    gr.set_defaults(func=cmd_group_raw)

    ghr = sub.add_parser("group-hyper-raw", help="Aggregate per-group hyper-raw SQMs.")
    ghr.add_argument("output_dir", type=Path,
                     help="fnirs-pipe derivatives directory (contains group-*/nirs/ SQM JSONs)")
    ghr.set_defaults(func=cmd_group_hyper_raw)

    ghw = sub.add_parser("group-hyper-wtc",
                         help="Merge per-dyad WTC band-mean tables into one long table.")
    ghw.add_argument("output_dir", type=Path,
                     help="Directory holding group-*_task-*_hyper-wtc.tsv (searched recursively)")
    ghw.set_defaults(func=cmd_group_hyper_wtc)

    pv = sub.add_parser("provenance", help="Render the file provenance graph from existing sidecars.")
    pv.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    pv.set_defaults(func=cmd_provenance)

    wr = sub.add_parser("window-raw", help="Windowed group raw QC report over [tstart, tend].")
    wr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    wr.add_argument("output_dir", type=Path, help="QC output directory (where group_nirs.html lives)")
    wr.add_argument("--task-label", required=True, help="BIDS task label (one task at a time).")
    wr.add_argument("--tstart", type=float, required=True, help="Window start time (s)")
    wr.add_argument("--tend",   type=float, required=True, help="Window end time (s)")
    wr.add_argument("--participant-label", nargs="+", action="extend",
                    help="Subject(s) to include (default: all).")
    wr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    wr.add_argument("--align", default="none",
                    help="t=0 origin: 'none' = recording start, 'trigger' = first matching annotation.")
    wr.add_argument("--trigger-name", default=None,
                    help="Annotation description used when --align trigger.")
    wr.add_argument("--name", default=None, help="Output suffix (default: window-{tstart}-{tend}).")
    wr.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI threshold for bad-channel detection.")
    wr.add_argument("--cardiac-l-freq", type=float, required=True,
                    help="Lower bound of cardiac band in Hz (required; population-dependent).")
    wr.add_argument("--cardiac-h-freq", type=float, required=True,
                    help="Upper bound of cardiac band in Hz (required; population-dependent).")
    wr.add_argument("--window-length", type=float, default=10.0,
                    help="Sliding-window length (s) for windowed SCI/PSP/GVTD series.")
    wr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    wr.set_defaults(func=cmd_window_raw)

    ep = sub.add_parser("epoch", help="Per-trial (epoch) QC report from task events.")
    ep.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    ep.add_argument("output_dir", type=Path, help="QC output directory")
    ep.add_argument("--task-label", required=True, help="BIDS task label (one task at a time).")
    ep.add_argument("--mode", choices=["epoch", "duration"], required=True,
                    help="epoch: fixed [onset+tmin, onset+tmax] window; duration: [onset, onset+event duration].")
    ep.add_argument("--tmin", type=float, default=None,
                    help="Epoch start relative to event onset in s (required for --mode epoch).")
    ep.add_argument("--tmax", type=float, default=None,
                    help="Epoch end relative to event onset in s (required for --mode epoch).")
    ep.add_argument("--events-csv", type=Path, default=None,
                    help="CSV with columns onset[,duration,trial_type]. Default: read events from the SNIRF.")
    ep.add_argument("--participant-label", nargs="+", action="extend",
                    help="Subject(s) to include (default: all).")
    ep.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    ep.add_argument("--sci-threshold", type=float, default=0.8,
                    help="SCI threshold for bad-channel detection.")
    ep.add_argument("--cardiac-l-freq", type=float, required=True,
                    help="Lower bound of cardiac band in Hz (required; population-dependent).")
    ep.add_argument("--cardiac-h-freq", type=float, required=True,
                    help="Upper bound of cardiac band in Hz (required; population-dependent).")
    ep.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    ep.set_defaults(func=cmd_epoch)

    hp = sub.add_parser("hyper-post", help="Hyperscanning post QC report (WTC, ISC, connectivity).")
    hp.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    hp.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    hp.add_argument("--pairs-csv", type=Path, required=True,
                    help="CSV with columns: group_id, subject_id, task. "
                         "Each unique (group_id, task) pair is processed as one session.")
    hp.add_argument("--group-id", default=None,
                    help="Process only this group_id. Omit to process all groups.")
    hp.add_argument("--desc", default="preproc",
                    help="desc entity of the per-subject stage to read, e.g. 'preproc' "
                         "(Beer-Lambert output) or 'errts' (confound-regression residual, "
                         "which is what short-channel regression leaves behind). Must be a "
                         "haemoglobin stage, not an optical-density one.")
    hp.add_argument("--roi-mapping", type=Path, default=None,
                    help="JSON file mapping ROI labels to lists of channel names. For ROI-level WTC. Optional.")
    hp.add_argument("--wtc-fmin", type=float, default=0.004, help="Lower bound (Hz) for WTC frequency axis.")
    hp.add_argument("--wtc-fmax", type=float, default=0.20,  help="Upper bound (Hz) for WTC frequency axis.")
    hp.add_argument("--wtc-band-fmin", type=float, default=None,
                    help="Lower bound (Hz) of the band the per-channel WTC TSV averages over. "
                         "Defaults to --wtc-fmin, i.e. the whole computed axis.")
    hp.add_argument("--wtc-band-fmax", type=float, default=None,
                    help="Upper bound (Hz) of that band. Defaults to --wtc-fmax.")
    hp.add_argument("--wtc-significance", action="store_true",
                    help="Overlay a Monte Carlo significance contour on WTC "
                         "(slow: see --wtc-mc-count for how slow).")
    hp.add_argument("--wtc-mc-count", type=int, default=300,
                    help="Surrogate series behind each --wtc-significance contour "
                         "(default 300). This is what the runtime is spent on and it "
                         "scales linearly; lower it to preview a run, raise it to settle "
                         "a contour. Ignored without --wtc-significance.")
    hp.add_argument("--wtc-seed", type=int, default=None,
                    help="Seed the Monte Carlo surrogates so --wtc-significance is "
                         "reproducible. Also bypasses pycwt's on-disk cache, which is not "
                         "keyed on the seed. Omit for the previous behaviour.")
    hp.add_argument("--wtc-roi-cross", action="store_true",
                    help="Cross every ROI with every other across the two brains instead of "
                         "pairing each ROI with its counterpart, so four ROIs give sixteen "
                         "coherence values rather than four. Needs --roi-mapping. The extra "
                         "pairs reach the TSV and an ROI x ROI matrix in the report; the "
                         "time-frequency heatmaps stay on the homologous pairs.")
    hp.add_argument("--wtc-channel-cross", action="store_true",
                    help="Cross every long channel with every other across the two brains "
                         "instead of pairing each channel with its counterpart, so 14 "
                         "channels give 196 coherence values rather than 14. The extra "
                         "pairs reach the channel TSV with a label2 column; the "
                         "time-frequency heatmaps stay on the homologous pairs. Single "
                         "channels are noisier than ROI averages, so treat the off-diagonal "
                         "as exploratory and correct for the number of tests.")
    hp.add_argument("--bads-scope", choices=("run", "subject"), default="run",
                    help="Which rejected channels are excluded from the inter-brain "
                         "metrics. 'run' (default) uses this task's own rejections. "
                         "'subject' unions them over every run of the subject, so all "
                         "conditions rest on the same channel set, which is what comparing "
                         "conditions needs; the cost is losing a channel everywhere because "
                         "one segment was bad.")
    hp.add_argument("--isc-threshold", type=float, default=0.3,
                    help="Minimum mean ISC to draw an arc in the connectivity circle.")
    hp.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=False,
                    help="Z-score each channel per subject after alignment.")
    hp.add_argument("--no-align", action="store_true",
                    help="Skip trigger-based alignment; trim all recordings to the shortest duration.")
    hp.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    hp.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    hp.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    hp.set_defaults(func=cmd_hyper_post)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
