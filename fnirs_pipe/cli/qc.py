"""fnirs-qc CLI (argparse) — quality control for fNIRS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fnirs_pipe.cli import _shared
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.utils.logging import get_logger, setup_logging

setup_logging()

logger = get_logger("cli.qc")


def cmd_prep_raw(
    bids_dir: Path, output_dir: Path, participant_label: str,
    session_label: list[str] | None, task_label: list[str] | None,
    dpf: list[float], sci_threshold: float, psp_threshold: float | None,
    min_good_frac: float | None, screen_scope: str,
    cardiac_l_freq: float, cardiac_h_freq: float,
    window_length: float,
    epoch_qc: bool, epoch_tmin: float | None, epoch_tmax: float | None,
    by_condition: bool,
    motion_correction: str,
    short_max_dist: float | None, long_min_dist: float | None,
    long_max_dist: float | None,
    skip_bids_validation: bool,
) -> None:
    """Generate static raw QC report for a single participant."""
    from collections import defaultdict

    from fnirs_pipe.cli._shared import separation_bands_from_args
    from fnirs_pipe.qc.metrics._helpers import separation_bands

    sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })))

    from fnirs_pipe.io.bids import bids_label, get_layout, get_nirs_files
    from fnirs_pipe.io.derivatives import subject_report_dir
    from fnirs_pipe.qc.subject.prep_raw_report import build_prep_raw_report

    if (epoch_tmin is None) != (epoch_tmax is None):
        print("Error: --epoch-tmin and --epoch-tmax must be given together.", file=sys.stderr)
        raise SystemExit(1)

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
        html_path = (subject_report_dir(output_dir, participant_label)
                     / ("_".join(name_parts) + "_desc-raw_nirs.html"))
        print(f"Generating raw QC report: {html_path.name} ...")
        try:
            build_prep_raw_report(group_runs, html_path, dpf=dpf, sci_threshold=sci_threshold,
                                  psp_threshold=psp_threshold,
                                  min_good_frac=min_good_frac,
                                  screen_scope=screen_scope,
                                  cardiac_l_freq=cardiac_l_freq, cardiac_h_freq=cardiac_h_freq,
                                  window_s=window_length, epoch_qc=epoch_qc,
                                  epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                  sep_bands=sep_bands, by_condition=by_condition,
                                  motion_correction=motion_correction)
            print(f"  -> {html_path}")
        except Exception as exc:
            logger.exception("Raw report generation failed for %s", html_path.name)
            print(f"  [error] {exc}", file=sys.stderr)

    # rebuilt rather than added to: it is assembled from the records on disk, so it comes
    # back carrying the pipeline's reports too where a run has been through both commands
    from fnirs_pipe.qc.subject.subject_index import write_subject_index
    try:
        index = write_subject_index(participant_label,
                                    subject_report_dir(output_dir, participant_label),
                                    " ".join(sys.argv))
        if index:
            print(f"  -> {index}")
    except Exception:
        logger.warning("sub-%s | subject index failed", participant_label, exc_info=True)

    print(f"Done. {len(all_runs)} run(s) processed.")


def cmd_hyper_raw(
    bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
    dpf: list[float], sci_threshold: float, psp_threshold: float | None,
    min_good_frac: float | None, screen_scope: str,
    cardiac_l_freq: float, cardiac_h_freq: float,
    coherence_fmin: float, coherence_fmax: float,
    normalize: bool, no_align: bool, tstart: float | None, tend: float | None,
    session_label: list[str] | None, task_label: list[str] | None,
    short_max_dist: float | None, long_min_dist: float | None,
    long_max_dist: float | None,
    skip_bids_validation: bool,
) -> None:
    """Generate hyperscanning raw QC report from BIDS raw data."""
    from fnirs_pipe.cli._shared import separation_bands_from_args
    from fnirs_pipe.qc.metrics._helpers import separation_bands

    sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })))

    from fnirs_pipe.cli.hyper import _run_groups, _select_groups
    from fnirs_pipe.pipeline.hyperscanning import (
        _raw_to_haemo,
        align_like,
        align_recordings,
        compute_group_sqm_raw,
        crop_aligned_window,
        load_group_raw_bids,
        load_group_stage,
        normalize_raws,
        trim_to_shortest,
    )
    from fnirs_pipe.qc.hyper.hyper_report import build_hyper_report

    groups = _select_groups(pairs_csv, group_id, task_label)
    ses = session_label[0] if session_label else None

    def _process(gid, task, members):
        raws_cw = load_group_raw_bids(bids_dir, members)
        sqm_data = compute_group_sqm_raw(members, raws_cw, sci_threshold, output_dir,
                                         cardiac_l_freq, cardiac_h_freq,
                                         psp_threshold=psp_threshold,
                                         min_good_frac=min_good_frac,
                                         screen_scope=screen_scope,
                                         sep_bands=sep_bands)
        raws_haemo = {sid: _raw_to_haemo(r, dpf) for sid, r in raws_cw.items()}
        if no_align:
            aligned_raws, offsets = trim_to_shortest(raws_haemo)
        else:
            aligned_raws, offsets = align_recordings(raws_haemo, task)
        aligned_raws = crop_aligned_window(aligned_raws, tstart, tend)
        if normalize:
            aligned_raws = normalize_raws(aligned_raws)
        # the motion panel needs optical density, which the haemoglobin conversion above
        # has already left behind, so the intensity copy is cut to the same window rather
        # than aligned a second time. The corrected file is whatever the member's own
        # `fnirs-pipe` run left in derivatives, and is simply absent for a member who has
        # not been through one.
        intensity_raws = align_like(raws_cw, aligned_raws)
        after_raws = align_like(load_group_stage(output_dir, members, "motcorrected"),
                                aligned_raws)
        return build_hyper_report(
            group_id=gid,
            task=task,
            group=members,
            sqm_data=sqm_data,
            aligned_raws=aligned_raws,
            offsets=offsets,
            raw_raws=raws_haemo,
            intensity_raws=intensity_raws,
            after_raws=after_raws,
            output_dir=output_dir,
            sep_bands=sep_bands,
            session=ses,
            sci_threshold=sci_threshold,
            cardiac_l_freq=cardiac_l_freq,
            cardiac_h_freq=cardiac_h_freq,
            coherence_fmin=coherence_fmin,
            coherence_fmax=coherence_fmax,
        )

    _run_groups(groups, _process)


def cmd_group_raw(output_dir: Path) -> None:
    """Aggregate per-subject SQMs into cohort_nirs.tsv + cohort_nirs.html."""
    from fnirs_pipe.qc.subject.group_writer import build_group_raw_report

    path = build_group_raw_report(output_dir)
    print(f"report -> {path}")


def cmd_group_hyper_raw(output_dir: Path) -> None:
    """Aggregate per-group hyper SQMs into cohort_hyper_nirs.tsv + cohort_hyper_nirs.html."""
    from fnirs_pipe.qc.hyper.group_hyper_writer import build_group_hyper_report

    path = build_group_hyper_report(output_dir)
    if path is None:
        print(f"no dyad records under {output_dir}; run `fnirs-qc hyper-raw` first")
        return
    print(f"report -> {path}")


def cmd_provenance(output_dir: Path) -> None:
    """Render the file provenance graph for every subject / group in a derivatives tree.

    Reads the JSON sidecars already on disk, so it works on any past run.
    """
    from fnirs_pipe.qc.figures.provenance_figure import write_provenance

    # the root is always searched too: hyper-raw writes its group TSVs there, not under nirs/
    targets = [
        *sorted(output_dir.glob("sub-*/nirs")),
        *sorted(output_dir.glob("group-*/nirs")),
        output_dir,
    ]

    from fnirs_pipe.qc.subject.sqm_record import scan_runs

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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-qc",
        description="fNIRS quality control: individual, hyperscanning and cohort-level reports.",
    )
    sub = p.add_subparsers(required=True)

    pr = sub.add_parser("prep-raw", parents=[_shared.screening(sci_default=SCI_PASS)],
                        help="Static raw QC report for a single participant.")
    pr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    pr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    pr.add_argument("participant_label", help="Subject ID to inspect, e.g. '01'")
    pr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    pr.add_argument("--task-label",    nargs="+", action="extend", help="Task label(s) to include.")
    pr.add_argument("--dpf", nargs="+", type=float, action="extend", required=True,
                    help="Differential pathlength factor. One value or one per wavelength.")
    pr.add_argument("--cardiac-l-freq", type=float, required=True,
                    help="Lower bound of cardiac band in Hz (required; population-dependent).")
    pr.add_argument("--cardiac-h-freq", type=float, required=True,
                    help="Upper bound of cardiac band in Hz (required; population-dependent).")
    pr.add_argument("--window-length", type=float, default=10.0,
                    help="Sliding-window length (s) for windowed SCI/PSP/GVTD series.")
    pr.add_argument("--epoch-qc", action="store_true",
                    help="Add a per-trial section: score each event window on its own and show "
                         "them as a trial x metric heatmap.")
    pr.add_argument("--epoch-tmin", type=float, default=None,
                    help="Trial window start relative to event onset in s; negative pulls in a "
                         "baseline. Omit to use each event's own duration.")
    pr.add_argument("--epoch-tmax", type=float, default=None,
                    help="Trial window end relative to event onset in s. Given together with "
                         "--epoch-tmin, or neither.")
    _shared.add_separation_bands(pr)
    pr.add_argument("--by-condition", action="store_true",
                    help="Also write one report per annotated condition, beside the run's "
                         "own and named the way `fnirs-prep crop` names a segment: the "
                         "condition becomes the task- entity. Their numbers are sliced out "
                         "of the run's windowed pass, so every condition sits on the same "
                         "window grid and the same filter as the run; nothing is cut and "
                         "nothing is re-measured. Each page screens on its own stretch, so "
                         "the verdict on it is that condition's; the recording was processed "
                         "under the run's, which its own page carries.")
    pr.add_argument("--motion-correction", choices=["tddr", "wavelet", "spline", "none"],
                    default="none",
                    help="Run this correction on a copy of the optical density and report "
                         "the recording either side of it: the correction's footprint, the "
                         "corrected file measured again on the keys the uncorrected one "
                         "carries, and one carpet showing both. Nothing is written back and "
                         "no other preprocessing runs, so the screening verdict and every "
                         "coupling metric still describe the recording as delivered. "
                         "Default none, which reports it uncorrected; the counts and the "
                         "GVTD series then have no corrected stage to be measured on, which "
                         "is what `fnirs-pipe run` gives them.")
    pr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    pr.set_defaults(func=cmd_prep_raw)

    hr = sub.add_parser("hyper-raw",
                        parents=[_shared.pairs_selection(),
                                 _shared.screening(sci_default=SCI_PASS),
                                 _shared.alignment_window()],
                        help="Hyperscanning raw QC report from BIDS raw data.")
    hr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    hr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    hr.add_argument("--dpf", nargs="+", type=float, action="extend", required=True,
                    help="Differential pathlength factor. One value or one per wavelength.")
    hr.add_argument("--cardiac-l-freq", type=float, required=True,
                    help="Lower bound of cardiac band in Hz (required; population-dependent).")
    hr.add_argument("--cardiac-h-freq", type=float, required=True,
                    help="Upper bound of cardiac band in Hz (required; population-dependent).")
    # named for what they set. The old --fmin / --fmax stay as aliases: they said nothing
    # about which of the report's frequency bands they were, and read as the analysis band
    # that fnirs-hyper spells --wtc-fmin
    hr.add_argument("--coh-fmin", "--fmin", dest="coherence_fmin", type=float, default=0.01,
                    help="Lower bound (Hz) of the band the Welch coherence is averaged over.")
    hr.add_argument("--coh-fmax", "--fmax", dest="coherence_fmax", type=float, default=0.10,
                    help="Upper bound (Hz) of that band.")
    _shared.add_separation_bands(hr)
    hr.add_argument("--session-label", nargs="+", action="extend", help="Session label(s) to include.")
    hr.add_argument("--skip-bids-validation", action=argparse.BooleanOptionalAction, default=False)
    hr.set_defaults(func=cmd_hyper_raw)

    gr = sub.add_parser(
        "cohort",
        help="Every subject in a tree on one page.",
        description="Aggregates the quality record of every run under OUTPUT_DIR. A run processed by both `fnirs-pipe` and `prep-raw` has two records and the pipeline one wins, being a superset, so the page carries every stage the run was measured at rather than the original recording alone.")
    gr.add_argument("output_dir", type=Path,
                    help="fnirs-pipe derivatives directory (contains sub-*/nirs/ SQM JSONs)")
    gr.set_defaults(func=cmd_group_raw)

    ghr = sub.add_parser(
        "cohort-hyper",
        help="Cohort report over every dyad: shared usable time, where it went, and each "
             "window against its own null.",
        description="One page for every dyad in a tree, built from the records and tables "
                    "the dyad runs already wrote. It reports what a dyad has and a subject "
                    "cannot; the per-subject quality distributions stay in `cohort`.")
    ghr.add_argument("output_dir", type=Path,
                     help="fnirs-pipe derivatives directory (contains group-*/nirs/ SQM JSONs)")
    ghr.set_defaults(func=cmd_group_hyper_raw)

    pv = sub.add_parser("provenance", help="Render the file provenance graph from existing sidecars.")
    pv.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    pv.set_defaults(func=cmd_provenance)


    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    kw = {k: v for k, v in vars(args).items() if k != "func"}
    args.func(**kw)
