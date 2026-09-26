"""fnirs-qc CLI (argparse): quality control for fNIRS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

from fnirs_pipe import __version__

from fnirs_pipe.io.naming import report_name
from fnirs_pipe.cli import _shared
from fnirs_pipe.qc.metrics import SCI_PASS
from fnirs_pipe.utils.logging import get_logger, setup_logging
from fnirs_pipe.cli._shared import separation_bands_from_args
from fnirs_pipe.qc.metrics._helpers import separation_bands

setup_logging()

logger = get_logger("cli.qc")


def cmd_prep_raw(
    bids_dir: Path, output_dir: Path, participant_label: list[str],
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
    """Generate static raw QC reports, one subject at a time."""
    _shared.refuse_output_in_input(bids_dir, output_dir, "fnirs-qc")
    if not skip_bids_validation:
        from fnirs_pipe.io.bids import validate_bids
        validate_bids(bids_dir)
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

    layout = get_layout(bids_dir)
    sessions = session_label or [None]
    tasks    = task_label    or [None]

    def one_subject(subject: str) -> int:
        """Every report this subject's runs produce, plus the index over them."""
        all_runs: list[dict] = []
        groups: dict[tuple, list[dict]] = defaultdict(list)

        for session in sessions:
            for task in tasks:
                files = get_nirs_files(layout, subject=subject, session=session, task=task)
                for f in files:
                    entities = layout.parse_file_entities(str(f))
                    actual_ses = entities.get("session")
                    actual_task = entities.get("task")
                    label = bids_label(subject, entities)

                    snirf_p = Path(f)
                    events_p = snirf_p.parent / (snirf_p.name.replace("_nirs.snirf",
                                                                     "_events.tsv"))
                    run_dict = {
                        "label":       label,
                        "subject_id":  subject,
                        "snirf_path":  str(f),
                        "events_path": str(events_p) if events_p.exists() else None,
                        "session":     actual_ses,
                        "task":        actual_task,
                    }
                    all_runs.append(run_dict)
                    groups[(actual_ses, actual_task)].append(run_dict)

        if not all_runs:
            print(f"Error: no SNIRF files found for sub-{subject}.", file=sys.stderr)
            return 0

        for (ses, task), group_runs in groups.items():
            name_parts = [f"sub-{subject}"]
            if ses:  name_parts.append(f"ses-{ses}")
            if task: name_parts.append(f"task-{task}")
            html_path = (subject_report_dir(output_dir, subject)
                         / report_name("_".join(name_parts), desc="raw"))
            print(f"Generating raw QC report: {html_path.name} ...")
            try:
                build_prep_raw_report(group_runs, html_path, dpf=dpf,
                                      sci_threshold=sci_threshold,
                                      psp_threshold=psp_threshold,
                                      min_good_frac=min_good_frac,
                                      screen_scope=screen_scope,
                                      cardiac_l_freq=cardiac_l_freq,
                                      cardiac_h_freq=cardiac_h_freq,
                                      window_s=window_length, epoch_qc=epoch_qc,
                                      epoch_tmin=epoch_tmin, epoch_tmax=epoch_tmax,
                                      sep_bands=sep_bands, by_condition=by_condition,
                                      motion_correction=motion_correction)
                print(f"  -> {html_path}")
            except Exception as exc:
                logger.exception("Raw report generation failed for %s", html_path.name)
                print(f"  [error] {exc}", file=sys.stderr)

        # rebuilt rather than added to: it is assembled from the records on disk, so it
        # comes back carrying the pipeline's reports too where a run has been through both
        # commands
        from fnirs_pipe.qc.subject.subject_index import write_subject_index
        try:
            index = write_subject_index(subject, subject_report_dir(output_dir, subject),
                                        " ".join(sys.argv))
            if index:
                print(f"  -> {index}")
        except Exception:
            logger.warning("sub-%s | subject index failed", subject, exc_info=True)
        return len(all_runs)

    # one subject's failure must not take the rest of the batch with it
    n_runs, failed = 0, []
    for subject in participant_label:
        try:
            found = one_subject(subject)
            if not found:
                failed.append(subject)
            n_runs += found
        except Exception as exc:
            logger.exception("sub-%s | raw QC failed", subject)
            print(f"  [error] sub-{subject}: {exc}", file=sys.stderr)
            failed.append(subject)

    print(f"Done. {n_runs} run(s) processed over {len(participant_label)} subject(s).")
    if failed:
        print(f"  {len(failed)} subject(s) produced nothing: {', '.join(failed)}",
              file=sys.stderr)
        raise SystemExit(1)


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
    _shared.refuse_output_in_input(bids_dir, output_dir, "fnirs-hyper")
    if not skip_bids_validation:
        from fnirs_pipe.io.bids import validate_bids
        validate_bids(bids_dir)
    from fnirs_pipe.io.derivatives import write_bidsignore, write_dataset_description

    # this writes group-*/ too, so the tree it lands in gets the same stamp fnirs-hyper
    # gives it, whichever of the two runs first
    write_dataset_description(output_dir, name="fnirs-hyper output",
                              generated_by="fnirs-hyper", source=bids_dir)
    write_bidsignore(output_dir)

    sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })))

    from fnirs_pipe.cli.hyper import _run_groups, _select_groups
    from fnirs_pipe.pipeline.hyper import (
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
    """Aggregate per-subject SQMs into desc-subjects_qc.tsv + desc-subjects_report.html."""
    from fnirs_pipe.qc.subject.group_writer import build_group_raw_report

    path = build_group_raw_report(output_dir)
    print(f"report -> {path}")


def cmd_group_hyper_raw(output_dir: Path) -> None:
    """Aggregate per-group hyper SQMs into desc-groups_qc.tsv + desc-groups_report.html."""
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
    from fnirs_pipe.qc.common.figure_io import figure_namer
    from fnirs_pipe.qc.figures.common.provenance_figure import write_provenance

    from fnirs_pipe.io.derivatives import entity_of
    from fnirs_pipe.qc.subject.sqm_record import scan_runs

    # the root is always searched too: the merged cross-dyad and group-null tables sit there
    targets = [
        *sorted(output_dir.glob("sub-*/nirs")),
        *sorted(output_dir.glob("sub-*/ses-*/nirs")),
        *sorted(output_dir.glob("group-*/nirs")),
        output_dir,
    ]

    total = 0
    for nirs_dir in targets:
        dest = nirs_dir.parent if nirs_dir.name == "nirs" else nirs_dir
        if dest.name.startswith("ses-"):
            dest = dest.parent  # a subject's figures sit beside its reports, above sessions
        # Same destinations and names the writers use, so re-rendering refreshes the image
        # an already-written report points at: a subject gets one graph per run, a group one
        # per task named as its dyad report names it, and the root a single graph
        if dest.name.startswith("sub-"):
            jobs = [(label, label) for label in scan_runs(nirs_dir)]
        elif dest.name.startswith("group-"):
            tasks = sorted({entity_of(p, "task") for p in nirs_dir.glob("*.json")} - {None})
            jobs = [(f"{dest.name}_task-{task}", None) for task in tasks]
        else:
            jobs = []
        for name, label in jobs or [(dest.name, None)]:
            written = write_provenance(nirs_dir, dest / "figures", figure_namer(name),
                                       title=name, label=label)
            for path in written:
                print(f"{path}")
            total += bool(written)

    if not total:
        print("no provenance sidecars found - run the pipeline first")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fnirs-qc",
        description="fNIRS quality control: individual, hyperscanning and cohort-level reports.",
    )
    p.add_argument("--version", action="version", version=f"fnirs-qc {__version__}")
    sub = p.add_subparsers(required=True)

    pr = sub.add_parser("prep-raw", parents=[_shared.screening(sci_default=SCI_PASS)],
                        help="Static raw QC report for a single participant.")
    pr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    pr.add_argument("output_dir", type=Path, help="fnirs-pipe derivatives directory")
    pr.add_argument("--participant-label", "--participant_label", nargs="+", action="extend", required=True,
                    type=_shared.BidsLabel,
                    help="Subject ID(s) to inspect, e.g. '01'. One report set per subject, "
                         "and one subject's failure does not stop the rest.")
    pr.add_argument("--session-label", "--session_label", nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Session label(s) to include.")
    pr.add_argument("--task-label", "--task_label",    nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Task label(s) to include.")
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
                         "own, as <run>_cond-<condition>_desc-raw_report.html. Their numbers "
                         "are sliced out "
                         "of the run's windowed pass, on the same window grid and filter as "
                         "the run. Each page's screening verdict is that condition's; the "
                         "run's own page carries the run's.")
    pr.add_argument("--motion-correction", choices=["tddr", "wavelet", "none"],
                    default="none",
                    help="Run this correction on a copy of the optical density and report "
                         "the recording either side of it: the correction's footprint, the "
                         "corrected file measured again on the keys the uncorrected one "
                         "carries, and one carpet showing both. Nothing is written back and "
                         "no other preprocessing runs, so the screening verdict and every "
                         "coupling metric still describe the recording as delivered. "
                         "Default none, which reports it uncorrected, with no corrected-stage "
                         "counts or GVTD series.")
    pr.add_argument(
        "--skip-bids-validation", "--skip_bids_validation",
        "--skip_bids_validator", dest="skip_bids_validation",
        action=argparse.BooleanOptionalAction, default=False,
        help="Do not check the input with bids-validator.")
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
    # named for what they set; --fmin / --fmax stay as aliases
    hr.add_argument("--coh-fmin", "--fmin", dest="coherence_fmin", type=float, default=0.01,
                    help="Lower bound (Hz) of the band the Welch coherence is averaged over.")
    hr.add_argument("--coh-fmax", "--fmax", dest="coherence_fmax", type=float, default=0.10,
                    help="Upper bound (Hz) of that band.")
    _shared.add_separation_bands(hr)
    hr.add_argument("--session-label", "--session_label", nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Session label(s) to include.")
    hr.add_argument(
        "--skip-bids-validation", "--skip_bids_validation",
        "--skip_bids_validator", dest="skip_bids_validation",
        action=argparse.BooleanOptionalAction, default=False,
        help="Do not check the input with bids-validator.")
    hr.set_defaults(func=cmd_hyper_raw)

    gr = sub.add_parser(
        "cohort",
        help="Every subject in a tree on one page.",
        description="Aggregates the quality record of every run under OUTPUT_DIR. A run processed by both `fnirs-pipe` and `prep-raw` has two records and the pipeline one is used, so the page carries every stage the run was measured at.")
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
