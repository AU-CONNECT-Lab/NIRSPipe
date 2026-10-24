"""nirspipe-qc CLI (argparse): quality control for fNIRS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache, partial
from types import SimpleNamespace
from uuid import uuid4

from nirspipe import __version__

from nirspipe.io.derivatives import group_label, group_report_dir, read_dataset_description
from nirspipe.io.head_template import resolve_template, space_of
from nirspipe.io.optode_space import hold_placement, place_optodes
from nirspipe.io.naming import report_name
from nirspipe.cli import _app_options
from nirspipe.cli import _shared
from nirspipe.utils.logging import get_logger, setup_logging
from nirspipe.cli._app_options import separation_bands_from_args
from nirspipe.qc.metrics._helpers import separation_bands

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
    ignore: list[str] | None = None,
    roi_mapping: Path | None = None,
    motion_std_thresh: float | None = None,
    wavelet_iqr: float | None = None,
    run_label: list[str] | None = None,
    nprocs: int | None = None,
    template: str | None = None,
) -> None:
    """Generate static raw QC reports, one per subject."""
    _app_options.check_motion_settings(motion_correction, motion_std_thresh, wavelet_iqr)
    _shared.refuse_output_in_input(bids_dir, output_dir, "nirspipe-qc")
    if not skip_bids_validation:
        from nirspipe.io.bids import validate_bids
        validate_bids(bids_dir)
    from nirspipe.io.derivatives import LINK_RAW, write_bidsignore, write_dataset_description
    # the records it writes name the raw recordings, which resolve through this link
    write_dataset_description(output_dir, source=bids_dir, link=LINK_RAW)
    write_bidsignore(output_dir)
    sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })))

    from nirspipe.io.bids import bids_label, channel_status_bads, get_layout, get_nirs_files
    from nirspipe.io.derivatives import entity_of, subject_report_dir
    from nirspipe.qc.subject.prep_raw_report import build_prep_raw_report

    if (epoch_tmin is None) != (epoch_tmax is None):
        print("Error: --epoch-tmin and --epoch-tmax must be given together.", file=sys.stderr)
        raise SystemExit(1)
    roi_map = _app_options.load_roi_mapping(roi_mapping)

    layout = get_layout(bids_dir)
    sessions = session_label or [None]
    tasks    = task_label    or [None]
    ages = _ages_for(dpf, bids_dir, participant_label)
    templates = _app_options.templates_for(template, participant_label, ages)

    def one_subject(subject: str) -> int:
        """Every report this subject's runs produce, plus the index over them."""
        all_runs: list[dict] = []
        groups: dict[tuple, list[dict]] = defaultdict(list)

        for session in sessions:
            for task in tasks:
                files = get_nirs_files(layout, subject=subject, session=session, task=task)
                for f in files:
                    run = entity_of(Path(f), "run")
                    if run_label and run not in run_label:
                        continue
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
                        "channels_tsv_bads": ([] if "channel-status" in (ignore or [])
                                              else channel_status_bads(layout, snirf_p)),
                        "age": ages.get(subject),
                        "template": templates[subject],
                    }
                    all_runs.append(run_dict)
                    # a chosen run gets a page of its own rather than replacing its task's page
                    groups[(actual_ses, actual_task, run if run_label else None)].append(run_dict)

        if not all_runs:
            print(f"Error: no SNIRF files found for sub-{subject}.", file=sys.stderr)
            return 0

        for (ses, task, run), group_runs in groups.items():
            name_parts = [f"sub-{subject}"]
            if ses:  name_parts.append(f"ses-{ses}")
            if task: name_parts.append(f"task-{task}")
            if run:  name_parts.append(f"run-{run}")
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
                                      motion_correction=motion_correction,
                                      motion_std_thresh=motion_std_thresh,
                                      wavelet_iqr=wavelet_iqr, roi_map=roi_map)
                print(f"  -> {html_path}")
            except Exception as exc:
                logger.exception("Raw report generation failed for %s", html_path.name)
                print(f"  [error] {exc}", file=sys.stderr)

        # rebuilt rather than added to: it is assembled from the records on disk, so it
        # comes back carrying the pipeline's reports too where a run has been through both
        # commands
        from nirspipe.qc.subject.subject_index import write_subject_index
        try:
            index = write_subject_index(subject, subject_report_dir(output_dir, subject),
                                        " ".join(sys.argv))
            if index:
                print(f"  -> {index}")
        except Exception:
            logger.warning("sub-%s | subject index failed", subject, exc_info=True)
        return len(all_runs)

    # one subject's failure must not take the rest of the batch with it
    def isolated(subject: str) -> int:
        try:
            return one_subject(subject)
        except Exception as exc:
            logger.exception("sub-%s | raw QC failed", subject)
            print(f"  [error] sub-{subject}: {exc}", file=sys.stderr)
            return 0

    found = _shared.map_subjects(isolated, participant_label, nprocs)
    n_runs = sum(found)
    failed = [subject for subject, n in zip(participant_label, found) if not n]

    print(f"Done. {n_runs} run(s) processed over {len(participant_label)} subject(s).")
    if failed:
        print(f"  {len(failed)} subject(s) produced nothing: {', '.join(failed)}",
              file=sys.stderr)
        raise SystemExit(1)


def _generated_by(tree: Path) -> set[str]:
    """The tool names in a tree's dataset_description.json, empty when there is none."""
    desc = read_dataset_description(Path(tree))
    return {str(e.get("Name", "")) for e in desc.get("GeneratedBy") or [] if isinstance(e, dict)}


@lru_cache(maxsize=1)
def _bids_layout(bids_dir: Path, run_id: str):
    """The BIDS layout of one run, built once per process, so each worker indexes the dataset once."""
    from nirspipe.io.bids import get_layout
    return get_layout(bids_dir)


def _raw_one_group(s: SimpleNamespace, gid, task, ses, members):
    """One dyad of `nirspipe-qc hyper-raw`: quality record and raw report. ``s`` is the run's settings."""
    from nirspipe.io.auxiliary import imu_traces, read_aux_snirf
    from nirspipe.io.bids import channel_status_bads
    from nirspipe.pipeline.hyper.alignment import (
        align_imu_like,
        align_like,
        aligned_offsets,
        align_recordings,
        crop_aligned_window,
        normalize_raws,
        trim_to_shortest,
    )
    from nirspipe.pipeline.hyper.group_io import (
        _raw_to_haemo,
        load_group_raw_bids,
        load_group_stage,
    )
    from nirspipe.pipeline.dpf import dpf_caveat, wavelengths
    from nirspipe.pipeline.hyper.group_quality import compute_group_sqm_raw
    from nirspipe.qc.hyper.hyper_report import build_hyper_report
    from nirspipe.utils.lineage import path_from

    # each member's channels.tsv, read off the BIDS tree its recording came from
    layout = _bids_layout(s.bids_dir, s.run_id) if s.read_channel_status else None
    raws_cw = load_group_raw_bids(s.bids_dir, members)
    ages = _ages_for(s.dpf, s.bids_dir, list(raws_cw))
    # each member on their own head, held so every page drawn from them uses it
    for sid, raw in raws_cw.items():
        space = space_of(resolve_template(s.template, ages.get(sid)))
        hold_placement(raw, place_optodes(path_from(raw), space))
    tsv_bads = ({sid: channel_status_bads(layout, Path(path_from(raw)))
                 for sid, raw in raws_cw.items()} if layout is not None else {})
    # the aux group rides only in the source file, on each member's own clock. A file
    # that will not read costs the IMU rows and not the report
    try:
        member_imu = {sid: imu_traces(*read_aux_snirf(path_from(raw)))
                      for sid, raw in raws_cw.items() if path_from(raw)}
    except Exception:
        logger.warning("%s: IMU could not be read; the report has no IMU rows",
                       gid, exc_info=True)
        member_imu = {}
    sqm_data = compute_group_sqm_raw(members, raws_cw, s.sci_threshold, s.output_dir,
                                     s.cardiac_l_freq, s.cardiac_h_freq,
                                     psp_threshold=s.psp_threshold,
                                     min_good_frac=s.min_good_frac,
                                     screen_scope=s.screen_scope,
                                     sep_bands=s.sep_bands, session=ses,
                                     channels_tsv_bads=tsv_bads, imu=member_imu)
    raws_haemo = {sid: _raw_to_haemo(r, s.dpf, ages.get(sid)) for sid, r in raws_cw.items()}
    # the dyad page says it the way it says a member screening could not judge
    for sid, age in ages.items():
        if caveat := dpf_caveat(age, wavelengths(raws_cw[sid])):
            sqm_data.setdefault(sid, {})["dpf_caveat"] = caveat
    if s.no_align:
        aligned_raws, offsets = trim_to_shortest(raws_haemo)
    else:
        aligned_raws, offsets = align_recordings(raws_haemo, task)
    aligned_raws = crop_aligned_window(aligned_raws, s.tstart, s.tend)
    # --tstart moves the shared clock's zero, so each member's crop point moves with it
    offsets = aligned_offsets(raws_haemo, aligned_raws)
    if s.normalize:
        aligned_raws = normalize_raws(aligned_raws)
    # the motion panel needs optical density, which the haemoglobin conversion above
    # has already left behind, so the intensity copy is cut to the same window rather
    # than aligned a second time. The corrected file is whatever the member's own
    # `nirspipe` run left in the tree --derivatives-dir names, and is simply absent for
    # a member who has not been through one.
    intensity_raws = align_like(raws_cw, aligned_raws)
    full_after = (load_group_stage(s.derivatives_dir, members, "motcorrected")
                  if s.derivatives_dir is not None else {})
    after_raws = align_like(full_after, aligned_raws) if full_after else {}
    # the panel's IMU, moved by the same shift the intensity copy was cut by
    try:
        imu = align_imu_like(member_imu, raws_cw, aligned_raws)
    except Exception:
        logger.warning("%s: IMU could not be aligned; the motion panel has no IMU rows",
                       gid, exc_info=True)
        imu = {}
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
        imu=imu,
        # the uncut recordings, so the motion panel filters before it cuts
        full_raws=raws_cw,
        full_after=full_after,
        output_dir=s.output_dir,
        sep_bands=s.sep_bands,
        session=ses,
        sci_threshold=s.sci_threshold,
        cardiac_l_freq=s.cardiac_l_freq,
        cardiac_h_freq=s.cardiac_h_freq,
        roi_map=s.roi_map,
    )


def cmd_hyper_raw(
    bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
    dpf: list[float], sci_threshold: float, psp_threshold: float | None,
    min_good_frac: float | None, screen_scope: str,
    cardiac_l_freq: float, cardiac_h_freq: float,
    normalize: bool, no_align: bool, tstart: float | None, tend: float | None,
    session_label: list[str] | None, task_label: list[str] | None,
    short_max_dist: float | None, long_min_dist: float | None,
    long_max_dist: float | None,
    skip_bids_validation: bool,
    derivatives_dir: Path | None = None,
    roi_mapping: Path | None = None,
    participant_label: list[str] | None = None,
    ignore: list[str] | None = None,
    nprocs: int | None = None,
    template: str | None = None,
) -> None:
    """Generate hyperscanning raw QC report from BIDS raw data."""
    _shared.refuse_output_in_input(bids_dir, output_dir, "nirspipe-hyper")
    if derivatives_dir is not None:
        _shared.refuse_output_is_source(derivatives_dir, output_dir)
    made_by = _generated_by(output_dir)
    if "nirspipe" in made_by:
        print(f"Error: {output_dir} is a nirspipe tree; the dyad reports go to a tree of their "
              f"own or to the nirspipe-hyper tree. Pass that as OUTPUT_DIR and this one as "
              f"--derivatives-dir.", file=sys.stderr)
        raise SystemExit(1)
    if not skip_bids_validation:
        from nirspipe.io.bids import validate_bids
        validate_bids(bids_dir)
    from nirspipe.io.derivatives import (
        LINK_PREPROCESSED, LINK_RAW, write_bidsignore, write_dataset_description,
    )

    # a dyad QC tree of its own is stamped as this command's; the nirspipe-hyper tree keeps its
    tool = "nirspipe-hyper" if "nirspipe-hyper" in made_by else "nirspipe-qc hyper-raw"
    write_dataset_description(output_dir, name=f"{tool} output",
                              generated_by=tool, source=bids_dir, link=LINK_RAW)
    if derivatives_dir is not None:
        write_dataset_description(output_dir, name=f"{tool} output",
                                  generated_by=tool, source=derivatives_dir,
                                  link=LINK_PREPROCESSED)
    write_bidsignore(output_dir)

    sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })))

    from nirspipe.cli.hyper import _run_groups, _select_groups

    groups = _select_groups(pairs_csv, group_id, task_label, participant_label)
    if session_label:
        # as a BIDS App reads it: the sessions to process. A key naming its session (or its
        # occasion) is kept when listed; a key naming none is read once per listed session
        groups = {(gid, task, s): [e if ses else replace(e, session=s) for e in members]
                  for (gid, task, ses), members in groups.items()
                  for s in ([ses] if ses else session_label) if s in session_label}
        if not groups:
            print(f"[error] session_label {session_label} not found in CSV", file=sys.stderr)
            raise SystemExit(1)
    if derivatives_dir is None:
        print("[info] no --derivatives-dir: the motion panel shows the recordings before "
              "correction only.", file=sys.stderr)
    roi_map = _app_options.load_roi_mapping(roi_mapping)
    subjects = sorted({e.subject_id for members in groups.values() for e in members})
    _app_options.templates_for(template, subjects, _ages_for(dpf, bids_dir, subjects))

    settings = SimpleNamespace(
        bids_dir=bids_dir, output_dir=output_dir, derivatives_dir=derivatives_dir,
        read_channel_status="channel-status" not in (ignore or []), run_id=uuid4().hex,
        sci_threshold=sci_threshold, cardiac_l_freq=cardiac_l_freq,
        cardiac_h_freq=cardiac_h_freq, psp_threshold=psp_threshold,
        min_good_frac=min_good_frac, screen_scope=screen_scope, sep_bands=sep_bands, dpf=dpf,
        no_align=no_align, normalize=normalize, tstart=tstart, tend=tend, roi_map=roi_map,
        template=template)
    _run_groups(groups, partial(_raw_one_group, settings), nprocs,
                log_file=lambda key: group_report_dir(output_dir, key[0]) / "logs"
                / f"{group_label(*key)}_desc-raw.log")


def cmd_group_raw(output_dir: Path) -> None:
    """Aggregate per-subject SQMs into desc-subjects_qc.tsv + desc-subjects_report.html."""
    from nirspipe.qc.subject.group_writer import build_group_raw_report

    path = build_group_raw_report(output_dir)
    if path is None:
        print(f"Error: no quality records under {output_dir}; run `nirspipe` or "
              "`nirspipe-qc prep-raw` into it first.", file=sys.stderr)
        raise SystemExit(1)
    print(f"report -> {path}")


def cmd_group_hyper_raw(output_dir: Path) -> None:
    """Aggregate per-group hyper SQMs into desc-groups_qc.tsv + desc-groups_report.html."""
    from nirspipe.qc.hyper.group_hyper_writer import build_group_hyper_report

    path = build_group_hyper_report(output_dir)
    if path is None:
        print(f"Error: no dyad records under {output_dir}; run `nirspipe-qc hyper-raw` "
              "into it first.", file=sys.stderr)
        raise SystemExit(1)
    print(f"report -> {path}")


def cmd_provenance(output_dir: Path) -> None:
    """Render the file provenance graph for every subject / group in a derivatives tree.

    Reads the JSON sidecars already on disk, so it works on any past run.
    """
    from nirspipe.qc.common.figure_io import figure_namer
    from nirspipe.qc.figures.common.provenance_figure import write_provenance

    from nirspipe.io.derivatives import entity_of, group_label
    from nirspipe.qc.subject.sqm_record import scan_runs

    # the root is always searched too: the merged cross-dyad and group-null tables sit there
    targets = [
        *sorted(output_dir.glob("sub-*/nirs")),
        *sorted(output_dir.glob("sub-*/ses-*/nirs")),
        *sorted(output_dir.glob("group-*/nirs")),
        *sorted(output_dir.glob("group-*/ses-*/nirs")),
        output_dir,
    ]

    total = 0
    for nirs_dir in targets:
        dest = nirs_dir.parent if nirs_dir.name == "nirs" else nirs_dir
        if dest.name.startswith("ses-"):
            dest = dest.parent  # a subject's figures sit beside its reports, above sessions
        # Same destinations and names the writers use, so re-rendering refreshes the image
        # an already-written report points at: a subject gets one graph per run, a group one
        # per task and session named as its dyad report names it, and the root a single graph
        if dest.name.startswith("sub-"):
            jobs = [(label, label) for label in scan_runs(nirs_dir)]
        elif dest.name.startswith("group-"):
            gid = dest.name.removeprefix("group-")
            labels = sorted({group_label(gid, entity_of(p, "task"), entity_of(p, "ses"))
                             for p in nirs_dir.glob("*.json") if entity_of(p, "task")})
            jobs = [(label, label) for label in labels]
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


def _ages_for(dpf: list, bids_dir: Path, subjects: "list[str]") -> dict:
    """Each subject's age when ``--dpf age`` asks for it, refusing a subject without one."""
    from nirspipe.io.bids import get_participant_age
    from nirspipe.pipeline.dpf import dpf_from_age

    try:
        if not dpf_from_age(dpf):
            return {}
        return {s: get_participant_age(bids_dir, s) for s in subjects}
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


def _add_dpf_and_cardiac(parser) -> None:
    """The optics and cardiac band both raw reports need, required: they are population values."""
    _app_options.add_dpf(parser, required=True)
    _app_options.add_template(parser)
    parser.add_argument("--cardiac-l-freq", type=float, required=True,
                        help="Lower bound of cardiac band in Hz (required; population-dependent).")
    parser.add_argument("--cardiac-h-freq", type=float, required=True,
                        help="Upper bound of cardiac band in Hz (required; population-dependent).")


def _add_ignore(parser) -> None:
    parser.add_argument("--ignore", nargs="+", action="extend", choices=["channel-status"],
                        help="Aspects to skip. 'channel-status': do not mark bad the channels each "
                             "recording's BIDS channels.tsv marks bad in its status column. "
                             "Pass it here when the nirspipe run being checked passed it.")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-qc",
        description="fNIRS quality control: individual, hyperscanning and cohort-level reports.",
    )
    p.add_argument("--version", action="version", version=f"nirspipe-qc {__version__}")
    sub = p.add_subparsers(required=True)

    pr = sub.add_parser("prep-raw", parents=[_app_options.screening()],
                        help="Static raw QC report for a single participant.")
    pr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    pr.add_argument("output_dir", type=Path, help="nirspipe derivatives directory")
    pr.add_argument("--participant-label", "--participant_label", nargs="+", action="extend", required=True,
                    type=_shared.BidsLabel,
                    help="Subject ID(s) to inspect, e.g. '01'. One report set per subject, "
                         "and one subject's failure does not stop the rest.")
    pr.add_argument("--session-label", "--session_label", nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Session label(s) to include.")
    pr.add_argument("--task-label", "--task_label",    nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Task label(s) to include.")
    pr.add_argument("--run-label", "--run_label", nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Run label(s) to include. Each chosen run gets a report of its own, named "
                         "with its run, so the report holding all of a task's runs is kept.")
    _add_dpf_and_cardiac(pr)
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
    _app_options.add_separation_bands(pr)
    pr.add_argument("--by-condition", action="store_true",
                    help="Also write one report per annotated condition, beside the run's "
                         "own, as <run>_cond-<condition>_desc-raw_report.html. Their numbers "
                         "are sliced out "
                         "of the run's windowed pass, on the same window grid and filter as "
                         "the run. Each page's screening verdict is that condition's; the "
                         "run's own page carries the run's.")
    pr.add_argument("--motion-correction",
                    choices=["tddr", "wavelet", "tpca", "spline", "spline_sg", "wavelet_tddr",
                             "spline_wavelet", "none"],
                    default="none",
                    help="Run this correction on a copy of the optical density and report the "
                         "recording before and after it. Nothing is written back and nothing "
                         "else runs, so screening and coupling still describe the recording as "
                         "delivered. Default none: report it uncorrected.")
    _app_options.add_motion_settings(pr)
    _shared.add_skip_bids_validation(pr)
    _shared.add_nprocs(pr)
    _add_ignore(pr)
    pr.add_argument("--roi-mapping", type=Path, default=None,
                    help="JSON file mapping ROI labels to lists of channel names. Groups the "
                         "motion carpet's rows by ROI. Optional.")
    pr.set_defaults(func=cmd_prep_raw)

    hr = sub.add_parser("hyper-raw",
                        parents=[_app_options.pairs_selection(),
                                 _app_options.screening(),
                                 _app_options.alignment_window()],
                        help="Hyperscanning raw QC report from BIDS raw data.")
    hr.add_argument("bids_dir",   type=Path, help="BIDS dataset root")
    hr.add_argument("output_dir", type=Path,
                    help="The tree the dyad reports go to: a raw QC tree of their own, or the "
                         "nirspipe-hyper tree.")
    hr.add_argument("analysis_level", choices=["group"],
                    help="Always `group`: every metric here needs both members present.")
    _add_dpf_and_cardiac(hr)
    _app_options.add_separation_bands(hr)
    hr.add_argument("--session-label", "--session_label", nargs="+", action="extend", type=_shared.BidsLabel,
                    help="Session label(s) to process: a group's occasion where the group "
                         "CSV has one, else its session. A group-CSV row without either is "
                         "read once per listed session.")
    _shared.add_skip_bids_validation(hr)
    _add_ignore(hr)
    _shared.add_nprocs(hr, unit="group")
    hr.add_argument("--derivatives-dir", "--derivatives_dir", type=Path, default=None,
                    help="The nirspipe tree holding each member's sub-<id>/nirs/ stages. The "
                         "motion panel draws its after-correction side from desc-motcorrected "
                         "there; without it, only the recordings before correction.")
    hr.add_argument("--roi-mapping", type=Path, default=None,
                    help="JSON file mapping ROI labels to lists of channel names. Groups each "
                         "member's motion carpet rows by ROI. Optional.")
    hr.set_defaults(func=cmd_hyper_raw)

    gr = sub.add_parser(
        "cohort",
        help="Every subject in a tree on one page.",
        description="Aggregates the quality record of every run under OUTPUT_DIR. A run processed by both `nirspipe` and `prep-raw` has two records and the pipeline one is used, so the page carries every stage the run was measured at.")
    gr.add_argument("output_dir", type=Path,
                    help="nirspipe derivatives directory (contains sub-*/nirs/ SQM JSONs)")
    gr.set_defaults(func=cmd_group_raw)

    ghr = sub.add_parser(
        "cohort-hyper",
        help="Cohort report over every dyad: shared usable time, where it went, and each "
             "window against its own null.",
        description="One page for every dyad in a tree, built from the records and tables "
                    "the dyad runs already wrote. It reports what a dyad has and a subject "
                    "cannot; the per-subject quality distributions stay in `cohort`.")
    ghr.add_argument("output_dir", type=Path,
                     help="nirspipe derivatives directory (contains group-*/nirs/ SQM JSONs)")
    ghr.set_defaults(func=cmd_group_hyper_raw)

    pv = sub.add_parser("provenance", help="Render the file provenance graph from existing sidecars.")
    pv.add_argument("output_dir", type=Path, help="nirspipe derivatives directory")
    pv.set_defaults(func=cmd_provenance)


    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    # analysis_level only gives hyper-raw the BIDS App shape; `group` is its one value
    kw = {k: v for k, v in vars(args).items() if k not in ("func", "analysis_level")}
    args.func(**kw)
