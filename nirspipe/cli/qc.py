"""nirspipe-qc CLI (argparse, BIDS App convention): raw quality control for fNIRS data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

from nirspipe import __version__

from nirspipe.io.derivatives import group_label, group_report_dir
from nirspipe.io.head_template import space_of
from nirspipe.io.optode_space import hold_placement, place_optodes
from nirspipe.io.naming import report_name
from nirspipe.cli import _app_options
from nirspipe.cli import _shared
from nirspipe.utils.logging import get_logger, setup_logging
from nirspipe.cli._app_options import separation_bands_from_args
from nirspipe.qc.metrics._helpers import separation_bands

setup_logging()

logger = get_logger("cli.qc")

# analysis levels and the flags each one cannot run without
_LEVEL_REQUIRES = {
    "participant": ("--dpf", "--sci-threshold", "--cardiac-l-freq", "--cardiac-h-freq"),
    "group":       (),
}
# a parser group's title names the level its flags belong to; one typed at the other level
# is refused rather than silently unused
_LEVEL_GROUPS = {"participant": "participant level", "group": "group level"}
# the selection a --config file's [execution] sets, beside its [qc] settings
_EXECUTION_KEYS = ("participant_label", "session_label", "task_label")


def cmd_participant(
    bids_dir: Path, output_dir: Path, participant_label: list[str] | None,
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
    bad_channels: str | None = None,
    keep_spans: Path | None = None,
    keep_spans_from: str | None = None,
    bad_spans: Path | None = None,
    bids_filter_file: Path | None = None,
    population: str | None = None,
    from_population: "list[str] | None" = None,
    record_settings: "dict | None" = None,
) -> None:
    """One raw QC report per recording, each subject on its own; ``record_settings`` goes in its run record."""
    from nirspipe.cli.workflows import (
        _bad_channels_for, _keep_channels_for, _refuse_bad_span_rows,
        _refuse_unmatched_bad_channel_rows, _spans_for,
    )

    _app_options.check_motion_settings(motion_correction, motion_std_thresh, wavelet_iqr)
    _app_options.check_keep_spans_from(keep_spans, keep_spans_from)
    _shared.refuse_output_in_input(bids_dir, output_dir, "nirspipe-qc")
    if not skip_bids_validation:
        from nirspipe.io.bids import validate_bids
        validate_bids(bids_dir)
    if (epoch_tmin is None) != (epoch_tmax is None):
        print("Error: --epoch-tmin and --epoch-tmax must be given together.", file=sys.stderr)
        raise SystemExit(1)

    from nirspipe.cli.run_record import write_raw_qc_record
    from nirspipe.io.bids import bids_label, channel_status_bads, get_layout, get_nirs_files
    from nirspipe.io.derivatives import (
        LINK_RAW, entity_of, subject_report_dir, write_bidsignore, write_dataset_description,
    )
    from nirspipe.qc.subject.prep_raw_report import build_prep_raw_report

    layout = get_layout(bids_dir)
    # as the pipeline reads it: every subject when none is named, a named one that is absent refused
    if not participant_label:
        participant_label = layout.get_subjects()
    else:
        missing = sorted(set(participant_label) - set(layout.get_subjects()))
        if missing:
            print(f"Error: participant label(s) not in {bids_dir}: {', '.join(missing)}",
                  file=sys.stderr)
            raise SystemExit(1)
    # the records it writes name the raw recordings, which resolve through this link
    write_dataset_description(output_dir, source=bids_dir, link=LINK_RAW)
    write_bidsignore(output_dir)
    sep_bands = separation_bands(type("Bands", (), separation_bands_from_args({
        "short_max_dist": short_max_dist, "long_min_dist": long_min_dist,
        "long_max_dist": long_max_dist,
    })))
    roi_map = _app_options.load_roi_mapping(roi_mapping)
    # the same refusals the pipeline makes, before any subject runs
    _refuse_unmatched_bad_channel_rows(bad_channels, layout)
    _refuse_bad_span_rows(keep_spans, layout, marker=keep_spans_from)
    _refuse_bad_span_rows(bad_spans, layout, flag="--bad-spans")
    sessions = session_label or [None]
    tasks    = task_label    or [None]
    ages = _app_options.participant_ages(dpf, from_population, bids_dir, participant_label)
    templates = _app_options.templates_for(template, participant_label, ages)
    batch_age = _app_options.population_age(population, from_population)

    def one_subject(subject: str) -> int:
        """Every report this subject's runs produce, plus the index over them."""
        all_runs: list[dict] = []
        groups: dict[tuple, list[dict]] = defaultdict(list)
        if record_settings is not None:
            write_raw_qc_record(record_settings["args"], record_settings["keys"], subject,
                                output_dir)

        for session in sessions:
            for task in tasks:
                files = get_nirs_files(layout, subject=subject, session=session, task=task,
                                       filter_file=bids_filter_file)
                for f in files:
                    run = entity_of(Path(f), "run")
                    if run_label and run not in run_label:
                        continue
                    entities = layout.parse_file_entities(str(f))
                    actual_ses = entities.get("session")
                    actual_task = entities.get("task")
                    label = bids_label(subject, entities)
                    keep = _spans_for(keep_spans, subject, entities,
                                      relative=bool(keep_spans_from))

                    snirf_p = Path(f)
                    run_dict = {
                        "label":       label,
                        "subject_id":  subject,
                        "snirf_path":  str(f),
                        "session":     actual_ses,
                        "task":        actual_task,
                        "channels_tsv_bads": ([] if "channel-status" in (ignore or [])
                                              else channel_status_bads(layout, snirf_p)),
                        "ignore": list(ignore or []),
                        "bad_channels": _bad_channels_for(bad_channels, subject, entities),
                        "keep_channels": _keep_channels_for(bad_channels, subject, entities),
                        "keep_spans": keep,
                        # a recording no row names is used whole, so no marker applies to it
                        "keep_spans_from": keep_spans_from if keep else None,
                        "bad_spans": _spans_for(bad_spans, subject, entities,
                                                flag="--bad-spans"),
                        "age": batch_age if batch_age is not None else ages.get(subject),
                        "template": templates[subject],
                        "population": population,
                        "from_population": list(from_population or []),
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


def _corrected(raws: dict, params: dict) -> dict:
    """Each member's recording through the motion correction its raw QC record names, as optical density."""
    from nirspipe.pipeline.motion import correct_motion
    from nirspipe.qc.metrics._helpers import as_optical_density

    out = {}
    for sid, raw in raws.items():
        method = params[sid].get("motion_correction") or "none"
        if method == "none":
            continue
        try:
            out[sid] = correct_motion(as_optical_density(raw), method=method,
                                      std_thresh=params[sid].get("motion_std_thresh"),
                                      iqr_factor=params[sid].get("wavelet_iqr"))
        except Exception:
            logger.warning("%s: %s could not be run; the motion panel shows the recording "
                           "before correction only", sid, method, exc_info=True)
    return out


def _raw_one_group(s: SimpleNamespace, gid, task, ses, members):
    """One dyad of the group level: each member's raw QC record, the aligned recordings, the page."""
    from nirspipe.io.auxiliary import imu_traces, read_aux_snirf
    from nirspipe.pipeline.hyper.alignment import (
        align_imu_like,
        align_like,
        aligned_offsets,
        align_recordings,
        crop_aligned_window,
        normalize_raws,
        trim_to_shortest,
    )
    from nirspipe.pipeline.hyper.group_io import _raw_to_haemo, load_group_raw_bids
    from nirspipe.pipeline.dpf import dpf_caveat, wavelengths
    from nirspipe.pipeline.hyper.group_quality import load_group_sqm_raw, member_raw_record
    from nirspipe.qc.hyper.hyper_report import build_hyper_report
    from nirspipe.qc.metrics._helpers import bands_from_record
    from nirspipe.utils.lineage import path_from
    from nirspipe.utils.spans import mark_manual, mark_unselected

    raws_cw = load_group_raw_bids(s.bids_dir, members)
    records = {e.subject_id: member_raw_record(s.output_dir, e, raws_cw[e.subject_id])
               for e in members}
    params = {sid: record["parameters"] for sid, (_path, record) in records.items()}
    # each member on its own head and with its own reviewer stretches, as its record says
    for sid, raw in raws_cw.items():
        hold_placement(raw, place_optodes(path_from(raw), space_of(params[sid]["template"])))
        mark_manual(mark_unselected(raw, params[sid].get("keep_spans") or []),
                    params[sid].get("bad_spans") or [])
    # the aux group rides only in the source file, on each member's own clock. A file
    # that will not read costs the IMU rows and not the report
    try:
        member_imu = {sid: imu_traces(*read_aux_snirf(path_from(raw)))
                      for sid, raw in raws_cw.items() if path_from(raw)}
    except Exception:
        logger.warning("%s: IMU could not be read; the report has no IMU rows",
                       gid, exc_info=True)
        member_imu = {}
    sqm_data = load_group_sqm_raw(s.output_dir, members, raws_cw, records, session=ses)
    # each member converted with the factors its own raw QC used
    raws_haemo = {sid: _raw_to_haemo(r, params[sid]["dpf"]) for sid, r in raws_cw.items()}
    # the dyad page says it the way it says a member screening could not judge
    for sid, raw in raws_cw.items():
        age = params[sid].get("dpf_age")
        if age is not None and (caveat := dpf_caveat(age, wavelengths(raw))):
            sqm_data[sid]["dpf_caveat"] = caveat
    bands = {sid: bands_from_record(record.get("raw") or {})
             for sid, (_path, record) in records.items()}
    sep_bands = bands[members[0].subject_id]
    if len(set(bands.values())) > 1:
        logger.warning("%s: the members' records split channels by different separation bands; "
                       "the dyad page uses %s's", gid, members[0].subject_id)
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
    # than aligned a second time; its corrected side is each member's own correction
    intensity_raws = align_like(raws_cw, aligned_raws)
    full_after = _corrected(raws_cw, params)
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
        sep_bands=sep_bands,
        session=ses,
        roi_map=s.roi_map,
    )


def _dyads(bids_dir: Path, output_dir: Path, pairs_csv: Path, group_id: str | None,
           participant_label: list[str] | None, task_label: list[str] | None,
           session_label: list[str] | None, normalize: bool, no_align: bool,
           tstart: float | None, tend: float | None, roi_mapping: Path | None,
           nprocs: int | None) -> bool:
    """Every selected dyad's raw page, read from its members' records; False when one failed."""
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
    settings = SimpleNamespace(
        bids_dir=bids_dir, output_dir=output_dir, no_align=no_align, normalize=normalize,
        tstart=tstart, tend=tend, roi_map=_app_options.load_roi_mapping(roi_mapping))
    done = _run_groups(groups, partial(_raw_one_group, settings), nprocs,
                       log_file=lambda key: group_report_dir(output_dir, key[0]) / "logs"
                       / f"{group_label(*key)}_desc-raw.log", exit_on_failure=False)
    return len(done) == len(groups)


def cmd_group(
    bids_dir: Path, output_dir: Path, participant_label: list[str] | None,
    session_label: list[str] | None, task_label: list[str] | None,
    skip_bids_validation: bool, roi_mapping: Path | None = None, nprocs: int | None = None,
    pairs_csv: Path | None = None, group_id: str | None = None, normalize: bool = False,
    no_align: bool = False, tstart: float | None = None, tend: float | None = None,
) -> None:
    """The cohort page over every subject in the tree, and given ``pairs_csv`` each dyad's page first."""
    from nirspipe.qc.hyper.group_hyper_writer import build_group_hyper_report
    from nirspipe.qc.subject.group_writer import build_group_raw_report

    _shared.refuse_output_in_input(bids_dir, output_dir, "nirspipe-qc")
    if group_id is not None and pairs_csv is None:
        print("Error: --group-id selects from --pairs-csv, which was not given.", file=sys.stderr)
        raise SystemExit(1)
    dyads_ok = True
    if pairs_csv is not None:
        if not skip_bids_validation:
            from nirspipe.io.bids import validate_bids
            validate_bids(bids_dir)
        from nirspipe.io.derivatives import LINK_RAW, write_dataset_description
        # refuses a BIDS dataset other than the one the participant level linked
        write_dataset_description(output_dir, source=bids_dir, link=LINK_RAW)
        dyads_ok = _dyads(bids_dir, output_dir, pairs_csv, group_id, participant_label,
                          task_label, session_label, normalize, no_align, tstart, tend,
                          roi_mapping, nprocs)

    path = build_group_raw_report(output_dir)
    if path is None:
        print(f"Error: no raw QC records under {output_dir}; run the participant level into "
              "it first.", file=sys.stderr)
        raise SystemExit(1)
    print(f"report -> {path}")
    # the dyad cohort page wherever the tree holds dyads, this run's or an earlier one's
    if any(output_dir.glob("group-*")):
        dyad_path = build_group_hyper_report(output_dir)
        if dyad_path is not None:
            print(f"report -> {dyad_path}")
    if not dyads_ok:
        raise SystemExit(1)


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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-qc",
        description="fNIRS raw quality control as a BIDS App: one report per recording at "
                    "participant level, then the cohort pages and, given a pairs table, each "
                    "dyad's page at group level, all in one tree.",
    )
    p.add_argument("--version", action="version", version=f"nirspipe-qc {__version__}")
    p.add_argument("bids_dir", type=Path, help="BIDS dataset root, read only.")
    p.add_argument("output_dir", type=Path,
                   help="The raw QC tree: each subject's reports under sub-<label>/, each "
                        "dyad's under group-<id>/, the cohort pages at the root.")
    p.add_argument("analysis_level", choices=list(_LEVEL_REQUIRES),
                   help="participant: screen and report each recording, subjects independently "
                        "and in parallel. group: after every participant run, the cohort page "
                        "and, with --pairs-csv, each dyad's page read from its members' "
                        "records, so a member keeps its own lines, band and verdict.")

    sel = p.add_argument_group("selection (both levels)")
    sel.add_argument("--participant-label", "--participant_label", nargs="+", action="extend",
                     type=_shared.BidsLabel,
                     help="Subject ID(s), e.g. '01'. Participant level: the subjects to report, "
                          "every one when omitted, one's failure not stopping the rest. Group "
                          "level: only the dyads with one of them as a member.")
    sel.add_argument("--session-label", "--session_label", nargs="+", action="extend",
                     type=_shared.BidsLabel,
                     help="Session label(s). At group level a group's occasion where the pairs "
                          "table has one, else its session; a row with neither is read once "
                          "per listed session.")
    sel.add_argument("--task-label", "--task_label", nargs="+", action="extend",
                     type=_shared.BidsLabel, help="Task label(s) to include.")
    sel.add_argument("--roi-mapping", type=Path, default=None,
                     help="JSON file mapping ROI labels to lists of channel names. Groups the "
                          "motion carpet's rows by ROI. Optional.")
    _shared.add_skip_bids_validation(sel)
    _shared.add_nprocs(sel, unit="job",
                       note="A job is a subject at participant level, a dyad at group level.")

    part = p.add_argument_group(_LEVEL_GROUPS["participant"])
    part.add_argument("--run-label", "--run_label", nargs="+", action="extend",
                      type=_shared.BidsLabel,
                      help="Run label(s) to include. Each chosen run gets a report of its own, "
                           "named with its run, so the report holding all of a task's runs is "
                           "kept.")
    _app_options.add_bids_filter_file(part)
    part.add_argument("--config", type=Path,
                      help="A raw QC run record (sub-<label>/logs/*_desc-raw.toml) or a TOML of "
                           "its shape: [qc] keyed by flag name. Reruns with its settings; flags "
                           "given here override it.")
    _app_options.add_dpf(part)
    _app_options.add_template(part)
    _app_options.add_population(part, "the two cardiac band edges")
    _app_options.add_sci_threshold(part)
    _app_options.add_psp_threshold(part)
    _app_options.add_min_good_frac(part)
    _app_options.add_screen_scope(part)
    part.add_argument("--cardiac-l-freq", type=float,
                      help="Lower bound of cardiac band in Hz (required at participant level "
                           "unless --population sets it).")
    part.add_argument("--cardiac-h-freq", type=float,
                      help="Upper bound of cardiac band in Hz (required at participant level "
                           "unless --population sets it).")
    part.add_argument("--window-length", type=float, default=10.0,
                      help="Sliding-window length (s) for windowed SCI/PSP/GVTD series.")
    part.add_argument("--epoch-qc", action="store_true",
                      help="Add a per-trial section: score each event window on its own and "
                           "show them as a trial x metric heatmap.")
    part.add_argument("--epoch-tmin", type=float, default=None,
                      help="Trial window start relative to event onset in s; negative pulls in "
                           "a baseline. Omit to use each event's own duration.")
    part.add_argument("--epoch-tmax", type=float, default=None,
                      help="Trial window end relative to event onset in s. Given together with "
                           "--epoch-tmin, or neither.")
    _app_options.add_separation_bands(part)
    part.add_argument("--by-condition", action="store_true",
                      help="Also write one report per annotated condition, beside the run's "
                           "own, as <run>_cond-<condition>_desc-raw_report.html. Their numbers "
                           "are sliced out of the run's windowed pass, on the same window grid "
                           "and filter as the run. Each page's screening verdict is that "
                           "condition's; the run's own page carries the run's.")
    part.add_argument("--motion-correction", choices=_app_options.MOTION_CHOICES,
                      default="tddr",
                      help="Run this correction on a copy of the optical density and report the "
                           "recording before and after it. Nothing is written back and nothing "
                           "else runs, so screening and coupling still describe the recording "
                           "as delivered. Default tddr, as nirspipe; none reports it "
                           "uncorrected only. The group level's motion panel shows each "
                           "member's.")
    _app_options.add_motion_settings(part)
    _app_options.add_reviewer_marks(part)
    part.add_argument("--ignore", nargs="+", action="extend", choices=["channel-status"],
                      help="Aspects to skip. 'channel-status': do not mark bad the channels "
                           "each recording's BIDS channels.tsv marks bad in its status column. "
                           "Pass it here when the nirspipe run being checked passed it.")

    grp = p.add_argument_group(_LEVEL_GROUPS["group"])
    grp.add_argument("--pairs-csv", type=Path,
                     help="CSV with columns: group_id, subject_id, task, and optionally "
                          "session, run and occasion. Each unique (group_id, task, session) is "
                          "one dyad page, read from its members' records; an occasion column "
                          "joins members whose session labels differ and names the group's "
                          "outputs ses-<occasion>. Omit for the cohort page alone.")
    grp.add_argument("--group-id", default=None, type=_shared.BidsLabel,
                     help="Process only this group_id of --pairs-csv.")
    _app_options.add_alignment_window(grp)
    return p


def _provenance_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nirspipe-provenance",
        description="Render the file provenance graph of every subject and dyad in a tree, "
                    "from the sidecars already on disk.")
    p.add_argument("--version", action="version", version=f"nirspipe-provenance {__version__}")
    p.add_argument("output_dir", type=Path, help="Any tree this package wrote.")
    return p


def _level_dests(parser: argparse.ArgumentParser, level: str) -> set[str]:
    return {a.dest for g in parser._action_groups if g.title == _LEVEL_GROUPS[level]
            for a in g._group_actions}


def dispatched(parser: argparse.ArgumentParser, level: str) -> set[str]:
    """The dests one level hands its command: the positionals, the shared selection, its own flags."""
    shared = {a.dest for g in parser._action_groups
              if g.title not in _LEVEL_GROUPS.values() for a in g._group_actions}
    return (shared | _level_dests(parser, level)) - {"help", "version", "config", "analysis_level"}


def read_config(path: Path) -> dict:
    """--config as ``{"qc": settings, "execution": selection, "version": writer}``; unknown keys refused."""
    from nirspipe.utils import load_toml

    raw = load_toml(path)
    stray = [k for k, v in raw.items() if not isinstance(v, dict)]
    if stray:
        print(f"Error: {path}: {stray[0]} sits outside any section; --config takes a [qc] "
              "section keyed by flag name, as a raw QC run record does.", file=sys.stderr)
        raise SystemExit(1)
    unknown = sorted(set(raw) - {"qc", "qc_sources", "execution", "environment"})
    if unknown:
        print(f"Error: {path}: no section [{unknown[0]}]; expected [qc] or [execution].",
              file=sys.stderr)
        raise SystemExit(1)
    parser = _build_parser()
    actions = {a.dest: a for a in parser._actions}
    allowed = _level_dests(parser, "participant") - {"config"}
    settings = raw.get("qc", {})
    bad = [k for k in settings if k not in allowed]
    if bad:
        print(f"Error: {path}: [qc] has no setting {bad[0]!r}.", file=sys.stderr)
        raise SystemExit(1)
    execution = {k: v for k, v in raw.get("execution", {}).items() if k in _EXECUTION_KEYS}
    return {"version": raw.get("environment", {}).get("nirspipe_version"),
            "qc": {k: _shared.coerce_setting(actions[k], k, v, path) for k, v in settings.items()},
            "execution": {k: _shared.coerce_setting(actions[k], k, v, path)
                          for k, v in execution.items()}}


def _apply_config(args: argparse.Namespace, argv: "list[str] | None") -> None:
    """Fill what the command line left out from --config."""
    config = read_config(args.config)
    if config["version"] and config["version"] != __version__:
        print(f"Warning: {args.config} was written by nirspipe {config['version']} and this is "
              f"{__version__}; a setting whose meaning or default changed in between runs as "
              "this version reads it.", file=sys.stderr)
    given = _shared.typed_flags(_build_parser(), argv)
    for key, value in {**config["qc"], **config["execution"]}.items():
        if key not in given:
            setattr(args, key, value)


def _refuse_other_level(args: argparse.Namespace, argv: "list[str] | None") -> None:
    """Exit non-zero naming a flag typed at the level it does not belong to."""
    parser = _build_parser()
    other = "group" if args.analysis_level == "participant" else "participant"
    typed = _shared.typed_flags(_build_parser(), argv) & _level_dests(parser, other)
    if typed:
        flag = next(a.option_strings[0] for a in parser._actions if a.dest in typed)
        why = (" The group level reads each member's settings from its raw QC record."
               if other == "participant" else "")
        print(f"Error: {flag} is a {other}-level setting, and this is the "
              f"{args.analysis_level} level.{why}", file=sys.stderr)
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _refuse_other_level(args, argv)
    if args.analysis_level == "participant" and args.config is not None:
        _apply_config(args, argv)
    filled = []
    if args.analysis_level == "participant":
        # below the command line and --config, so either overrides any value it fills
        filled = _app_options.fill_population(args, ("cardiac_l_freq", "cardiac_h_freq"))
    for flag in _LEVEL_REQUIRES[args.analysis_level]:
        if getattr(args, flag.lstrip("-").replace("-", "_")) is None:
            hint = ("; --population sets it" if flag in ("--dpf", "--cardiac-l-freq",
                                                          "--cardiac-h-freq") else "")
            print(f"Error: Missing option '{flag}' (required at {args.analysis_level} "
                  f"level{hint}).", file=sys.stderr)
            raise SystemExit(1)
    kw = vars(args)
    taken = {k: kw[k] for k in dispatched(parser, args.analysis_level)}
    if args.analysis_level == "participant":
        keys = tuple(sorted(_level_dests(parser, "participant") - {"config"}))
        cmd_participant(**taken, from_population=filled,
                        record_settings={"args": {**kw, "from_population": filled}, "keys": keys})
    else:
        cmd_group(**taken)


def main_provenance(argv: list[str] | None = None) -> None:
    cmd_provenance(_provenance_parser().parse_args(argv).output_dir)
