"""Execution layer between the CLI and the processing pipeline.

Resolves which subjects/sessions/tasks to run, merges CLI args with TOML config
into PrepConfig/PostConfig, and drives the per-subject prep → post → report loop.
Does no signal processing itself.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fnirs_pipe.io.bids import bids_label, get_layout, get_nirs_files
from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.io.snirf import read_snirf
import mne
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.utils import unwrap_enum as _v
from fnirs_pipe.utils import job_db as _jdb
from fnirs_pipe.utils.logging import get_logger, setup_logging
from fnirs_pipe.utils.run_record import write_run_record
from fnirs_pipe.utils.run_script import write_run_script

logger = get_logger("cli.workflows")

def _build_post_config(subject: str, session: str | None, args: dict[str, Any], toml: dict[str, Any], roi_map: dict | None = None) -> Any:
    from fnirs_pipe.pipeline.post_pipeline import PostConfig

    # CLI takes priority over TOML; Typer Enum values are unwrapped to plain strings.
    def pick(cli_key: str, toml_key: str | None = None, default: Any = None) -> Any:
        cli_val = args.get(cli_key)
        if cli_val is not None:
            return _v(cli_val)
        toml_val = toml.get(toml_key or cli_key)
        if toml_val is not None:
            return toml_val
        return default

    ep = pick("events_path")
    sc = pick("short_channel")
    raw_fir = pick("fir_delays")  # CLI passes a comma-separated string, e.g. "0,1,2"

    # contrast_def can live in a separate TOML file referenced by --contrast-file or config key.
    contrast_def = None
    contrast_file = args.get("contrast_file") or (Path(toml["contrast_file"]) if "contrast_file" in toml else None)
    if contrast_file:
        from fnirs_pipe.utils import load_toml
        contrast_def = load_toml(contrast_file)

    return PostConfig(
        subject=subject,
        cardiac_l_freq=pick("cardiac_l_freq"),
        cardiac_h_freq=pick("cardiac_h_freq"),
        resp_l_freq=pick("resp_l_freq"),
        resp_h_freq=pick("resp_h_freq"),
        session=session,
        dry_run=args.get("dry_run", False),
        high_pass=pick("high_pass"),
        low_pass=pick("low_pass"),
        resample_sfreq=pick("resample_sfreq"),
        stim_dur=pick("stim_dur"),
        hrf_model=pick("hrf_model"),
        noise_model=pick("noise_model"),
        drift_model=pick("drift_model"),
        drift_high_pass=pick("drift_high_pass"),
        drift_order=pick("drift_order"),
        fir_delays=tuple(int(x) for x in raw_fir.split(",")) if raw_fir else None,
        short_channel=sc if (sc and sc != "none") else None,
        events_path=str(ep) if ep else None,
        contrast_def=contrast_def,
        combine_runs=pick("combine_runs"),
        roi_map=roi_map,
    )


def run_participant_level(args: dict[str, Any]) -> None:
    bids_dir: Path             = args["bids_dir"]
    output_dir: Path           = args["output_dir"]
    participant_label: list[str] | None = args.get("participant_label")
    session_label: list[str] | None  = args.get("session_label")
    task_label: list[str] | None     = args.get("task_label")
    bids_filter_file: Path | None    = args.get("bids_filter_file")
    work_dir: Path | None            = args.get("work_dir")
    verbose: bool                    = args.get("verbose", False)

    setup_logging(verbose=verbose)
    logger.info("fnirs-pipe starting — output: %s", output_dir)

    skip_validation = (
        args.get("skip_bids_validation", False)
        or "bids-validation" in (args.get("ignore") or [])
    )
    layout = get_layout(bids_dir, validate=not skip_validation)
    # If no participant label provided, run on all subjects in BIDS dir. 
    # Write dataset_description to output for provenance.
    if not participant_label:
        participant_label = layout.get_subjects()
    write_dataset_description(output_dir)

    toml: dict[str, Any] = {}
    if args.get("config"):
        from fnirs_pipe.utils import load_toml
        toml = load_toml(args["config"])
        logger.debug("loaded post config: %s", args["config"])

    # cutoffs may come from CLI or TOML; resolve like PostConfig so DB log + report match what post applies
    cfg_high_pass = args.get("high_pass") if args.get("high_pass") is not None else toml.get("high_pass")
    cfg_low_pass  = args.get("low_pass")  if args.get("low_pass")  is not None else toml.get("low_pass")

    roi_map = None
    roi_mapping = args.get("roi_mapping")
    if roi_mapping is not None:
        import json
        try:
            roi_map = json.loads(Path(roi_mapping).read_text())
        except Exception as exc:
            logger.warning("failed to load ROI mapping %s: %s", roi_mapping, exc)

    tasks: list[str | None] = task_label if task_label else [None]

    from fnirs_pipe import __version__
    db_path = output_dir / "logs" / "fnirs_pipe.db"
    execution_id = _jdb.log_execution(
        db_path=db_path,
        command_line=" ".join(sys.argv),
        fnirs_pipe_version=__version__,
        input_dir=str(bids_dir),
        output_dir=str(output_dir),
        subjects=participant_label,
        work_dir=str(work_dir) if work_dir else None,
        session_labels=session_label,
        task_labels=task_label,
        mode=_v(args["mode"]) if args.get("mode") else None,
        dry_run=args.get("dry_run", False),
    )

    try:
        for subject in participant_label:
            sub_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub_dir = output_dir / f"sub-{subject}"
            log_file = sub_dir / "logs" / f"sub-{subject}.log"
            setup_logging(verbose=verbose, log_file=log_file)
            logger.info("sub-%s | starting", subject)
            # built here purely to record what will be used; the loops below build their own
            # per-session copies. Resolving now keeps the record even if the run then fails.
            write_run_record(
                args, subject, sub_timestamp, output_dir, sub_dir=sub_dir,
                prep_config=_make_prep_config(subject, None, args),
                post_config=(_build_post_config(subject, None, args, toml, roi_map=roi_map)
                             if args.get("mode") else None),
            )
            write_run_script(args, subject, sub_timestamp, output_dir, sub_dir=sub_dir)

            _jdb.log_run_start(
                db_path, execution_id, subject,
                sci_threshold=args["sci_threshold"],
                dpf=args["dpf"],
                motion_correction=_v(args["motion_correction"]),
                mode=_v(args["mode"]) if args.get("mode") else None,
                high_pass=cfg_high_pass,
                low_pass=cfg_low_pass,
                hrf_model=_v(args["hrf_model"]) if args.get("hrf_model") else toml.get("hrf_model"),
            )

            t0 = time.monotonic()
            subject_status = "SUCCESS"
            subject_error: str | None = None
            last_raw = last_result = last_label = None
            prep_config = None
            try:
                sessions: list[str | None] = session_label if session_label else [None]

                for session in sessions:
                    for task in tasks:
                        files = get_nirs_files(
                            layout, subject=subject, session=session,
                            task=task, filter_file=bids_filter_file,
                        )

                        if not files:
                            label = f"sub-{subject}" + (f" ses-{session}" if session else "") + (f" task-{task}" if task else "")
                            logger.warning("no snirf files found for %s, skipping", label)
                            continue

                        for snirf_path in files:
                            src_entities = layout.parse_file_entities(str(snirf_path))
                            prep_config = _make_prep_config(subject, session, args)
                            logger.info("processing: %s", snirf_path)
                            try:
                                raw = read_snirf(snirf_path)
                                result = run_prep(raw, prep_config, output_dir=output_dir, source_entities=src_entities, work_dir=work_dir, source_path=snirf_path)
                                logger.info("finished prep: %s", snirf_path.name)
                                last_raw, last_result = raw, result
                                last_label = bids_label(subject, src_entities)
                            except Exception:
                                logger.exception("prep failed for %s", snirf_path)
                                raise

                glm_est = dm = alff_df = fc_df = fc_hbr_df = last_denoised = gcor_reg = None
                if args.get("mode") is not None:
                    glm_est, dm, alff_df, fc_df, fc_hbr_df, last_denoised, gcor_reg = _run_post_for_subject(subject, sessions, args, toml, output_dir, roi_map=roi_map)

                # one SQM record per run, written once both passes have finished so the
                # final section can measure the last file post actually produced. The
                # database takes one row per section, which is what its checkpoint column
                # has always been for.
                try:
                    import json as _json
                    from fnirs_pipe.qc.sqm_record import SECTIONS, build_sqm_records, entities_of
                    for path in build_sqm_records(sub_dir / "nirs"):
                        logger.info("sub-%s | SQM record → %s", subject, path.name)
                        record = _json.loads(path.read_text(encoding="utf-8"))
                        ents = entities_of(path.stem)
                        for section in SECTIONS:
                            if record.get(section):
                                _jdb.log_sqm(db_path, execution_id, subject, section,
                                             record[section], session=ents["ses"],
                                             bids_task=ents["task"])
                except Exception:
                    logger.warning("sub-%s | SQM records failed", subject, exc_info=True)

                # rendered before the report, which embeds it: every sidecar it scans is
                # on disk by now, and --no-report still leaves the diagram behind
                provenance_path = None
                try:
                    from fnirs_pipe.qc.provenance import write_provenance
                    for path in write_provenance(
                        sub_dir / "nirs", sub_dir / "figures",
                        stem="provenance",
                        title=f"sub-{subject}" + (f"  |  mode: {args['mode']}" if args.get("mode") else ""),
                    ):
                        logger.info("sub-%s | provenance → %s", subject, path)
                        if path.suffix == ".png":
                            provenance_path = f"figures/{path.name}"
                except Exception:
                    logger.warning("sub-%s | provenance graph failed", subject, exc_info=True)

                if not args.get("no_report") and last_result is not None:
                    _emit_subject_report(subject, sub_dir, last_raw, last_result, prep_config, args, glm_est, dm, alff_df=alff_df, fc_df=fc_df, fc_hbr_df=fc_hbr_df, high_pass=cfg_high_pass, low_pass=cfg_low_pass, after_haemo=last_denoised, gcor_reg=gcor_reg, roi_map=roi_map, provenance_path=provenance_path, sqm_label=last_label)

            except Exception as exc:
                subject_status = "FAILED"
                subject_error = str(exc)
                raise
            finally:
                _jdb.log_run_end(
                    db_path, execution_id, subject,
                    status=subject_status,
                    error_msg=subject_error,
                    duration_seconds=time.monotonic() - t0,
                )

        _jdb.update_execution(db_path, execution_id, "COMPLETED")

    except Exception:
        _jdb.update_execution(db_path, execution_id, "FAILED")
        raise


def _make_prep_config(subject: str, session: str | None, args: dict[str, Any]) -> "PrepConfig":
    raw_bad = args.get("bad_channels")
    return PrepConfig(
        subject=subject,
        session=session,
        dpf=args["dpf"],
        sci_threshold=args["sci_threshold"],
        motion_correction=_v(args["motion_correction"]),
        bad_channels=[c.strip() for c in raw_bad.split(",")] if raw_bad else [],
        cardiac_l_freq=args["cardiac_l_freq"],
        cardiac_h_freq=args["cardiac_h_freq"],
        resp_l_freq=args["resp_l_freq"],
        resp_h_freq=args["resp_h_freq"],
        qc_window_s=args.get("window_length", 10.0),
        ignore=[_v(ig) for ig in (args.get("ignore") or [])],
    )


def _emit_subject_report(subject, sub_dir, last_raw, last_result, prep_config, args, glm_est, dm, alff_df=None, fc_df=None, fc_hbr_df=None, high_pass=None, low_pass=None, after_haemo=None, gcor_reg=None, roi_map=None, provenance_path=None, sqm_label=None):
    import mne
    import numpy as np
    from fnirs_pipe.qc.report import build_subject_report

    hbo_picks = mne.pick_types(last_result.raw_haemo.info, fnirs="hbo")
    coords_head = np.array([
        last_result.raw_haemo.info["chs"][i]["loc"][:3] for i in hbo_picks
    ])
    hbo_names = [last_result.raw_haemo.ch_names[i] for i in hbo_picks]
    bad_bases = {bc.rsplit(" ", 1)[0] for bc in last_result.bad_channels}
    good_mask = np.array([n.rsplit(" ", 1)[0] not in bad_bases for n in hbo_names])

    bad_annots: dict = {}
    for annot in last_raw.annotations:
        desc = annot["description"]
        if desc.upper().startswith("BAD"):
            bad_annots.setdefault(desc, []).append(
                (float(annot["onset"]), float(annot["duration"]))
            )

    build_subject_report(
        subject=subject,
        raw_intensity=last_raw,
        raw_haemo=last_result.raw_haemo,
        sci_scores=last_result.sci_scores,
        bad_channels=last_result.bad_channels,
        config=prep_config,
        run_command=" ".join(sys.argv),
        out_path=sub_dir / f"sub-{subject}_qc.html",
        sci_scores_matrix=last_result.sci_scores_matrix,
        sci_win_times=last_result.sci_win_times,
        psp_scores_matrix=last_result.psp_scores_matrix,
        psp_win_times=last_result.psp_win_times,
        coords_head=coords_head,
        good_mask=good_mask,
        ch_names_brain=hbo_names,
        segments=bad_annots or None,
        raw_before_motion=last_result.raw_od_before_motion,
        raw_after_motion=last_result.raw_od_after_motion,
        design_matrix=dm,
        glm_est=glm_est,
        l_freq=high_pass,
        h_freq=low_pass,
        mode=_v(args.get("mode")) if args.get("mode") else None,
        alff_df=alff_df,
        fc_df=fc_df,
        fc_hbr_df=fc_hbr_df,
        after_haemo=after_haemo,
        gcor_reg=gcor_reg,
        roi_map=roi_map,
        provenance_path=provenance_path,
        sqm_label=sqm_label,
    )


def _run_post_for_subject(
    subject: str,
    sessions: list[str | None],
    args: dict[str, Any],
    toml: dict[str, Any],
    output_dir: Path,
    roi_map: dict | None = None,
) -> tuple:
    """Run post-processing for all sessions/tasks. Returns (glm_est, design_matrix) from last file."""
    from fnirs_pipe.pipeline.post_pipeline import run_post

    mode = _v(args["mode"])
    task_label = args.get("task_label")
    tasks: list[str | None] = task_label if task_label else [None]

    post_layout = get_layout(output_dir, validate=False)
    # the layout indexes every nested derivative tree under output_dir, so an unrelated
    # sub-01 in a sibling output tree would be picked up and post-processed as if it
    # were ours. Only this run's own subject directory counts.
    subject_root = (output_dir / f"sub-{subject}").resolve()
    last_glm_est = last_dm = last_alff_df = last_fc_df = last_fc_hbr_df = last_denoised = last_gcor_reg = None
    for session in sessions:
        post_config = _build_post_config(subject, session, args, toml, roi_map=roi_map)
        for task in tasks:
            preproc_files = [
                p for p in get_nirs_files(
                    post_layout, subject=subject, session=session, task=task,
                )
                if "desc-preproc" in p.name and subject_root in Path(p).resolve().parents
            ]

            if not preproc_files:
                logger.warning("no desc-preproc snirf found for sub-%s, skipping post", subject)
                continue

            for snirf_path in preproc_files:
                src_entities = post_layout.parse_file_entities(str(snirf_path))
                logger.info("post (%s): %s", mode, snirf_path.name)
                try:
                    raw_haemo = read_snirf(snirf_path)
                    last_denoised, glm_est, dm, alff_df, fc_df, fc_hbr_df, gcor_reg = run_post(raw_haemo, post_config, output_dir=output_dir, mode=mode, source_entities=src_entities, source_path=snirf_path)
                    if glm_est is not None:
                        last_glm_est, last_dm = glm_est, dm
                    if fc_df is not None:
                        last_alff_df, last_fc_df, last_fc_hbr_df = alff_df, fc_df, fc_hbr_df
                    if gcor_reg is not None:
                        last_gcor_reg = gcor_reg
                except Exception:
                    logger.exception("post failed for %s", snirf_path)
                    raise

    return last_glm_est, last_dm, last_alff_df, last_fc_df, last_fc_hbr_df, last_denoised, last_gcor_reg


def run_group_level(args: dict[str, Any]) -> None:
    """BIDS Apps `group` entry point — aggregates per-subject (and per-group hyper,
    if present) SQM JSONs into cohort HTML reports under <output_dir>."""
    from pathlib import Path

    from fnirs_pipe.qc.group_writer import (
        build_group_hyper_raw_report,
        build_group_raw_report,
    )

    output_dir = Path(args["output_dir"])
    qc_root    = output_dir / "qc" if (output_dir / "qc").exists() else output_dir

    logger.info("fnirs-pipe group: aggregating individual SQMs from %s", qc_root)
    ind_path = build_group_raw_report(qc_root)
    logger.info("  -> %s", ind_path)

    if any(qc_root.glob("group-*/nirs/*_desc-sqm_nirs.json")):
        logger.info("fnirs-pipe group: also aggregating hyperscanning SQMs")
        hyper_path = build_group_hyper_raw_report(qc_root)
        logger.info("  -> %s", hyper_path)
