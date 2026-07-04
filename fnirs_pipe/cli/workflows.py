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

from fnirs_pipe.io.bids import get_layout, get_nirs_files
from fnirs_pipe.io.derivatives import write_dataset_description
import mne
from fnirs_pipe.pipeline.prep_pipeline import PrepConfig, run_prep
from fnirs_pipe.utils import unwrap_enum as _v
from fnirs_pipe.utils import job_db as _jdb
from fnirs_pipe.utils.logging import get_logger, setup_logging
from fnirs_pipe.utils.run_record import write_run_record
from fnirs_pipe.utils.run_script import write_run_script

logger = get_logger("cli.workflows")

def _build_post_config(subject: str, session: str | None, args: dict[str, Any], toml: dict[str, Any]) -> Any:
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
            log_file = sub_dir / "logs" / f"sub-{subject}_{sub_timestamp}.log"
            setup_logging(verbose=verbose, log_file=log_file)
            logger.info("sub-%s | starting", subject)
            write_run_record(args, subject, sub_timestamp, output_dir, sub_dir=sub_dir)
            write_run_script(args, subject, sub_timestamp, output_dir, sub_dir=sub_dir)

            _jdb.log_run_start(
                db_path, execution_id, subject,
                sci_threshold=args["sci_threshold"],
                dpf=args["dpf"],
                motion_correction=args["motion_correction"].value,
                mode=_v(args["mode"]) if args.get("mode") else None,
                high_pass=args.get("high_pass"),
                low_pass=args.get("low_pass"),
                hrf_model=_v(args["hrf_model"]) if args.get("hrf_model") else None,
            )

            t0 = time.monotonic()
            subject_status = "SUCCESS"
            subject_error: str | None = None
            last_raw = last_result = None
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
                                raw = mne.io.read_raw_snirf(str(snirf_path), preload=True)
                                result = run_prep(raw, prep_config, output_dir=output_dir, source_entities=src_entities, work_dir=work_dir)
                                logger.info("finished prep: %s", snirf_path.name)
                                last_raw, last_result = raw, result
                            except Exception:
                                logger.exception("prep failed for %s", snirf_path)
                                raise

                if last_result is not None:
                    if last_result.iqm_raw:
                        _jdb.log_iqm(db_path, execution_id, subject, "raw", last_result.iqm_raw)
                    if last_result.iqm_final:
                        _jdb.log_iqm(db_path, execution_id, subject, "final", last_result.iqm_final)

                glm_est = dm = alff_df = fc_df = None
                if args.get("mode") is not None:
                    glm_est, dm, alff_df, fc_df = _run_post_for_subject(subject, sessions, args, toml, output_dir)

                if not args.get("no_report") and last_result is not None:
                    _emit_subject_report(subject, sub_dir, last_raw, last_result, prep_config, args, glm_est, dm, alff_df=alff_df, fc_df=fc_df)

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
    raw_excl = args.get("exclude_channels")
    return PrepConfig(
        subject=subject,
        session=session,
        dpf=args["dpf"],
        sci_threshold=args["sci_threshold"],
        motion_correction=args["motion_correction"].value,
        exclude_channels=[c.strip() for c in raw_excl.split(",")] if raw_excl else [],
        cardiac_l_freq=args["cardiac_l_freq"],
        cardiac_h_freq=args["cardiac_h_freq"],
        ignore=[ig.value for ig in (args.get("ignore") or [])],
    )


def _emit_subject_report(subject, sub_dir, last_raw, last_result, prep_config, args, glm_est, dm, alff_df=None, fc_df=None):
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
        l_freq=args.get("high_pass"),
        h_freq=args.get("low_pass"),
        mode=_v(args.get("mode")) if args.get("mode") else None,
        alff_df=alff_df,
        fc_df=fc_df,
    )


def _run_post_for_subject(
    subject: str,
    sessions: list[str | None],
    args: dict[str, Any],
    toml: dict[str, Any],
    output_dir: Path,
) -> tuple:
    """Run post-processing for all sessions/tasks. Returns (glm_est, design_matrix) from last file."""
    from fnirs_pipe.pipeline.post_pipeline import run_post

    mode = _v(args["mode"])
    task_label = args.get("task_label")
    tasks: list[str | None] = task_label if task_label else [None]

    post_layout = get_layout(output_dir, validate=False)
    last_glm_est = last_dm = last_alff_df = last_fc_df = None
    for session in sessions:
        post_config = _build_post_config(subject, session, args, toml)
        for task in tasks:
            preproc_files = [
                p for p in get_nirs_files(
                    post_layout, subject=subject, session=session, task=task,
                )
                if "desc-preproc" in p.name
            ]

            if not preproc_files:
                logger.warning("no desc-preproc snirf found for sub-%s, skipping post", subject)
                continue

            for snirf_path in preproc_files:
                src_entities = post_layout.parse_file_entities(str(snirf_path))
                logger.info("post (%s): %s", mode, snirf_path.name)
                try:
                    raw_haemo = mne.io.read_raw_snirf(str(snirf_path), preload=True)
                    _, glm_est, dm, alff_df, fc_df = run_post(raw_haemo, post_config, output_dir=output_dir, mode=mode, source_entities=src_entities)
                    if glm_est is not None:
                        last_glm_est, last_dm = glm_est, dm
                    if fc_df is not None:
                        last_alff_df, last_fc_df = alff_df, fc_df
                except Exception:
                    logger.exception("post failed for %s", snirf_path)
                    raise

    return last_glm_est, last_dm, last_alff_df, last_fc_df


def run_group_level(args: dict[str, Any]) -> None:
    """BIDS Apps `group` entry point — aggregates per-subject (and per-group hyper,
    if present) IQM JSONs into cohort HTML reports under <output_dir>."""
    from pathlib import Path

    from fnirs_pipe.qc.group_writer import (
        build_group_hyper_raw_report,
        build_group_raw_report,
    )

    output_dir = Path(args["output_dir"])
    qc_root    = output_dir / "qc" if (output_dir / "qc").exists() else output_dir

    logger.info("fnirs-pipe group: aggregating individual IQMs from %s", qc_root)
    ind_path = build_group_raw_report(qc_root)
    logger.info("  -> %s", ind_path)

    if any(qc_root.glob("group-*/nirs/*_desc-iqm_nirs.json")):
        logger.info("fnirs-pipe group: also aggregating hyperscanning IQMs")
        hyper_path = build_group_hyper_raw_report(qc_root)
        logger.info("  -> %s", hyper_path)
