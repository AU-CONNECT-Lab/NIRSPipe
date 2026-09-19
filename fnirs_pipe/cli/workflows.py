"""Execution layer between the CLI and the processing pipeline.

Resolves which subjects/sessions/tasks to run, merges CLI args with TOML config
into PrepConfig/PostConfig, and drives the per-subject prep → post → report loop.
Does no signal processing itself.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fnirs_pipe.cli import _shared
from fnirs_pipe.io.bids import bids_label, get_layout, get_nirs_files
from fnirs_pipe.io.derivatives import write_dataset_description
from fnirs_pipe.io.snirf import read_snirf
from fnirs_pipe.pipeline.denoise import DEFAULT_FILTER_METHOD, DEFAULT_FILTER_ORDER
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
        high_pass=pick("high_pass"),
        low_pass=pick("low_pass"),
        filter_method=pick("filter_method", default=DEFAULT_FILTER_METHOD),
        filter_order=pick("filter_order", default=DEFAULT_FILTER_ORDER),
        resample_sfreq=pick("resample_sfreq"),
        stim_dur=pick("stim_dur"),
        hrf_model=pick("hrf_model"),
        noise_model=pick("noise_model", default="auto"),
        drift_model=pick("drift_model"),
        drift_high_pass=pick("drift_high_pass"),
        drift_order=pick("drift_order"),
        fir_delays=tuple(int(x) for x in raw_fir.split(",")) if raw_fir else None,
        short_channel=sc if (sc and sc != "none") else None,
        aux=bool(pick("aux_regressors", default=False)),
        aux_channels=pick("aux_channels"),
        events_path=str(ep) if ep else None,
        contrast_def=contrast_def,
        fc=bool(pick("fc", default=False)),
        combine_runs=pick("combine_runs"),
        roi_map=roi_map,
        # the same bands prep split with, so the regression and the reports agree
        **_shared.separation_bands_from_args(args),
    )


def _refuse_cropped_input(bids_dir: Path, allow: bool) -> None:
    """Stop a run whose input was cut into one file per condition before preprocessing.

    Motion correction fits its weighting over whatever series it is handed and the bandpass
    pads whatever it is given, so each condition preprocessed alone gets a different answer,
    the bandpass's being a baseline invented at the segment edges. Padding the crop fixes
    only the bandpass. The right order is to preprocess the recording and cut afterwards.

    Detected from the input tree's own `dataset_description.json`, which `fnirs-prep crop`
    stamps with its name, so nothing new has to be recorded for this to work.
    """
    if allow:
        return
    desc_path = Path(bids_dir) / "dataset_description.json"
    try:
        generated_by = json.loads(desc_path.read_text(encoding="utf-8")).get("GeneratedBy") or []
    except (OSError, json.JSONDecodeError):
        return
    names = {str(entry.get("Name", "")) for entry in generated_by if isinstance(entry, dict)}
    if "fnirs-prep crop" not in names:
        return
    raise SystemExit(
        f"[error] {bids_dir} was written by `fnirs-prep crop`, so every condition would be "
        "preprocessed on its own. Motion correction and the bandpass both read whatever "
        "series they are handed, so a short condition moves both.\n"
        "        Run this on the uncut recording instead, then cut what you need out of the "
        "result:\n"
        "          fnirs-pipe <bids> <out> participant ...\n"
        "          fnirs-prep crop <out> <out> --input-desc errts --segments-path <tsv> ...\n"
        "        Pass --allow-cropped-input to run on the cropped tree anyway."
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
    logger.info("fnirs-pipe starting - output: %s", output_dir)

    _refuse_cropped_input(bids_dir, allow=bool(args.get("allow_cropped_input")))

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

    # The separation bands are resolved here rather than in either config builder because only
    # the post builder is given the TOML, and prep is the step that stamps the bands into the
    # record. Resolving them in one place is what keeps the record, the regression and the
    # reports describing the same montage; half a threading is worse than no TOML support.
    # CLI still wins. `fnirs-prep` has no --config of its own, so its bands stay CLI-only.
    for _band in _shared.SEPARATION_BAND_KEYS:
        if args.get(_band) is None and toml.get(_band) is not None:
            args[_band] = toml[_band]

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

    run_notes: list[tuple[str, str]] = []
    failed: list[str] = []

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

            # The record and the script are the whole point of a dry run, and both are on
            # disk by here. Stopping before log_run_start keeps the database free of runs
            # that never happened.
            if args.get("dry_run"):
                logger.info("sub-%s | dry run: wrote the record and the script, processed "
                            "nothing", subject)
                continue

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
            # keyed by BIDS run stem: every QC figure is per run, so the report loop below
            # needs each run's own prep output rather than whichever finished last
            prep_runs: dict[str, tuple] = {}
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
                                prep_runs[bids_label(subject, src_entities)] = (raw, result, prep_config)
                            except Exception:
                                logger.exception("prep failed for %s", snirf_path)
                                raise

                post_runs: dict[str, dict] = {}
                if args.get("mode") is not None:
                    post_runs = _run_post_for_subject(subject, sessions, args, toml, output_dir, roi_map=roi_map)

                # one SQM record per run, written once both passes have finished so the
                # post-Beer-Lambert sections can measure the files post actually produced.
                # The database takes one row per section, which is what its checkpoint
                # column has always been for.
                import json as _json
                from fnirs_pipe.qc.subject.sqm_record import SECTIONS, build_sqm_records, entities_of
                try:
                    sqm_paths = build_sqm_records(
                        sub_dir / "nirs", bids_root=bids_dir,
                        qc_window_s=args.get("window_length", 10.0),
                        # nothing on disk records them, so the record would otherwise be
                        # split on the package defaults whatever this run was told
                        sep_bands=_shared.resolved_separation_bands(args),
                        labels=set(prep_runs))
                except Exception:
                    logger.error("sub-%s | SQM records failed", subject, exc_info=True)
                    sqm_paths = []
                for path in sqm_paths:
                    logger.info("sub-%s | SQM record -> %s", subject, path.name)
                    # the record is on disk either way; only the database rows are at risk here
                    try:
                        record = _json.loads(path.read_text(encoding="utf-8"))
                        ents = entities_of(path.stem)
                        for section in SECTIONS:
                            if record.get(section):
                                _jdb.log_sqm(db_path, execution_id, subject, section,
                                             record[section], session=ents["ses"],
                                             bids_task=ents["task"])
                    except Exception:
                        logger.warning("sub-%s | SQM database rows failed for %s",
                                       subject, path.name, exc_info=True)

                # one report per run: the figures, the provenance graph and the metrics all
                # describe a single recording, so a subject holding five tasks gets five
                for label, (raw, result, run_prep_config) in prep_runs.items():
                    # rendered before the report, which embeds it: every sidecar it scans is
                    # on disk by now, and --no-report still leaves the diagram behind
                    provenance_path = None
                    try:
                        from fnirs_pipe.qc.figures.common.provenance_figure import write_provenance
                        for path in write_provenance(
                            sub_dir / "nirs", sub_dir / "figures" / label,
                            stem="provenance", label=label,
                            title=label + (f"  |  mode: {args['mode']}" if args.get("mode") else ""),
                        ):
                            logger.info("sub-%s | provenance -> %s", subject, path)
                            if path.suffix == ".png":
                                provenance_path = f"figures/{label}/{path.name}"
                    except Exception:
                        logger.warning("%s | provenance graph failed", label, exc_info=True)

                    if args.get("no_report"):
                        continue
                    post = post_runs.get(label, {})
                    run_notes += [(label, n) for n in _emit_subject_report(
                        subject, sub_dir, raw, result, run_prep_config, args,
                        post.get("glm_est"), post.get("design_matrix"),
                        alff_df=post.get("alff_df"), fc_df=post.get("fc_df"),
                        fc_hbr_df=post.get("fc_hbr_df"), fc_seed=post.get("fc_seed") or {},
                        fc_roi=post.get("fc_roi") or {},
                        high_pass=cfg_high_pass, low_pass=cfg_low_pass,
                        after_haemo=post.get("denoised"),
                        roi_map=roi_map, provenance_path=provenance_path, sqm_label=label,
                    ) or []]

                if not args.get("no_report") and prep_runs:
                    from fnirs_pipe.qc.subject.subject_index import write_subject_index
                    try:
                        write_subject_index(subject, sub_dir, " ".join(sys.argv),
                                            mode=_v(args["mode"]) if args.get("mode") else None)
                    except Exception:
                        logger.warning("sub-%s | run index failed", subject, exc_info=True)

            # Recorded rather than raised: the subjects are independent and a rerun skips
            # what finished, so one bad recording must not strand the rest of the batch.
            except Exception as exc:
                subject_status = "FAILED"
                subject_error = str(exc)
                logger.exception("sub-%s | failed", subject)
                print(f"  [error] sub-{subject}: {exc}", file=sys.stderr)
                failed.append(subject)
            finally:
                _jdb.log_run_end(
                    db_path, execution_id, subject,
                    status=subject_status,
                    error_msg=subject_error,
                    duration_seconds=time.monotonic() - t0,
                )

        _jdb.update_execution(db_path, execution_id, "FAILED" if failed else "COMPLETED")
        _log_run_notes(run_notes)
        if failed:
            print(f"{len(failed)} of {len(participant_label)} subject(s) failed: "
                  f"{', '.join(failed)}", file=sys.stderr)
            raise SystemExit(1)

    except Exception:
        _jdb.update_execution(db_path, execution_id, "FAILED")
        raise


def _log_run_notes(run_notes: "list[tuple[str, str]]") -> None:
    """Repeat every report note once at the end, so a whole run's omissions read together.

    They were already logged where they happened, hundreds of lines back and one run at a
    time. Nothing here is an error; the same list is in each run's QC report.
    """
    if not run_notes:
        return
    logger.info("run notes (%d) - sections left out, not failures:", len(run_notes))
    for label, note in run_notes:
        logger.info("  %s | %s", label, note)


def _bad_channels_for(spec: str | None, subject: str) -> list[str]:
    """Resolve --bad-channels for one subject: a shared list, or a table with one row each.

    e.g. "S1_D1,S2_D3" gives that list for every subject, while a table

    ::

        participant_id  bad_channels
        sub-01          S1_D1,S2_D3
        sub-02          S4_D4

    gives "sub-01" the first row and "sub-02" the second. A subject the table does not list
    has none, which is how a cohort where only some caps slipped is described. The prefix is
    optional on either side, so "01" and "sub-01" name the same subject.
    """
    if not spec:
        return []
    spec = str(spec)
    path = Path(spec)
    if not path.exists():
        return [c.strip() for c in spec.split(",") if c.strip()]

    from fnirs_pipe.io.tables import read_table

    table = read_table(path, dtype=str).fillna("")
    missing = {"participant_id", "bad_channels"} - set(table.columns)
    if missing:
        raise ValueError(
            f"--bad-channels table {path} needs columns participant_id and bad_channels; "
            f"missing {sorted(missing)}"
        )
    wanted = subject.removeprefix("sub-")
    rows = table[table["participant_id"].str.removeprefix("sub-") == wanted]
    if rows.empty:
        return []
    return [c.strip() for c in ",".join(rows["bad_channels"]).split(",") if c.strip()]


def _make_prep_config(subject: str, session: str | None, args: dict[str, Any]) -> "PrepConfig":
    return PrepConfig(
        subject=subject,
        session=session,
        dpf=args["dpf"],
        sci_threshold=args["sci_threshold"],
        **({"psp_threshold": args["psp_threshold"]}
           if args.get("psp_threshold") is not None else {}),
        **({"min_good_frac": args["min_good_frac"]}
           if args.get("min_good_frac") is not None else {}),
        **({"screen_scope": args["screen_scope"]}
           if args.get("screen_scope") is not None else {}),
        motion_correction=_v(args["motion_correction"]),
        bad_channels=_bad_channels_for(args.get("bad_channels"), subject),
        cardiac_l_freq=args["cardiac_l_freq"],
        cardiac_h_freq=args["cardiac_h_freq"],
        resp_l_freq=args["resp_l_freq"],
        resp_h_freq=args["resp_h_freq"],
        qc_window_s=args.get("window_length", 10.0),
        epoch_tmin=args.get("epoch_tmin"),
        epoch_tmax=args.get("epoch_tmax"),
        epoch_chunk_duration=args.get("epoch_chunk_duration"),
        gvtd_censor=args.get("gvtd_censor"),
        gvtd_censor_n_std=args.get("gvtd_censor_n_std", 10.0),
        gvtd_min_epoch_s=args.get("gvtd_min_epoch_s", 30.0),
        **_shared.separation_bands_from_args(args),
        ignore=[_v(ig) for ig in (args.get("ignore") or [])],
    )


def _emit_subject_report(subject, sub_dir, last_raw, last_result, prep_config, args, glm_est, dm, alff_df=None, fc_df=None, fc_hbr_df=None, fc_seed=None, fc_roi=None, high_pass=None, low_pass=None, after_haemo=None, roi_map=None, provenance_path=None, sqm_label=None):
    import mne
    import numpy as np
    from fnirs_pipe.qc.subject.report import build_subject_report

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
    # censoring annotates the optical density, which last_raw is upstream of, so the spans
    # come off the result rather than out of the input
    if last_result.censor_spans:
        bad_annots["BAD_gvtd"] = [tuple(span) for span in last_result.censor_spans]

    return build_subject_report(
        subject=subject,
        raw_intensity=last_raw,
        raw_haemo=last_result.raw_haemo,
        sci_scores=last_result.sci_scores,
        bad_channels=last_result.bad_channels,
        config=prep_config,
        run_command=" ".join(sys.argv),
        out_path=sub_dir / f"{sqm_label or f'sub-{subject}'}_qc.html",
        coords_head=coords_head,
        good_mask=good_mask,
        ch_names_brain=hbo_names,
        segments=bad_annots or None,
        design_matrix=dm,
        glm_est=glm_est,
        l_freq=high_pass,
        h_freq=low_pass,
        mode=_v(args.get("mode")) if args.get("mode") else None,
        # read straight off args like `mode`, deliberately not through either PrepConfig
        # splat: this is a report option and PrepConfig has no field for it
        by_condition=bool(args.get("by_condition")),
        epoch_single_trial=bool(args.get("epoch_single_trial")),
        alff_df=alff_df,
        fc_df=fc_df,
        fc_hbr_df=fc_hbr_df,
        fc_seed=fc_seed,
        fc_roi=fc_roi,
        after_haemo=after_haemo,
        roi_map=roi_map,
        provenance_path=provenance_path,
        sqm_label=sqm_label,
    )


# What post leaves behind for one run, in the order the report section builders want it.
_POST_FIELDS = ("glm_est", "design_matrix", "alff_df", "fc_df", "fc_hbr_df",
                "denoised", "fc_seed", "fc_roi")


def _run_post_for_subject(
    subject: str,
    sessions: list[str | None],
    args: dict[str, Any],
    toml: dict[str, Any],
    output_dir: Path,
    roi_map: dict | None = None,
) -> dict[str, dict]:
    """Run post-processing for all sessions/tasks. Returns ``{bids_label: outputs}``.

    One entry per run, keyed the same way as the prep results, so the report loop can pair
    them up. A run whose post failed simply has no entry.
    """
    from fnirs_pipe.pipeline.post_pipeline import run_post

    mode = _v(args["mode"])
    task_label = args.get("task_label")
    tasks: list[str | None] = task_label if task_label else [None]

    post_layout = get_layout(output_dir, validate=False)
    # the layout indexes every nested derivative tree under output_dir, so an unrelated
    # sub-01 in a sibling output tree would be picked up and post-processed as if it
    # were ours. Only this run's own subject directory counts.
    subject_root = (output_dir / f"sub-{subject}").resolve()
    post_runs: dict[str, dict] = {}
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
                    denoised, glm_est, dm, alff_df, fc_df, fc_hbr_df, fc_seed, fc_roi = run_post(raw_haemo, post_config, output_dir=output_dir, mode=mode, source_entities=src_entities, source_path=snirf_path)
                except Exception:
                    logger.exception("post failed for %s", snirf_path)
                    raise
                post_runs[bids_label(subject, src_entities)] = dict(zip(
                    _POST_FIELDS,
                    (glm_est, dm, alff_df, fc_df, fc_hbr_df, denoised, fc_seed, fc_roi),
                ))

    return post_runs


def _warn_on_split_tree(output_dir: Path) -> None:
    """Say so when quality records sit in a subtree this aggregation does not reach.

    Two output directories means two records for one run, and the rule preferring the
    pipeline record over the `prep-raw` one can only choose between records one glob found.
    """
    for sub in (output_dir / "qc", output_dir / "derivatives"):
        if sub.is_dir() and any(sub.glob("*/**/nirs/*_desc-sqm*_nirs.json")):
            logger.warning(
                "quality records under %s are not part of this cohort page; point both "
                "`fnirs-pipe` and `fnirs-qc prep-raw` at one output directory, or aggregate "
                "that one separately with `fnirs-qc cohort %s`", sub, sub)


def run_group_level(args: dict[str, Any]) -> None:
    """BIDS Apps `group` entry point — aggregates per-subject (and per-group hyper,
    if present) SQM JSONs into cohort HTML reports under <output_dir>."""
    from pathlib import Path

    from fnirs_pipe.qc.hyper.group_hyper_writer import build_group_hyper_report
    from fnirs_pipe.qc.subject.group_writer import build_group_raw_report

    output_dir = Path(args["output_dir"])
    _warn_on_split_tree(output_dir)

    logger.info("fnirs-pipe group: aggregating individual SQMs from %s", output_dir)
    ind_path = build_group_raw_report(output_dir)
    logger.info("  -> %s", ind_path)

    if any(output_dir.glob("group-*/nirs/*_desc-sqm_nirs.json")):
        logger.info("fnirs-pipe group: also aggregating hyperscanning SQMs")
        hyper_path = build_group_hyper_report(output_dir)
        logger.info("  -> %s", hyper_path)

