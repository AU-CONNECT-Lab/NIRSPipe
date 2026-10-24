"""Execution layer between the CLI and the processing pipeline.

Resolves which subjects/sessions/tasks to run, merges CLI args with TOML config
into PrepConfig/PostConfig, and drives the per-subject prep → post → report loop.
Does no signal processing itself.
"""

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np

from nirspipe.cli import _app_options, _shared
from nirspipe.cli.run import mode_defaults, read_config
from nirspipe.exceptions import StageError
from nirspipe.io.bids import (bids_label, channel_status_bads, get_layout, get_nirs_files,
                              get_participant_age, validate_bids)
from nirspipe.io.head_template import resolve_template
from nirspipe.io.naming import report_name, roi_map_name
from nirspipe.io.derivatives import (
    LINK_RAW, entity_of, write_bidsignore, write_dataset_description,
)
from nirspipe.io.snirf import read_raw_snirf, read_snirf
from nirspipe.pipeline.denoise import DEFAULT_FILTER_METHOD, DEFAULT_FILTER_ORDER
from nirspipe.pipeline.censor_fill import DEFAULT_CENSOR_FILL
from nirspipe.pipeline.dpf import dpf_from_age
from nirspipe.pipeline.motion import motion_settings
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep
from nirspipe.utils import pair_of, unwrap_enum as _v
from nirspipe.utils.logging import get_logger, setup_logging, thread_log_file
from nirspipe.cli.run_record import (RUN_TIMESTAMP_FORMAT, error_line, record_stem,
                                     write_run_outcome, write_run_record)
from nirspipe.utils.spans import bad_spans, marker_onset

logger = get_logger("cli.workflows")

def _roi_map_name(args: dict[str, Any]) -> str:
    """The seg- entity of this run's ROI map, from ``--roi-mapping``."""
    return roi_map_name(args.get("roi_mapping"))


def _build_post_config(subject: str, session: str | None, args: dict[str, Any], toml: dict[str, Any], roi_map: dict | None = None) -> Any:
    from nirspipe.pipeline.post_pipeline import PostConfig

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
        from nirspipe.utils import load_toml
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
        censor_fill=pick("censor_fill", default=DEFAULT_CENSOR_FILL),
        resample_sfreq=pick("resample_sfreq"),
        stim_dur=pick("stim_dur"),
        hrf_model=pick("hrf_model"),
        noise_model=pick("noise_model", default="auto"),
        drift_model=pick("drift_model"),
        drift_high_pass=pick("drift_high_pass"),
        drift_order=pick("drift_order", default=1),
        fir_delays=tuple(int(x) for x in raw_fir.split(",")) if raw_fir else None,
        short_channel=sc if (sc and sc != "none") else None,
        aux=bool(pick("aux_regressors", default=False)),
        aux_channels=pick("aux_channels"),
        events_path=str(ep) if ep else None,
        contrast_def=contrast_def,
        fc=bool(pick("fc", default=False)),
        min_time=float(pick("min_time", default=0.0)),
        export_mean_filled=bool(pick("export_mean_filled", default=False)),
        combine_runs=pick("combine_runs"),
        roi_map=roi_map,
        roi_map_name=_roi_map_name(args),
        # the same bands prep split with, so the regression and the reports agree
        **_app_options.separation_bands_from_args(args),
    )


# ---- post settings: command line over --config over the mode's defaults ----

_NOT_SETTINGS = {"subject", "session", "roi_map", "roi_map_name", "contrast_def"}
_ARG_OF_FIELD = {"aux": "aux_regressors"}


def _post_setting_args() -> dict[str, str]:
    """PostConfig field -> the argument, and TOML key, that sets it."""
    from dataclasses import fields
    from nirspipe.pipeline.post_pipeline import PostConfig
    return {f.name: _ARG_OF_FIELD.get(f.name, f.name)
            for f in fields(PostConfig) if f.name not in _NOT_SETTINGS}


def _resolve_post_settings(args: dict[str, Any],
                           config_toml: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Fill the gaps the command line left in ``args`` from --config, then the mode's defaults.

    Returns the two file layers merged, and which layer each PostConfig field came from:
    ``cli``, ``config``, ``mode`` or ``default``. The sources are read before the fill,
    after which every value sits in ``args`` and looks typed.

    e.g. ``--mode rest --low-pass 0.08`` with no --config gives low_pass "cli" (0.08),
    high_pass "mode" (the rest preset's 0.01) and resample_sfreq "default" (None).
    """
    defaults = mode_defaults(_v(args.get("mode")))
    layered = {**defaults, **config_toml}
    setting_args = _post_setting_args()

    sources: dict[str, str] = {}
    for field, arg in setting_args.items():
        if args.get(arg) is not None:
            sources[field] = "cli"
        elif arg in config_toml:
            sources[field] = "config"
        elif arg in defaults:
            sources[field] = "mode"
        else:
            sources[field] = "default"
    sources["roi_map_name"] = "cli" if args.get("roi_mapping") else "default"

    # one resolved set for every reader
    for arg in [*setting_args.values(), "contrast_file"]:
        if args.get(arg) is None and layered.get(arg) is not None:
            args[arg] = layered[arg]
    # `--low-pass none` switches a cutoff off even where a file layer sets one
    for arg in ("high_pass", "low_pass"):
        if args.get(arg) == "none":
            args[arg] = None
            layered.pop(arg, None)
    return layered, sources


def _refuse_cosine_without_cutoff(args: dict[str, Any], sources: dict[str, str]) -> None:
    """Stop before any subject runs, naming the layer that chose cosine."""
    if _v(args.get("drift_model")) != "cosine" or args.get("drift_high_pass") is not None:
        return
    chosen_by = {"mode": f"--mode {_v(args['mode'])} uses", "config": "--config sets"}.get(
        sources.get("drift_model"), "--drift-model asks for")
    raise SystemExit(
        f"[error] {chosen_by} a cosine drift model, which needs --drift-high-pass. The "
        "cutoff depends on the design: use 1/(2 x the slowest repeat of any condition), "
        "or pick another --drift-model.")


def _refuse_glm_without_durations(args: dict[str, Any]) -> None:
    """Stop before any subject runs when a GLM has no way to set its event durations."""
    if _v(args.get("mode")) != "glm" or args.get("stim_dur") is not None or args.get("events_path"):
        return
    raise SystemExit(
        "[error] --mode glm needs --stim-dur (one duration for every annotation) or "
        "--events-path (a table of onsets and durations).")


def _refuse_cropped_input(bids_dir: Path, allow: bool) -> None:
    """Stop a run whose input was cut into one file per condition before preprocessing.

    Motion correction fits its weighting over whatever series it is handed and the bandpass
    pads whatever it is given, so each condition preprocessed alone gets a different answer.

    Detected from the input tree's own `dataset_description.json`, which `nirspipe-prep crop`
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
    if "nirspipe-prep crop" not in names:
        return
    raise SystemExit(
        f"[error] {bids_dir} was written by `nirspipe-prep crop`, so every condition would be "
        "preprocessed on its own. Motion correction and the bandpass both read whatever "
        "series they are handed, so a short condition moves both.\n"
        "        Run this on the uncut recording instead, then cut what you need out of the "
        "result:\n"
        "          nirspipe <bids> <out> participant ...\n"
        "          nirspipe-prep crop <out> <out> --input-desc errts --segments-path <tsv> ...\n"
        "        Pass --allow-cropped-input to run on the cropped tree anyway."
    )


def _refuse_missing_ages(dpf: list, bids_dir: Path, subjects: "list[str]") -> dict:
    """With ``--dpf age``, each subject's age, stopping before the first when any has none."""
    try:
        if dpf_from_age(dpf):
            return {subject: get_participant_age(bids_dir, subject) for subject in subjects}
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None
    return {}


def run_participant_level(args: dict[str, Any]) -> None:
    bids_dir: Path             = args["bids_dir"]
    output_dir: Path           = args["output_dir"]
    participant_label: list[str] | None = args.get("participant_label")
    session_label: list[str] | None  = args.get("session_label")
    task_label: list[str] | None     = args.get("task_label")
    bids_filter_file: Path | None    = args.get("bids_filter_file")
    verbose: bool                    = args.get("verbose", False)

    setup_logging(verbose=verbose)
    logger.info("nirspipe starting - output: %s", output_dir)

    _refuse_cropped_input(bids_dir, allow=bool(args.get("allow_cropped_input")))

    skip_validation = (
        args.get("skip_bids_validation", False)
        or "bids-validation" in (args.get("ignore") or [])
    )
    if not skip_validation:
        validate_bids(bids_dir)
    layout = get_layout(bids_dir)
    # If no participant label provided, run on all subjects in BIDS dir.
    if not participant_label:
        participant_label = layout.get_subjects()
    else:
        missing = sorted(set(participant_label) - set(layout.get_subjects()))
        if missing:
            raise SystemExit(f"Error: participant label(s) not in {bids_dir}: "
                             f"{', '.join(missing)}")
    _refuse_unmatched_bad_channel_rows(args.get("bad_channels"), layout)
    ages = _refuse_missing_ages(args.get("dpf") or [], bids_dir, participant_label)
    _app_options.templates_for(args.get("template"), participant_label, ages)
    _refuse_bad_span_rows(args.get("keep_spans"), layout, marker=args.get("keep_spans_from"))
    _refuse_bad_span_rows(args.get("bad_spans"), layout, flag="--bad-spans")
    write_dataset_description(output_dir, source=bids_dir, link=LINK_RAW)
    write_bidsignore(output_dir)

    config_toml: dict[str, Any] = {}
    if args.get("config"):
        config_toml = read_config(args["config"])["post"]
        logger.debug("loaded post settings from %s", args["config"])

    # folded into args before anything reads them: prep stamps the separation bands, and the
    # report reads the cutoffs straight off args
    toml, post_sources = _resolve_post_settings(args, config_toml)
    cfg_high_pass = args.get("high_pass")
    cfg_low_pass  = args.get("low_pass")
    # (argument, value, layer) for what the report lists as not typed
    filled_settings: list[tuple[str, Any, str]] = []
    if args.get("mode"):
        _refuse_cosine_without_cutoff(args, post_sources)
        _refuse_glm_without_durations(args)
        # a cutoff a file switched off was popped from the layers, so it is absent there
        filled_settings = [(arg, toml.get(arg, "none"), post_sources[field])
                           for field, arg in _post_setting_args().items()
                           if post_sources[field] in ("config", "mode")]

    roi_map = _app_options.load_roi_mapping(args.get("roi_mapping"))

    tasks: list[str | None] = task_label if task_label else [None]

    run_notes: list[tuple[str, str]] = []
    failed: list[str] = []

    def _one(subject: str) -> None:
        sub_timestamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
        sub_dir = output_dir / f"sub-{subject}"
        log_file = sub_dir / "logs" / f"{record_stem(subject, session_label, task_label)}.log"
        with thread_log_file(log_file), _isolate(subject, failed):
            logger.info("sub-%s | starting", subject)
            # built here purely to record what will be used; the loops below build their own
            # per-session copies. Resolving now keeps the record even if the run then fails.
            record = write_run_record(
                args, subject, sub_timestamp, output_dir, sub_dir=sub_dir,
                prep_config=_make_prep_config(subject, None, args),
                post_config=(_build_post_config(subject, None, args, toml, roi_map=roi_map)
                             if args.get("mode") else None),
                post_sources=post_sources,
            )

            # The record is the whole point of a dry run, and it is on disk by here.
            if args.get("dry_run"):
                logger.info("sub-%s | dry run: wrote the record, processed nothing", subject)
                return

            # keyed by BIDS run stem: every QC figure is per run, so the report loop below
            # needs each run's own prep output rather than whichever finished last
            prep_runs: dict[str, tuple] = {}
            t0 = time.monotonic()
            status, error = "success", None
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
                            # the file's own session, which --session-label only filters on;
                            # without it the outputs of a session tree lose their ses- level
                            prep_config = _make_prep_config(
                                subject, src_entities.get("session") or session, args,
                                entities=src_entities,
                                channels_tsv_bads=(
                                    [] if "channel-status" in (args.get("ignore") or [])
                                    else channel_status_bads(layout, snirf_path)))
                            logger.info("processing: %s", snirf_path)
                            try:
                                raw = read_snirf(snirf_path)
                                result = run_prep(raw, prep_config, output_dir=output_dir, source_entities=src_entities, source_path=snirf_path,
                                                  source_metadata=layout.get_metadata(str(snirf_path)))
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
                from nirspipe.qc.subject.sqm_record import build_sqm_records
                try:
                    # one nirs/ per session the runs came from
                    sqm_paths = []
                    for ses in sorted({entity_of(label, "ses") or "" for label in prep_runs}):
                        sqm_paths += build_sqm_records(
                            sub_dir / (f"ses-{ses}" if ses else "") / "nirs",
                            bids_root=bids_dir,
                            qc_window_s=args.get("window_length", 10.0),
                            # nothing on disk records them, so the record would otherwise
                            # be split on the package defaults whatever this run was told
                            sep_bands=_app_options.resolved_separation_bands(args),
                            labels=set(prep_runs))
                except Exception:
                    logger.error("sub-%s | SQM records failed", subject, exc_info=True)
                    sqm_paths = []
                for path in sqm_paths:
                    logger.info("sub-%s | SQM record -> %s", subject, path.name)

                # one report per run: the figures, the provenance graph and the metrics all
                # describe a single recording, so a subject holding five tasks gets five
                for label, (raw, result, run_prep_config) in prep_runs.items():
                    # rendered before the report, which embeds it: every sidecar it scans is
                    # on disk by now, and --no-report still leaves the diagram behind
                    provenance_path = None
                    try:
                        from nirspipe.qc.figures.common.provenance_figure import write_provenance
                        from nirspipe.qc.common.figure_io import figure_namer
                        # the run's own nirs/, session level included; the figure stays
                        # beside the subject's reports
                        ses = entity_of(label, "ses")
                        for path in write_provenance(
                            sub_dir / (f"ses-{ses}" if ses else "") / "nirs",
                            sub_dir / "figures",
                            figure_namer(label), label=label,
                            title=label + (f"  |  mode: {args['mode']}" if args.get("mode") else ""),
                        ):
                            logger.info("sub-%s | provenance -> %s", subject, path)
                            if path.suffix == ".png":
                                provenance_path = f"figures/{path.name}"
                    except Exception:
                        logger.warning("%s | provenance graph failed", label, exc_info=True)

                    if args.get("no_report"):
                        continue
                    post = post_runs.get(label, {})
                    # extend rather than +=: inside the per-subject function that is a
                    # rebind, which makes the name local and unreadable, and several
                    # subjects may be appending at once
                    run_notes.extend([(label, n) for n in _emit_subject_report(
                        subject, sub_dir, raw, result, run_prep_config, args,
                        post.get("glm_est"), post.get("design_matrix"),
                        alff_df=post.get("alff_df"), fc_df=post.get("fc_df"),
                        fc_hbr_df=post.get("fc_hbr_df"), fc_seed=post.get("fc_seed") or {},
                        fc_roi=post.get("fc_roi") or {},
                        high_pass=cfg_high_pass, low_pass=cfg_low_pass,
                        after_haemo=post.get("denoised"),
                        roi_map=roi_map, provenance_path=provenance_path, sqm_label=label,
                        roi_map_name=_roi_map_name(args),
                        filled_settings=filled_settings,
                    ) or []])

                if not args.get("no_report") and prep_runs:
                    from nirspipe.qc.subject.subject_index import write_subject_index
                    try:
                        write_subject_index(subject, sub_dir, " ".join(sys.argv),
                                            mode=_v(args["mode"]) if args.get("mode") else None)
                    except Exception:
                        logger.warning("sub-%s | run index failed", subject, exc_info=True)

            # recorded rather than raised, so one bad recording does not strand the batch
            except Exception as exc:
                status, error = "failed", error_line(exc)
                logger.exception("sub-%s | failed", subject)
                print(f"  [error] sub-{subject}: {exc}", file=sys.stderr)
                failed.append(subject)
            finally:
                write_run_outcome(record, status, time.monotonic() - t0, error)

    nprocs = args.get("nprocs")
    if nprocs:
        jobs, blas = _shared.split_cpus(nprocs, len(participant_label))
        logger.info("%d subject(s) within %d cores: %d at once, %d BLAS thread(s) each",
                    len(participant_label), nprocs, jobs, blas)
    _shared.map_subjects(_one, participant_label, nprocs)

    _log_run_notes(run_notes)
    if failed:
        print(f"{len(failed)} of {len(participant_label)} subject(s) failed: "
              f"{', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)


@contextmanager
def _isolate(subject: str, failed: list[str]):
    """Record a subject's failure instead of raising it, so the rest of the batch still runs.

    The inner ``try`` around the pipeline itself already does this for anything the stages
    raise; this covers the run record that sits outside it, which would otherwise end the
    batch. A subject that failed inside is not listed twice.
    """
    try:
        yield
    except Exception as exc:
        logger.exception("sub-%s | failed", subject)
        print(f"  [error] sub-{subject}: {exc}", file=sys.stderr)
        if subject not in failed:
            failed.append(subject)


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


# --bad-channels table column -> (pybids entity, optional prefix)
_BAD_CHANNEL_KEYS = {"participant_id": ("subject", "sub-"), "session": ("session", "ses-"),
                     "task": ("task", "task-"), "acq": ("acquisition", "acq-"),
                     "run": ("run", "run-")}


def _label_table_rows(path: Path, flag: str, data_columns: tuple[str, ...],
                      optional: tuple[str, ...] = ()) -> list[dict]:
    """The rows of a per-recording table, each label cell with its prefix taken off.

    A ``--bad-channels`` row ``sub-01 | ses-02 | | 1 | S1_D1`` comes back as
    ``{"participant_id": "01", "session": "02", "task": "", "run": "1",
    "bad_channels": "S1_D1", "line": 2}``; a blank or absent label column means all.
    """
    from nirspipe.io.tables import read_table

    table = read_table(path, dtype=str).fillna("")
    missing = [c for c in data_columns if c not in table.columns]
    if missing:
        raise ValueError(f"{flag} table {path} needs a {' and a '.join(missing)} column")
    # a misspelt label column would otherwise read as blank, i.e. every recording
    unknown = set(table.columns) - set(_BAD_CHANNEL_KEYS) - set(data_columns) - set(optional)
    if unknown:
        raise ValueError(
            f"{flag} table {path} has unknown columns {sorted(unknown)}; "
            f"allowed: {', '.join([*_BAD_CHANNEL_KEYS, *data_columns, *optional])}")
    rows = []
    for i, record in enumerate(table.to_dict("records")):
        line = i + 2  # the header is line 1
        row: dict[str, Any] = {**{c: record[c] for c in data_columns},
                               **{c: record.get(c, "") for c in optional}, "line": line}
        for column, (_, prefix) in _BAD_CHANNEL_KEYS.items():
            label = str(record.get(column, "")).strip().removeprefix(prefix)
            if label and not (label.isdigit() if column == "run" else label.isalnum()):
                kind = "a run number" if column == "run" else "a BIDS label (letters and digits)"
                raise ValueError(
                    f"{flag} table {path} line {line}: {column} "
                    f"{record.get(column, '')!r} is not {kind}")
            row[column] = label
        rows.append(row)
    return rows


def _bad_channel_rows(path: Path) -> list[dict[str, str]]:
    # keep_channels: the reviewer kept these although screening rejected them
    return _label_table_rows(path, "--bad-channels", ("bad_channels",), optional=("keep_channels",))


def _span_rows(path: Path, flag: str = "--keep-spans",
               relative: bool = False) -> list[dict[str, Any]]:
    """The rows of a --keep-spans or --bad-spans table, onset and duration read as seconds.

    ``sub-01 | | rest | | 30 | 240`` comes back with ``"onset": 30.0, "duration": 240.0``.
    A ``relative`` onset is measured from a marker and may be negative.
    """
    rows = _label_table_rows(path, flag, ("onset", "duration"))
    for row in rows:
        for column in ("onset", "duration"):
            cell = row[column]
            try:
                row[column] = float(cell)
            except ValueError:
                row[column] = float("nan")
            if not np.isfinite(row[column]) or (column == "duration" and row[column] <= 0):
                raise ValueError(
                    f"{flag} table {path} line {row['line']}: {column} {cell!r} is not "
                    f"a {'positive ' if column == 'duration' else ''}number of seconds")
        if row["onset"] < 0 and not relative:
            raise ValueError(f"{flag} table {path} line {row['line']}: onset "
                             f"{row['onset']:g} s lies before the recording starts")
    return rows


def _bad_channel_row_matches(row: dict[str, Any], subject: str, entities: dict | None) -> bool:
    if row["participant_id"] and row["participant_id"] != subject.removeprefix("sub-"):
        return False
    for column in ("session", "task", "acq", "run"):
        if not row[column]:
            continue
        if entities is None:
            return False
        value = entities.get(_BAD_CHANNEL_KEYS[column][0])
        if value is None:
            return False
        # pybids reads run as a padded integer, so "1" and "01" are one run
        same = (int(row[column]) == int(value) if column == "run"
                else row[column] == str(value))
        if not same:
            return False
    return True


def _bad_channels_for(spec: str | None, subject: str, entities: dict | None = None) -> list[str]:
    """Resolve --bad-channels for one recording: a shared list, or a table of rows.

    e.g. "S1_D1,S2_D3" gives that list for every recording, while a table

    ::

        participant_id  session  run  bad_channels
        sub-01                        S1_D1,S2_D3
        sub-01          02       1    S4_D4
        sub-02                        S4_D4
                        02            S5_D5

    gives every recording of "sub-01" the first row and its ses-02 run-1 both rows, and
    every subject's ses-02 the last row. A recording's list is the union of every row that
    matches it, and a blank or absent participant_id, session, task, acq or run matches all.
    ``entities`` are the recording's parsed BIDS entities; without them only the rows that
    hold for every recording of the subject count. A subject no row matches has none. The
    prefixes are optional, so "01" and "sub-01" name the same subject.
    """
    if not spec:
        return []
    spec = str(spec)
    path = Path(spec)
    if not path.exists():
        return [c.strip() for c in spec.split(",") if c.strip()]

    return _channel_cells_for(path, subject, entities, "bad_channels")


def _keep_channels_for(spec: str | None, subject: str, entities: dict | None = None) -> list[str]:
    """The keep_channels cells of a --bad-channels table for one recording; a bare list keeps none."""
    if not spec or not Path(spec).exists():
        return []
    return _channel_cells_for(Path(spec), subject, entities, "keep_channels")


def _channel_cells_for(path: Path, subject: str, entities: dict | None, column: str) -> list[str]:
    rows = [row for row in _bad_channel_rows(path)
            if _bad_channel_row_matches(row, subject, entities)]
    return [c.strip() for c in ",".join(r[column] for r in rows).split(",") if c.strip()]


def _spans_for(spec: str | None, subject: str, entities: dict | None = None,
               flag: str = "--keep-spans",
               relative: bool = False) -> list[tuple[float, float]]:
    """The stretches a --keep-spans or --bad-spans table gives one recording, as ``(onset, duration)``.

    Rows are selected as the --bad-channels table's are, so a row with only ``task-rest``
    applies to every subject's rest recordings, and a recording gets every row that matches.
    No row means none: a --keep-spans recording is then used whole.
    """
    if not spec:
        return []
    return [(row["onset"], row["duration"])
            for row in _span_rows(Path(str(spec)), flag, relative)
            if _bad_channel_row_matches(row, subject, entities)]


def _refuse_unmatched_rows(rows: list[dict], flag: str, spec: str, recordings: list) -> None:
    for row in rows:
        if not any(_bad_channel_row_matches(row, ent["subject"], ent) for ent in recordings):
            named = " ".join(f"{_BAD_CHANNEL_KEYS[c][1]}{row[c]}" for c in _BAD_CHANNEL_KEYS
                             if row[c])
            raise SystemExit(
                f"[error] {flag} table {spec} line {row['line']} ({named}) matches no "
                f"recording in the dataset.")


def _refuse_unmatched_bad_channel_rows(spec: str | None, layout: Any) -> None:
    """Stop before any subject runs when a --bad-channels table row names no recording."""
    if not spec or not Path(str(spec)).exists():
        return
    try:
        rows = _bad_channel_rows(Path(str(spec)))
    except ValueError as err:
        raise SystemExit(f"[error] {err}") from None
    _refuse_unmatched_rows(rows, "--bad-channels", str(spec),
                           [f.get_entities() for f in layout.get(extension=".snirf")])


def _refuse_bad_span_rows(spec: str | None, layout: Any,
                          flag: str = "--keep-spans", marker: str | None = None) -> None:
    """Stop before any subject runs on a --keep-spans or --bad-spans row that matches no
    recording or starts outside one it matches, or on a matched recording without ``marker``."""
    if not spec:
        return
    path = Path(str(spec))
    if not path.exists():
        raise SystemExit(f"[error] {flag} table {spec} not found.")
    try:
        rows = _span_rows(path, flag, relative=marker is not None)
    except ValueError as err:
        raise SystemExit(f"[error] {err}") from None
    files = layout.get(extension=".snirf")
    _refuse_unmatched_rows(rows, flag, str(spec), [f.get_entities() for f in files])
    # file -> (length, the marker's onset or 0), both in seconds on its data axis
    headers: dict[str, tuple[float, float]] = {}
    for row in rows:
        for f in files:
            ent = f.get_entities()
            if not _bad_channel_row_matches(row, ent["subject"], ent):
                continue
            if f.path not in headers:
                raw = read_raw_snirf(f.path, preload=False, verbose="error")
                try:
                    origin = marker_onset(raw, marker) if marker else 0.0
                except ValueError:
                    raise SystemExit(
                        f"[error] --keep-spans-from marker {marker!r} is not among the "
                        f"annotations of {Path(f.path).name}, which {flag} table {spec} "
                        f"line {row['line']} names.") from None
                headers[f.path] = (raw.n_times / float(raw.info["sfreq"]), origin)
            length, origin = headers[f.path]
            onset = origin + row["onset"]
            measured = f" (from {marker!r} at {origin:g} s)" if marker else ""
            if onset >= length:
                raise SystemExit(
                    f"[error] {flag} table {spec} line {row['line']}: onset "
                    f"{row['onset']:g} s{measured} lies past the end of {Path(f.path).name} "
                    f"({length:g} s).")
            if onset < 0:
                raise SystemExit(
                    f"[error] {flag} table {spec} line {row['line']}: onset "
                    f"{row['onset']:g} s{measured} lies before the start of "
                    f"{Path(f.path).name}.")


def _make_prep_config(subject: str, session: str | None, args: dict[str, Any],
                      entities: dict | None = None,
                      channels_tsv_bads: list[str] | None = None) -> "PrepConfig":
    marker = args.get("keep_spans_from")
    keep_spans = _spans_for(args.get("keep_spans"), subject, entities, relative=bool(marker))
    # resolved here so the run record writes the value the method ran with
    motion = motion_settings(_v(args["motion_correction"]), args.get("motion_std_thresh"),
                             args.get("wavelet_iqr"))
    age = (get_participant_age(args["bids_dir"], subject) if dpf_from_age(args["dpf"])
           else None)
    return PrepConfig(
        subject=subject,
        session=session,
        dpf=args["dpf"],
        **({"age": age} if age is not None else {}),
        template=resolve_template(args.get("template"), age),
        sci_threshold=args["sci_threshold"],
        **({"psp_threshold": args["psp_threshold"]}
           if args.get("psp_threshold") is not None else {}),
        **({"min_good_frac": args["min_good_frac"]}
           if args.get("min_good_frac") is not None else {}),
        **({"screen_scope": args["screen_scope"]}
           if args.get("screen_scope") is not None else {}),
        motion_correction=_v(args["motion_correction"]),
        motion_std_thresh=motion.get("motion_std_thresh"),
        wavelet_iqr=motion.get("wavelet_iqr"),
        bad_channels=_bad_channels_for(args.get("bad_channels"), subject, entities),
        keep_channels=_keep_channels_for(args.get("bad_channels"), subject, entities),
        channels_tsv_bads=channels_tsv_bads or [],
        keep_spans=keep_spans,
        # a recording no row names is used whole, so no marker applies to it
        keep_spans_from=marker if keep_spans else None,
        bad_spans=_spans_for(args.get("bad_spans"), subject, entities, flag="--bad-spans"),
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
        **_app_options.separation_bands_from_args(args),
        ignore=[_v(ig) for ig in (args.get("ignore") or [])],
    )


def _emit_subject_report(subject, sub_dir, last_raw, last_result, prep_config, args, glm_est, dm, alff_df=None, fc_df=None, fc_hbr_df=None, fc_seed=None, fc_roi=None, high_pass=None, low_pass=None, after_haemo=None, roi_map=None, provenance_path=None, sqm_label=None, roi_map_name=None, filled_settings=None):
    from nirspipe.qc.subject.report import build_subject_report

    # rejected channels included, so the brain figures can draw them as rejected
    hbo_picks = mne.pick_types(last_result.raw_haemo.info, fnirs="hbo", exclude=[])
    coords_head = np.array([
        last_result.raw_haemo.info["chs"][i]["loc"][:3] for i in hbo_picks
    ])
    hbo_names = [last_result.raw_haemo.ch_names[i] for i in hbo_picks]
    bad_bases = {pair_of(bc) for bc in last_result.bad_channels}
    good_mask = np.array([pair_of(n) not in bad_bases for n in hbo_names])

    # off the processed data rather than the input, which lacks the spans prep added
    # (--keep-spans, --bad-spans, --gvtd-censor); on the data axis the figures draw
    bad_annots: dict = {}
    for start, stop, desc in bad_spans(last_result.raw_haemo):
        bad_annots.setdefault(desc, []).append((start, stop - start))

    return build_subject_report(
        subject=subject,
        raw_intensity=last_raw,
        raw_haemo=last_result.raw_haemo,
        sci_scores=last_result.sci_scores,
        bad_channels=last_result.bad_channels,
        config=prep_config,
        run_command=" ".join(sys.argv),
        out_path=sub_dir / report_name(sqm_label or f"sub-{subject}"),
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
        alff_df=alff_df,
        fc_df=fc_df,
        fc_hbr_df=fc_hbr_df,
        fc_seed=fc_seed,
        fc_roi=fc_roi,
        after_haemo=after_haemo,
        roi_map=roi_map,
        provenance_path=provenance_path,
        sqm_label=sqm_label,
        roi_map_name=roi_map_name,
        filled_settings=filled_settings,
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
    from nirspipe.pipeline.post_pipeline import run_post, write_combined_runs

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

            # the runs of one session and task, which --combine-runs joins once all are done
            by_task: dict = {}
            for snirf_path in preproc_files:
                src_entities = post_layout.parse_file_entities(str(snirf_path))
                # per file, for the session the file sits in; see the prep loop
                post_config = _build_post_config(subject, src_entities.get("session") or session,
                                                 args, toml, roi_map=roi_map)
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
                if src_entities.get("run") is not None:
                    # every entity but the run: files told apart by acq are not runs of one series
                    key = tuple(sorted((k, str(v)) for k, v in src_entities.items() if k != "run"))
                    by_task.setdefault(key, []).append((src_entities, post_config))

            for entries in by_task.values():
                config = entries[0][1]
                if not config.combine_runs or len(entries) < 2:
                    continue
                runs = [e for e, _ in sorted(entries, key=lambda e: int(e[0]["run"]))]
                try:
                    write_combined_runs(config, output_dir, mode, runs)
                except StageError as exc:
                    logger.warning("sub-%s | --combine-runs: task %s not joined: %s",
                                   subject, runs[0].get("task"), exc)

    return post_runs


def _warn_on_split_tree(output_dir: Path) -> None:
    """Say so when quality records sit in a subtree this aggregation does not reach.

    Two output directories means two records for one run, and the rule preferring the
    pipeline record over the `prep-raw` one can only choose between records one glob found.
    """
    from nirspipe.qc.subject.sqm_record import RECORD_SUFFIXES

    for sub in (output_dir / "qc", output_dir / "derivatives"):
        if sub.is_dir() and any(any(sub.glob(f"*/**/nirs/*{suffix}"))
                                for suffix in RECORD_SUFFIXES.values()):
            logger.warning(
                "quality records under %s are not part of this cohort page; point both "
                "`nirspipe` and `nirspipe-qc prep-raw` at one output directory, or aggregate "
                "that one separately with `nirspipe-qc cohort %s`", sub, sub)


def run_group_level(args: dict[str, Any]) -> None:
    """BIDS Apps `group` entry point: aggregates per-subject (and per-group hyper,
    if present) SQM JSONs into cohort HTML reports under <output_dir>."""
    from nirspipe.qc.hyper.group_hyper_writer import build_group_hyper_report
    from nirspipe.qc.subject.group_writer import build_group_raw_report
    from nirspipe.qc.subject.sqm_record import RECORD_SUFFIX

    output_dir = Path(args["output_dir"])
    _warn_on_split_tree(output_dir)

    logger.info("nirspipe group: aggregating individual SQMs from %s", output_dir)
    ind_path = build_group_raw_report(output_dir)
    if ind_path is not None:
        logger.info("  -> %s", ind_path)

    hyper_path = None
    if any([*output_dir.glob(f"group-*/nirs/*{RECORD_SUFFIX}"),
            *output_dir.glob(f"group-*/ses-*/nirs/*{RECORD_SUFFIX}")]):
        logger.info("nirspipe group: also aggregating hyperscanning SQMs")
        hyper_path = build_group_hyper_report(output_dir)
        logger.info("  -> %s", hyper_path)
    if ind_path is None and hyper_path is None:
        raise SystemExit(f"Error: no quality records under {output_dir}; run the "
                         "participant level into it first.")

