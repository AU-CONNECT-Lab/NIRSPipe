"""Write a per-subject TOML run record to output_dir/logs/."""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from nirspipe.io.derivatives import group_label
from nirspipe import __version__


# a run's id, made when it starts and read back wherever its time is printed
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# flag -> config field, where the two are spelled differently
_FIELD_OF = {"window_length": "qc_window_s", "aux_regressors": "aux"}
# recorded as typed: a table resolves per recording, after the record is written, and the
# bands are typed in mm where the configs hold metres
_AS_GIVEN = ("bad_channels", "keep_spans", "keep_spans_from", "bad_spans", "events_path",
             "contrast_file", "roi_mapping", "short_max_dist", "long_min_dist", "long_max_dist")


def _fwd(p: Any) -> str | None:
    """Forward-slash-normalized string of a path, or None if falsy."""
    return str(p).replace("\\", "/") if p else None


def _command_line() -> str:
    """The command this process was started with, slashes forward so it reads the same on every OS."""
    return " ".join(sys.argv).replace("\\", "/")


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, (int, float)):
        return str(v)
    elif isinstance(v, Path):
        return f'"{v.as_posix()}"'
    elif isinstance(v, str):
        escaped = v.replace("\\", "/").replace('"', '\\"')
        return f'"{escaped}"'
    else:
        escaped = str(v).replace("\\", "/").replace('"', '\\"')
        return f'"{escaped}"'


def _toml_value(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        if not v:
            return "[]"
        return "[ " + ", ".join(_toml_value(list(i)) if isinstance(i, (list, tuple))
                                else _toml_scalar(i) for i in v) + " ]"
    return _toml_scalar(v)


def _section(name: str, data: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    for k, v in data.items():
        if v is None:
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


def _settings(keys: tuple[str, ...], args: dict[str, Any], config: Any) -> dict[str, Any]:
    """Each flag as the run resolved it, spelled and scaled as typed, so the record reads back as --config.

    A flag with a config field takes the field, which holds the code default where the flag
    was left out; values supplied by --config never appear in argv, so the field is the only
    place they show. The rest, and the inputs in _AS_GIVEN, take the argument.
    """
    out: dict[str, Any] = {}
    for key in keys:
        field = _FIELD_OF.get(key, key)
        if key not in _AS_GIVEN and config is not None and hasattr(config, field):
            value = getattr(config, field)
        else:
            value = args.get(key)
        # a table is read where the record is read, so it needs a path that works from anywhere
        if isinstance(value, Path) or (key == "bad_channels" and value and Path(str(value)).exists()):
            value = Path(value).resolve().as_posix()
        elif key == "fir_delays" and value is not None:
            value = ",".join(str(d) for d in value)
        out[key] = value
    return out


def _settings_sections(settings: dict[str, Any], from_population, name: str) -> list[str]:
    """``[name]``, and ``[name_sources]`` naming what ``--population`` filled.

    A DPF the population set is left out: written as ``age`` it would read each participant's
    age on rerun, where the rerun's ``--population`` sets the same one again.
    """
    filled = list(from_population or ())
    if "dpf" in filled:
        settings = {k: v for k, v in settings.items() if k != "dpf"}
    sections = [_section(name, settings)]
    if filled:
        sections.append(_section(f"{name}_sources", {k: "population" for k in filled}))
    return sections


def _environment() -> dict[str, Any]:
    """Versions and machine, the same for every record a run writes bar free_mem_gb,
    which is read at the moment of writing."""
    from nirspipe.qc.boilerplate import collect_software_versions

    env: dict[str, Any] = {
        "nirspipe_version": __version__,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil
        env["free_mem_gb"] = round(psutil.virtual_memory().available / 1024 ** 3, 1)
    except ImportError:
        pass

    for pkg_name, pkg_ver in collect_software_versions().items():
        if pkg_name in ("python", "nirspipe"):
            continue
        env[pkg_name.replace("-", "_") + "_version"] = pkg_ver
    return env


def write_run_record(
    args: dict[str, Any],
    subject: str,
    timestamp: str,
    output_dir: Path,
    sub_dir: Path | None = None,
    prep_config: Any = None,
    post_config: Any = None,
    post_sources: dict[str, str] | None = None,
) -> Path:
    """Write a TOML run record for one subject, returning its path.

    Output: sub_dir/logs/<record_stem>.toml  (sub_dir defaults to output_dir)

    Sections:
      [environment]: software versions + system info
      [execution]:   invocation: the verbatim command, paths, selection filters
      [prep]:        preprocessing flags as resolved, keyed and scaled as typed
      [prep_sources]: the [prep] values --population filled, each named "population"
      [post]:        postprocessing flags as resolved (only with --mode)
      [post_sources]: which layer set each [post] value: cli, config, mode or default
      [outcome]:     success or failed, appended by write_run_outcome when the subject ends

    [execution] answers "what was run"; [prep]/[post] answer "what was used" and are what
    --config reads back, so the record reruns its run. A post setting the mode would fill
    is written "none" where it was switched off, or the rerun would get the mode's value;
    so is --short-channel, whose default is auto.

    # NOTE: a [qc] section with per-subject results (bad channels, SCI
    # distribution, motion summary) will be added here once the QC text
    # summary pipeline is implemented.
    """
    # the parser decides which flags --config reads back, so the record writes those
    from nirspipe.cli.run import config_keys, mode_defaults

    env = _environment()
    keys = config_keys()
    out = record_path(sub_dir if sub_dir is not None else output_dir, subject,
                      args.get("session_label"), args.get("task_label"))

    execution: dict[str, Any] = {
        "run_uuid": timestamp,
        "run_timestamp": datetime.strptime(timestamp, RUN_TIMESTAMP_FORMAT).isoformat(),
        "run_command": _command_line(),
        "bids_dir": _fwd(args["bids_dir"]),
        "output_dir": _fwd(output_dir),
        # the folder the record and the live log share
        "log_dir": _fwd(out.parent),
        "participant_label": [subject],
        "session_label": args.get("session_label"),
        "task_label": args.get("task_label"),
        "bids_filter_file": (Path(args["bids_filter_file"]).resolve().as_posix()
                             if args.get("bids_filter_file") else None),
        "analysis_level": args.get("analysis_level") or "participant",
        "nprocs": args.get("nprocs"),
        "skip_bids_validation": args.get("skip_bids_validation", False),
        "ignore": list(args.get("ignore") or []),
        "allow_cropped_input": args.get("allow_cropped_input", False),
        "config": Path(args["config"]).resolve().as_posix() if args.get("config") else None,
        "dry_run": args.get("dry_run", False),
        "verbose": args.get("verbose", False),
        "no_report": args.get("no_report", False),
    }

    sections = [
        _section("environment", env),
        _section("execution", execution),
    ]

    if prep_config is not None:
        sections += _settings_sections(_settings(keys["prep"], args, prep_config),
                                       args.get("from_population"), "prep")

    mode = args.get("mode")
    if mode is not None and post_config is not None:
        post = _settings(keys["post"], args, post_config)
        # short_channel's code default is auto, so a run told none says it too
        for key in (*mode_defaults(mode), "short_channel"):
            if post.get(key) is None:
                post[key] = "none"
        sections.append(_section("post", post))
        if post_sources is not None:
            from_file = "mode" in (args.get("config_applied") or ())
            sections.append(_section("post_sources", {
                "mode": "config" if from_file else "cli",
                **{k: post_sources[_FIELD_OF.get(k, k)] for k, v in post.items()
                   if v is not None and _FIELD_OF.get(k, k) in post_sources}}))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    return out


def write_raw_qc_record(args: dict[str, Any], keys: tuple[str, ...], subject: str,
                        output_dir: Path) -> Path:
    """Write one subject's raw QC run record, ``sub-<label>/logs/<record_stem>_desc-raw.toml``.

    [qc] holds the participant-level flags as typed and is what ``nirspipe-qc --config`` reads
    back; [qc_sources] names what --population filled; [execution] the invocation. Its own name, so a pipeline record in the same tree is
    never replaced.
    """
    timestamp = datetime.now()
    execution = {
        "run_timestamp": timestamp.isoformat(timespec="seconds"),
        "run_command": _command_line(),
        "bids_dir": _fwd(args["bids_dir"]),
        "output_dir": _fwd(output_dir),
        "analysis_level": "participant",
        "participant_label": [subject],
        "session_label": args.get("session_label"),
        "task_label": args.get("task_label"),
        "nprocs": args.get("nprocs"),
        "skip_bids_validation": args.get("skip_bids_validation", False),
    }
    sub_dir = output_dir / f"sub-{subject}"
    out = sub_dir / "logs" / (record_stem(subject, args.get("session_label"),
                                          args.get("task_label")) + "_desc-raw.toml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([_section("environment", _environment()),
                              _section("execution", execution),
                              *_settings_sections(_settings(keys, args, None),
                                                  args.get("from_population"), "qc")]),
                   encoding="utf-8")
    return out


def record_stem(subject: str, sessions: "list[str] | None" = None,
                tasks: "list[str] | None" = None) -> str:
    """The name a subject's run record and log share: the subject and the run's selection.

    ::

      sub 01                                        ->  sub-01
      sub 01, --session-label 2 1                   ->  sub-01_ses-1+2
      sub 01, --session-label 1, --task-label rest  ->  sub-01_ses-1_task-rest

    Runs of one subject started in parallel on different sessions or tasks each keep their
    own record, and rerunning a selection replaces its record.
    """
    stem = f"sub-{subject}"
    for entity, labels in (("ses", sessions), ("task", tasks)):
        if labels:
            stem += f"_{entity}-" + "+".join(sorted(set(labels)))
    return stem


def record_path(sub_dir: Path, subject: str, sessions: "list[str] | None" = None,
                tasks: "list[str] | None" = None) -> Path:
    return sub_dir / "logs" / f"{record_stem(subject, sessions, tasks)}.toml"


def recorded_post(sub_dir: Path, label: "str | None" = None) -> dict:
    """The ``[post]`` settings of the newest pipeline run record under ``sub_dir/logs`` covering ``label``, or {}.

    A record covers a run its ``[execution]`` task and session selection left in; with no
    ``label`` any record covers. Raw QC's own records are not the pipeline's.
    """
    import tomllib

    from nirspipe.io.derivatives import entity_of

    newest = ("", {})
    for path in sorted(Path(sub_dir, "logs").glob("*.toml")):
        if path.name.endswith("_desc-raw.toml"):
            continue
        try:
            record = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        execution = record.get("execution") or {}
        if label is not None and any(
                execution.get(key) and entity_of(label, entity) not in execution[key]
                for key, entity in (("task_label", "task"), ("session_label", "ses"))):
            continue
        stamp = str(execution.get("run_timestamp") or "")
        if stamp >= newest[0]:
            newest = (stamp, record.get("post") or {})
    return newest[1]


def error_line(exc: BaseException) -> str:
    """An exception as one line: its type and the first line of its message."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__


def write_run_outcome(record: Path, status: str, duration_s: float,
                      error: str | None = None) -> None:
    """Append [outcome] to a run record; a record without one never finished."""
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open("a", encoding="utf-8") as f:
        f.write("\n" + _section("outcome", {
            "status": status,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "duration_s": round(duration_s, 1),
            "error": error,
        }))


def write_group_run_record(
    args: dict[str, Any],
    group_id: str,
    task: str,
    timestamp: str,
    output_dir: Path,
    group_dir: Path,
    members: list[str] | None = None,
    session: str | None = None,
    device=None,
) -> Path:
    """Write a TOML run record for one hyperscanning group.

    Output: group_dir/logs/group-{group_id}[_ses-{session}]_task-{task}.toml, the mirror of a subject's
    logs/sub-{id}.toml.

    Sections:
      [environment] - software versions + system info, the same block a subject gets
      [execution]   - the verbatim command, the paths, and which group and task it covered
      [hyper]       - every option the invocation resolved to

    [hyper] is the whole argument dict rather than a config dataclass because the
    hyperscanning CLI has none: its options go straight from argparse into the report
    builder, so the args *are* the resolved settings.

    ``device`` is the card the transforms ran on: its versions join [environment], its check
    against the CPU [execution].
    """
    execution: dict[str, Any] = {
        "run_uuid": timestamp,
        "run_timestamp": datetime.strptime(timestamp, RUN_TIMESTAMP_FORMAT).isoformat(),
        "run_command": _command_line(),
        "output_dir": _fwd(output_dir),
        "log_dir": _fwd(group_dir / "logs"),
        "group_id": group_id,
        **({"session_label": session} if session else {}),
        "task_label": task,
        "participant_label": list(members or []),
        "nprocs": args.get("nprocs"),
        "device": args.get("device"),
        **((device.check or {"device_check_cells": 0}) if device is not None else {}),
        **({"config": _fwd(args["config"])} if args.get("config") else {}),
    }

    skip = {"func", "output_dir", "verbose", "config", "nprocs", "device"}
    hyper = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in args.items()
             if k not in skip and not isinstance(v, dict)}

    out = group_dir / "logs" / f"{group_label(group_id, task, session)}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        _section("environment", {**_environment(),
                                 **(device.environment if device is not None else {})}),
        _section("execution", execution),
        _section("hyper", hyper),
    ]), encoding="utf-8")
    return out
