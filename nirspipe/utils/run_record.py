"""Write a per-subject TOML run record to output_dir/logs/."""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from nirspipe.io.derivatives import group_label
from nirspipe.utils import unwrap_enum as _unwrap
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
    if isinstance(v, list):
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
        value = _unwrap(value)
        # a table is read where the record is read, so it needs a path that works from anywhere
        if isinstance(value, Path) or (key == "bad_channels" and value and Path(str(value)).exists()):
            value = Path(value).resolve().as_posix()
        elif key == "fir_delays" and value is not None:
            value = ",".join(str(d) for d in value)
        out[key] = value
    return out


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
) -> None:
    """Write a TOML run record for one subject.

    Output: sub_dir/logs/sub-{subject}.toml  (sub_dir defaults to output_dir)

    Sections:
      [environment]: software versions + system info
      [execution]:   invocation: the verbatim command, paths, selection filters
      [prep]:        preprocessing flags as resolved, keyed and scaled as typed
      [post]:        postprocessing flags as resolved (only with --mode)
      [post_sources]: which layer set each [post] value: cli, config, mode or default

    [execution] answers "what was run"; [prep]/[post] answer "what was used" and are what
    --config reads back, so the record reruns its run. A post setting the mode would fill
    is written "none" where it was switched off, or the rerun would get the mode's value.

    # NOTE: a [qc] section with per-subject results (bad channels, SCI
    # distribution, motion summary) will be added here once the QC text
    # summary pipeline is implemented.
    """
    # the parser decides which flags --config reads back, so the record writes those
    from nirspipe.cli.run import config_keys, mode_defaults

    env = _environment()
    keys = config_keys()

    execution: dict[str, Any] = {
        "run_uuid": timestamp,
        "run_timestamp": datetime.strptime(timestamp, RUN_TIMESTAMP_FORMAT).isoformat(),
        "run_command": _command_line(),
        "bids_dir": _fwd(args["bids_dir"]),
        "output_dir": _fwd(output_dir),
        "work_dir": _fwd(args.get("work_dir")),
        "log_dir": _fwd(output_dir / "logs"),
        "participant_label": [subject],
        "session_label": args.get("session_label"),
        "task_label": args.get("task_label"),
        "bids_filter_file": (Path(args["bids_filter_file"]).resolve().as_posix()
                             if args.get("bids_filter_file") else None),
        "analysis_level": _unwrap(args.get("analysis_level"), "participant"),
        "n_jobs": args.get("n_jobs", 1),
        "skip_bids_validation": args.get("skip_bids_validation", False),
        "ignore": [_unwrap(ig) for ig in (args.get("ignore") or [])],
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
        sections.append(_section("prep", _settings(keys["prep"], args, prep_config)))

    mode = _unwrap(args.get("mode"))
    if mode is not None and post_config is not None:
        post = _settings(keys["post"], args, post_config)
        for key in mode_defaults(mode):
            if post.get(key) is None:
                post[key] = "none"
        sections.append(_section("post", post))
        if post_sources is not None:
            from_file = "mode" in (args.get("config_applied") or ())
            sections.append(_section("post_sources", {
                "mode": "config" if from_file else "cli",
                **{k: post_sources[_FIELD_OF.get(k, k)] for k, v in post.items()
                   if v is not None and _FIELD_OF.get(k, k) in post_sources}}))

    base = sub_dir if sub_dir is not None else output_dir
    out = base / "logs" / f"sub-{subject}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")


def write_group_run_record(
    args: dict[str, Any],
    group_id: str,
    task: str,
    timestamp: str,
    output_dir: Path,
    group_dir: Path,
    members: list[str] | None = None,
    session: str | None = None,
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
    }

    skip = {"func", "output_dir", "verbose"}
    hyper = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in args.items()
             if k not in skip and not isinstance(v, dict)}

    out = group_dir / "logs" / f"{group_label(group_id, task, session)}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        _section("environment", _environment()),
        _section("execution", execution),
        _section("hyper", hyper),
    ]), encoding="utf-8")
    return out
