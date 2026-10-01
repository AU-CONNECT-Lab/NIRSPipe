"""Write a per-subject TOML run record to output_dir/logs/."""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fnirs_pipe.utils import unwrap_enum as _unwrap
from fnirs_pipe import __version__


def _fwd(p: Any) -> str | None:
    """Forward-slash-normalized string of a path, or None if falsy."""
    return str(p).replace("\\", "/") if p else None


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
        return "[ " + ", ".join(_toml_scalar(i) for i in v) + " ]"
    return _toml_scalar(v)


def _section(name: str, data: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    for k, v in data.items():
        if v is None:
            continue
        lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


def _config_section(config: Any) -> dict[str, Any]:
    """Scalar fields of a config dataclass, minus what [execution] already carries.

    Read off the config rather than the CLI args, since values supplied by --config TOML
    never appear in argv. Iterating fields also means new config options are recorded
    without touching this file.
    """
    from dataclasses import fields

    skip = {"subject", "session", "dry_run"}
    out: dict[str, Any] = {}
    for f in fields(config):
        if f.name in skip:
            continue
        value = getattr(config, f.name)
        if isinstance(value, dict):      # roi_map / contrast_def are structures, not settings
            continue
        out[f.name] = list(value) if isinstance(value, tuple) else value
    return out


def _environment() -> dict[str, Any]:
    """Versions and machine, the same for every record a run writes bar free_mem_gb,
    which is read at the moment of writing."""
    from fnirs_pipe.qc.boilerplate import collect_software_versions

    env: dict[str, Any] = {
        "fnirs_pipe_version": __version__,
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
        if pkg_name in ("python", "fnirs-pipe"):
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
      [prep]:        preprocessing parameters as resolved, from PrepConfig
      [post]:        postprocessing parameters as resolved, from PostConfig (only with --mode)
      [post_sources]: which layer set each [post] value: cli, config, mode or default

    [execution] answers "what was run" and stays copy-pasteable; [prep]/[post] answer
    "what was used" and come from the config objects the pipeline actually received.

    # NOTE: a [qc] section with per-subject results (bad channels, SCI
    # distribution, motion summary) will be added here once the QC text
    # summary pipeline is implemented.
    """
    env = _environment()

    execution: dict[str, Any] = {
        "run_uuid": timestamp,
        "run_timestamp": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").isoformat(),
        "run_command": " ".join(sys.argv).replace("\\", "/"),
        "bids_dir": _fwd(args["bids_dir"]),
        "output_dir": _fwd(output_dir),
        "work_dir": _fwd(args.get("work_dir")),
        "log_dir": _fwd(output_dir / "logs"),
        "participant_label": [subject],
        "session_label": args.get("session_label"),
        "task_label": args.get("task_label"),
        "bids_filter_file": _fwd(args.get("bids_filter_file")),
        "analysis_level": _unwrap(args.get("analysis_level"), "participant"),
        "n_jobs": args.get("n_jobs", 1),
        "skip_bids_validation": args.get("skip_bids_validation", False),
        "ignore": [_unwrap(ig) for ig in (args.get("ignore") or [])],
        "dry_run": args.get("dry_run", False),
        "verbose": args.get("verbose", False),
        "no_report": args.get("no_report", False),
    }

    sections = [
        _section("environment", env),
        _section("execution", execution),
    ]

    if prep_config is not None:
        sections.append(_section("prep", _config_section(prep_config)))

    mode = args.get("mode")
    if mode is not None and post_config is not None:
        # mode is not a PostConfig field; it selects which branch run_post takes
        post = _config_section(post_config)
        sections.append(_section("post", {"mode": _unwrap(mode), **post}))
        if post_sources is not None:
            sections.append(_section("post_sources", {
                "mode": "cli",
                **{k: post_sources.get(k, "default") for k, v in post.items() if v is not None}}))

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
) -> Path:
    """Write a TOML run record for one hyperscanning group.

    Output: group_dir/logs/group-{group_id}_task-{task}.toml, the mirror of a subject's
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
        "run_timestamp": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").isoformat(),
        "run_command": " ".join(sys.argv).replace("\\", "/"),
        "output_dir": _fwd(output_dir),
        "log_dir": _fwd(group_dir / "logs"),
        "group_id": group_id,
        "task_label": task,
        "participant_label": list(members or []),
    }

    skip = {"func", "output_dir", "verbose"}
    hyper = {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in args.items()
             if k not in skip and not isinstance(v, dict)}

    out = group_dir / "logs" / f"group-{group_id}_task-{task}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([
        _section("environment", _environment()),
        _section("execution", execution),
        _section("hyper", hyper),
    ]), encoding="utf-8")
    return out
