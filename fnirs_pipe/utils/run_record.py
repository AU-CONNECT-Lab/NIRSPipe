"""Write a per-subject TOML run record to output_dir/logs/."""

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fnirs_pipe.utils import unwrap_enum as _unwrap


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


def write_run_record(
    args: dict[str, Any],
    subject: str,
    timestamp: str,
    output_dir: Path,
    sub_dir: Path | None = None,
) -> None:
    """Write a TOML run record for one subject.

    Output: sub_dir/logs/sub-{subject}_{timestamp}.toml  (sub_dir defaults to output_dir)

    Sections:
      [environment]  — software versions + system info
      [execution]    — all CLI parameters (paths, filters, switches)
      [prep]         — preprocessing parameters
      [post]         — postprocessing parameters (only when --mode is set)

    # NOTE: a [qc] section with per-subject results (bad channels, SCI
    # distribution, motion summary) will be added here once the QC text
    # summary pipeline is implemented.
    """
    from fnirs_pipe.qc.boilerplate import collect_software_versions
    from fnirs_pipe import __version__

    versions = collect_software_versions()

    free_mem_gb = None
    try:
        import psutil
        free_mem_gb = round(psutil.virtual_memory().available / 1024 ** 3, 1)
    except ImportError:
        pass

    env: dict[str, Any] = {
        "fnirs_pipe_version": __version__,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    if free_mem_gb is not None:
        env["free_mem_gb"] = free_mem_gb

    for pkg_name, pkg_ver in versions.items():
        if pkg_name in ("python", "fnirs-pipe"):
            continue
        env[pkg_name.replace("-", "_") + "_version"] = pkg_ver

    execution: dict[str, Any] = {
        "run_uuid": timestamp,
        "run_timestamp": datetime.strptime(timestamp, "%Y%m%d_%H%M%S").isoformat(),
        "run_command": " ".join(sys.argv).replace("\\", "/"),
        "bids_dir": str(args["bids_dir"]).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "work_dir": str(args["work_dir"]).replace("\\", "/") if args.get("work_dir") else None,
        "log_dir": str(output_dir / "logs").replace("\\", "/"),
        "participant_label": [subject],
        "session_label": args.get("session_label"),
        "task_label": args.get("task_label"),
        "bids_filter_file": (
            str(args["bids_filter_file"]).replace("\\", "/")
            if args.get("bids_filter_file") else None
        ),
        "analysis_level": _unwrap(args.get("analysis_level"), "participant"),
        "n_jobs": args.get("n_jobs", 1),
        "skip_bids_validation": args.get("skip_bids_validation", False),
        "ignore": [_unwrap(ig) for ig in (args.get("ignore") or [])],
        "dry_run": args.get("dry_run", False),
        "verbose": args.get("verbose", False),
        "no_report": args.get("no_report", False),
    }

    prep: dict[str, Any] = {
        "dpf": args["dpf"],
        "sci_threshold": args["sci_threshold"],
        "motion_correction": _unwrap(args.get("motion_correction"), "tddr"),
    }

    sections = [
        _section("environment", env),
        _section("execution", execution),
        _section("prep", prep),
    ]

    mode = args.get("mode")
    if mode is not None:
        post: dict[str, Any] = {
            "mode": _unwrap(mode),
            "atlas": _unwrap(args.get("atlas")),
            "high_pass": args.get("high_pass"),
            "low_pass": args.get("low_pass"),
            "detrend": args.get("detrend", True),
            "short_channel_regression": _unwrap(
                args.get("short_channel_regression"), "none"
            ),
            "combine_runs": args.get("combine_runs", False),
            "hrf_model": _unwrap(args.get("hrf_model"), "canonical"),
            "glm_solver": _unwrap(args.get("glm_solver"), "ar-irls"),
            "contrast_file": (
                str(args["contrast_file"]).replace("\\", "/")
                if args.get("contrast_file") else None
            ),
            "connectivity_measure": _unwrap(
                args.get("connectivity_measure"), "correlation"
            ),
            "parcellate": args.get("parcellate", False),
            "min_coverage": args.get("min_coverage", 0.5),
            "custom_config": (
                str(args["custom_config"]).replace("\\", "/")
                if args.get("custom_config") else None
            ),
        }
        sections.append(_section("post", post))

    base = sub_dir if sub_dir is not None else output_dir
    out = base / "logs" / f"sub-{subject}_{timestamp}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
