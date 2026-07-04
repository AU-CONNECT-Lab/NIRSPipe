"""Shared utilities."""

from pathlib import Path
from typing import Any


def unwrap_enum(val: Any, default: Any = None) -> Any:
    if val is None:
        return default
    return val.value if hasattr(val, "value") else val


def is_optical_density(raw: Any) -> bool:
    """True if the data is already optical density (OD conversion done upstream)."""
    types = set(raw.get_channel_types())
    return "fnirs_od" in types and "fnirs_cw_amplitude" not in types


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as fh:
        return tomllib.load(fh)
