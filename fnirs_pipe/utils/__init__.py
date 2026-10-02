"""Shared utilities."""

from pathlib import Path
from typing import Any

import numpy as np

_R_CLIP = 0.999999


def unwrap_enum(val: Any, default: Any = None) -> Any:
    if val is None:
        return default
    return val.value if hasattr(val, "value") else val


def is_optical_density(raw: Any) -> bool:
    """True if the data is already optical density (OD conversion done upstream)."""
    types = set(raw.get_channel_types())
    return "fnirs_od" in types and "fnirs_cw_amplitude" not in types


def fisher_r_to_z(r: Any) -> np.ndarray:
    """Fisher r-to-z, elementwise; r is clipped below +/-1 and a non-finite r gives NaN.

    z = arctanh(clip(r, -0.999999, 0.999999))
    """
    r = np.asarray(r, dtype=float)
    finite = np.isfinite(r)
    return np.where(finite, np.arctanh(np.clip(np.where(finite, r, 0.0), -_R_CLIP, _R_CLIP)), np.nan)


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as fh:
        return tomllib.load(fh)
