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


def pair_of(ch: Any) -> str:
    """A channel's source-detector pair: its name without the wavelength or chromophore.

    "S1_D1 760" -> "S1_D1",  "S1_D1 hbo" -> "S1_D1",  "S1_D1" -> "S1_D1"
    """
    return str(ch).rsplit(" ", 1)[0]


def fisher_r_to_z(r: Any) -> np.ndarray:
    """Fisher r-to-z, elementwise; r is clipped below +/-1 and a non-finite r gives NaN.

    z = arctanh(clip(r, -0.999999, 0.999999))
    """
    r = np.asarray(r, dtype=float)
    finite = np.isfinite(r)
    return np.where(finite, np.arctanh(np.clip(np.where(finite, r, 0.0), -_R_CLIP, _R_CLIP)), np.nan)


def bare_roi_map(roi_map: "dict[str, list[str]]") -> "dict[str, list[str]]":
    """ROI entries as S-D pairs: a trailing " hbo" / " hbr" is dropped, repeats collapse.

    {"L": ["S1_D1 hbo", "S1_D1", "S2_D2"]}  ->  {"L": ["S1_D1", "S2_D2"]}
    """
    def _bare(ch: str) -> str:
        return ch[:-4] if ch.endswith((" hbo", " hbr")) else ch
    return {roi: list(dict.fromkeys(_bare(str(ch)) for ch in chs)) for roi, chs in roi_map.items()}


def roi_overlaps(roi_map: "dict[str, list[str]]") -> "dict[str, list[str]]":
    """Each channel more than one ROI lists, with the ROIs that list it, in map order."""
    listed: dict[str, list[str]] = {}
    for roi, chs in bare_roi_map(roi_map).items():
        for ch in chs:
            listed.setdefault(ch, []).append(roi)
    return {ch: rois for ch, rois in listed.items() if len(rois) > 1}


def load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]
    with open(path, "rb") as fh:
        return tomllib.load(fh)
