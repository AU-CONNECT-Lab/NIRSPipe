"""Shared utilities."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

_R_CLIP = 0.999999

# channels each member must keep in a region for its ROI cell to be reported
ROI_MIN_CHANNELS = 1

# seconds within which two times are one edge: far under a sample, far over round-off
TIME_TOL_S = 1e-6


def unwrap_enum(val: Any, default: Any = None) -> Any:
    if val is None:
        return default
    return val.value if hasattr(val, "value") else val


def is_optical_density(raw: Any) -> bool:
    """True if the data is already optical density (OD conversion done upstream)."""
    types = set(raw.get_channel_types())
    return "fnirs_od" in types and "fnirs_cw_amplitude" not in types


# lower-case prefixes of annotations that are not events
NOT_MARKERS = ("bad", "edge")


def is_marker(description: Any) -> bool:
    """Whether an annotation is an event: not a BAD_ span, nor the EDGE boundary mne writes at a join.

    "task" -> True,  "BAD_gvtd" -> False,  "EDGE boundary" -> False
    """
    return not str(description).lower().startswith(NOT_MARKERS)


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


def zscore_rows(x: np.ndarray) -> np.ndarray:
    """Each row to zero mean and unit deviation, a flat row left alone rather than divided by 0."""
    mu  = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mu) / np.where(std < 1e-12, 1.0, std)


# the multipletests method names; none leaves the raw p as the only one written
P_CORRECTIONS = ("none", "fdr_bh", "fdr_by", "holm", "bonferroni")


def correct_p(frame: "pd.DataFrame", keys: "list[str]", method: str, p: str = "p",
              eligible: "pd.Series | None" = None) -> "pd.DataFrame":
    """Add ``family`` and, unless ``method`` is none, a ``p_<method>`` column beside ``p``.

    ::

      4 channels in one condition, holm  ->  family 4, p kept, p_holm added

    Rows whose ``p`` is absent, or that ``eligible`` leaves out, take no part in any family
    and keep a blank corrected p.
    """
    from statsmodels.stats.multitest import multipletests

    if method not in P_CORRECTIONS:
        raise ValueError(f"unknown p correction {method!r}; one of {P_CORRECTIONS}")
    frame = frame.copy()
    col = f"p_{method}"
    if method != "none":
        frame[col] = np.nan
    frame["family"] = 0
    if p not in frame.columns or not keys:
        return frame
    tested = frame[p].notna() if eligible is None else frame[p].notna() & eligible
    for _, part in frame.groupby(keys, dropna=False):
        usable = part[tested[part.index]]
        if usable.empty:
            continue
        if method != "none":
            frame.loc[usable.index, col] = multipletests(usable[p], method=method)[1]
        frame.loc[part.index, "family"] = len(usable)
    return frame


def cone_margin_s(band_fmin: float, factor: float = 2.0) -> float:
    r"""Seconds of recording to keep either side of a window so its band is clear of the cone.

    ::

      cone_margin_s(0.06)  ->  47.1     cone_margin_s(0.02)  ->  141.4

    A wavelet coefficient near the end of a series is computed partly against the padding
    beyond it, and the cone of influence is where that contamination has decayed to
    :math:`e^{-2}`. For the Morlet wavelet the cone reaches :math:`\sqrt{2}\,P` in from each
    edge at period :math:`P`, so the lowest frequency a band mean uses, ``band_fmin``, is the
    one that reaches furthest:

    .. math::

        \mathrm{margin} = \mathrm{factor} \times \frac{\sqrt{2}}{f_{\min}}

    Returned in seconds, so a caller crops ``[t0 - margin, t1 + margin]`` and windows the
    result back to ``[t0, t1]``.
    """
    if band_fmin <= 0:
        raise ValueError(f"band_fmin must be positive, got {band_fmin}")
    return float(factor) * np.sqrt(2.0) / float(band_fmin)


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
