"""``--population``: the values that follow the participants' age group, each overridden by its own flag."""

from __future__ import annotations

import math
from pathlib import Path

from nirspipe.utils import load_toml

POPULATIONS: "dict[str, dict]" = load_toml(
    Path(__file__).resolve().parent.parent / "data" / "populations.toml")
CHOICES = tuple(POPULATIONS)
BAND_KEYS = ("cardiac_l_freq", "cardiac_h_freq", "resp_l_freq", "resp_h_freq")


def population_values(name: str) -> dict:
    """One population's row; ValueError naming the choices for any other name."""
    if name not in POPULATIONS:
        raise ValueError(f"--population {name!r} is none of {', '.join(CHOICES)}")
    return POPULATIONS[name]


def age_span(name: str) -> str:
    """``"12 to 65 years"``, ``"under 1 year"`` or ``"65 years and over"``."""
    low, high = population_values(name)["ages"]
    if low == 0:
        return f"under {high:g} year{'s' if high != 1 else ''}"
    if math.isinf(high):
        return f"{low:g} years and over"
    return f"{low:g} to {high:g} years"
