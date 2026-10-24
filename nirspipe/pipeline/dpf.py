"""The differential pathlength factor a run converts with: the numbers given, or one per wavelength from age."""

from __future__ import annotations

import mne
import numpy as np

AGE = "age"


def age_dpf(wavelength_nm: float, age_years: float) -> float:
    """DPF of the frontal head at one wavelength and age.

    DPF(λ, A) = 223.3 + 0.05624 A^0.8493 − 5.723e-7 λ³ + 0.001245 λ² − 0.9025 λ, with λ in nm and
    A in years: Equation 7 of Scholkmann & Wolf (2013), J Biomed Opt 18(10):105004.
    """
    lam = float(wavelength_nm)
    return (223.3 + 0.05624 * float(age_years) ** 0.8493
            - 5.723e-7 * lam ** 3 + 0.001245 * lam ** 2 - 0.9025 * lam)


def dpf_from_age(dpf: "list") -> bool:
    """Whether ``--dpf`` asked for the age equation; the word mixed with numbers is refused."""
    words = [d for d in dpf if isinstance(d, str)]
    if words and len(dpf) > 1:
        raise ValueError(f"--dpf takes numbers or the single word {AGE!r}, not both: {dpf}")
    return bool(words)


def wavelengths(raw: mne.io.Raw) -> list[float]:
    """The recording's wavelengths in nm, ascending, the order MNE pairs factors with."""
    picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
    return sorted({float(raw.info["chs"][p]["loc"][9]) for p in picks})


def resolve_dpf(dpf: "list", raw: mne.io.Raw, age: "float | None") -> list[float]:
    """The factors a run converts with: the numbers given, or :func:`age_dpf` per wavelength."""
    if not dpf_from_age(dpf):
        return [float(d) for d in dpf]
    if age is None or not np.isfinite(age):
        raise ValueError("--dpf age needs the participant's age, and none was found")
    return [age_dpf(w, age) for w in wavelengths(raw)]


def as_ppf(dpf: "list[float]") -> "float | list[float]":
    """MNE's ``ppf``: one value for every wavelength, or the list."""
    return dpf[0] if len(dpf) == 1 else dpf
