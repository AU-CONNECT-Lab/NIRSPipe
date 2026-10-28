"""The differential pathlength factor a run converts with: the numbers given, or one per wavelength from age."""

from __future__ import annotations

import mne
import numpy as np

from nirspipe.qc.boilerplate.notes import section_note
from nirspipe.utils.logging import get_logger

logger = get_logger("pipeline.dpf")

AGE = "age"
# where Scholkmann & Wolf (2013) recommend the equation (section 3): 0 to 70 years, 690 to
# 832 nm and "safe" on to 950 nm
AGE_RANGE = (0.0, 70.0)
WAVELENGTH_RANGE = (690.0, 950.0)


def age_dpf(wavelength_nm: float, age_years: float) -> float:
    """DPF of the frontal head at one wavelength and age.

    DPF(λ, A) = 223.3 + 0.05624 A^0.8493 − 5.723e-7 λ³ + 0.001245 λ² − 0.9025 λ, with λ in nm and
    A in years: Equation 7 of Scholkmann & Wolf (2013), J Biomed Opt 18(10):105004.
    """
    lam = float(wavelength_nm)
    return (223.3 + 0.05624 * float(age_years) ** 0.8493
            - 5.723e-7 * lam ** 3 + 0.001245 * lam ** 2 - 0.9025 * lam)


def dpf_caveat(age: float, wavelengths_nm: "list[float]") -> "str | None":
    """What of an age DPF lies outside the equation's recommended range, as a page note, or None."""
    outside = ([f"age {age:g} years"] if not AGE_RANGE[0] <= age <= AGE_RANGE[1] else [])
    outside += [f"{w:g} nm" for w in wavelengths_nm
                if not WAVELENGTH_RANGE[0] <= w <= WAVELENGTH_RANGE[1]]
    return section_note("caveat.dpf_age_range", outside=", ".join(outside)) if outside else None


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
    if caveat := dpf_caveat(age, wavelengths(raw)):
        logger.warning(caveat)
    return [age_dpf(w, age) for w in wavelengths(raw)]


def dpf_text(dpf: "list", wavelengths_nm: "list[float]", age: "float | None") -> str:
    """The factors as the reports print them: ``age 30 (6.30, 5.23)``, or the numbers given."""
    if not dpf_from_age(dpf):
        return ", ".join(f"{float(d):g}" for d in dpf)
    if age is None or not np.isfinite(age):
        return AGE
    factors = ", ".join(f"{age_dpf(w, age):.2f}" for w in wavelengths_nm)
    return f"{AGE} {age:g} ({factors})"


def as_ppf(dpf: "list[float]") -> "float | list[float]":
    """MNE's ``ppf``: one value for every wavelength, or the list."""
    return dpf[0] if len(dpf) == 1 else dpf
