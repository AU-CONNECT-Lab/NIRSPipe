"""The head a run's optodes are placed on and drawn over: fsaverage, or one of MNE's infant templates.

Each infant template (2 weeks to 2 years) has its own scalp, cortex, fiducials and 10-10
positions. It is fetched on first use into the folder fsaverage lives in, so every renderer
finds both under one ``subjects_dir``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.defaults import HEAD_SIZE_DEFAULT
from scipy.spatial import cKDTree

FSAVERAGE = "fsaverage"
# --template value -> (MNE subject name, age in years)
INFANT = {
    "2wk": ("ANTS2-0Weeks3T", 14 / 365.25),
    "1mo": ("ANTS1-0Months3T", 1 / 12),
    "2mo": ("ANTS2-0Months3T", 2 / 12),
    "3mo": ("ANTS3-0Months3T", 3 / 12),
    "4.5mo": ("ANTS4-5Months3T", 4.5 / 12),
    "6mo": ("ANTS6-0Months3T", 6 / 12),
    "7.5mo": ("ANTS7-5Months3T", 7.5 / 12),
    "9mo": ("ANTS9-0Months3T", 9 / 12),
    "10.5mo": ("ANTS10-5Months3T", 10.5 / 12),
    "12mo": ("ANTS12-0Months3T", 1.0),
    "15mo": ("ANTS15-0Months3T", 15 / 12),
    "18mo": ("ANTS18-0Months3T", 1.5),
    "2yr": ("ANTS2-0Years3T", 2.0),
}
CHOICES = (FSAVERAGE, *INFANT)
# from this age on fsaverage stands for the head; between it and the oldest infant template
# no template fits, and the run asks for one
ADULT_AGE = 18.0


# ---- Which template ----

def template_for_age(age_years: float) -> str:
    """The template an age takes: the infant template nearest in age, or fsaverage from 18.

    ``0.33`` -> ``"4.5mo"``; ``30`` -> ``"fsaverage"``; ``5`` -> ValueError
    """
    if age_years >= ADULT_AGE:
        return FSAVERAGE
    oldest = max(age for _, age in INFANT.values())
    if age_years > oldest:
        raise ValueError(
            f"no head template fits age {age_years:g} years: MNE's infant templates end at 2 "
            f"years and fsaverage is an adult head; pass --template ({', '.join(CHOICES)})")
    return min(INFANT, key=lambda name: abs(INFANT[name][1] - age_years))


def resolve_template(template: "str | None", age: "float | None") -> str:
    """The template a participant runs on: the one given, else by ``age``, else fsaverage.

    ``age`` is the participant's age when ``--dpf age`` reads it, None otherwise.
    """
    if template is not None:
        if template not in CHOICES:
            raise ValueError(f"--template {template!r} is none of {', '.join(CHOICES)}")
        return template
    return FSAVERAGE if age is None else template_for_age(age)


def space_of(template: str) -> str:
    """The BIDS ``space`` label: fsaverage, or the MNE subject name with its hyphen as ``p``.

    ``"4.5mo"`` -> ``"ANTS4p5Months3T"``; a BIDS label is alphanumeric
    """
    return space_of_subject(FSAVERAGE if template == FSAVERAGE else INFANT[template][0])


def space_of_subject(subject: str) -> str:
    """The ``space`` label of a template's folder name: ``ANTS4-5Months3T`` -> ``ANTS4p5Months3T``."""
    return subject.replace("-", "p")


def template_of_space(space: str) -> str:
    """The ``--template`` value a ``space`` label names; ValueError for any other label."""
    for template in CHOICES:
        if space_of(template) == space:
            return template
    raise ValueError(f"space {space!r} is no head template this package knows")


def subject_of(space: str) -> str:
    """The template's folder name under ``subjects_dir``: ``ANTS4p5Months3T`` -> ``ANTS4-5Months3T``."""
    template = template_of_space(space)
    return FSAVERAGE if template == FSAVERAGE else INFANT[template][0]


def describe(space: str) -> str:
    """One phrase naming the template, for the coordinate sidecar and the page notes."""
    template = template_of_space(space)
    if template == FSAVERAGE:
        return "fsaverage surface RAS, which is MNI305"
    return f"surface RAS of MNE's {INFANT[template][0]} infant template ({template})"


# ---- Files ----

@lru_cache(maxsize=1)
def subjects_dir() -> Path:
    """The folder fsaverage is in, fetching it on first use; the infant templates go beside it."""
    try:
        return Path(mne.datasets.fetch_fsaverage(verbose=False)).parent
    except Exception as exc:
        raise RuntimeError(
            f"fsaverage is needed to place the optodes on the brain and could not be loaded "
            f"({exc}); run mne.datasets.fetch_fsaverage() once on a machine with internet "
            f"access") from exc


@lru_cache(maxsize=None)
def template_dir(space: str) -> Path:
    """The template's FreeSurfer folder, fetched on first use."""
    root = subjects_dir()
    template = template_of_space(space)
    if template == FSAVERAGE:
        return root / FSAVERAGE
    try:
        return root / mne.datasets.fetch_infant_template(template, root, verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"the {template} infant template could not be loaded ({exc}); on a machine with "
            f"internet access run mne.datasets.fetch_infant_template({template!r}, "
            f"subjects_dir) and copy its {INFANT[template][0]} folder into {root}") from exc


def source_space_path(space: str) -> Path:
    """The cortical source space a projection spreads channel values over."""
    folder = template_dir(space)
    name = ("fsaverage-ico-5-src.fif" if space == FSAVERAGE
            else f"{subject_of(space)}-oct-6-src.fif")
    return folder / "bem" / name


# ---- Geometry ----

@lru_cache(maxsize=None)
def _fiducials(space: str) -> dict[str, np.ndarray]:
    """Nasion, LPA and RPA on the template's surface RAS, in metres."""
    points, _ = mne.io.read_fiducials(
        template_dir(space) / "bem" / f"{subject_of(space)}-fiducials.fif")
    names = {mne.io.constants.FIFF.FIFFV_POINT_NASION: "nasion",
             mne.io.constants.FIFF.FIFFV_POINT_LPA: "lpa",
             mne.io.constants.FIFF.FIFFV_POINT_RPA: "rpa"}
    return {names[p["ident"]]: np.asarray(p["r"], float) for p in points}


@lru_cache(maxsize=None)
def head_to_mri(space: str = FSAVERAGE) -> np.ndarray:
    """The template's head frame to its surface RAS, as a 4x4 in metres.

    fsaverage ships its own; an infant template's follows from its fiducials, which define
    the head frame.
    """
    if space == FSAVERAGE:
        trans = mne.read_trans(template_dir(space) / "bem" / "fsaverage-trans.fif")
        return np.asarray(trans["trans"], float)
    fid = _fiducials(space)
    return np.linalg.inv(mne.transforms.get_ras_to_neuromag_trans(
        fid["nasion"], fid["lpa"], fid["rpa"]))


@lru_cache(maxsize=None)
def scalp(space: str) -> cKDTree:
    """The template's scalp surface, in mm, for nearest-point queries."""
    surface = mne.read_bem_surfaces(
        template_dir(space) / "bem" / f"{subject_of(space)}-head.fif", verbose=False)[0]
    return cKDTree(surface["rr"] * 1000)


# FreeSurfer annotation -> the channel-table column naming a channel's region in it
CORTEX_ATLASES = {"aparc": "aparc", "PALS_B12_Brodmann": "brodmann"}
_HEMIS = ("lh", "rh")


def cortex_atlases(space: str) -> list[str]:
    """The annotations a template's channels are labelled from: aparc on every template, Brodmann on fsaverage, which alone ships it."""
    return list(CORTEX_ATLASES) if space == FSAVERAGE else ["aparc"]


@lru_cache(maxsize=None)
def _pial(space: str) -> cKDTree:
    """Both hemispheres' pial vertices, lh then rh, in surface RAS mm."""
    surf = template_dir(space) / "surf"
    return cKDTree(np.vstack([mne.read_surface(surf / f"{hemi}.pial")[0] for hemi in _HEMIS]))


@lru_cache(maxsize=None)
def _vertex_regions(space: str, atlas: str) -> np.ndarray:
    """Each pial vertex's region in ``atlas``, lh then rh, as ``name-hemi``; None off every region."""
    import nibabel.freesurfer as fs

    regions = []
    for hemi in _HEMIS:
        labels, _, names = fs.read_annot(str(template_dir(space) / "label" / f"{hemi}.{atlas}.annot"))
        named = np.array([f"{n.decode()}-{hemi}" for n in names] + [None], dtype=object)
        # -1, a vertex in no region, indexes the trailing None
        regions.append(named[labels])
    return np.concatenate(regions)


def nearest_cortex_regions(space: str, points_mm: np.ndarray) -> dict[str, list]:
    """Each point's region in every atlas the template ships, read at the pial vertex nearest it.

    ``{"aparc": ["precentral-lh", None], "brodmann": ["Brodmann.4-lh", None]}`` for a point
    over the left motor strip and one with no position (NaN).
    """
    points = np.asarray(points_mm, float).reshape(-1, 3)
    placed = np.isfinite(points).all(axis=1)
    out = {}
    for atlas in cortex_atlases(space):
        column = [None] * len(points)
        if placed.any():
            nearest = _pial(space).query(points[placed])[1]
            for i, region in zip(np.flatnonzero(placed), _vertex_regions(space, atlas)[nearest]):
                column[i] = region
        out[CORTEX_ATLASES[atlas]] = column
    return out


@lru_cache(maxsize=None)
def reference_points(space: str) -> dict[str, np.ndarray]:
    """10-10 (fsaverage: 10-5) positions and the three fiducials on the template, in mm, by lower-case label."""
    if space == FSAVERAGE:
        montage = mne.channels.make_standard_montage("standard_1005")
        # the montage sits in its own frame, so its native->head step comes before fsaverage's
        trans = head_to_mri(space) @ mne.channels.compute_native_head_t(montage)["trans"]
        pos = montage.get_positions()
        points = {**{k.lower(): v for k, v in pos["ch_pos"].items()},
                  **{k: pos[k] for k in ("nasion", "lpa", "rpa")}}
        return {k: mne.transforms.apply_trans(trans, v) * 1000 for k, v in points.items()}
    # the 10-5 table names its in-between points by their two neighbours (af3_f1), which
    # standard_1005 does not, so only the 10-10 names are shared
    table = pd.read_csv(template_dir(space) / "montages" / "10-5_electrodes.tsv", sep="\t")
    points = {str(n).lower(): np.array([x, y, z], float)
              for n, x, y, z in table[["name", "x", "y", "z"]].itertuples(index=False)}
    points.pop("nz", None)
    return {**points, **{k: v * 1000 for k, v in _fiducials(space).items()}}


@lru_cache(maxsize=None)
def from_fsaverage(space: str) -> np.ndarray:
    """fsaverage surface RAS (mm) to the template's, the affine that best carries their shared 10-10 points.

    The identity for fsaverage. A cap stored on an adult head reaches an infant template this
    way, by where its optodes sit in the 10-10 system rather than by their distance in mm.
    """
    if space == FSAVERAGE:
        return np.eye(4)
    adult, infant = reference_points(FSAVERAGE), reference_points(space)
    shared = sorted(set(adult) & set(infant))
    src = np.column_stack([np.array([adult[k] for k in shared]), np.ones(len(shared))])
    dst = np.array([infant[k] for k in shared])
    affine = np.eye(4)
    affine[:3] = np.linalg.lstsq(src, dst, rcond=None)[0].T
    return affine


def linear_scale(space: str) -> float:
    """How much smaller or larger the template's head is than fsaverage's, as one length ratio."""
    return float(np.cbrt(abs(np.linalg.det(from_fsaverage(space)[:3, :3]))))


def head_sphere(space: str) -> "np.ndarray | None":
    """The sphere a head map projects onto, in the template's head frame (m); None keeps MNE's default.

    MNE's default is sized for an adult head; an infant template's is shrunk by the template's
    size, so a cap sits inside its head outline as it does on fsaverage.
    """
    if space == FSAVERAGE:
        return None
    return np.array([0.0, 0.0, 0.0, HEAD_SIZE_DEFAULT * linear_scale(space)])
