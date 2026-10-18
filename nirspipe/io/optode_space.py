"""Where a recording's optodes sit on fsaverage, decided once per run and recorded beside it.

SNIRF does not say which space its optode coordinates are in, and a dataset's
``_coordsystem.json`` is often wrong about it. So the placement is decided from the file:
its labelled landmarks registered onto the same points on fsaverage, or, when that leaves
the cap off the scalp, the reading of the coordinates (head frame or already on fsaverage)
that puts the optodes on the fsaverage scalp. Prep writes the result as the run's channel
table, and every later reader takes the placement from that record instead of deciding again.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from nirspipe import __version__
from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import entity_of, read_json, write_sidecar_json
from nirspipe.io.naming import derivative_path
from nirspipe.io.tables import write_tsv
from nirspipe.utils.lineage import path_from
from nirspipe.utils.logging import get_logger

logger = get_logger("io.optode_space")

SPACE = "fsaverage"
# a placed cap sits this close to the fsaverage scalp, as the median over its optodes
SCALP_FIT_MAX_MM = 5.0
_MIN_LANDMARKS = 3
_UNIT_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
# spellings of the three fiducials that standard_1005 does not use itself
_LANDMARK_ALIASES = {"nz": "nasion", "al": "lpa", "ar": "rpa"}
_HELD_KEY = "nirspipe_placement"
_M_TO_MM = np.diag([1000.0, 1000.0, 1000.0, 1.0])


# ---- fsaverage ----

def _fsaverage_dir() -> str:
    try:
        return mne.datasets.fetch_fsaverage(verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"fsaverage is needed to place the optodes on the brain and could not be loaded "
            f"({exc}); run mne.datasets.fetch_fsaverage() once on a machine with internet "
            f"access") from exc


@lru_cache(maxsize=1)
def head_to_mri() -> np.ndarray:
    """fsaverage head->MRI (surface RAS) as a 4x4 in metres; raises when fsaverage cannot be loaded."""
    trans = mne.read_trans(os.path.join(_fsaverage_dir(), "bem", "fsaverage-trans.fif"))
    return np.asarray(trans["trans"], float)


@lru_cache(maxsize=1)
def _scalp() -> cKDTree:
    """The fsaverage scalp surface, in mm, for nearest-point queries."""
    surface = mne.read_bem_surfaces(
        os.path.join(_fsaverage_dir(), "bem", "fsaverage-head.fif"), verbose=False)[0]
    return cKDTree(surface["rr"] * 1000)


@lru_cache(maxsize=1)
def _reference_landmarks() -> dict[str, np.ndarray]:
    """standard_1005 positions and the three fiducials on fsaverage, in mm, by lower-case label."""
    montage = mne.channels.make_standard_montage("standard_1005")
    # the montage sits in its own frame, so its native->head step comes before fsaverage's
    trans = head_to_mri() @ mne.channels.compute_native_head_t(montage)["trans"]
    pos = montage.get_positions()
    points = {**{k.lower(): v for k, v in pos["ch_pos"].items()},
              **{k: pos[k] for k in ("nasion", "lpa", "rpa")}}
    return {k: mne.transforms.apply_trans(trans, v) * 1000 for k, v in points.items()}


def _scalp_fit(points_mm: np.ndarray) -> float:
    return float(np.median(_scalp().query(points_mm)[0]))


# ---- The placement ----

@dataclass(frozen=True, eq=False)
class Placement:
    """How a recording's optode coordinates reach fsaverage surface RAS (MNI305).

    ``method`` is ``"landmarks"`` (the file's landmarks registered onto fsaverage's),
    ``"head"`` or ``"mni"`` (the coordinates read in that frame), or ``"none"`` (no reading
    puts the cap on the scalp). ``transform`` takes coordinates as MNE holds them, in metres,
    to fsaverage in millimetres.
    """
    method: str
    transform: np.ndarray | None = None
    scalp_fit_mm: float | None = None
    landmarks: int = 0
    note: str | None = None

    @property
    def placed(self) -> bool:
        return self.transform is not None

    def apply(self, coords_m: np.ndarray) -> np.ndarray:
        if self.transform is None:
            raise StageError(f"optodes were not placed on fsaverage: {self.note}")
        return mne.transforms.apply_trans(self.transform, np.asarray(coords_m, float))

    def to_json(self) -> dict:
        return {"method": self.method,
                "landmarks_used": self.landmarks,
                "scalp_fit_mm": self.scalp_fit_mm,
                "transform": None if self.transform is None else self.transform.tolist(),
                "note": self.note}

    @classmethod
    def from_json(cls, record: dict) -> "Placement":
        transform = record.get("transform")
        return cls(method=record["method"],
                   transform=None if transform is None else np.asarray(transform, float),
                   scalp_fit_mm=record.get("scalp_fit_mm"),
                   landmarks=int(record.get("landmarks_used") or 0),
                   note=record.get("note"))


def _text(value) -> str:
    """A SNIRF string field, stored as bytes, as a one-element array, or as either."""
    if np.ndim(value):
        value = np.asarray(value).ravel()[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_probe(snirf_path) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    """Optode positions and labelled landmarks off a SNIRF probe, in metres as MNE scales them.

    ``LengthUnit "mm"``, ``sourcePos3D [[-30, 60, 40]]`` -> optode ``[-0.03, 0.06, 0.04]``
    """
    with h5py.File(snirf_path, "r") as f:
        probe = f["nirs/probe"]
        scale = _UNIT_TO_M[_text(f["nirs/metaDataTags/LengthUnit"][()]).strip()]
        if "sourcePos3D" not in probe or "detectorPos3D" not in probe:
            return None, {}
        optodes = np.vstack([probe["sourcePos3D"][()], probe["detectorPos3D"][()]])[:, :3]
        landmarks = {}
        if "landmarkPos3D" in probe and "landmarkLabels" in probe:
            for label, pos in zip(probe["landmarkLabels"][()], probe["landmarkPos3D"][()]):
                landmarks[_text(label).strip().lower()] = np.asarray(pos[:3], float) * scale
    # an optode number no channel uses is stored as a NaN row
    optodes = optodes[np.isfinite(optodes).all(axis=1)] * scale
    return (optodes if len(optodes) else None), landmarks


def copy_landmarks(source_path, dest_path) -> None:
    """Give a SNIRF rewritten through MNE its source's probe landmarks, labels and all.

    ``source landmarkLabels [Nz, Cz, F3]`` in mm -> the same three in ``dest``, in its own unit

    MNE keeps a landmark's position but not its name (anything but the three fiducials reads
    back as EEG001, EEG002, ...), and mne-nirs writes template fiducials where the source had
    none, so a cropped or re-marked recording would otherwise be placed by landmarks it never
    had. A source without landmarks leaves the copy without any.
    """
    with h5py.File(source_path, "r") as src, h5py.File(dest_path, "r+") as dest:
        ratio = (_UNIT_TO_M[_text(src["nirs/metaDataTags/LengthUnit"][()]).strip()]
                 / _UNIT_TO_M[_text(dest["nirs/metaDataTags/LengthUnit"][()]).strip()])
        source, target = src["nirs/probe"], dest["nirs/probe"]
        for key in ("landmarkPos3D", "landmarkPos2D", "landmarkLabels"):
            if key in target:
                del target[key]
            if key not in source:
                continue
            value = source[key][()]
            if key.startswith("landmarkPos"):
                value = np.asarray(value, float).copy()
                # the last column of a landmark position is its index, not a coordinate
                value[:, :int(key[-2])] *= ratio
            target.create_dataset(key, data=value)


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation, translation and one scale taking ``src`` onto ``dst`` in least squares, as a 4x4."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - src_mean, dst - dst_mean
    u, s, vt = np.linalg.svd(b.T @ a / len(src))
    # keeps a mirror image from passing as a rotation
    flip = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ flip @ vt
    scale = np.trace(np.diag(s) @ flip) / a.var(axis=0).sum()
    out = np.eye(4)
    out[:3, :3] = scale * rotation
    out[:3, 3] = dst_mean - scale * rotation @ src_mean
    return out


def place_optodes(snirf_path) -> Placement:
    """Decide where a recording's optodes sit on fsaverage, from the SNIRF file alone.

    Landmarks first: every landmark label the file shares with standard_1005 (the three
    fiducials included) is registered onto its fsaverage position, and the result is kept
    when the optodes land on the scalp. Otherwise the coordinates are read as head frame and
    as already on fsaverage, and the reading that lands on the scalp is kept. A file no
    reading places, or one whose probe cannot be read, gets ``method "none"`` with the reason.

    ::

      10-10 labelled landmarks, cap on the scalp once registered       -> "landmarks"
      fiducials from another template tilt the cap, raw values fit     -> "mni"
      digitiser coordinates never registered to a head                 -> "none"
    """
    try:
        optodes, landmarks = _read_probe(snirf_path)
    except (OSError, KeyError, ValueError) as exc:
        return Placement("none", note=f"the probe could not be read ({exc})")
    if optodes is None:
        return Placement("none", note="the file has no 3-D optode positions")
    try:
        return _place(optodes, landmarks)
    except RuntimeError as exc:
        # fsaverage could not be loaded; the run goes on without a placement
        return Placement("none", note=str(exc))


def _place(optodes: np.ndarray, landmarks: dict[str, np.ndarray]) -> Placement:
    reference = _reference_landmarks()
    shared = {}
    for label, pos in landmarks.items():
        key = label if label in reference else _LANDMARK_ALIASES.get(label)
        if key in reference:
            shared[key] = (pos, reference[key])
    tried = None
    if len(shared) >= _MIN_LANDMARKS:
        src, dst = (np.array(side) for side in zip(*shared.values()))
        transform = _similarity(src, dst)
        fit = _scalp_fit(mne.transforms.apply_trans(transform, optodes))
        if fit <= SCALP_FIT_MAX_MM:
            return Placement("landmarks", transform, fit, len(shared))
        tried = (f"registering {len(shared)} landmarks left the optodes {fit:.1f} mm off the "
                 f"fsaverage scalp")

    readings = {"mni": _M_TO_MM, "head": _M_TO_MM @ head_to_mri()}
    fits = {name: _scalp_fit(mne.transforms.apply_trans(t, optodes)) for name, t in readings.items()}
    best = min(fits, key=fits.get)
    if fits[best] <= SCALP_FIT_MAX_MM:
        return Placement(best, readings[best], fits[best], len(shared), note=tried)
    reason = (f"no reading puts the optodes on the fsaverage scalp: as head frame "
              f"{fits['head']:.1f} mm off, as fsaverage {fits['mni']:.1f} mm off")
    return Placement("none", landmarks=len(shared),
                     note=f"{tried}; {reason}" if tried else reason)


# ---- The run's record ----

def channel_coords_path(run_file) -> Path:
    """The channel table recording a run's placement, beside any file of that run.

    ``.../sub-01_ses-02_task-rest_run-01_desc-preproc_nirs.snirf``
        -> ``.../sub-01_ses-02_task-rest_run-01_space-fsaverage_channels.tsv``
    """
    run_file = Path(run_file)
    keep = {name: entity_of(run_file.name, key)
            for name, key in (("subject", "sub"), ("session", "ses"), ("task", "task"),
                              ("run", "run"))}
    return run_file.parent / derivative_path(".", "channels", ".tsv", space=SPACE, **keep).name


def write_channel_coords(raw: mne.io.Raw, placement: Placement, out_path: Path,
                         sources: "list[str | Path]" = ()) -> Path:
    """One row per fNIRS channel at its source-detector midpoint on fsaverage, and the placement.

    ``S1_D1 hbo`` with source (0.01, 0.05, 0.06) and detector (0.04, 0.05, 0.06) m
        -> ``name x y z`` = ``S1_D1 hbo`` and the transformed midpoint in mm; n/a when unplaced
    """
    picks = mne.pick_types(raw.info, fnirs=True, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    mid = np.array([(raw.info["chs"][i]["loc"][3:6] + raw.info["chs"][i]["loc"][6:9]) / 2
                    for i in picks]).reshape(-1, 3)
    xyz = placement.apply(mid) if placement.placed else np.full((len(names), 3), np.nan)
    write_tsv(pd.DataFrame({"name": names, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}),
              out_path)
    column = "Channel position on fsaverage: the midpoint of its source and detector"
    write_sidecar_json(Path(out_path), {
        "pipeline_version": __version__,
        "step": "optode_placement",
        "Sources": [Path(s).as_posix() for s in sources],
        "NIRSCoordinateSystem": SPACE,
        "NIRSCoordinateSystemDescription": "fsaverage surface RAS, which is MNI305",
        "NIRSCoordinateUnits": "mm",
        **{axis: {"Description": f"{column}, {axis} axis", "Units": "mm"} for axis in "xyz"},
        "optode_placement": placement.to_json(),
    })
    return Path(out_path)


def read_placement(run_file) -> Placement:
    """The placement prep recorded for a run, or StageError when the tree predates the record."""
    record_path = channel_coords_path(run_file).with_suffix(".json")
    record = read_json(record_path).get("optode_placement")
    if record is None:
        raise StageError(f"{record_path.name} is missing beside {Path(run_file).name}; this "
                         f"derivative tree predates optode placement, rerun nirspipe")
    return Placement.from_json(record)


def copy_placement(source_file, dest_file) -> None:
    """Carry a run's placement record to a file written under another run's name.

    ``sub-01_task-rest_desc-preproc`` cut into ``sub-01_task-early_desc-preproc``
        -> ``sub-01_task-early_space-fsaverage_channels.tsv`` and ``.json`` beside it
    """
    src, dest = channel_coords_path(source_file), channel_coords_path(dest_file)
    if src == dest:
        return
    read_placement(source_file)
    for suffix in (".tsv", ".json"):
        dest.with_suffix(suffix).write_bytes(src.with_suffix(suffix).read_bytes())


def hold_placement(raw: mne.io.Raw, placement: Placement) -> mne.io.Raw:
    """Attach a placement to the Raw, for the life of the object; returns raw."""
    raw.info["temp"] = {**(raw.info.get("temp") or {}), _HELD_KEY: placement}
    return raw


def _from_frame(info) -> Placement:
    """An in-memory Raw with no file behind it: MNE's own frame label says how to read it."""
    from mne._fiff.constants import FIFF

    chs = info["chs"]
    frame = chs[0]["coord_frame"] if len(chs) else FIFF.FIFFV_COORD_UNKNOWN
    if frame == FIFF.FIFFV_COORD_HEAD:
        return Placement("head", _M_TO_MM @ head_to_mri())
    if frame == FIFF.FIFFV_COORD_MRI:
        return Placement("mni", _M_TO_MM)
    return Placement("none", note="an in-memory recording in no known frame")


def placement_of(raw: mne.io.Raw) -> Placement:
    """The placement of a Raw's optodes: held on it, recorded for its run, or read off its file.

    A derivative's own landmarks are the ones mne-nirs wrote, not the recording's, so a
    derivative always takes the record prep left beside it.
    """
    held = (raw.info.get("temp") or {}).get(_HELD_KEY)
    if held is not None:
        return held
    path = path_from(raw) or (raw.filenames[0] if raw.filenames else None)
    if path is None:
        return _from_frame(raw.info)
    if entity_of(Path(path).name, "desc"):
        return read_placement(path)
    return place_optodes(path)
