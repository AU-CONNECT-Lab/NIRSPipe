"""Optode placement on fsaverage: decided from the recording's own file, recorded once, read back.

Every case plants the answer: optodes and landmarks put at known points on fsaverage, then
moved by a known rotation, scale and shift, or read in a known frame. A placement is right
when it brings the moved coordinates back to where they were planted.
"""

import json

import h5py
import mne
import numpy as np
import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.derivatives import write_dataset_description
from nirspipe.io.optode_space import (
    _reference_landmarks,
    channel_coords_path,
    head_to_mri,
    place_optodes,
    placement_of,
)
from nirspipe.io.snirf import read_snirf, write_snirf
from nirspipe.io.tables import read_table
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import _write_dataset_root, synth_raw

# (source, detector) per synthetic pair S1_D1 ... S4_D4; the fifth, short pair is built below
_LONG_PAIRS = [("f3", "fc3"), ("f4", "fc4"), ("c3", "cp3"), ("c4", "cp4")]
_LANDMARKS = ["Nz", "LPA", "RPA", "Cz", "Iz", "T7", "T8", "Fpz", "Oz"]
_BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.1, resp_h_freq=0.4)


def _planted_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    """Source and detector of each pair on fsaverage, in mm."""
    ref = _reference_landmarks()
    pairs = [(ref[s], ref[d]) for s, d in _LONG_PAIRS]
    pairs.append((ref["cz"], ref["cz"] + np.array([0.0, 8.0, 0.0])))
    return pairs


def _move() -> np.ndarray:
    """A rotation of 20 degrees about z and 10 about x, scale 1.08, shift (20, -10, 30) mm, in metres."""
    a, b = np.deg2rad(20.0), np.deg2rad(10.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    out = np.eye(4)
    out[:3, :3] = 1.08 * rz @ rx
    out[:3, 3] = [0.02, -0.01, 0.03]
    return out


def _write(path, frame: np.ndarray, landmarks: "list[str] | None" = None,
           landmark_shift_mm: "dict[str, np.ndarray] | None" = None, unit: str = "m",
           duration: float | None = None):
    """A recording whose planted mm positions are put through ``frame`` (mm on fsaverage -> m in file)."""
    raw = synth_raw("01", "rest", **({"duration": duration} if duration else {}))
    pairs = _planted_pairs()
    for ch, name in zip(raw.info["chs"], raw.ch_names):
        src, det = pairs[int(name.split("_")[0][1:]) - 1]
        src, det = (mne.transforms.apply_trans(frame, p / 1000) for p in (src, det))
        ch["loc"][3:6], ch["loc"][6:9], ch["loc"][0:3] = src, det, (src + det) / 2
    write_snirf(raw, path)
    ref = _reference_landmarks()
    with h5py.File(path, "r+") as f:
        probe = f["nirs/probe"]
        for key in ("landmarkPos3D", "landmarkLabels"):
            if key in probe:
                del probe[key]
        if landmarks:
            shift = landmark_shift_mm or {}
            # Nz is the file's spelling of the nasion, which standard_1005 lists as a fiducial
            pos = np.array([mne.transforms.apply_trans(
                frame, (ref["nasion" if k == "Nz" else k.lower()] + shift.get(k, 0.0)) / 1000)
                for k in landmarks])
            probe.create_dataset("landmarkPos3D",
                                 data=np.column_stack([pos, np.arange(1, len(pos) + 1)]))
            probe.create_dataset("landmarkLabels", data=[k.encode() for k in landmarks])
        if unit == "mm":
            for key in ("sourcePos3D", "detectorPos3D", "landmarkPos3D"):
                if key in probe:
                    value = probe[key][()]
                    value[:, :3] *= 1000
                    del probe[key]
                    probe.create_dataset(key, data=value)
            del f["nirs/metaDataTags/LengthUnit"]
            f["nirs/metaDataTags"].create_dataset("LengthUnit", data=b"mm")
    return path


def _sources_mm(raw, placement) -> np.ndarray:
    """Each pair's source as the placement puts it, in pair order."""
    seen = {}
    for ch, name in zip(raw.info["chs"], raw.ch_names):
        seen.setdefault(name.split(" ")[0], placement.apply(ch["loc"][3:6][None, :])[0])
    return np.array([seen[f"S{i}_D{i}"] for i in range(1, len(_planted_pairs()) + 1)])


def _planted_sources() -> np.ndarray:
    return np.array([s for s, _ in _planted_pairs()])


def test_landmarks_bring_a_moved_cap_back_where_it_was_planted(tmp_path):
    path = _write(tmp_path / "sub-01_task-rest_nirs.snirf", _move(), landmarks=_LANDMARKS)
    placement = place_optodes(path)
    assert placement.method == "landmarks" and placement.landmarks == len(_LANDMARKS)
    raw = mne.io.read_raw_snirf(str(path), verbose="error")
    np.testing.assert_allclose(_sources_mm(raw, placement), _planted_sources(), atol=1e-6)


def test_a_cap_already_on_fsaverage_without_landmarks_reads_as_it_stands(tmp_path):
    path = _write(tmp_path / "sub-01_task-rest_nirs.snirf", np.eye(4))
    placement = place_optodes(path)
    assert placement.method == "mni"
    raw = mne.io.read_raw_snirf(str(path), verbose="error")
    np.testing.assert_allclose(_sources_mm(raw, placement), _planted_sources(), atol=1e-6)


def test_head_frame_coordinates_are_read_as_head_frame(tmp_path):
    path = _write(tmp_path / "sub-01_task-rest_nirs.snirf", np.linalg.inv(head_to_mri()))
    placement = place_optodes(path)
    assert placement.method == "head"
    raw = mne.io.read_raw_snirf(str(path), verbose="error")
    np.testing.assert_allclose(_sources_mm(raw, placement), _planted_sources(), atol=1e-6)


def test_fiducials_from_another_template_fall_back_to_reading_the_coordinates(tmp_path):
    # three fiducials 15 mm low, as a template with its own fiducial convention puts them
    low = {k: np.array([0.0, 0.0, -15.0]) for k in ("Nz", "LPA", "RPA")}
    path = _write(tmp_path / "sub-01_task-rest_nirs.snirf", np.eye(4),
                  landmarks=["Nz", "LPA", "RPA"], landmark_shift_mm=low)
    placement = place_optodes(path)
    assert placement.method == "mni"
    assert "3 landmarks" in placement.note


def test_coordinates_far_from_any_head_are_not_placed(tmp_path):
    far = np.eye(4)
    far[:3, 3] = [0.15, 0.10, 0.20]
    placement = place_optodes(_write(tmp_path / "sub-01_task-rest_nirs.snirf", far))
    assert placement.method == "none" and not placement.placed
    assert "no reading" in placement.note


def test_a_probe_with_only_2d_positions_is_not_placed(tmp_path):
    path = _write(tmp_path / "sub-01_task-rest_nirs.snirf", np.eye(4))
    with h5py.File(path, "r+") as f:
        for key in ("sourcePos3D", "detectorPos3D"):
            f["nirs/probe"].create_dataset(key.replace("3D", "2D"), data=f["nirs/probe"][key][()][:, :2])
            del f["nirs/probe"][key]
    placement = place_optodes(path)
    assert placement.method == "none" and "3-D" in placement.note


@pytest.fixture(scope="module")
def prepped(tmp_path_factory):
    """Prep run on a moved, landmarked recording; the output tree and the source path."""
    root = tmp_path_factory.mktemp("placement")
    _write_dataset_root(root / "raw", ["01"])
    source = _write(root / "raw" / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf", _move(),
                    landmarks=_LANDMARKS)
    write_dataset_description(root / "out", source=root / "raw")
    config = PrepConfig(subject="01", dpf=[6.0], sci_threshold=0.8, motion_correction="none",
                        **_BANDS)
    raw = mne.io.read_raw_snirf(str(source), preload=True, verbose="error")
    run_prep(raw, config, root / "out", source_entities={"task": "rest"}, source_path=source)
    return root / "out" / "sub-01" / "nirs", source


def test_prep_writes_each_channel_at_its_planted_midpoint(prepped):
    nirs_dir, _ = prepped
    table = read_table(nirs_dir / "sub-01_task-rest_space-fsaverage_channels.tsv").set_index("name")
    pairs = _planted_pairs()
    for i, (src, det) in enumerate(pairs, start=1):
        for chroma in ("hbo", "hbr"):
            np.testing.assert_allclose(table.loc[f"S{i}_D{i} {chroma}", ["x", "y", "z"]].to_numpy(float),
                                       (src + det) / 2, atol=1e-6)
    side = json.loads((nirs_dir / "sub-01_task-rest_space-fsaverage_channels.json").read_text())
    assert side["NIRSCoordinateSystem"] == "fsaverage" and side["NIRSCoordinateUnits"] == "mm"
    assert side["optode_placement"]["method"] == "landmarks"


def test_a_derivative_is_placed_by_the_record_not_by_its_own_landmarks(prepped):
    # mne-nirs writes no landmark the recording had, so only the record can place this file
    nirs_dir, source = prepped
    od = read_snirf(nirs_dir / "sub-01_task-rest_desc-od_nirs.snirf")
    from_record = placement_of(od)
    assert from_record.method == "landmarks"
    raw = mne.io.read_raw_snirf(str(source), verbose="error")
    np.testing.assert_allclose(_sources_mm(od, from_record), _sources_mm(raw, placement_of(raw)),
                               atol=1e-6)


def test_a_derivative_without_the_record_is_refused(prepped, tmp_path):
    nirs_dir, _ = prepped
    lone = tmp_path / "sub-01_task-rest_desc-od_nirs.snirf"
    lone.write_bytes((nirs_dir / lone.name).read_bytes())
    with pytest.raises(StageError, match="predates optode placement"):
        placement_of(read_snirf(lone))


def test_the_record_is_named_for_the_run_not_the_stage():
    assert channel_coords_path("d/sub-01_ses-2_task-rest_run-01_desc-preproc_nirs.snirf").name == \
        "sub-01_ses-2_task-rest_run-01_space-fsaverage_channels.tsv"


def test_a_cropped_recording_keeps_the_landmarks_that_place_it(tmp_path):
    from nirspipe.pipeline.crop import crop_snirf_from_path

    # in mm, so the copy has to convert into the metres mne-nirs writes
    _write_dataset_root(tmp_path / "raw", ["01"])
    source = _write(tmp_path / "raw" / "sub-01" / "nirs" / "sub-01_task-rest_nirs.snirf",
                    _move(), landmarks=_LANDMARKS, unit="mm", duration=60.0)
    (out,) = crop_snirf_from_path(source, tmp_path / "derivatives", "01", tmin=10.0, tmax=40.0)
    with h5py.File(out, "r") as f:
        labels = [x.decode() for x in f["nirs/probe/landmarkLabels"][()]]
    assert labels == _LANDMARKS
    placement = place_optodes(out)
    assert placement.method == "landmarks"
    cropped = mne.io.read_raw_snirf(str(out), verbose="error")
    np.testing.assert_allclose(_sources_mm(cropped, placement), _planted_sources(), atol=1e-6)
