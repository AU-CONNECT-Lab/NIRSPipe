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
from nirspipe.io.head_template import FSAVERAGE, reference_points
from nirspipe.io.optode_space import (
    Placement,
    channel_coords_path,
    head_to_mri,
    place_optodes,
    placement_of,
    read_placement,
    write_channel_coords,
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
    ref = reference_points(FSAVERAGE)
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
    ref = reference_points(FSAVERAGE)
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
    assert side["optode_placement"]["space"] == "fsaverage"


def test_prep_names_each_channels_region_in_both_fsaverage_atlases(prepped):
    nirs_dir, _ = prepped
    path = nirs_dir / "sub-01_task-rest_space-fsaverage_channels.tsv"
    table = read_table(path)
    assert table.columns.tolist() == ["name", "x", "y", "z", "aparc", "brodmann"]
    assert table[["aparc", "brodmann"]].notna().all().all()
    # one position per pair, so both chromophores sit over one region
    pair = table["name"].str.rsplit(" ", n=1).str[0]
    assert (table.groupby(pair)[["aparc", "brodmann"]].nunique() == 1).all().all()
    side = json.loads(path.with_suffix(".json").read_text())
    assert {"aparc", "brodmann"} <= set(side)


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


# ---- the group copy ----

def _member(derivs, subject, x, method="landmarks", space="fsaverage", regions=None):
    """A member's preproc file and the record prep would leave beside it, hand-written."""
    nirs = derivs / subject / "nirs"
    nirs.mkdir(parents=True, exist_ok=True)
    (nirs / f"{subject}_task-rest_desc-preproc_nirs.snirf").write_bytes(b"")
    head = "".join(f"\t{name}" for name in regions or {})
    cells = "".join(f"\t{value}" for value in (regions or {}).values())
    (nirs / f"{subject}_task-rest_space-{space}_channels.tsv").write_text(
        f"name\tx\ty\tz{head}\nS1_D1 hbo\t{x}\t2.0\t3.0{cells}\n"
        f"S1_D1 hbr\t{x}\t2.0\t3.0{cells}\n")
    (nirs / f"{subject}_task-rest_space-{space}_channels.json").write_text(json.dumps(
        {"optode_placement": {"method": method, "space": space, "landmarks_used": 3,
                              "scalp_fit_mm": 1.0, "transform": np.eye(4).tolist(),
                              "note": None}}))


def test_a_group_table_keeps_each_members_positions_under_their_subject_id(tmp_path):
    from nirspipe.io.groups import GroupEntry
    from nirspipe.pipeline.hyper.group_quality import write_group_channel_coords

    derivs, out = tmp_path / "derivatives", tmp_path / "hyper"
    write_dataset_description(derivs)
    write_dataset_description(out, source=derivs)
    _member(derivs, "sub-01", 10.0)
    _member(derivs, "sub-02", 12.5, method="mni")
    group = [GroupEntry("G1", "sub-01", "rest"), GroupEntry("G1", "sub-02", "rest")]

    path = write_group_channel_coords(derivs, out, group, "preproc")

    assert path == out / "group-G1" / "nirs" / "group-G1_task-rest_space-fsaverage_channels.tsv"
    table = read_table(path)
    assert table.columns.tolist() == ["subject_id", "space", "name", "x", "y", "z"]
    assert set(table["space"]) == {"fsaverage"}
    assert table.set_index(["subject_id", "name"])["x"].to_dict() == {
        ("sub-01", "S1_D1 hbo"): 10.0, ("sub-01", "S1_D1 hbr"): 10.0,
        ("sub-02", "S1_D1 hbo"): 12.5, ("sub-02", "S1_D1 hbr"): 12.5}
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["NIRSCoordinateSystem"] == "fsaverage"
    assert {k: v["method"] for k, v in side["optode_placement"].items()} == {
        "sub-01": "landmarks", "sub-02": "mni"}


def test_a_group_table_carries_the_members_regions(tmp_path):
    from nirspipe.io.groups import GroupEntry
    from nirspipe.pipeline.hyper.group_quality import write_group_channel_coords

    derivs, out = tmp_path / "derivatives", tmp_path / "hyper"
    write_dataset_description(derivs)
    write_dataset_description(out, source=derivs)
    _member(derivs, "sub-01", 10.0, regions={"aparc": "precentral-lh", "brodmann": "Brodmann.4-lh"})
    _member(derivs, "sub-02", 12.5, space=_INFANT_SPACE, regions={"aparc": "postcentral-lh"})
    group = [GroupEntry("G1", "sub-01", "rest"), GroupEntry("G1", "sub-02", "rest")]

    path = write_group_channel_coords(derivs, out, group, "preproc")

    table = read_table(path).set_index(["subject_id", "name"])
    assert table.columns.tolist() == ["space", "x", "y", "z", "aparc", "brodmann"]
    assert table.loc[("sub-01", "S1_D1 hbo"), "brodmann"] == "Brodmann.4-lh"
    # an infant template has no Brodmann atlas, so its members' cells are blank
    assert table.loc[("sub-02", "S1_D1 hbo"), "aparc"] == "postcentral-lh"
    assert table.loc[[("sub-02", "S1_D1 hbo")], "brodmann"].isna().all()
    side = json.loads(path.with_suffix(".json").read_text())
    assert "Brodmann" in side["brodmann"]["Description"] and "aparc" in side["aparc"]["Description"]


def test_a_group_with_a_member_from_an_older_tree_is_refused(tmp_path):
    from nirspipe.io.groups import GroupEntry
    from nirspipe.pipeline.hyper.group_quality import write_group_channel_coords

    derivs = tmp_path / "derivatives"
    _member(derivs, "sub-01", 10.0)
    (derivs / "sub-02" / "nirs").mkdir(parents=True)
    (derivs / "sub-02" / "nirs" / "sub-02_task-rest_desc-preproc_nirs.snirf").write_bytes(b"")
    group = [GroupEntry("G1", "sub-01", "rest"), GroupEntry("G1", "sub-02", "rest")]
    with pytest.raises(StageError, match="predates optode placement"):
        write_group_channel_coords(derivs, tmp_path / "hyper", group, "preproc")


# ---- the template a run was placed on ----

_INFANT_SPACE = "ANTS6p0Months3T"


def test_a_rerun_on_another_template_leaves_one_table(tmp_path):
    run_file = tmp_path / "sub-01_task-rest_desc-preproc_nirs.snirf"
    old = tmp_path / "sub-01_task-rest_space-fsaverage_channels.tsv"
    old.write_text("name\tx\ty\tz\n")
    old.with_suffix(".json").write_text("{}")
    unplaced = Placement("none", note="2-D probe", space=_INFANT_SPACE)

    write_channel_coords(synth_raw("01", "rest"), unplaced,
                         channel_coords_path(run_file, _INFANT_SPACE))

    assert not old.exists() and not old.with_suffix(".json").exists()
    placement = read_placement(run_file)
    assert placement.space == _INFANT_SPACE and not placement.placed
    # an infant template is labelled in aparc alone, and an unplaced run nowhere
    table = read_table(channel_coords_path(run_file, _INFANT_SPACE))
    assert table.columns.tolist() == ["name", "x", "y", "z", "aparc"]
    assert table["aparc"].isna().all()
    side = json.loads(channel_coords_path(run_file, _INFANT_SPACE).with_suffix(".json").read_text())
    # BIDS has no keyword for an infant template; the description names it
    assert side["NIRSCoordinateSystem"] == "Other"
    assert "ANTS6-0Months3T" in side["NIRSCoordinateSystemDescription"]


def test_a_run_with_tables_in_two_spaces_is_refused(tmp_path):
    derivs = tmp_path / "derivatives"
    _member(derivs, "sub-01", 10.0)
    _member(derivs, "sub-01", 10.0, space=_INFANT_SPACE)
    with pytest.raises(StageError, match="more than one space"):
        read_placement(derivs / "sub-01" / "nirs" / "sub-01_task-rest_desc-preproc_nirs.snirf")


def test_a_group_on_two_templates_names_each_members_and_no_shared_space(tmp_path):
    from nirspipe.io.groups import GroupEntry
    from nirspipe.pipeline.hyper.group_quality import write_group_channel_coords

    derivs, out = tmp_path / "derivatives", tmp_path / "hyper"
    write_dataset_description(derivs)
    write_dataset_description(out, source=derivs)
    _member(derivs, "sub-01", 10.0)
    _member(derivs, "sub-02", 12.5, space=_INFANT_SPACE)
    group = [GroupEntry("G1", "sub-01", "rest"), GroupEntry("G1", "sub-02", "rest")]

    path = write_group_channel_coords(derivs, out, group, "preproc")

    assert path.name == "group-G1_task-rest_channels.tsv"
    table = read_table(path)
    assert table.groupby("subject_id")["space"].unique().map(list).to_dict() == {
        "sub-01": ["fsaverage"], "sub-02": [_INFANT_SPACE]}
    side = json.loads(path.with_suffix(".json").read_text())
    assert side["NIRSCoordinateSystem"] == "Other"
    assert {k: v["space"] for k, v in side["optode_placement"].items()} == {
        "sub-01": "fsaverage", "sub-02": _INFANT_SPACE}
