"""SNIRF write/read: the guards on writing, and the stage round trip.

The lineage stamp does not survive a SNIRF round trip, so the filename carries it:
the writer puts the stage in the desc- entity and read_snirf reads the same string
back. That handoff is the one place the whole provenance chain touches disk.
"""

import pytest

from fnirs_pipe.io.snirf import read_snirf, write_snirf
from fnirs_pipe.utils.lineage import lineage_of, path_from, stage_of

from tests._synth import synth_raw


@pytest.fixture(scope="module")
def raw():
    return synth_raw("01", "tapping", duration=60.0)


# ---- write guards ----

def test_write_rejects_a_raw_without_a_measurement_date(raw, tmp_path):
    raw = raw.copy()
    raw.set_meas_date(None)
    with pytest.raises(ValueError, match="meas_date"):
        write_snirf(raw, tmp_path / "x.snirf")


def test_write_rejects_a_raw_without_subject_info(raw, tmp_path):
    raw = raw.copy()
    raw.info["subject_info"] = None
    with pytest.raises(ValueError, match="subject_info"):
        write_snirf(raw, tmp_path / "x.snirf")


def test_write_creates_missing_parent_directories(raw, tmp_path):
    path = tmp_path / "sub-01" / "nirs" / "sub-01_nirs.snirf"
    write_snirf(raw, path)
    assert path.exists()


# ---- stage round trip ----

def test_desc_entity_restores_the_stage(raw, tmp_path):
    path = tmp_path / "sub-01_task-tapping_desc-preproc_nirs.snirf"
    write_snirf(raw, path)
    assert stage_of(read_snirf(path)) == "preproc"


def test_a_file_without_desc_is_the_raw_input(raw, tmp_path):
    path = tmp_path / "sub-01_task-tapping_nirs.snirf"
    write_snirf(raw, path)
    assert stage_of(read_snirf(path)) == "raw"


def test_read_records_the_file_it_came_from(raw, tmp_path):
    path = tmp_path / "sub-01_desc-od_nirs.snirf"
    write_snirf(raw, path)
    back = read_snirf(path)
    assert path_from(back) == path.as_posix()
    assert lineage_of(back).step == "load"


def test_annotations_survive_the_round_trip(raw, tmp_path):
    path = tmp_path / "sub-01_task-tapping_nirs.snirf"
    write_snirf(raw, path)
    back = read_snirf(path)
    assert len(back.annotations) == len(raw.annotations)
    assert set(back.annotations.description) == set(raw.annotations.description)


def test_channel_names_and_types_survive_the_round_trip(raw, tmp_path):
    path = tmp_path / "sub-01_nirs.snirf"
    write_snirf(raw, path)
    back = read_snirf(path)
    assert back.ch_names == raw.ch_names
    assert back.get_channel_types() == raw.get_channel_types()


# ---- bad channels ----
#
# Bads are the only thing the SNIRF round trip drops: the format has nowhere to put them,
# so without the sidecar post-processing would silently un-reject every channel SCI had
# marked. The sidecar carries them instead, which couples the two files.

def _write(raw, path, bads=None):
    """Write a SNIRF and the sidecar the pipeline writes beside it."""
    from fnirs_pipe.io.derivatives import write_sidecar_json

    write_snirf(raw, path)
    if bads is not None:
        write_sidecar_json(path, {"step": "sci_pruning", "Sources": [], "bad_channels": bads})
    return path


def test_bad_channels_come_back_from_the_sidecar(raw, tmp_path):
    marked = raw.copy()
    marked.info["bads"] = [raw.ch_names[0], raw.ch_names[3]]
    path = _write(marked, tmp_path / "sub-01_desc-preproc_nirs.snirf", bads=marked.info["bads"])

    assert read_snirf(path).info["bads"] == marked.info["bads"]


def test_the_file_alone_loses_them(raw, tmp_path):
    # documents the coupling rather than guarding it: no sidecar means no restore, and
    # that is the design, not a gap to fill in later
    marked = raw.copy()
    marked.info["bads"] = [raw.ch_names[0]]
    path = _write(marked, tmp_path / "sub-01_desc-preproc_nirs.snirf")

    assert read_snirf(path).info["bads"] == []


def test_a_sidecar_without_bad_channels_marks_nothing(raw, tmp_path):
    path = _write(raw, tmp_path / "sub-01_desc-od_nirs.snirf", bads=[])
    assert read_snirf(path).info["bads"] == []


def test_a_name_the_file_does_not_have_is_ignored(raw, tmp_path):
    # only reachable through a hand-edited sidecar, but MNE raises on an unknown name
    # and that would turn a typo into a failed run
    path = _write(raw, tmp_path / "sub-01_desc-od_nirs.snirf",
                  bads=[raw.ch_names[1], "S99_D99 760"])

    assert read_snirf(path).info["bads"] == [raw.ch_names[1]]


def test_an_unreadable_sidecar_does_not_break_the_read(raw, tmp_path):
    path = tmp_path / "sub-01_desc-od_nirs.snirf"
    write_snirf(raw, path)
    path.with_suffix(".json").write_text("{not json")

    assert read_snirf(path).info["bads"] == []
