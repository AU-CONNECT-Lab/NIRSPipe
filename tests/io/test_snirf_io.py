"""SNIRF write/read: the guards on writing, and the stage round trip.

The lineage stamp does not survive a SNIRF round trip, so the filename carries it:
the writer puts the stage in the desc- entity and read_snirf reads the same string
back. That handoff is the one place the whole provenance chain touches disk.
"""

import logging

import h5py
import mne
import numpy as np
import pytest

from nirspipe.io.auxiliary import read_aux_snirf
from nirspipe.io.bids import write_bids_from_snirf
from nirspipe.io.snirf import read_raw_snirf, read_snirf, write_snirf
from nirspipe.io.tables import read_table
from nirspipe.utils.lineage import lineage_of, path_from, stage_of

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
    from nirspipe.io.derivatives import write_sidecar_json

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


# ---- a file whose clock does not start at 0 ----

def _on_its_own_clock(raw, path, t0, unit="s"):
    """The recording with its clock starting at t0, events a and b at 10 s and 55 s of its data, and one aux channel."""
    write_snirf(raw, path)
    k = 1000.0 if unit == "ms" else 1.0
    with h5py.File(path, "r+") as f:
        nirs = f["nirs"]
        times = (nirs["data1/time"][...] + t0) * k
        nirs["data1/time"][...] = times
        for key in [key for key in nirs if key.startswith("stim")]:
            del nirs[key]
        for i, (name, onset) in enumerate((("a", 10.0), ("b", 55.0)), start=1):
            stim = nirs.create_group(f"stim{i}")
            stim.create_dataset("name", data=name.encode())
            stim.create_dataset("data", data=np.array([[(onset + t0) * k, 2.0 * k, 1.0]]))
        aux = nirs.create_group("aux1")
        aux.create_dataset("name", data=b"ACCEL_X")
        aux.create_dataset("dataTimeSeries", data=np.zeros(len(times)))
        aux.create_dataset("time", data=times)
        tags = nirs["metaDataTags"]
        if "TimeUnit" in tags:
            del tags["TimeUnit"]
        tags.create_dataset("TimeUnit", data=unit.encode())
    return path


def _events(raw):
    a = raw.annotations
    return [(round(float(o), 6), round(float(d), 6), str(n)) for o, d, n in zip(a.onset, a.duration, a.description)]


def test_events_stay_on_the_data_when_the_clock_starts_late(raw, tmp_path):
    path = _on_its_own_clock(raw, tmp_path / "sub-01_task-tapping_nirs.snirf", t0=70.0)

    assert _events(read_snirf(path)) == [(10.0, 2.0, "a"), (55.0, 2.0, "b")]


def test_mne_alone_places_them_past_the_end_and_drops_them(raw, tmp_path):
    path = _on_its_own_clock(raw, tmp_path / "sub-01_task-tapping_nirs.snirf", t0=70.0)

    assert _events(mne.io.read_raw_snirf(path, verbose="error")) == []


def test_the_move_is_reported_with_the_clock(raw, tmp_path, caplog):
    path = _on_its_own_clock(raw, tmp_path / "sub-01_task-tapping_nirs.snirf", t0=70.0)
    with caplog.at_level(logging.WARNING):
        read_raw_snirf(path, verbose="error")
    assert "starts at 70 s" in caplog.text


def test_a_file_in_milliseconds_has_its_events_in_seconds(raw, tmp_path):
    path = _on_its_own_clock(raw, tmp_path / "sub-01_task-tapping_nirs.snirf", t0=0.0, unit="ms")

    back = read_raw_snirf(path, verbose="error")
    assert back.info["sfreq"] == pytest.approx(raw.info["sfreq"])
    assert _events(back) == [(10.0, 2.0, "a"), (55.0, 2.0, "b")]


def test_a_file_starting_at_zero_reads_as_mne_reads_it(raw, tmp_path, caplog):
    path = _on_its_own_clock(raw, tmp_path / "sub-01_task-tapping_nirs.snirf", t0=0.0)
    with caplog.at_level(logging.WARNING):
        back = read_raw_snirf(path, verbose="error")
    assert _events(back) == [(10.0, 2.0, "a"), (55.0, 2.0, "b")]
    assert _events(back) == _events(mne.io.read_raw_snirf(path, verbose="error"))
    assert "clock" not in caplog.text


def test_aux_stamps_sit_on_the_data_axis(raw, tmp_path):
    path = _on_its_own_clock(raw, tmp_path / "sub-01_task-tapping_nirs.snirf", t0=70.0)

    times, _, _ = read_aux_snirf(path)
    assert np.allclose(times["ACCEL_X"], raw.times)


def test_conversion_to_bids_writes_the_events_on_the_data(raw, tmp_path):
    path = _on_its_own_clock(raw, tmp_path / "recording.snirf", t0=70.0)
    write_bids_from_snirf(path, tmp_path / "bids", subject="01", task="tapping")

    events = read_table(next((tmp_path / "bids").rglob("*_events.tsv")))
    assert events["onset"].tolist() == [10.0, 55.0]
    assert events["trial_type"].tolist() == ["a", "b"]
