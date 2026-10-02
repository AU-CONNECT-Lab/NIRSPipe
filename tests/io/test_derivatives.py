"""The `data` field a sidecar records about the signal it sits beside.

This is the producer side of the shape shown on every provenance node. The reader is
covered in test_provenance_graph.py; if the two disagree the diagram loses the field
without anything failing, so both ends are pinned.
"""

import json

from fnirs_pipe.io.derivatives import data_state, write_sidecar_json


def test_data_state_records_what_the_step_left_behind(make_raw):
    raw = make_raw(n_ch=8, dur=60.0, sfreq=10.0)
    raw.info["bads"] = [raw.ch_names[0], raw.ch_names[1]]

    assert data_state(raw) == {"n_channels": 8, "n_bad": 2, "sfreq": 10.0, "duration_s": 60.0}


def test_no_bad_channels_is_zero_not_absent(make_raw):
    # the reader subtracts n_bad unconditionally, so the key has to be there
    assert data_state(make_raw(n_ch=4))["n_bad"] == 0


def test_duration_follows_the_sampling_rate(make_raw):
    # a resample changes both, and reporting the old rate against the new sample count
    # is the mistake this catches
    resampled = make_raw(n_ch=4, dur=60.0).resample(2.0, verbose="error")
    state = data_state(resampled)

    assert state["sfreq"] == 2.0
    assert state["duration_s"] == 60.0


def test_the_sidecar_timestamps_itself(tmp_path):
    out = tmp_path / "sub-01_desc-od_nirs.snirf"
    write_sidecar_json(out, {"step": "od_conversion", "Sources": []})

    assert "timestamp" in json.loads(out.with_suffix(".json").read_text())


# ---- reading what is beside an output ----

def test_read_json_gives_an_empty_object_for_anything_it_cannot_use(tmp_path):
    from fnirs_pipe.io.derivatives import read_json

    (tmp_path / "ok.json").write_text('{"parameters": {"a": 1}}', encoding="utf-8")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert read_json(tmp_path / "ok.json") == {"parameters": {"a": 1}}
    for name in ("bad.json", "list.json", "missing.json"):
        assert read_json(tmp_path / name) == {}


def test_subject_labels_are_the_bare_sorted_sub_folders(tmp_path):
    from fnirs_pipe.io.derivatives import subject_labels

    for name in ("sub-02", "sub-01", "logs"):
        (tmp_path / name).mkdir()
    (tmp_path / "sub-03.txt").write_text("")
    assert subject_labels(tmp_path) == ["01", "02"]
