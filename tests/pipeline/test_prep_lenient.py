"""Preprocessing's lenient switch: what screening cannot judge rejects nothing, and a step that fails is skipped and named."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from nirspipe.exceptions import StageError
from nirspipe.io.auxiliary import read_aux_table
from nirspipe.io.derivatives import write_dataset_description
from nirspipe.io.snirf import read_snirf, write_snirf
from nirspipe.pipeline import prep_pipeline
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import _write_dataset_root, _write_subject, synth_raw

GAP = "it has 1 window(s) of 10 s to count and needs 2"


def _config(**overrides):
    return PrepConfig(**{"subject": "01", "dpf": [6.0], "sci_threshold": 0.8,
                         "cardiac_l_freq": 0.7, "cardiac_h_freq": 1.5, "resp_l_freq": 0.1,
                         "resp_h_freq": 0.5, "motion_correction": "tddr", **overrides})


def _run(raw, tmp_path, **overrides):
    return run_prep(raw, _config(**overrides), tmp_path, source_entities={"task": "rest"})


def _stages(out):
    return sorted(p.name.split("_desc-")[1].split("_")[0]
                  for p in out.rglob("*_desc-*_nirs.snirf"))


def _sci_sidecar(out):
    (path,) = out.rglob("*_desc-sci_nirs.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_recording_too_short_to_judge_rejects_nothing_and_says_why(tmp_path):
    result = _run(synth_raw("01", "rest", duration=15.0), tmp_path, lenient=True)
    assert result.bad_channels == []
    assert _sci_sidecar(tmp_path)["screening_gap"] == GAP
    assert _stages(tmp_path) == ["motcorrected", "od", "preproc", "sci"]


def test_the_same_recording_is_refused_without_the_switch(tmp_path):
    with pytest.raises(StageError, match="needs 2"):
        _run(synth_raw("01", "rest", duration=15.0), tmp_path)


def test_a_judged_recording_records_no_gap_and_screens_as_the_strict_path_does(tmp_path):
    strict = _run(synth_raw("01", "rest"), tmp_path / "strict")
    lenient = _run(synth_raw("01", "rest"), tmp_path / "lenient", lenient=True)
    assert lenient.bad_channels == strict.bad_channels
    assert "screening_gap" not in _sci_sidecar(tmp_path / "lenient")
    assert "screening_gap" not in _sci_sidecar(tmp_path / "strict")


def test_every_channel_failing_goes_on_with_all_marked(tmp_path):
    raw = synth_raw("01", "rest")
    result = _run(raw, tmp_path, lenient=True, sci_threshold=1.0, min_good_frac=1.0)
    assert sorted(result.bad_channels) == sorted(raw.ch_names)
    assert sorted(_sci_sidecar(tmp_path)["bad_channels"]) == sorted(raw.ch_names)
    # every step still runs, as raw QC draws the recording whatever its verdict
    assert result.skipped == []
    assert _stages(tmp_path) == ["motcorrected", "od", "preproc", "sci"]


def test_every_channel_failing_is_refused_without_the_switch(tmp_path):
    with pytest.raises(StageError, match="every channel failed"):
        _run(synth_raw("01", "rest"), tmp_path, sci_threshold=1.0, min_good_frac=1.0)


def test_every_channel_marked_by_hand_goes_on(tmp_path):
    raw = synth_raw("01", "rest")
    pairs = sorted({name.rsplit(" ", 1)[0] for name in raw.ch_names})
    result = _run(raw, tmp_path, lenient=True, bad_channels=pairs)
    assert sorted(result.bad_channels) == sorted(raw.ch_names)


def _boom(*args, **kwargs):
    raise ValueError("planted failure")


def test_a_failed_correction_is_skipped_and_named(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_pipeline, "correct_motion", _boom)
    result = _run(synth_raw("01", "rest"), tmp_path, lenient=True)
    assert [name for name, _ in result.skipped] == ["Motion correction"]
    assert _stages(tmp_path) == ["od", "preproc", "sci"]
    assert result.raw_haemo is not None


def test_a_failed_conversion_is_skipped_and_named(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_pipeline, "od_to_haemo", _boom)
    result = _run(synth_raw("01", "rest"), tmp_path, lenient=True)
    assert [name for name, _ in result.skipped] == ["Beer-Lambert"]
    assert _stages(tmp_path) == ["motcorrected", "od", "sci"]
    assert result.raw_haemo is None


@pytest.mark.parametrize("step", ["correct_motion", "od_to_haemo"])
def test_a_failed_step_stops_the_run_without_the_switch(tmp_path, monkeypatch, step):
    monkeypatch.setattr(prep_pipeline, step, _boom)
    with pytest.raises(ValueError, match="planted failure"):
        _run(synth_raw("01", "rest"), tmp_path)


def _fail_on(desc, real):
    def write(raw, path, *args, **kwargs):
        if f"_desc-{desc}_" in Path(path).name:
            raise OSError("planted write failure")
        return real(raw, path, *args, **kwargs)
    return write


def _preproc(out):
    (path,) = out.rglob("*_desc-preproc_nirs.snirf")
    return read_snirf(path).get_data()


def test_a_correction_that_could_not_be_saved_is_not_carried_into_the_conversion(tmp_path, monkeypatch):
    raw = synth_raw("01", "rest")
    _run(raw.copy(), tmp_path / "none", lenient=True, motion_correction="none")
    _run(raw.copy(), tmp_path / "tddr", lenient=True)
    monkeypatch.setattr(prep_pipeline, "write_snirf", _fail_on("motcorrected", write_snirf))
    result = _run(raw.copy(), tmp_path / "unsaved", lenient=True)
    assert [name for name, _ in result.skipped] == ["Motion correction"]
    assert not np.allclose(_preproc(tmp_path / "tddr"), _preproc(tmp_path / "none"))
    assert np.array_equal(_preproc(tmp_path / "unsaved"), _preproc(tmp_path / "none"))


def test_a_failed_positions_table_is_named_as_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(prep_pipeline, "write_channel_coords", _boom)
    result = _run(synth_raw("01", "rest"), tmp_path, lenient=True)
    assert [name for name, _ in result.skipped] == ["Channel positions"]
    assert result.raw_haemo is not None
    assert "preproc" in _stages(tmp_path)


def test_the_aux_table_is_written_when_the_conversion_fails(tmp_path, monkeypatch):
    bids = tmp_path / "bids"
    _write_dataset_root(bids, ["01"])
    source = _write_subject(bids, "01", "rest", synth_raw("01", "rest"))
    with h5py.File(source, "a") as handle:
        aux = handle["nirs"].create_group("aux1")
        aux.create_dataset("name", data=b"ACC_X")
        aux.create_dataset("dataTimeSeries", data=np.linspace(0.0, 1.0, 600))
        aux.create_dataset("time", data=np.arange(600) / 20.0)
        aux.create_dataset("dataUnit", data=b"m/s^2")
    monkeypatch.setattr(prep_pipeline, "od_to_haemo", _boom)
    out = tmp_path / "out"
    write_dataset_description(out, source=bids)
    run_prep(read_snirf(source), _config(lenient=True), out, source_entities={"task": "rest"},
             source_path=source)
    (table,) = out.rglob("*_desc-aux_timeseries.tsv.gz")
    assert "ACC_X" in read_aux_table(table).columns


def test_every_channel_failing_is_said_by_every_stage_after_screening(tmp_path):
    _run(synth_raw("01", "rest"), tmp_path, lenient=True, sci_threshold=1.0, min_good_frac=1.0)
    for desc in ("sci", "motcorrected", "preproc"):
        (path,) = tmp_path.rglob(f"*_desc-{desc}_nirs.snirf")
        sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        names = read_snirf(path).ch_names
        assert sorted(sidecar["bad_channels"]) == sorted(names), desc
        assert sidecar["data"]["n_bad"] == len(names), desc
