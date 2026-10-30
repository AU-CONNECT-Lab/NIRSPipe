"""Preprocessing's lenient switch: what screening cannot judge rejects nothing, and a step that fails is skipped and named."""

import json

import pytest

from nirspipe.exceptions import StageError
from nirspipe.pipeline import prep_pipeline
from nirspipe.pipeline.prep_pipeline import PrepConfig, run_prep

from tests._synth import synth_raw

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
