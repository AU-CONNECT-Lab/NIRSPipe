"""Every derivative _nirs.json carries the REQUIRED fields of the raw sidecar it came from."""

import json
import shutil

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.exceptions import StageError
from nirspipe.io.snirf import read_snirf
from nirspipe.pipeline.crop import crop_snirf_from_path
from nirspipe.pipeline.post_pipeline import PostConfig, run_post

from tests._synth import make_bids_dataset

BANDS = dict(cardiac_l_freq=0.7, cardiac_h_freq=1.5, resp_l_freq=0.2, resp_h_freq=0.5)
ARGS = ("--dpf", "6", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
        "--cardiac-h-freq", "1.5", "--resp-l-freq", "0.2", "--resp-h-freq", "0.5",
        "--mode", "rest", "--drift-model", "polynomial", "--drift-order", "1",
        "--high-pass", "0.01", "--low-pass", "0.1", "--resample-sfreq", "5",
        "--export-mean-filled", "--participant-label", "01", "--no-report",
        "--skip-bids-validation")
STAGES = ("od", "sci", "motcorrected", "preproc", "filtered", "resampled", "errts",
          "errtsbroad", "meanfilled")
REQUIRED = ("TaskName", "SamplingFrequency", "NIRSChannelCount", "NIRSSourceOptodeCount",
            "NIRSDetectorOptodeCount")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("bids_metadata")
    bids = make_bids_dataset(root, subjects=("01",), tasks=("tapping",))
    out = root / "out"
    run_cli.main([str(bids), str(out), "participant", *ARGS])
    (raw_json,) = (bids / "sub-01" / "nirs").glob("*_nirs.json")
    files = {p.name.split("_desc-")[1].split("_")[0]: p for p in out.rglob("*_desc-*_nirs.snirf")}
    return json.loads(raw_json.read_text(encoding="utf-8")), files, out


def _sidecar(path):
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


def _assert_describes(side, path, raw_side):
    data = read_snirf(path)
    assert side["TaskName"] == raw_side["TaskName"]
    assert side["NIRSSourceOptodeCount"] == raw_side["NIRSSourceOptodeCount"]
    assert side["NIRSDetectorOptodeCount"] == raw_side["NIRSDetectorOptodeCount"]
    assert side["SamplingFrequency"] == pytest.approx(data.info["sfreq"])
    assert side["NIRSChannelCount"] == len(data.ch_names)


# ---- the run under test ----

def test_the_raw_sidecar_has_them_and_every_stage_is_written(run):
    raw_side, files, _ = run
    assert all(k in raw_side for k in REQUIRED)
    assert sorted(files) == sorted(STAGES)


# ---- propagation ----

@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_carries_the_required_fields(run, stage):
    _, files, _ = run
    side = _sidecar(files[stage])
    assert [k for k in REQUIRED if k not in side] == []


@pytest.mark.parametrize("stage", STAGES)
def test_unchanged_fields_are_copied_and_changed_ones_describe_the_file(run, stage):
    raw_side, files, _ = run
    _assert_describes(_sidecar(files[stage]), files[stage], raw_side)


def test_resampling_changes_the_rate_and_nothing_else(run):
    raw_side, files, _ = run
    assert _sidecar(files["preproc"])["SamplingFrequency"] == raw_side["SamplingFrequency"]
    assert _sidecar(files["resampled"])["SamplingFrequency"] == 5.0


def test_a_cut_of_a_stage_carries_them(run, tmp_path):
    raw_side, files, _ = run
    (cut,) = crop_snirf_from_path(files["errts"], tmp_path / "cropped", "01", tmin=0, tmax=100,
                                  derivative=True)
    side = _sidecar(cut)
    _assert_describes(side, cut, raw_side)
    assert side["SamplingFrequency"] == 5.0


# ---- trees written before the fields ----

def _without_rate(files, stage, tmp_path):
    copy = tmp_path / files[stage].name
    shutil.copy2(files[stage], copy)
    side = _sidecar(files[stage])
    del side["SamplingFrequency"]
    copy.with_suffix(".json").write_text(json.dumps(side), encoding="utf-8")
    return copy


def test_post_refuses_an_input_written_before_the_fields(run, tmp_path):
    _, files, _ = run
    preproc = _without_rate(files, "preproc", tmp_path)
    with pytest.raises(StageError, match="records no SamplingFrequency"):
        run_post(read_snirf(preproc), PostConfig(subject="01", high_pass=0.01, low_pass=0.1,
                                                 **BANDS),
                 output_dir=tmp_path / "out", mode="denoise", source_entities={"task": "tapping"},
                 source_path=preproc)
    assert not (tmp_path / "out").exists()


def test_crop_refuses_a_stage_written_before_the_fields(run, tmp_path):
    _, files, _ = run
    errts = _without_rate(files, "errts", tmp_path)
    with pytest.raises(StageError, match="records no SamplingFrequency"):
        crop_snirf_from_path(errts, tmp_path / "cropped", "01", tmin=0, tmax=100,
                             derivative=True)
    assert not list((tmp_path / "cropped").rglob("*.snirf"))
