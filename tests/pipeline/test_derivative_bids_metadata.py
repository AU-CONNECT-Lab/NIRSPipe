"""Every derivative _nirs.json carries the REQUIRED fields of the raw sidecar it came from."""

import json

import pytest

from nirspipe.cli import run as run_cli
from nirspipe.io.snirf import read_snirf

from tests._synth import make_bids_dataset

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
    return json.loads(raw_json.read_text(encoding="utf-8")), files


def _sidecar(path):
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))


# ---- the run under test ----

def test_the_raw_sidecar_has_them_and_every_stage_is_written(run):
    raw_side, files = run
    assert all(k in raw_side for k in REQUIRED)
    assert sorted(files) == sorted(STAGES)


# ---- propagation ----

@pytest.mark.xfail(strict=True, reason="derivative sidecars carry none of the raw REQUIRED fields")
@pytest.mark.parametrize("stage", STAGES)
def test_each_stage_carries_the_required_fields(run, stage):
    _, files = run
    side = _sidecar(files[stage])
    assert [k for k in REQUIRED if k not in side] == []


@pytest.mark.xfail(strict=True, reason="derivative sidecars carry none of the raw REQUIRED fields")
@pytest.mark.parametrize("stage", STAGES)
def test_unchanged_fields_are_copied_and_changed_ones_describe_the_file(run, stage):
    raw_side, files = run
    side = _sidecar(files[stage])
    data = read_snirf(files[stage])
    assert side.get("TaskName") == raw_side["TaskName"]
    assert side.get("NIRSSourceOptodeCount") == raw_side["NIRSSourceOptodeCount"]
    assert side.get("NIRSDetectorOptodeCount") == raw_side["NIRSDetectorOptodeCount"]
    assert side.get("SamplingFrequency") == pytest.approx(data.info["sfreq"])
    assert side.get("NIRSChannelCount") == len(data.ch_names)
