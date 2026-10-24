"""A recording channel screening cannot judge: the raw reports show it and say why, prep refuses it."""

import pandas as pd
import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.io.groups import GroupEntry
from nirspipe.pipeline.hyper.group_quality import compute_group_sqm_raw
from nirspipe.pipeline.prep_pipeline import intensity_to_od
from nirspipe.qc.metrics import screening_gap

from tests._synth import make_bids_dataset, synth_raw

RAW_ARGS = ("--dpf", "6", "--sci-threshold", "0.8", "--cardiac-l-freq", "0.7",
            "--cardiac-h-freq", "1.5", "--skip-bids-validation")
NOTE = "Channel screening could not judge this recording"


# ---- the reason ----

def test_a_recording_under_two_windows_has_a_gap():
    od = intensity_to_od(synth_raw("01", "rest", duration=15.0))
    assert screening_gap(od, {"good_frac": {"S1_D1 760": 1.0}}) == (
        "it has 1 window(s) of 10 s to count and needs 2")


def test_an_unmeasured_criterion_is_a_gap():
    od = intensity_to_od(synth_raw("01", "rest"))
    assert screening_gap(od, {"good_frac": {}}).startswith("passing windows could not be measured")


def test_no_optical_density_is_a_gap():
    assert screening_gap(None, {}) == "its optical density could not be computed"


def test_a_recording_screening_can_judge_has_none():
    od = intensity_to_od(synth_raw("01", "rest"))
    assert screening_gap(od, {"good_frac": {"S1_D1 760": 1.0}}) is None


# ---- the subject raw report ----

@pytest.fixture(scope="module")
def short_raw_qc(tmp_path_factory):
    root = tmp_path_factory.mktemp("unscreened")
    bids = make_bids_dataset(root, subjects=("01",), tasks=("rest",), duration=15.0)
    out = root / "out"
    qc_cli.main(["prep-raw", str(bids), str(out), "--participant-label", "01", *RAW_ARGS])
    return out


def test_the_raw_report_renders_and_says_why(short_raw_qc):
    (page,) = short_raw_qc.rglob("*_desc-raw_report.html")
    html = page.read_text(encoding="utf-8")
    assert NOTE in html and "it has 1 window(s) of 10 s to count and needs 2" in html


def test_the_raw_report_rejects_nothing_on_screening(short_raw_qc):
    (path,) = short_raw_qc.rglob("*_desc-rawchannel_qc.tsv")
    assert not pd.read_csv(path, sep="\t")["is_bad"].any()


# ---- the dyad raw pass ----

def test_a_member_screening_cannot_judge_carries_the_reason_and_rejects_nothing(tmp_path):
    raws = {"sub-11": synth_raw("11", "hold", duration=15.0), "sub-12": synth_raw("12", "hold")}
    sqm = compute_group_sqm_raw([GroupEntry("G01", sid, "hold") for sid in raws], raws, 0.8,
                                tmp_path, 0.7, 1.5)
    assert sqm["sub-11"]["screening_gap"] == "it has 1 window(s) of 10 s to count and needs 2"
    assert sqm["sub-11"]["bad_channels"] == []
    assert sqm["sub-12"]["screening_gap"] is None
