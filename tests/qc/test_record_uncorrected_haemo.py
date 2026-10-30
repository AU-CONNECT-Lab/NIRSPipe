"""The pipeline's record carries the HbO-HbR correlation before the motion step, run level and per condition."""

import pytest

from nirspipe.cli import qc as qc_cli
from nirspipe.cli import run as run_cli
from nirspipe.qc.subject.group_writer import condition_rows
from nirspipe.qc.subject.record_io import read_record

from tests._fingerprint import DPF, make_fingerprint_dataset

SCREENING = ("--dpf", *(str(v) for v in DPF), "--sci-threshold", "0.75",
             "--psp-threshold", "0.12", "--min-good-frac", "0.85",
             "--cardiac-l-freq", "0.8", "--cardiac-h-freq", "1.9",
             "--resp-l-freq", "0.15", "--resp-h-freq", "0.45", "--skip-bids-validation")


def _record(out, desc):
    (path,) = out.rglob(f"*_desc-{desc}_qc.json")
    return read_record(path)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("uncorrected")
    bids, _ = make_fingerprint_dataset(tmp, blocks=True)
    for name in ("none", "tddr"):
        run_cli.main([str(bids), str(tmp / name), "participant", *SCREENING, "--no-report",
                      "--motion-correction", name])
    qc_cli.main([str(bids), str(tmp / "qc"), "participant", "--participant-label", "01",
                 *SCREENING, "--motion-correction", "tddr"])
    return {"none": _record(tmp / "none", "sqm"), "tddr": _record(tmp / "tddr", "sqm"),
            "raw": _record(tmp / "qc", "sqmraw")}


def test_without_a_correction_before_is_the_preproc_file(records):
    record = records["none"]
    assert record["rawhaemo"]["hbo_hbr_corr_mean"] == pytest.approx(
        record["preproc"]["hbo_hbr_corr_mean"], abs=1e-12)
    assert record["per_channel"]["rawhaemo"]["hbo_hbr_corr_per_channel"] == pytest.approx(
        record["per_channel"]["preproc"]["hbo_hbr_corr_per_channel"], abs=1e-12)


def test_without_a_correction_each_condition_before_is_its_preproc_value(records):
    entries = records["none"]["by_condition"]
    assert entries
    for entry in entries.values():
        for row in entry["haemo_by_set"].values():
            if row:
                assert row["hbo_hbr_corr_mean_uncorrected"] == pytest.approx(
                    row["hbo_hbr_corr_mean"], abs=1e-12)
        assert entry["per_channel"]["hbo_hbr_corr_uncorrected_per_channel"] == pytest.approx(
            entry["per_channel"]["hbo_hbr_corr_per_channel"], abs=1e-12)


@pytest.mark.parametrize("section", ["rawhaemo", "rawhaemo_long", "rawhaemo_short"])
def test_under_a_correction_before_is_what_raw_qc_measured(records, section):
    assert records["tddr"][section]["hbo_hbr_corr_mean"] == pytest.approx(
        records["raw"][section]["hbo_hbr_corr_mean"], abs=1e-9)


def test_the_correction_moved_the_pair_in_this_recording(records):
    record = records["tddr"]
    assert record["rawhaemo"]["hbo_hbr_corr_mean"] != pytest.approx(
        record["preproc"]["hbo_hbr_corr_mean"], abs=1e-6)


def test_under_a_correction_each_condition_before_is_what_raw_qc_measured(records):
    pipeline, raw = records["tddr"]["by_condition"], records["raw"]["by_condition"]
    assert set(pipeline) == set(raw)
    for label, entry in pipeline.items():
        for name, row in entry["haemo_by_set"].items():
            if row:
                assert row["hbo_hbr_corr_mean_uncorrected"] == pytest.approx(
                    raw[label]["haemo_by_set"][name]["hbo_hbr_corr_mean"], abs=1e-9)


def test_the_cohort_names_a_condition_before_as_its_run_before(records):
    row = {"bids_name": "sub-01_task-x", **records["tddr"]}
    rows = condition_rows(row)
    assert rows
    for flat in rows:
        entry = records["tddr"]["by_condition"][flat["condition"]]
        assert flat["rawhaemo_hbo_hbr_corr_mean"] == pytest.approx(
            entry["haemo_by_set"]["all"]["hbo_hbr_corr_mean_uncorrected"])
