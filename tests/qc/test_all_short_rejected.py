"""A subject whose every short channel was rejected loses its short sections, quietly, and nothing else."""

import logging

import pytest

from nirspipe.cli.run import main
from nirspipe.qc.subject.record_io import read_record
from tests._synth import N_LONG_PAIRS, make_bids_dataset

SHORT_PAIR = f"S{N_LONG_PAIRS + 1}_D{N_LONG_PAIRS + 1}"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("allshort")
    bids = make_bids_dataset(root, subjects=("01",), tasks=("mixed",))
    out = root / "out"
    caplog_records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            caplog_records.append(record)

    handler = _Keep(level=logging.WARNING)
    logging.getLogger().addHandler(handler)
    try:
        main([str(bids), str(out), "participant", "--participant-label", "01",
              "--bad-channels", SHORT_PAIR, "--dpf", "6", "--sci-threshold", "0.8",
              "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
              "--resp-l-freq", "0.1", "--resp-h-freq", "0.4",
              "--mode", "denoise", "--by-condition", "--no-report", "--skip-bids-validation"])
    finally:
        logging.getLogger().removeHandler(handler)
    record = read_record(next((out / "sub-01").rglob("*_desc-sqm_qc.json")))
    return record, caplog_records


def test_no_section_fails_on_an_empty_good_set(run):
    _record, warnings = run
    failed = [r.getMessage() for r in warnings if "failed" in r.getMessage()]
    assert not failed, failed


def test_the_short_sections_are_left_out_and_the_long_ones_kept(run):
    record, _warnings = run
    assert "preproc_long" in record and "preproc_short" not in record
    assert "raw_short" not in record


def test_the_conditions_survive_with_an_empty_short_set(run):
    record, _warnings = run
    by_condition = record.get("by_condition") or {}
    assert by_condition
    for entry in by_condition.values():
        assert entry["haemo_by_set"]["long"] and not entry["haemo_by_set"]["short"]
