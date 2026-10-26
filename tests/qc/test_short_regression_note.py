"""A run whose every short channel was rejected: its rows stay short, and the report says why no regressor ran."""

import pytest

from nirspipe.cli.run import main
from nirspipe.qc.common.channel_table import channel_rows
from tests._synth import N_LONG_PAIRS, make_bids_dataset

SHORT_PAIR = f"S{N_LONG_PAIRS + 1}_D{N_LONG_PAIRS + 1}"
NOTE = "Short-channel regression did not run: all"
NEITHER = "neither range covers"


@pytest.fixture(scope="module")
def bids(tmp_path_factory):
    return make_bids_dataset(tmp_path_factory.mktemp("short_note"), subjects=("01",),
                             tasks=("rest",))


def _report(bids, out, *flags):
    main([str(bids), str(out), "participant", "--participant-label", "01",
          "--bad-channels", SHORT_PAIR, "--dpf", "6", "--sci-threshold", "0.8",
          "--cardiac-l-freq", "0.7", "--cardiac-h-freq", "1.5",
          "--resp-l-freq", "0.1", "--resp-h-freq", "0.4",
          "--mode", "denoise", *flags, "--skip-bids-validation"])
    return (out / "sub-01" / "sub-01_task-rest_report.html").read_text(encoding="utf-8")


def test_the_default_regression_with_every_short_channel_rejected_is_noted(bids, tmp_path):
    page = _report(bids, tmp_path)
    assert NOTE in page
    # an 8 mm channel is short, rejected or not
    assert NEITHER not in page


def test_a_run_told_none_gets_no_note(bids, tmp_path):
    assert NOTE not in _report(bids, tmp_path, "--short-channel", "none")


def test_a_rejected_short_set_keeps_its_rows_short_with_the_whole_montage_scores():
    long_ch, short_ch = "S1_D1 760", "S5_D5 760"
    record = {
        "raw": {"n_long_channels": 1, "n_short_channels": 1, "sep_short_max_mm": 10.0,
                "sep_long_min_mm": 15.0, "sep_long_max_mm": None},
        "per_channel": {
            "raw": {"ch_dist_per_channel": {long_ch: 30.0, short_ch: 8.0},
                    "sci_per_channel": {long_ch: 0.9, short_ch: 0.4}},
            # no raw_short: every short channel was rejected
            "raw_long": {"sci_per_channel": {long_ch: 0.9}},
        },
    }
    rows = {r["name"]: r for r in channel_rows(record, {long_ch: 0.9, short_ch: 0.4},
                                               [short_ch])}
    assert (rows[short_ch]["separation"], rows[short_ch]["sci_whole"]) == ("short", 0.4)
    assert rows[short_ch]["is_bad"] and rows[long_ch]["separation"] == "long"


def _unsplit_record(dist: dict) -> dict:
    return {
        "raw": {"n_long_channels": len(dist), "n_short_channels": 0, "sep_short_max_mm": 10.0,
                "sep_long_min_mm": 15.0, "sep_long_max_mm": 45.0},
        "per_channel": {"raw": {"ch_dist_per_channel": dist,
                                "sci_per_channel": {ch: 0.9 for ch in dist}}},
    }


def test_a_montage_without_short_channels_calls_its_channels_long_by_their_separation():
    # no raw_long or raw_short: every channel is long, so the record was never split
    dist = {"S1_D1 760": 30.0, "S2_D2 760": 40.0}
    rows = channel_rows(_unsplit_record(dist), {ch: 0.9 for ch in dist}, [])
    assert [r["separation"] for r in rows] == ["long", "long"]
    assert rows[0]["sci_whole"] == 0.9


def test_a_montage_without_positions_has_no_separation():
    dist = {"S1_D1 760": 0.0, "S2_D2 760": 0.0}
    rows = channel_rows(_unsplit_record(dist), {ch: 0.9 for ch in dist}, [])
    assert {r["separation"] for r in rows} == {""}
