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
            "raw": {"ch_dist_per_channel": {long_ch: 0.03, short_ch: 0.008},
                    "sci_per_channel": {long_ch: 0.9, short_ch: 0.4}},
            # no raw_short: every short channel was rejected
            "raw_long": {"sci_per_channel": {long_ch: 0.9}},
        },
    }
    rows = {r["name"]: r for r in channel_rows(record, {long_ch: 0.9, short_ch: 0.4},
                                               [short_ch])}
    assert (rows[short_ch]["separation"], rows[short_ch]["sci_whole"]) == ("short", 0.4)
    assert rows[short_ch]["is_bad"] and rows[long_ch]["separation"] == "long"
