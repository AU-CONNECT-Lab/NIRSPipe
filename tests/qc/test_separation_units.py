"""The record keeps source-detector separations in mm, the unit the flags and the band stamps use."""

import pytest

from nirspipe.qc.boilerplate.vocabulary import metric_label
from nirspipe.qc.metrics.coupling import _channel_distance_metrics
from tests._synth import N_LONG_PAIRS, synth_raw


def test_a_30_mm_and_an_8_mm_pair_are_recorded_as_30_and_8():
    # the synthetic montage puts every long detector 30 mm from its source and the short one 8 mm
    metrics = _channel_distance_metrics(synth_raw("01", "rest"))
    assert metrics["ch_dist_max"] == pytest.approx(30.0)
    assert metrics["ch_dist_min"] == pytest.approx(8.0)
    short = f"S{N_LONG_PAIRS + 1}_D{N_LONG_PAIRS + 1} 760"
    assert metrics["ch_dist_per_channel"][short] == pytest.approx(8.0)


def test_the_report_labels_them_in_mm():
    assert metric_label("ch_dist_mean") == "Mean separation (mm)"
