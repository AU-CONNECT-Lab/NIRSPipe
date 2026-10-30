"""A flat channel leaves a window's SCI and PSP means to the other channels, and the retained share is over the recording's length."""

import mne
import numpy as np
import pytest

from nirspipe.pipeline.prep_pipeline import intensity_to_od
from nirspipe.qc.metrics.haemo import _retention_metrics
from nirspipe.qc.metrics.windowed import attach_windowed_series

from tests._synth import synth_raw


def _flat_between(x):
    x = x.copy()
    x[200:400] = x[200]
    return x


@pytest.mark.parametrize("stem", ["sci", "psp"])
def test_a_window_one_flat_pair_blanks_keeps_the_other_channels_mean(stem):
    raw = synth_raw("01", "rest", duration=120.0, motion_onset=None)
    # 20 to 40 s on the 10 Hz synthetic clock
    raw.apply_function(_flat_between, picks=["S1_D1 760", "S1_D1 850"])
    windowed: dict = {}
    series = attach_windowed_series(windowed, intensity_to_od(raw), 0.7, 1.5, 10.0,
                                    raw_intensity=raw)
    matrix = np.asarray(series[f"{stem}_matrix"], dtype=float)
    blanked = np.isnan(matrix).any(axis=0)
    assert blanked.any() and not np.isnan(matrix).all(axis=0).any()
    means = np.asarray(windowed[f"{stem}_per_window"], dtype=float)
    assert not np.isnan(means).any()
    assert means[blanked] == pytest.approx(
        [np.mean(col[~np.isnan(col)]) for col in matrix[:, blanked].T])


def test_the_retained_share_is_over_the_recording_s_length():
    raw = synth_raw("01", "rest", duration=600.0, motion_onset=None)
    raw.set_annotations(mne.Annotations([100.0], [60.0], ["BAD_manual"]))
    # 60 s of 600: the share the record's excluded section gives the same run
    assert _retention_metrics(raw)["pct_data_retained"] == pytest.approx(0.9, abs=1e-12)
