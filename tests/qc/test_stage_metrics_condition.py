"""A condition page band-limits before it cuts; what left the recording stays as stored."""

import numpy as np

from nirspipe.pipeline.denoise import band_limited
from nirspipe.pipeline.prep_pipeline import intensity_to_od, od_to_haemo
from nirspipe.qc.metrics import comparable_stage_metrics
from tests._synth import synth_raw

BANDS = (0.7, 1.5, 0.2, 0.5)


def test_the_removed_rows_read_the_stored_stages_when_the_caller_band_limited_first():
    haemo = od_to_haemo(intensity_to_od(synth_raw("01", "rest")), [6.0])
    # a pulse well inside the cardiac band and well outside the passband
    haemo._data += 1e-6 * np.sin(2 * np.pi * 1.0 * haemo.times)
    filtered = band_limited(haemo, 0.01, 0.2)

    split = comparable_stage_metrics([("a", haemo), ("b", haemo)], 0.01, 0.2, *BANDS,
                                     limited=[filtered, filtered])
    prefiltered = comparable_stage_metrics([("a", filtered), ("b", filtered)], None, None,
                                           *BANDS)

    # the cardiac band sits outside the passband: as stored it is there, filtered it is gone
    stored = split["removed"]["cardiac_band_power_hbo"][0]
    assert stored > 10 * prefiltered["removed"]["cardiac_band_power_hbo"][0]
    assert "drift_band_power_hbo" in split["removed"]
    # and the quality rows are the band-limited ones either way
    assert split["quality"]["gcor_hbo"] == prefiltered["quality"]["gcor_hbo"]


def test_the_band_is_applied_with_the_pipeline_design_not_mne_default():
    # MNE's default FIR leaves a 0.2 Hz low-pass a transition band of 2 Hz, so a 1 Hz pulse
    # would pass almost untouched; the pipeline's Butterworth takes it out
    haemo = od_to_haemo(intensity_to_od(synth_raw("01", "rest")), [6.0])
    haemo._data += 1e-6 * np.sin(2 * np.pi * 1.0 * haemo.times)
    ours = band_limited(haemo, 0.01, 0.2)
    mne_default = haemo.copy().filter(0.01, 0.2, verbose=False)
    one_hz = lambda r: r.compute_psd(fmin=0.9, fmax=1.1, verbose=False).get_data().mean()
    assert one_hz(ours) < one_hz(mne_default) / 100
