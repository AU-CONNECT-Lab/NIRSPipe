"""SCI, PSP and CV are drawn and sliced on one window grid, at any sampling rate."""

import numpy as np
import pytest
from mne_nirs.preprocessing import peak_power, scalp_coupling_index_windowed

from fnirs_pipe.pipeline.prep_pipeline import intensity_to_od
from fnirs_pipe.qc.metrics.windowed import compute_windowed_cv
from tests._synth import synth_raw


@pytest.mark.parametrize("sfreq", [10.0, 7.8125])
def test_cv_windows_are_the_sci_and_psp_windows(sfreq):
    # at 7.8125 Hz a 10 s window is 78.125 samples: rounding it gave CV one sample less per
    # window than mne-nirs, and the two grids drifted apart over the run
    raw = synth_raw("01", "rest")
    if raw.info["sfreq"] != sfreq:
        raw = raw.resample(sfreq, verbose=False)
    od = intensity_to_od(raw.copy())

    _, _, sci_times = scalp_coupling_index_windowed(od, time_window=10.0, l_freq=0.7,
                                                    h_freq=1.5, verbose=False)
    _, _, psp_times = peak_power(od.copy(), time_window=10.0, l_freq=0.7, h_freq=1.5,
                                 verbose=False)
    _, cv_times = compute_windowed_cv(raw, 10.0)

    # starts and count, not ends: mne-nirs clips the last window's end to the final sample
    cv_starts = np.asarray(cv_times)[:, 0]
    np.testing.assert_allclose(cv_starts, np.asarray(sci_times)[:, 0])
    np.testing.assert_allclose(cv_starts, np.asarray(psp_times)[:, 0])


def test_a_flat_channel_is_counted_and_kept_out_of_both_means():
    import mne

    from fnirs_pipe.qc.metrics.coupling import _intensity_metrics

    rng = np.random.default_rng(0)
    data = np.vstack([1.0 + 0.02 * rng.standard_normal(400),   # a real channel, CV ~ 2%
                      np.full(400, 1.0)])                      # flat at a non-zero level
    raw = mne.io.RawArray(data, mne.create_info(["S1_D1 760", "S1_D1 850"], 10.0,
                                                "fnirs_cw_amplitude"), verbose="error")
    out = _intensity_metrics(raw)
    assert out["n_flat_channels"] == 1
    assert out["cv_mean"] == pytest.approx(out["cv_per_channel"]["S1_D1 760"])
    assert "S1_D1 850" not in out["cv_per_channel"]
