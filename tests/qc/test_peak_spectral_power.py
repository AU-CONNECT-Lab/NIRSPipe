"""PSP on the scale its 0.1 line is defined on: a power, not a density, spectrum.

Expected values come from the definition rather than the code: a unit cosine carries power
1/2 on a one-sided power spectrum, the similarity is a cosine similarity so neither
wavelength's amplitude can move it, and two wavelengths sharing no pulse have no peak.
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from nirspipe.pipeline.prep_pipeline import intensity_to_od
from nirspipe.qc.metrics._helpers import PSP_PASS
from nirspipe.qc.metrics.coupling import _pair_peak_power, peak_spectral_power
from tests._synth import synth_raw

SFREQ = 10.0
N = 100          # one 10 s window
BAND = (0.7, 1.5)


def _cosine(f0: float) -> np.ndarray:
    return np.cos(2 * np.pi * f0 * np.arange(N) / SFREQ + 0.3)


@pytest.mark.parametrize("bin_index", [19, 20, 21])
def test_two_identical_unit_cosines_score_one_half(bin_index):
    # on a bin of the 2N - 1 point spectrum, so the Hamming window loses nothing to scalloping
    x = _cosine(bin_index * SFREQ / (2 * N - 1))
    assert _pair_peak_power(x, x, SFREQ, BAND[1]) == pytest.approx(0.5, abs=0.02)


def test_neither_wavelength_s_amplitude_moves_it():
    x = _cosine(1.0)
    assert _pair_peak_power(3.0 * x, 0.2 * x, SFREQ, BAND[1]) == pytest.approx(
        _pair_peak_power(x, x, SFREQ, BAND[1]))


def test_a_peak_above_the_band_is_not_read():
    x = _cosine(2.5)
    assert _pair_peak_power(x, x, SFREQ, BAND[1]) < 0.01


def test_a_flat_window_scores_zero():
    assert _pair_peak_power(np.zeros(N), _cosine(1.0), SFREQ, BAND[1]) == 0.0


def test_two_wavelengths_with_no_shared_pulse_stay_under_the_line():
    rng = np.random.default_rng(0)
    sos = butter(4, BAND, btype="band", fs=SFREQ, output="sos")
    scores = [_pair_peak_power(*sosfiltfilt(sos, rng.standard_normal((2, N))), SFREQ, BAND[1])
              for _ in range(200)]
    assert np.percentile(scores, 95) < PSP_PASS


def test_a_pair_s_two_channels_carry_one_score_in_channel_order():
    od = intensity_to_od(synth_raw("01", "rest"))
    scores, times = peak_spectral_power(od, *BAND, 10.0)
    assert scores.shape == (len(od.ch_names), len(times))
    np.testing.assert_array_equal(scores[0::2], scores[1::2])
