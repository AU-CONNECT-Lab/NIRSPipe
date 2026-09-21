"""Edge/mid RMS: the filter transient the reports never mentioned.

A high-pass with a low cutoff needs a long impulse response, so its output starts and ends
with the filter settling. On real recordings that runs well above the middle of the record,
and a detrend does not touch it, so the report says the number rather than leaving every
figure drawn on a filtered stage to include it silently.
"""

import numpy as np
import pytest

from fnirs_pipe.qc.metrics import EDGE_S, edge_to_mid_rms

from tests._synth import synth_raw


@pytest.fixture(scope="module")
def raw():
    return synth_raw("01", "rest", duration=400.0, n_long_pairs=4,
                     bad_pair=None, motion_onset=None)


def test_a_flat_record_reads_about_one(raw):
    """Nothing louder at the ends means a ratio at 1, which is what makes a larger value
    readable as the transient rather than as an arbitrary scale."""
    flat = raw.copy()
    rng = np.random.default_rng(0)
    flat._data[:] = rng.normal(0, 1e-6, flat._data.shape)
    assert edge_to_mid_rms(flat) == pytest.approx(1.0, abs=0.1)


def test_a_loud_pair_of_ends_is_detected(raw):
    loud = raw.copy()
    n = int(EDGE_S * loud.info["sfreq"])
    loud._data[:] = 1e-6
    loud._data[:, :n] *= 4.0
    loud._data[:, -n:] *= 4.0
    # both ends at 4x, so the RMS over the two of them together is 4x the middle
    assert edge_to_mid_rms(loud) == pytest.approx(4.0, rel=0.05)


def test_a_real_high_pass_leaves_the_ends_louder(raw):
    """The measurement that opened this: a bandpass alone produces the elevation, with no
    artefact put in by hand."""
    import mne

    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    haemo = mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)
    filtered = haemo.copy().filter(0.01, 0.2, method="iir",
                                   iir_params=dict(order=4, ftype="butter"), verbose="error")
    assert edge_to_mid_rms(filtered) > 1.0


def test_a_record_with_no_middle_declines_to_answer(raw):
    """Three edge spans is the floor: two ends and something between them. Returning a
    number off a shorter record would be comparing an end against an end."""
    short = raw.copy().crop(tmax=2.0 * EDGE_S - 1.0 / raw.info["sfreq"])
    assert edge_to_mid_rms(short) is None
    assert edge_to_mid_rms(raw, edge_s=1.0) is not None
