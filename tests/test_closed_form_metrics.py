"""Metrics whose answer can be worked out by hand, checked against the hand-worked answer.

These four have no library between the input and the output, so a closed-form input pins
them exactly rather than approximately. That matters most for `_gcor`, whose value on
uncorrelated channels is not the 0 an intuition about "no correlation" suggests: the
statistic averages *normalised* channels and then squares the norm, so n mutually
orthogonal channels give exactly 1/n. A test written against 0 would fail on correct code,
and a test written against "small" would pass on a version that had lost the normalisation.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from fnirs_pipe.qc.quantitative_metrics import (
    _gcor,
    _mask_to_segments,
    channel_cv,
    channel_snr,
)


# ---- GCOR ----

def test_identical_channels_give_one():
    x = np.random.default_rng(0).normal(size=500)
    assert_allclose(_gcor(np.vstack([x, x, x])), 1.0, atol=1e-12)


def test_a_channel_and_its_negative_cancel_exactly():
    x = np.random.default_rng(1).normal(size=500)
    assert_allclose(_gcor(np.vstack([x, -x])), 0.0, atol=1e-12)


@pytest.mark.parametrize("n", [2, 4, 8, 16])
def test_orthogonal_channels_give_one_over_n_not_zero(n):
    # g is the mean of n unit vectors; orthogonality makes the cross terms vanish, so
    # ||g||^2 = n * (1/n)^2 = 1/n. The floor of GCOR is set by channel count, not by
    # correlation, which is why a bare "close to 0" assertion would be wrong here
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(4000, n)))
    q -= q.mean(axis=0, keepdims=True)
    assert_allclose(_gcor(q.T), 1.0 / n, rtol=1e-3)


def test_one_channel_has_no_pairwise_mean_to_take():
    assert _gcor(np.random.default_rng(3).normal(size=(1, 100))) is None


def test_gcor_is_blind_to_channel_scale():
    # each channel is unit-normalised before averaging, so rescaling one cannot move it
    rng = np.random.default_rng(4)
    data = rng.normal(size=(5, 800))
    scaled = data.copy()
    scaled[0] *= 1e4
    assert_allclose(_gcor(data), _gcor(scaled), rtol=1e-10)


# ---- SNR and CV ----

def test_snr_and_cv_recover_mu_over_sigma():
    # sample statistics, not population ones: compare against the realised mean and std of
    # the draw rather than the 5.0 / 0.5 they were drawn from, or the test measures the RNG
    rng = np.random.default_rng(5)
    data = rng.normal(loc=5.0, scale=0.5, size=(3, 20000))
    mu, sigma = data.mean(axis=1), data.std(axis=1)
    assert_allclose(channel_snr(data), mu / sigma, rtol=1e-12)
    assert_allclose(channel_cv(data), sigma / mu, rtol=1e-12)


def test_the_two_are_reciprocal():
    rng = np.random.default_rng(6)
    data = rng.normal(loc=3.0, scale=0.4, size=(4, 5000))
    assert_allclose(channel_snr(data) * channel_cv(data), 1.0, rtol=1e-12)


def test_a_flat_channel_has_no_snr_but_zero_cv():
    flat = np.full((1, 100), 3.0)
    assert np.isnan(channel_snr(flat)).all()      # sigma == 0, the division is undefined
    assert_allclose(channel_cv(flat), 0.0)        # sigma == 0, the ratio is not


def test_a_zero_mean_channel_has_no_cv():
    assert np.isnan(channel_cv(np.array([[-1.0, 1.0]]))).all()


def test_cv_is_blind_to_scale_and_snr_with_it():
    rng = np.random.default_rng(7)
    data = rng.normal(loc=2.0, scale=0.3, size=(3, 4000))
    assert_allclose(channel_cv(data * 1000), channel_cv(data), rtol=1e-12)
    assert_allclose(channel_snr(data * 1000), channel_snr(data), rtol=1e-12)


# ---- mask to segments ----

SECONDS = np.arange(5.0)


# A run of n samples is n sample periods wide, at the end of the recording as anywhere else.
# The last-sample cases used to expect zero width, which is the span no figure can draw.
@pytest.mark.parametrize("mask, expected", [
    ([0, 1, 1, 0, 1], [(1.0, 2.0), (4.0, 1.0)]),   # the docstring's own example
    ([0, 0, 0, 0, 0], []),
    ([1, 1, 1, 1, 1], [(0.0, 5.0)]),               # one run touching both boundaries
    ([1, 0, 0, 0, 0], [(0.0, 1.0)]),               # opens at sample 0, no rising edge to find
    ([0, 0, 0, 0, 1], [(4.0, 1.0)]),               # closes at the last sample, no falling edge
    ([0, 0, 1, 0, 0], [(2.0, 1.0)]),
    ([1, 0, 1, 0, 1], [(0.0, 1.0), (2.0, 1.0), (4.0, 1.0)]),
])
def test_runs_become_intervals(mask, expected):
    assert _mask_to_segments(np.array(mask, dtype=bool), SECONDS) == expected


def test_the_interval_is_read_off_the_time_axis_not_the_sample_index():
    # the closing index is exclusive: samples 1, 2, 3 flagged at 10 Hz opens at times[1] and
    # runs to times[4], a 0.3 s span. Reading it as times[3] would silently shorten every
    # segment by one sample
    times = np.arange(5) / 10.0
    (onset, duration), = _mask_to_segments(np.array([0, 1, 1, 1, 0], dtype=bool), times)
    assert (onset, duration) == pytest.approx((0.1, 0.3))


def test_a_duration_is_never_negative():
    # a trailing run has no falling edge, so its end is pinned to the last sample; the max()
    # in the implementation is what stops that from going backwards on a one-sample run
    for _, duration in _mask_to_segments(np.array([0, 0, 0, 0, 1], dtype=bool), SECONDS):
        assert duration >= 0.0
