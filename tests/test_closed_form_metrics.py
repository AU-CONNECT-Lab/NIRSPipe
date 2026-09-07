"""Metrics whose answer can be worked out by hand, checked against the hand-worked answer.

These four have no library between the input and the output, so a closed-form input pins
them exactly rather than approximately. That matters most for `_gcor`, whose value on
uncorrelated channels is not the 0 an intuition about "no correlation" suggests: the
statistic averages *normalised* channels and then squares the norm, so n mutually
orthogonal channels give exactly 1/n. A test written against 0 would fail on correct code,
and a test written against "small" would pass on a version that had lost the normalisation.

The two GVTD sections at the end are deterministic rather than closed form, and they say so
where they start: a bandpass sits between their input and their output. They are here for
the determinism, since neither needs a recording or a seed.
"""

import mne
import numpy as np
import pytest
from numpy.testing import assert_allclose

from fnirs_pipe.qc.metrics import (
    _gcor,
    _mask_to_segments,
    channel_cv,
    channel_snr,
    gvtd_censor_spans,
    gvtd_channel_picks,
    long_short_channels,
)

from ._synth import synth_raw


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


# ---- GVTD censoring: the min_epoch_s pass ----
# Not closed form, and deliberately not asserted as if it were: the motion band's 0.01 Hz
# high-pass rings for roughly 1/0.01 = 100 s, so one artifact does not produce one span and
# no hand-drawn span geometry survives contact with the filter. What is pinned instead is
# the contract of the second pass, which is the part a threshold alone does not give and the
# part that has no other test.

_SF = 10.0


def _od_with_bursts(*onsets, duration=300.0, n_ch=4, width=3.0, freq=0.3):
    """An OD recording that is flat apart from a motion-band burst at each onset.

    Flat, not quiet: exact zeros make the baseline GVTD exactly zero, so the threshold is
    set by the bursts alone and the test needs no seed. The burst is a Hann-windowed 0.3 Hz
    sinusoid, i.e. already inside GVTD_MOTION_BAND, so the bandpass passes it rather than
    turning it into a decaying oscillation the way it does an impulse.
    """
    n = int(duration * _SF)
    data = np.zeros((n_ch, n))
    k = int(width * _SF)
    burst = np.hanning(k) * np.sin(2 * np.pi * freq * np.arange(k) / _SF)
    for onset in onsets:
        i = int(onset * _SF)
        data[:, i:i + k] += burst
    info = mne.create_info([f"S{i + 1}_D{i + 1} 760" for i in range(n_ch)],
                           _SF, ["fnirs_od"] * n_ch)
    return mne.io.RawArray(data, info, verbose="error")


def _surviving(spans, duration=300.0):
    """Durations of what the spans leave behind, i.e. the epochs the caller gets to keep."""
    edges = [0.0, *(x for onset, dur in spans for x in (onset, onset + dur)), duration]
    return [b - a for a, b in zip(edges[::2], edges[1::2]) if b > a]


def _censor(raw, min_epoch_s):
    return gvtd_censor_spans(raw, n_std=10.0, min_epoch_s=min_epoch_s, channel_set="all")


def test_the_short_island_between_two_artifacts_is_censored_with_them():
    """The whole point of the second pass. Two bursts 20 s apart, `min_epoch_s` 15: the
    stretch between them is too short to analyse, so it goes as well and the two artifacts
    collapse into one span."""
    raw = _od_with_bursts(100.0, 120.0)
    loose_spans, loose = _censor(raw, 0.0)
    tight_spans, tight = _censor(raw, 15.0)

    assert min(_surviving(loose_spans)) < 15.0      # the island exists before the pass
    assert min(_surviving(tight_spans)) >= 15.0     # and no survivor is short after it
    assert tight["gvtd_censor_n_spans"] < loose["gvtd_censor_n_spans"]
    assert tight["gvtd_censor_n_epochs"] < loose["gvtd_censor_n_epochs"]


def test_the_second_pass_only_ever_censors_more():
    """It absorbs survivors, so it cannot hand any sample back. A pass that recomputed the
    mask instead of adding to it would show up here."""
    raw = _od_with_bursts(100.0, 120.0)
    _, loose = _censor(raw, 0.0)
    _, tight = _censor(raw, 15.0)
    assert tight["gvtd_censor_pct"] > loose["gvtd_censor_pct"]
    assert tight["gvtd_censor_retained_s"] < loose["gvtd_censor_retained_s"]


def test_the_head_goes_too_not_only_the_islands():
    """`min_epoch_s` is a rule about every survivor, not about the gaps between artifacts.
    A single artifact 40 s in, with `min_epoch_s` 60, loses the 40 s before it as well."""
    spans, metrics = _censor(_od_with_bursts(40.0), 60.0)
    assert spans[0][0] == 0.0
    assert min(_surviving(spans)) >= 60.0
    assert metrics["gvtd_censor_n_epochs"] == 1


def test_min_epoch_zero_asks_for_the_threshold_alone():
    """The opt-out: no survivor is absorbed, however short, so the epoch count is the span
    count give or take the two boundaries."""
    spans, metrics = _censor(_od_with_bursts(100.0, 120.0), 0.0)
    assert metrics["gvtd_censor_n_epochs"] == pytest.approx(len(spans) + 1, abs=1)
    assert min(_surviving(spans)) < 1.0             # a sub-second survivor was left alone


def test_censoring_everything_reports_nothing_left_rather_than_raising():
    """`min_epoch_s` past the recording length censors it whole. The caller has to be able
    to read that off the metrics, so it is a record of zero epochs, not an exception."""
    spans, metrics = _censor(_od_with_bursts(30.0, duration=60.0), 1000.0)
    assert spans == [(0.0, 60.0)]
    assert metrics["gvtd_censor_n_epochs"] == 0
    assert metrics["gvtd_censor_retained_s"] == 0.0
    assert metrics["gvtd_censor_pct"] == 1.0


def test_a_flat_recording_has_no_threshold_and_censors_nothing():
    """Zero derivative everywhere means `gvtd_threshold` returns None. The metrics dict
    still has to arrive with every key, or a record built from it loses the columns rather
    than recording that there was nothing to measure."""
    flat = mne.io.RawArray(
        np.zeros((4, 3000)),
        mne.create_info([f"S{i + 1}_D{i + 1} 760" for i in range(4)], _SF, ["fnirs_od"] * 4),
        verbose="error")
    spans, metrics = _censor(flat, 30.0)
    assert spans == []
    # the request is recorded, the result is None: the two are not the same thing
    assert metrics["gvtd_censor_n_std"] == 10.0
    assert metrics["gvtd_censor_min_epoch_s"] == 30.0
    assert metrics["gvtd_censor_channel_set"] == "all"
    for key in ("gvtd_censor_thresh", "gvtd_censor_pct", "gvtd_censor_n_spans",
                "gvtd_censor_n_epochs", "gvtd_censor_retained_s"):
        assert metrics[key] is None, key


# ---- GVTD channel set: the label has to say what happened ----

def test_a_montage_with_no_long_channels_falls_back_to_all_and_says_so():
    """The label is printed on the carpet and the trace, and stored in the censor section.
    A fallback that kept reporting "long" would make a run measured over every channel look
    comparable with one measured over the long ones, which H7 in the method note says it is
    not: on the hyper montage `all` is 40 channels where `long` is 22.
    """
    short_only = synth_raw("01", "rest", duration=60.0, n_long_pairs=0,
                           bad_pair=None, motion_onset=None)
    assert long_short_channels(short_only) == ([], short_only.ch_names)

    picks, label = gvtd_channel_picks(short_only, "long")
    assert picks == short_only.ch_names
    assert label == "all"


def test_a_normal_montage_picks_the_long_channels_and_keeps_the_label():
    normal = synth_raw("01", "rest", duration=60.0, bad_pair=None, motion_onset=None)
    long_names, short_names = long_short_channels(normal)
    assert long_names and short_names           # the fallback above is not what runs here
    assert gvtd_channel_picks(normal, "long") == (long_names, "long")
    assert gvtd_channel_picks(normal, "all") == (normal.ch_names, "all")


def test_an_unknown_channel_set_is_refused_rather_than_guessed():
    normal = synth_raw("01", "rest", duration=60.0, bad_pair=None, motion_onset=None)
    with pytest.raises(ValueError, match="channel_set"):
        gvtd_channel_picks(normal, "short")
