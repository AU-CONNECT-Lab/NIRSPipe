"""What `_pairwise_wtc` does to pycwt's return values, not what pycwt computes.

The transform and the coherence ratio are `pycwt.wct` and are taken as correct; the mother
wavelet handed to it is ours, because its smoothing operator is (see `_morlet`), and the
fixture below passes the same one so these tests stay about the adaptation. That adaptation
is trim, sort to ascending frequency, band-limit, decimate time. Four things can go wrong
there and none of them are pycwt's, so the structural tests below rebuild each step by hand
from the raw pycwt output and require equality.

A structural test alone cannot catch a convention we and pycwt express identically but label
wrongly, so the semantic test injects a burst and requires it to show up in the right row and
the right columns. It is built as a contrast across *time* within one frequency row, not as a
peak across frequency, because a peak-across-frequency assertion cannot work here: pycwt's own
time-averaged coherence peaks at the lowest frequency in the band whatever is injected, since
the smoothing window at the largest scales spans the record and drives coherence toward 1.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline.hyper.synchrony import _pairwise_wtc

SFREQ = 5.0
DT = 1.0 / SFREQ
STEP = int(round(SFREQ))          # what _wtc_over_pairs passes
FMIN, FMAX = 0.02, 0.5
F0 = 0.1


def _pair(n, seed=0, f0=None, mask=None):
    t = np.arange(n) * DT
    rng = np.random.default_rng(seed)
    shared = np.sin(2 * np.pi * (f0 or F0) * t)
    if mask is not None:
        shared = shared * mask
    return shared + 0.5 * rng.standard_normal(n), shared + 0.5 * rng.standard_normal(n)


def _ours(sig1, sig2):
    return _pairwise_wtc(sig1, sig2, DT, STEP, FMIN, FMAX, significance=False, cache=False)


@pytest.fixture(scope="module")
def short_pair():
    return _pair(400)


@pytest.fixture(scope="module")
def raw_pycwt(short_pair):
    import pycwt

    from fnirs_pipe.pipeline.hyper.synchrony import _morlet

    sig1, sig2 = short_pair
    WCT, aWCT, coi, freqs, _ = pycwt.wct(sig1, sig2, dt=DT, dj=1.0 / 12, sig=False,
                                         normalize=True, cache=False, wavelet=_morlet())
    return WCT, coi, freqs, aWCT


# ---- structural: rebuild each step by hand ----

def test_the_whole_adaptation_reproduces_a_hand_built_one(short_pair, raw_pycwt):
    WCT, coi, freqs, _ = raw_pycwt
    order = np.argsort(freqs)
    freqs_s = freqs[order]
    band = (freqs_s >= FMIN) & (freqs_s <= FMAX)

    W, f, c, sig, _ = _ours(*short_pair)
    assert sig is None
    assert f.tolist() == pytest.approx(freqs_s[band].tolist())
    assert W == pytest.approx(WCT[order][band][:, ::STEP].astype(np.float32))
    assert c == pytest.approx(coi[::STEP].astype(np.float32))


def test_pycwt_returns_frequency_in_descending_order(raw_pycwt):
    # the sort exists only because of this; if pycwt ever returns ascending, the sort is a
    # no-op rather than a bug, but the assertion below stops the reason being forgotten
    _, _, freqs, _ = raw_pycwt
    assert np.all(np.diff(freqs) < 0)


def test_every_row_still_belongs_to_its_own_frequency(short_pair, raw_pycwt):
    WCT, _, freqs, _ = raw_pycwt
    W, f, _, _, _ = _ours(*short_pair)
    for i, fi in enumerate(f):
        j = int(np.argmin(np.abs(freqs - fi)))
        assert W[i] == pytest.approx(WCT[j][::STEP].astype(np.float32)), f"row {i}, {fi} Hz"


def test_the_phase_is_carried_through_the_same_reordering_as_the_coherence(short_pair, raw_pycwt):
    """pycwt's second return is the relative phase, and we used to drop it.

    It is the one output that has to travel through the identical sort, band mask and
    decimation as the coherence: a mismatch anywhere in that chain rotates every arrow on
    the figure while leaving the map underneath it correct, so nothing looks wrong.
    """
    _, _, freqs, aWCT = raw_pycwt
    order = np.argsort(freqs)
    band = (freqs[order] >= FMIN) & (freqs[order] <= FMAX)

    W, f, _, _, phase = _ours(*short_pair)
    assert phase.shape == W.shape
    assert phase == pytest.approx(aWCT[order][band][:, ::STEP].astype(np.float32))


def test_the_phase_is_an_angle_rather_than_a_coherence(short_pair):
    """Read as a coherence it would look like noise in [0, 1]; it is radians on (-pi, pi]."""
    _, _, _, _, phase = _ours(*short_pair)
    assert phase.min() >= -np.pi - 1e-6
    assert phase.max() <= np.pi + 1e-6
    assert phase.min() < 0, "an all-positive range would mean it was never a signed angle"


def test_the_returned_frequencies_are_ascending_and_inside_the_band(short_pair):
    _, f, _, _, _ = _ours(*short_pair)
    assert np.all(np.diff(f) > 0)
    assert f.min() >= FMIN and f.max() <= FMAX


def test_the_three_decimated_axes_have_the_same_length(short_pair):
    # the figures pair WCT columns with ref_raw.times[::step] and with coi; a mismatch there
    # shifts the whole heatmap in time without failing anywhere
    n = len(short_pair[0])
    W, _, c, _, _ = _ours(*short_pair)
    assert W.shape[1] == len(np.arange(n)[::STEP]) == len(c)


def test_pycwt_does_not_pad_the_time_axis(short_pair, raw_pycwt):
    """The trim in `_pairwise_wtc` is currently a no-op, and that is worth pinning.

    The docstring calls it trimming pycwt's zero-padding, but this pycwt returns exactly
    len(sig) columns. If a future version starts padding, this test fails and the trim stops
    being dead code; if the trim is deleted as unused, nothing else notices.
    """
    WCT, coi, _, _ = raw_pycwt
    assert WCT.shape[1] == len(short_pair[0])
    assert len(coi) == len(short_pair[0])


# ---- semantic: does the axis mean what it says ----

def test_a_burst_lands_in_the_right_row_and_the_right_columns():
    n = 6000
    mask = np.zeros(n)
    mask[n // 3:2 * n // 3] = 1.0
    W, f, _, _, _ = _ours(*_pair(n, seed=11, mask=mask))

    row = int(np.argmin(np.abs(f - F0)))
    lo, hi = W.shape[1] // 3, 2 * W.shape[1] // 3
    inside = W[row][lo:hi].mean()
    outside = np.r_[W[row][:lo], W[row][hi:]].mean()
    assert inside > 0.8
    assert inside > 3 * outside
    assert lo <= int(np.argmax(W[row])) < hi


def test_a_burst_does_not_light_up_an_unrelated_frequency():
    # the contrast above must be specific to the injected frequency, or it would also pass
    # with the rows shifted by a constant offset
    n = 6000
    mask = np.zeros(n)
    mask[n // 3:2 * n // 3] = 1.0
    W, f, _, _, _ = _ours(*_pair(n, seed=11, mask=mask))

    row = int(np.argmin(np.abs(f - 0.35)))
    lo, hi = W.shape[1] // 3, 2 * W.shape[1] // 3
    inside = W[row][lo:hi].mean()
    outside = np.r_[W[row][:lo], W[row][hi:]].mean()
    assert inside == pytest.approx(outside, abs=0.1)


# ---- the smoothing operator is ours, not the stock one ----

@pytest.mark.parametrize("dj", [1.0 / 12, 1.0 / 14, 1.0 / 8, 1.0 / 20])
def test_the_scale_window_spans_dj0_whatever_the_grid_is(dj):
    """The width is fixed in log2(scale), so the point count follows dj rather than the
    other way round."""
    from fnirs_pipe.pipeline.hyper.synchrony import _SCALE_SMOOTH_DJ0, _scale_window

    win = _scale_window(dj)
    # normalised back to unit end-weights, the span is what the definition fixes
    assert win.sum() == pytest.approx(1.0)
    assert (win / win.max()).sum() == pytest.approx(_SCALE_SMOOTH_DJ0 / dj)
    assert len(win) % 2 == 1


def test_the_stock_mother_would_give_a_different_coherence(short_pair):
    """A silent return to pycwt's own Morlet would move every coherence in the package, so
    the difference is asserted rather than assumed. Wider smoothing reads lower."""
    import pycwt

    from fnirs_pipe.pipeline.hyper.synchrony import _morlet

    sig1, sig2 = short_pair
    kw = dict(dt=DT, dj=1.0 / 12, sig=False, normalize=True, cache=False)
    ours, *_ = pycwt.wct(sig1, sig2, wavelet=_morlet(), **kw)
    stock, *_ = pycwt.wct(sig1, sig2, **kw)

    assert np.nanmean(stock) < np.nanmean(ours)
    assert np.nanmax(np.abs(ours - stock)) > 0.01
