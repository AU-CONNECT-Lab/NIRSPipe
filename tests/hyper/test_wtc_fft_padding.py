"""Choosing the transform length must not move a coherence, only the time it takes.

pycwt rounds every transform up to the next power of two, which on an hour-long recording is
40% zeros. Padding is not decoration: without it the circular convolution wraps the end of the
record onto its start. What it has to clear is how far the widest wavelet reaches in from an
edge, and a power of two is not that number.

The failure this guards against is silent. Too little padding does not raise; it returns a
coherence that is wrong at the edges and, at the shortest paddings, wrong well inside the cone
of influence too. So the tests below assert `array_equal` against the power-of-two transform
rather than a tolerance, and one of them pins the minimum itself, since a rule that stopped
clearing the cone would still return plausible numbers.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline.hyper.wtc import (
    _FLAMBDA,
    _cwt,
    _pair_from_prepared,
    _prepare_channel,
    _trim_pair,
    _wavelet_grid,
    cone_margin_s,
)

pycwt = pytest.importorskip("pycwt")

SFREQ = 10.1725


def _signal(n, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SFREQ
    return np.sin(2 * np.pi * 0.08 * t) + 0.5 * np.sin(2 * np.pi * 0.12 * t) + rng.normal(0, 1, n)


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _at_length(grid, n_fft):
    """The same grid forced onto another transform length, smoothing included.

    The mother carries the length too, so a grid rebuilt without a fresh one would transform
    at the length asked for and smooth at the length it picked for itself.
    """
    from dataclasses import replace
    from fnirs_pipe.pipeline.hyper.wtc import _morlet
    mother = _morlet()
    mother.n_fft = n_fft
    return replace(grid, n_fft=n_fft, mother=mother)


# ---- the length itself ----

def test_the_length_is_never_worse_than_the_power_of_two():
    """A recording shorter than its own padding must not be padded past the power of two."""
    for n, fmin in ((3052, 0.01), (7800, 0.01), (5000, 0.05), (39611, 0.01), (120000, 0.01)):
        dt = 1 / SFREQ
        grid = _wavelet_grid(dt, n, fmin, 0.20, limit_scales=True)
        assert grid.n_fft <= _pow2(n), f"n={n} fmin={fmin} padded past the power of two"
        assert grid.n_fft >= n, f"n={n} fmin={fmin} truncates the record"


def test_the_length_clears_the_widest_wavelet():
    """The rule's whole justification. Checked against `cone_margin_s` at the largest scale."""
    dt = 1 / SFREQ
    for n, fmin in ((39611, 0.01), (20000, 0.01), (5000, 0.05)):
        grid = _wavelet_grid(dt, n, fmin, 0.20, limit_scales=True)
        if grid.n_fft == _pow2(n):
            continue  # the minimum took over, and a power of two clears it by construction
        reach = cone_margin_s(1.0 / (_FLAMBDA * grid.sj[-1])) / dt
        assert grid.n_fft - n >= reach, f"n={n} fmin={fmin} pads {grid.n_fft-n} < {reach:.0f}"


def test_the_saving_is_real_on_an_hour_long_recording():
    """39611 samples sit just above a power of two."""
    grid = _wavelet_grid(1 / SFREQ, 39611, 0.01, 0.20, limit_scales=True)
    assert grid.n_fft < 0.80 * 65536


# ---- the transform ----

def test_our_cwt_reproduces_pycwt_exactly_at_the_same_length():
    """`_cwt` is pycwt's arithmetic with the length pulled out, so at pycwt's own length it
    must return pycwt's own numbers, not numbers close to them."""
    n, dt = 5000, 1 / SFREQ
    sig = _signal(n)
    y = (sig - sig.mean()) / sig.std()
    grid = _at_length(_wavelet_grid(dt, n, 0.05, 0.20, limit_scales=True), _pow2(n))

    W, sj, freqs, coi = _cwt(y, dt, grid)
    W_ref, sj_ref, freqs_ref, coi_ref, _, _ = pycwt.cwt(
        y, dt, dj=1 / 12, s0=grid.s0, J=grid.J, wavelet=grid.mother)

    assert np.array_equal(W, W_ref), f"W differs, max {np.abs(W - W_ref).max()}"
    assert np.array_equal(sj, sj_ref)
    assert np.array_equal(freqs, freqs_ref)
    assert np.array_equal(coi, coi_ref)


# ---- the whole coherence ----

def _both_lengths(n, fmin):
    """One pairing computed at the chosen length and at the power of two, before trimming."""
    dt = 1 / SFREQ
    a, b = _signal(n, seed=4), _signal(n, seed=9)
    short = _wavelet_grid(dt, n, fmin, 0.20, limit_scales=True)
    assert short.n_fft < _pow2(n), "this shape does not exercise the change"
    out = []
    for grid in (short, _at_length(short, _pow2(n))):
        p1, p2 = _prepare_channel(a, dt, grid), _prepare_channel(b, dt, grid)
        W, aW = _pair_from_prepared(p1, p2, dt, grid)
        out.append((W, aW, p1.coi, p1.freqs))
    return out


@pytest.mark.parametrize("n,fmin", [(5000, 0.05), (20000, 0.01)])
def test_what_the_padding_moves_is_confined_to_the_scales_nothing_reads(n, fmin):
    """Shorter padding does move cells, and this pins where.

    The margin scales below ``fmin`` are the widest wavelets and the ones padding protects, and
    they exist only to give the band's own scales neighbours to be smoothed against. They are
    filtered out before anything is written. Inside the band that survives, the difference has
    to be back down at floating-point noise, and this asserts the whole ladder rather than the
    one rung a caller happens to look at.
    """
    (W1, _, _, freqs), (W2, _, _, _) = _both_lengths(n, fmin)
    d = np.abs(W1 - W2)
    kept = (freqs >= fmin) & (freqs <= 0.20)
    band = (freqs >= 0.06) & (freqs <= 0.15)

    # the margin scales do move, by four orders more than anything that is kept
    assert d.max() > 10 * d[kept].max(), (
        "the discarded margin scales moved no more than the kept ones, so this shape is not "
        "exercising what the test claims")
    assert d[kept].max() < 1e-5, "the written map moved, not just the discarded margin"
    assert d[band].max() < 1e-9, "the reported band moved by more than rounding"


@pytest.mark.parametrize("n,fmin", [(5000, 0.05), (20000, 0.01)])
def test_the_stored_map_and_the_band_mean_do_not_move(n, fmin):
    """What a reader actually gets: the float32 map on disk, and the number in the table.

    The map's lowest rows can move by a few units in float32's last place, in the cells at the
    record's edges that the cone masks off anyway. The band mean, which is what every table
    carries, does not move at all.
    """
    step = int(round(SFREQ))
    got, ref = (_trim_pair(W, aW, coi, freqs, n, step, fmin, 0.20)
                for W, aW, coi, freqs in _both_lengths(n, fmin))

    for name, g, r in zip(("wtc", "freqs", "coi", "sig", "phase"), got, ref):
        if r is None:
            assert g is None
            continue
        if name in ("freqs", "coi"):
            assert np.array_equal(g, r), f"{name} differs"
            continue
        diff = np.abs(g.astype(np.float64) - r.astype(np.float64)).max()
        assert diff < 1e-5, f"{name} moved by {diff}"

    band = (ref[1] >= 0.06) & (ref[1] <= 0.15)
    assert float(np.nanmean(got[0][band])) == pytest.approx(
        float(np.nanmean(ref[0][band])), abs=1e-12), "the band mean moved"


def test_too_little_padding_really_does_break_it():
    """The control. Without it the test above would pass on a rule that padded nothing."""
    n, fmin, dt, step = 20000, 0.01, 1 / SFREQ, int(round(SFREQ))
    a, b = _signal(n, seed=4), _signal(n, seed=9)
    ref_grid = _at_length(_wavelet_grid(dt, n, fmin, 0.20, limit_scales=True), _pow2(n))
    bare = _at_length(ref_grid, n)

    out = []
    for grid in (bare, ref_grid):
        p1, p2 = _prepare_channel(a, dt, grid), _prepare_channel(b, dt, grid)
        W, aW = _pair_from_prepared(p1, p2, dt, grid)
        out.append(_trim_pair(W, aW, p1.coi, p1.freqs, n, step, fmin, 0.20))

    assert np.abs(out[0][0] - out[1][0]).max() > 1e-3, (
        "transforming with no padding at all left the coherence alone, so this test is not "
        "measuring what it claims and neither is the one above")
