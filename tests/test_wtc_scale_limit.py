"""Limiting the wavelet scales must not move a single coherence value.

`_scale_range` computes only the scales the [fmin, fmax] filter keeps, plus margin, instead
of every scale the record length allows. That is a speed change dressed as a correctness
risk: it is safe only because the scales it picks land on pycwt's own grid, and because the
margin is wider than the boxcar `wct` smooths across scales with.

Both halves have already been wrong once. The first version anchored the grid at `2 * dt`
where pycwt anchors it at `2 * dt / flambda`, which shifted every frequency and moved the
band means with it. Nothing failed; the numbers were just quietly different. These tests
compare against the unrestricted transform so that cannot happen silently again.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline.synchrony import _FLAMBDA, _pairwise_wtc, _scale_range

pycwt = pytest.importorskip("pycwt")

DJ = 1 / 12


def _pair(duration_s: float, sfreq: float, seed: int = 7):
    """Two coupled slow signals plus noise, the shape of a filtered HbO pair."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    t = np.arange(n) / sfreq
    common = np.sin(2 * np.pi * 0.08 * t) + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    return common + rng.normal(0, 1, n), np.roll(common, 3) + rng.normal(0, 1, n)


CASES = [
    (duration, sfreq, fmin, fmax)
    for duration, sfreq in ((300.0, 10.1725), (900.0, 10.1725), (120.0, 5.0))
    for fmin, fmax in ((0.01, 0.25), (0.004, 0.20), (0.05, 0.15))
]


@pytest.mark.parametrize("duration,sfreq,fmin,fmax", CASES)
def test_limiting_scales_reproduces_the_full_transform(duration, sfreq, fmin, fmax):
    x, y = _pair(duration, sfreq)
    dt, step = 1 / sfreq, int(round(sfreq))

    full = _pairwise_wtc(x, y, dt, step, fmin, fmax, cache=False, limit_scales=False)
    limited = _pairwise_wtc(x, y, dt, step, fmin, fmax, cache=False, limit_scales=True)

    wtc_full, freqs_full, coi_full, _, phase_full = full
    wtc_lim, freqs_lim, coi_lim, _, phase_lim = limited

    assert freqs_full.shape == freqs_lim.shape
    np.testing.assert_allclose(freqs_full, freqs_lim, rtol=1e-12)
    np.testing.assert_allclose(coi_full, coi_lim, rtol=1e-12)
    # the coherences themselves, which is what a band mean is taken over
    np.testing.assert_allclose(wtc_full, wtc_lim, atol=1e-6)
    # and the phase, which the arrows are drawn from. It is band-limited and decimated by
    # the same indexing, so a slip there would rotate every arrow without touching the map
    np.testing.assert_allclose(phase_full, phase_lim, atol=1e-6)


@pytest.mark.parametrize("duration,sfreq,fmin,fmax", CASES)
def test_the_limited_scales_lie_on_pycwt_own_grid(duration, sfreq, fmin, fmax):
    """s0 must be a whole number of dj steps from pycwt's default, or the grid shifts."""
    dt = 1 / sfreq
    n = int(duration * sfreq)
    s0, _ = _scale_range(dt, DJ, fmin, fmax, n)

    steps = np.log2(s0 / (2 * dt / _FLAMBDA)) / DJ
    assert steps == pytest.approx(round(steps), abs=1e-9)
    assert steps >= -1e-9, "s0 below pycwt's smallest default scale"


def test_limiting_actually_drops_scales():
    """A guard on the point of the exercise: it has to be cheaper than not doing it."""
    sfreq, duration = 10.1725, 900.0
    dt, n = 1 / sfreq, int(duration * sfreq)
    s0, J = _scale_range(dt, DJ, 0.01, 0.25, n)

    default_s0 = 2 * dt / _FLAMBDA
    default_J = int(round(np.log2(n * dt / default_s0) / DJ))
    assert J + 1 < 0.7 * (default_J + 1)
