"""Reusing a channel's transform across its pairings must not move a single coherence value.

`_pairwise_wtc` transforms both signals of every pair. Four of the six steps behind a
coherence depend on one signal only, so crossing a montage repeats them once per pairing.
`_prepare_channel` hoists those four out of the pair loop and `_pair_from_prepared` keeps the
two that need both.

That is a speed change with no numerical excuse: it calls the same pycwt functions on the
same inputs in the same order, so the result is not merely close, it is identical. These
tests assert `array_equal` rather than `allclose` on purpose. A tolerance here would hide
exactly the mistakes worth catching, a scale grid that shifted or a pairing that ended up
against the wrong channel's auto spectrum, since both produce plausible coherences.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline.synchrony import (
    _pair_from_prepared, _pairwise_wtc, _prepare_channel, _trim_pair, _wavelet_grid,
    _wtc_over_pairs,
)

pycwt = pytest.importorskip("pycwt")

SFREQ = 10.1725
FMIN, FMAX = 0.01, 0.20


def _signals(n_channels: int, duration_s: float = 300.0, seed: int = 3) -> list[np.ndarray]:
    """Slow coupled signals plus noise, the shape of a filtered haemoglobin trace."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SFREQ)
    t = np.arange(n) / SFREQ
    common = np.sin(2 * np.pi * 0.08 * t) + 0.5 * np.sin(2 * np.pi * 0.12 * t)
    return [common * (0.5 + 0.2 * k) + rng.normal(0, 1, n) for k in range(n_channels)]


def _raw(n_times: int) -> mne.io.Raw:
    """Only sfreq and times are read off the recordings, so one channel is enough."""
    info = mne.create_info(["x"], SFREQ, "misc")
    return mne.io.RawArray(np.zeros((1, n_times)), info, verbose="ERROR")


def _reused(sig1, sig2, dt, step, limit_scales):
    grid = _wavelet_grid(dt, len(sig1), FMIN, FMAX, limit_scales)
    p1, p2 = _prepare_channel(sig1, dt, grid), _prepare_channel(sig2, dt, grid)
    WCT, aWCT = _pair_from_prepared(p1, p2, dt, grid)
    return _trim_pair(WCT, aWCT, p1.coi, p1.freqs, len(sig1), step, FMIN, FMAX)


# ---- the primitives ----

@pytest.mark.parametrize("limit_scales", [True, False])
def test_a_reused_transform_reproduces_pycwt_exactly(limit_scales):
    sig1, sig2 = _signals(2)
    dt, step = 1 / SFREQ, int(round(SFREQ))

    ref = _pairwise_wtc(sig1, sig2, dt, step, FMIN, FMAX, cache=False,
                        limit_scales=limit_scales)
    got = _reused(sig1, sig2, dt, step, limit_scales)

    names = ("wtc", "freqs", "coi", "sig", "phase")
    for name, a, b in zip(names, ref, got):
        if a is None:
            assert b is None, f"{name} appeared out of nowhere"
            continue
        assert np.array_equal(a, b), f"{name} differs, max {np.abs(a - b).max()}"


def test_the_grid_matches_the_scales_pycwt_picks():
    """The reuse rests on every channel landing on one grid, so the grid has to be pycwt's."""
    sig, = _signals(1)
    dt = 1 / SFREQ
    grid = _wavelet_grid(dt, len(sig), FMIN, FMAX, limit_scales=True)

    _, sj, _, _, _, _ = pycwt.cwt((sig - sig.mean()) / sig.std(), dt, dj=1 / 12,
                                  s0=grid.s0, J=grid.J, wavelet=grid.mother)
    assert np.array_equal(sj, grid.sj)


def test_a_signal_of_another_length_is_refused_rather_than_broadcast():
    """A shorter signal would broadcast against the grid instead of failing."""
    sig, = _signals(1)
    grid = _wavelet_grid(1 / SFREQ, len(sig), FMIN, FMAX, limit_scales=True)
    with pytest.raises(ValueError, match="differs from the grid"):
        _prepare_channel(sig[:-1], 1 / SFREQ, grid)


def test_a_grid_the_scales_fell_off_is_refused(monkeypatch):
    """A scale whose row comes back all-NaN is dropped, which would pair two channels on
    different grids. Simulated at the transform, since that is the only thing that drops one."""
    import fnirs_pipe.pipeline.synchrony as syn
    sig, = _signals(1)
    grid = _wavelet_grid(1 / SFREQ, len(sig), FMIN, FMAX, limit_scales=True)
    real = syn._cwt
    monkeypatch.setattr(syn, "_cwt",
                        lambda s, dt, g: tuple(x[:-1] for x in real(s, dt, g)[:3])
                                         + (real(s, dt, g)[3],))
    with pytest.raises(ValueError, match="wavelet scales differ"):
        syn._prepare_channel(sig, 1 / SFREQ, grid)


# ---- the loop that uses them ----

@pytest.mark.parametrize("cross", [True, False])
def test_the_pair_loop_reproduces_the_per_pair_route(cross):
    """The whole of `_wtc_over_pairs`, against calling `_pairwise_wtc` on each pairing."""
    labels = ["S1_D1", "S1_D2", "S2_D1"]
    sigs1 = _signals(len(labels), seed=3)
    sigs2 = _signals(len(labels), seed=11)
    n = len(sigs1[0])
    raws = {"sub-01": _raw(n), "sub-02": _raw(n)}
    signals = {"sub-01": dict(zip(labels, sigs1)), "sub-02": dict(zip(labels, sigs2))}

    result = _wtc_over_pairs(raws, signals, FMIN, FMAX, cross=cross, axis=labels)

    dt, step = 1 / SFREQ, int(round(SFREQ))
    pairs = result.pairs[("sub-01", "sub-02")]
    expected_keys = ([(a, b) for a in labels for b in labels] if cross else labels)
    assert list(pairs) == expected_keys

    for key in expected_keys:
        label1, label2 = key if cross else (key, key)
        ref = _pairwise_wtc(signals["sub-01"][label1], signals["sub-02"][label2],
                            dt, step, FMIN, FMAX, cache=False)
        got = pairs[key]
        assert np.array_equal(got["wtc"], ref[0]), f"{key} coherence differs"
        assert np.array_equal(got["phase"], ref[4]), f"{key} phase differs"
        assert np.array_equal(got["coi"], ref[2]), f"{key} cone differs"
    assert np.array_equal(result.freqs, _pairwise_wtc(
        sigs1[0], sigs2[0], dt, step, FMIN, FMAX, cache=False)[1])


def test_a_rejected_channel_still_blanks_only_its_own_pairings():
    """The reuse must not let one missing channel take its neighbours' pairings with it."""
    labels = ["S1_D1", "S1_D2"]
    sigs1, sigs2 = _signals(2, seed=3), _signals(2, seed=11)
    n = len(sigs1[0])
    raws = {"sub-01": _raw(n), "sub-02": _raw(n)}
    signals = {"sub-01": {labels[0]: sigs1[0]},
               "sub-02": dict(zip(labels, sigs2))}

    pairs = _wtc_over_pairs(raws, signals, FMIN, FMAX, cross=True,
                           axis=labels).pairs[("sub-01", "sub-02")]

    assert pairs[(labels[1], labels[0])] is None
    assert pairs[(labels[1], labels[1])] is None
    assert pairs[(labels[0], labels[0])] is not None
    assert pairs[(labels[0], labels[1])] is not None


def test_the_channel_cache_is_not_carried_between_members():
    """Two members share label names, so a cache keyed on the label alone would cross-feed."""
    labels = ["S1_D1"]
    a, b, c = _signals(3, seed=5)
    n = len(a)
    raws = {"sub-01": _raw(n), "sub-02": _raw(n), "sub-03": _raw(n)}
    signals = {"sub-01": {labels[0]: a}, "sub-02": {labels[0]: b}, "sub-03": {labels[0]: c}}

    result = _wtc_over_pairs(raws, signals, FMIN, FMAX, cross=True, axis=labels)

    dt, step = 1 / SFREQ, int(round(SFREQ))
    for (s1, s2), sig_a, sig_b in [(("sub-01", "sub-02"), a, b),
                                   (("sub-01", "sub-03"), a, c),
                                   (("sub-02", "sub-03"), b, c)]:
        ref = _pairwise_wtc(sig_a, sig_b, dt, step, FMIN, FMAX, cache=False)
        got = result.pairs[(s1, s2)][(labels[0], labels[0])]
        assert np.array_equal(got["wtc"], ref[0]), f"{s1}-{s2} coherence differs"


# ---- the cache carried between calls ----

def test_a_cache_carried_between_calls_changes_no_value():
    """`cache1` reuses the first side's transforms across calls; the numbers must not move.

    The pseudo-dyad null calls this once per iteration with the first subject unchanged, so
    the second call is the one that reads from the cache rather than filling it.
    """
    labels = ["S1_D1", "S1_D2", "S2_D1"]
    sigs1 = _signals(len(labels), seed=3)
    n = len(sigs1[0])
    raws = {"sub-01": _raw(n), "sub-02": _raw(n)}
    cache = {}

    for call, seed2 in enumerate((11, 23)):
        signals = {"sub-01": dict(zip(labels, sigs1)),
                   "sub-02": dict(zip(labels, _signals(len(labels), seed=seed2)))}
        cached = _wtc_over_pairs(raws, signals, FMIN, FMAX, axis=labels, cache1=cache)
        plain = _wtc_over_pairs(raws, signals, FMIN, FMAX, axis=labels)
        for label in labels:
            assert np.array_equal(cached.pairs[("sub-01", "sub-02")][label]["wtc"],
                                  plain.pairs[("sub-01", "sub-02")][label]["wtc"]), \
                f"call {call} label {label} differs"
    assert set(cache) == {("sub-01", label) for label in labels}


def test_the_carried_cache_is_keyed_by_member():
    """Same guard as the within-call cache: two members share label names."""
    labels = ["S1_D1"]
    a, b, c = _signals(3, seed=5)
    n = len(a)
    raws = {"sub-01": _raw(n), "sub-02": _raw(n), "sub-03": _raw(n)}
    signals = {"sub-01": {labels[0]: a}, "sub-02": {labels[0]: b}, "sub-03": {labels[0]: c}}

    result = _wtc_over_pairs(raws, signals, FMIN, FMAX, axis=labels, cache1={})

    dt, step = 1 / SFREQ, int(round(SFREQ))
    for (s1, s2), sig_a, sig_b in [(("sub-01", "sub-02"), a, b),
                                   (("sub-01", "sub-03"), a, c),
                                   (("sub-02", "sub-03"), b, c)]:
        ref = _pairwise_wtc(sig_a, sig_b, dt, step, FMIN, FMAX, cache=False)
        got = result.pairs[(s1, s2)][labels[0]]
        assert np.array_equal(got["wtc"], ref[0]), f"{s1}-{s2} coherence differs"


def test_a_cached_transform_from_another_grid_is_refused():
    """A stale entry misaligns the cross spectrum without changing its shape, so it raises."""
    labels = ["S1_D1"]
    short, long_ = _signals(1, duration_s=200.0)[0], _signals(1, duration_s=300.0)[0]
    cache = {}
    raws = {"sub-01": _raw(len(short)), "sub-02": _raw(len(short))}
    _wtc_over_pairs(raws, {"sub-01": {labels[0]: short}, "sub-02": {labels[0]: short}},
                    FMIN, FMAX, axis=labels, cache1=cache)

    raws = {"sub-01": _raw(len(long_)), "sub-02": _raw(len(long_))}
    with pytest.raises(ValueError, match="different grid"):
        _wtc_over_pairs(raws, {"sub-01": {labels[0]: long_}, "sub-02": {labels[0]: long_}},
                        FMIN, FMAX, axis=labels, cache1=cache)
