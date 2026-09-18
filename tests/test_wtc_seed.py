"""Reproducible Monte Carlo significance for wavelet coherence.

The surrogates are drawn inside pycwt from numpy's global legacy RNG, which takes no seed
argument. What we control is: seeding that RNG once before the pairs are walked, restoring
it afterwards, and switching off pycwt's on-disk cache, which is keyed on the wavelet grid
but not on the seed and would otherwise answer with a curve computed under unknown settings.

Seeding once rather than per pair is deliberate and is what the first two tests below pin.
One seed reused for every pair hands all but identical surrogates to channels with similar
autocorrelation, which would turn the Monte Carlo error into a bias shared by the whole
montage instead of noise that averages out across channels.

pycwt's own determinism given a seeded RNG is verified by hand, not here: 300 surrogates
per pair is far too slow for a suite. These assert our half of the contract.
"""

import numpy as np
import pytest

from fnirs_pipe.pipeline import synchrony

LABELS = ["S1_D1", "S1_D2", "S2_D2"]


@pytest.fixture
def calls(monkeypatch):
    """Replace pycwt.wct with a stand-in that records its kwargs and one random draw."""
    import pycwt

    recorded: list[dict] = []

    def _fake_wct(sig1, sig2, dt, dj, sig, normalize, **kwargs):
        recorded.append({"kwargs": kwargs, "draw": float(np.random.rand())})
        n_freq, n_time = 8, len(sig1)
        freqs = np.linspace(0.5, 0.001, n_freq)      # descending, as pycwt returns them
        return (np.zeros((n_freq, n_time)), None, np.zeros(n_time), freqs,
                np.linspace(0.1, 0.9, n_freq))

    monkeypatch.setattr(pycwt, "wct", _fake_wct)
    return recorded


@pytest.fixture
def inputs(make_raw):
    """Two subjects, three shared channel labels."""
    raws = {"sub-01": make_raw(n_ch=1, dur=32.0), "sub-02": make_raw(n_ch=1, dur=32.0)}
    signal = np.sin(np.linspace(0, 20, 64))
    signals = {sid: {label: signal for label in LABELS} for sid in raws}
    return raws, signals


def _run(inputs, seed, significance=True, mc_count=300):
    raws, signals = inputs
    return synchrony._wtc_over_pairs(
        raws, signals, fmin=0.004, fmax=0.2,
        significance=significance, seed=seed, mc_count=mc_count,
    )


def _draws(calls):
    return [c["draw"] for c in calls]


# ---- one seed, independent pairs ----

def test_each_pair_draws_its_own_surrogates(calls, inputs):
    # the whole reason the seed is applied once instead of per pair
    _run(inputs, seed=42)
    draws = _draws(calls)
    assert len(draws) == len(LABELS)
    assert len(set(draws)) == len(draws)


def test_the_same_seed_repeats_the_whole_run(calls, inputs):
    _run(inputs, seed=42)
    first = _draws(calls)
    calls.clear()
    _run(inputs, seed=42)

    assert _draws(calls) == first


def test_a_different_seed_gives_a_different_run(calls, inputs):
    _run(inputs, seed=42)
    first = _draws(calls)
    calls.clear()
    _run(inputs, seed=7)

    assert _draws(calls) != first


def test_no_seed_leaves_it_to_chance(calls, inputs):
    _run(inputs, seed=None)
    first = _draws(calls)
    calls.clear()
    _run(inputs, seed=None)

    assert _draws(calls) != first


def test_the_global_rng_is_put_back(calls, inputs):
    # np.random.seed is process-wide; leaving it set would quietly determine every later
    # draw anywhere in the run
    np.random.seed(1234)
    before = np.random.rand()
    np.random.seed(1234)

    _run(inputs, seed=42)

    assert np.random.rand() == before


# ---- the cache ----

def test_seeding_bypasses_the_cache(calls, inputs):
    # the cache key covers the AR1 coefficients and the wavelet grid, not the seed, so a
    # file from an earlier run would silently answer for the one this seed asks for
    _run(inputs, seed=42)
    assert all(c["kwargs"]["cache"] is False for c in calls)


def test_without_a_seed_the_cache_is_left_alone(calls, inputs):
    # unseeded runs keep the previous behaviour, including the speed the cache buys
    _run(inputs, seed=None)
    assert all("cache" not in c["kwargs"] for c in calls)


# ---- the surrogate count ----

def test_the_surrogate_count_reaches_pycwt(calls, inputs):
    # it drives the whole runtime, so a value that silently failed to arrive would look
    # like the flag doing nothing rather than like an error
    _run(inputs, seed=42, mc_count=17)
    assert all(c["kwargs"]["mc_count"] == 17 for c in calls)


def test_the_surrogate_count_defaults_to_pycwts_own(calls, inputs):
    _run(inputs, seed=42)
    assert all(c["kwargs"]["mc_count"] == 300 for c in calls)


def test_no_surrogate_count_without_significance(calls, inputs):
    # pycwt takes it through **kwargs and never looks at it when sig is off; passing it
    # anyway would suggest a cost that is not being paid
    _run(inputs, seed=None, significance=False, mc_count=17)
    assert all("mc_count" not in c["kwargs"] for c in calls)
