"""The ISC phase null with one shared set of phases per draw, against each channel scrambled alone.

Each check is a property of the surrogate a hand calculation fixes: a rotation by a shared
angle per frequency keeps every spectrum and every cross-spectrum, so two identical channels
stay identical; scrambling each alone keeps the spectra only.
"""

import numpy as np

from nirspipe.pipeline.hyper.isc import _isc_null_draws
from nirspipe.pipeline.hyper.surrogate import phase_rotate, phase_scramble


def _rows(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal((3, n)), axis=1)
    base[2] = 0.6 * base[0] + 0.4 * base[1]
    return base


def test_each_row_keeps_its_amplitude_spectrum_and_its_mean():
    rows = _rows()
    out = phase_rotate(rows, np.random.default_rng(1))
    np.testing.assert_allclose(np.abs(np.fft.rfft(out, axis=1)),
                               np.abs(np.fft.rfft(rows, axis=1)), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(out.mean(axis=1), rows.mean(axis=1), atol=1e-12)
    assert not np.allclose(out, rows)


def test_the_rows_keep_their_cross_spectrum():
    rows = _rows()
    out = phase_rotate(rows, np.random.default_rng(2))
    fx, fy = np.fft.rfft(rows, axis=1), np.fft.rfft(out, axis=1)
    np.testing.assert_allclose(fy[0] * np.conj(fy[2]), fx[0] * np.conj(fx[2]),
                               rtol=1e-9, atol=1e-6)


def test_two_identical_channels_stay_identical_only_under_the_shared_rotation():
    rows = _rows()
    rows[1] = rows[0]
    shared = phase_rotate(rows, np.random.default_rng(3))
    np.testing.assert_allclose(shared[1], shared[0], atol=1e-12)
    rng = np.random.default_rng(3)
    alone = np.array([phase_scramble(r, rng) for r in rows])
    assert not np.allclose(alone[1], alone[0])


def test_the_null_draws_follow_the_switch():
    member1, member2 = _rows(seed=4), _rows(seed=5)
    member2[1] = member2[0]
    shared = _isc_null_draws(member1, member2, 0, 6, seed=7, shared=True)
    alone = _isc_null_draws(member1, member2, 0, 6, seed=7)
    # every member-1 channel correlates with the two copies alike in each shared draw
    np.testing.assert_allclose(shared[:, :, 1], shared[:, :, 0], atol=1e-12)
    assert not np.allclose(alone[:, :, 1], alone[:, :, 0])


def test_a_blank_row_stays_blank_under_the_shared_rotation():
    member1, member2 = _rows(seed=8), _rows(seed=9)
    member2[2] = np.nan
    draws = _isc_null_draws(member1, member2, 0, 3, seed=1, shared=True)
    assert np.isnan(draws[:, :, 2]).all()
    assert np.isfinite(draws[:, :, :2]).all()


# ---- the coherence's phase null ----

def test_the_coherence_draw_keeps_two_identical_channels_identical_only_when_shared():
    from nirspipe.pipeline.hyper.surrogate import _surrogate_draw

    rows = _rows()
    signals = {"S1_D1": rows[0], "S2_D2": rows[0].copy(), "S3_D3": rows[2]}
    shared = _surrogate_draw(signals, np.random.default_rng(4), shared=True)
    alone = _surrogate_draw(signals, np.random.default_rng(4), shared=False)
    np.testing.assert_allclose(shared["S1_D1"], shared["S2_D2"], atol=1e-9)
    assert not np.allclose(alone["S1_D1"], alone["S2_D2"])
    for label, sig in signals.items():
        for out in (shared, alone):
            np.testing.assert_allclose(np.abs(np.fft.rfft(out[label])),
                                       np.abs(np.fft.rfft(sig)), rtol=1e-9, atol=1e-9)


def test_a_coherence_channel_with_a_gap_is_left_as_it_is_and_the_rest_still_rotate():
    from nirspipe.pipeline.hyper.surrogate import _surrogate_draw

    rows = _rows()
    blank = rows[1].copy()
    blank[10] = np.nan
    out = _surrogate_draw({"a": rows[0], "b": blank, "c": rows[0].copy()},
                          np.random.default_rng(5), shared=True)
    np.testing.assert_array_equal(out["b"], blank)
    np.testing.assert_allclose(out["a"], out["c"], atol=1e-9)
    assert not np.allclose(out["a"], rows[0])
