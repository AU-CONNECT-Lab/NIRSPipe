"""A condition's own cone: the cells near its start and end that carry the neighbouring condition."""

import numpy as np
import pytest

from nirspipe.pipeline.hyper.wtc import (WTCResult, _pair_from_prepared, _prepare_channel,
                                         _wavelet_grid, condition_cone, window_result,
                                         wtc_band_mean)
from nirspipe.qc.common.windows import condition_margins, split_windows
from nirspipe.qc.boilerplate.vocabulary import _condition_route


def _result(times, coi, n_freqs=3):
    data = {"wtc": np.full((n_freqs, len(times)), 0.5), "coi": np.asarray(coi, dtype=float),
            "phase": np.zeros((n_freqs, len(times)))}
    return WTCResult(pairs={("a", "b"): {"S1_D1": data}}, freqs=np.array([0.02, 0.05, 0.1]),
                     times=np.asarray(times, dtype=float))


def test_the_cone_is_the_one_pycwt_gives_a_series_cut_at_the_condition():
    import pycwt

    times = np.arange(0.0, 1000.0)
    got = condition_cone(_result(times, np.full(times.size, 1e6)), 300.0, 600.0)
    cut = times[(times >= 300.0) & (times <= 600.0)]
    *_, coi, _, _ = pycwt.cwt(np.random.default_rng(0).standard_normal(cut.size), 1.0)
    inside = _map(got)["coi"][(times >= 300.0) & (times <= 600.0)]
    # pycwt counts half a sample more distance at each edge
    assert inside == pytest.approx(coi, abs=0.4)


def test_at_0_02_hz_the_cone_reaches_68_s_into_the_condition():
    times = np.arange(0.0, 1000.0)
    coi = _map(condition_cone(_result(times, np.full(times.size, 1e6)), 300.0, 600.0))["coi"]
    reliable = times[1.0 / np.where(coi > 0, coi, np.nan) <= 0.02]
    assert reliable.min() == pytest.approx(300.0 + 68.4, abs=1.0)
    assert reliable.max() == pytest.approx(600.0 - 68.4, abs=1.0)


def test_the_record_cone_is_kept_where_it_reaches_further():
    times = np.arange(0.0, 400.0)
    record = np.full(times.size, 1e6)
    record[:50] = 0.0
    coi = _map(condition_cone(_result(times, record), 0.0, 400.0))["coi"]
    assert np.all(coi[:50] == 0.0)
    assert coi[200] > 0


def test_an_empty_result_comes_back_as_it_was():
    empty = WTCResult(pairs={}, freqs=np.array([]), times=np.array([]))
    assert condition_cone(empty, 0.0, 10.0) is empty


def test_a_window_keeps_its_condition_edges():
    conditions = [("rest", 0.0, 100.0), ("task", 200.0, 500.0)]
    windows, sources = split_windows(conditions, 100.0)
    assert condition_margins(conditions, windows, sources) == {
        "rest-w1": (0.0, 0.0), "task-w1": (0.0, 200.0), "task-w2": (100.0, 100.0),
        "task-w3": (200.0, 0.0)}


def test_an_unsplit_condition_is_its_own_edges():
    conditions = [("task", 200.0, 500.0)]
    assert condition_margins(conditions, conditions, {}) == {"task": (0.0, 0.0)}


def _band_noise(rng, n, lo=0.01, hi=0.2):
    f = np.fft.rfftfreq(n, 1.0)
    spec = np.where((f >= lo) & (f <= hi), 1.0, 0.0) * np.exp(1j * rng.uniform(0, 2 * np.pi, f.size))
    x = np.fft.irfft(spec, n=n)
    return x / x.std()


def test_the_coupled_condition_beside_an_uncoupled_one_regains_its_level():
    """Coupling only after 600 s: read whole, the coupled block's first minute is diluted."""
    n = 1800
    grid = _wavelet_grid(1.0, n, 0.01, 0.15, limit_scales=True)
    plain, coned, interior = [], [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        shared, coupled = _band_noise(rng, n), np.arange(n) >= 600
        x = np.where(coupled, shared, _band_noise(rng, n)) + 0.3 * _band_noise(rng, n)
        y = np.where(coupled, shared, _band_noise(rng, n)) + 0.3 * _band_noise(rng, n)
        p1, p2 = _prepare_channel(x, 1.0, grid), _prepare_channel(y, 1.0, grid)
        r2, _ = _pair_from_prepared(p1, p2, 1.0, grid)
        whole = WTCResult(pairs={("a", "b"): {"S1_D1": {"wtc": r2, "coi": p1.coi,
                                                        "phase": np.zeros_like(r2)}}},
                          freqs=p1.freqs, times=np.arange(n, dtype=float))
        block = window_result(whole, 600.0, 900.0)
        plain.append(_band_mean(block))
        coned.append(_band_mean(condition_cone(block, 600.0, 900.0)))
        interior.append(_band_mean(window_result(whole, 750.0, 1650.0)))
    plain, coned, interior = map(np.mean, (plain, coned, interior))
    assert coned - plain > 0.015
    assert abs(coned - interior) < abs(plain - interior) / 2


def _band_mean(result):
    return float(wtc_band_mean(result, 0.02, 0.10)["coherence"].iloc[0])


@pytest.mark.parametrize("cone, mask, said", [(True, True, True), (False, True, False),
                                              (True, False, False)])
def test_the_methods_name_the_condition_cone_only_where_it_was_applied(cone, mask, said):
    text = _condition_route({"condition_windows_s": {"task": [0, 300]}, "wtc_cond_cone": cone,
                             "mask_coi": mask})
    assert ("own start and end" in text) is said


def _map(result):
    return next(iter(next(iter(result.pairs.values())).values()))
