"""The coherence on a CUDA card agrees with the CPU path, and a card that fails stops the run."""

import numpy as np
import pandas as pd
import pytest

from nirspipe.exceptions import DeviceError
from nirspipe.pipeline.hyper import _gpu
from nirspipe.pipeline.hyper._gpu import CudaDevice, open_device
from nirspipe.pipeline.hyper.surrogate import compute_wtc_phase_null
from nirspipe.pipeline.hyper.wtc import compute_wtc, wtc_band_mean
from nirspipe.qc.common.report_shell import guard
from tests.hyper.test_hyper_censoring import BAND, SPANS, _dyad, _silent_dyad


def _card():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA card")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("no CUDA card")
    return open_device("cuda")


def _within_one_ulp(cpu, gpu, columns=slice(None)):
    a, b = np.asarray(cpu)[:, columns], np.asarray(gpu)[:, columns]
    assert np.array_equal(np.isnan(a), np.isnan(b))
    defined = ~np.isnan(a)
    gap = np.abs(a[defined].astype(np.float64) - b[defined])
    assert (gap <= np.spacing(np.maximum(np.abs(a[defined]), np.abs(b[defined])))).all()


def _same_maps(cpu, gpu, kept_only=False):
    assert np.array_equal(cpu.freqs, gpu.freqs) and np.array_equal(cpu.times, gpu.times)
    assert cpu.pairs.keys() == gpu.pairs.keys()
    for pair_key, labels in cpu.pairs.items():
        assert labels.keys() == gpu.pairs[pair_key].keys()
        columns = ~cpu.excluded[pair_key] if kept_only else slice(None)
        for label, data in labels.items():
            other = gpu.pairs[pair_key][label]
            assert np.array_equal(data["coi"], other["coi"])
            _within_one_ulp(data["wtc"], other["wtc"], columns)
            _within_one_ulp(data["phase"], other["phase"], columns)


# ---- the card against the CPU ----

def test_a_crossed_run_on_the_card_matches_the_cpu():
    device = _card()
    cpu = compute_wtc(_dyad(), fmin=0.02, fmax=0.2, cross=True)
    gpu = compute_wtc(_dyad(), fmin=0.02, fmax=0.2, cross=True, device=device)
    _same_maps(cpu, gpu)
    pd.testing.assert_frame_equal(wtc_band_mean(cpu, *BAND), wtc_band_mean(gpu, *BAND))
    assert device.check["device_check_cells"] > 0


@pytest.mark.parametrize("fill", ["flat", "linear"])
def test_a_long_filled_span_is_undefined_in_the_same_cells_on_the_card(fill):
    device = _card()
    cpu = compute_wtc(_silent_dyad(fill), fmin=0.02, fmax=0.2)
    gpu = compute_wtc(_silent_dyad(fill), fmin=0.02, fmax=0.2, device=device)
    # round-off inside the fill differs between FFT libraries; the kept columns may not
    _same_maps(cpu, gpu, kept_only=True)
    for pair_key, labels in gpu.pairs.items():
        for label, data in labels.items():
            assert np.array_equal(np.isnan(data["wtc"]),
                                  np.isnan(cpu.pairs[pair_key][label]["wtc"]))
    pd.testing.assert_frame_equal(wtc_band_mean(cpu, *BAND), wtc_band_mean(gpu, *BAND))


def test_a_burst_inside_the_spans_passes_the_check_and_gives_the_same_tables():
    device = _card()
    # the burst sets each row's largest power, so kept cells sit near the floor
    cpu = compute_wtc(_dyad(SPANS, garbage=5.0), fmin=0.02, fmax=0.2, cross=True)
    gpu = compute_wtc(_dyad(SPANS, garbage=5.0), fmin=0.02, fmax=0.2, cross=True,
                      device=device)
    assert device.check["device_check_cells_near_floor"] > 0
    pd.testing.assert_frame_equal(wtc_band_mean(cpu, *BAND), wtc_band_mean(gpu, *BAND))


def test_a_phase_null_on_the_card_draws_the_cpus_null():
    device = _card()
    kwargs = dict(n_iter=3, fmin=0.02, fmax=0.2, seed=7, cross=True)
    cpu = compute_wtc_phase_null(_dyad(), *BAND, **kwargs)
    gpu = compute_wtc_phase_null(_dyad(), *BAND, **kwargs, device=device)
    for a, b in zip(cpu.draws, gpu.draws):
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-6)


def test_transforms_left_uncached_on_a_full_card_give_the_same_maps():
    device = _card()
    cached = compute_wtc(_dyad(), fmin=0.02, fmax=0.2, cross=True, device=device)
    device.has_room = lambda: False
    recomputed = compute_wtc(_dyad(), fmin=0.02, fmax=0.2, cross=True, device=device)
    for pair_key, labels in cached.pairs.items():
        for label, data in labels.items():
            assert np.array_equal(data["wtc"], recomputed.pairs[pair_key][label]["wtc"],
                                  equal_nan=True)


# ---- a card that fails ----

def test_a_card_out_of_memory_stops_the_run_instead_of_blanking_a_pairing():
    device = _card()
    import cupy

    def out_of_memory(*args):
        raise cupy.cuda.memory.OutOfMemoryError(1 << 30, 0, 1 << 30)

    device.pair_from_prepared = out_of_memory
    with pytest.raises(DeviceError, match="ran out of memory"):
        compute_wtc(_dyad(), fmin=0.02, fmax=0.2, device=device)


def test_more_processes_than_the_card_holds_are_refused():
    _card()
    with pytest.raises(DeviceError, match="Lower --nprocs"):
        _gpu.check_card(1000)


def test_a_device_error_is_not_swallowed_by_a_report_section():
    errors: list = []
    with pytest.raises(DeviceError):
        with guard("WTC computation (hbo)", errors, "group-01"):
            raise DeviceError("the card failed")
    assert errors == []


def test_cpu_opens_no_device_and_an_unknown_name_is_refused():
    assert open_device("cpu") is None
    with pytest.raises(ValueError, match="unknown device"):
        open_device("tpu")


# ---- the check's rule, on hand-made maps ----

def _checker():
    # the rule needs no card: only the comparison runs
    device = object.__new__(CudaDevice)
    device.check = None
    return device


def _trimmed(wtc, phase):
    wtc, phase = np.asarray(wtc, np.float32), np.asarray(phase, np.float32)
    return wtc, np.array([0.05, 0.1]), np.zeros(wtc.shape[1], np.float32), None, phase


ONE = np.float32(1.0)
NEXT = np.nextafter(ONE, np.float32(2.0))        # one float32 unit above 1.0
SECOND = np.nextafter(NEXT, np.float32(2.0))     # two units above


FULL = np.ones((2, 2))                           # every cell well above the power floor


def test_one_float32_unit_passes_and_is_counted():
    device = _checker()
    cpu = _trimmed([[0.5, ONE], [0.25, 0.75]], [[0.1, 3.0], [-1.0, 2.0]])
    gpu = _trimmed([[0.5, NEXT], [0.25, 0.75]], [[0.1, 3.0], [-1.0, 2.0]])
    device.verify(cpu, gpu, FULL, None, "a S1_D1 - b S1_D1")
    assert device.check == {"device_check_pairing": "a S1_D1 - b S1_D1",
                            "device_check_cells": 4, "device_check_cells_one_ulp": 1,
                            "device_check_cells_near_floor": 0}


def test_two_float32_units_refuse_the_run():
    with pytest.raises(DeviceError, match="1 of 4 cells"):
        _checker().verify(_trimmed([[0.5, ONE], [0.25, 0.75]], np.zeros((2, 2))),
                          _trimmed([[0.5, SECOND], [0.25, 0.75]], np.zeros((2, 2))),
                          FULL, None, "p")


def test_a_cell_near_the_power_floor_is_counted_but_not_judged():
    device = _checker()
    device.verify(_trimmed([[0.5, 0.9]], [[0.0, 0.0]]), _trimmed([[0.5, 0.1]], [[0.0, 2.0]]),
                  np.array([[1.0, 1e-8]]), None, "p")
    assert device.check["device_check_cells"] == 1
    assert device.check["device_check_cells_near_floor"] == 1


def test_undefined_cells_must_be_the_same_ones():
    with pytest.raises(DeviceError, match="undefined in other cells"):
        _checker().verify(_trimmed([[np.nan, 0.5]], [[0.0, 0.0]]),
                          _trimmed([[0.4, 0.5]], [[0.0, 0.0]]), FULL[:1], None, "p")


def test_excluded_columns_are_not_judged():
    device = _checker()
    device.verify(_trimmed([[0.5, 0.9]], [[0.0, 0.0]]),
                  _trimmed([[0.5, 0.1]], [[0.0, 1.0]]),
                  FULL[:1], np.array([False, True]), "p")
    assert device.check["device_check_cells"] == 1


def test_the_same_angle_either_side_of_the_branch_cut_agrees():
    device = _checker()
    edge = np.float32(np.pi)
    device.verify(_trimmed([[0.5]], [[edge]]), _trimmed([[0.5]], [[-edge]]), FULL[:1, :1],
                  None, "p")
    assert device.check["device_check_cells_one_ulp"] == 1


def test_a_pairing_with_no_defined_kept_cell_leaves_the_check_open():
    device = _checker()
    device.verify(_trimmed([[np.nan, 0.5]], [[0.0, 0.0]]),
                  _trimmed([[np.nan, 0.5]], [[0.0, 0.0]]), np.array([[0.0, 1.0]]),
                  np.array([False, True]), "p")
    assert device.check is None
