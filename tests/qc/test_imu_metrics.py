"""The IMU summaries and their agreement with GVTD, on traces whose answers are known.

The IMU is the one record of movement the optical data cannot produce for itself, so what
is pinned here is that its numbers are the plain statistics of the trace over the stretch
asked for, that a missing sensor is a missing number rather than a zero, and that the
agreement ranks GVTD against the sensor rather than against anything else.
"""

import numpy as np
from numpy.testing import assert_allclose

from nirspipe.io.auxiliary import ImuTrace
from nirspipe.qc.metrics import (IMU_STAT_KEYS, compute_windowed_gvtd, imu_gvtd_agreement,
                                   imu_scalars, imu_section, imu_windowed, window_grid)
from nirspipe.qc.metrics.imu import _window_mean
from tests._synth import synth_raw


def _gyro(t, y):
    return {"gyro": ImuTrace(np.asarray(t, float), np.asarray(y, float), "°/s")}


def test_the_summary_is_the_trace_statistics_over_the_stretch():
    t = np.arange(10.0)
    s = imu_scalars(_gyro(t, t), 2.0, 5.0)
    assert s["gyro_speed_mean"] == 3.5
    assert_allclose(s["gyro_speed_p95"], np.percentile([2.0, 3.0, 4.0, 5.0], 95))


def test_a_sensor_the_recording_lacks_is_none_not_zero():
    s = imu_scalars(_gyro(np.arange(10.0), np.ones(10)), 0.0, 9.0)
    assert set(s) == set(IMU_STAT_KEYS)
    assert s["gyro_speed_mean"] == 1.0
    assert all(s[k] is None for k in IMU_STAT_KEYS if k.startswith("accel_jerk"))


def test_a_stretch_with_no_sample_is_none():
    s = imu_scalars(_gyro(np.arange(10.0), np.ones(10)), 20.0, 30.0)
    assert s["gyro_speed_mean"] is None


def test_the_window_mean_takes_every_sample_within_reach():
    out = _window_mean(np.arange(4.0), np.array([0.0, 2.0, 4.0, 6.0]),
                       np.array([1.5, 10.0]), 1.0)
    assert out[0] == 3.0
    assert np.isnan(out[1])


def test_agreement_follows_the_sensor_and_its_sign():
    """GVTD built from the gyroscope ranks with it; built against it, ranks the other way."""
    t = np.arange(0.0, 100.0, 0.01)
    y = np.zeros_like(t)
    for onset in (10.0, 40.0, 70.0):
        y[(t >= onset) & (t < onset + 3.0)] = 5.0
    frames = np.arange(0.5, 99.5, 0.1)
    moved = _window_mean(t, y, frames, 1.0)
    assert imu_gvtd_agreement(_gyro(t, y), moved, frames)["gyro_speed_gvtd_rho"] > 0.99
    assert imu_gvtd_agreement(_gyro(t, y), -moved, frames)["gyro_speed_gvtd_rho"] < -0.99


def test_agreement_needs_frames_the_sensor_covers():
    t = np.arange(0.0, 10.0, 0.01)
    rho = imu_gvtd_agreement(_gyro(t, t), np.ones(5), np.arange(50.0, 55.0))
    assert rho["gyro_speed_gvtd_rho"] is None


def test_the_run_section_carries_the_unit_beside_the_numbers():
    raw = synth_raw("01", "tapping", duration=120.0)
    t = np.arange(0.0, raw.times[-1], 0.01)
    section = imu_section(_gyro(t, 1.0 + np.abs(np.sin(t))), raw)
    assert section["gyro_speed_unit"] == "°/s"
    assert "accel_jerk_unit" not in section
    assert -1.0 <= section["gyro_speed_gvtd_rho"] <= 1.0


# ---- per window, on the grid the GVTD series use ----

def test_the_grid_is_whole_windows_of_ceil_samples():
    win, starts, ends = window_grid(100, 10.0, 3.0)
    assert win == 30
    assert starts.tolist() == [0, 30, 60] and ends.tolist() == [30, 60, 90]


def test_imu_windows_are_the_gvtd_windows():
    """A trace equal to its own clock averages to each window's centre, so the IMU's window
    means land on the GVTD's window times one for one."""
    raw = synth_raw("01", "tapping", duration=120.0)
    _, _, gvtd_times = compute_windowed_gvtd(raw, 10.0)
    t = np.arange(0.0, raw.times[-1], 0.001)
    series = imu_windowed(_gyro(t, t), raw.times, raw.info["sfreq"], 10.0)
    assert len(series["gyro_speed_per_window"]) == len(gvtd_times)
    assert_allclose(series["gyro_speed_per_window"], gvtd_times, atol=0.01)
    assert "accel_jerk_per_window" not in series


def test_a_window_the_sensor_missed_is_nan_not_dropped():
    raw = synth_raw("01", "tapping", duration=120.0)
    t = np.arange(0.0, 50.0, 0.01)
    series = imu_windowed(_gyro(t, np.ones_like(t)), raw.times, raw.info["sfreq"], 10.0)
    means = np.asarray(series["gyro_speed_per_window"])
    assert len(means) == len(window_grid(len(raw.times), raw.info["sfreq"], 10.0)[1])
    assert np.all(means[:4] == 1.0) and np.all(np.isnan(means[6:]))
