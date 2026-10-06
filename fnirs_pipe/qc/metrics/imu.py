"""Head movement from the IMU: its summary over a stretch, and its agreement with GVTD.

The IMU is the one record of movement not taken from the optical data. The traces come from
:func:`~fnirs_pipe.io.auxiliary.imu_traces`; nothing here reads a file.
"""

from typing import Any

import mne
import numpy as np
from scipy.stats import spearmanr

from fnirs_pipe.qc.metrics._helpers import _safe_metrics, as_optical_density
from fnirs_pipe.qc.metrics.gvtd import GVTD_MOTION_BAND, gvtd_channel_picks, gvtd_timetrace

# record name of each magnitude -> its key in `imu_traces`
IMU_QUANTITIES = {"gyro_speed": "gyro", "accel_jerk": "accel"}
# the keys `imu_scalars` declares, in print order
IMU_STAT_KEYS = ("gyro_speed_mean", "gyro_speed_median", "gyro_speed_p95",
                 "accel_jerk_mean", "accel_jerk_median", "accel_jerk_p95")
# seconds either side of a GVTD frame the IMU is averaged over before the two are ranked
AGREEMENT_HALF_WINDOW_S = 1.0


@_safe_metrics("IMU summary", (
    "gyro_speed_mean", "gyro_speed_median", "gyro_speed_p95",
    "accel_jerk_mean", "accel_jerk_median", "accel_jerk_p95",
))
def imu_scalars(imu: dict, t0: float, t1: float) -> dict[str, float]:
    """Mean, median and 95th percentile of each IMU magnitude between ``t0`` and ``t1``.

    ::

        {"gyro": ImuTrace(t, |omega|, "°/s")}, 0.0, 300.0
        -> {"gyro_speed_mean": 1.4, "gyro_speed_median": 0.6, "gyro_speed_p95": 5.2,
            "accel_jerk_mean": None, ...}

    A sensor the recording lacks, or one with no sample in the stretch, stays None.
    """
    out: dict[str, float] = {}
    for name, sensor in IMU_QUANTITIES.items():
        trace = imu.get(sensor)
        if trace is None:
            continue
        t, y = np.asarray(trace.t, dtype=float), np.asarray(trace.y, dtype=float)
        y = y[(t >= t0) & (t <= t1) & np.isfinite(y)]
        if not y.size:
            continue
        out[f"{name}_mean"] = float(y.mean())
        out[f"{name}_median"] = float(np.median(y))
        out[f"{name}_p95"] = float(np.percentile(y, 95))
    return out


def _window_mean(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray,
                 half: float) -> np.ndarray:
    """Mean of ``y`` within ``half`` seconds of each ``t_dst``, NaN where no sample falls.

    Running sums, so the cost is one pass over the IMU whatever the window::

        t_src [0, 1, 2, 3], y [0, 2, 4, 6], t_dst [1.5], half 1.0  ->  [3.0]
    """
    csum = np.concatenate(([0.0], np.cumsum(y)))
    i0 = np.searchsorted(t_src, t_dst - half)
    i1 = np.searchsorted(t_src, t_dst + half, side="right")
    n = i1 - i0
    return np.where(n > 0, (csum[i1] - csum[i0]) / np.maximum(n, 1), np.nan)


@_safe_metrics("IMU agreement with GVTD", ("gyro_speed_gvtd_rho", "accel_jerk_gvtd_rho"))
def imu_gvtd_agreement(imu: dict, gvtd: np.ndarray, gvtd_times: np.ndarray) -> dict[str, float]:
    """Spearman rho between GVTD and each IMU magnitude, frame by frame.

    Each GVTD frame is paired with the IMU averaged over ``AGREEMENT_HALF_WINDOW_S`` either
    side of it, about the smear the motion band puts on a movement; frames the IMU does not
    cover are left out. A sensor with fewer than three paired frames, or a side that never
    varies, has no rank order and stays None.
    """
    gvtd = np.asarray(gvtd, dtype=float)
    gvtd_times = np.asarray(gvtd_times, dtype=float)
    out: dict[str, float] = {}
    for name, sensor in IMU_QUANTITIES.items():
        trace = imu.get(sensor)
        if trace is None:
            continue
        t, y = np.asarray(trace.t, dtype=float), np.asarray(trace.y, dtype=float)
        keep = np.isfinite(t) & np.isfinite(y)
        paired = _window_mean(t[keep], y[keep], gvtd_times, AGREEMENT_HALF_WINDOW_S)
        ok = np.isfinite(paired) & np.isfinite(gvtd)
        if ok.sum() < 3 or np.ptp(paired[ok]) == 0 or np.ptp(gvtd[ok]) == 0:
            continue
        out[f"{name}_gvtd_rho"] = float(spearmanr(gvtd[ok], paired[ok])[0])
    return out


def imu_section(imu: dict, raw: mne.io.Raw, sep_bands=None) -> dict[str, Any]:
    """The run's ``imu`` record section: the summaries, their units and the agreement.

    Summarised over the optical recording's span, and ranked against the motion-band GVTD
    of the channel set a run is judged on, so the number reads beside ``gvtd_filt_mean``.
    ``raw`` is intensity or optical density.
    """
    raw_od = as_optical_density(raw)
    times = raw_od.times
    picks, _ = gvtd_channel_picks(raw_od, sep_bands)
    gvtd = gvtd_timetrace(raw_od.get_data(picks=picks), float(raw_od.info["sfreq"]),
                          *GVTD_MOTION_BAND)
    # the unit the recording declared, so a number is never read on another device's scale
    units = {f"{name}_unit": imu[sensor].unit for name, sensor in IMU_QUANTITIES.items()
             if sensor in imu and imu[sensor].unit}
    return {**imu_scalars(imu, float(times[0]), float(times[-1])),
            **imu_gvtd_agreement(imu, gvtd, times[1:]),
            **units}
