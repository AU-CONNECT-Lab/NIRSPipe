"""The dyad motion panel's IMU rows sit on the shared clock, not on each member's own.

Two members aligned by the same amount would hide a clock mix-up, since both columns would be
wrong together, so every test here gives the members different shifts.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.pipeline.hyper.alignment import align_imu_like, align_like
from fnirs_pipe.qc.figures.hyper.hyper_figures import build_motion_panel, motion_series
from tests._synth import synth_raw

SHIFTS = {"sub-01": 5.0, "sub-02": 12.0}
EVENT_OWN_S = 30.0


def _aligned(raws):
    """Each member cut as trigger alignment would: a different start, one shared length."""
    length = min(float(r.times[-1]) - SHIFTS[sid] for sid, r in raws.items()) - 1.0
    return {sid: r.copy().crop(tmin=SHIFTS[sid], tmax=SHIFTS[sid] + length)
            for sid, r in raws.items()}


def _imu(t_end: float, scale: float = 1.0):
    t = np.arange(0.0, t_end, 0.01)
    y = scale * (1.0 + 0.1 * np.abs(np.sin(t)))
    y[np.argmin(np.abs(t - EVENT_OWN_S))] = 50.0 * scale
    return {"gyro": (t, y)}


@pytest.fixture(scope="module")
def members():
    raws = {sid: synth_raw(sid.removeprefix("sub-"), "tapping", duration=120.0)
            for sid in SHIFTS}
    return raws, _aligned(raws)


def test_each_member_moves_by_its_own_shift(members):
    raws, aligned = members
    imu = align_imu_like({sid: _imu(120.0) for sid in raws}, raws, aligned)

    for sid, shift in SHIFTS.items():
        t, y = imu[sid]["gyro"]
        assert t[np.argmax(y)] == pytest.approx(EVENT_OWN_S - shift, abs=0.01)
        assert t[0] >= 0.0 and t[-1] <= aligned[sid].times[-1] + 1e-9


def test_imu_and_optical_copy_land_on_the_same_sample(members):
    """The IMU is moved by the shift ``align_like`` cuts the optical copy by, so an event
    stamped at one own-clock time on both is found at one shared-clock time on both."""
    raws, aligned = members
    marked = {}
    for sid, raw in raws.items():
        data = raw.get_data()
        data[:, np.argmin(np.abs(raw.times - EVENT_OWN_S))] *= 3.0
        marked[sid] = mne.io.RawArray(data, raw.info, verbose="error")
    optical = align_like(marked, aligned)
    imu = align_imu_like({sid: _imu(120.0) for sid in raws}, marked, aligned)

    for sid in SHIFTS:
        od_t = optical[sid].times[np.argmax(optical[sid].get_data()[0])]
        t, y = imu[sid]["gyro"]
        assert t[np.argmax(y)] == pytest.approx(od_t, abs=1.0 / optical[sid].info["sfreq"])


def test_each_member_is_divided_by_its_own_median(members):
    """Two IMUs share no absolute scale, so each row reads x its own median, and the
    median itself is kept for the label."""
    raws, aligned = members
    imu = align_imu_like({"sub-01": _imu(120.0), "sub-02": _imu(120.0, scale=0.3)},
                         raws, aligned)
    motion = motion_series(align_like(raws, aligned), None, list(SHIFTS), imu=imu)
    row = motion["imu"]["gyro"]

    for _, y in row["members"]:
        assert np.median(y) == pytest.approx(1.0)
    assert row["medians"]["sub-02"] / row["medians"]["sub-01"] == pytest.approx(0.3, rel=0.05)
    assert np.array_equal(row["both"], np.minimum(*[y for _, y in row["members"]]))


def test_the_panel_draws_the_imu_above_the_gvtd_rows(members):
    raws, aligned = members
    imu = align_imu_like({sid: _imu(120.0) for sid in raws}, raws, aligned)
    motion = motion_series(align_like(raws, aligned), None, list(SHIFTS), imu=imu)
    fig = build_motion_panel(motion, "before")

    labels = [str(a.text) for a in fig.layout.annotations]
    assert any("gyroscope" in text for text in labels)
    assert any(text.startswith("median") for text in labels)
    imu_axis = next(a.yref for a in fig.layout.annotations if "gyroscope" in str(a.text))
    gvtd_axis = next(a.yref for a in fig.layout.annotations if "GVTD" in str(a.text))
    # rows are numbered top down, so the IMU row's axis comes first
    assert int(imu_axis.split()[0][1:] or 1) < int(gvtd_axis.split()[0][1:] or 1)


def test_no_imu_draws_no_row(members):
    raws, aligned = members
    motion = motion_series(align_like(raws, aligned), None, list(SHIFTS))
    fig = build_motion_panel(motion, "before")

    assert motion["imu"] == {}
    assert not any("gyroscope" in str(a.text) for a in fig.layout.annotations)
