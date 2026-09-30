import numpy as np
import pandas as pd
import pytest

from fnirs_pipe.io.auxiliary import (
    TIME_COLUMN, accel_jerk, gyro_speed, imu_traces, table_channels,
)


def test_speed_is_the_magnitude_over_the_axes_with_each_offset_removed():
    t = np.arange(0.0, 10.0, 0.01)
    x = np.full_like(t, 0.7)             # a resting offset, not movement
    x[500] = 3.7
    y = np.full_like(t, -0.2)
    y[500] = 3.8
    times = {"GYRO_X_1": t, "GYRO_Y_1": t, "ACCEL_X_1": t}
    values = {"GYRO_X_1": x, "GYRO_Y_1": y, "ACCEL_X_1": np.full_like(t, 9.81)}

    t_out, speed, _ = gyro_speed(times, values)

    assert t_out is t
    assert speed[500] == pytest.approx(5.0)
    assert np.delete(speed, 500) == pytest.approx(0.0)


def test_jerk_ignores_gravity_and_sees_a_jolt():
    """A still sensor reads gravity split over its axes by the tilt; the derivative is zero
    there whatever the tilt, and a step of 3 and 4 m/s^2 over one 0.01 s sample is 500 m/s^3."""
    t = np.arange(0.0, 10.0, 0.01)
    x = np.full_like(t, 9.81 * 0.6)
    y = np.full_like(t, 9.81 * 0.8)
    x[500:] += 3.0
    y[500:] += 4.0
    t_out, jerk, _ = accel_jerk({"ACCEL_X_1": t, "ACCEL_Y_1": t},
                                {"ACCEL_X_1": x, "ACCEL_Y_1": y})

    assert len(jerk) == len(t_out) == len(t)
    assert jerk[500] == pytest.approx(500.0)
    assert np.delete(jerk, 500) == pytest.approx(0.0)


def test_each_sensor_is_there_only_if_the_recording_has_it():
    t = np.arange(0.0, 1.0, 0.1)
    assert gyro_speed({"ACCEL_X_1": t}, {"ACCEL_X_1": t}) is None
    assert set(imu_traces({"ACCEL_X_1": t}, {"ACCEL_X_1": t})) == {"accel"}
    assert imu_traces({"TEMP_1": t}, {"TEMP_1": t}) == {}


def test_the_table_reads_the_same_as_the_snirf_dicts():
    t = np.arange(0.0, 5.0, 0.01)
    rng = np.random.default_rng(0)
    cols = {f"{s}_{a}_1": rng.standard_normal(t.size) for s in ("GYRO", "ACCEL") for a in "XYZ"}
    table = pd.DataFrame({TIME_COLUMN: t, **cols})

    from_table = imu_traces(*table_channels(table))
    from_dicts = imu_traces({c: t for c in cols}, cols)

    assert set(from_table) == {"gyro", "accel"}
    for sensor in from_table:
        assert np.allclose(from_table[sensor][1], from_dicts[sensor][1])


def test_the_subject_report_finds_the_table_beside_the_stages(tmp_path):
    """The report has only the run's stage files; the aux table sits beside them under the
    same entities, and a run without one draws no row rather than failing."""
    import gzip
    from fnirs_pipe.qc.subject.report import _load_imu

    (tmp_path / "sub-01_task-rest_desc-sci_nirs.snirf").touch()
    t = np.arange(0.0, 2.0, 0.01)
    table = pd.DataFrame({TIME_COLUMN: t, "GYRO_X_1": np.where(t > 1, 2.0, 0.0),
                          "ACCEL_X_1": np.full_like(t, 9.81)})
    with gzip.open(tmp_path / "sub-01_task-rest_desc-aux_timeseries.tsv.gz", "wt") as fh:
        table.to_csv(fh, sep="\t", index=False)
    (tmp_path / "sub-01_task-rest_desc-aux_timeseries.json").write_text(
        '{"Units": {"GYRO_X_1": "o/s", "ACCEL_X_1": "m/s^2"}}', encoding="utf-8")

    errors: list = []
    imu = _load_imu(tmp_path, "sub-01_task-rest", "01", errors)
    assert errors == []
    assert set(imu) == {"gyro", "accel"}
    assert imu["gyro"][1].max() == pytest.approx(2.0)
    assert (imu["gyro"].unit, imu["accel"].unit) == ("°/s", "m/s³")

    (tmp_path / "sub-02_task-rest_desc-sci_nirs.snirf").touch()
    assert _load_imu(tmp_path, "sub-02_task-rest", "02", errors) is None
    assert errors == []


def test_each_sensor_carries_the_unit_its_axis_prints():
    """The vendor's spelling is the one an axis prints: NIRx writes o/s and m/s^2. Jerk is
    the acceleration per second; axes that disagree, or record none, print nothing."""
    t = np.arange(0.0, 1.0, 0.1)
    names = ["GYRO_X_1", "GYRO_Y_1", "ACCEL_X_1", "ACCEL_Y_1"]
    times, values = {n: t for n in names}, {n: np.sin(t) for n in names}

    traces = imu_traces(times, values, {"GYRO_X_1": "o/s", "GYRO_Y_1": "o/s",
                                        "ACCEL_X_1": "m/s^2", "ACCEL_Y_1": "m/s^2"})
    assert (traces["gyro"].unit, traces["accel"].unit) == ("°/s", "m/s³")

    mixed = imu_traces(times, values, {"GYRO_X_1": "o/s", "GYRO_Y_1": "rad/s"})
    assert (mixed["gyro"].unit, mixed["accel"].unit) == ("", "")


def test_the_table_units_come_from_its_sidecar(tmp_path):
    import json
    from fnirs_pipe.io.auxiliary import aux_table_units

    table = tmp_path / "sub-01_task-rest_desc-aux_timeseries.tsv.gz"
    table.touch()
    (tmp_path / "sub-01_task-rest_desc-aux_timeseries.json").write_text(
        json.dumps({"Units": {"GYRO_X_1": "o/s"}}), encoding="utf-8")

    assert aux_table_units(table) == {"GYRO_X_1": "o/s"}
    assert aux_table_units(tmp_path / "missing_desc-aux_timeseries.tsv.gz") == {}
