"""Which group-table columns the cohort report charts, and under which heading.

A column reaches a chart only through its bare metric name, so a section prefix the parser
does not know leaves the metric unrecognised and drops it into "Other", settings included.
"""

from fnirs_pipe.qc.figures.subject.group_figures import _split_column, group_metrics


def test_an_optional_section_is_split_off_like_any_other():
    assert _split_column("censor_gvtd_censor_pct") == ("censor", "all", "gvtd_censor_pct")
    assert _split_column("imu_gyro_speed_mean") == ("imu", "all", "gyro_speed_mean")


def test_imu_columns_get_their_own_chart_and_the_check_only_ones_none():
    groups, ordered = group_metrics([
        "imu_gyro_speed_mean", "imu_accel_jerk_p95", "imu_gyro_speed_gvtd_rho",
        "censor_gvtd_censor_pct", "censor_gvtd_censor_n_std",
    ])
    charts = dict(groups)
    assert charts["Motion sensor: gyroscope speed"] == ["imu_gyro_speed_mean"]
    assert charts["Motion sensor: accelerometer jerk"] == ["imu_accel_jerk_p95"]
    assert "censor_gvtd_censor_pct" in charts["Motion & spike fraction"]
    assert "Other" not in charts
    assert "imu_gyro_speed_gvtd_rho" not in ordered
    assert "censor_gvtd_censor_n_std" not in ordered
