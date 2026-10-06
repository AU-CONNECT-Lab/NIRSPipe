"""Which group-table columns the cohort report charts, and under which heading.

A column reaches a chart only through its bare metric name, so a section prefix the parser
does not know leaves the metric unrecognised and drops it into "Other", settings included.
"""

import numpy as np

from fnirs_pipe.qc.figures.subject.group_figures import (_split_column, build_condition_panels,
                                                         build_window_grid, group_metrics)
from fnirs_pipe.qc.metrics import GVTD_MOTION_BAND


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


def test_the_over_time_grid_draws_the_motion_band_gvtd():
    """The unfiltered trace does not follow head movement; the grid draws the one that does."""
    times = [5.0 + 10.0 * i for i in range(30)]
    rows = [{"bids_name": f"sub-0{k}_task-rest", "gvtd_window_times_s": times,
             "gvtd_per_window": [1.0] * 30, "gvtd_filt_per_window": [2.0 + k] * 30}
            for k in range(3)]
    fig = build_window_grid(rows)
    drawn = {round(float(v), 6) for tr in fig.data if tr.y is not None for v in tr.y
             if v is not None and np.isfinite(v)}
    assert drawn and 1.0 not in drawn
    # the row says which trace it is, the band read off the constant
    assert f"{GVTD_MOTION_BAND[0]:g}-{GVTD_MOTION_BAND[1]:g} Hz" in fig.layout.yaxis.title.text


def test_the_condition_panels_draw_the_motion_band_gvtd():
    rows = [{"bids_name": f"sub-0{k}_task-full",
             "by_condition": {c: {"scalars": {"gvtd_mean": 1.0, "gvtd_filt_mean": 2.0 + k}}
                              for c in ("rest", "talk")}}
            for k in range(3)]
    fig = build_condition_panels(rows)
    drawn = {round(float(v), 6) for tr in fig.data if tr.y is not None for v in tr.y
             if v is not None and np.isfinite(v)}
    assert drawn and 1.0 not in drawn
