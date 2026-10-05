"""The motion figures show movement, not pulse, and not a downsampling artefact.

Two properties are easy to lose in a refactor and invisible in the rendered HTML, because a
wrong trace still looks like a plausible squiggle:

* the derivative rows are band-limited before differencing. Differencing is a high-pass, so
  on unfiltered OD the ~1 Hz cardiac component dominates head motion and the row reads as a
  pulse envelope.
* the derivative is taken at full resolution and *then* pooled. Differencing an already
  decimated trace aliases the cardiac band back in and scales the values by the decimation
  step, so the peaks move and their heights stop meaning anything.

The second is pinned by comparing a pooled figure against an unpooled one rather than by
recomputing the expected array, which would only restate the implementation.
"""

import mne
import numpy as np
import pytest

from fnirs_pipe.qc.figures.common.motion_panel import (
    IMU_SLOT, _GVTD_ROW_PX, _SPIKE_LABEL, _maxpool_xy, build_motion_detail_figure,
    carpet_gvtd_figure, carpet_z,
)
from fnirs_pipe.io.auxiliary import ImuTrace
from fnirs_pipe.qc.metrics import GVTD_MOTION_BAND, _mask_to_segments
from tests._synth import synth_raw

CARDIAC_HZ = 1.0
CARDIAC_AMP = 0.02


@pytest.fixture(scope="module")
def od_with_cardiac():
    """OD carrying a strong 1 Hz oscillation on every channel."""
    raw = synth_raw("01", "tapping", duration=400.0)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    pulse = CARDIAC_AMP * np.sin(2 * np.pi * CARDIAC_HZ * od.times)
    return mne.io.RawArray(od.get_data() + pulse, od.info, verbose="error")


def _derivative_row(fig):
    return np.asarray(next(t for t in fig.data if t.name == "|dOD/dt|").y, dtype=float)


def test_channel_derivative_row_is_not_the_cardiac_envelope(od_with_cardiac):
    ch = od_with_cardiac.ch_names[0]
    fig = build_motion_detail_figure(od_with_cardiac, od_with_cardiac, ch, max_pts=10 ** 6)

    unfiltered = np.abs(np.diff(od_with_cardiac.get_data(picks=[ch])[0]))
    # the row keeps peaks (it is max-pooled), so compare typical level, not the maximum
    assert np.median(_derivative_row(fig)) < 0.5 * np.median(unfiltered)


def test_channel_derivative_peak_survives_pooling(od_with_cardiac):
    """Pooling must not move the peak: max over bins of a full-res trace is the full-res max.

    Differencing after decimation instead would change the value, since the difference then
    spans `step` samples rather than one.
    """
    ch = od_with_cardiac.ch_names[0]
    unpooled = build_motion_detail_figure(od_with_cardiac, od_with_cardiac, ch, max_pts=10 ** 6)
    pooled = build_motion_detail_figure(od_with_cardiac, od_with_cardiac, ch, max_pts=300)

    assert len(_derivative_row(pooled)) < len(_derivative_row(unpooled))
    assert _derivative_row(pooled).max() == pytest.approx(_derivative_row(unpooled).max())


def _panel_labels(fig):
    """Every label the figure prints: subplot titles, annotations and axis titles."""
    labels = [a.text for a in fig.layout.annotations if a.text]
    labels += [axis["title"]["text"] for axis in fig.layout.to_plotly_json().values()
               if isinstance(axis, dict) and (axis.get("title") or {}).get("text")]
    return [str(t) for t in labels]


def test_carpet_draws_only_the_motion_band_gvtd(od_with_cardiac):
    """One GVTD panel, the band-limited one. There is no unfiltered trace: it is the
    pipeline stage with the worst artifact-to-background ratio."""
    raw = synth_raw("01", "tapping", duration=200.0)
    fig = carpet_gvtd_figure(raw, raw.ch_names[:6])

    labels = _panel_labels(fig)
    assert len([t for t in labels if "GVTD" in t]) == 1
    # the band and the channel set are printed beside the panel, not inside its axis title
    assert any(f"{GVTD_MOTION_BAND[0]:g}" in t for t in labels)
    assert any("ch" in t for t in labels)


# ---- the spans are drawn as given -------------------------------------------------------
SPIKE_SPANS = [(10.0, 2.0), (12.5, 0.1), (60.0, 0.0)]
CORRECTED_SPANS = [(30.0, 5.0)]


@pytest.fixture(scope="module")
def span_fig():
    raw = synth_raw("01", "tapping", duration=200.0)
    return carpet_gvtd_figure(raw, raw.ch_names[:6],
                              spike_segments=SPIKE_SPANS,
                              corrected_segments=CORRECTED_SPANS)


def _polygon(fig, name):
    return next(t for t in fig.data if t.name == name and t.fill == "toself")


def test_spike_shading_keeps_every_span_separate(span_fig):
    """Each span is its own rectangle: the gap between two spans is the claim that nothing
    happened there, so merging neighbours would print an artifact that was not detected."""
    xs = list(_polygon(span_fig, _SPIKE_LABEL).x)

    assert xs.count(None) == len(SPIKE_SPANS)          # one closed rectangle per span
    edges = [x for x in xs if x is not None]
    for i, (onset, duration) in enumerate(SPIKE_SPANS):
        assert edges[4 * i:4 * i + 4] == [onset, onset, onset + duration, onset + duration]


def test_correction_footprint_sits_above_the_gvtd_trace(span_fig):
    """The green bar is a strip of its own on the row over the trace, not a band below the
    carpet: it is read against the peaks the correction was aimed at."""
    corrected = _polygon(span_fig, "corrected")
    gvtd = next(t for t in span_fig.data if t.name in ("GVTD", "before"))

    assert corrected.yaxis == "y"        # the first row
    assert gvtd.yaxis == "y2"            # the GVTD panel directly under it
    assert list(corrected.x)[:4] == [30.0, 30.0, 35.0, 35.0]


def test_single_sample_span_is_one_sample_wide():
    """A run of one flagged sample covers one sample period. Closing it at its own timestamp
    would give a zero-width span, which no renderer draws, so a spike on the last sample of
    the recording would be silently missing from the figure."""
    times = np.arange(5, dtype=float)                  # 1 Hz
    flagged = np.array([False, True, True, False, True])

    assert _mask_to_segments(flagged, times) == [(1.0, 2.0), (4.0, 1.0)]


def test_maxpool_keeps_the_time_the_peak_happened():
    t = np.arange(8, dtype=float)
    y = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    t_ds, y_ds = _maxpool_xy(t, y, max_pts=4)

    assert y_ds.max() == 5.0
    assert t_ds[y_ds.argmax()] == 2.0


# ---- the carpet is detrended before it is scaled -----------------------------------------

def test_carpet_grey_is_the_fluctuation_not_the_drift():
    """A row that is a slow ramp plus a small oscillation reads as the oscillation. Scaled
    without detrending, the ramp sets the SD and the row is a left-to-right gradient."""
    t = np.linspace(0.0, 600.0, 6000)
    wobble = np.sin(2 * np.pi * 0.05 * t)
    data = np.vstack([10.0 * t / t[-1] + 0.1 * wobble])

    z, t_ds, _ = carpet_z(data, t)

    assert abs(np.corrcoef(z[0], t_ds)[0, 1]) < 0.1
    assert np.corrcoef(z[0], wobble[::len(t) // len(t_ds)][:z.shape[1]])[0, 1] > 0.99


def test_corrected_carpet_is_still_scaled_by_the_uncorrected_sd():
    """Detrending does not change the before/after contract: the corrected side is divided by
    the uncorrected SD, so a correction that halves the signal draws half as dark."""
    t = np.linspace(0.0, 300.0, 3000)
    rng = np.random.default_rng(0)
    before = rng.standard_normal((4, t.size)) + np.linspace(0.0, 5.0, t.size)
    after = 0.5 * before

    z_before, _, stats = carpet_z(before, t, z_threshold=100.0)
    z_after, _, _ = carpet_z(after, t, z_threshold=100.0, stats=stats)

    assert np.median(z_after.std(axis=1) / z_before.std(axis=1)) == pytest.approx(0.5, abs=0.01)


# ---- the IMU rows ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def imu_trace():
    t = np.arange(0.0, 200.0, 0.01)
    speed = np.abs(np.sin(2 * np.pi * 0.1 * t))
    speed[(t > 50) & (t < 51)] = 40.0
    jerk = np.abs(np.cos(2 * np.pi * 0.1 * t))
    jerk[(t > 120) & (t < 121)] = 90.0
    return {"gyro": ImuTrace(t, speed, "°/s"), "accel": ImuTrace(t, jerk, "m/s³")}


def _trace(fig, name):
    return next(t for t in fig.data if t.name == name)


def test_imu_rows_sit_under_the_strip_and_over_the_gvtd_rows(imu_trace):
    """The correction strip stays the top row; then the movement, gyroscope first, then the
    index computed from the data, and the data."""
    raw = synth_raw("01", "tapping", duration=200.0)
    fig = carpet_gvtd_figure(raw, raw.ch_names[:6], imu=imu_trace,
                             corrected_segments=CORRECTED_SPANS)

    assert _polygon(fig, "corrected").yaxis == "y"
    assert _trace(fig, "gyroscope").yaxis == "y2"
    assert _trace(fig, "accelerometer").yaxis == "y3"
    assert _trace(fig, "GVTD").yaxis == "y4"
    assert {a.name for a in fig.layout.annotations} >= {f"{IMU_SLOT}gyro", f"{IMU_SLOT}accel"}
    # each IMU row's axis names the unit its sensor recorded
    assert (fig.layout.yaxis2.title.text, fig.layout.yaxis3.title.text) == ("°/s", "m/s³")
    # the strip still sits directly on the row under it
    gap = fig.layout.yaxis.domain[0] - fig.layout.yaxis2.domain[1]
    assert 0 < gap < 0.01


def test_a_sensor_the_recording_lacks_gets_no_row(imu_trace):
    raw = synth_raw("01", "tapping", duration=200.0)
    without = carpet_gvtd_figure(raw, raw.ch_names[:6])
    gyro_only = carpet_gvtd_figure(raw, raw.ch_names[:6], imu={"gyro": imu_trace["gyro"]})

    assert {t.name for t in without.data}.isdisjoint({"gyroscope", "accelerometer"})
    assert "accelerometer" not in {t.name for t in gyro_only.data}
    # a row adds its own height rather than squeezing the others
    assert gyro_only.layout.height - without.layout.height >= _GVTD_ROW_PX


def test_imu_rows_keep_the_jolt_and_stop_at_the_recording(imu_trace, od_with_cardiac):
    """Max-pooled like the GVTD rows, so a one-second jolt survives the display cap, and
    cut to the optical recording's span. The per-channel figure carries both sensors."""
    ch = od_with_cardiac.ch_names[0]
    fig = build_motion_detail_figure(od_with_cardiac, od_with_cardiac, ch, imu=imu_trace)
    gyro, accel = _trace(fig, "gyroscope"), _trace(fig, "accelerometer")

    assert (gyro.yaxis, accel.yaxis) == ("y", "y2")
    assert np.max(gyro.y) == 40.0 and np.max(accel.y) == 90.0
    assert np.max(gyro.x) <= od_with_cardiac.times[-1]
