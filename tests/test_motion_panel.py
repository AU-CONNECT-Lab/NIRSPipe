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

from fnirs_pipe.qc.figures.motion_panel import (
    _SPIKE_LABEL, _maxpool_xy, build_motion_detail_figure, carpet_gvtd_figure,
)
from fnirs_pipe.qc.quantitative_metrics import _mask_to_segments
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
    """One GVTD panel, the band-limited one. The unfiltered trace was dropped: it is the
    pipeline stage Sherafati 2020 measured the worst artifact-to-background ratio at."""
    raw = synth_raw("01", "tapping", duration=200.0)
    fig = carpet_gvtd_figure(raw, raw.ch_names[:6])

    labels = _panel_labels(fig)
    assert len([t for t in labels if "GVTD" in t]) == 1
    # the band and the channel set are printed beside the panel, not inside its axis title
    assert any("0.01" in t for t in labels)
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
    instead gave a zero-width span, which no renderer draws, so a spike on the last sample of
    the recording was silently missing from the figure."""
    times = np.arange(5, dtype=float)                  # 1 Hz
    flagged = np.array([False, True, True, False, True])

    assert _mask_to_segments(flagged, times) == [(1.0, 2.0), (4.0, 1.0)]


def test_maxpool_keeps_the_time_the_peak_happened():
    t = np.arange(8, dtype=float)
    y = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    t_ds, y_ds = _maxpool_xy(t, y, max_pts=4)

    assert y_ds.max() == 5.0
    assert t_ds[y_ds.argmax()] == 2.0
