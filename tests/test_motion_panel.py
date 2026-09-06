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

from fnirs_pipe.qc.figures.motion_panel import build_motion_detail_figure, carpet_gvtd_figure
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


def test_carpet_draws_only_the_motion_band_gvtd(od_with_cardiac):
    """One GVTD panel, the band-limited one. The unfiltered trace was dropped: it is the
    pipeline stage Sherafati 2020 measured the worst artifact-to-background ratio at."""
    raw = synth_raw("01", "tapping", duration=200.0)
    fig = carpet_gvtd_figure(raw, raw.ch_names[:6])

    titles = [a.text for a in fig.layout.annotations if a.text]
    gvtd_titles = [t for t in titles if "GVTD" in t]
    assert len(gvtd_titles) == 1
    assert "0.01" in gvtd_titles[0]
