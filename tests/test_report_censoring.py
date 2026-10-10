"""Report figures read kept frames only, and draw what they drew before on a run without spans."""

from datetime import datetime, timezone

import mne
import numpy as np
from mne.time_frequency import psd_array_welch

from nirspipe.qc.figures.common._utils import PSD_NFFT, line_xy
from nirspipe.qc.figures.subject.raw_figures import build_channel_figure
from nirspipe.qc.subject.report import _segments_in_window

SFREQ = 10.0


def _pair(spans=(), garbage=0.0):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((2, int(SFREQ * 300))) * 1e-6
    info = mne.create_info(["S1_D1 hbo", "S1_D1 hbr"], SFREQ, ["hbo", "hbr"])
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations([s[0] for s in spans], [s[1] for s in spans],
                                        ["BAD_gvtd"] * len(spans)))
    for onset, duration in spans:
        raw._data[:, int(onset * SFREQ):int((onset + duration) * SFREQ)] += garbage
    return raw


def _psd_trace(raw):
    _, psd_fig, _ = build_channel_figure(raw, [], "S1_D1", max_ts_pts=4000)
    return None if psd_fig is None else np.asarray(psd_fig.data[0].y)


def test_the_channel_spectrum_is_unchanged_without_a_span():
    raw = _pair()
    psds, freqs = psd_array_welch(raw.get_data(), SFREQ, n_fft=PSD_NFFT, verbose=False)
    mask = freqs <= 2.0
    np.testing.assert_array_equal(_psd_trace(raw), line_xy(freqs[mask], psds[0][mask])["y"])


def test_the_channel_spectrum_leaves_out_bad_frames():
    quiet, loud = _pair([(100.0, 60.0)]), _pair([(100.0, 60.0)], garbage=1e-3)
    np.testing.assert_allclose(_psd_trace(loud), _psd_trace(quiet), rtol=1e-12)
    every_frame, _ = psd_array_welch(loud.get_data(), SFREQ, n_fft=PSD_NFFT, verbose=False)
    assert not np.allclose(_psd_trace(loud), every_frame[0][:len(_psd_trace(loud))])


def test_a_stretch_wholly_under_a_span_draws_no_spectrum():
    assert _psd_trace(_pair([(0.0, 300.0)])) is None


def test_the_bad_segment_zoom_draws_corrupted_spans_only():
    segments = {"BAD_gvtd": [(10.0, 5.0)], "BAD_unselected": [(100.0, 200.0)],
                "BAD_manual": [(50.0, 2.0)]}
    assert _segments_in_window(segments, None) == [(10.0, 5.0), (50.0, 2.0)]
