"""The windowed SCI / PSP / CV strip: every colour bar is pinned, threshold at mid-scale."""

import numpy as np
import pytest

from fnirs_pipe.qc.figures.subject.sci_psp_panel import build_sci_psp_figure


def _figure(sci_threshold=0.8, psp_threshold=0.1, cv_threshold=0.05):
    chans = ["A", "B"]
    times = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]])
    # the windows that used to stretch an open scale: a negative SCI and a PSP far above its line
    sci = np.array([[0.99, -0.9, 0.95], [0.97, 0.98, 0.3]])
    psp = np.array([[3.2, 0.05, 1.1], [0.4, 2.7, 0.01]])
    cv = np.array([[0.01, 0.4, 0.02], [0.03, 0.02, 0.01]])
    return build_sci_psp_figure(
        {c: 0.9 for c in chans}, {c: 0.5 for c in chans}, set(),
        sci_threshold=sci_threshold, psp_threshold=psp_threshold, cv_threshold=cv_threshold,
        sci_matrix=sci, sci_win_times=times, psp_matrix=psp, psp_win_times=times,
        cv_matrix=cv, cv_win_times=times)


def _ranges(fig) -> dict:
    return {t.name: (t.zmin, t.zmax) for t in fig.data if t.type == "heatmap"}


def test_each_bar_stays_inside_what_its_metric_can_take():
    ranges = _ranges(_figure())
    assert ranges["SCI"] == pytest.approx((0.6, 1.0))
    assert ranges["PSP"] == pytest.approx((0.0, 0.2))
    assert ranges["CV"] == pytest.approx((0.0, 0.1))


def test_the_threshold_sits_mid_scale_whatever_it_is():
    ranges = _ranges(_figure(sci_threshold=0.7, psp_threshold=0.3, cv_threshold=0.08))
    for name, line in (("SCI", 0.7), ("PSP", 0.3), ("CV", 0.08)):
        lo, hi = ranges[name]
        assert (lo + hi) / 2 == pytest.approx(line)


def test_no_row_falls_back_to_an_open_scale():
    assert all(t.zmid is None for t in _figure().data if t.type == "heatmap")
