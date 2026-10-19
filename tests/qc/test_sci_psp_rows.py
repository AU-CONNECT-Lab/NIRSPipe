"""The SCI/PSP strip: a row per pair for the pair measures, a row per wavelength for CV, all named."""

import numpy as np

from nirspipe.qc.figures.subject.sci_psp_panel import build_sci_psp_figure

NAMES = [f"S{p}_D{p} {wl}" for p in (1, 2, 3) for wl in (760, 850)]
PAIRS = ["S1_D1", "S2_D2", "S3_D3"]
TIMES = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [30.0, 40.0]])
# one number per pair, written into both wavelength rows as mne-nirs writes it
SCI = np.repeat([[0.90, 0.85, 0.95, 0.70],
                 [0.30, 0.20, 0.40, 0.10],
                 [0.99, 0.90, 0.90, 0.90]], 2, axis=0)
PSP = np.repeat([[0.20, 0.30, 0.40, 0.50],
                 [0.02, 0.01, 0.03, 0.04],
                 [0.60, 0.50, 0.40, 0.30]], 2, axis=0)
# its own number on each wavelength
CV = np.arange(24, dtype=float).reshape(6, 4) / 200 + 0.01


def _strip(sci=SCI, bad=(), warn=None):
    return build_sci_psp_figure(
        {n: 0.9 for n in NAMES}, {n: 0.4 for n in NAMES}, set(bad), sci_threshold=0.8,
        sci_matrix=sci, sci_win_times=TIMES, psp_matrix=PSP, psp_win_times=TIMES,
        cv_matrix=CV, cv_win_times=TIMES, window_s=10.0, warn=warn)


def _heat(fig, name):
    return next(t for t in fig.data if t.type == "heatmap" and t.name == name)


def test_sci_and_psp_get_a_row_per_pair_and_cv_a_row_per_wavelength():
    warn = []
    fig = _strip(warn=warn)
    assert list(_heat(fig, "SCI").y) == PAIRS
    np.testing.assert_array_equal(np.asarray(_heat(fig, "SCI").z), SCI[::2])
    assert list(_heat(fig, "PSP").y) == PAIRS
    np.testing.assert_array_equal(np.asarray(_heat(fig, "PSP").z), PSP[::2])
    assert list(_heat(fig, "CV").y) == NAMES
    np.testing.assert_array_equal(np.asarray(_heat(fig, "CV").z), CV)
    assert warn == []


def test_every_row_is_named():
    fig = _strip()
    # the heatmaps sit in the left column: axes 1, 3 and 5 of a three-by-two grid
    named = [list(axis.tickvals) for axis in (fig.layout.yaxis, fig.layout.yaxis3,
                                              fig.layout.yaxis5)]
    assert named == [PAIRS, PAIRS, NAMES]


def test_a_rejected_pair_is_red_in_the_sci_row_alone():
    fig = _strip(bad={"S3_D3 760", "S3_D3 850"})
    sci_dots, psp_dots = [t for t in fig.data if t.type == "scatter" and t.mode == "markers"][:2]
    # S3_D3's SCI mean is 0.92, over the line: red only because screening rejected it
    assert dict(zip(sci_dots.y, sci_dots.marker.color))["S3_D3"] == "#e74c3c"
    assert dict(zip(psp_dots.y, psp_dots.marker.color))["S3_D3"] == "#27ae60"


def test_a_pair_whose_wavelengths_disagree_is_drawn_twice_and_said_so():
    sci = SCI.copy()
    sci[3, 2] = 0.95                                  # S2_D2 850 alone, in one window
    warn = []
    fig = _strip(sci=sci, warn=warn)
    assert list(_heat(fig, "SCI").y) == ["S1_D1", "S2_D2 760", "S2_D2 850", "S3_D3"]
    np.testing.assert_array_equal(np.asarray(_heat(fig, "SCI").z)[1:3], sci[2:4])
    assert len(warn) == 1 and "SCI" in warn[0] and "S2_D2" in warn[0]
    assert list(_heat(fig, "PSP").y) == PAIRS
