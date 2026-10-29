"""Windowed SCI / PSP / GVTD: one time axis, and a PSP scalar that a display setting cannot move.

The three series are stored side by side in one SQM record and drawn on one axis, but only
SCI gets its windows from mne_nirs; PSP copies that grid and GVTD's is ours. These tests pin
the seam.

A non-integer sampling rate is the point of `od_raw`. At 10 Hz a 10 s window is exactly 100
samples however you round, so every grid disagreement here is invisible; at 7.8125 Hz it is
78.125 samples and the rounding difference accumulates over the recording.
"""

import warnings

import numpy as np
import pytest

from nirspipe.qc.metrics import (
    PSP_WINDOW_S,
    attach_windowed_series,
    compute_windowed_gvtd,
    compute_windowed_psp,
    compute_windowed_sci,
)

SFREQ = 7.8125
WINDOW_S = 10.0
CARDIAC = (0.7, 1.5)


@pytest.fixture(scope="module")
def od_raw():
    import mne

    from tests._synth import synth_raw

    raw = synth_raw("01", "tapping", duration=400.0)
    raw.resample(SFREQ, verbose="error")
    return mne.preprocessing.nirs.optical_density(raw, verbose="error")


def _sci_window_centers(raw):
    _, times = compute_windowed_sci(raw, *CARDIAC, WINDOW_S)
    return np.asarray(times, dtype=float).mean(axis=1)


# ---- the GVTD grid is the SCI/PSP grid ----
def test_gvtd_windows_land_on_the_sci_window_grid(od_raw):
    sci_centers = _sci_window_centers(od_raw)
    _, _, gvtd_centers = compute_windowed_gvtd(od_raw, WINDOW_S)

    assert len(gvtd_centers) == len(sci_centers)
    assert np.allclose(gvtd_centers, sci_centers)


def test_gvtd_centers_are_read_off_real_sample_times(od_raw):
    """Centres derived from the nominal window length drift, and this is by how much."""
    _, _, gvtd_centers = compute_windowed_gvtd(od_raw, WINDOW_S)
    nominal = np.arange(len(gvtd_centers)) * WINDOW_S + WINDOW_S / 2

    assert abs(gvtd_centers[-1] - nominal[-1]) > 1.0


# ---- psp_mean is pinned, because the window moves the uncoupled pair, not the coupled ----
def test_the_window_moves_an_uncoupled_pair_s_psp_and_not_a_coupled_one_s(od_raw):
    from tests._synth import BAD_PAIR

    def psp_at(window_s):
        scores, _ = compute_windowed_psp(od_raw, *CARDIAC, window_s)
        return np.nanmedian(scores, axis=1)

    at_10, at_20 = psp_at(10.0), psp_at(20.0)
    uncoupled = [2 * BAD_PAIR, 2 * BAD_PAIR + 1]
    coupled = [i for i in range(len(at_10)) if i not in uncoupled]
    # a power spectrum of the same pulse holds; a density spectrum would double
    np.testing.assert_allclose(at_20[coupled], at_10[coupled], rtol=0.05)
    assert (at_20[uncoupled] < 0.7 * at_10[uncoupled]).all()

    def sci_at(window_s):
        scores, _ = compute_windowed_sci(od_raw, *CARDIAC, window_s)
        return float(np.mean(scores))

    assert sci_at(20.0) == pytest.approx(sci_at(10.0), abs=0.05)


def test_psp_mean_uses_the_pinned_window_not_the_library_default(od_raw, monkeypatch):
    from nirspipe.qc import metrics as qm
    from nirspipe.qc.metrics import coupling

    seen = {}
    real = coupling.peak_spectral_power

    def spy(raw, l_freq, h_freq, window_s):
        seen["window_s"] = window_s
        return real(raw, l_freq, h_freq, window_s)

    monkeypatch.setattr(coupling, "peak_spectral_power", spy)
    qm._psp_metrics(od_raw.copy(), *CARDIAC)

    assert seen["window_s"] == PSP_WINDOW_S


# ---- a cardiac band the filter rejects must not take GVTD down with it ----
def test_gvtd_series_survives_an_unusable_cardiac_band(od_raw):
    sqm: dict = {}
    above_nyquist = SFREQ / 2 + 1.0
    attach_windowed_series(sqm, od_raw, 0.7, above_nyquist, WINDOW_S)

    assert "sci_per_window" not in sqm
    assert len(sqm["gvtd_per_window"]) > 0
    assert len(sqm["gvtd_window_times_s"]) == len(sqm["gvtd_per_window"])


# ---- provenance: a record says which grid its series were binned on ----
def test_the_record_carries_the_window_the_series_were_binned_on(od_raw):
    sqm: dict = {}
    attach_windowed_series(sqm, od_raw, *CARDIAC, 20.0)

    assert sqm["qc_window_s"] == 20.0
    assert len(sqm["sci_per_window"]) == len(sqm["sci_window_times_s"])


def test_the_window_is_recorded_even_when_every_series_fails(od_raw):
    sqm: dict = {}
    attach_windowed_series(sqm, od_raw, 0.7, SFREQ / 2 + 1.0, 0.0)

    assert sqm["qc_window_s"] == 0.0
    assert "sci_per_window" not in sqm


# ---- the heatmap gets one x value per column ----
def test_window_centers_collapses_start_end_pairs():
    from nirspipe.qc.metrics.windowed import window_centers as _window_centers

    pairs = [(0.0, 10.1), (10.1, 20.2), (20.2, 30.3)]
    assert np.allclose(_window_centers(pairs), [5.05, 15.15, 25.25])
    assert np.allclose(_window_centers([5.05, 15.15]), [5.05, 15.15])


def test_our_heatmap_has_one_column_per_window_like_mne_nirs_own_figure(od_raw):
    """The cross-check proper: mne_nirs plots the same scores itself, so the two can be compared.

    It labels each column by the window start; we use the centre. What must match is the
    column count.
    """
    import matplotlib.pyplot as plt
    from mne_nirs.visualisation import plot_timechannel_quality_metric

    from nirspipe.qc.figures.subject.sci_psp_panel import build_sci_psp_figure

    plt.switch_backend("Agg")
    sci_matrix, sci_times = compute_windowed_sci(od_raw, *CARDIAC, WINDOW_S)

    ref = plot_timechannel_quality_metric(od_raw, sci_matrix, sci_times)
    ref_starts = np.array([float(t.get_text()) for t in ref.axes[0].get_xticklabels()])
    plt.close(ref)

    fig = build_sci_psp_figure(
        sci_scores={ch: 0.9 for ch in od_raw.ch_names},
        psp_per_channel={ch: 0.5 for ch in od_raw.ch_names},
        bad_channels=set(),
        sci_threshold=0.8,
        sci_matrix=sci_matrix, sci_win_times=sci_times,
        psp_matrix=sci_matrix, psp_win_times=sci_times,
    )
    our_x = np.asarray(next(t for t in fig.data if t.type == "heatmap").x, dtype=float)

    assert len(our_x) == len(ref_starts)
    assert np.allclose(our_x - ref_starts, (our_x[1] - our_x[0]) / 2, atol=1.0)


def test_sci_heatmap_time_axis_spans_the_recording(od_raw):
    from nirspipe.qc.figures.subject.sci_psp_panel import build_sci_psp_figure

    sci_matrix, sci_times = compute_windowed_sci(od_raw, *CARDIAC, WINDOW_S)
    fig = build_sci_psp_figure(
        sci_scores={ch: 0.9 for ch in od_raw.ch_names},
        psp_per_channel={ch: 0.5 for ch in od_raw.ch_names},
        bad_channels=set(),
        sci_threshold=0.8,
        sci_matrix=sci_matrix, sci_win_times=sci_times,
        psp_matrix=sci_matrix, psp_win_times=sci_times,
    )
    heatmap = next(t for t in fig.data if t.type == "heatmap")

    assert len(heatmap.x) == sci_matrix.shape[1]
    assert max(heatmap.x) > 0.9 * od_raw.times[-1]


def test_a_flat_channel_leaves_the_windowed_sci_without_a_warning(od_raw):
    from nirspipe.qc.metrics.coupling import _sci_win_metrics

    raw = od_raw.copy().apply_function(lambda x: x * 0, picks=[od_raw.ch_names[0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _sci_win_metrics(raw, *CARDIAC)
    assert [str(w.message) for w in caught if w.category is RuntimeWarning] == []
    assert od_raw.ch_names[0] not in out["sci_win_per_channel"]
    assert out["sci_win_per_channel"]


# ---- every channel under the plain key, the sets suffixed, as the record's sections ----
def test_the_plain_gvtd_series_is_every_channel_and_each_set_is_suffixed():
    import mne

    from nirspipe.qc.metrics import long_short_channels
    from tests._synth import synth_raw

    od = mne.preprocessing.nirs.optical_density(synth_raw("01", "rest", duration=200.0),
                                                verbose="error")
    sqm: dict = {}
    attach_windowed_series(sqm, od, *CARDIAC, WINDOW_S)
    long_names, short_names = long_short_channels(od)
    assert long_names and short_names
    for key, names in (("gvtd_per_window", od.ch_names), ("gvtd_per_window_long", long_names),
                       ("gvtd_per_window_short", short_names)):
        means, _, _ = compute_windowed_gvtd(od.copy().pick(names), WINDOW_S)
        np.testing.assert_array_equal(np.asarray(sqm[key]), np.asarray(means), err_msg=key)
    assert not [k for k in sqm if k.endswith("_all")]
