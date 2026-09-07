"""Windowed SCI / PSP / GVTD: one time axis, and a PSP scalar that a display setting cannot move.

The three series are stored side by side in one SQM record and drawn on one axis, but only
SCI and PSP get their windows from mne_nirs; GVTD's are ours. These tests pin the seam.

A non-integer sampling rate is the point of `od_raw`. At 10 Hz a 10 s window is exactly 100
samples however you round, so every grid disagreement here is invisible; at 7.8125 Hz it is
78.125 samples and the rounding difference accumulates over the recording.
"""

import numpy as np
import pytest

from fnirs_pipe.qc.metrics import (
    PSP_WINDOW_S,
    attach_windowed_series,
    compute_windowed_gvtd,
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


# ---- F2: the GVTD grid is the SCI/PSP grid ----
def test_gvtd_windows_land_on_the_sci_window_grid(od_raw):
    sci_centers = _sci_window_centers(od_raw)
    _, _, gvtd_centers = compute_windowed_gvtd(od_raw, WINDOW_S)

    assert len(gvtd_centers) == len(sci_centers)
    assert np.allclose(gvtd_centers, sci_centers)


def test_gvtd_centers_are_read_off_real_sample_times(od_raw):
    """Regression pin: centres derived from the nominal window length drift, and this is by how much."""
    _, _, gvtd_centers = compute_windowed_gvtd(od_raw, WINDOW_S)
    nominal = np.arange(len(gvtd_centers)) * WINDOW_S + WINDOW_S / 2

    assert abs(gvtd_centers[-1] - nominal[-1]) > 1.0


# ---- F3: psp_mean is pinned, because the score scales with the window ----
def test_psp_scales_with_window_length_and_sci_does_not(od_raw):
    from mne_nirs.preprocessing import peak_power

    def psp_at(window_s):
        _, scores, _ = peak_power(od_raw.copy(), time_window=window_s,
                                  l_freq=CARDIAC[0], h_freq=CARDIAC[1], verbose=False)
        return float(np.mean(scores))

    assert psp_at(20.0) / psp_at(10.0) == pytest.approx(2.0, rel=0.15)

    def sci_at(window_s):
        scores, _ = compute_windowed_sci(od_raw, *CARDIAC, window_s)
        return float(np.mean(scores))

    assert sci_at(20.0) == pytest.approx(sci_at(10.0), abs=0.05)


def test_psp_mean_uses_the_pinned_window_not_the_library_default(od_raw, monkeypatch):
    import mne_nirs.preprocessing as nirs_prep

    from fnirs_pipe.qc import metrics as qm

    seen = {}
    real = nirs_prep.peak_power

    def spy(raw, **kwargs):
        seen.update(kwargs)
        return real(raw, **kwargs)

    monkeypatch.setattr(nirs_prep, "peak_power", spy)
    qm._psp_metrics(od_raw.copy(), *CARDIAC)

    assert seen["time_window"] == PSP_WINDOW_S


# ---- F4: a cardiac band the filter rejects must not take GVTD down with it ----
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


def test_an_older_database_gains_the_columns_it_is_missing(tmp_path):
    import sqlite3

    from fnirs_pipe.utils import job_db

    db = tmp_path / "runs.db"
    # the schema as the previous version left it: everything but the newest column
    previous = job_db._SCHEMA.replace("    qc_window_s                 REAL,\n", "")
    assert previous != job_db._SCHEMA
    with sqlite3.connect(db) as old:
        old.executescript(previous)

    conn = job_db._get_conn(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sqm)")}
    conn.close()

    assert set(job_db._SQM_COLS) <= cols


# ---- F6: the heatmap gets one x value per column ----
def test_window_centers_collapses_start_end_pairs():
    from fnirs_pipe.qc.figures.raw_figures import _window_centers

    pairs = [(0.0, 10.1), (10.1, 20.2), (20.2, 30.3)]
    assert np.allclose(_window_centers(pairs), [5.05, 15.15, 25.25])
    assert np.allclose(_window_centers([5.05, 15.15]), [5.05, 15.15])


def test_our_heatmap_has_one_column_per_window_like_mne_nirs_own_figure(od_raw):
    """The cross-check proper: mne_nirs plots the same scores itself, so the two can be compared.

    It labels each column by the window start; we use the centre. What must match is the
    column count, which is what the flattening bug got wrong.
    """
    import matplotlib.pyplot as plt
    from mne_nirs.visualisation import plot_timechannel_quality_metric

    from fnirs_pipe.qc.figures.raw_figures import build_sci_psp_figure

    plt.switch_backend("Agg")
    sci_matrix, sci_times = compute_windowed_sci(od_raw, *CARDIAC, WINDOW_S)

    ref = plot_timechannel_quality_metric(od_raw, sci_matrix, sci_times)
    ref_starts = np.array([float(t.get_text()) for t in ref.axes[0].get_xticklabels()])
    plt.close(ref)

    fig = build_sci_psp_figure(
        sci_scores={ch: 0.9 for ch in od_raw.ch_names},
        psp_per_channel={ch: 0.5 for ch in od_raw.ch_names},
        bad_channels=set(),
        sci_matrix=sci_matrix, sci_win_times=sci_times,
        psp_matrix=sci_matrix, psp_win_times=sci_times,
    )
    our_x = np.asarray(next(t for t in fig.data if t.type == "heatmap").x, dtype=float)

    assert len(our_x) == len(ref_starts)
    assert np.allclose(our_x - ref_starts, (our_x[1] - our_x[0]) / 2, atol=1.0)


def test_sci_heatmap_time_axis_spans_the_recording(od_raw):
    from fnirs_pipe.qc.figures.raw_figures import build_sci_psp_figure

    sci_matrix, sci_times = compute_windowed_sci(od_raw, *CARDIAC, WINDOW_S)
    fig = build_sci_psp_figure(
        sci_scores={ch: 0.9 for ch in od_raw.ch_names},
        psp_per_channel={ch: 0.5 for ch in od_raw.ch_names},
        bad_channels=set(),
        sci_matrix=sci_matrix, sci_win_times=sci_times,
        psp_matrix=sci_matrix, psp_win_times=sci_times,
    )
    heatmap = next(t for t in fig.data if t.type == "heatmap")

    assert len(heatmap.x) == sci_matrix.shape[1]
    assert max(heatmap.x) > 0.9 * od_raw.times[-1]
