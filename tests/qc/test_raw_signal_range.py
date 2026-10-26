"""The raw signal overview's y range hugs its traces instead of Plotly's padded autorange."""

import numpy as np

from nirspipe.qc.figures import build_ts_figure
from tests._synth import synth_raw


def test_the_range_is_the_traces_extent():
    raw = synth_raw("01", "rest", seed=0)
    fig = build_ts_figure(raw, [], set(), 2000)[0]
    ys = np.concatenate([np.asarray(t.y, float) for t in fig.data])
    lo, hi = fig.layout.yaxis.range
    assert fig.layout.yaxis.autorange is False
    ticks = fig.layout.yaxis.tickvals
    half = (ticks[1] - ticks[0]) / 2
    assert lo == min(ticks[0] - half, ys.min()) and hi == max(ticks[-1] + half, ys.max())


def test_rows_sit_four_sds_apart_each_trace_centred_on_its_tick():
    raw = synth_raw("01", "rest", seed=0)
    fig = build_ts_figure(raw, [], set(), 2000)[0]
    ticks = np.asarray(fig.layout.yaxis.tickvals, float)
    assert np.allclose(np.diff(ticks), 4.0)
    for trace, tick in zip(fig.data, ticks):
        y = np.asarray(trace.y, float)
        # z-scored, so a trace's mean is its row and its spread one unit
        assert abs(np.mean(y) - tick) < 0.5


def test_a_pairs_wavelengths_sit_on_adjacent_rows_whatever_the_file_order():
    raw = synth_raw("01", "rest", seed=0)
    # one wavelength block before the other, as some files are written
    by_wavelength = sorted(raw.ch_names, key=lambda ch: (ch.split()[1], raw.ch_names.index(ch)))
    raw = raw.copy().reorder_channels(by_wavelength)
    names = [t.name for t in build_ts_figure(raw, [], set(), 2000)[0].data]
    pairs = [n.split()[0] for n in names]
    assert all(pairs[k] == pairs[k + 1] for k in range(0, len(pairs), 2))
    # pairs keep the order the file first lists them in
    assert list(dict.fromkeys(pairs)) == list(dict.fromkeys(ch.split()[0] for ch in by_wavelength))
