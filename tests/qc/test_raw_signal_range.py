"""The raw signal overview's y range hugs its traces instead of Plotly's padded autorange."""

import numpy as np

from nirspipe.qc.figures import build_ts_figure
from tests._synth import N_LONG_PAIRS, synth_raw

_LONG = {f"S{k}_D{k}" for k in range(1, N_LONG_PAIRS + 1)}


def test_the_range_is_the_traces_extent():
    raw = synth_raw("01", "rest", seed=0)
    fig = build_ts_figure(raw, [], set(), 2000)[0]
    ys = np.concatenate([np.asarray(t.y, float) for t in fig.data])
    lo, hi = fig.layout.yaxis.range
    assert fig.layout.yaxis.autorange is False
    ticks = fig.layout.yaxis.tickvals
    half = 2.0
    assert lo == min(min(ticks) - half, ys.min()) and hi == max(max(ticks) + half, ys.max())


def test_rows_sit_four_sds_apart_each_trace_centred_on_its_tick():
    raw = synth_raw("01", "rest", seed=0, short_channels=False)
    fig = build_ts_figure(raw, [], set(), 2000)[0]
    ticks = np.asarray(fig.layout.yaxis.tickvals, float)
    # read top-down: the first channel highest
    assert np.allclose(np.diff(ticks), -4.0)
    for trace, tick in zip(fig.data, ticks):
        y = np.asarray(trace.y, float)
        # z-scored, so a trace's mean is its row and its spread one unit
        assert abs(np.mean(y) - tick) < 0.5


def test_a_pairs_wavelengths_sit_on_adjacent_rows_whatever_the_file_order():
    raw = synth_raw("01", "rest", seed=0)
    # one wavelength block before the other, pairs last to first
    by_wavelength = sorted(raw.ch_names, key=lambda ch: (ch.split()[1], -raw.ch_names.index(ch)))
    raw = raw.copy().reorder_channels(by_wavelength)
    names = [t.name for t in build_ts_figure(raw, [], set(), 2000)[0].data]
    pairs = [n.split()[0] for n in names]
    assert all(pairs[k] == pairs[k + 1] for k in range(0, len(pairs), 2))
    # long pairs by source then detector number, whatever order the file lists them in
    long_pairs = [p for p in dict.fromkeys(pairs) if p in _LONG]
    assert long_pairs == [f"S{k}_D{k}" for k in range(1, N_LONG_PAIRS + 1)]


def test_short_channels_sit_in_their_own_labelled_block_below_the_long_ones():
    raw = synth_raw("01", "rest", seed=0)
    fig = build_ts_figure(raw, [], set(), 2000)[0]
    y_of = dict(zip((t.name for t in fig.data), fig.layout.yaxis.tickvals))
    long_y = [y for name, y in y_of.items() if name.split()[0] in _LONG]
    short_y = [y for name, y in y_of.items() if name.split()[0] not in _LONG]
    assert short_y and max(short_y) < min(long_y)
    # one blank row between the blocks, holding the label
    assert min(long_y) - max(short_y) == 8.0
    label = [a for a in fig.layout.annotations if a.text == "Short channels"]
    assert len(label) == 1 and max(short_y) < label[0].y < min(long_y)
    # no dashes left: the block says short, the colour stays the wavelength
    assert all(t.line.dash in (None, "solid") for t in fig.data)
