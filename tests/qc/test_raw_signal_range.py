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
    assert lo == min(-1.5, ys.min()) and hi == max(3 * (len(fig.data) - 1) + 1.5, ys.max())
