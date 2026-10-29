"""The cohort box charts' slanted x labels run left from their ticks; the left margin has to
hold the part that reaches past the plot's edge, or the browser drops the first label."""

import math

import pandas as pd
import pytest

from nirspipe.qc.figures.subject.group_figures import _group_box

# a lower bound for Plotly's 12 px tick font, measured off a rendered chart
_MIN_CHAR_PX = 6.5


def _box(cols):
    df = pd.DataFrame({c: [0.9, 0.8, 0.7] for c in cols})
    return _group_box("Coupling", cols, df, ["01", "02", "03"])


@pytest.mark.parametrize("cols", [
    ["motion_post_long_sci_win_mean", "raw_long_sci_mean", "raw_short_sci_mean"],
    ["raw_long_sci_mean", "raw_long_cp_mean"],
])
def test_every_slanted_label_fits_left_of_its_tick(cols):
    fig = _box(cols)
    lo, hi = fig.layout.xaxis.range
    plot_px = fig.layout.width - fig.layout.margin.l - fig.layout.margin.r
    for tick, label in zip(fig.layout.xaxis.tickvals, fig.layout.xaxis.ticktext):
        tick_px = fig.layout.margin.l + (tick - lo) / (hi - lo) * plot_px
        assert tick_px >= len(label) * _MIN_CHAR_PX * math.cos(math.radians(30)), label


def test_the_margin_grows_without_narrowing_the_plot():
    short = _box(["raw_long_sci_mean", "raw_long_cp_mean", "raw_long_snr_mean"])
    long = _box(["motion_post_long_sci_win_mean", "raw_long_cp_mean", "raw_long_snr_mean"])
    assert long.layout.margin.l > short.layout.margin.l
    assert (long.layout.width - long.layout.margin.l) == (short.layout.width - short.layout.margin.l)
