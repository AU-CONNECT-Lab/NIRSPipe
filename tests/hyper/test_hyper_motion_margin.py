"""The dyad motion panel's row names sit in the left margin; the margin has to hold the widest
one, or the browser cuts its first letters off."""

import re

import pytest

from nirspipe.qc.figures.hyper.hyper_figures import build_motion_panel, motion_series
from tests._synth import synth_raw

# a lower bound for Plotly's 10 px Open Sans, bold or not, measured off a rendered page
_MIN_CHAR_PX = 5.5


def _panel(sids):
    raws = {sid: synth_raw(sid.removeprefix("sub-"), "tapping", duration=120.0) for sid in sids}
    return build_motion_panel(motion_series(raws, None, sids), "before")


def _widest_line(text):
    return max(len(re.sub(r"<[^>]+>", "", line)) for line in text.split("<br>"))


@pytest.mark.parametrize("sids", [["sub-01", "sub-02"], ["sub-pilot2024site01", "sub-02"]])
def test_every_row_name_fits_in_the_left_margin(sids):
    fig = _panel(sids)
    names = [a for a in fig.layout.annotations if a.xref == "paper" and a.x == 0
             and a.xanchor == "right"]
    assert names
    for name in names:
        room = fig.layout.margin.l - abs(name.xshift or 0)
        assert room >= _MIN_CHAR_PX * _widest_line(name.text), name.text


def test_a_longer_member_label_widens_the_margin():
    assert _panel(["sub-pilot2024site01", "sub-02"]).layout.margin.l \
        > _panel(["sub-01", "sub-02"]).layout.margin.l
