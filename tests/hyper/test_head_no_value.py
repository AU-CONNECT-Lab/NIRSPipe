"""A channel with no SCI in a window is drawn white on the dyad's head maps, never black."""

import numpy as np

from nirspipe.qc.figures.common.head_map import head_geometry
from nirspipe.qc.figures.hyper.hyper_figures import build_head_slider
from nirspipe.utils import pair_of
from tests._synth import synth_raw


def test_the_slider_draws_a_channel_with_no_value_white_in_every_frame():
    raw = synth_raw("01", "tapping")
    pairs = list(dict.fromkeys(pair_of(ch) for ch in raw.ch_names))
    geo = head_geometry(raw, pairs)
    t = np.array([5.0, 15.0, 25.0])
    sci = {p: np.array([0.9, 0.8, 0.95]) for p in pairs}
    gone = geo["long"]["names"][0]
    sci[gone] = np.array([np.nan, 0.9, np.nan])        # saturated in the first and last window
    fig = build_head_slider({"A": geo}, ["A"], {"t": t, "pairs": pairs},
                            {"A": {"per_pair_sci": sci}}, step=1)

    white = [i for i, tr in enumerate(fig.data)
             if isinstance(getattr(tr.marker, "color", None), str) and tr.marker.color == "#ffffff"]
    assert white, "no white twin drawn"
    coloured = white[0] - 1
    first = list(geo["long"]["names"]).index(gone)
    reach = int(np.sum(geo["long"]["per_pair"][:first]))
    assert fig.data[coloured].marker.size[reach] == 0, "the coloured marker hides"
    assert fig.data[white[0]].marker.size[reach] > 0, "its white twin shows"
    # the middle frame has a value there, so the twin hides and the colour shows
    middle = fig.frames[1].data
    assert middle[0].marker.size[reach] > 0 and middle[1].marker.size[reach] == 0
