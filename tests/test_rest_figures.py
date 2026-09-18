"""The two rest-mode figures whose numbers already existed with nothing drawing them.

`_fcroi.tsv` had been written since ROI FC landed and no panel showed it; ALFF had a bar
chart ordered by channel name, which cannot be read as the spatial claim it is. Both are
Plotly figures, so what is asserted is the geometry that carries the meaning and the
refusals: no frames and no optode positions must give None rather than an exception, because
the report treats None as "skip the panel" and an exception as a broken report.

The seed map's own route is pinned in test_rest_seed_figure.
"""

import numpy as np
import pandas as pd
import plotly.colors as pc
import pytest

from fnirs_pipe.qc.figures.common.matrix_map import BLANK_CELL, CORRELATION_SCALE
from fnirs_pipe.qc.figures.subject.rest_figures import (
    alff_topo_figure, fc_matrix_figure, fc_roi_matrix_figure,
)


def _panels(fig) -> list[str]:
    """The subplot titles, one per panel."""
    return [a.text for a in fig.layout.annotations if a.text]


def _roi_frame(labels):
    rng = np.random.default_rng(0)
    m = rng.uniform(-1, 1, (len(labels), len(labels)))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 1.0)
    return pd.DataFrame(m, index=labels, columns=labels)


# ---- ROI to ROI ----

def test_the_roi_matrix_draws_both_chromophores():
    frame = _roi_frame(["PFC", "TPJ", "M1"])
    assert _panels(fc_roi_matrix_figure({"hbo": frame})) == ["ROI FC - HbO"]
    assert _panels(fc_roi_matrix_figure({"hbo": frame, "hbr": frame})) ==            ["ROI FC - HbO", "ROI FC - HbR"]


def test_the_roi_matrix_prints_every_cell():
    """A handful of ROIs means the numbers fit, and a matrix whose values all sit near one
    another renders as a flat square without them."""
    frame = _roi_frame(["PFC", "TPJ", "M1"])
    fig = fc_roi_matrix_figure({"hbo": frame})
    printed = [t for t in fig.data if getattr(t, "mode", None) == "text"]
    # the diagonal says nothing and is blanked, so three ROIs leave six cells
    assert sum(len(t.text) for t in printed) == 6


def test_the_roi_diagonal_is_blank():
    """An ROI's correlation with itself is 1 by construction."""
    fig = fc_roi_matrix_figure({"hbo": _roi_frame(["PFC", "TPJ"])})
    z = np.asarray(next(t for t in fig.data if t.type == "heatmap").z, dtype=float)
    assert np.isnan(np.diag(z)).all()


def test_the_roi_matrix_declines_rather_than_raises():
    assert fc_roi_matrix_figure({}) is None
    assert fc_roi_matrix_figure({"hbo": pd.DataFrame()}) is None


# ---- the channel matrix ----

def test_the_channel_matrix_splits_by_chromophore():
    labels = ["S1_D1 hbo", "S1_D2 hbo"], ["S1_D1 hbr", "S1_D2 hbr"]
    hbo, hbr = (_roi_frame(l) for l in labels)
    assert _panels(fc_matrix_figure(hbo)) == ["FC - HbO"]
    assert _panels(fc_matrix_figure(hbo, hbr)) == ["FC - HbO", "FC - HbR"]


def test_the_channel_matrix_keeps_a_rejected_channel_as_a_grey_row():
    """compute_fc blanks a rejected channel, and the row has to stay in the matrix: a
    dropped row would renumber the others and hide that anything was rejected."""
    frame = _roi_frame(["S1_D1 hbo", "S1_D2 hbo", "S2_D1 hbo"])
    frame.loc["S1_D2 hbo", :] = np.nan
    frame.loc[:, "S1_D2 hbo"] = np.nan
    fig = fc_matrix_figure(frame)
    heat = next(t for t in fig.data if t.type == "heatmap")
    assert list(heat.y) == list(frame.index)
    assert np.isnan(np.asarray(heat.z, dtype=float)[1]).all()
    # the grey is painted under the cells, so a blank reads as grey and not as the page
    assert any(sh.fillcolor == BLANK_CELL for sh in fig.layout.shapes)


def test_both_matrices_use_the_report_s_one_correlation_scale():
    """Three figures used to carry their own copy, so a red cell could have meant +1 in one
    panel and -1 in the next."""
    frame = _roi_frame(["S1_D1 hbo", "S1_D2 hbo"])
    for fig in (fc_matrix_figure(frame), fc_roi_matrix_figure({"hbo": _roi_frame(["A", "B"])})):
        heat = next(t for t in fig.data if t.type == "heatmap")
        assert [c for _, c in heat.colorscale] ==                [c for _, c in pc.get_colorscale(CORRELATION_SCALE)]
        assert (heat.zmin, heat.zmax) == (-1.0, 1.0)


# ---- ALFF on the layout ----

@pytest.fixture(scope="module")
def haemo():
    import mne

    from ._synth import synth_raw

    raw = synth_raw("01", "rest", duration=40.0, motion_onset=None)
    od = mne.preprocessing.nirs.optical_density(raw, verbose="error")
    return mne.preprocessing.nirs.beer_lambert_law(od, ppf=6.0)


@pytest.fixture
def alff_df(haemo):
    """The columns compute_alff writes, mALFF included: the figure draws that one."""
    rng = np.random.default_rng(1)
    alff = rng.uniform(0.1, 1.0, len(haemo.ch_names))
    return pd.DataFrame({
        "channel": haemo.ch_names,
        "alff": alff,
        "malff": alff / alff.mean(),
        "falff": rng.uniform(0.1, 0.9, len(haemo.ch_names)),
    })


def test_the_flat_map_is_drawn_from_the_montage(haemo, alff_df):
    assert alff_topo_figure(haemo, alff_df) is not None


def test_a_channel_is_one_disc_and_not_a_path(haemo, alff_df):
    """An amplitude is a property of a place. Drawn as a source-to-detector bar it chained
    into the neighbouring channels wherever they share an optode, and a montage of
    independent measurements read as one connected polyline."""
    from fnirs_pipe.qc.metrics import long_short_channels

    long_names, _ = long_short_channels(haemo, None)
    # short channels are off every map in this module, so the count is the long ones
    n_drawn = len({c.split(" ")[0] for c in long_names})
    fig = alff_topo_figure(haemo, alff_df)
    panel = [t for t in fig.data
             if t.marker.colorscale is not None and t.xaxis == fig.data[-1].xaxis]
    # the bar mark would put a string of markers on every channel instead of one disc
    assert sum(len(t.x) for t in panel) == n_drawn


def test_the_two_chromophores_of_a_row_share_one_bar(haemo, alff_df):
    """mALFF and fALFF are both dimensionless, so HbO and HbR can be read against each
    other; the four-bar version could compare nothing with anything."""
    fig = alff_topo_figure(haemo, alff_df)
    bars = [t for t in fig.data if getattr(t.marker, "showscale", False)]
    assert len(bars) == 2
    assert {b.marker.colorbar.title.text for b in bars} == {"mALFF", "fALFF"}


def test_the_row_scale_spans_both_chromophores(haemo, alff_df):
    fig = alff_topo_figure(haemo, alff_df)
    by_row = {}
    for t in fig.data:
        if t.marker.colorscale is not None and t.marker.cmin is not None:
            by_row.setdefault(t.yaxis, []).append((t.marker.cmin, t.marker.cmax))
    # two panels a row, and both ends of the scale identical across them
    for panels in by_row.values():
        assert len(set(panels)) == 1


def test_alff_stands_in_where_a_frame_carries_no_malff(haemo, alff_df):
    """Only a hand-built frame does; the figure still draws two rows rather than one."""
    fig = alff_topo_figure(haemo, alff_df.drop(columns=["malff"]))
    bars = [t for t in fig.data if getattr(t.marker, "showscale", False)]
    assert {b.marker.colorbar.title.text for b in bars} == {"ALFF", "fALFF"}


def test_the_hover_keeps_the_measured_amplitude(haemo, alff_df):
    """The colour is a multiple of the chromophore mean, so the molar value it came from
    has nowhere else to go."""
    fig = alff_topo_figure(haemo, alff_df)
    carried = [t for t in fig.data if t.customdata is not None]
    assert carried and all("customdata" in t.hovertemplate for t in carried)


def test_no_optode_positions_means_no_figure(haemo, alff_df):
    """The bar chart still works without coordinates; this panel cannot, and says so."""
    flat = haemo.copy()
    for ch in flat.info["chs"]:
        ch["loc"][:] = 0.0
    assert alff_topo_figure(flat, alff_df) is None


def _scales(fig) -> list[tuple]:
    """(cmin, cmax) of every trace that carries a colour scale, i.e. one per panel."""
    return [(t.marker.cmin, t.marker.cmax) for t in fig.data
            if getattr(t.marker, "cmin", None) is not None]


def test_a_rejected_channel_does_not_set_the_colour_scale(haemo, alff_df):
    """One dead channel with a runaway amplitude would otherwise flatten every real
    difference into one colour.

    Asked the direct way: does the spike move the scale? Kept, it must; rejected, it must
    not, which a figure that only faded the channel and went on reading its value would fail.
    Comparing a rejected run against a clean one instead would compare two different sets of
    channels, since rejecting one removes its fALFF from that panel too.
    """
    spiked = alff_df.copy()
    # the drawn column is what sets the scale, so that is the one to spike
    spiked.loc[spiked.index[0], "malff"] = 1e6
    outlier = str(spiked.iloc[0]["channel"])

    assert _scales(alff_topo_figure(haemo, spiked)) != _scales(alff_topo_figure(haemo, alff_df))

    marked = haemo.copy()
    marked.info["bads"] = [outlier]
    assert _scales(alff_topo_figure(marked, spiked)) == _scales(alff_topo_figure(marked, alff_df))
