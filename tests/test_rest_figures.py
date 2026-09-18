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
    alff_topo_figure, fc_roi_matrix_figure, rest_channel_panel,
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


# ---- the channel panel ----

def _fc_frame(pairs, chromo):
    labels = [f"{p} {chromo}" for p in pairs]
    return _roi_frame(labels)


def test_a_matrix_is_drawn_as_a_triangle():
    """It is symmetric, so the upper half is the same values read the other way round;
    drawing both doubles the ink for no second reading."""
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    fig = rest_channel_panel(_fc_frame(pairs, "hbo"), None, None)
    z = np.asarray(next(t for t in fig.data if t.type == "heatmap").z, dtype=float)
    assert np.isnan(z[np.triu_indices(len(pairs))]).all()      # upper half and diagonal
    assert np.isfinite(z[np.tril_indices(len(pairs), -1)]).all()   # lower half kept


def test_the_panel_splits_the_matrices_by_chromophore():
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    hbo, hbr = _fc_frame(pairs, "hbo"), _fc_frame(pairs, "hbr")
    assert _panels(rest_channel_panel(hbo, hbr))[:2] == ["HbO", "HbR"]
    heat = [t for t in rest_channel_panel(hbo, hbr).data if t.type == "heatmap"]
    assert len(heat) == 2
    # the matrices are labelled by pair, the chromophore being the panel's own title
    assert list(heat[0].x) == pairs


def test_the_panel_rows_share_one_channel_order():
    """A channel has to be in the same relative place in every row, or the panel is three
    figures that happen to be stacked."""
    pairs = ["S2_D1", "S1_D1", "S1_D2"]
    alff = pd.DataFrame({"channel": [f"{p} {c}" for p in pairs for c in ("hbo", "hbr")],
                         "alff": np.arange(6.0), "falff": np.arange(6.0) / 10})
    fig = rest_channel_panel(_fc_frame(pairs, "hbo"), None, alff)
    heat = next(t for t in fig.data if t.type == "heatmap")
    strip = next(ax for name, ax in fig.layout.to_plotly_json().items()
                 if name.startswith("xaxis") and ax.get("ticktext"))
    assert list(heat.x) == list(strip["ticktext"])


def test_a_rejected_channel_sits_below_the_measured_range_not_on_zero():
    """A channel whose amplitude really is near zero belongs on the axis; if the two land on
    the same row a reader cannot tell a dead channel from a rejected one."""
    pairs = ["S1_D1", "S1_D2"]
    alff = pd.DataFrame({
        "channel": [f"{p} {c}" for p in pairs for c in ("hbo", "hbr")],
        "alff": [1.0, 1.2, np.nan, np.nan], "falff": [0.2, 0.2, np.nan, np.nan]})
    fig = rest_channel_panel(_fc_frame(pairs, "hbo"), None, alff)
    ring = next(t for t in fig.data if getattr(t, "name", None) == "rejected")
    assert all(y < 1.0 for y in ring.y)


def test_the_panel_is_the_matrices_alone_without_amplitudes():
    fig = rest_channel_panel(_fc_frame(["S1_D1", "S1_D2"], "hbo"), None, None)
    assert [t.type for t in fig.data].count("heatmap") == 1
    assert not any(getattr(t, "name", None) in ("HbO", "HbR") for t in fig.data)


def test_the_panel_declines_rather_than_raises():
    assert rest_channel_panel(None, None, None) is None
    assert rest_channel_panel(pd.DataFrame(), None, None) is None


def test_both_matrices_use_the_report_s_one_correlation_scale():
    """Three figures used to carry their own copy, so a red cell could have meant +1 in one
    panel and -1 in the next."""
    for fig in (rest_channel_panel(_fc_frame(["S1_D1", "S1_D2"], "hbo"), None, None),
                fc_roi_matrix_figure({"hbo": _roi_frame(["A", "B"])})):
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
