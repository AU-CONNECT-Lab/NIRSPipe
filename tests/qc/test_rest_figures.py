"""The rest-mode figures: the ROI FC matrix, the channel panel and mALFF on the cortex.

For the Plotly figures what is asserted is the geometry that carries the meaning and the
refusals: no frames must give None rather than an exception, because the report treats
None as "skip the panel" and an exception as a broken report. The brain figure is checked
through what it hands the projection, and once rendered for real.

The seed map's own route is pinned in test_rest_seed_figure.
"""

import numpy as np
import pandas as pd
import plotly.colors as pc
import pytest
from matplotlib import colormaps

from nirspipe.qc.figures.common.matrix_map import CORRELATION_SCALE
from nirspipe.qc.figures.subject import rest_figures
from nirspipe.qc.figures.subject.rest_figures import (
    alff_brain_figure, fc_roi_matrix_figure, rest_channel_panel,
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


def test_the_roi_matrix_draws_hbt_after_the_others():
    frame = _roi_frame(["PFC", "TPJ"])
    assert _panels(fc_roi_matrix_figure({"hbt": frame, "hbo": frame})) ==            ["ROI FC - HbO", "ROI FC - HbT"]


def test_the_roi_matrix_prints_every_cell():
    """A handful of ROIs means the numbers fit, and a matrix whose values all sit near one
    another renders as a flat square without them."""
    frame = _roi_frame(["PFC", "TPJ", "M1"])
    fig = fc_roi_matrix_figure({"hbo": frame})
    printed = [t for t in fig.data if getattr(t, "mode", None) == "text"]
    # one half of a symmetric matrix, diagonal blanked: three ROIs leave three cells
    assert sum(len(t.text) for t in printed) == 3


def test_the_roi_matrix_is_a_triangle_with_a_blank_diagonal():
    """Symmetric like the channel matrices, so the same half is drawn; and an ROI's
    correlation with itself is 1 by construction and says nothing."""
    fig = fc_roi_matrix_figure({"hbo": _roi_frame(["PFC", "TPJ", "M1"])})
    z = np.asarray(next(t for t in fig.data if t.type == "heatmap").z, dtype=float)
    assert np.isnan(z[np.triu_indices(3)]).all()
    assert np.isfinite(z[np.tril_indices(3, -1)]).all()


def test_the_roi_matrix_declines_rather_than_raises():
    assert fc_roi_matrix_figure({}) is None
    assert fc_roi_matrix_figure({"hbo": pd.DataFrame()}) is None


# ---- the channel panel ----

def _fc_frame(pairs, chromo):
    labels = [f"{p} {chromo}" for p in pairs]
    return _roi_frame(labels)


def test_a_matrix_is_drawn_whole_with_a_blank_diagonal():
    """Whole, so an ROI's block reads as a square; the diagonal is 1 by construction."""
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    fig = rest_channel_panel({"hbo": _fc_frame(pairs, "hbo")})
    z = np.asarray(next(t for t in fig.data if t.type == "heatmap").z, dtype=float)
    assert np.isnan(np.diag(z)).all()
    off = ~np.eye(len(pairs), dtype=bool)
    assert np.isfinite(z[off]).all()
    np.testing.assert_allclose(z, z.T, equal_nan=True)


def test_an_roi_map_orders_the_panel_and_marks_each_roi():
    pairs = ["S1_D1", "S1_D2", "S2_D1", "S3_D3"]
    roi_map = {"right": ["S2_D1", "S1_D2"], "left": ["S1_D1"]}
    alff = pd.DataFrame({"channel": [f"{p} hbo" for p in pairs],
                         "alff": np.arange(4.0), "falff": np.arange(4.0) / 10})
    fig = rest_channel_panel({"hbo": _fc_frame(pairs, "hbo")}, alff, roi_map=roi_map)
    heat = next(t for t in fig.data if t.type == "heatmap")
    # ROIs in name order, members by name inside, a pair no ROI claims last
    assert list(heat.x) == list(heat.y) == ["S1_D1", "S1_D2", "S2_D1", "S3_D3"]
    strip = next(ax for name, ax in fig.layout.to_plotly_json().items()
                 if name.startswith("xaxis") and ax.get("ticktext"))
    assert list(strip["ticktext"]) == list(heat.x)
    filled = [s for s in fig.layout.shapes if s.fillcolor and s.xref == "x"]
    # a strip along the top and one down the side for each of left, right and unassigned
    assert len(filled) == 6
    rules = [s for s in fig.layout.shapes if s.type == "line" and s.xref == "x"]
    # one vertical and one horizontal rule between each two neighbouring ROIs
    assert sorted({s.x0 for s in rules if s.x0 == s.x1}) == [0.5, 2.5]
    legend = {t.name for t in fig.data if getattr(t, "showlegend", None) is not False}
    assert {"left", "right", "unassigned"} <= legend


def test_the_panel_splits_the_matrices_by_chromophore():
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    hbo, hbr = _fc_frame(pairs, "hbo"), _fc_frame(pairs, "hbr")
    assert _panels(rest_channel_panel({"hbo": hbo, "hbr": hbr}))[:2] == ["HbO", "HbR"]
    heat = [t for t in rest_channel_panel({"hbo": hbo, "hbr": hbr}).data if t.type == "heatmap"]
    assert len(heat) == 2
    # the matrices are labelled by pair, the chromophore being the panel's own title
    assert list(heat[0].x) == pairs


def test_the_panel_draws_a_matrix_and_a_marker_per_chromophore():
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    chromas = ("hbo", "hbr", "hbt")
    alff = pd.DataFrame({"channel": [f"{p} {c}" for p in pairs for c in chromas],
                         "alff": np.arange(9.0), "falff": np.arange(9.0) / 10})
    fig = rest_channel_panel({c: _fc_frame(pairs, c) for c in chromas}, alff)
    assert _panels(fig)[:3] == ["HbO", "HbR", "HbT"]
    assert [t.type for t in fig.data].count("heatmap") == 3
    markers = {t.name for t in fig.data if getattr(t, "mode", None) == "markers"}
    assert {"HbO", "HbR", "HbT"} <= markers


def test_an_hbr_only_run_puts_its_matrix_first():
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    fig = rest_channel_panel({"hbr": _fc_frame(pairs, "hbr")})
    assert _panels(fig) == ["HbR"]


def test_the_panel_rows_share_one_channel_order():
    """A channel has to be in the same relative place in every row, or the panel is three
    figures that happen to be stacked."""
    pairs = ["S2_D1", "S1_D1", "S1_D2"]
    alff = pd.DataFrame({"channel": [f"{p} {c}" for p in pairs for c in ("hbo", "hbr")],
                         "alff": np.arange(6.0), "falff": np.arange(6.0) / 10})
    fig = rest_channel_panel({"hbo": _fc_frame(pairs, "hbo")}, alff)
    heat = next(t for t in fig.data if t.type == "heatmap")
    strip = next(ax for name, ax in fig.layout.to_plotly_json().items()
                 if name.startswith("xaxis") and ax.get("ticktext"))
    assert list(heat.x) == list(strip["ticktext"])


def test_the_panel_orders_channels_by_source_then_detector_number():
    """S2_D9 before S2_D10 and S9 before S10, which string order gets backwards."""
    pairs = ["S10_D1", "S2_D10", "S1_D1", "S2_D9", "S9_D2"]
    alff = pd.DataFrame({"channel": [f"{p} hbo" for p in pairs], "alff": np.arange(5.0),
                         "falff": np.arange(5.0) / 10})
    fig = rest_channel_panel({"hbo": _fc_frame(pairs, "hbo")}, alff)
    heat = next(t for t in fig.data if t.type == "heatmap")
    assert list(heat.x) == ["S1_D1", "S2_D9", "S2_D10", "S9_D2", "S10_D1"]


def test_a_rejected_channel_sits_below_the_measured_range_not_on_zero():
    """A channel whose amplitude really is near zero belongs on the axis; if the two land on
    the same row a reader cannot tell a dead channel from a rejected one."""
    pairs = ["S1_D1", "S1_D2"]
    alff = pd.DataFrame({
        "channel": [f"{p} {c}" for p in pairs for c in ("hbo", "hbr")],
        "alff": [1.0, 1.2, np.nan, np.nan], "falff": [0.2, 0.2, np.nan, np.nan]})
    fig = rest_channel_panel({"hbo": _fc_frame(pairs, "hbo")}, alff)
    ring = next(t for t in fig.data if getattr(t, "name", None) == "no value")
    assert all(y < 1.0 for y in ring.y)


def test_the_panel_is_the_matrices_alone_without_amplitudes():
    fig = rest_channel_panel({"hbo": _fc_frame(["S1_D1", "S1_D2"], "hbo")})
    assert [t.type for t in fig.data].count("heatmap") == 1
    assert not any(getattr(t, "name", None) in ("HbO", "HbR") for t in fig.data)


def test_the_panel_declines_rather_than_raises():
    assert rest_channel_panel({}) is None
    assert rest_channel_panel({"hbo": pd.DataFrame()}) is None


def test_both_matrices_use_the_report_s_one_correlation_scale():
    """A copy of the scale per figure would let a red cell mean +1 in one panel and -1 in
    the next."""
    for fig in (rest_channel_panel({"hbo": _fc_frame(["S1_D1", "S1_D2"], "hbo")}),
                fc_roi_matrix_figure({"hbo": _roi_frame(["A", "B"])})):
        heat = next(t for t in fig.data if t.type == "heatmap")
        assert [c for _, c in heat.colorscale] ==                [c for _, c in pc.get_colorscale(CORRELATION_SCALE)]


def test_the_range_is_symmetric_and_reaches_the_largest_correlation_of_either_panel():
    """All-positive correlations still span the red half, and 0 stays white."""
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    hbo, hbr = _fc_frame(pairs, "hbo"), _fc_frame(pairs, "hbr") * 0.5
    off = ~np.eye(3, dtype=bool)
    largest = max(np.abs(hbo.to_numpy()[off]).max(), np.abs(hbr.to_numpy()[off]).max())
    heats = [t for t in rest_channel_panel({"hbo": hbo, "hbr": hbr}).data if t.type == "heatmap"]
    assert [(h.zmin, h.zmax) for h in heats] == [(-largest, largest)] * 2
    roi = _roi_frame(["A", "B", "C"])
    heat = next(t for t in fc_roi_matrix_figure({"hbo": roi}).data if t.type == "heatmap")
    top = np.abs(roi.to_numpy()[off]).max()
    assert (heat.zmin, heat.zmax) == pytest.approx((-top, top))


# ---- only the measured part is drawn ----

def _without(frame, label):
    out = frame.copy()
    out.loc[label, :] = np.nan
    out.loc[:, label] = np.nan
    out.loc[label, label] = 1.0
    return out


def test_an_roi_with_no_value_leaves_the_roi_matrix():
    fig = fc_roi_matrix_figure({"hbo": _without(_roi_frame(["PFC", "TPJ", "M1"]), "TPJ")})
    heat = next(t for t in fig.data if t.type == "heatmap")
    assert list(heat.x) == list(heat.y) == ["PFC", "M1"]
    assert not any(s.fillcolor for s in fig.layout.shapes)


def test_a_channel_with_no_correlation_leaves_every_row_of_the_panel():
    pairs = ["S1_D1", "S1_D2", "S2_D1"]
    alff = pd.DataFrame({"channel": [f"{p} {c}" for p in pairs for c in ("hbo", "hbr")],
                         "alff": np.arange(6.0), "falff": np.arange(6.0) / 10})
    fig = rest_channel_panel({"hbo": _without(_fc_frame(pairs, "hbo"), "S1_D2 hbo")}, alff)
    heat = next(t for t in fig.data if t.type == "heatmap")
    strip = next(ax for name, ax in fig.layout.to_plotly_json().items()
                 if name.startswith("xaxis") and ax.get("ticktext"))
    assert list(heat.x) == list(strip["ticktext"]) == ["S1_D1", "S2_D1"]




# ---- ALFF on the cortex ----

@pytest.fixture(scope="module")
def haemo():
    import mne

    from tests._synth import synth_raw

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


@pytest.fixture
def projected(monkeypatch):
    """What alff_brain_figure hands to the projection and the renderer, with neither run."""
    seen = {}

    def project(raw, columns):
        seen.setdefault("blocks", []).append(list(raw.ch_names))
        # the first block, HbO, is the one the tests below read
        seen.setdefault("names", list(raw.ch_names))
        seen.setdefault("columns", np.asarray(columns))
        return None, None

    def render(stc, colour, subjects_dir, size, views):
        seen.setdefault("colour", colour)
        seen["views"] = views
        return [np.full((48, 56, 3), 255, np.uint8)] * len(views)

    monkeypatch.setattr(rest_figures, "project_to_cortex", project)
    monkeypatch.setattr(rest_figures, "render_cortex", render)
    return seen


def _long_hbo(raw):
    from nirspipe.qc.metrics import long_short_channels

    long_names, _ = long_short_channels(raw, None)
    return [ch for ch in long_names if ch.endswith(" hbo") and ch not in raw.info["bads"]]


def test_only_the_kept_long_hbo_channels_are_projected(haemo, alff_df, projected):
    marked = haemo.copy()
    marked.info["bads"] = [_long_hbo(haemo)[0]]
    assert alff_brain_figure(marked, alff_df) is not None

    names = _long_hbo(marked)
    assert projected["names"] == names
    malff = alff_df.set_index("channel")["malff"]
    np.testing.assert_array_equal(projected["columns"][:, 0], malff[names].to_numpy())
    # the second column is the coverage every vertex is divided by
    np.testing.assert_array_equal(projected["columns"][:, 1], 1.0)


def test_each_chromophore_of_the_table_is_a_block_of_its_own(haemo, alff_df, projected):
    from nirspipe.pipeline.chromophores import hbt_of

    hbt = hbt_of(haemo)
    table = pd.concat([alff_df, pd.DataFrame({"channel": hbt.ch_names, "alff": 1.0,
                                              "malff": 1.0, "falff": 0.5})])
    half = _long_hbo(haemo)[0].replace(" hbo", " hbr")
    marked = haemo.copy()
    marked.info["bads"] = [half]
    alff_brain_figure(marked, table)
    hbo, hbr, total = projected["blocks"]
    assert [{n.rsplit(" ", 1)[1] for n in names} for names in (hbo, hbr, total)] == [
        {"hbo"}, {"hbr"}, {"hbt"}]
    # one bad half takes its HbT out, not its other half
    pair = half.rsplit(" ", 1)[0]
    assert f"{pair} hbo" in hbo and f"{pair} hbt" not in total


def test_a_rejected_channel_does_not_set_the_colour_scale(haemo, alff_df, projected):
    """One dead channel with a runaway amplitude would otherwise flatten every real
    difference into one colour: the largest kept value must still take the top colour."""
    outlier = _long_hbo(haemo)[0]
    spiked = alff_df.copy()
    spiked.loc[spiked["channel"] == outlier, "malff"] = 1e6
    marked = haemo.copy()
    marked.info["bads"] = [outlier]
    alff_brain_figure(marked, spiked)

    top = spiked.set_index("channel")["malff"][_long_hbo(marked)].max()
    rgba = projected["colour"](np.array([-1.0]), np.array([[top, 1.0]]))
    np.testing.assert_allclose(rgba[0, :3], colormaps["viridis"](1.0)[:3], atol=1e-12)


def test_alff_stands_in_where_a_frame_carries_no_malff(haemo, alff_df, projected):
    """Only a hand-built frame lacks mALFF; the molar ALFF is drawn instead."""
    alff_brain_figure(haemo, alff_df.drop(columns=["malff"]))
    alff = alff_df.set_index("channel")["alff"]
    np.testing.assert_array_equal(projected["columns"][:, 0],
                                  alff[projected["names"]].to_numpy())


def test_the_views_skip_the_medial_wall(haemo, alff_df, projected):
    alff_brain_figure(haemo, alff_df)
    assert [(v["azimuth"], v["elevation"]) for v in projected["views"]] == [
        (180, 90), (0, 90), (0, 0), (90, 90)]


def test_no_value_anywhere_means_no_figure(haemo, alff_df, projected):
    assert alff_brain_figure(haemo, alff_df.assign(malff=np.nan)) is None
    assert "names" not in projected


def test_a_low_and_a_high_channel_land_on_their_own_sides(tmp_path):
    """Rendered for real: from above, the left channel's low value is purple on the
    left of the picture and the right channel's high value yellow on the right."""
    import base64
    import io
    import os

    import mne
    from PIL import Image

    subjects_dir = mne.utils.get_subjects_dir()
    if subjects_dir is None or not os.path.isfile(
            os.path.join(subjects_dir, "fsaverage", "surf", "lh.pial")):
        pytest.skip("fsaverage is not on this machine")

    names = [f"S{s}_D{d} {c}" for s, d in ((1, 1), (2, 2)) for c in ("hbo", "hbr")]
    hbo = [n for n in names if n.endswith(" hbo")]
    raw = mne.io.RawArray(np.zeros((4, 100)),
                          mne.create_info(names, 10.0, ["hbo", "hbr", "hbo", "hbr"]),
                          verbose=False)
    pos = {"S1": [-0.05, 0.02, 0.10], "D1": [-0.02, 0.02, 0.11],
           "S2": [0.05, 0.02, 0.10], "D2": [0.02, 0.02, 0.11]}
    fid = dict(nasion=[0.0, 0.085, -0.035], lpa=[-0.081, -0.029, -0.041],
               rpa=[0.084, -0.029, -0.041])
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=pos, coord_frame="head", **fid),
                    verbose="error")
    # HbO alone, so the picture is the one 2 x 2 block
    frame = pd.DataFrame({"channel": hbo, "malff": [0.5, 1.5]})

    img = np.array(Image.open(io.BytesIO(base64.b64decode(
        alff_brain_figure(raw, frame)))).convert("RGB")).astype(int)
    h, w, _ = img.shape
    superior = img[h // 2:, : int(w * 0.46)]   # bottom left of the 2 x 2, bar excluded

    def count(part, rgb):
        return int(np.all(np.abs(part - np.array(rgb) * 255) < 40, axis=2).sum())

    purple, yellow = colormaps["viridis"](0.0)[:3], colormaps["viridis"](1.0)[:3]
    half = superior.shape[1] // 2
    left, right = superior[:, :half], superior[:, half:]
    assert count(left, purple) > 50 and count(left, yellow) == 0
    assert count(right, yellow) > 50 and count(right, purple) == 0
