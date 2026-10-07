"""The resting-state figures: the written connectivity and amplitude, on the channels they name."""

import numpy as np
import pandas as pd
import pytest

from tests.figure_accuracy._payload import one_figure
from tests.figure_accuracy.conftest import ROI_MAP


def _table(run, suffix, index):
    return pd.read_csv(run.table(f"_{suffix}"), sep="\t", na_values="n/a").set_index(index)


def _kept_long(run):
    return [p for p in run.truth.long_pairs if not p.bad]


@pytest.fixture(scope="module")
def panel(rest_run):
    return one_figure(rest_run.figure("restpanel"))


# ---- channel connectivity ----

@pytest.mark.parametrize("index, chroma", [(0, "hbo"), (1, "hbr")])
def test_each_fc_matrix_is_the_written_matrix(rest_run, panel, index, chroma):
    heat = next(t for t in panel["data"] if t["type"] == "heatmap" and (t.get("xaxis") or "x")
                == ("x" if index == 0 else "x2"))
    written = _table(rest_run, f"chromo-{chroma}_stat-pearson_relmat.tsv", "channel")
    drawn = np.asarray(heat["z"], float)
    for i, row in enumerate(heat["y"]):
        for j, col in enumerate(heat["x"]):
            if np.isfinite(drawn[i, j]):
                assert drawn[i, j] == pytest.approx(written.loc[f"{row} {chroma}", f"{col} {chroma}"],
                                                    abs=1e-6), (row, col)


def test_the_fc_matrix_recovers_the_designed_network(rest_run, panel):
    heat = panel["data"][0]
    drawn = pd.DataFrame(np.asarray(heat["z"], float), index=list(heat["y"]), columns=list(heat["x"]))
    clean = rest_run.truth.haemo_no_systemic
    kept = _kept_long(rest_run)
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            value = drawn.loc[b.name, a.name]
            if not np.isfinite(value):
                value = drawn.loc[a.name, b.name]
            designed = np.corrcoef(clean[f"{a.name} hbo"], clean[f"{b.name} hbo"])[0, 1]
            assert value == pytest.approx(designed, abs=0.2), (a.name, b.name)


# ---- amplitude ----

def test_the_alff_strip_and_the_alff_map_are_the_written_alff(rest_run, panel):
    written = _table(rest_run, "stat-alff_nirsmap.tsv", "channel")
    strip = next(t for t in panel["data"] if t.get("name") == "HbO"
                 and "ALFF (M)" in (t.get("hovertemplate") or ""))
    for pair, value in zip(strip["customdata"], strip["y"]):
        assert value == pytest.approx(written.loc[f"{pair} hbo", "alff"], rel=1e-5), pair
    topo = one_figure(rest_run.figure("topo", suffix="nirsmap", entities="stat-alff"))
    hbo = next(t for t in topo["data"] if (t.get("xaxis") or "x") == "x" and t.get("customdata") is not None)
    for pair, malff, alff in zip(hbo["text"], hbo["marker"]["color"], hbo["customdata"]):
        assert malff == pytest.approx(written.loc[f"{pair} hbo", "malff"], rel=1e-5), pair
        assert alff == pytest.approx(written.loc[f"{pair} hbo", "alff"], rel=1e-5), pair


def test_alff_is_the_band_amplitude_of_the_broadband_residual_of_the_same_channel(rest_run):
    written = _table(rest_run, "stat-alff_nirsmap.tsv", "channel")
    raw = rest_run.read("errtsbroad")
    freqs = np.fft.rfftfreq(len(raw.times), 1 / raw.info["sfreq"])
    band = (freqs >= 0.01) & (freqs <= 0.08)
    ratios = []
    for pair in _kept_long(rest_run):
        x = raw.get_data(picks=[f"{pair.name} hbo"])[0]
        amplitude = np.abs(np.fft.rfft(x - x.mean()))[band].mean()
        ratios.append(written.loc[f"{pair.name} hbo", "alff"] / amplitude)
    # one constant normalisation for every channel; a value on the wrong channel breaks it
    assert np.std(ratios) < 0.01 * np.mean(ratios)


# ---- ROIs ----

@pytest.mark.parametrize("index, chroma", [(0, "hbo"), (1, "hbr")])
def test_the_roi_matrix_is_the_written_roi_matrix(rest_run, index, chroma):
    fig = one_figure(rest_run.figure("matrix", suffix="relmat",
                                     entities="seg-roinet_agg-roi_stat-pearson"))
    heat = [t for t in fig["data"] if t["type"] == "heatmap"][index]
    written = _table(rest_run, f"chromo-{chroma}_seg-roinet_agg-roi_stat-pearson_relmat.tsv", "roi")
    drawn = np.asarray(heat["z"], float)
    assert list(heat["x"]) == list(ROI_MAP)
    for i, row in enumerate(heat["y"]):
        for j, col in enumerate(heat["x"]):
            if np.isfinite(drawn[i, j]):
                assert drawn[i, j] == pytest.approx(written.loc[row, col], abs=1e-6), (row, col)


def test_the_roi_matrix_shows_the_network_between_its_rois(rest_run):
    written = _table(rest_run, "chromo-hbo_seg-roinet_agg-roi_stat-pearson_relmat.tsv", "roi")
    assert written.loc["A", "B"] < -0.2
    assert abs(written.loc["A", "C"]) < 0.15 and abs(written.loc["B", "C"]) < 0.15


def test_each_seed_map_colours_a_channel_by_its_correlation_with_that_seed(rest_run):
    fig = one_figure(rest_run.figure("matrix", suffix="relmat",
                                     entities="seg-roinet_agg-seed_stat-pearson"))
    titles = [a["text"] for a in fig["layout"]["annotations"]]
    written = _table(rest_run, "chromo-hbo_seg-roinet_agg-seed_stat-pearson_relmat.tsv", "roi")
    for k, seed in enumerate(ROI_MAP):
        axis = "x" if 2 * k == 0 else f"x{2 * k + 1}"
        assert titles[2 * k] == f"{seed} - HbO"
        for trace in fig["data"]:
            colour = (trace.get("marker") or {}).get("color")
            if (trace.get("xaxis") or "x") != axis or isinstance(colour, str) or colour is None:
                continue
            for pair, value in zip(trace["text"], colour):
                assert value == pytest.approx(written.loc[seed, f"{pair} hbo"], abs=1e-6), (seed, pair)
    assert written.loc["A", "S4_D4 hbo"] < -0.2
    assert abs(written.loc["A", "S6_D6 hbo"]) < 0.15
