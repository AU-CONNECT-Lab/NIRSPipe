"""The haemoglobin panels after denoising: the right stage, the right rows, the right pairs."""

import numpy as np
import pytest

from tests._fingerprint import HBR_FREQ, SPIKE, SYSTEMIC_FREQ
from tests.figure_accuracy._payload import one_figure
from tests.figure_accuracy._read import share_at, which_pair


def _heatmaps(fig):
    return [t for t in fig["data"] if t["type"] == "heatmap"]


# ---- HbO-HbR correlation ----

@pytest.fixture(scope="module")
def corr(denoise_run):
    return one_figure(denoise_run.figure("hbohbrcorr"))


@pytest.mark.parametrize("index, desc", [(0, "preproc"), (1, "errts")])
def test_each_matrix_is_the_channel_correlation_of_its_stage(denoise_run, corr, index, desc):
    heat = _heatmaps(corr)[index]
    assert list(heat["x"]) == list(heat["y"])
    expected = np.corrcoef(denoise_run.read(desc).get_data(picks=list(heat["x"])))
    drawn = np.asarray(heat["z"], float)
    shown = np.isfinite(drawn)
    assert shown.sum() == len(drawn) * (len(drawn) - 1) // 2
    np.testing.assert_allclose(drawn[shown], expected[shown], atol=1e-6)


@pytest.mark.parametrize("name, desc", [("before denoising", "preproc"), ("after denoising", "errts")])
def test_each_pair_dot_is_that_pair_s_hbo_hbr_correlation(denoise_run, corr, name, desc):
    dots = next(t for t in corr["data"] if t.get("name") == name)
    raw = denoise_run.read(desc)
    for pair, r in zip(dots["customdata"], dots["y"]):
        hbo, hbr = raw.get_data(picks=[f"{pair} hbo", f"{pair} hbr"])
        assert r == pytest.approx(np.corrcoef(hbo, hbr)[0, 1], abs=1e-6), pair


def test_after_denoising_the_dots_recover_the_designed_correlation(denoise_run, corr):
    dots = next(t for t in corr["data"] if t.get("name") == "after denoising")
    drawn = dict(zip(dots["customdata"], dots["y"]))
    clean = denoise_run.truth.haemo_no_systemic
    for pair in denoise_run.truth.long_pairs:
        if pair.bad or pair.name in denoise_run.truth.moved:
            continue
        designed = np.corrcoef(clean[f"{pair.name} hbo"], clean[f"{pair.name} hbr"])[0, 1]
        assert drawn[pair.name] == pytest.approx(designed, abs=0.15), pair.name


@pytest.mark.xfail(strict=True, reason="rejected pairs are ranked best-to-worst with the kept ones, unmarked")
def test_a_rejected_pair_is_not_ranked_among_the_kept_ones(denoise_run, corr):
    rejected = {p.name for p in denoise_run.truth.pairs if p.bad}
    for name in ("before denoising", "after denoising"):
        dots = next(t for t in corr["data"] if t.get("name") == name)
        assert not rejected & set(dots["customdata"])


# ---- carpets ----

def test_each_motion_carpet_row_is_the_channel_it_is_labelled(denoise_run):
    fig = one_figure(denoise_run.figure("carpet"))
    for heat in _heatmaps(fig):
        x = np.asarray(heat["x"], float)
        for label, row in zip(heat["y"], np.asarray(heat["z"], float)):
            pair = denoise_run.truth.pair(label.rsplit(" ", 1)[0])
            if not pair.short and not pair.bad:
                assert which_pair(x, row, denoise_run.truth) == pair.name, label


def test_the_motion_carpet_shows_the_spike_on_its_channel_at_its_time(denoise_run):
    fig = one_figure(denoise_run.figure("carpet"))
    before = _heatmaps(fig)[0]
    x = np.asarray(before["x"], float)
    window = (x >= SPIKE[1]) & (x <= SPIKE[1] + 2)
    spike_pair = denoise_run.truth.spike[0]
    for label, row in zip(before["y"], np.asarray(before["z"], float)):
        pair = label.rsplit(" ", 1)[0]
        if pair == spike_pair:
            assert np.abs(row[window]).max() >= 2.9, label
        elif not denoise_run.truth.pair(pair).bad and pair != denoise_run.truth.step[0]:
            assert np.abs(row[window]).max() < 2.5, label


@pytest.fixture(scope="module")
def stage_carpet(denoise_run):
    heat = _heatmaps(one_figure(denoise_run.figure("carpetstage")))[0]
    return np.asarray(heat["x"], float), dict(zip(heat["y"], np.asarray(heat["z"], float)))


def test_the_denoised_carpet_rows_are_their_own_pairs(denoise_run, stage_carpet):
    x, rows = stage_carpet
    for pair in denoise_run.truth.long_pairs:
        if not pair.bad:
            assert which_pair(x, rows[f"{pair.name} hbo"], denoise_run.truth) == pair.name


def test_the_denoised_carpet_is_after_regression_and_keeps_the_chromophores_apart(denoise_run, stage_carpet):
    x, rows = stage_carpet
    for pair in denoise_run.truth.long_pairs:
        if pair.bad:
            continue
        assert share_at(x, rows[f"{pair.name} hbo"], SYSTEMIC_FREQ) < 0.02
        assert share_at(x, rows[f"{pair.name} hbr"], HBR_FREQ) > 10 * share_at(x, rows[f"{pair.name} hbo"], HBR_FREQ)
