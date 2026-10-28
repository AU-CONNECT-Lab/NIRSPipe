"""The GLM figures: the design the model fitted, and each pair's own estimate."""

import numpy as np
import pandas as pd
import pytest

from tests._fingerprint import AUX_NAME, EVENT_ONSETS, RESPONSE_AMP
from tests.figure_accuracy._payload import one_figure


def _conditions(run, name):
    args, kwargs = run.captured[name]
    return list(kwargs.get("conditions") or args[1])


def test_the_model_carries_the_aux_regressor(glm_run):
    design = glm_run.captured["design_matrix_static_figure"][0][0]
    assert f"aux_{AUX_NAME}" in design.columns


def test_the_design_time_series_draws_the_conditions_and_nothing_else(glm_run):
    assert _conditions(glm_run, "design_matrix_static_figure") == [glm_run.task]


def test_the_heatmap_draws_the_whole_design_written_to_disk(glm_run):
    drawn = glm_run.captured["design_matrix_heatmap"][0][0]
    on_disk = pd.read_csv(glm_run.table("_design.tsv"), sep="\t")
    # bookkeeping for the frames left out of the fit, never drawn; this run has none
    assert (on_disk.pop("censored") == 0).all()
    assert list(drawn.columns) == list(on_disk.columns)
    np.testing.assert_allclose(drawn.to_numpy(float), on_disk.to_numpy(float), rtol=1e-5, atol=1e-9)


def test_the_activation_draws_the_conditions_and_nothing_else(glm_run):
    assert list(glm_run.captured["activation_condition_figures"][0][1]) == [glm_run.task]


def test_every_activation_file_written_is_a_condition_s(glm_run):
    written = sorted(p.name for p in glm_run.figures.glob("*desc-glmactivation*"))
    assert written and all(f"_cond-{glm_run.task}_" in name for name in written), written


def test_the_task_regressor_rises_after_each_event_and_not_before(glm_run):
    design = glm_run.captured["design_matrix_static_figure"][0][0]
    t = design.index.to_numpy(float)
    regressor = design[glm_run.task].to_numpy(float)
    for onset in EVENT_ONSETS:
        after = (t >= onset) & (t < onset + 30)
        before = (t >= onset - 5) & (t < onset)
        assert np.abs(regressor[before]).max() < 0.05 * regressor[after].max(), onset
        assert 3 < t[after][np.argmax(regressor[after])] - onset < 20, onset


@pytest.fixture(scope="module")
def estimates(glm_run):
    results = glm_run.captured["activation_condition_figures"][0][1]
    return results[glm_run.task].set_index("ch_name")["theta"]


def test_the_activation_draws_the_estimates_written_to_disk(glm_run, estimates):
    table = pd.read_csv(glm_run.table("_desc-glm_nirsmap.tsv"), sep="\t")
    on_disk = table[table["Condition"] == glm_run.task].set_index("ch_name")["theta"]
    pd.testing.assert_series_equal(estimates.sort_index(), on_disk.sort_index(),
                                   check_names=False, rtol=1e-6)


def test_only_the_responders_activate(glm_run, estimates):
    for pair in glm_run.truth.pairs:
        hbo, hbr = estimates[f"{pair.name} hbo"], estimates[f"{pair.name} hbr"]
        if pair.responds:
            assert hbo > 0.5 * RESPONSE_AMP, pair.name
            assert hbr == pytest.approx(-hbo / 3, rel=0.3), pair.name
        else:
            assert abs(hbo) < 0.1 * RESPONSE_AMP, pair.name


def test_the_correlation_panel_says_its_after_stage_is_the_glm_residual(glm_run):
    names = {t.get("name") for t in one_figure(glm_run.figure("hbohbrcorr"))["data"]}
    assert "after GLM (task removed)" in names and "after denoising" not in names


# ---- whitening ----

def test_the_whitening_spectrum_draws_the_table_written_to_disk(glm_run):
    on_disk = pd.read_csv(glm_run.table("_stat-psd_desc-whitening_timeseries.tsv"), sep="\t")
    traces = {t.get("name"): t for t in one_figure(glm_run.figure("whitening", suffix="nirsmap"))["data"]}
    for chroma, label in (("hbo", "HbO"), ("hbr", "HbR")):
        drawn = traces[f"{label} whitened"]
        np.testing.assert_allclose(np.asarray(drawn["x"], float), on_disk["freq"], rtol=1e-12)
        np.testing.assert_allclose(np.asarray(drawn["y"], float), on_disk[f"{chroma}_whitened"],
                                   rtol=1e-12)


def test_the_record_and_the_report_carry_the_low_band_power(glm_run):
    channels = pd.read_csv(glm_run.table("_desc-whitening_nirsmap.tsv"), sep="\t")
    good = channels[channels["long"] & ~channels["bad"]]
    section = glm_run.record()["whitening"]
    for chroma in ("hbo", "hbr"):
        expected = good.loc[good["Chroma"] == chroma, "lowband_whitened"].median()
        assert section[f"whitening_lowband_{chroma}"] == pytest.approx(expected)
    html = glm_run.report.read_text(encoding="utf-8")
    assert "GLM residual" in html and "Residual low-band power HbO" in html


def test_each_contrast_is_drawn_with_the_effect_written_to_disk(glm_contrast_run):
    """The second activation call draws the contrasts: each one's effect, read back from its table."""
    calls = glm_contrast_run.captured["_calls"]["activation_condition_figures"]
    assert len(calls) == 2
    drawn, kwargs = calls[1][0][1], calls[1][1]
    assert list(drawn) == ["whole", "half"] and kwargs["quantity"] == "contrast"
    table = pd.read_csv(glm_contrast_run.table("_desc-contrast_nirsmap.tsv"), sep="\t")
    for name, rows in drawn.items():
        on_disk = table[table["contrast"] == name].set_index("ch_name")["effect"]
        np.testing.assert_allclose(rows.set_index("ch_name")["theta"], on_disk.loc[rows["ch_name"]])
    # the half contrast is half the whole one, channel by channel
    whole, half = (drawn[n].set_index("ch_name")["theta"] for n in ("whole", "half"))
    np.testing.assert_allclose(half, whole / 2, rtol=1e-6)


def test_the_page_shows_one_contrast_figure_per_contrast(glm_contrast_run):
    written = sorted(p.name for p in glm_contrast_run.figures.glob("*desc-glmcontrast*"))
    assert [n.split("_cond-")[1].split("_")[0] for n in written] == ["half", "whole"]
    page = (glm_contrast_run.out / "sub-01" / "sub-01_task-tapping_report.html").read_text(
        encoding="utf-8")
    assert "<h3>Contrasts</h3>" in page and all(f"figures/{n}" in page for n in written)
