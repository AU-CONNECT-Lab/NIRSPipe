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


@pytest.mark.parametrize("name", ["design_matrix_static_figure", "design_matrix_heatmap"])
def test_both_design_figures_draw_the_conditions_and_nothing_else(glm_run, name):
    assert _conditions(glm_run, name) == [glm_run.task]


def test_the_heatmap_draws_the_whole_design_written_to_disk(glm_run):
    drawn = glm_run.captured["design_matrix_heatmap"][0][0]
    on_disk = pd.read_csv(glm_run.table("_design.tsv"), sep="\t")
    assert list(drawn.columns) == list(on_disk.columns)
    np.testing.assert_allclose(drawn.to_numpy(float), on_disk.to_numpy(float), rtol=1e-5, atol=1e-9)


def test_the_activation_draws_the_conditions_and_nothing_else(glm_run):
    assert list(glm_run.captured["activation_condition_figures"][0][1]) == [glm_run.task]


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
