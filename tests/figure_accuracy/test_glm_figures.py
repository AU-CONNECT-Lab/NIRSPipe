"""The GLM figures: the design the model fitted, and each pair's own estimate."""

import numpy as np
import pandas as pd
import pytest

from tests._fingerprint import EVENT_ONSETS, RESPONSE_AMP
from tests.figure_accuracy._payload import one_figure


def _conditions(run, name):
    args, kwargs = run.captured[name]
    return list(kwargs.get("conditions") or args[1])


@pytest.mark.parametrize("name", ["design_matrix_static_figure", "design_matrix_heatmap"])
def test_both_design_figures_draw_the_conditions_and_nothing_else(glm_run, name):
    assert _conditions(glm_run, name) == [glm_run.task]


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
    assert list(results) == [glm_run.task]
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
