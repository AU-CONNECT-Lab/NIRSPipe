"""A condition matrix with a few rows still has a plotting area; Plotly throws when its colour
bar gets no length, and the cohort page then shows an empty frame."""

import numpy as np
import pytest

from nirspipe.qc.figures.subject.group_figures import _condition_heatmap


@pytest.mark.parametrize("n_rows", [1, 3])
def test_a_short_condition_matrix_keeps_a_plotting_area(n_rows):
    labels = [f"sub-{i:02d}" for i in range(n_rows)]
    values = np.arange(n_rows * 4, dtype=float).reshape(n_rows, 4)
    fig = _condition_heatmap([("SCI", values), ("PSP", values)],
                             ["Collaboration#1", "Individual#1", "Collaboration#2", "Individual#2"],
                             labels, "run")

    margin = fig.layout.margin
    assert fig.layout.height - margin.t - margin.b >= 100
