"""The Analysis page's schematic before a run: the stages each mode runs, in the pipeline's order."""

from __future__ import annotations

import pytest

from nirspipe.interface.callbacks.analysis_callbacks import _schematic_elements


def _nodes(elements) -> list[str]:
    return [e["data"]["id"] for e in elements if "id" in e["data"]]


def _edges(elements) -> set[tuple[str, str]]:
    return {(e["data"]["source"], e["data"]["target"]) for e in elements if "source" in e["data"]}


def test_motion_correction_runs_on_optical_density_before_beer_lambert():
    edges = _edges(_schematic_elements(None, False, False))
    assert ("sci", "motion") in edges and ("motion", "bll") in edges
    assert ("bll", "motion") not in edges


@pytest.mark.parametrize("mode, fc, confounds, expected", [
    ("none", False, False, set()),
    ("denoise", False, False, {"filter"}),
    ("denoise", False, True, {"filter", "regress"}),
    ("denoise", True, False, {"filter", "fc"}),
    ("glm", False, True, {"filter", "glm"}),
    ("glm", True, False, {"filter", "glm", "fc"}),
    ("rest", False, False, {"filter", "regress", "alff", "fc"}),
])
def test_each_mode_draws_the_stages_it_runs(mode, fc, confounds, expected):
    nodes = set(_nodes(_schematic_elements(mode, fc, confounds)))
    assert nodes - {"load", "od", "sci", "motion", "bll"} == expected


def test_every_edge_joins_two_drawn_nodes():
    elements = _schematic_elements("rest", True, True)
    nodes = set(_nodes(elements))
    assert all(a in nodes and b in nodes for a, b in _edges(elements))
