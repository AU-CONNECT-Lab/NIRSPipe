"""The brain figure leaves out its static flat map where the page draws the clickable one."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from nirspipe.qc.common.figure_io import figure_namer
from nirspipe.qc.subject import report
from tests._synth import synth_raw


def _png_b64(width: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, 40), "white").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(report, "quality_brain_views", lambda *a, **k: _png_b64(120))
    raw = synth_raw("01", "tapping")
    names = [n for n in raw.ch_names if n.endswith("760")]
    coords = np.array([raw.info["chs"][raw.ch_names.index(n)]["loc"][:3] for n in names])
    return raw, names, coords


def _draw(inputs, tmp_path, flat_map):
    raw, names, coords = inputs
    errors = []
    out = report._section_brain({n: 0.9 for n in raw.ch_names}, [], coords,
                                np.ones(len(names), bool), raw, "01", errors, tmp_path,
                                figure_namer("sub-01_task-tapping"), ch_names_brain=names,
                                sci_threshold=0.8, flat_map=flat_map)
    assert errors == []
    return Image.open(tmp_path / out["brain_views_path"].rsplit("/", 1)[-1]).size


def test_without_its_flat_map_the_figure_is_the_3d_views_alone(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "optode_layout_static",
                        lambda *a, **k: pytest.fail("the flat map was drawn"))
    assert _draw(inputs, tmp_path, flat_map=False) == (120, 40)


def test_with_it_the_flat_map_sits_beside_the_views(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "optode_layout_static", lambda *a, **k: _png_b64(40))
    assert _draw(inputs, tmp_path, flat_map=True) == (160, 40)
